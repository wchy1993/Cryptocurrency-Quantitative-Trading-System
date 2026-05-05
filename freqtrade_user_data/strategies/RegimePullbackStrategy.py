from __future__ import annotations

from datetime import datetime

import talib.abstract as ta
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, merge_informative_pair
from freqtrade.vendor.qtpylib import indicators as qtpylib
from pandas import DataFrame


class RegimePullbackStrategy(IStrategy):
    """
    Deterministic futures baseline.

    This strategy deliberately avoids FreqAI. It uses BTC 1h market regime as the
    top-level filter, then trades only SOL/ADA/LTC with trend pullback/breakout
    entries or range reversion entries.
    """

    timeframe = "15m"
    can_short = True
    process_only_new_candles = True
    startup_candle_count = 240

    minimal_roi = {"0": 0.028}
    stoploss = -0.016
    trailing_stop = True
    trailing_stop_positive = 0.008
    trailing_stop_positive_offset = 0.016
    trailing_only_offset_is_reached = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    position_adjustment_enable = False

    trade_pairs = {"SOL/USDT:USDT", "ADA/USDT:USDT", "LTC/USDT:USDT"}

    @property
    def protections(self) -> list[dict]:
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 3},
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 36,
                "trade_limit": 2,
                "stop_duration_candles": 18,
                "only_per_pair": False,
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 144,
                "trade_limit": 15,
                "stop_duration_candles": 48,
                "max_allowed_drawdown": 0.04,
            },
        ]

    def informative_pairs(self) -> list[tuple[str, str]]:
        return [("BTC/USDT:USDT", "1h")]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"]
        dataframe["volume_ratio"] = dataframe["volume"] / dataframe["volume"].rolling(20).mean()
        dataframe["ema50_slope"] = dataframe["ema50"] / dataframe["ema50"].shift(8) - 1.0
        dataframe["ema_dist"] = (dataframe["close"] - dataframe["ema200"]) / dataframe["close"]
        dataframe["ema50_extension"] = (dataframe["close"] - dataframe["ema50"]).abs() / dataframe["close"]
        dataframe["donchian_high"] = dataframe["high"].rolling(36).max().shift(1)
        dataframe["donchian_low"] = dataframe["low"].rolling(36).min().shift(1)

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2.2)
        dataframe["bb_lower"] = bollinger["lower"]
        dataframe["bb_mid"] = bollinger["mid"]
        dataframe["bb_upper"] = bollinger["upper"]
        dataframe["bb_width"] = (bollinger["upper"] - bollinger["lower"]) / bollinger["mid"]
        dataframe["bb_width_rank"] = dataframe["bb_width"].rolling(96).rank(pct=True)
        dataframe["bb_lower_reclaim"] = (dataframe["low"] < dataframe["bb_lower"]) & (
            dataframe["close"] > dataframe["bb_lower"]
        )
        dataframe["bb_upper_reclaim"] = (dataframe["high"] > dataframe["bb_upper"]) & (
            dataframe["close"] < dataframe["bb_upper"]
        )

        return self._merge_btc_regime(dataframe)

    def _merge_btc_regime(self, dataframe: DataFrame) -> DataFrame:
        if not self.dp:
            dataframe["btc_trend_long"] = True
            dataframe["btc_trend_short"] = True
            dataframe["btc_range"] = True
            dataframe["btc_risk_ok"] = True
            return dataframe

        btc = self.dp.get_pair_dataframe(pair="BTC/USDT:USDT", timeframe="1h")
        if btc.empty:
            dataframe["btc_trend_long"] = True
            dataframe["btc_trend_short"] = True
            dataframe["btc_range"] = True
            dataframe["btc_risk_ok"] = True
            return dataframe

        btc = btc.copy()
        btc["btc_ema50"] = ta.EMA(btc, timeperiod=50)
        btc["btc_ema200"] = ta.EMA(btc, timeperiod=200)
        btc["btc_adx"] = ta.ADX(btc, timeperiod=14)
        btc["btc_rsi"] = ta.RSI(btc, timeperiod=14)
        btc["btc_roc3"] = btc["close"].pct_change(3)
        btc["btc_ema50_slope"] = btc["btc_ema50"] / btc["btc_ema50"].shift(6) - 1.0
        btc["btc_close"] = btc["close"]
        btc = btc[
            [
                "date",
                "btc_close",
                "btc_ema50",
                "btc_ema200",
                "btc_adx",
                "btc_rsi",
                "btc_roc3",
                "btc_ema50_slope",
            ]
        ]

        dataframe = merge_informative_pair(
            dataframe,
            btc,
            self.timeframe,
            "1h",
            ffill=True,
            append_timeframe=True,
        )

        dataframe["btc_trend_long"] = (
            (dataframe["btc_close_1h"] > dataframe["btc_ema200_1h"])
            & (dataframe["btc_ema50_1h"] > dataframe["btc_ema200_1h"])
            & (dataframe["btc_ema50_slope_1h"] > -0.0015)
            & (dataframe["btc_rsi_1h"] < 76)
        ).fillna(False)
        dataframe["btc_trend_short"] = (
            (dataframe["btc_close_1h"] < dataframe["btc_ema200_1h"])
            & (dataframe["btc_ema50_1h"] < dataframe["btc_ema200_1h"])
            & (dataframe["btc_ema50_slope_1h"] < 0.0015)
            & (dataframe["btc_rsi_1h"] > 24)
        ).fillna(False)
        dataframe["btc_range"] = (
            dataframe["btc_rsi_1h"].between(34, 66)
            & (dataframe["btc_adx_1h"] <= 23)
            & (dataframe["btc_roc3_1h"].abs() < 0.018)
        ).fillna(False)
        dataframe["btc_risk_ok"] = (
            dataframe["btc_rsi_1h"].between(24, 78)
            & (dataframe["btc_roc3_1h"].abs() < 0.035)
        ).fillna(False)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        if metadata["pair"] not in self.trade_pairs:
            return dataframe

        vol_ok = dataframe["atr_pct"].between(0.0015, 0.026)
        not_chasing = dataframe["ema50_extension"] <= 0.032

        trend_long_base = (
            dataframe["btc_trend_long"]
            & dataframe["btc_risk_ok"]
            & (dataframe["close"] > dataframe["ema200"])
            & (dataframe["ema50"] > dataframe["ema200"])
            & (dataframe["ema50_slope"] > 0.0002)
            & dataframe["rsi"].between(42, 70)
            & (dataframe["adx"] >= 16)
            & (dataframe["volume_ratio"] >= 0.75)
            & vol_ok
            & not_chasing
        )
        trend_short_base = (
            dataframe["btc_trend_short"]
            & dataframe["btc_risk_ok"]
            & (dataframe["close"] < dataframe["ema200"])
            & (dataframe["ema50"] < dataframe["ema200"])
            & (dataframe["ema50_slope"] < -0.0002)
            & dataframe["rsi"].between(30, 58)
            & (dataframe["adx"] >= 16)
            & (dataframe["volume_ratio"] >= 0.75)
            & vol_ok
            & not_chasing
        )

        pullback_long = (
            trend_long_base
            & (dataframe["low"] <= dataframe["ema20"] * 1.004)
            & (dataframe["close"] > dataframe["ema20"])
            & ((dataframe["close"] > dataframe["open"]) | (dataframe["rsi"] > dataframe["rsi"].shift(1)))
        )
        pullback_short = (
            trend_short_base
            & (dataframe["high"] >= dataframe["ema20"] * 0.996)
            & (dataframe["close"] < dataframe["ema20"])
            & ((dataframe["close"] < dataframe["open"]) | (dataframe["rsi"] < dataframe["rsi"].shift(1)))
        )

        breakout_long = (
            trend_long_base
            & (dataframe["close"] > dataframe["donchian_high"])
            & (dataframe["volume_ratio"] >= 1.05)
            & (dataframe["rsi"] <= 68)
        )
        breakout_short = (
            trend_short_base
            & (dataframe["close"] < dataframe["donchian_low"])
            & (dataframe["volume_ratio"] >= 1.05)
            & (dataframe["rsi"] >= 32)
        )

        range_base = (
            dataframe["btc_range"]
            & dataframe["btc_risk_ok"]
            & (dataframe["adx"] <= 24)
            & (dataframe["bb_width_rank"] < 0.80)
            & dataframe["atr_pct"].between(0.0012, 0.020)
            & (dataframe["volume_ratio"] >= 0.65)
        )
        range_long = (
            range_base
            & dataframe["bb_lower_reclaim"]
            & (dataframe["rsi"] < 40)
            & ((dataframe["bb_mid"] - dataframe["close"]) / dataframe["close"] > 0.006)
        )
        range_short = (
            range_base
            & dataframe["bb_upper_reclaim"]
            & (dataframe["rsi"] > 60)
            & ((dataframe["close"] - dataframe["bb_mid"]) / dataframe["close"] > 0.006)
        )

        dataframe.loc[pullback_long, ["enter_long", "enter_tag"]] = (1, "trend_pullback_long")
        dataframe.loc[pullback_short, ["enter_short", "enter_tag"]] = (1, "trend_pullback_short")
        dataframe.loc[breakout_long, ["enter_long", "enter_tag"]] = (1, "trend_breakout_long")
        dataframe.loc[breakout_short, ["enter_short", "enter_tag"]] = (1, "trend_breakout_short")
        dataframe.loc[range_long, ["enter_long", "enter_tag"]] = (1, "range_reclaim_long")
        dataframe.loc[range_short, ["enter_short", "enter_tag"]] = (1, "range_reclaim_short")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        trend_long_exit = (
            ((dataframe["close"] < dataframe["ema50"]) & (dataframe["rsi"] < 46))
            | (dataframe["rsi"] > 76)
            | (~dataframe["btc_risk_ok"])
        )
        trend_short_exit = (
            ((dataframe["close"] > dataframe["ema50"]) & (dataframe["rsi"] > 54))
            | (dataframe["rsi"] < 24)
            | (~dataframe["btc_risk_ok"])
        )
        range_long_exit = (dataframe["close"] >= dataframe["bb_mid"]) | (dataframe["rsi"] > 56)
        range_short_exit = (dataframe["close"] <= dataframe["bb_mid"]) | (dataframe["rsi"] < 44)

        dataframe.loc[trend_long_exit | range_long_exit, ["exit_long", "exit_tag"]] = (
            1,
            "rule_long_exit",
        )
        dataframe.loc[trend_short_exit | range_short_exit, ["exit_short", "exit_tag"]] = (
            1,
            "rule_short_exit",
        )
        return dataframe

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        return min(3.0, max_leverage)

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> str | bool | None:
        tag = trade.enter_tag or ""
        if "range" in tag:
            if current_profit >= 0.008:
                return "range_profit_take"
            if current_profit <= -0.007:
                return "range_loss_cut"
        if current_profit >= 0.018:
            return "trend_profit_take"
        if current_profit <= -0.012:
            return "trend_loss_cut"
        return None
