from __future__ import annotations

from datetime import datetime

import talib.abstract as ta
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, merge_informative_pair
from pandas import DataFrame


class BtcHourlyMomentumStrategy(IStrategy):
    """
    Conservative 1h BTC-regime breakout strategy for Binance USDT futures.

    This replaces the noisy 15m pullback approach with slower entries:
    only trade liquid pairs, only take short breakouts aligned with the pair
    trend, and use BTC 4h as a market-risk filter.
    """

    timeframe = "1h"
    can_short = True
    process_only_new_candles = True
    startup_candle_count = 260

    minimal_roi = {"0": 0.07}
    stoploss = -0.03
    trailing_stop = True
    trailing_stop_positive = 0.012
    trailing_stop_positive_offset = 0.034
    trailing_only_offset_is_reached = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    position_adjustment_enable = False

    trade_pairs = {
        "BTC/USDT:USDT",
        "ETH/USDT:USDT",
        "BNB/USDT:USDT",
        "SOL/USDT:USDT",
        "XRP/USDT:USDT",
        "ADA/USDT:USDT",
        "DOGE/USDT:USDT",
        "LINK/USDT:USDT",
        "AVAX/USDT:USDT",
        "LTC/USDT:USDT",
    }

    @property
    def protections(self) -> list[dict]:
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 4},
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 48,
                "trade_limit": 3,
                "stop_duration_candles": 18,
                "only_per_pair": False,
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 240,
                "trade_limit": 20,
                "stop_duration_candles": 48,
                "max_allowed_drawdown": 0.06,
            },
        ]

    def informative_pairs(self) -> list[tuple[str, str]]:
        configured_pairs = set(self.trade_pairs)
        if getattr(self, "config", None):
            configured_pairs.update(self.config.get("exchange", {}).get("pair_whitelist", []))

        informative = {(pair, "4h") for pair in configured_pairs}
        informative.add(("BTC/USDT:USDT", "4h"))
        return sorted(informative)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["atr_pct"] = ta.ATR(dataframe, timeperiod=14) / dataframe["close"]
        dataframe["volume_ratio"] = dataframe["volume"] / dataframe["volume"].rolling(24).mean()
        dataframe["ema50_slope"] = dataframe["ema50"] / dataframe["ema50"].shift(12) - 1.0
        dataframe["donchian_high"] = dataframe["high"].rolling(36).max().shift(1)
        dataframe["donchian_low"] = dataframe["low"].rolling(36).min().shift(1)
        dataframe["range_high"] = dataframe["high"].rolling(48).max()
        dataframe["range_low"] = dataframe["low"].rolling(48).min()
        dataframe["range_pos"] = (
            (dataframe["close"] - dataframe["range_low"])
            / (dataframe["range_high"] - dataframe["range_low"])
        )

        dataframe = self._merge_pair_4h(dataframe, metadata)
        dataframe = self._merge_btc_4h(dataframe)
        return dataframe

    def _merge_pair_4h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not self.dp:
            dataframe["pair_4h_long"] = True
            dataframe["pair_4h_short"] = True
            return dataframe

        pair_4h = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe="4h")
        if pair_4h.empty:
            dataframe["pair_4h_long"] = True
            dataframe["pair_4h_short"] = True
            return dataframe

        pair_4h = pair_4h.copy()
        pair_4h["p4h_close"] = pair_4h["close"]
        pair_4h["p4h_ema50"] = ta.EMA(pair_4h, timeperiod=50)
        pair_4h["p4h_ema200"] = ta.EMA(pair_4h, timeperiod=200)
        pair_4h["p4h_rsi"] = ta.RSI(pair_4h, timeperiod=14)
        pair_4h["p4h_adx"] = ta.ADX(pair_4h, timeperiod=14)
        pair_4h["p4h_roc3"] = pair_4h["close"].pct_change(3)
        pair_4h["p4h_ema50_slope"] = pair_4h["p4h_ema50"] / pair_4h["p4h_ema50"].shift(6) - 1.0
        pair_4h = pair_4h[
            [
                "date",
                "p4h_close",
                "p4h_ema50",
                "p4h_ema200",
                "p4h_rsi",
                "p4h_adx",
                "p4h_roc3",
                "p4h_ema50_slope",
            ]
        ]

        dataframe = merge_informative_pair(
            dataframe,
            pair_4h,
            self.timeframe,
            "4h",
            ffill=True,
            append_timeframe=True,
        )
        dataframe["pair_4h_long"] = (
            (dataframe["p4h_close_4h"] > dataframe["p4h_ema200_4h"])
            & (dataframe["p4h_close_4h"] > dataframe["p4h_ema50_4h"])
            & (dataframe["p4h_ema50_slope_4h"] > 0)
            & dataframe["p4h_rsi_4h"].between(52, 74)
            & (dataframe["p4h_roc3_4h"] > -0.012)
        ).fillna(False)
        dataframe["pair_4h_short"] = (
            (dataframe["p4h_close_4h"] < dataframe["p4h_ema50_4h"])
            & (dataframe["p4h_ema50_slope_4h"] < 0)
            & dataframe["p4h_rsi_4h"].between(24, 48)
            & (dataframe["p4h_roc3_4h"] < 0.012)
        ).fillna(False)
        return dataframe

    def _merge_btc_4h(self, dataframe: DataFrame) -> DataFrame:
        if not self.dp:
            dataframe["btc_long_ok"] = True
            dataframe["btc_short_ok"] = True
            dataframe["btc_risk_ok"] = True
            return dataframe

        btc_4h = self.dp.get_pair_dataframe(pair="BTC/USDT:USDT", timeframe="4h")
        if btc_4h.empty:
            dataframe["btc_long_ok"] = True
            dataframe["btc_short_ok"] = True
            dataframe["btc_risk_ok"] = True
            return dataframe

        btc_4h = btc_4h.copy()
        btc_4h["btc_close"] = btc_4h["close"]
        btc_4h["btc_ema50"] = ta.EMA(btc_4h, timeperiod=50)
        btc_4h["btc_ema200"] = ta.EMA(btc_4h, timeperiod=200)
        btc_4h["btc_rsi"] = ta.RSI(btc_4h, timeperiod=14)
        btc_4h["btc_roc3"] = btc_4h["close"].pct_change(3)
        btc_4h["btc_ema50_slope"] = btc_4h["btc_ema50"] / btc_4h["btc_ema50"].shift(6) - 1.0
        btc_4h = btc_4h[
            [
                "date",
                "btc_close",
                "btc_ema50",
                "btc_ema200",
                "btc_rsi",
                "btc_roc3",
                "btc_ema50_slope",
            ]
        ]

        dataframe = merge_informative_pair(
            dataframe,
            btc_4h,
            self.timeframe,
            "4h",
            ffill=True,
            append_timeframe=True,
        )
        dataframe["btc_long_ok"] = (
            (dataframe["btc_close_4h"] > dataframe["btc_ema200_4h"] * 0.985)
            & (dataframe["btc_close_4h"] > dataframe["btc_ema50_4h"])
            & (dataframe["btc_ema50_slope_4h"] > -0.003)
            & dataframe["btc_rsi_4h"].between(38, 76)
            & (dataframe["btc_roc3_4h"] > -0.035)
        ).fillna(False)
        dataframe["btc_short_ok"] = (
            (dataframe["btc_close_4h"] < dataframe["btc_ema50_4h"])
            & (dataframe["btc_ema50_slope_4h"] < 0.004)
            & dataframe["btc_rsi_4h"].between(20, 62)
            & (dataframe["btc_roc3_4h"] < 0.04)
        ).fillna(False)
        dataframe["btc_risk_ok"] = (
            dataframe["btc_rsi_4h"].between(20, 82)
            & (dataframe["btc_roc3_4h"].abs() < 0.075)
        ).fillna(False)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        if metadata["pair"] not in self.trade_pairs:
            return dataframe

        vol_ok = dataframe["atr_pct"].between(0.003, 0.055)
        common = (
            dataframe["btc_risk_ok"]
            & (dataframe["volume_ratio"] >= 0.85)
            & vol_ok
        )

        long_signal = (
            common
            & dataframe["btc_long_ok"]
            & dataframe["pair_4h_long"]
            & (dataframe["close"] > dataframe["ema200"])
            & (dataframe["ema20"] > dataframe["ema50"])
            & (dataframe["ema50"] > dataframe["ema200"] * 0.998)
            & (dataframe["ema50_slope"] > 0)
            & (dataframe["adx"] >= 18)
            & dataframe["rsi"].between(50, 72)
            & (dataframe["close"] > dataframe["donchian_high"])
            & (dataframe["range_pos"] > 0.82)
        )
        short_signal = (
            common
            & dataframe["btc_short_ok"]
            & dataframe["pair_4h_short"]
            & (dataframe["close"] < dataframe["ema200"])
            & (dataframe["ema20"] < dataframe["ema50"])
            & (dataframe["ema50"] < dataframe["ema200"] * 1.002)
            & (dataframe["ema50_slope"] < 0)
            & (dataframe["adx"] >= 18)
            & dataframe["rsi"].between(28, 50)
            & (dataframe["close"] < dataframe["donchian_low"])
            & (dataframe["range_pos"] < 0.18)
        )

        dataframe.loc[short_signal, ["enter_short", "enter_tag"]] = (1, "hourly_breakout_short")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        long_exit = (
            ((dataframe["close"] < dataframe["ema50"]) & (dataframe["rsi"] < 46))
            | (dataframe["rsi"] > 78)
            | (~dataframe["btc_risk_ok"])
        )
        short_exit = (
            ((dataframe["close"] > dataframe["ema50"]) & (dataframe["rsi"] > 54))
            | (dataframe["rsi"] < 22)
            | (~dataframe["btc_risk_ok"])
        )
        dataframe.loc[long_exit, ["exit_long", "exit_tag"]] = (1, "hourly_long_exit")
        dataframe.loc[short_exit, ["exit_short", "exit_tag"]] = (1, "hourly_short_exit")
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
        return min(2.0, max_leverage)

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> str | bool | None:
        if current_profit >= 0.055:
            return "hourly_profit_take"

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if dataframe is None or dataframe.empty:
            return None

        last = dataframe.iloc[-1]
        duration_candles = (current_time - trade.open_date_utc).total_seconds() / (60 * 60)

        if current_profit <= -0.022:
            if trade.is_short:
                if last["close"] > last["ema20"] or last["rsi"] > 53:
                    return "hourly_loss_cut"
            else:
                if last["close"] < last["ema20"] or last["rsi"] < 47:
                    return "hourly_loss_cut"

        if duration_candles >= 30 and current_profit < 0.003:
            return "hourly_time_exit"

        return None
