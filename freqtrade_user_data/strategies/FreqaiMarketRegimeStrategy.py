from __future__ import annotations

from datetime import datetime

import talib.abstract as ta
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, merge_informative_pair
from freqtrade.vendor.qtpylib import indicators as qtpylib
from pandas import DataFrame


class FreqaiMarketRegimeStrategy(IStrategy):
    """
    FreqAI futures strategy for liquid USDT perpetuals.

    The model predicts the average forward return. Entries are only allowed
    when the prediction agrees with simple market-structure filters so the
    model is not allowed to trade every small statistical wiggle.
    """

    timeframe = "15m"
    can_short = True
    process_only_new_candles = True
    startup_candle_count = 240

    minimal_roi = {"0": 0.026}
    stoploss = -0.018
    trailing_stop = True
    trailing_stop_positive = 0.010
    trailing_stop_positive_offset = 0.018
    trailing_only_offset_is_reached = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    position_adjustment_enable = False

    trade_pair_whitelist = {"SOL/USDT:USDT", "ADA/USDT:USDT", "LTC/USDT:USDT"}

    prediction_zscore = 1.20
    min_prediction_edge = 0.0065
    max_prediction_edge = 0.07
    min_adx = 18.0
    long_rsi_min = 42.0
    long_rsi_max = 70.0
    short_rsi_min = 30.0
    short_rsi_max = 58.0
    min_atr_pct = 0.0015
    max_atr_pct = 0.022
    min_ema_dist = 0.002
    min_ema50_slope = 0.0005
    range_min_atr_pct = 0.0012
    range_max_atr_pct = 0.018
    min_trend_prediction_atr_ratio = 1.05
    min_range_prediction_atr_ratio = 0.45
    max_ema50_extension = 0.035
    min_range_target_space = 0.006

    def informative_pairs(self) -> list[tuple[str, str]]:
        return [("BTC/USDT:USDT", "1h")]

    @property
    def protections(self) -> list[dict]:
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 4},
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 48,
                "trade_limit": 3,
                "stop_duration_candles": 16,
                "only_per_pair": False,
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 192,
                "trade_limit": 20,
                "stop_duration_candles": 48,
                "max_allowed_drawdown": 0.08,
            },
        ]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.freqai.start(dataframe, metadata, self)

        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["atr_pct"] = ta.ATR(dataframe, timeperiod=14) / dataframe["close"]
        dataframe["volume_ratio"] = dataframe["volume"] / dataframe["volume"].rolling(20).mean()
        dataframe["ema_dist"] = (dataframe["close"] - dataframe["ema200"]) / dataframe["close"]
        dataframe["ema50_slope"] = dataframe["ema50"] / dataframe["ema50"].shift(8) - 1.0
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2.2)
        dataframe["bb_lower"] = bollinger["lower"]
        dataframe["bb_mid"] = bollinger["mid"]
        dataframe["bb_upper"] = bollinger["upper"]
        dataframe["bb_width"] = (bollinger["upper"] - bollinger["lower"]) / bollinger["mid"]
        dataframe["bb_lower_reclaim"] = (dataframe["low"] < dataframe["bb_lower"]) & (
            dataframe["close"] > dataframe["bb_lower"]
        )
        dataframe["bb_upper_reclaim"] = (dataframe["high"] > dataframe["bb_upper"]) & (
            dataframe["close"] < dataframe["bb_upper"]
        )

        base_target = self.min_prediction_edge
        dataframe["target_roi"] = base_target
        dataframe["sell_roi"] = -base_target
        if "&-s_close_mean" in dataframe and "&-s_close_std" in dataframe:
            dataframe["target_roi"] = (
                dataframe["&-s_close_mean"] + dataframe["&-s_close_std"] * self.prediction_zscore
            ).clip(lower=base_target, upper=self.max_prediction_edge)
            dataframe["sell_roi"] = (
                dataframe["&-s_close_mean"] - dataframe["&-s_close_std"] * self.prediction_zscore
            ).clip(lower=-self.max_prediction_edge, upper=-base_target)

        dataframe = self._merge_btc_market_filter(dataframe)
        return dataframe

    def _merge_btc_market_filter(self, dataframe: DataFrame) -> DataFrame:
        if not self.dp:
            dataframe["btc_long_ok"] = True
            dataframe["btc_short_ok"] = True
            dataframe["btc_range_ok"] = True
            dataframe["btc_risk_ok"] = True
            return dataframe

        btc = self.dp.get_pair_dataframe(pair="BTC/USDT:USDT", timeframe="1h")
        if btc.empty:
            dataframe["btc_long_ok"] = True
            dataframe["btc_short_ok"] = True
            dataframe["btc_range_ok"] = True
            dataframe["btc_risk_ok"] = True
            return dataframe

        btc = btc.copy()
        btc["btc_ema50"] = ta.EMA(btc, timeperiod=50)
        btc["btc_ema200"] = ta.EMA(btc, timeperiod=200)
        btc["btc_adx"] = ta.ADX(btc, timeperiod=14)
        btc["btc_rsi"] = ta.RSI(btc, timeperiod=14)
        btc["btc_ema50_slope"] = btc["btc_ema50"] / btc["btc_ema50"].shift(6) - 1.0
        btc["btc_roc_3"] = btc["close"].pct_change(3)
        btc["btc_close"] = btc["close"]
        btc = btc[
            [
                "date",
                "btc_close",
                "btc_ema50",
                "btc_ema200",
                "btc_adx",
                "btc_rsi",
                "btc_ema50_slope",
                "btc_roc_3",
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
        btc_close = "btc_close_1h"
        btc_ema50 = "btc_ema50_1h"
        btc_ema200 = "btc_ema200_1h"
        btc_adx = "btc_adx_1h"
        btc_rsi = "btc_rsi_1h"
        btc_ema50_slope = "btc_ema50_slope_1h"
        btc_roc_3 = "btc_roc_3_1h"
        dataframe["btc_long_ok"] = (
            (dataframe[btc_close] > dataframe[btc_ema200])
            & (dataframe[btc_ema50_slope] > -0.002)
            & (dataframe[btc_roc_3] > -0.012)
            & dataframe[btc_rsi].between(38, 74)
            & (dataframe[btc_adx] >= 12)
        )
        dataframe["btc_short_ok"] = (
            (dataframe[btc_close] < dataframe[btc_ema200])
            & (dataframe[btc_ema50_slope] < 0.002)
            & (dataframe[btc_roc_3] < 0.012)
            & dataframe[btc_rsi].between(26, 62)
            & (dataframe[btc_adx] >= 12)
        )
        dataframe["btc_range_ok"] = (
            dataframe[btc_rsi].between(34, 66)
            & (dataframe[btc_adx] <= 24)
            & (dataframe[btc_roc_3].abs() < 0.018)
            & (dataframe[btc_ema50_slope].abs() < 0.004)
        )
        dataframe["btc_risk_ok"] = (
            dataframe[btc_rsi].between(26, 76)
            & (dataframe[btc_roc_3].abs() < 0.035)
        )
        dataframe["btc_long_ok"] = dataframe["btc_long_ok"].fillna(False)
        dataframe["btc_short_ok"] = dataframe["btc_short_ok"].fillna(False)
        dataframe["btc_range_ok"] = dataframe["btc_range_ok"].fillna(False)
        dataframe["btc_risk_ok"] = dataframe["btc_risk_ok"].fillna(False)
        return dataframe

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs
    ) -> DataFrame:
        ema = ta.EMA(dataframe, timeperiod=period)
        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=period, stds=2.2
        )
        return dataframe.assign(
            **{
                "%-rsi-period": ta.RSI(dataframe, timeperiod=period),
                "%-adx-period": ta.ADX(dataframe, timeperiod=period),
                "%-ema_dist-period": (dataframe["close"] - ema) / dataframe["close"],
                "%-roc-period": ta.ROC(dataframe, timeperiod=period),
                "%-atr_pct-period": ta.ATR(dataframe, timeperiod=period) / dataframe["close"],
                "%-bb_width-period": (bollinger["upper"] - bollinger["lower"]) / bollinger["mid"],
                "%-relative_volume-period": dataframe["volume"] / dataframe["volume"].rolling(period).mean(),
            }
        )

    def feature_engineering_expand_basic(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        return dataframe.assign(
            **{
                "%-pct_change": dataframe["close"].pct_change(),
                "%-raw_volume": dataframe["volume"],
                "%-range_pct": (dataframe["high"] - dataframe["low"]) / dataframe["close"],
                "%-body_pct": (dataframe["close"] - dataframe["open"]) / dataframe["close"],
            }
        )

    def feature_engineering_standard(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        return dataframe.assign(
            **{
                "%-day_of_week": (dataframe["date"].dt.dayofweek + 1) / 7,
                "%-hour_of_day": (dataframe["date"].dt.hour + 1) / 25,
                "%-ema50_ema200_dist": (
                    ta.EMA(dataframe, timeperiod=50) - ta.EMA(dataframe, timeperiod=200)
                )
                / dataframe["close"],
                "%-volume_zscore": (
                    dataframe["volume"] - dataframe["volume"].rolling(48).mean()
                )
                / dataframe["volume"].rolling(48).std(),
            }
        )

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        label_period = self.freqai_info["feature_parameters"]["label_period_candles"]
        dataframe["&-s_close"] = (
            dataframe["close"].shift(-label_period).rolling(label_period).mean() / dataframe["close"] - 1.0
        )
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        if metadata["pair"] not in self.trade_pair_whitelist:
            return dataframe

        prediction_ok = dataframe["do_predict"] == 1
        volatility_ok = dataframe["atr_pct"].between(self.min_atr_pct, self.max_atr_pct)
        range_volatility_ok = dataframe["atr_pct"].between(self.range_min_atr_pct, self.range_max_atr_pct)
        long_structure_ok = (dataframe["ema_dist"] > self.min_ema_dist) & (
            dataframe["ema50_slope"] > self.min_ema50_slope
        )
        short_structure_ok = (dataframe["ema_dist"] < -self.min_ema_dist) & (
            dataframe["ema50_slope"] < -self.min_ema50_slope
        )
        long_edge_ok = (
            (dataframe["&-s_close"] > dataframe["target_roi"])
            & ((dataframe["&-s_close"] / dataframe["atr_pct"]) >= self.min_trend_prediction_atr_ratio)
        )
        short_edge_ok = (
            (dataframe["&-s_close"] < dataframe["sell_roi"])
            & ((-dataframe["&-s_close"] / dataframe["atr_pct"]) >= self.min_trend_prediction_atr_ratio)
        )
        long_extension_ok = (
            (dataframe["close"] > dataframe["ema50"])
            & (((dataframe["close"] - dataframe["ema50"]) / dataframe["close"]) <= self.max_ema50_extension)
        )
        short_extension_ok = (
            (dataframe["close"] < dataframe["ema50"])
            & (((dataframe["ema50"] - dataframe["close"]) / dataframe["close"]) <= self.max_ema50_extension)
        )
        range_structure_ok = (
            (dataframe["adx"] <= 24)
            & (dataframe["ema_dist"].abs() < 0.04)
            & (dataframe["bb_width"] < dataframe["bb_width"].rolling(96).quantile(0.75))
            & dataframe["btc_range_ok"]
        )
        strong_long_prediction = dataframe["&-s_close"] > dataframe["target_roi"] * 1.35
        strong_short_prediction = dataframe["&-s_close"] < dataframe["sell_roi"] * 1.35
        long_momentum_ok = (
            (dataframe["close"] > dataframe["close"].shift(1))
            | (dataframe["rsi"] > dataframe["rsi"].shift(1))
            | strong_long_prediction
        )
        short_momentum_ok = (
            (dataframe["close"] < dataframe["close"].shift(1))
            | (dataframe["rsi"] < dataframe["rsi"].shift(1))
            | strong_short_prediction
        )

        long_conditions = (
            prediction_ok
            & long_edge_ok
            & (dataframe["close"] > dataframe["ema200"])
            & (dataframe["ema50"] > dataframe["ema200"])
            & (dataframe["adx"] >= self.min_adx)
            & dataframe["rsi"].between(self.long_rsi_min, self.long_rsi_max)
            & (dataframe["volume_ratio"] >= 0.85)
            & long_momentum_ok
            & long_structure_ok
            & long_extension_ok
            & dataframe["btc_long_ok"]
            & dataframe["btc_risk_ok"]
            & volatility_ok
        )

        short_conditions = (
            prediction_ok
            & short_edge_ok
            & (dataframe["close"] < dataframe["ema200"])
            & (dataframe["ema50"] < dataframe["ema200"])
            & (dataframe["adx"] >= self.min_adx)
            & dataframe["rsi"].between(self.short_rsi_min, self.short_rsi_max)
            & (dataframe["volume_ratio"] >= 0.85)
            & short_momentum_ok
            & short_structure_ok
            & short_extension_ok
            & dataframe["btc_short_ok"]
            & dataframe["btc_risk_ok"]
            & volatility_ok
        )

        dataframe.loc[long_conditions, ["enter_long", "enter_tag"]] = (1, "freqai_trend_long")
        dataframe.loc[short_conditions, ["enter_short", "enter_tag"]] = (1, "freqai_trend_short")

        range_long_conditions = (
            prediction_ok
            & range_structure_ok
            & range_volatility_ok
            & dataframe["bb_lower_reclaim"]
            & (dataframe["rsi"] < 42)
            & ((dataframe["bb_mid"] - dataframe["close"]) / dataframe["close"] > self.min_range_target_space)
            & ((dataframe["&-s_close"] / dataframe["atr_pct"]) >= self.min_range_prediction_atr_ratio)
            & (dataframe["volume_ratio"] >= 0.65)
            & dataframe["btc_risk_ok"]
        )
        range_short_conditions = (
            prediction_ok
            & range_structure_ok
            & range_volatility_ok
            & dataframe["bb_upper_reclaim"]
            & (dataframe["rsi"] > 58)
            & ((dataframe["close"] - dataframe["bb_mid"]) / dataframe["close"] > self.min_range_target_space)
            & ((-dataframe["&-s_close"] / dataframe["atr_pct"]) >= self.min_range_prediction_atr_ratio)
            & (dataframe["volume_ratio"] >= 0.65)
            & dataframe["btc_risk_ok"]
        )

        dataframe.loc[range_long_conditions, ["enter_long", "enter_tag"]] = (
            1,
            "freqai_range_long",
        )
        dataframe.loc[range_short_conditions, ["enter_short", "enter_tag"]] = (
            1,
            "freqai_range_short",
        )
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        long_exit = (
            ((dataframe["&-s_close"] < dataframe["sell_roi"]) & (dataframe["rsi"] < 50))
            | (dataframe["rsi"] > 74)
            | ((dataframe["close"] < dataframe["ema50"]) & (dataframe["rsi"] < 45))
        )
        short_exit = (
            ((dataframe["&-s_close"] > dataframe["target_roi"]) & (dataframe["rsi"] > 50))
            | (dataframe["rsi"] < 26)
            | ((dataframe["close"] > dataframe["ema50"]) & (dataframe["rsi"] > 55))
        )

        dataframe.loc[long_exit, ["exit_long", "exit_tag"]] = (1, "freqai_long_invalidated")
        dataframe.loc[short_exit, ["exit_short", "exit_tag"]] = (1, "freqai_short_invalidated")
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
        if current_profit >= 0.018:
            return "hard_profit_take"
        if trade.enter_tag and "range" in trade.enter_tag and current_profit >= 0.008:
            return "range_profit_take"
        if trade.enter_tag and "range" in trade.enter_tag and current_profit <= -0.007:
            return "range_loss_cut"
        if trade.nr_of_successful_entries <= 1 and current_profit <= -0.014:
            return "hard_loss_cut"
        return None
