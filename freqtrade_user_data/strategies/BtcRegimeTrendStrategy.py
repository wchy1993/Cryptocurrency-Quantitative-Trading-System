from __future__ import annotations

from datetime import datetime

import talib.abstract as ta
from freqtrade.persistence import Trade
from freqtrade.strategy import BooleanParameter, DecimalParameter, IStrategy, IntParameter, merge_informative_pair
from pandas import DataFrame


class BtcRegimeTrendStrategy(IStrategy):
    """
    BTC-regime trend strategy for Binance USDT futures.

    The previous version became too restrictive and also had stale hyperopt
    parameters disabling shorts. This version trades only liquid majors, uses
    BTC 1h/4h as a risk filter, and requires the traded pair's own 1h trend to
    agree with the 15m entry.
    """

    timeframe = "15m"
    can_short = False
    process_only_new_candles = True
    startup_candle_count = 260

    minimal_roi = {"0": 0.055}
    stoploss = -0.022
    trailing_stop = True
    trailing_stop_positive = 0.010
    trailing_stop_positive_offset = 0.024
    trailing_only_offset_is_reached = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    position_adjustment_enable = False

    trade_pairs = {
        "ADA/USDT:USDT",
    }

    buy_enable_long = BooleanParameter(default=True, space="buy", optimize=False)
    buy_enable_short = BooleanParameter(default=False, space="buy", optimize=False)
    buy_adx = DecimalParameter(12.0, 30.0, decimals=1, default=22.0, space="buy", optimize=True)
    buy_btc_adx = DecimalParameter(8.0, 28.0, decimals=1, default=12.0, space="buy", optimize=True)
    buy_volume = DecimalParameter(0.60, 1.35, decimals=2, default=0.90, space="buy", optimize=True)
    buy_breakout_volume = DecimalParameter(0.90, 1.80, decimals=2, default=1.25, space="buy", optimize=True)
    buy_slope = DecimalParameter(0.0000, 0.0020, decimals=4, default=0.0002, space="buy", optimize=True)
    buy_touch = DecimalParameter(0.000, 0.018, decimals=3, default=0.006, space="buy", optimize=True)
    buy_max_extension = DecimalParameter(0.020, 0.080, decimals=3, default=0.045, space="buy", optimize=True)
    buy_rsi_long_min = IntParameter(36, 52, default=42, space="buy", optimize=True)
    buy_rsi_long_max = IntParameter(58, 76, default=69, space="buy", optimize=True)
    buy_rsi_short_min = IntParameter(24, 42, default=31, space="buy", optimize=True)
    buy_rsi_short_max = IntParameter(45, 64, default=57, space="buy", optimize=True)

    sell_take_profit = DecimalParameter(0.018, 0.050, decimals=3, default=0.034, space="sell", optimize=True)
    sell_loss_cut = DecimalParameter(0.010, 0.026, decimals=3, default=0.016, space="sell", optimize=True)
    sell_time_loss = IntParameter(8, 24, default=14, space="sell", optimize=True)
    sell_rsi_long_exit = IntParameter(66, 82, default=74, space="sell", optimize=True)
    sell_rsi_short_exit = IntParameter(18, 36, default=28, space="sell", optimize=True)

    @property
    def protections(self) -> list[dict]:
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 3},
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 64,
                "trade_limit": 3,
                "stop_duration_candles": 18,
                "only_per_pair": False,
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 192,
                "trade_limit": 18,
                "stop_duration_candles": 48,
                "max_allowed_drawdown": 0.055,
            },
        ]

    def informative_pairs(self) -> list[tuple[str, str]]:
        configured_pairs = set(self.trade_pairs)
        if getattr(self, "config", None):
            configured_pairs.update(self.config.get("exchange", {}).get("pair_whitelist", []))
        configured_pairs.add("BTC/USDT:USDT")

        informative = {(pair, "1h") for pair in configured_pairs}
        informative.update({(pair, "4h") for pair in self.trade_pairs})
        informative.add(("BTC/USDT:USDT", "4h"))
        return sorted(informative)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema10"] = ta.EMA(dataframe, timeperiod=10)
        dataframe["ema20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema100"] = ta.EMA(dataframe, timeperiod=100)
        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["atr_pct"] = ta.ATR(dataframe, timeperiod=14) / dataframe["close"]
        dataframe["volume_ratio"] = dataframe["volume"] / dataframe["volume"].rolling(20).mean()
        dataframe["ema20_slope"] = dataframe["ema20"] / dataframe["ema20"].shift(4) - 1.0
        dataframe["ema50_slope"] = dataframe["ema50"] / dataframe["ema50"].shift(8) - 1.0
        dataframe["ema50_extension"] = (dataframe["close"] - dataframe["ema50"]).abs() / dataframe["close"]
        dataframe["donchian_high"] = dataframe["high"].rolling(48).max().shift(1)
        dataframe["donchian_low"] = dataframe["low"].rolling(48).min().shift(1)
        dataframe["range_high"] = dataframe["high"].rolling(32).max()
        dataframe["range_low"] = dataframe["low"].rolling(32).min()
        dataframe["range_pos"] = (
            (dataframe["close"] - dataframe["range_low"])
            / (dataframe["range_high"] - dataframe["range_low"])
        )

        dataframe = self._merge_pair_trend(dataframe, metadata)
        dataframe = self._merge_btc_regime(dataframe)
        return dataframe

    def _set_default_regime(self, dataframe: DataFrame) -> DataFrame:
        dataframe["pair_long_regime"] = True
        dataframe["pair_short_regime"] = True
        dataframe["pair_macro_long"] = True
        dataframe["pair_momentum_up"] = True
        dataframe["pair_momentum_down"] = True
        return dataframe

    def _merge_pair_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not self.dp:
            return self._set_default_regime(dataframe)

        pair_1h = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe="1h")
        if pair_1h.empty:
            return self._set_default_regime(dataframe)

        pair_1h = pair_1h.copy()
        pair_1h["p1h_close"] = pair_1h["close"]
        pair_1h["p1h_ema20"] = ta.EMA(pair_1h, timeperiod=20)
        pair_1h["p1h_ema50"] = ta.EMA(pair_1h, timeperiod=50)
        pair_1h["p1h_ema200"] = ta.EMA(pair_1h, timeperiod=200)
        pair_1h["p1h_rsi"] = ta.RSI(pair_1h, timeperiod=14)
        pair_1h["p1h_adx"] = ta.ADX(pair_1h, timeperiod=14)
        pair_1h["p1h_roc3"] = pair_1h["close"].pct_change(3)
        pair_1h["p1h_ema50_slope"] = pair_1h["p1h_ema50"] / pair_1h["p1h_ema50"].shift(6) - 1.0
        pair_1h = pair_1h[
            [
                "date",
                "p1h_close",
                "p1h_ema20",
                "p1h_ema50",
                "p1h_ema200",
                "p1h_rsi",
                "p1h_adx",
                "p1h_roc3",
                "p1h_ema50_slope",
            ]
        ]

        dataframe = merge_informative_pair(
            dataframe,
            pair_1h,
            self.timeframe,
            "1h",
            ffill=True,
            append_timeframe=True,
        )
        dataframe["pair_long_regime"] = (
            (dataframe["p1h_close_1h"] > dataframe["p1h_ema200_1h"] * 1.005)
            & (dataframe["p1h_close_1h"] > dataframe["p1h_ema50_1h"])
            & (dataframe["p1h_ema20_1h"] > dataframe["p1h_ema50_1h"])
            & (dataframe["p1h_ema50_1h"] > dataframe["p1h_ema200_1h"] * 1.002)
            & (dataframe["p1h_ema50_slope_1h"] > 0.0005)
            & dataframe["p1h_rsi_1h"].between(40, 74)
        ).fillna(False)
        dataframe["pair_short_regime"] = (
            (dataframe["p1h_close_1h"] < dataframe["p1h_ema200_1h"] * 1.01)
            & (dataframe["p1h_ema50_1h"] < dataframe["p1h_ema200_1h"] * 1.015)
            & (dataframe["p1h_ema50_slope_1h"] < 0.004)
            & dataframe["p1h_rsi_1h"].between(22, 66)
        ).fillna(False)
        dataframe["pair_momentum_up"] = (
            (dataframe["p1h_close_1h"] > dataframe["p1h_ema20_1h"])
            & (dataframe["p1h_roc3_1h"] > -0.006)
        ).fillna(False)
        dataframe["pair_momentum_down"] = (
            (dataframe["p1h_close_1h"] < dataframe["p1h_ema20_1h"])
            & (dataframe["p1h_roc3_1h"] < 0.018)
        ).fillna(False)

        pair_4h = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe="4h")
        if pair_4h.empty:
            dataframe["pair_macro_long"] = dataframe["pair_long_regime"]
            return dataframe

        pair_4h = pair_4h.copy()
        pair_4h["p4h_close"] = pair_4h["close"]
        pair_4h["p4h_ema50"] = ta.EMA(pair_4h, timeperiod=50)
        pair_4h["p4h_ema200"] = ta.EMA(pair_4h, timeperiod=200)
        pair_4h["p4h_rsi"] = ta.RSI(pair_4h, timeperiod=14)
        pair_4h["p4h_roc3"] = pair_4h["close"].pct_change(3)
        pair_4h["p4h_ema50_slope"] = pair_4h["p4h_ema50"] / pair_4h["p4h_ema50"].shift(3) - 1.0
        pair_4h = pair_4h[
            [
                "date",
                "p4h_close",
                "p4h_ema50",
                "p4h_ema200",
                "p4h_rsi",
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
        dataframe["pair_macro_long"] = (
            (dataframe["p4h_close_4h"] > dataframe["p4h_ema50_4h"])
            & (dataframe["p4h_ema50_slope_4h"] > 0.0)
            & (dataframe["p4h_roc3_4h"] > 0.015)
            & dataframe["p4h_rsi_4h"].between(60, 75)
        ).fillna(False)
        return dataframe

    def _merge_btc_regime(self, dataframe: DataFrame) -> DataFrame:
        if not self.dp:
            dataframe["btc_long_ok"] = True
            dataframe["btc_short_ok"] = True
            dataframe["btc_risk_ok"] = True
            dataframe["btc_macro_long"] = True
            dataframe["btc_macro_short"] = True
            return dataframe

        btc = self.dp.get_pair_dataframe(pair="BTC/USDT:USDT", timeframe="1h")
        if btc.empty:
            dataframe["btc_long_ok"] = True
            dataframe["btc_short_ok"] = True
            dataframe["btc_risk_ok"] = True
            dataframe["btc_macro_long"] = True
            dataframe["btc_macro_short"] = True
            return dataframe

        btc = btc.copy()
        btc["btc_close"] = btc["close"]
        btc["btc_ema50"] = ta.EMA(btc, timeperiod=50)
        btc["btc_ema200"] = ta.EMA(btc, timeperiod=200)
        btc["btc_adx"] = ta.ADX(btc, timeperiod=14)
        btc["btc_rsi"] = ta.RSI(btc, timeperiod=14)
        btc["btc_roc3"] = btc["close"].pct_change(3)
        btc["btc_ema50_slope"] = btc["btc_ema50"] / btc["btc_ema50"].shift(6) - 1.0
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
        dataframe["btc_long_ok"] = (
            (dataframe["btc_close_1h"] > dataframe["btc_ema200_1h"] * 1.002)
            & (dataframe["btc_ema50_slope_1h"] > 0.0002)
            & (dataframe["btc_adx_1h"] >= float(self.buy_btc_adx.value))
            & dataframe["btc_rsi_1h"].between(40, 76)
            & (dataframe["btc_roc3_1h"] > -0.018)
        ).fillna(False)
        dataframe["btc_short_ok"] = (
            (dataframe["btc_close_1h"] < dataframe["btc_ema200_1h"] * 1.02)
            & (dataframe["btc_ema50_slope_1h"] < 0.005)
            & (dataframe["btc_adx_1h"] >= float(self.buy_btc_adx.value))
            & dataframe["btc_rsi_1h"].between(22, 68)
            & (dataframe["btc_roc3_1h"] < 0.04)
        ).fillna(False)
        dataframe["btc_risk_ok"] = (
            dataframe["btc_rsi_1h"].between(20, 82)
            & (dataframe["btc_roc3_1h"].abs() < 0.060)
        ).fillna(False)

        btc_4h = self.dp.get_pair_dataframe(pair="BTC/USDT:USDT", timeframe="4h")
        if btc_4h.empty:
            dataframe["btc_macro_long"] = dataframe["btc_long_ok"]
            dataframe["btc_macro_short"] = dataframe["btc_short_ok"]
            return dataframe

        btc_4h = btc_4h.copy()
        btc_4h["btc4h_close"] = btc_4h["close"]
        btc_4h["btc4h_ema50"] = ta.EMA(btc_4h, timeperiod=50)
        btc_4h["btc4h_ema200"] = ta.EMA(btc_4h, timeperiod=200)
        btc_4h["btc4h_rsi"] = ta.RSI(btc_4h, timeperiod=14)
        btc_4h["btc4h_roc3"] = btc_4h["close"].pct_change(3)
        btc_4h["btc4h_ema50_slope"] = btc_4h["btc4h_ema50"] / btc_4h["btc4h_ema50"].shift(3) - 1.0
        btc_4h = btc_4h[
            [
                "date",
                "btc4h_close",
                "btc4h_ema50",
                "btc4h_ema200",
                "btc4h_rsi",
                "btc4h_roc3",
                "btc4h_ema50_slope",
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
        dataframe["btc_macro_long"] = (
            (dataframe["btc4h_close_4h"] > dataframe["btc4h_ema200_4h"] * 0.99)
            & (dataframe["btc4h_close_4h"] > dataframe["btc4h_ema50_4h"])
            & (dataframe["btc4h_ema50_4h"] > dataframe["btc4h_ema200_4h"] * 0.985)
            & (dataframe["btc4h_ema50_slope_4h"] > -0.001)
            & (dataframe["btc4h_roc3_4h"] > -0.035)
            & dataframe["btc4h_rsi_4h"].between(38, 76)
        ).fillna(False)
        dataframe["btc_macro_short"] = (
            (dataframe["btc4h_close_4h"] < dataframe["btc4h_ema200_4h"] * 1.06)
            & (dataframe["btc4h_ema50_slope_4h"] < 0.010)
            & (dataframe["btc4h_roc3_4h"] < 0.080)
            & dataframe["btc4h_rsi_4h"].between(18, 74)
        ).fillna(False)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        if metadata["pair"] not in self.trade_pairs:
            return dataframe

        vol_ok = dataframe["atr_pct"].between(0.0014, 0.036)
        not_extended = dataframe["ema50_extension"] <= float(self.buy_max_extension.value)
        common = (
            dataframe["btc_risk_ok"]
            & (dataframe["volume_ratio"] >= float(self.buy_volume.value))
            & vol_ok
            & not_extended
        )

        long_base = (
            bool(self.buy_enable_long.value)
            & common
            & dataframe["btc_macro_long"]
            & dataframe["btc_long_ok"]
            & dataframe["pair_macro_long"]
            & dataframe["pair_long_regime"]
            & dataframe["pair_momentum_up"]
            & (dataframe["close"] > dataframe["ema100"])
            & (dataframe["ema20"] > dataframe["ema50"] * (1.0 - float(self.buy_slope.value)))
            & (dataframe["ema50_slope"] > -0.0015)
            & dataframe["rsi"].between(int(self.buy_rsi_long_min.value), int(self.buy_rsi_long_max.value))
        )
        short_base = (
            bool(self.buy_enable_short.value)
            & common
            & dataframe["btc_macro_short"]
            & dataframe["btc_short_ok"]
            & dataframe["pair_short_regime"]
            & dataframe["pair_momentum_down"]
            & (dataframe["close"] < dataframe["ema100"])
            & (dataframe["ema20"] < dataframe["ema50"] * (1.0 + float(self.buy_slope.value)))
            & (dataframe["ema50_slope"] < 0.0015)
            & dataframe["rsi"].between(int(self.buy_rsi_short_min.value), int(self.buy_rsi_short_max.value))
        )

        touch = float(self.buy_touch.value)
        pullback_long = (
            long_base
            & (dataframe["adx"] >= float(self.buy_adx.value))
            & (dataframe["low"] <= dataframe["ema20"] * (1.0 + touch))
            & (dataframe["close"] > dataframe["ema20"])
            & (dataframe["close"] > dataframe["open"])
            & (dataframe["rsi"] > dataframe["rsi"].shift(1))
        )
        breakout_long = (
            long_base
            & (dataframe["adx"] >= float(self.buy_adx.value))
            & (dataframe["ema20_slope"] > 0)
            & (dataframe["close"] > dataframe["donchian_high"])
            & (dataframe["volume_ratio"] >= float(self.buy_breakout_volume.value))
            & (dataframe["rsi"].between(52, 74))
            & (dataframe["range_pos"] > 0.82)
        )
        pullback_short = (
            short_base
            & (dataframe["adx"] >= float(self.buy_adx.value))
            & (dataframe["high"] >= dataframe["ema20"] * (1.0 - touch))
            & (dataframe["close"] < dataframe["ema20"])
            & (dataframe["close"] < dataframe["open"])
            & (dataframe["rsi"] < dataframe["rsi"].shift(1))
        )
        breakout_short = (
            short_base
            & (dataframe["adx"] >= float(self.buy_adx.value))
            & (dataframe["ema20_slope"] < 0)
            & (dataframe["close"] < dataframe["donchian_low"])
            & (dataframe["volume_ratio"] >= float(self.buy_breakout_volume.value))
            & (dataframe["rsi"].between(24, 48))
            & (dataframe["range_pos"] < 0.28)
        )

        dataframe.loc[breakout_long, ["enter_long", "enter_tag"]] = (1, "trend_breakout_long")
        dataframe.loc[breakout_short, ["enter_short", "enter_tag"]] = (1, "trend_breakout_short")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        long_exit = (
            ((dataframe["close"] < dataframe["ema50"]) & (dataframe["rsi"] < 44))
            | (dataframe["rsi"] > int(self.sell_rsi_long_exit.value))
            | (~dataframe["btc_risk_ok"])
        )
        short_exit = (
            ((dataframe["close"] > dataframe["ema50"]) & (dataframe["rsi"] > 56))
            | (dataframe["rsi"] < int(self.sell_rsi_short_exit.value))
            | (~dataframe["btc_risk_ok"])
        )
        dataframe.loc[long_exit, ["exit_long", "exit_tag"]] = (1, "trend_long_exit")
        dataframe.loc[short_exit, ["exit_short", "exit_tag"]] = (1, "trend_short_exit")
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
        if current_profit >= float(self.sell_take_profit.value):
            return "trend_profit_take"

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if dataframe is None or dataframe.empty:
            return None

        last = dataframe.iloc[-1]
        duration_candles = (current_time - trade.open_date_utc).total_seconds() / (15 * 60)

        if current_profit <= -float(self.sell_loss_cut.value):
            if trade.is_short:
                if last["close"] > last["ema20"] or last["rsi"] > 52:
                    return "trend_loss_cut"
            else:
                if last["close"] < last["ema20"] or last["rsi"] < 48:
                    return "trend_loss_cut"

        if duration_candles >= int(self.sell_time_loss.value) and current_profit < 0.002:
            return "trend_time_exit"

        return None
