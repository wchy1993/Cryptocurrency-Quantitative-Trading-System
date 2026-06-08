# 15m 60币组合优化报告

- 最大回撤硬约束: 15.00%
- 最小交易数: 200
- 最优候选是否满足约束: 是
- 数据: `data/binance_15m_100d` 60 个 Binance USD-M 15m CSV。
- 执行限制: 本轮只做本地组合回测，没有连接真实账户，也没有真实下单。

## 最优结果
- 初始权益: 120.00 U
- 最终权益: 295.51 U
- 净收益: 146.26%
- 折算月收益: 44.52%
- 最大回撤: 10.54%
- 总交易: 5882
- 胜率: 83.73%
- Profit Factor: 1.17
- 多头: 3499 笔，PnL 113.99 U，PF 1.16
- 空头: 2383 笔，PnL 61.52 U，PF 1.19

## 最优参数
```json
{
  "trading": {
    "timeframe": "15m",
    "max_open_positions": 5,
    "max_new_entries_per_cycle": 1,
    "symbol_reentry_cooldown_seconds": 7200,
    "initial_entry_fraction": 0.6,
    "scale_in_entry_fraction": 0.15,
    "max_scale_ins_per_symbol": 1,
    "scale_in_min_profit_pct": 0.008,
    "scale_in_cooldown_seconds": 7200,
    "breakeven_trigger_pct": 0.002,
    "breakeven_lock_pct": 0.0008,
    "trailing_activation_pct": 0.012,
    "trailing_pullback_pct": 0.0035,
    "momentum_exit_min_profit_pct": 0.0045,
    "quick_take_profit_pct": 0.0075,
    "strong_take_profit_pct": 0.024
  },
  "strategy": {
    "fast_ema": 8,
    "slow_ema": 89,
    "atr_period": 14,
    "channel_period": 72,
    "min_atr_pct": 0.004,
    "max_atr_pct": 0.035,
    "breakout_buffer_atr": 0.1,
    "ema_gap_atr": 0.35,
    "volume_period": 30,
    "min_volume_ratio": 1.5,
    "stop_loss_atr": 2.5,
    "take_profit_atr": 2.4,
    "breakeven_atr": 1.5,
    "trailing_activation_atr": 0.5,
    "trailing_stop_atr": 0.5,
    "max_holding_bars": 24,
    "spike_guard_enabled": false,
    "spike_min_range_atr": 2.5,
    "spike_min_wick_atr": 1.0,
    "spike_min_wick_ratio": 0.7,
    "spike_min_volume_ratio": 1.6,
    "spike_block_bars": 3,
    "spike_trade_enabled": true,
    "spike_recovery_ratio": 0.45,
    "spike_stop_atr": 0.7,
    "spike_take_profit_atr": 1.2,
    "spike_risk_multiplier": 0.5,
    "spike_max_holding_bars": 12,
    "allow_short": true,
    "long_score_threshold": 0.85,
    "short_score_threshold": 0.55,
    "long_risk_bias": 0.75,
    "short_risk_bias": 0.35,
    "regime_filter_enabled": false,
    "regime_lookback": 48,
    "long_min_slow_slope_atr": -0.75,
    "short_max_slow_slope_atr": 1.5
  },
  "filters": {
    "enabled": true,
    "timeframes": [
      "15m",
      "30m",
      "1h"
    ],
    "min_score": 5,
    "extreme_reversal_entry_enabled": true,
    "pre_cross_entry_enabled": false,
    "reversal_cross_lookback_bars": 2,
    "confirmed_cross_risk_multiplier": 0.5,
    "pre_cross_risk_multiplier": 0.35,
    "rsi_long_floor": 30.0,
    "rsi_long_ceiling": 76.0,
    "rsi_short_floor": 28.0,
    "rsi_short_ceiling": 68.0
  },
  "risk": {
    "starting_capital_usdt": 120.0,
    "max_account_margin_usage_pct": 0.06,
    "max_symbol_margin_pct": 0.04,
    "risk_per_trade_pct": 0.03,
    "max_daily_loss_pct": 0.08,
    "max_drawdown_pct": 0.15,
    "soft_drawdown_reduce_pct": 0.04,
    "soft_drawdown_stop_pct": 0.15,
    "min_profit_after_cost_pct": 0.002
  }
}
```

## 候选排名
| 排名 | Trial | Score | 合格 | 收益% | 月化% | 回撤% | 胜率% | PF | 交易 | 多头PnL | 空头PnL |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 18 | 1029.99 | Y | 146.26 | 44.52 | 10.54 | 83.73 | 1.17 | 5882 | 113.99 | 61.52 |
| 2 | 28 | -9.92 | N | 0.47 | 0.14 | 7.18 | 86.04 | 1.02 | 265 | 0.00 | 0.57 |
| 3 | 20 | -66.63 | N | -0.78 | -0.24 | 8.75 | 82.11 | 0.95 | 95 | 0.00 | -0.93 |
| 4 | 27 | -70.86 | N | -2.15 | -0.65 | 7.17 | 81.37 | 0.95 | 306 | 1.40 | -3.98 |
| 5 | 30 | -108.32 | N | -4.02 | -1.22 | 5.89 | 63.35 | 0.84 | 251 | -5.53 | 0.70 |
| 6 | 12 | -110.03 | N | -5.60 | -1.71 | 7.38 | 70.40 | 0.89 | 375 | 0.89 | -7.62 |
| 7 | 19 | -111.92 | N | -6.35 | -1.93 | 10.38 | 78.28 | 0.93 | 861 | -6.89 | -0.73 |
| 8 | 15 | -115.41 | N | -4.33 | -1.32 | 5.15 | 61.71 | 0.81 | 269 | -0.83 | -4.37 |
| 9 | 22 | -123.49 | N | -7.05 | -2.14 | 14.18 | 77.17 | 0.93 | 635 | -5.57 | -2.88 |
| 10 | 26 | -131.43 | N | -6.47 | -1.97 | 8.27 | 78.89 | 0.83 | 199 | 0.00 | -7.76 |
| 11 | 24 | -138.18 | N | -7.87 | -2.40 | 13.76 | 65.14 | 0.93 | 743 | -22.48 | 13.04 |
| 12 | 14 | -139.31 | N | -5.09 | -1.55 | 7.58 | 75.43 | 0.76 | 175 | -7.74 | 1.63 |
