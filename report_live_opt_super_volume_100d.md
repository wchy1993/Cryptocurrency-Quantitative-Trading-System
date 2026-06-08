# 15m 60币组合优化报告

- 最大回撤硬约束: 15.00%
- 最小交易数: 300
- 最优候选是否满足约束: 是
- 数据: `data/binance_15m_100d` 60 个 Binance USD-M 15m CSV。
- 执行限制: 本轮只做本地组合回测，没有连接真实账户，也没有真实下单。

## 最优结果
- 初始权益: 10000.00 U
- 最终权益: 20115.35 U
- 净收益: 101.15%
- 折算月收益: 30.79%
- 最大回撤: 10.57%
- 总交易: 5597
- 胜率: 83.69%
- Profit Factor: 1.16
- 多头: 3289 笔，PnL 6046.67 U，PF 1.15
- 空头: 2308 笔，PnL 4068.68 U，PF 1.18

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
    "short_max_slow_slope_atr": 1.5,
    "super_volume_breakout_enabled": true,
    "super_volume_min_ratio": 3.0,
    "super_volume_min_breakout_atr": 0.75,
    "super_volume_min_body_atr": 0.35,
    "super_volume_confidence_boost": 0.15,
    "super_volume_risk_multiplier": 1.35,
    "super_volume_take_profit_multiplier": 1.35
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
| 1 | 1 | 713.23 | Y | 101.15 | 30.79 | 10.57 | 83.69 | 1.16 | 5597 | 6046.67 | 4068.68 |
| 2 | 3 | 54.77 | Y | 10.30 | 3.13 | 7.49 | 83.97 | 1.14 | 686 | -672.53 | 1702.14 |
| 3 | 15 | 39.70 | Y | 6.84 | 2.08 | 5.24 | 84.92 | 1.21 | 451 | -96.92 | 780.84 |
| 4 | 9 | -27.68 | N | 1.80 | 0.55 | 9.22 | 78.85 | 1.06 | 156 | -412.62 | 592.20 |
| 5 | 11 | -34.99 | N | 0.68 | 0.21 | 7.66 | 74.67 | 1.02 | 300 | -110.28 | 177.99 |
| 6 | 6 | -119.68 | N | -4.32 | -1.32 | 6.90 | 65.01 | 0.90 | 483 | 0.00 | -432.39 |
| 7 | 4 | -129.97 | N | -4.31 | -1.31 | 6.88 | 63.06 | 0.87 | 222 | 524.43 | -955.02 |
| 8 | 5 | -133.53 | N | -3.88 | -1.18 | 7.29 | 71.95 | 0.94 | 524 | -228.05 | -159.67 |
| 9 | 16 | -160.26 | N | -4.75 | -1.45 | 4.96 | 65.97 | 0.74 | 191 | -555.90 | 81.07 |
| 10 | 14 | -166.06 | N | -4.06 | -1.24 | 4.74 | 71.55 | 0.72 | 116 | -546.47 | 140.45 |
