# 30m 组合优化报告

- 最大回撤硬约束: 20.00%
- 最小交易数: 500
- 最优候选是否满足约束: 是
- 数据: `data/binance_30m_180d` Binance USD-M 30m CSV。
- 执行限制: 本轮只做本地组合回测，没有连接真实账户，也没有真实下单。

## 最优结果
- 初始权益: 100.00 U
- 最终权益: 3414.02 U
- 净收益: 3314.02%
- 折算月收益: 560.46%
- 最大回撤: 16.18%
- 总交易: 5965
- 胜率: 87.83%
- Profit Factor: 1.56
- 多头: 3575 笔，PnL 2359.91 U，PF 1.54
- 空头: 2390 笔，PnL 954.11 U，PF 1.64

## 最优参数
```json
{
  "trading": {
    "timeframe": "30m",
    "max_open_positions": 8,
    "max_new_entries_per_cycle": 1,
    "symbol_reentry_cooldown_seconds": 3600,
    "initial_entry_fraction": 0.75,
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
    "super_volume_min_ratio": 3.5,
    "super_volume_min_breakout_atr": 0.85,
    "super_volume_min_body_atr": 0.35,
    "super_volume_confidence_boost": 0.2,
    "super_volume_risk_multiplier": 1.5,
    "super_volume_take_profit_multiplier": 1.6
  },
  "filters": {
    "enabled": true,
    "timeframes": [
      "30m",
      "1h",
      "2h"
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
    "starting_capital_usdt": 100.0,
    "max_account_margin_usage_pct": 0.1,
    "max_symbol_margin_pct": 0.04,
    "risk_per_trade_pct": 0.04,
    "max_daily_loss_pct": 0.08,
    "max_drawdown_pct": 0.2,
    "starting_capital_drawdown_stop_pct": 0.2,
    "weekly_profit_drawdown_stop_pct": 0.15,
    "soft_drawdown_reduce_pct": 0.04,
    "soft_drawdown_stop_pct": 0.15,
    "min_profit_after_cost_pct": 0.002
  }
}
```

## 候选排名
| 排名 | Trial | Score | 合格 | 收益% | 月化% | 回撤% | 胜率% | PF | 交易 | 多头PnL | 空头PnL |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 23235.42 | Y | 3314.02 | 560.46 | 16.18 | 87.83 | 1.56 | 5965 | 2359.91 | 954.11 |
| 2 | 5 | 42.90 | N | 4.63 | 0.78 | 6.91 | 78.57 | 1.37 | 84 | 0.54 | 4.09 |
| 3 | 4 | 21.53 | N | 3.92 | 0.66 | 8.16 | 83.65 | 1.11 | 312 | 3.41 | 0.51 |
| 4 | 8 | -107.04 | N | -2.54 | -0.43 | 5.09 | 60.47 | 0.86 | 172 | 0.00 | -2.54 |
| 5 | 3 | -121.37 | N | -4.65 | -0.79 | 9.49 | 62.96 | 0.89 | 189 | 0.00 | -4.65 |
| 6 | 7 | -165.12 | N | -6.53 | -1.10 | 8.57 | 56.49 | 0.77 | 154 | 0.00 | -6.53 |
| 7 | 2 | -227.66 | N | -12.48 | -2.11 | 12.73 | 56.35 | 0.70 | 252 | -5.48 | -7.00 |
| 8 | 6 | -816.34 | N | 4.50 | 0.76 | 26.07 | 80.77 | 1.01 | 6510 | 7.05 | -2.54 |
