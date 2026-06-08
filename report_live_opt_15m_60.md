# 15m 60币组合优化报告

- 最大回撤硬约束: 15.00%
- 最小交易数: 300
- 最优候选是否满足约束: 是
- 数据: `data/binance_15m_30d` 60 个 Binance USD-M 15m CSV。
- 执行限制: 本轮只做本地组合回测，没有连接真实账户，也没有真实下单。

## 最优结果
- 初始权益: 120.00 U
- 最终权益: 375.16 U
- 净收益: 212.63%
- 折算月收益: 215.81%
- 最大回撤: 11.51%
- 总交易: 1688
- 胜率: 85.25%
- Profit Factor: 1.47
- 多头: 973 笔，PnL 143.24 U，PF 1.41
- 空头: 715 笔，PnL 111.92 U，PF 1.56

## 最优参数
```json
{
  "trading": {
    "timeframe": "15m",
    "max_open_positions": 3,
    "max_new_entries_per_cycle": 1,
    "symbol_reentry_cooldown_seconds": 7200,
    "initial_entry_fraction": 0.75,
    "scale_in_entry_fraction": 0.25,
    "max_scale_ins_per_symbol": 0,
    "scale_in_min_profit_pct": 0.003,
    "scale_in_cooldown_seconds": 7200,
    "breakeven_trigger_pct": 0.002,
    "breakeven_lock_pct": 0.0018,
    "trailing_activation_pct": 0.005,
    "trailing_pullback_pct": 0.0035,
    "momentum_exit_min_profit_pct": 0.0045,
    "quick_take_profit_pct": 0.01,
    "strong_take_profit_pct": 0.014
  },
  "strategy": {
    "fast_ema": 18,
    "slow_ema": 55,
    "atr_period": 14,
    "channel_period": 20,
    "min_atr_pct": 0.004,
    "max_atr_pct": 0.02,
    "breakout_buffer_atr": 0.0,
    "ema_gap_atr": 0.0,
    "volume_period": 48,
    "min_volume_ratio": 1.2,
    "stop_loss_atr": 2.5,
    "take_profit_atr": 0.9,
    "breakeven_atr": 1.5,
    "trailing_activation_atr": 0.5,
    "trailing_stop_atr": 0.5,
    "max_holding_bars": 12,
    "spike_guard_enabled": false,
    "spike_min_range_atr": 4.0,
    "spike_min_wick_atr": 1.0,
    "spike_min_wick_ratio": 0.7,
    "spike_min_volume_ratio": 1.2,
    "spike_block_bars": 3,
    "spike_trade_enabled": true,
    "spike_recovery_ratio": 0.35,
    "spike_stop_atr": 0.5,
    "spike_take_profit_atr": 1.2,
    "spike_risk_multiplier": 0.35,
    "spike_max_holding_bars": 4,
    "allow_short": true,
    "long_score_threshold": 0.45,
    "short_score_threshold": 0.45,
    "long_risk_bias": 1.0,
    "short_risk_bias": 0.35,
    "regime_filter_enabled": false,
    "regime_lookback": 12,
    "long_min_slow_slope_atr": -0.75,
    "short_max_slow_slope_atr": 0.75
  },
  "filters": {
    "enabled": true,
    "timeframes": [
      "15m",
      "30m",
      "1h"
    ],
    "min_score": 5,
    "reversal_cross_lookback_bars": 2,
    "confirmed_cross_risk_multiplier": 0.5,
    "pre_cross_risk_multiplier": 0.25,
    "rsi_long_floor": 32.0,
    "rsi_long_ceiling": 76.0,
    "rsi_short_floor": 28.0,
    "rsi_short_ceiling": 64.0
  },
  "risk": {
    "starting_capital_usdt": 120.0,
    "max_account_margin_usage_pct": 0.05,
    "max_symbol_margin_pct": 0.04,
    "risk_per_trade_pct": 0.04,
    "max_daily_loss_pct": 0.15,
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
| 1 | 27 | 1711.27 | Y | 212.63 | 215.81 | 11.51 | 85.25 | 1.47 | 1688 | 143.24 | 111.92 |
| 2 | 6 | 1654.36 | Y | 205.80 | 208.88 | 8.77 | 84.37 | 1.32 | 3083 | 45.60 | 201.36 |
| 3 | 5 | 1505.18 | Y | 187.77 | 190.57 | 12.98 | 87.04 | 1.32 | 2353 | 46.17 | 179.16 |
| 4 | 8 | 1505.09 | Y | 185.97 | 188.75 | 5.52 | 80.96 | 1.47 | 2716 | 117.50 | 105.67 |
| 5 | 20 | 993.04 | Y | 121.81 | 123.62 | 6.38 | 86.79 | 1.53 | 1976 | 59.50 | 86.67 |
| 6 | 12 | 989.68 | Y | 123.59 | 125.44 | 11.32 | 84.61 | 1.21 | 2274 | 43.52 | 104.79 |
| 7 | 29 | 479.28 | Y | 58.56 | 59.43 | 5.20 | 81.97 | 1.28 | 2157 | 32.18 | 38.08 |
| 8 | 30 | 288.17 | Y | 36.02 | 36.55 | 10.35 | 80.38 | 1.17 | 1371 | 11.17 | 32.05 |
| 9 | 24 | 141.74 | Y | 17.23 | 17.49 | 7.07 | 82.77 | 1.16 | 969 | 7.91 | 12.77 |
| 10 | 2 | 73.62 | Y | 9.81 | 9.96 | 8.30 | 76.68 | 1.08 | 1102 | -2.82 | 14.59 |
