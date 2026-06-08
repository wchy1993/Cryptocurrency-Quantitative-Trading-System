# 15m 60币组合优化报告

- 最大回撤硬约束: 15.00%
- 最小交易数: 300
- 最优候选是否满足约束: 是
- 数据: `data/binance_15m_30d` 60 个 Binance USD-M 15m CSV。
- 执行限制: 本轮只做本地组合回测，没有连接真实账户，也没有真实下单。

## 最优结果
- 初始权益: 120.00 U
- 最终权益: 538.29 U
- 净收益: 348.57%
- 折算月收益: 353.78%
- 最大回撤: 11.46%
- 总交易: 3285
- 胜率: 82.53%
- Profit Factor: 1.38
- 多头: 1718 笔，PnL 107.88 U，PF 1.16
- 空头: 1567 笔，PnL 310.41 U，PF 1.77

## 最优参数
```json
{
  "trading": {
    "timeframe": "15m",
    "max_open_positions": 5,
    "max_new_entries_per_cycle": 2,
    "symbol_reentry_cooldown_seconds": 7200,
    "initial_entry_fraction": 0.75,
    "scale_in_entry_fraction": 0.15,
    "max_scale_ins_per_symbol": 2,
    "scale_in_min_profit_pct": 0.008,
    "scale_in_cooldown_seconds": 7200,
    "breakeven_trigger_pct": 0.002,
    "breakeven_lock_pct": 0.0012,
    "trailing_activation_pct": 0.005,
    "trailing_pullback_pct": 0.0025,
    "momentum_exit_min_profit_pct": 0.002,
    "quick_take_profit_pct": 0.005,
    "strong_take_profit_pct": 0.014
  },
  "strategy": {
    "fast_ema": 18,
    "slow_ema": 89,
    "atr_period": 21,
    "channel_period": 48,
    "min_atr_pct": 0.0025,
    "max_atr_pct": 0.02,
    "breakout_buffer_atr": 0.0,
    "ema_gap_atr": 0.0,
    "volume_period": 20,
    "min_volume_ratio": 1.5,
    "stop_loss_atr": 2.0,
    "take_profit_atr": 0.9,
    "breakeven_atr": 0.5,
    "trailing_activation_atr": 1.0,
    "trailing_stop_atr": 0.8,
    "max_holding_bars": 12,
    "spike_guard_enabled": true,
    "spike_min_range_atr": 2.5,
    "spike_min_wick_atr": 2.0,
    "spike_min_wick_ratio": 0.7,
    "spike_min_volume_ratio": 1.6,
    "spike_block_bars": 3,
    "spike_trade_enabled": false,
    "spike_recovery_ratio": 0.45,
    "spike_stop_atr": 0.7,
    "spike_take_profit_atr": 0.9,
    "spike_risk_multiplier": 0.35,
    "spike_max_holding_bars": 12,
    "allow_short": true,
    "long_score_threshold": 0.55,
    "short_score_threshold": 0.65,
    "long_risk_bias": 0.75,
    "short_risk_bias": 0.5,
    "regime_filter_enabled": false,
    "regime_lookback": 24,
    "long_min_slow_slope_atr": -0.25,
    "short_max_slow_slope_atr": 0.75
  },
  "filters": {
    "enabled": true,
    "timeframes": [
      "15m",
      "30m",
      "1h"
    ],
    "min_score": 6,
    "reversal_cross_lookback_bars": 2,
    "confirmed_cross_risk_multiplier": 0.65,
    "pre_cross_risk_multiplier": 0.15,
    "rsi_long_floor": 32.0,
    "rsi_long_ceiling": 76.0,
    "rsi_short_floor": 28.0,
    "rsi_short_ceiling": 68.0
  },
  "risk": {
    "starting_capital_usdt": 120.0,
    "max_account_margin_usage_pct": 0.1,
    "max_symbol_margin_pct": 0.03,
    "risk_per_trade_pct": 0.05,
    "max_daily_loss_pct": 0.1,
    "max_drawdown_pct": 0.15,
    "soft_drawdown_reduce_pct": 0.04,
    "soft_drawdown_stop_pct": 0.15,
    "min_profit_after_cost_pct": 0.0015
  }
}
```

## 候选排名
| 排名 | Trial | Score | 合格 | 收益% | 月化% | 回撤% | 胜率% | PF | 交易 | 多头PnL | 空头PnL |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 15 | 2795.47 | Y | 348.57 | 353.78 | 11.46 | 82.53 | 1.38 | 3285 | 107.88 | 310.41 |
| 2 | 3 | 2256.72 | Y | 279.53 | 283.70 | 6.64 | 82.41 | 1.60 | 2922 | 125.54 | 209.89 |
| 3 | 1 | 1711.27 | Y | 212.63 | 215.81 | 11.51 | 85.25 | 1.47 | 1688 | 143.24 | 111.92 |
| 4 | 6 | 1558.85 | Y | 192.47 | 195.34 | 4.94 | 84.87 | 1.49 | 3060 | 112.79 | 118.17 |
| 5 | 16 | 728.37 | Y | 90.50 | 91.85 | 9.55 | 80.83 | 1.25 | 1753 | 36.61 | 71.98 |
| 6 | 2 | 363.91 | Y | 44.29 | 44.95 | 4.64 | 78.80 | 1.23 | 2524 | 3.97 | 49.18 |
| 7 | 8 | 123.56 | Y | 15.92 | 16.16 | 8.45 | 79.85 | 1.11 | 1911 | -1.92 | 21.03 |
| 8 | 18 | 110.54 | N | 12.95 | 13.14 | 7.02 | 83.64 | 1.48 | 220 | 18.90 | -3.36 |
| 9 | 4 | 95.15 | Y | 12.12 | 12.30 | 9.16 | 77.03 | 1.08 | 1010 | 2.39 | 12.15 |
| 10 | 19 | 40.98 | N | 5.35 | 5.43 | 6.63 | 85.84 | 1.22 | 219 | 10.15 | -3.74 |
