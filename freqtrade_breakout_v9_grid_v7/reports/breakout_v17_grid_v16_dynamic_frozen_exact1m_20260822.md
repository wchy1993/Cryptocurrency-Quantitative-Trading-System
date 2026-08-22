# Breakout V17 / Grid V16 Dynamic 冻结基线

日期：2026-08-22

冻结策略：`BreakoutV17GridV16DynamicRiskBalancedResearchFreqtrade`

策略版本：`breakout_v17_grid_v16_dynamic_risk_balanced_20260821`

## 冻结结论

本分支把当前正式回测基线冻结为 V17 策略。它包含策略完整继承链、定向测试、专用回测配置、exact-1m 原始结果归档和汇总工具。V19 动态候选、15m 压力结果、临时日志以及失败实验不属于本冻结版本。

验收口径：1h 主周期、1m 撮合明细、初始钱包 200U、`stake_amount=unlimited`、最多 2 个 campaign、期货逐仓、固定 Active50、关闭回测缓存。跨年度数据来自同一个连续钱包，没有逐年重置资金。

## Exact-1m 回测结果

| 区间 | 交易 | 胜 / 负 | 净利润 | PF | 胜率 | Freqtrade 摘要回撤 | 严格钱包回撤 | Sharpe | Sortino |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2025-08-11 至 2026-08-11 | 225 | 143 / 82 | 421,749.637U | 16.489 | 63.556% | 1.255% | 16.222% | 2.995 | 28.790 |
| 2024-01-01 至 2026-08-11 | 528 | 298 / 230 | 3,355,291.801U | 10.075084 | 56.439394% | 4.708230% | 22.309034% | 1.583340 | 12.243520 |

## 跨年度逐年结果

年度净利润按平仓年份汇总；年度收益率和严格回撤来自该年连续钱包切片。2026 只统计至 08-11。

| 年份 | 交易 | 胜 / 负 | 胜率 | 净利润 | 连续钱包收益率 | PF | 严格回撤 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2024 | 219 | 125 / 94 | 57.077626% | 10,732.109U | 5,366.054% | 3.599621 | 20.195952% |
| 2025 | 163 | 69 / 94 | 42.331288% | 9,334.234U | 86.253% | 1.623587 | 22.309034% |
| 2026 至 08-11 | 146 | 104 / 42 | 71.232877% | 3,335,225.459U | 16,379.688% | 10.512128 | 20.579439% |

## 跨年度多空与组件

| 分组 | 交易 | 占比 | 胜 / 负 | 胜率 | 净利润 | PF |
|---|---:|---:|---:|---:|---:|---:|
| 多单 | 136 | 25.757576% | 57 / 79 | 41.911765% | 2,088,897.879U | 14.292912 |
| 空单 | 392 | 74.242424% | 241 / 151 | 61.479592% | 1,266,393.922U | 6.957203 |
| Breakout | 231 | 43.750000% | 81 / 150 | 35.064935% | 2,972,660.454U | 12.366579 |
| Grid | 297 | 56.250000% | 217 / 80 | 73.063973% | 382,631.347U | 4.536357 |

## 归档与校验

- 最近一年 exact-1m：`user_data/backtest_results/v18_dynamic_position_20260821/risk_balanced_partial90_recent1y_exact1m-2026-08-21_13-09-30.zip`
- SHA-256：`DD65704D765FCAFE255B64831A62C18B6149ED42B5D4C55ADDD35694B1957FBE`
- 跨年度 exact-1m：`user_data/backtest_results/v19_dynamic_1m_20260821/backtest-result-2026-08-21_16-16-16.zip`
- SHA-256：`A3CB17D3BBC7FCEF214990D9B1E219BA3D2A43B1D59B0CF8E1EB1B141159F2B0`

跨年度归档虽然位于 `v19_dynamic_1m_20260821` 目录，但归档元数据中的实际策略是本冻结基线 `BreakoutV17GridV16DynamicRiskBalancedResearchFreqtrade`；V19 候选策略没有进入本分支。

## 复跑命令

```powershell
.\.runtime\python312\Scripts\freqtrade.exe backtesting `
  --config config.breakout-v17-grid-v16-dynamic-baseline.backtest.json `
  --strategy BreakoutV17GridV16DynamicRiskBalancedResearchFreqtrade `
  --timerange 20240101-20260811 `
  --timeframe-detail 1m `
  --cache none `
  --export trades `
  --export-directory user_data/backtest_results/v17_grid_v16_dynamic_frozen_recheck `
  --no-color
```

汇总已有归档：

```powershell
.\.runtime\python312\python.exe tools\summarize_v19_exact1m.py `
  user_data\backtest_results\v19_dynamic_1m_20260821\backtest-result-2026-08-21_16-16-16.zip
```

## 使用边界

- 1m OHLCV 仍不能模拟盘口深度、滑点、延迟、排队和部分成交。
- 固定 Active50 存在幸存者偏差和币池前视风险。
- 200U 无限复利后的绝对金额远超真实盘口容量，绝对 U 金额不能直接外推为实盘可成交收益。
- 正式实盘切换前仍需隔离 dry-run 和逐单对账。
