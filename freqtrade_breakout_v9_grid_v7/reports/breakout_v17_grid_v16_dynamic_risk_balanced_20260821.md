# V17 / Grid V16 动态仓位管理验收

日期：2026-08-21

## 结论

本轮选定的研究策略为：

`BreakoutV17GridV16DynamicRiskBalancedResearchFreqtrade`

冻结基线 `BreakoutV17GridV16ParetoFinalResearchFreqtrade` 及其回测归档未被替换或修改。新策略在最近一年精确 1m 主验收和 2024+ 跨年度 15m 压力验收中，所有约定指标均不低于 V17 / Grid V16 Final。

## 动态管理规则

1. Breakout 普通 score-4 多单若入场后 15 分钟最高进展不超过 0.10R，则减仓 90%，保留 10% 尾仓走原止损；只在已实现回撤 8%（含）至 10%（不含）的压力带启用。
2. Grid score-3 空单首次止盈后准备第一次重建时，若已完成 4h 动量绝对值不超过 0.20%，平掉剩余尾仓，不把已经释放的风险重新加回。
3. Grid score-4 空单三层仓位已打满、尚未止盈、持仓至少 120 分钟、已完成 4h 逆向动量至少 1.5%、campaign 不超过 -0.33R 且账户层亏损仍在 -6% 以内时，减仓 90%，保留 10% 尾仓走原止损。
4. Breakout 高分趋势 runner、未止盈 Grid、已有成功重建记录的 Grid 均不受上述退出影响。
5. 没有币种或日期硬编码，所有判断只使用已完成 K 线、交易状态和已实现账户状态。

## 最近一年精确 1m 主验收

区间：2025-08-11 至 2026-08-11；1h 主周期；1m detail；200U 初始钱包；最多 2 仓；缓存关闭。

| 指标 | V17 / Grid V16 Final | Dynamic Risk Balanced | 变化 |
|---|---:|---:|---:|
| 交易数 | 225 | 225 | 持平 |
| 净利润 | 411,480.153U | 421,749.637U | +10,269.483U / +2.50% |
| PF | 14.861 | 16.489 | +10.96% |
| 胜率 | 63.556% | 63.556% | 持平 |
| Freqtrade 汇总回撤 | 1.937% | 1.255% | -0.683 个百分点 |
| 严格小时钱包回撤 | 19.284% | 16.222% | -3.062 个百分点 |
| Sharpe | 2.973 | 2.995 | +0.75% |
| Sortino | 23.224 | 28.790 | +23.97% |

实际动态触发：

- HYPE：弱跟随 Breakout 减仓 90%，最终亏损由基线的 -12.520U 降至 -2.715U。
- XMR：首次 Grid 止盈后 4h 动量转平，尾仓退出；该笔为 +1,168.928U。
- ONDO：满层 Grid 严重逆向动量时减仓 90%，10% 尾仓继续走原 trailing；该笔为 -3,165.137U。

精确归档：

`user_data/backtest_results/v18_dynamic_position_20260821/risk_balanced_partial90_recent1y_exact1m-2026-08-21_13-09-30.zip`

SHA-256：`DD65704D765FCAFE255B64831A62C18B6149ED42B5D4C55ADDD35694B1957FBE`

## 2024+ 跨年度 15m 压力验收

区间：2024-01-01 至 2026-08-11；1h 主周期；15m detail；其他配置相同。

| 指标 | V17 / Grid V16 Final | Dynamic Risk Balanced | 变化 |
|---|---:|---:|---:|
| 交易数 | 518 | 519 | +1 |
| 净利润 | 2,823,053.406U | 2,893,024.847U | +69,971.441U / +2.48% |
| PF | 9.282 | 10.145 | +9.29% |
| 胜率 | 53.089% | 53.179% | +0.090 个百分点 |
| Freqtrade 汇总回撤 | 2.730% | 1.835% | -0.895 个百分点 |
| 严格小时钱包回撤 | 21.544% | 21.544% | 持平 |
| Sharpe | 1.391 | 1.426 | +2.57% |
| Sortino | 10.458 | 12.458 | +19.12% |

压力归档：

`user_data/backtest_results/v18_dynamic_position_20260821/risk_balanced_partial90_crossyear_2024_20260811_15m-2026-08-21_12-35-41.zip`

SHA-256：`560B9D56EE41B505FEB5C623A66F2B9AF2CBA0CE2DA32737D91438024A7158CC`

## 代码与验证

- 策略：`user_data/strategies/BreakoutV17GridV16DynamicPositionResearchFreqtrade.py`
- 定向测试：`tests/test_breakout_v17_grid_v16_dynamic_position.py`
- 测试：18 passed；唯一警告为工作区 `.pytest_cache` 无写权限，不影响结果。
- `py_compile` 与 `git diff --check` 通过。

## 使用边界

- 该版本仍是研究策略，不自动替换 GUI、LIVE 或 DRY-RUN 的冻结策略。
- 回测采用固定 Active50，存在幸存者偏差；无限复利后的绝对 U 金额不能直接外推为真实可成交规模。
- 1m / 15m OHLCV 无法完整模拟订单簿深度、滑点、延迟、排队和部分成交。
- 下一步应先做与冻结 Final 并行的 dry-run 对账，重点观察 partial fill、custom data 持久化、尾仓最小名义价值和实际 slot 占用。
