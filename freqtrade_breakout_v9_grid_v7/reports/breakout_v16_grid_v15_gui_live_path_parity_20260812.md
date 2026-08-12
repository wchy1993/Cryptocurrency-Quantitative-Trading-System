# Breakout V16 + Grid V15 GUI / 回测路径一致性验收

验收日期：2026-08-12

## 结论

通过。GUI、DRY-RUN、LIVE 与正式回测配置均加载
`BreakoutV16GridV15QualityPfCombinedLiveParityFreqtrade`。该生产类直接继承
冻结的 `BreakoutV16GridV15QualityPfCombinedResearchFreqtrade`，没有重新实现
信号、跨币排序、Breakout 优先仲裁、仓位、杠杆、Grid DCA、止损或退出。

账户契约保持为共享 Max2：最多一个 Breakout 和一个 Grid；只剩最后一格时
Breakout 优先。生产类在 LIVE/DRY-RUN 中只增加数据完整性和墙钟安全门。

## 同时间轴逐笔回放

- 时间：2026-07-01 00:00 UTC 至 2026-08-01 00:00 UTC
- 币池：与 GUI 完全相同的 Binance USDT-M 固定 50 币
- 周期：1h 信号、V16 因果 15m 路径、1m 细节回放
- 钱包：200U，无限仓位复利
- 保证金：逐仓；策略原始动态杠杆逻辑
- 最大仓位：2（Breakout 1 + Grid 1）
- 成本：每侧 0.08%
- 缓存：关闭
- Freqtrade protection handlers：策略未定义

| 路径 | 交易数 | 净利润 | 总收益 | PF | 胜率 | 钱包路径最大回撤 |
|---|---:|---:|---:|---:|---:|---:|
| 冻结研究组合类 | 33 | +246.90822429U | +123.45% | 5.238838 | 66.67% | 9.3605% |
| GUI/LIVE 生产类 | 33 | +246.90822429U | +123.45% | 5.238838 | 66.67% | 9.3605% |

Freqtrade 闭仓汇总口径回撤亦完全一致，为 8.7691%。回测截止时的一笔
`force_exit` 同样出现在两条路径中。

两组完整导出交易对象逐字段完全相等，包括币种、时间、价格、方向、标签、
退出原因、仓位、数量、杠杆、盈亏、资金费和全部订单明细。将交易数组以
UTF-8、键排序、紧凑 JSON 规范化后的共同 SHA-256 为：

`3750a2f018bbdf78cbebe3d3eaea7ccd1ba5ced2061f126c66148e092aec82ad`

验收归档：
`user_data/backtest_results/v16_grid_v15_gui_path_parity_20260812/backtest-result-2026-08-12_13-19-32.zip`

归档 SHA-256：
`77ccb00d20b523318890ae8d3dfd7e16216c7691abde81be81127856917f89a0`

冻结源码 SHA-256：

- 研究组合类：`263a7be9a2506c8b48fcb5d94e8d41ffc4092cbafd3d8170d6ecf02cc6729e30`
- GUI/LIVE 类：`3a3b40b23cd34e06cd8cc95fb8c0319591e0419cf78991331e929e37d90964ce`

归档内附带的两份源码与上述哈希一致。

## LIVE 专属一致性保护

- Breakout 与 Grid 的入口都要求 50 个币完成同一个刚收盘的 1h 批次。
- 缺失、时间戳不一致、部分同步或整点后超过 90 秒时，本轮开仓失败关闭。
- Breakout 还必须确认信号小时的四根 15m 路径全部完成；缺一根即拒绝开仓，
  避免在没有 V16 no-follow / runner 保护状态时进入实盘。
- Grid 不依赖 V16 15m 路径，但仍必须通过同一 1h 跨币排序批次。
- 回测仅绕过真实墙钟和实时数据可用性门，其余方法均走同一个冻结父类。
- 首次 Breakout 成交将 `v16_no_follow_watch` 写入交易自定义数据；Grid 的
  DCA/TP 计数继续使用原有按订单 ID 幂等的持久化路径。

## GUI 与账本隔离

- DRY-RUN：`user_data/tradesv3.breakout_v16_grid_v15_combined.dryrun.sqlite`
- LIVE：`user_data/tradesv3.breakout_v16_grid_v15_combined.live.sqlite`
- 不复用纯 V16、研究回测或旧 V11/Grid 账本。
- 继续共用账户级 PID 锁，防止旧 GUI 和新组合 GUI 同时下单。
- 组合高水位状态使用独立 basename，并按 DRY-RUN/LIVE 分离保存。

## 自动验收

- 相关测试：79 passed，另有 9 个参数化子测试通过
- GUI 静态校验：策略类、50 币、共享 Max2、双袖套限制、配置与依赖哈希通过
- 临时 DRY-RUN：组合策略解析、本地 API 启停、50/50 同步市场环境及
  50/50 完整 V16 四段路径通过
- DRY-RUN 模拟权益：200.00U
- 真实订单提交：0

## 无法完全消除的现实差异

实际 LIVE 的盘口、点差、网络延迟、部分成交、资金费、最小数量和市场冲击无法
由 1m OHLC 回放精确复制，因此实盘成交价和后续复利金额仍可能偏离回测。GUI
没有另一套交易逻辑；这些差异属于交易所执行现实，而不是策略分叉。
