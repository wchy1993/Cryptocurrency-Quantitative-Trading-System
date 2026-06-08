# 加密合约超短线量化 MVP

这是一个研究优先的模拟交易系统：先做 CSV 行情回测、样例策略、交易成本、风控、参数优化和模拟成交，不接真实交易所密钥，也不承诺盈利。

## 当前能力

- 生成确定性的 1 分钟样例 OHLCV 数据
- 从 CSV 加载 K 线数据
- 运行波动率突破策略
- 支持突破缓冲、趋势强度、成交量和波动率过滤
- 支持插针保护和确认后小仓位反打
- 计入手续费、滑点、止损、止盈、保本止损、移动止损、最长持仓时间和简化强平价
- 单仓位回测，输出胜率、净收益、最大回撤、盈亏比等指标
- 随机参数搜索，可按净收益、胜率、盈亏比、Calmar 或综合盈利分数排序
- 内置基础测试

## 快速开始

生成样例数据：

```powershell
python -m crypto_scalper.cli generate-sample --output data/sample_btcusdt_1m.csv --bars 3000
```

运行回测：

```powershell
python -m crypto_scalper.cli backtest --config config.example.json
```

查看逐笔交易：

```powershell
python -m crypto_scalper.cli backtest --config config.example.json --trades
```

扫描策略参数，默认按综合盈利分数排序：

```powershell
python -m crypto_scalper.cli optimize --config config.example.json --top 10 --min-trades 20 --trials 250
```

只按净收益最大化排序：

```powershell
python -m crypto_scalper.cli optimize --config config.example.json --metric net_return_pct --top 10 --min-trades 20 --trials 500
```

把最佳参数写成新配置：

```powershell
python -m crypto_scalper.cli optimize --config config.example.json --metric net_return_pct --trials 500 --write-config config.optimized.json
```

运行测试：

```powershell
python -m unittest discover -s tests
```

## Binance U 本位自动交易

当前实盘模块默认是 `testnet + dry-run`，不会真实下单。图形界面入口：

```powershell
python -m crypto_scalper.gui
```

命令行跑一轮检查：

```powershell
python -m crypto_scalper.cli trade-live --config config.live.json --once
```

配置文件：

- `config.live.json`: 本机运行配置，默认不提交
- `config.live.example.json`: 示例配置
- `.env.example`: API 密钥示例

把 `.env.example` 复制成 `.env` 后，填入你自己的变量。不要把密钥发到聊天里，也不要提交 `.env`。

```text
BINANCE_FUTURES_API_KEY=你的 API Key
BINANCE_FUTURES_API_SECRET=你的 API Secret
```

主网真实下单需要同时满足：

- `environment` 改成 `mainnet`
- `dry_run` 改成 `false`
- `mainnet_confirmation_text` 填 `CONFIRM_MAINNET`
- 账户是单向持仓 One-way，本版本默认不在 Hedge Mode 下交易

给 100U 本金的默认保守风控：

- 当前策略按 `30m` 信号入场，用 `30m/1h/2h` 做多周期过滤
- 最大杠杆上限 `30x`，实际仓位仍按止损距离和信号质量控制，不会每笔都满 30x
- 总保证金占用不超过权益 `8%`
- 单币种保证金占用不超过权益 `4%`
- 默认观察并允许开仓 60 个 U 本位合约
- 单周期最多新开 `1` 个仓位，最多同时持有 `4` 个仓位
- 单笔风险参数为 `6%`，实际下单还会受信号质量、保证金上限和最小/最大名义仓位限制
- 日亏损上限 `15%`
- 总最大回撤 `15%`
- 保留至少 `20U` 可用余额

下载 15m 历史数据并跑组合级回测：

```powershell
python -m crypto_scalper.cli download-history --timeframe 15m --days 90 --output-dir data/binance_15m_90d
python -m crypto_scalper.live_portfolio_backtest --config config.live.json --data-dir data/binance_15m_90d
```

新的 60 币 15m 初步组合回测记录在 `report_15m_30d.md`。旧的 30 币 1h 历史测试记录保留在 `report.md`，只作为旧周期参考，不能把 1h 结果当作 15m 表现。

基于 60 币 15m 数据重新优化后的配置在 `config.live.optimized.json`，报告在 `report_live_opt_15m_60_round2.md`。复跑命令：

```powershell
python -m crypto_scalper.live_portfolio_backtest --config config.live.optimized.json --data-dir data/binance_15m_30d
```

至少 3 个月的 100 天复测数据在 `data/binance_15m_100d`，复测报告在 `report_live_15m_100d.md`。这个长样本复测显示 30 天优化配置没有通过稳定性要求。

基于 100 天数据重新优化后的配置在 `config.live.optimized_100d.json`，报告在 `report_live_opt_15m_100d.md`。复跑命令：

```powershell
python -m crypto_scalper.live_portfolio_backtest --config config.live.optimized_100d.json --data-dir data/binance_15m_100d
```

针对超级放量突破进一步优化后的配置在 `config.live.optimized_super_volume.json`，报告在 `report_live_super_volume_100d.md`。复跑命令：

```powershell
python -m crypto_scalper.live_portfolio_backtest --config config.live.optimized_super_volume.json --data-dir data/binance_15m_100d
```

最新 30m 版本使用 60 个币种，数据在 `data/binance_30m_180d`，100U 最新优化报告在 `report_live_super_volume_30m_180d_100u.md`。15m 100U 报告保留在 `report_live_super_volume_180d_100u.md`，10000U 优化报告保留在 `report_live_super_volume_180d_weekly_risk_optimized.md`。复跑命令：

```powershell
python -m crypto_scalper.resample_history --input-dir data/binance_15m_180d --output-dir data/binance_30m_180d --source-timeframe 15m --target-timeframe 30m
python -m crypto_scalper.live_portfolio_backtest --config config.live.optimized_super_volume.json --data-dir data/binance_30m_180d
```

## 数据格式

CSV 需要包含以下列：

```text
timestamp,open,high,low,close,volume
```

`timestamp` 支持 ISO 格式，例如 `2025-01-01T00:00:00` 或带 `Z` 的 UTC 时间。

## 配置重点

`config.example.json` 里的关键字段：

- `fee_bps`: 单边手续费，基点单位，4 表示 0.04%
- `slippage_bps`: 单边滑点，基点单位
- `max_leverage`: 最大杠杆，用于简化强平价估算
- `risk_per_trade_pct`: 单笔最大风险占权益比例
- `max_daily_loss_pct`: 当日亏损熔断
- `max_drawdown_pct`: 总回撤熔断
- `min_atr_pct` / `max_atr_pct`: 过滤过低或过高波动行情
- `breakout_buffer_atr`: 要求突破超过通道一定 ATR，减少假突破
- `ema_gap_atr`: 要求快慢 EMA 有足够趋势差
- `min_volume_ratio`: 要求当前成交量高于近期均量比例
- `breakeven_atr`: 盈利达到指定 ATR 后把止损推到开仓价
- `trailing_stop_atr`: 移动止损距离
- `max_holding_bars`: 最长持仓 K 线数，0 表示不限制
- `spike_guard_enabled`: 遇到异常插针时暂停普通追单
- `spike_trade_enabled`: 允许插针后确认反打
- `spike_min_range_atr`: K 线总振幅至少达到多少 ATR 才算插针
- `spike_min_wick_ratio`: 影线占整根 K 线的最小比例
- `spike_risk_multiplier`: 插针单相对普通单的风险倍数，默认更小
- `spike_max_holding_bars`: 插针单最长持仓 K 线数，默认短持仓

## 优化原则

盈利最大化不要只看胜率。建议先按 `net_return_pct` 或 `profit_score` 排序，再检查：

- 交易次数是否足够，避免偶然样本
- 最大回撤是否可承受
- 盈亏比是否大于 1
- 手续费和滑点是否按真实交易所水平设置
- 最佳参数换一段历史数据后是否仍然有效

## 下一步建议

1. 接入真实历史数据下载器，例如 Binance/OKX/Kraken 的 K 线。
2. 增加 walk-forward 验证，避免只在一段行情上过拟合。
3. 加入盘口深度和订单簿滑点模型。
4. 加入模拟盘执行层，再考虑小资金实盘。
5. 做参数扫描时以净期望、回撤和稳定性为主，不只看胜率。

## 风险说明

这只是研究和模拟交易工具。加密合约具有高波动、高杠杆和强平风险；任何策略都可能亏损，甚至在异常行情中快速亏完本金。
