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

给 120U 本金的默认保守风控：

- 最大杠杆上限 `20x`
- 总保证金占用不超过权益 `10%`
- 单币种保证金占用不超过权益 `3%`
- 单仓名义价值不超过 `50U`
- 最大同时持仓 `3`
- 默认观察 18 个主流 U 本位合约：BTC、ETH、BNB、SOL、XRP、DOGE、ADA、AVAX、LINK、LTC、BCH、DOT、TRX、MATIC、NEAR、APT、ARB、OP
- 单笔风险约 `0.25%`
- 日亏损上限 `3%`
- 总最大回撤 `10%`
- 保留至少 `20U` 可用余额

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
