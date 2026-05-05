# FreqAI 策略说明

这是一套独立于当前 `crypto_scalper` GUI 的 Freqtrade/FreqAI 策略骨架。当前自研交易循环继续保留；这套目录用于迁移到 Freqtrade 生态后做机器学习回测、dry-run 和后续小资金验证。

## 文件

- `config_freqai.example.json`: Freqtrade + FreqAI 示例配置，默认 `dry_run=true`，`dry_run_wallet=10000`。
- `strategies/FreqaiMarketRegimeStrategy.py`: FreqAI 策略，支持多空，但默认只在模型预测和趋势过滤同时通过时入场。
- `models/`: FreqAI 训练模型保存位置。
- `data/`: Freqtrade 下载的行情数据位置。

## 策略逻辑

模型目标：

```text
预测未来 8 根 15m K 线的平均收益率
```

入场不是只看模型预测，还会同时过滤：

```text
do_predict == 1
预测收益 > 动态阈值
EMA50 / EMA200 顺势
ADX >= 18
RSI 没有过热/过冷
ATR% 在合理区间
成交量不低于近期均量
```

这样做的目的不是追求高频开仓，而是减少你前面遇到的两个问题：

```text
1. 手写规则过拟合、经常没有 edge
2. 50 个币同时扫描后交易次数太多，手续费吃掉利润
```

## 安装

建议单独建一个虚拟环境，不要和当前项目混在一起。

```powershell
python -m venv .venv-freqtrade
.\.venv-freqtrade\Scripts\Activate.ps1
pip install "freqtrade[freqai]"
```

如果 LightGBM 没装上，再单独装：

```powershell
pip install lightgbm
```

## 下载数据

回测 FreqAI 前必须先下载足够长的数据。因为训练窗口是 30 天，回测开始前也需要预留训练数据，所以下载范围要比回测范围更早。

示例：

```powershell
freqtrade download-data `
  --userdir freqtrade_user_data `
  --config freqtrade_user_data/config_freqai.example.json `
  --timeframes 15m 1h 4h `
  --trading-mode futures `
  --timerange 20240101-
```

## 回测

先用 2-3 个月回测，不要直接实盘。

```powershell
freqtrade backtesting `
  --userdir freqtrade_user_data `
  --strategy FreqaiMarketRegimeStrategy `
  --strategy-path freqtrade_user_data/strategies `
  --config freqtrade_user_data/config_freqai.example.json `
  --freqaimodel LightGBMRegressor `
  --timerange 20250301-20250501
```

如果你改了特征、目标或模型参数，要修改配置里的：

```text
freqai.identifier
```

否则 FreqAI 可能复用旧模型。

## Dry-run

只在回测结果稳定后再跑 dry-run：

```powershell
freqtrade trade `
  --userdir freqtrade_user_data `
  --strategy FreqaiMarketRegimeStrategy `
  --strategy-path freqtrade_user_data/strategies `
  --config freqtrade_user_data/config_freqai.example.json `
  --freqaimodel LightGBMRegressor
```

## 调参优先级

优先调这些：

```text
prediction_zscore
min_prediction_edge
label_period_candles
train_period_days
include_timeframes
max_open_trades
stake_amount
```

不要一开始就加更多模型或更多特征。先看：

```text
手续费后净利润
最大回撤
平均单笔收益
交易次数
多空分别表现
不同月份是否都能接受
```

## 重要提醒

50 个币 + 3 个时间周期 + 4 个相关币特征会比较耗时。如果训练太慢，先把 `pair_whitelist` 缩到：

```text
BTC, ETH, BNB, SOL, XRP, ADA, AVAX, LINK, LTC, BCH
```

跑通回测和 dry-run 后，再逐步扩到 50 个。
