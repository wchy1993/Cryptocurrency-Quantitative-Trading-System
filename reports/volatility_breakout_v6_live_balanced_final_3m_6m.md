# Breakout v6 core/runner optimization

- Gap-free 1m execution, fees, slippage, funding and point-in-time market context
- Core and runner use independent risk/exit profiles; global max position remains one
- GUI, Hybrid v5 and Grid v3 are unchanged
- Selection status: `strict_robust_improvement`

| Period | Version | Trades | Net | PF | Win | Max DD |
|---|---|---:|---:|---:|---:|---:|
| 3 months | Hybrid v5 baseline | 147 | +1723.23U | 1.616 | 31.97% | 39.94% |
| 3 months | Breakout v6 core/runner | 49 | +3481.58U | 4.847 | 42.86% | 29.15% |
| 6 months | Hybrid v5 baseline | 277 | +5735.00U | 1.620 | 31.41% | 39.94% |
| 6 months | Breakout v6 core/runner | 92 | +10537.28U | 4.396 | 36.96% | 31.76% |
