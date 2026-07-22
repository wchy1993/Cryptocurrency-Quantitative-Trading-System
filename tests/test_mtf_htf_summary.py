from __future__ import annotations

from scripts.mtf_htf_summary import summarize


def _trade(symbol: str, pnl: float, month: str, side: str = "LONG") -> dict[str, object]:
    return {
        "symbol": symbol,
        "side": side,
        "entry_time": f"{month}-01T00:00:00",
        "entry_reason": (
            "long_mtf_4h_rsi_regime_pullback regime=LONG_BIAS "
            "regime_tf=4h trigger=sweep trigger_tf=15m"
        ),
        "exit_reason": "take_profit_1m" if pnl > 0 else "stop_loss_1m",
        "gross_pnl": pnl + 0.2,
        "fee": 0.1,
        "slippage_cost": 0.1,
        "net_pnl": pnl,
        "entry_price": 100.0,
        "initial_stop_price": 99.0,
        "quantity": 1.0,
        "mfe": 1.0 if pnl > 0 else 0.2,
        "mae": -0.2 if pnl > 0 else -1.0,
    }


def test_mtf_summary_reports_robustness_and_groups() -> None:
    report = {
        "initial_equity": 160.0,
        "final_equity": 160.5,
        "trades": [
            _trade("BTCUSDT", 1.0, "2026-01"),
            _trade("ETHUSDT", -0.5, "2026-02"),
        ],
        "mtf_report": {"reject_stats": {"4h_no_regime": 7}},
    }

    result = summarize(report)

    assert result["overall"]["trades"] == 2
    assert result["overall"]["profit_factor"] == 2.0
    assert result["by_trigger"]["sweep"]["trades"] == 2
    assert result["by_month"]["2026-01"]["net_pnl"] == 1.0
    assert result["path"]["samples"] == 2
    assert result["path"]["reached_1_00r_pct"] == 50.0
    assert result["top_symbol"] == "BTCUSDT"
    assert result["reject_stats"]["4h_no_regime"] == 7


def test_combined_summary_includes_mtf_and_mtpc_sleeves() -> None:
    mtf_trade = _trade("BTCUSDT", 1.0, "2026-01")
    mtpc_trade = {
        **_trade("ETHUSDT", -0.25, "2026-01"),
        "entry_reason": "multi_timeframe_trend_pullback_continuation entry_mode=first_pullback",
    }
    report = {
        "initial_equity": 200.0,
        "final_equity": 200.75,
        "net_pnl": 0.75,
        "trades": [mtf_trade, mtpc_trade],
        "mtf_report": {"reject_stats": {}},
        "mtpc_report": {"candidate_count": 2},
    }

    result = summarize(report)

    assert result["overall"]["trades"] == 2
    assert result["overall"]["net_pnl"] == 0.75
    assert result["by_strategy"]["mtf_reversal"]["trades"] == 1
    assert result["by_strategy"]["mtpc"]["trades"] == 1
    assert result["active_sleeves"] == {"mtf_reversal": True, "mtpc": True}
