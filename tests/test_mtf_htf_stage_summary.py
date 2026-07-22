from __future__ import annotations

import json

from scripts.mtf_htf_stage_summary import summarize_stage
from scripts.mtf_htf_stage_runner import _manifest_configs, _signal_strategy_key
from crypto_scalper.live_config import default_live_config, load_live_config
from dataclasses import replace


def test_stage_summary_removes_full_report_payload() -> None:
    payload = {
        "trade_start": "2025-06-12T00:00:00",
        "trade_end": "2026-04-01T00:00:00",
        "cost_experiment": "full_cost",
        "backtest_mode": "conservative",
        "symbols": ["BTCUSDT"],
        "experiments": [
            {
                "name": "baseline",
                "config": "config.json",
                "summary": {"overall": {"trades": 1}},
                "diagnostics": {"splits": {}},
                "report": {"trades": [{"net_pnl": 1.0}]},
            }
        ],
    }

    compact = summarize_stage(payload)

    assert "report" not in compact["experiments"][0]
    assert compact["experiments"][0]["summary"]["overall"]["trades"] == 1


def test_signal_cache_key_ignores_exit_only_parameters() -> None:
    strategy = default_live_config().strategy
    changed_exit = replace(strategy, mtf_move_stop_to_breakeven_r=0.8, mtf_exit_on_1h_confirm_lost=False)
    changed_signal = replace(strategy, mtf_trigger_max_volume_ratio=2.0)

    assert _signal_strategy_key(strategy) == _signal_strategy_key(changed_exit)
    assert _signal_strategy_key(strategy) != _signal_strategy_key(changed_signal)


def test_signal_cache_key_ignores_portfolio_scheduling_parameters() -> None:
    strategy = default_live_config().strategy
    changed = replace(
        strategy,
        mtf_max_open_positions=3,
        mtf_max_daily_trades=5,
        mtf_symbol_cooldown_hours=2,
    )

    assert _signal_strategy_key(strategy) == _signal_strategy_key(changed)


def test_manifest_can_import_mtpc_sleeve_and_enable_explicit_combination(tmp_path) -> None:
    manifest = tmp_path / "combined.json"
    manifest.write_text(
        json.dumps(
            {
                "base_config": "config.mtf-htf.long-rank3p5-aggressive-3p5.json",
                "experiments": [
                    {
                        "name": "combined",
                        "mtpc_source_config": "config.mtpc.frequency-profit-v1.json",
                        "universe_source_config": "config.mtpc.frequency-profit-v1.json",
                        "append_universe_to_base": True,
                        "mtpc": {"combine_with_mtf": True},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    configs = _manifest_configs(str(manifest))

    assert len(configs) == 1
    combined = configs[0][2]
    assert combined.strategy.mtf_4h_rsi_regime_enabled
    assert combined.mtpc.enabled
    assert combined.mtpc.combine_with_mtf
    assert not combined.mtpc.allow_short
    assert len(combined.trading.symbols) == 60
    assert len(combined.trading.entry_symbols) == 57
    frozen = load_live_config("config.mtf-htf.long-rank3p5-aggressive-3p5.json")
    assert combined.trading.symbols[:30] == frozen.trading.symbols
