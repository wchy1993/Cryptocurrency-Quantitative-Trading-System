from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

from .config import StrategyConfig


DEFAULT_SYMBOLS = (
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "LTCUSDT",
    "BCHUSDT",
    "DOTUSDT",
    "TRXUSDT",
    "NEARUSDT",
    "APTUSDT",
    "ARBUSDT",
    "OPUSDT",
    "UNIUSDT",
    "AAVEUSDT",
    "FILUSDT",
    "ETCUSDT",
    "ATOMUSDT",
    "INJUSDT",
    "SUIUSDT",
    "SEIUSDT",
    "WIFUSDT",
    "1000PEPEUSDT",
    "1000SHIBUSDT",
    "TIAUSDT",
    "POLUSDT",
    "WLDUSDT",
    "XLMUSDT",
    "XMRUSDT",
    "ALGOUSDT",
    "VETUSDT",
    "THETAUSDT",
    "RUNEUSDT",
    "CRVUSDT",
    "SANDUSDT",
    "MANAUSDT",
    "HBARUSDT",
    "GALAUSDT",
    "ARUSDT",
    "LDOUSDT",
    "ICPUSDT",
    "FETUSDT",
    "STXUSDT",
    "CFXUSDT",
    "PENDLEUSDT",
    "ORDIUSDT",
    "JUPUSDT",
    "PYTHUSDT",
    "STRKUSDT",
    "TAOUSDT",
    "ENAUSDT",
    "1000FLOKIUSDT",
    "JASMYUSDT",
    "IMXUSDT",
    "APEUSDT",
    "CHZUSDT",
)

DEFAULT_ENTRY_SYMBOLS = DEFAULT_SYMBOLS


@dataclass(frozen=True)
class ExchangeConfig:
    environment: str = "testnet"
    api_key_env: str = "BINANCE_FUTURES_API_KEY"
    api_secret_env: str = "BINANCE_FUTURES_API_SECRET"
    recv_window: int = 5_000
    timeout_seconds: int = 10


@dataclass(frozen=True)
class LiveTradingConfig:
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS
    entry_symbols: tuple[str, ...] = DEFAULT_ENTRY_SYMBOLS
    timeframe: str = "30m"
    kline_limit: int = 500
    poll_seconds: int = 60
    entry_scan_seconds: int = 300
    symbol_reentry_cooldown_seconds: int = 3600
    dry_run: bool = True
    require_mainnet_confirmation: bool = True
    mainnet_confirmation_text: str = ""
    leverage: int = 30
    symbol_leverage_overrides: dict[str, int] = field(default_factory=dict)
    margin_type: str = "CROSSED"
    require_one_way_mode: bool = True
    use_market_orders: bool = True
    reduce_only_exit: bool = True
    use_protective_orders: bool = True
    working_type: str = "MARK_PRICE"
    max_open_positions: int = 5
    super_volume_extra_slot_enabled: bool = False
    super_volume_extra_max_open_positions: int = 5
    super_volume_extra_min_rank_score: float = 6.2
    super_volume_extra_min_momentum_pct: float = 0.035
    super_volume_extra_min_volume_ratio: float = 2.8
    max_new_entries_per_cycle: int = 1
    min_managed_exit_bars: int = 1
    stats_log_interval_seconds: int = 300
    initial_entry_fraction: float = 0.75
    scale_in_entry_fraction: float = 0.35
    max_scale_ins_per_symbol: int = 2
    scale_in_min_profit_pct: float = 0.004
    scale_in_cooldown_seconds: int = 3600
    allow_loss_scale_in: bool = False
    loss_scale_in_trigger_pct: float = 0.006
    loss_scale_in_entry_fraction: float = 0.25
    profit_exit_enabled: bool = True
    breakeven_trigger_pct: float = 0.003
    breakeven_lock_pct: float = 0.0012
    trailing_activation_pct: float = 0.008
    trailing_pullback_pct: float = 0.0035
    momentum_exit_min_profit_pct: float = 0.003
    quick_take_profit_pct: float = 0.0075
    strong_take_profit_pct: float = 0.018
    profit_exit_rsi_long: float = 62.0
    profit_exit_rsi_short: float = 38.0
    session_profit_guard_enabled: bool = False
    session_profit_guard_trigger_usdt: float = 0.45
    session_profit_guard_pullback_usdt: float = 0.25
    session_profit_guard_cooldown_seconds: int = 600


@dataclass(frozen=True)
class MultiTimeframeFilterConfig:
    enabled: bool = True
    timeframes: tuple[str, ...] = ("30m", "1h", "2h")
    kline_limit: int = 240
    min_score: int = 4
    rsi_period: int = 14
    rsi_long_floor: float = 32.0
    rsi_long_ceiling: float = 72.0
    rsi_short_floor: float = 28.0
    rsi_short_ceiling: float = 68.0
    rsi_oversold: float = 35.0
    rsi_overbought: float = 65.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    kdj_period: int = 9
    higher_tf_alignment_enabled: bool = False
    higher_tf_alignment_timeframe: str = "1h"
    higher_tf_alignment_conflict_score: int = 2
    higher_tf_alignment_rsi_buffer: float = 5.0
    extreme_reversal_entry_enabled: bool = True
    pre_cross_entry_enabled: bool = True
    reversal_cross_lookback_bars: int = 3
    long_extreme_rsi: float = 30.0
    short_extreme_rsi: float = 70.0
    long_extreme_kdj: float = 25.0
    short_extreme_kdj: float = 75.0
    confirmed_cross_risk_multiplier: float = 0.65
    pre_cross_risk_multiplier: float = 0.25


@dataclass(frozen=True)
class LiveRiskConfig:
    starting_capital_usdt: float = 100.0
    max_account_margin_usage_pct: float = 0.10
    max_symbol_margin_pct: float = 0.04
    min_symbol_margin_pct: float = 0.01
    max_position_notional_usdt: float = 10_000.0
    risk_per_trade_pct: float = 0.06
    max_daily_loss_pct: float = 0.15
    max_drawdown_pct: float = 0.15
    starting_capital_drawdown_stop_pct: float = 0.20
    weekly_profit_drawdown_stop_pct: float = 0.15
    soft_drawdown_reduce_pct: float = 0.05
    soft_drawdown_stop_pct: float = 0.10
    soft_drawdown_min_size_multiplier: float = 0.35
    estimated_fee_bps: float = 5.0
    estimated_slippage_bps: float = 2.0
    backtest_mode: str = "conservative"
    cost_experiment: str = "full_cost"
    market_slippage_bps: float = 2.0
    stop_slippage_bps: float = 5.0
    take_profit_slippage_bps: float = 2.0
    maker_fee_rate: float = 0.0002
    taker_fee_rate: float = 0.0005
    funding_enabled: bool = False
    funding_data_dir: str = ""
    funding_default_rate: float = 0.0
    dynamic_slippage_enabled: bool = False
    impact_coefficient_bps: float = 25.0
    impact_exponent: float = 0.5
    max_bar_participation_rate: float = 0.003
    min_partial_fill_ratio: float = 0.10
    point_in_time_universe_enabled: bool = False
    point_in_time_universe_top_n: int = 100
    universe_lookback_days: int = 1
    new_symbol_warmup_days: int = 20
    alpha_diagnostics_enabled: bool = False
    min_profit_after_cost_pct: float = 0.0010
    min_available_balance_usdt: float = 20.0
    min_order_notional_usdt: float = 5.0
    cooldown_seconds_after_loss: int = 300


@dataclass(frozen=True)
class RegimeScoreConfig:
    enabled: bool = False
    shadow_mode: bool = True
    minimum_history_bars: int = 30
    trend_min_score: float = 60.0
    reversal_min_score: float = 60.0
    minimum_score_gap: float = 10.0
    trend_ema_slope_weight: float = 20.0
    trend_ema_alignment_weight: float = 20.0
    trend_macd_weight: float = 15.0
    trend_atr_expansion_weight: float = 10.0
    trend_btc_weight: float = 20.0
    trend_breadth_weight: float = 15.0
    reversal_extension_weight: float = 25.0
    reversal_rsi_weight: float = 15.0
    reversal_kdj_weight: float = 10.0
    reversal_macd_weight: float = 20.0
    reversal_no_extreme_weight: float = 15.0
    reversal_btc_weight: float = 10.0
    reversal_exhaustion_weight: float = 5.0
    ema_slope_lookback_bars: int = 3
    ema_slope_full_score_atr: float = 0.08
    ema_alignment_full_score_atr: float = 0.50
    macd_level_full_score_atr: float = 0.12
    macd_delta_full_score_atr: float = 0.06
    atr_expansion_min_ratio: float = 0.90
    atr_expansion_full_score_ratio: float = 1.25
    atr_chaos_ratio: float = 2.0
    btc_15m_full_score_return: float = 0.006
    btc_1h_full_score_return: float = 0.015
    btc_alignment_full_score_atr: float = 0.40
    btc_adverse_return_limit: float = 0.015
    breadth_full_score: float = 0.68
    trend_max_extension_atr: float = 3.0
    reversal_recovery_lookback_bars: int = 5
    reversal_extension_min_atr: float = 0.50
    reversal_extension_full_score_atr: float = 2.0
    reversal_rsi_neutral: float = 50.0
    reversal_rsi_extreme: float = 35.0
    reversal_rsi_full_score: float = 25.0
    reversal_kdj_extreme: float = 30.0
    reversal_macd_delta_full_score_atr: float = 0.06


@dataclass(frozen=True)
class ReversalAlphaConfig:
    enabled: bool = False
    shadow_mode: bool = True
    setup_lookback_bars: int = 5
    setup_extension_atr_min: float = 0.50
    setup_long_rsi_extreme_max: float = 40.0
    setup_short_rsi_extreme_min: float = 60.0
    setup_long_kdj_extreme_max: float = 35.0
    setup_short_kdj_extreme_min: float = 65.0
    setup_require_rsi_recovery: bool = True
    setup_require_kdj_recovery: bool = True
    setup_macd_improvement_bars: int = 2
    setup_require_no_new_extreme: bool = True
    trigger_ema_period: int = 9
    trigger_require_directional_candle: bool = True
    trigger_close_position_min: float = 0.55
    trigger_require_macd_improvement: bool = True
    btc_adverse_15m_return: float = 0.006
    btc_adverse_1h_return: float = 0.012


@dataclass(frozen=True)
class MacroEventConfig:
    enabled: bool = False
    events_path: str = "data/macro_events_us_major_2025_2026.csv"
    symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    primary_symbol: str = "BTCUSDT"
    event_types: tuple[str, ...] = ("NFP", "CPI_YOY")
    leverage: int = 50
    pre_event_flatten_seconds: int = 120
    post_event_lockout_seconds: int = 3600
    post_event_entry_delay_seconds: int = 60
    margin_pct: float = 0.05
    stop_loss_pct: float = 0.0025
    take_profit_pct: float = 0.0040
    max_holding_seconds: int = 900
    nfp_min_surprise_k: float = 25.0
    cpi_min_surprise_pct: float = 0.10


@dataclass(frozen=True)
class VbpUniverseConfig:
    rank_jump_threshold: int = 50
    rank_window_minutes: int = 30
    rvol_entry_threshold: float = 1.5
    rvol_trigger_threshold: float = 3.0
    rvol_lookback_days: int = 20
    watchlist_ttl_minutes: int = 60


@dataclass(frozen=True)
class VbpStructureFilterConfig:
    consolidation_bars: int = 48
    consolidation_threshold_pct: float = 0.02
    funding_rate_max: float = 0.0001
    daily_high_lookback_days: int = 90
    daily_high_zone_pct: float = 0.9
    compression_quality_enabled: bool = False
    compression_atr_lookback_bars: int = 80
    compression_atr_percentile_max: float = 1.0
    compression_range_atr_max: float = 999.0
    compression_volume_recent_bars: int = 12
    compression_volume_baseline_bars: int = 48
    compression_volume_contraction_max: float = 999.0
    compression_prior_move_lookback_bars: int = 30
    compression_prior_move_max_atr: float = 999.0


@dataclass(frozen=True)
class VbpEntryConfig:
    pullback_volume_ratio: float = 0.4
    timeout_bars: int = 15
    use_vwap_as_pullback_target: bool = True
    breakout_quality_enabled: bool = False
    breakout_distance_atr_min: float = 0.0
    breakout_distance_atr_max: float = 999.0
    breakout_body_atr_min: float = 0.0
    breakout_close_position_min: float = 0.0
    breakout_upper_wick_ratio_max: float = 1.0
    breakout_volume_ratio_min: float = 0.0
    pullback_quality_enabled: bool = False
    pullback_depth_atr_min: float = 0.0
    pullback_depth_atr_max: float = 999.0
    pullback_depth_to_breakout_max: float = 999.0
    pullback_bars_max: int = 999
    pullback_require_hold_breakout_level: bool = False
    confirmation_quality_enabled: bool = False
    confirmation_body_atr_min: float = 0.0
    confirmation_close_position_min: float = 0.0
    confirmation_upper_wick_ratio_max: float = 1.0
    max_entry_chase_atr: float = 999.0
    relative_strength_enabled: bool = False
    relative_strength_lookback_minutes: int = 60
    relative_strength_min_vs_btc_pct: float = 0.0015
    relative_strength_min_vs_market_pct: float = -1.0
    relative_strength_max_rank_pct: float = 1.0


@dataclass(frozen=True)
class VbpExitConfig:
    stop_loss_pct: float = 0.025
    tp1_rr_ratio: float = 1.5
    tp1_close_ratio: float = 0.5
    trailing_stop_after_tp1: bool = True
    runner_after_tp1_enabled: bool = False
    peak_giveback_enabled: bool = True
    peak_giveback_trigger_pct: float = 0.012
    peak_giveback_floor_pct: float = 0.002
    peak_giveback_retrace_pct: float = 0.008
    peak_giveback_pre_tp1_trigger_pct: float = 0.020
    peak_giveback_require_cost_coverage: bool = True
    peak_tracking_mode: str = "close"
    large_bear_exit_enabled: bool = True
    large_bear_min_peak_profit_pct: float = 0.010
    large_bear_min_current_profit_pct: float = 0.002
    large_bear_volume_multiplier: float = 3.0
    large_bear_lookback_bars: int = 20
    large_bear_max_close_position: float = 0.25
    large_bear_min_body_pct: float = 0.0025


@dataclass(frozen=True)
class VbpPositionConfig:
    size_multiplier: float = 1.5
    max_positions: int = 2


@dataclass(frozen=True)
class VbpMarketFilterConfig:
    enabled: bool = True
    btc_15m_drop_block: float = -0.006
    btc_1h_drop_block: float = -0.012
    btc_1h_ema_bear_block_enabled: bool = True
    breadth_enabled: bool = True
    breadth_min_15m_up_pct: float = 0.38
    breadth_min_1h_up_pct: float = 0.42
    breadth_min_above_ema21_pct: float = 0.35


@dataclass(frozen=True)
class VbpRiskControlConfig:
    enabled: bool = True
    consecutive_loss_limit: int = 2
    consecutive_loss_pause_minutes: int = 60
    symbol_loss_cooldown_minutes: int = 240
    daily_loss_stop_pct: float = 0.05
    adaptive_exposure_enabled: bool = True
    daily_loss_reduce_pct: float = 0.03
    monthly_loss_reduce_pct: float = 0.08
    drawdown_reduce_pct: float = 0.12
    reduced_size_multiplier: float = 0.75
    reduced_max_positions: int = 1
    monthly_drawdown_control_enabled: bool = True
    monthly_drawdown_half_size_pct: float = 0.10
    monthly_drawdown_one_position_pct: float = 0.15
    monthly_drawdown_stop_pct: float = 0.20
    weekly_drawdown_control_enabled: bool = False
    weekly_drawdown_half_size_pct: float = 0.08
    weekly_drawdown_one_position_pct: float = 0.12
    weekly_drawdown_stop_pct: float = 0.16
    weekly_loss_reduce_pct: float = 0.06
    weekly_loss_one_position_pct: float = 0.10
    weekly_loss_stop_pct: float = 0.14
    consecutive_loss_reduce_enabled: bool = False
    consecutive_loss_reduce_losses: int = 1
    consecutive_loss_reduce_minutes: int = 1440
    consecutive_loss_reduced_size_multiplier: float = 0.65
    consecutive_loss_reduced_max_positions: int = 1
    consecutive_loss_quality_filter_enabled: bool = False
    consecutive_loss_quality_losses: int = 2
    consecutive_loss_quality_minutes: int = 240
    consecutive_loss_quality_rvol_multiplier: float = 1.25
    consecutive_loss_quality_min_close_position: float = 0.65
    consecutive_loss_quality_pullback_volume_ratio: float = 0.30
    consecutive_loss_quality_size_multiplier: float = 0.50
    consecutive_loss_quality_max_positions: int = 1
    frequency_control_enabled: bool = True
    symbol_entry_cooldown_minutes: int = 720
    max_24h_return_pct: float = 0.18
    correlated_alt_max_positions: int = 2
    symbol_performance_guard_enabled: bool = True
    symbol_recent_trade_window: int = 8
    symbol_recent_min_trades: int = 5
    symbol_recent_pf_min: float = 0.8
    symbol_recent_net_pnl_min: float = 0.0
    symbol_performance_cooldown_minutes: int = 1440
    fail_fast_enabled: bool = True
    fail_fast_minutes: int = 15
    fail_fast_min_mfe_r: float = 0.25
    fail_fast_lost_level_enabled: bool = True
    fail_fast_lost_vwap_enabled: bool = True
    breakeven_enabled: bool = True
    breakeven_trigger_r: float = 0.8
    breakeven_offset_pct: float = 0.0005


@dataclass(frozen=True)
class VbpStrategyConfig:
    enabled: bool = False
    enabled_symbols: tuple[str, ...] = ()
    universe: VbpUniverseConfig = field(default_factory=VbpUniverseConfig)
    structure_filter: VbpStructureFilterConfig = field(default_factory=VbpStructureFilterConfig)
    entry: VbpEntryConfig = field(default_factory=VbpEntryConfig)
    exit: VbpExitConfig = field(default_factory=VbpExitConfig)
    position: VbpPositionConfig = field(default_factory=VbpPositionConfig)
    market_filter: VbpMarketFilterConfig = field(default_factory=VbpMarketFilterConfig)
    risk_control: VbpRiskControlConfig = field(default_factory=VbpRiskControlConfig)


@dataclass(frozen=True)
class CmiprRegimeConfig:
    btc_symbol: str = "BTCUSDT"
    eth_symbol: str = "ETHUSDT"
    ema_period: int = 21
    ema_slope_lookback: int = 3
    min_ema_slope_pct: float = 0.001
    min_breadth_above_ema21: float = 0.55
    min_breadth_positive_1h: float = 0.52
    max_breadth_above_ema21_overheated: float = 0.88
    btc_shock_1h_pct: float = 0.035
    max_direction_conflict: float = 0.35
    allow_neutral_high_quality: bool = False
    enter_confirmation_bars_1h: int = 2
    exit_confirmation_bars_1h: int = 2
    enter_breadth_above_ema21: float = 0.58
    exit_breadth_above_ema21: float = 0.48
    enter_ema_slope_pct: float = 0.0012
    exit_ema_slope_pct: float = -0.0002
    min_state_hold_bars_1h: int = 3


@dataclass(frozen=True)
class CmiprRankingConfig:
    long_top_fraction: float = 0.15
    short_bottom_fraction: float = 0.10
    max_candidates_per_scan: int = 8
    weight_return_30m: float = 0.15
    weight_return_1h: float = 0.25
    weight_return_4h: float = 0.25
    weight_relative_btc_1h: float = 0.15
    weight_relative_btc_4h: float = 0.10
    weight_volume_trend: float = 0.05
    weight_trend_alignment: float = 0.05
    max_extension_atr: float = 3.0


@dataclass(frozen=True)
class CmiprCompressionConfig:
    lookback_bars_30m: int = 12
    atr_period: int = 14
    atr_percentile_lookback: int = 96
    max_atr_percentile: float = 0.45
    max_atr_to_average: float = 0.90
    max_channel_width_atr: float = 5.0
    max_volume_contraction: float = 0.85
    min_range_bars: int = 6
    max_failed_breakouts: int = 2
    max_prior_move_atr: float = 2.5


@dataclass(frozen=True)
class CmiprIgnitionConfig:
    breakout_lookback_15m: int = 20
    min_breakout_distance_atr: float = 0.10
    max_breakout_distance_atr: float = 1.50
    min_body_atr: float = 0.45
    min_close_position: float = 0.70
    max_wick_ratio: float = 0.30
    min_volume_ratio: float = 1.50
    macd_hist_expanding_bars: int = 2
    max_ema21_distance_atr: float = 2.5
    require_oi: bool = False
    min_oi_change_30m: float = -0.01
    max_oi_change_30m: float = 0.04
    require_funding: bool = False
    max_long_funding_rate: float = 0.0005
    min_short_funding_rate: float = -0.0003
    short_risk_multiplier: float = 0.60
    min_derivatives_coverage_pct: float = 0.95
    require_taker_flow: bool = False
    min_long_taker_buy_ratio: float = 0.52
    max_short_taker_buy_ratio: float = 0.48
    require_basis: bool = False
    max_long_basis_pct: float = 0.003
    min_short_basis_pct: float = -0.003


@dataclass(frozen=True)
class CmiprEntryConfig:
    mode: str = "first_pullback"
    pending_expiry_minutes: int = 90
    pullback_timeframe: str = "5m"
    pullback_min_depth_atr: float = 0.10
    pullback_max_depth_atr: float = 1.20
    pullback_max_volume_ratio: float = 0.90
    confirmation_min_close_position: float = 0.60
    max_chase_distance_atr: float = 0.50
    stop_atr_buffer: float = 0.25
    min_stop_atr: float = 0.50
    max_stop_atr: float = 2.50
    initial_risk_fraction: float = 0.40
    min_target_to_cost_ratio: float = 4.0
    extra_execution_delay_minutes: int = 0


@dataclass(frozen=True)
class CmiprPyramidConfig:
    enabled: bool = True
    max_addons: int = 2
    addon_1_risk_fraction: float = 0.30
    addon_2_risk_fraction: float = 0.30
    addon_1_trigger_r: float = 0.60
    addon_2_trigger_r: float = 1.20
    confirmation_timeframe: str = "5m"
    min_confirmation_volume_ratio: float = 1.05
    min_bars_between_addons: int = 2
    extra_execution_delay_minutes: int = 0
    require_full_cost_current_r: bool = True
    include_stop_slippage_in_risk: bool = True
    min_new_stop_noise_atr: float = 0.50


@dataclass(frozen=True)
class CmiprExitConfig:
    fail_fast_bars_5m: int = 3
    fail_fast_min_mfe_r: float = 0.20
    breakeven_trigger_r: float = 0.80
    breakeven_cost_buffer_pct: float = 0.0015
    runner_activation_r: float = 1.20
    trailing_type: str = "ema21_15m"
    chandelier_atr_mult: float = 3.0
    giveback_mode: str = "segmented"
    giveback_r_low: float = 0.70
    giveback_r_mid: float = 0.90
    giveback_r_high: float = 1.20
    giveback_mid_mfe_r: float = 1.50
    giveback_high_mfe_r: float = 3.00
    max_holding_minutes: int = 2880
    runner_enabled: bool = True
    fixed_take_profit_r: float = 1.50


@dataclass(frozen=True)
class CmiprRiskControlConfig:
    max_open_positions: int = 2
    max_same_direction_positions: int = 1
    global_min_entry_interval_seconds: int = 300
    symbol_cooldown_minutes: int = 240
    consecutive_loss_limit: int = 3
    consecutive_loss_pause_minutes: int = 360
    daily_loss_stop_pct: float = 0.015
    soft_drawdown_pct: float = 0.05
    soft_drawdown_risk_multiplier: float = 0.60
    hard_drawdown_pct: float = 0.10


@dataclass(frozen=True)
class CmiprExecutionSafetyConfig:
    protective_stop_required: bool = True
    close_on_protection_failure: bool = True
    max_protection_attempts: int = 2
    order_pending_timeout_seconds: int = 30
    cancel_pending_timeout_seconds: int = 15
    recover_positions_after_restart: bool = True
    adverse_same_bar_ordering: bool = True


@dataclass(frozen=True)
class CmiprResearchConfig:
    model_variant: str = "core"
    max_experiments_per_stage: int = 24
    near_optimal_pf_tolerance: float = 0.03
    prefer_lower_drawdown: bool = True
    historical_test_start: str = "2026-04-01T00:00:00"
    historical_test_end: str = "2026-06-30T23:59:59"
    final_acceptance_source: str = "post_freeze_shadow_or_dry_run"
    sizing_mode: str = "compounding"
    fixed_equity_usdt: float = 160.0
    fixed_trade_risk_usdt: float = 0.80
    min_research_trades: int = 30


@dataclass(frozen=True)
class CmiprStrategyConfig:
    enabled: bool = False
    enabled_symbols: tuple[str, ...] = ()
    allow_long: bool = True
    allow_short: bool = True
    disable_legacy_strategies: bool = True
    auxiliary_oi_data_dir: str = "data/binance_oi_taker_5m"
    auxiliary_funding_data_dir: str = "data/binance_oi_flush_funding"
    auxiliary_basis_data_dir: str = ""
    regime: CmiprRegimeConfig = field(default_factory=CmiprRegimeConfig)
    ranking: CmiprRankingConfig = field(default_factory=CmiprRankingConfig)
    compression: CmiprCompressionConfig = field(default_factory=CmiprCompressionConfig)
    ignition: CmiprIgnitionConfig = field(default_factory=CmiprIgnitionConfig)
    entry: CmiprEntryConfig = field(default_factory=CmiprEntryConfig)
    pyramid: CmiprPyramidConfig = field(default_factory=CmiprPyramidConfig)
    exit: CmiprExitConfig = field(default_factory=CmiprExitConfig)
    risk_control: CmiprRiskControlConfig = field(default_factory=CmiprRiskControlConfig)
    execution_safety: CmiprExecutionSafetyConfig = field(default_factory=CmiprExecutionSafetyConfig)
    research: CmiprResearchConfig = field(default_factory=CmiprResearchConfig)


@dataclass(frozen=True)
class PortfolioControlConfig:
    enabled: bool = False
    max_open_positions: int = 5
    max_vbp_positions: int = 2
    max_indicator_positions: int = 3
    max_altcoin_positions: int = 0
    prevent_same_symbol_overlap: bool = True
    symbol_cooldown_minutes: int = 0
    symbol_loss_cooldown_minutes: int = 0
    vbp_risk_multiplier: float = 1.0
    indicator_risk_multiplier: float = 1.0
    btc_weak_risk_reduction_enabled: bool = False
    btc_weak_1h_return_pct: float = -0.012
    btc_weak_4h_return_pct: float = -0.025
    btc_weak_risk_multiplier: float = 0.5
    weekly_drawdown_control_enabled: bool = False
    weekly_drawdown_reduce_pct: float = 0.10
    weekly_drawdown_stop_pct: float = 0.18
    weekly_drawdown_risk_multiplier: float = 0.5
    weekly_loss_reduce_pct: float = 0.08
    weekly_loss_stop_pct: float = 0.15
    weekly_loss_risk_multiplier: float = 0.5


@dataclass(frozen=True)
class LiveAppConfig:
    exchange: ExchangeConfig
    trading: LiveTradingConfig
    strategy: StrategyConfig
    filters: MultiTimeframeFilterConfig
    risk: LiveRiskConfig
    macro_events: MacroEventConfig
    vbp_strategy: VbpStrategyConfig = field(default_factory=VbpStrategyConfig)
    portfolio_control: PortfolioControlConfig = field(default_factory=PortfolioControlConfig)
    regime_score: RegimeScoreConfig = field(default_factory=RegimeScoreConfig)
    reversal_alpha: ReversalAlphaConfig = field(default_factory=ReversalAlphaConfig)
    cmipr: CmiprStrategyConfig = field(default_factory=CmiprStrategyConfig)


T = TypeVar("T")


def _coerce_dataclass(cls: type[T], values: dict[str, Any]) -> T:
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown config keys for {cls.__name__}: {', '.join(unknown)}")
    if cls is LiveTradingConfig and "symbols" in values:
        values = dict(values)
        values["symbols"] = tuple(_normalize_symbols(values["symbols"]))
    if cls is LiveTradingConfig and "entry_symbols" in values:
        values = dict(values)
        values["entry_symbols"] = tuple(_normalize_symbols(values["entry_symbols"]))
    if cls is LiveTradingConfig and "symbol_leverage_overrides" in values:
        values = dict(values)
        values["symbol_leverage_overrides"] = _normalize_symbol_leverage_overrides(values["symbol_leverage_overrides"])
    if cls is MultiTimeframeFilterConfig and "timeframes" in values:
        values = dict(values)
        values["timeframes"] = tuple(_normalize_timeframes(values["timeframes"]))
    if cls is MacroEventConfig and "symbols" in values:
        values = dict(values)
        values["symbols"] = tuple(_normalize_symbols(values["symbols"]))
    if cls is MacroEventConfig and "primary_symbol" in values:
        values = dict(values)
        primary = _normalize_symbols([values["primary_symbol"]])
        values["primary_symbol"] = primary[0] if primary else "BTCUSDT"
    if cls is MacroEventConfig and "event_types" in values:
        values = dict(values)
        raw_types = values["event_types"]
        if isinstance(raw_types, str):
            parts = raw_types.replace("，", ",").replace("\n", ",").split(",")
        else:
            parts = list(raw_types)
        values["event_types"] = tuple(str(item).strip().upper() for item in parts if str(item).strip())
    return cls(**values)


def _normalize_symbols(value: Any) -> list[str]:
    if isinstance(value, str):
        parts = value.replace("，", ",").replace("\n", ",").split(",")
    else:
        parts = list(value)
    symbols: list[str] = []
    for part in parts:
        symbol = str(part).strip().upper()
        if not symbol:
            continue
        if not symbol.endswith("USDT"):
            symbol = f"{symbol}USDT"
        if symbol not in symbols:
            symbols.append(symbol)
    return symbols


def _normalize_timeframes(value: Any) -> list[str]:
    if isinstance(value, str):
        parts = value.replace("，", ",").replace("\n", ",").split(",")
    else:
        parts = list(value)
    timeframes: list[str] = []
    for part in parts:
        timeframe = str(part).strip()
        if timeframe and timeframe not in timeframes:
            timeframes.append(timeframe)
    return timeframes


def _normalize_symbol_leverage_overrides(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, int] = {}
    for raw_symbol, raw_leverage in value.items():
        symbols = _normalize_symbols([raw_symbol])
        if not symbols:
            continue
        try:
            leverage = int(float(raw_leverage))
        except (TypeError, ValueError):
            continue
        if leverage > 0:
            normalized[symbols[0]] = leverage
    return normalized


def _coerce_vbp_config(values: dict[str, Any]) -> VbpStrategyConfig:
    values = dict(values or {})
    if "enabled_symbols" in values:
        values["enabled_symbols"] = tuple(_normalize_symbols(values["enabled_symbols"]))
    nested = {
        "universe": VbpUniverseConfig,
        "structure_filter": VbpStructureFilterConfig,
        "entry": VbpEntryConfig,
        "exit": VbpExitConfig,
        "position": VbpPositionConfig,
        "market_filter": VbpMarketFilterConfig,
        "risk_control": VbpRiskControlConfig,
    }
    for key, cls in nested.items():
        values[key] = _coerce_dataclass(cls, values.get(key, {}))
    return _coerce_dataclass(VbpStrategyConfig, values)


def _coerce_cmipr_config(values: dict[str, Any]) -> CmiprStrategyConfig:
    values = dict(values or {})
    if "enabled_symbols" in values:
        values["enabled_symbols"] = tuple(_normalize_symbols(values["enabled_symbols"]))
    nested = {
        "regime": CmiprRegimeConfig,
        "ranking": CmiprRankingConfig,
        "compression": CmiprCompressionConfig,
        "ignition": CmiprIgnitionConfig,
        "entry": CmiprEntryConfig,
        "pyramid": CmiprPyramidConfig,
        "exit": CmiprExitConfig,
        "risk_control": CmiprRiskControlConfig,
        "execution_safety": CmiprExecutionSafetyConfig,
        "research": CmiprResearchConfig,
    }
    for key, cls in nested.items():
        values[key] = _coerce_dataclass(cls, values.get(key, {}))
    return _coerce_dataclass(CmiprStrategyConfig, values)


def load_live_config(path: str | Path) -> LiveAppConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return LiveAppConfig(
        exchange=_coerce_dataclass(ExchangeConfig, raw.get("exchange", {})),
        trading=_coerce_dataclass(LiveTradingConfig, raw.get("trading", {})),
        strategy=_coerce_dataclass(StrategyConfig, raw.get("strategy", {})),
        filters=_coerce_dataclass(MultiTimeframeFilterConfig, raw.get("filters", {})),
        risk=_coerce_dataclass(LiveRiskConfig, raw.get("risk", {})),
        macro_events=_coerce_dataclass(MacroEventConfig, raw.get("macro_events", {})),
        vbp_strategy=_coerce_vbp_config(raw.get("vbp_strategy", {})),
        portfolio_control=_coerce_dataclass(PortfolioControlConfig, raw.get("portfolio_control", {})),
        regime_score=_coerce_dataclass(RegimeScoreConfig, raw.get("regime_score", {})),
        reversal_alpha=_coerce_dataclass(ReversalAlphaConfig, raw.get("reversal_alpha", {})),
        cmipr=_coerce_cmipr_config(raw.get("cmipr", {})),
    )


def write_live_config(path: str | Path, config: LiveAppConfig) -> None:
    payload = {
        "exchange": asdict(config.exchange),
        "trading": asdict(config.trading),
        "strategy": asdict(config.strategy),
        "filters": asdict(config.filters),
        "risk": asdict(config.risk),
        "macro_events": asdict(config.macro_events),
        "vbp_strategy": asdict(config.vbp_strategy),
        "portfolio_control": asdict(config.portfolio_control),
        "regime_score": asdict(config.regime_score),
        "reversal_alpha": asdict(config.reversal_alpha),
        "cmipr": asdict(config.cmipr),
    }
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def default_live_config() -> LiveAppConfig:
    return LiveAppConfig(
        exchange=ExchangeConfig(),
        trading=LiveTradingConfig(
            initial_entry_fraction=0.75,
            scale_in_entry_fraction=0.35,
            max_scale_ins_per_symbol=2,
            scale_in_min_profit_pct=0.004,
        ),
        strategy=StrategyConfig(
            fast_ema=8,
            slow_ema=55,
            atr_period=10,
            channel_period=48,
            min_atr_pct=0.004,
            max_atr_pct=0.02,
            breakout_buffer_atr=0.1,
            ema_gap_atr=0.1,
            volume_period=20,
            min_volume_ratio=1.5,
            stop_loss_atr=3.0,
            take_profit_atr=1.2,
            breakeven_atr=0.5,
            trailing_activation_atr=0.5,
            trailing_stop_atr=0.5,
            max_holding_bars=48,
            spike_guard_enabled=True,
            spike_min_range_atr=2.5,
            spike_min_wick_atr=1.4,
            spike_min_wick_ratio=0.7,
            spike_min_volume_ratio=1.0,
            spike_block_bars=3,
            spike_trade_enabled=False,
            spike_recovery_ratio=0.45,
            spike_stop_atr=0.5,
            spike_take_profit_atr=0.7,
            spike_risk_multiplier=0.5,
            spike_max_holding_bars=12,
            rsi_reversal_enabled=False,
            allow_short=True,
            long_score_threshold=0.55,
            short_score_threshold=0.55,
            long_risk_bias=1.0,
            short_risk_bias=0.2,
            regime_filter_enabled=True,
            regime_lookback=12,
            long_min_slow_slope_atr=-0.25,
            short_max_slow_slope_atr=1.5,
        ),
        filters=MultiTimeframeFilterConfig(),
        risk=LiveRiskConfig(),
        macro_events=MacroEventConfig(),
    )
