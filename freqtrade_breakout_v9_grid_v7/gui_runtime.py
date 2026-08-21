from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import secrets
import socket
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


PROJECT_DIR = Path(__file__).resolve().parent
BASE_CONFIG_PATH = PROJECT_DIR / "config.gui.json"
BACKTEST_CONFIG_PATH = PROJECT_DIR / "config.backtest.json"
MANIFEST_PATH = PROJECT_DIR / "gui_manifest.json"
STRATEGY_PATH = (
    PROJECT_DIR
    / "user_data"
    / "strategies"
    / "BreakoutV16GridV15PrecisionGuardScore2DecayLiveParityFreqtrade.py"
)
_LOCAL_STRATEGY_DEPENDENCY_NAMES = (
    "BreakoutV16Score2StructuralDecayResearchFreqtrade.py",
    "BreakoutV16GridV15PrecisionGuardLiveParityFreqtrade.py",
    "BreakoutV16GridV15QualityPfCombinedLiveParityFreqtrade.py",
    "BreakoutV16GridV15QualityPfCombinedResearchFreqtrade.py",
    "GridV15Fixed50QualityProtectedResearchFreqtrade.py",
    "GridV15Fixed50OnlyResearchFreqtrade.py",
    "_GridOnlyResearchMixin.py",
    "BreakoutV16Fixed50MtfAdaptiveBreakoutMax2ResearchFreqtrade.py",
    "BreakoutV15Fixed50BreakoutMax2ResearchFreqtrade.py",
    "_BreakoutMax2ResearchMixin.py",
    "_BreakoutOnlyResearchMixin.py",
    "BreakoutV15Fixed50ShortImpulseStableSelectedFreqtrade.py",
    "BreakoutV15Fixed50CoreShortImpulseMinimumOrderNeighborhoodResearchFreqtrade.py",
    "BreakoutV15Fixed50CoreShortImpulseMinimumOrderResearchFreqtrade.py",
    "BreakoutV15Fixed50CoreTransitionResearchFreqtrade.py",
    "BreakoutV15Fixed50TransitionDefenseResearchFreqtrade.py",
    "BreakoutV15Fixed50GridCampaignLossBudgetResearchFreqtrade.py",
    "BreakoutV15Fixed50ConfirmedRelativeSqueezeResearchFreqtrade.py",
    "BreakoutV15Fixed50WeakImpulseScore5FailureExitResearchFreqtrade.py",
    "BreakoutV15Fixed50Score5FailureExitResearchFreqtrade.py",
    "BreakoutV15Fixed50CrossYearQualityResearchFreqtrade.py",
    "BreakoutV15Fixed50RelativeSqueezeAlwaysResearchFreqtrade.py",
    "BreakoutV15Fixed50RobustGridMomentumResearchFreqtrade.py",
    "BreakoutV15Fixed50RobustQualityResearchFreqtrade.py",
    "BreakoutV15Fixed50WeakImpulseGridLongDefenseResearchFreqtrade.py",
    "BreakoutV14AdaptiveAllocationRangeResearchFreqtrade.py",
    "V14DynamicUniverseSupport.py",
    "BreakoutV13BullCaptureGridRelativeSqueezeSelectedFreqtrade.py",
    "BreakoutV13GridRelativeSqueezeStage29Freqtrade.py",
    "BreakoutV13BullCaptureDrawdownResearchFreqtrade.py",
    "BreakoutV12RegimeAdaptiveGridV9SelectedFreqtrade.py",
    "BreakoutV12RegimeAdaptiveGridV9Freqtrade.py",
    "BreakoutV12MultiRegimeGridV9Freqtrade.py",
    "BreakoutV11AdaptiveGridV8DualSideFreqtrade.py",
    "BreakoutV11DefensiveBoS3Risk50Recovery60Q112GridV8DualSideFreqtrade.py",
    "BreakoutV11PortfolioRecovery60ExhaustionQ112GridV8DualSideFreqtrade.py",
    "BreakoutV11PortfolioRecovery100ExhaustionQ112GridV8DualSideFreqtrade.py",
    "BreakoutV11Trigger18GridV8DualSideFreqtrade.py",
    "BreakoutV11GridV8DualSideFreqtrade.py",
    "BreakoutV10FGridV8DualSideFreqtrade.py",
    "BreakoutV10GridV7Freqtrade.py",
    "BreakoutV9GridV7Freqtrade.py",
)
STRATEGY_DEPENDENCIES = {
    f"user_data/strategies/{name}": (
        PROJECT_DIR / "user_data" / "strategies" / name
    )
    for name in _LOCAL_STRATEGY_DEPENDENCY_NAMES
}
STRATEGY_DEPENDENCIES[
    "../freqtrade_grid_v8_dual_side/user_data/strategies/"
    "GridV8DualSideFreqtrade.py"
] = (
    PROJECT_DIR.parent
    / "freqtrade_grid_v8_dual_side"
    / "user_data"
    / "strategies"
    / "GridV8DualSideFreqtrade.py"
)
USER_DATA_DIR = PROJECT_DIR / "user_data"
LOG_DIR = USER_DATA_DIR / "logs"
# The PID locks intentionally retain their original names so an already-open
# pure-V16 or older console cannot run beside the combined GUI on one account.
MANUAL_RECONCILIATION_LOG_PATH = (
    LOG_DIR / "manual_exchange_reconciliation_v16_grid_v15.jsonl"
)
GUI_LOCK_PATH = LOG_DIR / "breakout_v15_max2_gui.lock"
ENGINE_LOCK_PATH = LOG_DIR / "breakout_v15_max2_engine.lock"
RUNTIME_DIR = PROJECT_DIR / ".runtime" / "conda"
FREQTRADE_BIN = RUNTIME_DIR / "bin" / "freqtrade"
RUNTIME_PYTHON = RUNTIME_DIR / "bin" / "python"

STRATEGY_CLASS = (
    "BreakoutV16GridV15PrecisionGuardScore2DecayLiveParityFreqtrade"
)
RELEASE_LABEL = (
    "Breakout V16 + Grid V15 PF · Score 2 结构衰减 · 1m 精度止损 · 共享 Max2"
)
DEFAULT_DRY_WALLET = 200.0
MAX_OPEN_TRADES = 2
BREAKOUT_OPEN_LIMIT = 1
GRID_OPEN_LIMIT = 1
PAIR_COUNT = 50

MODE_DRY = "DRY-RUN"
MODE_LIVE = "LIVE"
VALID_MODES = {MODE_DRY, MODE_LIVE}

BINANCE_FUTURES_TIME_URL = "https://fapi.binance.com/fapi/v1/time"
CLOCK_SAFETY_LAG_MS = 2_000
GUI_LOG_ARCHIVE_BYTES = 10 * 1024 * 1024

KEY_NAMES = (
    "BINANCE_FUTURES_API_KEY",
    "BINANCE_API_KEY",
    "FREQTRADE__EXCHANGE__KEY",
)
SECRET_NAMES = (
    "BINANCE_FUTURES_API_SECRET",
    "BINANCE_API_SECRET",
    "FREQTRADE__EXCHANGE__SECRET",
)


@dataclass(frozen=True)
class SecretBundle:
    key: str
    secret: str
    source: str

    @property
    def ready(self) -> bool:
        return bool(self.key and self.secret)


@dataclass(frozen=True)
class ValidationReport:
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    details: dict[str, Any]

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class ApiCredentials:
    username: str
    password: str
    jwt_secret: str
    ws_token: str

    @classmethod
    def create(cls) -> "ApiCredentials":
        return cls(
            username=f"gui_{secrets.token_hex(5)}",
            password=secrets.token_urlsafe(24),
            jwt_secret=secrets.token_urlsafe(48),
            ws_token=secrets.token_urlsafe(32),
        )


@dataclass(frozen=True)
class LaunchSpec:
    mode: str
    command: tuple[str, ...]
    environment: dict[str, str]
    overlay_path: Path
    database_path: Path
    logfile_path: Path
    api_port: int
    api_credentials: ApiCredentials
    secret_bundle: SecretBundle
    time_difference_ms: int | None
    nfp_enabled: bool
    strategy_class: str
    max_open_trades: int


@dataclass(frozen=True)
class ManualPositionChange:
    """A ledger position that became smaller on the exchange without a bot order."""

    trade_id: int
    pair: str
    expected_side: str
    kind: str
    managed_amount: float
    exchange_amount: float

    @property
    def description(self) -> str:
        action = "已在交易所全部平仓" if self.kind == "full_close" else "已在交易所部分减仓"
        return (
            f"{self.pair} {action}：账本={self.managed_amount:g}，"
            f"交易所={self.exchange_amount:g}"
        )


@dataclass(frozen=True)
class PositionReconciliationReport:
    """Classify safe manual reductions separately from unsafe position drift."""

    manual_changes: tuple[ManualPositionChange, ...]
    errors: tuple[str, ...]


@dataclass(frozen=True)
class ManualReconciliationMarkers:
    trade_ids: frozenset[int]
    order_ids: frozenset[str]


@dataclass(frozen=True)
class ClockSyncReport:
    local_minus_server_ms: int
    round_trip_ms: float
    time_difference_ms: int
    sample_count: int


class EngineOutputReducer:
    """Keep the GUI readable while the complete engine log remains on disk."""

    _timestamped_log = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[,.]\d+ - ")
    _clock_error = re.compile(
        r"InvalidNonce|(?:code[\"']?\s*:\s*)?-1021|"
        r"Timestamp for this request was 1000ms ahead",
        re.IGNORECASE,
    )
    _traceback_start = re.compile(
        r"^(?:Traceback \(most recent call last\):|"
        r"During handling of the above exception|"
        r"The above exception was the direct cause)"
    )

    def __init__(self, *, repeat_window_seconds: float = 60.0) -> None:
        self.repeat_window_seconds = float(repeat_window_seconds)
        self.in_traceback = False
        self.last_clock_emit_at = -float("inf")
        self.last_clock_seen_at = -float("inf")

    def reduce(
        self,
        line: str,
        *,
        now: float | None = None,
    ) -> tuple[str, str] | None:
        cleaned = str(line).strip()
        if not cleaned:
            return None
        current = time.monotonic() if now is None else float(now)

        if self._clock_error.search(cleaned):
            self.last_clock_seen_at = current
            self.in_traceback = True
            if current - self.last_clock_emit_at < self.repeat_window_seconds:
                return None
            self.last_clock_emit_at = current
            return (
                "Binance 时间校验异常（-1021）：已暂停高频账户刷新并进入退避；"
                "完整堆栈保留在 Freqtrade 原始日志。",
                "clock_error",
            )

        if (
            "Exception in ASGI application" in cleaned
            and current - self.last_clock_seen_at < self.repeat_window_seconds
        ):
            self.in_traceback = True
            return None
        if self._traceback_start.match(cleaned):
            self.in_traceback = True
            return None
        if self._timestamped_log.match(cleaned):
            self.in_traceback = False
            return cleaned, "engine"
        if self.in_traceback:
            return None
        return cleaned, "engine"


def process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def active_pid_lock(path: Path, *, clean_stale: bool = True) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = load_json(path)
        pid = int(payload.get("pid") or 0)
    except (OSError, ValueError, json.JSONDecodeError):
        return {
            "pid": 0,
            "path": str(path),
            "error": "锁文件损坏，需人工检查",
        }
    if process_is_alive(pid):
        return payload
    if clean_stale:
        try:
            path.unlink()
        except OSError:
            return {
                "pid": pid,
                "path": str(path),
                "error": "过期锁文件无法清理",
            }
    return None


def acquire_pid_lock(
    path: Path,
    *,
    pid: int,
    kind: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    existing = active_pid_lock(path)
    if existing is not None:
        raise RuntimeError(
            f"{kind} 已由 PID {existing.get('pid') or '?'} 占用：{path}"
        )
    payload: dict[str, Any] = {
        "pid": int(pid),
        "kind": kind,
        "created_at": datetime_now_iso(),
    }
    if details:
        payload.update(details)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as exc:
        existing = active_pid_lock(path, clean_stale=False) or {}
        raise RuntimeError(
            f"{kind} 已由 PID {existing.get('pid') or '?'} 占用：{path}"
        ) from exc
    try:
        os.write(descriptor, encoded)
    finally:
        os.close(descriptor)
    return payload


def release_pid_lock(path: Path, *, pid: int) -> bool:
    if not path.is_file():
        return True
    try:
        payload = load_json(path)
        if int(payload.get("pid") or 0) != int(pid):
            return False
        path.unlink()
        return True
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def datetime_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def universe_sha256(pairs: list[str]) -> str:
    encoded = json.dumps(pairs, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strip_optional_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value


def parse_dotenv(path: Path) -> dict[str, str]:
    """Read simple dotenv assignments without mutating process environment."""

    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        value = raw_value.strip()
        if value and value[0] not in {"'", '"'}:
            value = value.split(" #", 1)[0].rstrip()
        values[key] = _strip_optional_quotes(value)
    return values


def dotenv_candidates() -> tuple[Path, ...]:
    override = (
        os.environ.get("BREAKOUT_V16_GUI_ENV_FILE", "").strip()
        or os.environ.get("BREAKOUT_V15_GUI_ENV_FILE", "").strip()
        or os.environ.get("V11_ADAPTIVE_GUI_ENV_FILE", "").strip()
        or os.environ.get("V10F_GUI_ENV_FILE", "").strip()
    )
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override).expanduser())
    candidates.extend(
        (
            PROJECT_DIR.parent / ".env",
            PROJECT_DIR / ".env",
        )
    )
    result: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in result:
            result.append(resolved)
    return tuple(result)


def _first_value(
    sources: Iterable[tuple[str, dict[str, str]]],
    names: Iterable[str],
) -> tuple[str, str]:
    for source_name, values in sources:
        for name in names:
            value = values.get(name, "").strip()
            if value:
                return value, source_name
    return "", ""


def load_exchange_secrets() -> SecretBundle:
    source_maps: list[tuple[str, dict[str, str]]] = []
    for path in dotenv_candidates():
        if path.is_file():
            source_maps.append((str(path), parse_dotenv(path)))
    source_maps.append(("当前进程环境", dict(os.environ)))

    key, key_source = _first_value(source_maps, KEY_NAMES)
    secret, secret_source = _first_value(source_maps, SECRET_NAMES)
    if key_source and key_source == secret_source:
        source = key_source
    elif key_source and secret_source:
        source = f"{key_source} / {secret_source}"
    else:
        source = key_source or secret_source
    return SecretBundle(key=key, secret=secret, source=source)


def redact_sensitive(text: str, bundle: SecretBundle | None = None) -> str:
    redacted = str(text)
    if bundle:
        for value in (bundle.key, bundle.secret):
            if value and len(value) >= 4:
                redacted = redacted.replace(value, "[REDACTED]")
    patterns = (
        r"(?i)(api[_ -]?key\s*[=:]\s*)[^\s,;]+",
        r"(?i)(secret(?:[_ -]?key)?\s*[=:]\s*)[^\s,;]+",
        r'(?i)(["\'](?:key|secret)["\']\s*:\s*["\'])[^"\']+',
        r"(?i)(signature=)[0-9a-f]+",
    )
    for pattern in patterns:
        redacted = re.sub(pattern, r"\1[REDACTED]", redacted)
    return redacted


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} 顶层必须是 JSON 对象")
    return value


def clock_report_from_samples(
    samples: Iterable[tuple[float, float]],
    *,
    safety_lag_ms: int = CLOCK_SAFETY_LAG_MS,
) -> ClockSyncReport:
    values = tuple(
        (float(round_trip_ms), float(local_minus_server_ms))
        for round_trip_ms, local_minus_server_ms in samples
    )
    if not values:
        raise ValueError("Binance 时间采样为空")
    best_round_trip, best_offset = min(values, key=lambda item: item[0])
    offset_ms = int(round(best_offset))
    return ClockSyncReport(
        local_minus_server_ms=offset_ms,
        round_trip_ms=round(best_round_trip, 1),
        time_difference_ms=offset_ms + int(safety_lag_ms),
        sample_count=len(values),
    )


def measure_binance_clock(
    *,
    sample_count: int = 3,
    timeout: float = 4.0,
) -> ClockSyncReport:
    """Measure Binance time and add a safe lag for long-running signed requests."""

    if sample_count < 1:
        raise ValueError("时间采样次数必须大于 0")
    samples: list[tuple[float, float]] = []
    request = urllib.request.Request(
        BINANCE_FUTURES_TIME_URL,
        headers={
            "Accept": "application/json",
            "User-Agent": "breakout-v16-grid-v15-combined-gui",
        },
    )
    for _index in range(sample_count):
        before_ms = time.time_ns() / 1_000_000
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
            json.JSONDecodeError,
        ) as exc:
            if not samples:
                raise ConnectionError(f"无法读取 Binance 服务器时间：{exc}") from exc
            break
        after_ms = time.time_ns() / 1_000_000
        try:
            server_ms = float(payload["serverTime"])
        except (KeyError, TypeError, ValueError) as exc:
            if not samples:
                raise RuntimeError("Binance 时间响应格式错误") from exc
            break
        midpoint_ms = (before_ms + after_ms) / 2.0
        samples.append((after_ms - before_ms, midpoint_ms - server_ms))
        if sample_count > 1:
            time.sleep(0.05)
    return clock_report_from_samples(samples)


def friendly_runtime_error(message: str) -> str:
    normalized = " ".join(str(message).split())
    lowered = normalized.lower()
    if "invalidnonce" in lowered or "-1021" in lowered:
        return (
            "Binance 时间校验失败（-1021）：本机时间与交易所时间发生偏移，"
            "账户刷新已自动退避。"
        )
    if "too many requests" in lowered or "rate limit" in lowered or " 429" in lowered:
        return "Binance 请求频率受限（429）：账户刷新已自动降频。"
    if "freqtrade api 500" in lowered:
        return "Freqtrade 账户接口暂时不可用，已自动降低刷新频率。"
    if len(normalized) > 320:
        return normalized[:317] + "..."
    return normalized or "未知错误"


def poll_backoff_seconds(
    consecutive_failures: int,
    *,
    base_seconds: float = 10.0,
    maximum_seconds: float = 300.0,
) -> float:
    failures = max(1, int(consecutive_failures))
    return min(float(maximum_seconds), float(base_seconds) * (2 ** failures))


def archive_oversized_log(
    path: Path,
    *,
    maximum_bytes: int = GUI_LOG_ARCHIVE_BYTES,
) -> Path | None:
    """Move an oversized GUI log aside without deleting diagnostic history."""

    if not path.is_file() or path.stat().st_size <= int(maximum_bytes):
        return None
    stamp = time.strftime("%Y%m%d_%H%M%S")
    candidate = path.with_name(f"{path.stem}.{stamp}.archive{path.suffix}")
    counter = 1
    while candidate.exists():
        candidate = path.with_name(
            f"{path.stem}.{stamp}.{counter}.archive{path.suffix}"
        )
        counter += 1
    path.replace(candidate)
    return candidate


def verify_release(
    *,
    base_config_path: Path = BASE_CONFIG_PATH,
    manifest_path: Path = MANIFEST_PATH,
    strategy_path: Path = STRATEGY_PATH,
    require_runtime: bool = True,
) -> ValidationReport:
    errors: list[str] = []
    warnings: list[str] = []
    details: dict[str, Any] = {}

    for label, path in (
        ("基础配置", base_config_path),
        ("回测配置", BACKTEST_CONFIG_PATH),
        ("GUI 冻结清单", manifest_path),
        ("策略文件", strategy_path),
    ):
        if not path.is_file():
            errors.append(f"{label}不存在：{path}")
    for relative, path in STRATEGY_DEPENDENCIES.items():
        if not path.is_file():
            errors.append(f"策略依赖不存在：{relative}")

    if errors:
        return ValidationReport(tuple(errors), tuple(warnings), details)

    try:
        config = load_json(base_config_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"基础配置读取失败：{exc}")
        config = {}
    try:
        manifest = load_json(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"冻结清单读取失败：{exc}")
        manifest = {}
    try:
        backtest_config = load_json(BACKTEST_CONFIG_PATH)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"回测配置读取失败：{exc}")
        backtest_config = {}

    if config:
        whitelist = config.get("exchange", {}).get("pair_whitelist", [])
        checks = (
            (
                config.get("strategy") == STRATEGY_CLASS,
                f"策略必须为 {STRATEGY_CLASS}",
            ),
            (
                config.get("max_open_trades") == MAX_OPEN_TRADES,
                "最大持仓必须为 2",
            ),
            (
                config.get("trading_mode") == "futures",
                "交易模式必须为 futures",
            ),
            (
                config.get("margin_mode") == "isolated",
                "保证金模式必须为 isolated（逐仓）",
            ),
            (
                config.get("exchange", {}).get("name") == "binance",
                "交易所必须为 Binance",
            ),
            (
                len(whitelist) == PAIR_COUNT,
                f"交易池必须为 {PAIR_COUNT} 个币，当前为 {len(whitelist)}",
            ),
            (
                config.get("force_entry_enable") is False,
                "必须禁用 API 强制开仓",
            ),
            (
                config.get("pairlists")
                == [{"method": "StaticPairList", "allow_inactive": False}],
                "GUI 执行配置必须过滤交易所不活跃市场",
            ),
        )
        errors.extend(message for passed, message in checks if not passed)
        exchange_config = config.get("exchange", {})
        if exchange_config.get("key") or exchange_config.get("secret"):
            errors.append("基础配置中不得保存 API Key/Secret")
        details.update(
            {
                "strategy": config.get("strategy"),
                "pair_count": len(whitelist),
                "universe_sha256": universe_sha256(whitelist),
                "max_open_trades": config.get("max_open_trades"),
                "trading_mode": config.get("trading_mode"),
                "margin_mode": config.get("margin_mode"),
            }
        )

    strategy_hash = sha256_file(strategy_path)
    config_hash = sha256_file(base_config_path)
    backtest_config_hash = sha256_file(BACKTEST_CONFIG_PATH)
    details["strategy_sha256"] = strategy_hash
    details["config_sha256"] = config_hash
    details["backtest_config_sha256"] = backtest_config_hash
    dependency_hashes = {
        relative: sha256_file(path)
        for relative, path in STRATEGY_DEPENDENCIES.items()
        if path.is_file()
    }
    details["dependency_sha256"] = dependency_hashes
    if config and backtest_config:
        gui_pairs = config.get("exchange", {}).get("pair_whitelist", [])
        backtest_pairs = (
            backtest_config.get("exchange", {}).get("pair_whitelist", [])
        )
        if gui_pairs != backtest_pairs:
            errors.append("GUI 与回测配置的50币名单不一致")
        if backtest_config.get("strategy") != STRATEGY_CLASS:
            errors.append("回测配置的策略类与 GUI 不一致")
        parity_keys = (
            "timeframe",
            "max_open_trades",
            "stake_currency",
            "stake_amount",
            "tradable_balance_ratio",
            "trading_mode",
            "margin_mode",
            "liquidation_buffer",
        )
        mismatched = [
            key
            for key in parity_keys
            if config.get(key) != backtest_config.get(key)
        ]
        if mismatched:
            errors.append(
                "GUI 与回测关键执行参数不一致：" + ",".join(mismatched)
            )
    if manifest:
        if manifest.get("strategy_class") != STRATEGY_CLASS:
            errors.append("冻结清单的策略类不一致")
        if manifest.get("strategy_sha256") != strategy_hash:
            errors.append("策略文件校验失败：内容已偏离 GUI 冻结版本")
        if manifest.get("config_sha256") != config_hash:
            errors.append("基础配置校验失败：内容已偏离 GUI 冻结版本")
        if manifest.get("backtest_config_sha256") != backtest_config_hash:
            errors.append("回测配置校验失败：内容已偏离 GUI 冻结版本")
        if manifest.get("dependency_sha256") != dependency_hashes:
            errors.append("策略依赖校验失败：V16 + Grid V15 已偏离冻结版本")
        if manifest.get("pair_count") != PAIR_COUNT:
            errors.append("冻结清单的币种数量不一致")
        if manifest.get("universe_sha256") != details.get("universe_sha256"):
            errors.append("冻结清单的50币名单与 GUI 配置不一致")
        if manifest.get("max_open_trades") != MAX_OPEN_TRADES:
            errors.append("冻结清单的最大持仓数不一致")
        if manifest.get("breakout_open_limit") != BREAKOUT_OPEN_LIMIT:
            errors.append("冻结清单的 Breakout 同时持仓上限必须为 1")
        if manifest.get("grid_open_limit") != GRID_OPEN_LIMIT:
            errors.append("冻结清单的 Grid 同时持仓上限必须为 1")
        if manifest.get("grid_enabled") is not True:
            errors.append("冻结清单必须明确启用 Grid V15 PF")
        if manifest.get("last_slot_priority") != "breakout":
            errors.append("冻结清单必须保留最后一格 Breakout 优先规则")
        if manifest.get("nfp_enabled") is not False:
            errors.append("冻结清单必须明确禁用非农追加仓")

    if require_runtime:
        if not FREQTRADE_BIN.is_file():
            errors.append(f"Freqtrade 运行文件不存在：{FREQTRADE_BIN}")
        elif not os.access(FREQTRADE_BIN, os.X_OK):
            errors.append(f"Freqtrade 运行文件不可执行：{FREQTRADE_BIN}")
        if not RUNTIME_PYTHON.is_file():
            warnings.append(f"独立 Python 运行文件不存在：{RUNTIME_PYTHON}")

    return ValidationReport(tuple(errors), tuple(warnings), details)


def find_free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def build_runtime_overlay(
    mode: str,
    port: int,
    credentials: ApiCredentials,
    *,
    time_difference_ms: int | None = None,
    nfp_enabled: bool = False,
) -> dict[str, Any]:
    if mode not in VALID_MODES:
        raise ValueError(f"无效运行模式：{mode}")
    if nfp_enabled:
        raise ValueError("当前 GUI 仅允许运行 V16 + Grid V15 PF 组合策略")
    overlay: dict[str, Any] = {
        "bot_name": (
            "breakout_v16_grid_v15_quality_pf_combined_dry"
            if mode == MODE_DRY
            else "breakout_v16_grid_v15_quality_pf_combined_live"
        ),
        "strategy": STRATEGY_CLASS,
        "max_open_trades": MAX_OPEN_TRADES,
        "dry_run": mode == MODE_DRY,
        "dry_run_wallet": DEFAULT_DRY_WALLET,
        "initial_state": "stopped",
        "force_entry_enable": False,
        "api_server": {
            "enabled": True,
            "listen_ip_address": "127.0.0.1",
            "listen_port": int(port),
            "verbosity": "error",
            "enable_openapi": False,
            "jwt_secret_key": credentials.jwt_secret,
            "ws_token": credentials.ws_token,
            "CORS_origins": [],
            "username": credentials.username,
            "password": credentials.password,
        },
        "internals": {
            "process_throttle_secs": 5,
        },
    }
    if time_difference_ms is not None:
        base_exchange = json.loads(
            json.dumps(load_json(BASE_CONFIG_PATH).get("exchange", {}))
        )
        if not base_exchange.get("name"):
            raise RuntimeError("基础配置缺少交易所名称")
        base_exchange.pop("key", None)
        base_exchange.pop("secret", None)
        options = {
            "defaultType": "future",
            "adjustForTimeDifference": False,
            "timeDifference": int(time_difference_ms),
            # Binance -4046 means the requested margin mode is already active.
            # Returning it as an idempotent success lets Freqtrade continue to
            # leverage/order creation instead of retrying a completed action.
            "setMarginMode": {
                "throwMarginModeAlreadySet": False,
            },
        }
        base_exchange["ccxt_config"] = {
            "enableRateLimit": True,
            "options": dict(options),
        }
        base_exchange["ccxt_async_config"] = {
            "enableRateLimit": True,
            "options": dict(options),
        }
        overlay["exchange"] = base_exchange
    return overlay


def secure_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
        0o600,
    )
    try:
        os.write(descriptor, encoded)
    finally:
        os.close(descriptor)


def sqlite_url(path: Path) -> str:
    return f"sqlite:///{path.resolve()}"


def _clean_child_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for name in (
        *KEY_NAMES,
        *SECRET_NAMES,
        "FREQTRADE__DRY_RUN",
        "FREQTRADE__INITIAL_STATE",
        "FREQTRADE__API_SERVER__USERNAME",
        "FREQTRADE__API_SERVER__PASSWORD",
        "FREQTRADE__API_SERVER__JWT_SECRET_KEY",
        "FREQTRADE__API_SERVER__WS_TOKEN",
    ):
        environment.pop(name, None)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["NO_COLOR"] = "1"
    return environment


def build_launch_spec(
    mode: str,
    *,
    port: int | None = None,
    credentials: ApiCredentials | None = None,
    bundle: SecretBundle | None = None,
    session_token: str | None = None,
    time_difference_ms: int | None = None,
    write_overlay: bool = True,
    nfp_enabled: bool = False,
) -> LaunchSpec:
    if mode not in VALID_MODES:
        raise ValueError(f"无效运行模式：{mode}")
    if nfp_enabled:
        raise ValueError("当前 GUI 仅允许运行 V16 + Grid V15 PF 组合策略")
    report = verify_release()
    if not report.ok:
        raise RuntimeError("；".join(report.errors))

    api_port = int(port or find_free_local_port())
    api_credentials = credentials or ApiCredentials.create()
    secret_bundle = bundle or load_exchange_secrets()
    if mode == MODE_LIVE and not secret_bundle.ready:
        raise RuntimeError("原 .env 中未找到完整的 Binance Futures API Key/Secret")

    token = session_token or secrets.token_hex(8)
    suffix = "dryrun" if mode == MODE_DRY else "live"
    overlay_path = (
        LOG_DIR
        / "runtime"
        / f"breakout_v16_grid_v15_combined_{suffix}_{token}.json"
    )
    database_path = (
        USER_DATA_DIR
        / f"tradesv3.breakout_v16_grid_v15_combined.{suffix}.sqlite"
    )
    logfile_path = (
        LOG_DIR / f"freqtrade_breakout_v16_grid_v15_combined_{suffix}.log"
    )
    overlay = build_runtime_overlay(
        mode,
        api_port,
        api_credentials,
        time_difference_ms=time_difference_ms,
        nfp_enabled=nfp_enabled,
    )
    if write_overlay:
        secure_write_json(overlay_path, overlay)

    environment = _clean_child_environment()
    if mode == MODE_LIVE:
        environment["FREQTRADE__EXCHANGE__KEY"] = secret_bundle.key
        environment["FREQTRADE__EXCHANGE__SECRET"] = secret_bundle.secret

    command: list[str] = [
        str(FREQTRADE_BIN),
        "trade",
        "--no-color",
        "--config",
        str(BASE_CONFIG_PATH),
        "--config",
        str(overlay_path),
        "--userdir",
        str(USER_DATA_DIR),
        "--strategy",
        STRATEGY_CLASS,
        "--db-url",
        sqlite_url(database_path),
        "--logfile",
        str(logfile_path),
    ]
    if mode == MODE_DRY:
        command.extend(
            (
                "--dry-run",
                "--dry-run-wallet",
                f"{DEFAULT_DRY_WALLET:.2f}",
            )
        )

    return LaunchSpec(
        mode=mode,
        command=tuple(command),
        environment=environment,
        overlay_path=overlay_path,
        database_path=database_path,
        logfile_path=logfile_path,
        api_port=api_port,
        api_credentials=api_credentials,
        secret_bundle=secret_bundle,
        time_difference_ms=time_difference_ms,
        nfp_enabled=False,
        strategy_class=STRATEGY_CLASS,
        max_open_trades=MAX_OPEN_TRADES,
    )


class FreqtradeApiClient:
    def __init__(
        self,
        port: int,
        credentials: ApiCredentials,
        *,
        timeout: float = 4.0,
    ) -> None:
        self.base_url = f"http://127.0.0.1:{int(port)}/api/v1"
        token = base64.b64encode(
            f"{credentials.username}:{credentials.password}".encode("utf-8")
        ).decode("ascii")
        self.authorization = f"Basic {token}"
        self.timeout = timeout

    def request(
        self,
        path: str,
        *,
        method: str = "GET",
        payload: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Any:
        normalized = "/" + path.lstrip("/")
        data = None
        headers = {
            "Accept": "application/json",
            "Authorization": self.authorization,
        }
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.base_url + normalized,
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout if timeout is None else timeout,
            ) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Freqtrade API {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ConnectionError(f"Freqtrade API 不可用：{exc}") from exc
        if not body:
            return {}
        try:
            return json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError("Freqtrade API 返回了无效 JSON") from exc

    def ping(self, *, timeout: float = 1.0) -> bool:
        try:
            result = self.request("/ping", timeout=timeout)
        except (ConnectionError, RuntimeError):
            return False
        return isinstance(result, dict) and result.get("status") == "pong"

    def get(self, path: str) -> Any:
        return self.request(path)

    def post(self, path: str) -> Any:
        return self.request(path, method="POST")


def validate_running_config(
    config: dict[str, Any],
    mode: str,
    *,
    nfp_enabled: bool = False,
) -> tuple[str, ...]:
    errors: list[str] = []
    expected_dry = mode == MODE_DRY
    if nfp_enabled:
        errors.append("当前 GUI 仅允许运行 V16 + Grid V15 PF 组合策略")
    checks = (
        (
            config.get("strategy") == STRATEGY_CLASS,
            f"运行策略不是 {STRATEGY_CLASS}",
        ),
        (
            bool(config.get("dry_run")) is expected_dry,
            "运行模式与 GUI 选择不一致",
        ),
        (
            config.get("trading_mode") == "futures",
            "运行交易模式不是 futures",
        ),
        (
            config.get("margin_mode") == "isolated",
            "运行保证金模式不是 isolated",
        ),
        (
            int(config.get("max_open_trades", -1)) == MAX_OPEN_TRADES,
            f"运行最大持仓数不是 {MAX_OPEN_TRADES}",
        ),
        (
            config.get("force_entry_enable") is False,
            "运行配置意外启用了强制开仓",
        ),
        (
            str(config.get("state", "")).lower() == "stopped",
            "预检时 Freqtrade 未处于停止交易状态",
        ),
    )
    errors.extend(message for passed, message in checks if not passed)
    return tuple(errors)


def extract_account_snapshot(balance: dict[str, Any]) -> dict[str, Any]:
    currencies = balance.get("currencies") or []
    stake = str(balance.get("stake") or "USDT")
    stake_row = next(
        (
            row
            for row in currencies
            if str(row.get("currency") or "").upper() == stake.upper()
            and not row.get("is_position")
        ),
        {},
    )
    positions = [
        row
        for row in currencies
        if row.get("is_position") and abs(float(row.get("position") or 0.0)) > 0
    ]
    return {
        "equity": float(balance.get("total") or 0.0),
        "available": float(stake_row.get("free") or 0.0),
        "stake": stake,
        "exchange_positions": positions,
        "unmanaged_positions": [
            row for row in positions if not bool(row.get("is_bot_managed"))
        ],
    }


def normalize_pair_key(value: str) -> str:
    normalized = re.sub(r"[^A-Z0-9]", "", str(value).upper())
    return normalized.replace("USDTUSDT", "USDT")


def analyze_position_reconciliation(
    managed_trades: list[dict[str, Any]],
    exchange_positions: list[dict[str, Any]],
) -> PositionReconciliationReport:
    """Separate verifiable manual reductions from unsafe account drift.

    Missing or smaller exchange positions are eligible for Freqtrade's official
    ``reload trade from exchange`` recovery path.  Direction changes, manual
    additions, malformed trade identifiers, and numeric parsing failures remain
    hard errors and must never be auto-adopted by the bot.
    """

    exchange_by_pair = {
        normalize_pair_key(str(position.get("currency") or position.get("pair") or "")):
        position
        for position in exchange_positions
    }
    changes: list[ManualPositionChange] = []
    errors: list[str] = []
    for trade in managed_trades:
        pair = str(trade.get("pair") or "")
        key = normalize_pair_key(pair)
        position = exchange_by_pair.get(key)
        has_open_orders = bool(trade.get("has_open_orders"))
        expected_side = "short" if bool(trade.get("is_short")) else "long"

        try:
            trade_id = int(trade.get("trade_id") or trade.get("id") or 0)
            managed_amount = abs(float(trade.get("amount") or 0.0))
        except (TypeError, ValueError):
            errors.append(f"{pair or '?'} 账本持仓字段无法解析")
            continue

        if position is None:
            if has_open_orders:
                # An order fill and the wallet snapshot can briefly cross in flight.
                continue
            if trade_id > 0 and managed_amount > 0:
                changes.append(
                    ManualPositionChange(
                        trade_id=trade_id,
                        pair=pair,
                        expected_side=expected_side,
                        kind="full_close",
                        managed_amount=managed_amount,
                        exchange_amount=0.0,
                    )
                )
            else:
                errors.append(f"{pair} 账本有持仓但交易所无对应仓位")
            continue

        exchange_side = str(position.get("side") or "").lower()
        if exchange_side and exchange_side != expected_side:
            errors.append(
                f"{pair} 方向不一致：账本={expected_side}，交易所={exchange_side}"
            )
            continue

        # Older/mocked balance payloads may omit an amount.  Direction checking
        # remains useful, but amount reconciliation is impossible for those rows.
        if "position" not in position and "amount" not in position and "contracts" not in position:
            continue
        try:
            exchange_amount = abs(
                float(
                    position.get("position")
                    if position.get("position") is not None
                    else position.get("amount")
                    if position.get("amount") is not None
                    else position.get("contracts")
                )
            )
        except (TypeError, ValueError):
            errors.append(f"{pair} 交易所持仓数量无法解析")
            continue

        if managed_amount <= 0:
            errors.append(f"{pair} 账本持仓数量无效：{managed_amount:g}")
            continue
        tolerance = max(1e-12, managed_amount * 1e-6)
        if exchange_amount < managed_amount - tolerance:
            if has_open_orders:
                continue
            if trade_id <= 0:
                errors.append(f"{pair} 缺少有效 trade_id，无法安全同步人工减仓")
                continue
            changes.append(
                ManualPositionChange(
                    trade_id=trade_id,
                    pair=pair,
                    expected_side=expected_side,
                    kind="full_close" if exchange_amount <= tolerance else "partial_close",
                    managed_amount=managed_amount,
                    exchange_amount=exchange_amount,
                )
            )
        elif exchange_amount > managed_amount + tolerance:
            errors.append(
                f"{pair} 交易所仓位大于账本：账本={managed_amount:g}，"
                f"交易所={exchange_amount:g}；疑似人工加仓，禁止自动接管"
            )

    return PositionReconciliationReport(tuple(changes), tuple(errors))


def validate_position_reconciliation(
    managed_trades: list[dict[str, Any]],
    exchange_positions: list[dict[str, Any]],
) -> tuple[str, ...]:
    """Detect stale ledger trades and direction mismatches before LIVE starts."""

    report = analyze_position_reconciliation(managed_trades, exchange_positions)
    manual_errors = tuple(change.description for change in report.manual_changes)
    return report.errors + manual_errors


def append_manual_reconciliation_audit(
    records: Iterable[dict[str, Any]],
    *,
    path: Path = MANUAL_RECONCILIATION_LOG_PATH,
) -> int:
    """Append sanitized manual-close records to a durable, owner-only JSONL file."""

    prepared: list[bytes] = []
    for record in records:
        payload = {
            "recorded_at": datetime_now_iso(),
            "source": "freqtrade_reload_trade_from_exchange",
            **record,
        }
        prepared.append(
            (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        )
    if not prepared:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o600,
    )
    try:
        for row in prepared:
            os.write(descriptor, row)
    finally:
        os.close(descriptor)
    return len(prepared)


def load_manual_reconciliation_markers(
    *,
    path: Path = MANUAL_RECONCILIATION_LOG_PATH,
) -> ManualReconciliationMarkers:
    """Load idempotency markers without treating a malformed audit row as truth."""

    trade_ids: set[int] = set()
    order_ids: set[str] = set()
    if not path.is_file():
        return ManualReconciliationMarkers(frozenset(), frozenset())
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ManualReconciliationMarkers(frozenset(), frozenset())
    for line in lines:
        try:
            record = json.loads(line)
            trade_id = int(record.get("trade_id") or 0)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if trade_id > 0:
            trade_ids.add(trade_id)
        for order in record.get("exchange_orders") or []:
            order_id = str(order.get("order_id") or "")
            if order_id:
                order_ids.add(order_id)
    return ManualReconciliationMarkers(
        frozenset(trade_ids),
        frozenset(order_ids),
    )


def classify_component(enter_tag: str | None) -> str:
    tag = str(enter_tag or "").lower()
    if tag.startswith("nfp_v4_"):
        return "非农 Stability v4（旧）"
    if tag.startswith("bo_"):
        return "Breakout V16 MTF"
    if tag.startswith("grid_v8_"):
        return "Grid V15 PF"
    if tag.startswith("grid_v7_"):
        return "Grid v7（旧）"
    return "未知"


def has_open_nfp_trade(trades: Iterable[dict[str, Any]]) -> bool:
    return any(
        str(trade.get("enter_tag") or "").lower().startswith("nfp_v4_")
        for trade in trades
    )


def compact_command(command: Iterable[str]) -> str:
    """Return a safe diagnostic command that never contains exchange secrets."""

    return " ".join(
        f'"{item}"' if any(character.isspace() for character in item) else item
        for item in command
    )


def run_static_check() -> int:
    report = verify_release()
    bundle = load_exchange_secrets()
    print(f"release={RELEASE_LABEL}")
    print(f"strategy={report.details.get('strategy', '-')}")
    print(f"pairs={report.details.get('pair_count', '-')}")
    print(f"max_open_trades={report.details.get('max_open_trades', '-')}")
    print(f"runtime={'OK' if FREQTRADE_BIN.is_file() else 'MISSING'}")
    print(f"env_keys={'FOUND' if bundle.ready else 'MISSING'}")
    print(f"env_source={bundle.source or '-'}")
    for warning in report.warnings:
        print(f"warning={warning}")
    for error in report.errors:
        print(f"error={error}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(run_static_check())
