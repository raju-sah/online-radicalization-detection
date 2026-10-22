"""Configuration and environment handling for radpath.

The YAML file in ``configs/`` is the single source of research parameters.
Secrets (only the hashing salt for now) come from ``.env`` at the repository
root and are never written into configuration files or data outputs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "configs" / "default.yaml"
PLACEHOLDER_SALT: str = "change-me"

_REQUIRED_KEYS = (
    "paths",
    "target_subreddits",
    "gateway_subreddits",
    "feature_window",
    "pre_entry_days",
    "followup_days",
    "deep_min_contributions",
    "deep_min_active_weeks",
    "split",
    "seed",
)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the YAML config, also loading ``.env`` if present.

    Args:
        path: YAML path. Relative paths are resolved against the repository
            root. Defaults to ``configs/default.yaml``.

    Returns:
        Validated configuration dictionary.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ValueError: If required keys or parameter values are invalid.
    """
    config_path = _resolve_project_path(path or DEFAULT_CONFIG_PATH)
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    """Validate required keys and invariants in-place.

    Raises:
        ValueError: If a required key is missing or a value is out of range.
    """
    missing = [key for key in _REQUIRED_KEYS if key not in cfg]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    window = cfg["feature_window"]
    if not isinstance(window, dict) or not {"first_n", "days"} <= set(window):
        raise ValueError("feature_window must define first_n and days")
    if int(window["first_n"]) < 1:
        raise ValueError("feature_window.first_n must be >= 1")
    if int(window["days"]) < 1:
        raise ValueError("feature_window.days must be >= 1")
    if int(cfg["pre_entry_days"]) < 1:
        raise ValueError("pre_entry_days must be >= 1")
    if int(cfg["followup_days"]) < 1:
        raise ValueError("followup_days must be >= 1")
    if int(cfg["deep_min_contributions"]) < 1:
        raise ValueError("deep_min_contributions must be >= 1")
    if int(cfg["deep_min_active_weeks"]) < 1:
        raise ValueError("deep_min_active_weeks must be >= 1")

    fractions = cfg["split"]
    if not isinstance(fractions, dict) or set(fractions) != {"train", "val", "test"}:
        raise ValueError("split must define exactly train, val and test")
    total = sum(float(value) for value in fractions.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"split fractions must sum to 1.0 (got {total})")
    if any(float(value) <= 0 for value in fractions.values()):
        raise ValueError("split fractions must be positive")

    for key in ("target_subreddits", "gateway_subreddits"):
        values = cfg[key]
        if not isinstance(values, list) or not values:
            raise ValueError(f"{key} must be a non-empty list")
        if not all(isinstance(value, str) and value.strip() for value in values):
            raise ValueError(f"{key} must contain non-empty strings")
    controls = cfg.get("control_subreddits", [])
    if not isinstance(controls, list):
        raise ValueError("control_subreddits must be a list")

    overrides = cfg.get("coverage_overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("coverage_overrides must be a mapping")
    for name, override in overrides.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("coverage_overrides keys must be subreddit names")
        if not isinstance(override, dict):
            raise ValueError(f"coverage_overrides.{name} must be a mapping")
        if set(override) - {"start_utc", "end_utc"}:
            raise ValueError(
                f"coverage_overrides.{name} may only define start_utc and end_utc"
            )

    near_days = cfg.get("coverage", {}).get("near_start_days", 30)
    if int(near_days) < 0:
        raise ValueError("coverage.near_start_days must be >= 0")

    robustness = cfg.get("robustness", {})
    if robustness:
        for key in ("min_contributions", "min_active_weeks", "followup_days"):
            values = robustness.get(key, [])
            if not isinstance(values, list) or not values:
                raise ValueError(f"robustness.{key} must be a non-empty list")
            if any(int(value) < 1 for value in values):
                raise ValueError(f"robustness.{key} values must be >= 1")
        if not isinstance(robustness.get("comments_only", True), bool):
            raise ValueError("robustness.comments_only must be a boolean")


def _resolve_project_path(path: str | Path) -> Path:
    """Resolve ``path`` against the repository root when relative."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def resolve_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    """Return configured paths as absolute :class:`Path` objects."""
    paths = cfg["paths"]
    resolved = {name: _resolve_project_path(value) for name, value in paths.items()}
    for name in ("raw", "interim", "processed", "reports"):
        if name not in resolved:
            raise ValueError(f"paths.{name} is missing from config")
    return resolved


def ensure_data_dirs(cfg: dict[str, Any]) -> dict[str, Path]:
    """Create configured data/output directories if needed."""
    paths = resolve_paths(cfg)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def configured_subreddits(cfg: dict[str, Any]) -> set[str]:
    """Return lower-cased target, control, gateway and extra subreddits."""
    names: list[str] = []
    for key in ("target_subreddits", "control_subreddits", "gateway_subreddits"):
        names.extend(cfg.get(key, []))
    names.extend(cfg.get("ingest", {}).get("extra_subreddits", []))
    return {name.strip().lower() for name in names if name and name.strip()}


def get_salt(required: bool = True) -> str:
    """Read the username hashing salt from the environment / ``.env``.

    Args:
        required: If true, raise when the salt is absent or a placeholder.

    Returns:
        Salt string, or ``""`` when absent and ``required`` is false.

    Raises:
        ValueError: If the salt is missing, too short, or still the example
            placeholder.
    """
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    salt = os.environ.get("RADPATH_SALT", "").strip()
    if not salt:
        if required:
            raise ValueError(
                "RADPATH_SALT is not set. Copy .env.example to .env and use a "
                "long random value, e.g. `openssl rand -hex 32`."
            )
        return ""
    if salt == PLACEHOLDER_SALT:
        if required:
            raise ValueError("RADPATH_SALT is still the example placeholder; set a real salt.")
        return ""
    if len(salt) < 8:
        if required:
            raise ValueError("RADPATH_SALT must be at least 8 characters.")
        return ""
    return salt
