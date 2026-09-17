"""Safe YAML loading and default -> YAML -> CLI precedence."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Mapping

import yaml
from dotenv import dotenv_values, find_dotenv

from vifinqa.config.models import ExperimentConfig


# ``src/vifinqa/config/loader.py`` -> repository root.
_REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_asset_path(path: str | Path) -> Path:
    """Resolve a repository-relative asset (prompt template) independently of the CWD.

    An absolute path is used as given. A relative path is tried against the current
    working directory first, then against the repository root, so the CLI keeps working
    when it is invoked from outside the project directory.
    """

    candidate = Path(path)
    if candidate.is_absolute():
        if not candidate.is_file():
            raise FileNotFoundError(f"Asset not found: {candidate}")
        return candidate
    tried = [Path.cwd() / candidate, _REPO_ROOT / candidate]
    for option in tried:
        if option.is_file():
            return option
    locations = ", ".join(str(option) for option in tried)
    raise FileNotFoundError(f"Asset {candidate} not found; tried: {locations}")


DEFAULTS: dict[str, Any] = {
    "run": {"name": "vifinqa-run"},
    "paths": {
        "data_root": "data/ocr_filter",
        "questions_dir": "data/questions",
        "cache_dir": ".cache/vifinqa",
        "runs_dir": "runs",
        "company_meta_path": "data/file_filter.csv",
    },
    "evaluation": {
        "ks": [1, 3, 5, 10, 20, 50, 100],
        "answer_abs_tolerance": 0.01,
    },
}


def _deep_merge(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _set_dotted(target: dict[str, Any], dotted: str, value: Any) -> None:
    cursor = target
    parts = dotted.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def load_config(
    path: Path,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> tuple[ExperimentConfig, dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    resolved = _deep_merge(copy.deepcopy(DEFAULTS), loaded)
    for key, value in (overrides or {}).items():
        if value is not None:
            _set_dotted(resolved, key, value)
    model = ExperimentConfig.model_validate(resolved)
    return model, model.model_dump(mode="json", exclude_none=True)


def env_value(config: Mapping[str, Any], field: str, *, required: bool = False) -> str | None:
    """Resolve `<field>_env` without placing a secret in serialized config."""

    env_name = config.get(f"{field}_env")
    if not isinstance(env_name, str) or not env_name:
        if required:
            raise ValueError(f"Missing {field}_env")
        return None
    value = os.getenv(env_name)
    if value is None:
        dotenv_path = find_dotenv(usecwd=True)
        dotenv = dotenv_values(dotenv_path) if dotenv_path else {}
        dotenv_value = dotenv.get(env_name)
        value = str(dotenv_value) if dotenv_value is not None else None
    if required and not value:
        raise ValueError(f"Environment variable {env_name} is not set")
    return value
