"""Import shim for the numbered private target preparation script."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_PATH = Path(__file__).with_name("20_prepare_private_code_targets.py")
_SPEC = importlib.util.spec_from_file_location("prepare_private_code_targets", _PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load {_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

build_target_cohort = _MODULE.build_target_cohort
