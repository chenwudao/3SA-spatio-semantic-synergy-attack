"""3SA research scaffold."""

from pathlib import Path
import os

_workspace_root = Path(__file__).resolve().parents[2]
_workspace_tmp = _workspace_root / ".tmp"
_workspace_tmp.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TMPDIR", str(_workspace_tmp))
os.environ.setdefault("TEMP", str(_workspace_tmp))
os.environ.setdefault("TMP", str(_workspace_tmp))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(_workspace_tmp / "torchinductor"))

from .config import AttackConfig, DefenseConfig, ExperimentConfig

__all__ = ["AttackConfig", "DefenseConfig", "ExperimentConfig"]
