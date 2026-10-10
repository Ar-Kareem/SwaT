from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

VARIABLES = {
    "DATASETS_ROOT": "MULTIKV_DATASETS_ROOT",
    "MODELS_ROOT": "MULTIKV_MODELS_ROOT",
    "SCRATCH_ROOT": "MULTIKV_SCRATCH_ROOT",
    "PYTHON": "MULTIKV_PYTHON",
    "VLLM_PYTHON": "MULTIKV_VLLM_PYTHON",
    "TEXMF_ROOT": "MULTIKV_TEXMF_ROOT",
}


def env_path(variable: str) -> Path:
    value = os.environ.get(variable)
    if not value:
        raise RuntimeError(f"{variable} is not set; define it in {REPO_ROOT / '.env'} (see .env.example)")
    return Path(value).expanduser()


def __getattr__(name: str) -> Path:
    if name in VARIABLES:
        return env_path(VARIABLES[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def portable(path: str | Path) -> str:
    path = Path(path).resolve()
    for variable in ("MULTIKV_MODELS_ROOT", "MULTIKV_DATASETS_ROOT"):
        try:
            return f"${variable}/{path.relative_to(env_path(variable).resolve())}"
        except ValueError:
            continue
    return str(path)
