from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_ROOT = PROJECT_ROOT / ".aether"
PROJECTS_DIR = DATA_ROOT / "projects"
KNOWLEDGE_BASE_DIR = DATA_ROOT / "knowledge_base"
RUNTIME_DIR = DATA_ROOT / "runtime"
RUNS_DIR = DATA_ROOT / "runs"
CACHE_DIR = DATA_ROOT / "cache"


def ensure_runtime_dir(*parts: str) -> Path:
    path = RUNTIME_DIR.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_project_dirs() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
