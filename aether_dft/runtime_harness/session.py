from __future__ import annotations

from pathlib import Path

from aether_dft.session_store import AetherSessionStore


class HarnessSessionStore:
    def __init__(self, base_dir: Path | None = None):
        self.store = AetherSessionStore(base_dir)

