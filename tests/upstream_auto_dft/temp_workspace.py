from __future__ import annotations

import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4


TESTS_ROOT = Path(__file__).resolve().parent
WORKSPACE_TMP_ROOT = TESTS_ROOT / ".tmp_workspace"


@contextmanager
def workspace_tempdir(prefix: str = "auto_dft_"):
    WORKSPACE_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{prefix}{uuid4().hex[:6]}_", dir=str(WORKSPACE_TMP_ROOT)))
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
