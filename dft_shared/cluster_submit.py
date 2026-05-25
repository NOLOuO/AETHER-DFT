"""统一集群提交模块：一套逻辑，支持 relax / ts_dimer / 自定义任务类型。

用法::

    from dft_shared.cluster_submit import prepare_and_submit, TaskSubmission

    # 结构优化
    submit = TaskSubmission(
        task_name="C7H14-re",
        task_type="relax",
        poscar_path=Path("POSCAR C7H14-re.txt"),
        incar_template=Path("templates/INCAR"),
        kpoints_template=Path("templates/KPOINTS"),
    )

    # TS Dimer（自动附带 MODECAR）
    submit = TaskSubmission(
        task_name="TS1",
        task_type="ts_dimer",
        poscar_path=Path("POSCAR TS.txt"),
        incar_template=Path("templates/INCAR_DIMER"),
        kpoints_template=Path("templates/KPOINTS"),
        extra_files={"MODECAR": Path("MODECAR")},
    )

    result = prepare_and_submit(submit)
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from .remote_backend import RemoteBackend, RunResult, get_backend, shell_quote
from .remote_config import RemoteConfig


LOGGER = logging.getLogger(__name__)
TASK_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\u4e00-\u9fff ]+$")
DEFAULT_REMOTE_RETRY_ATTEMPTS = 3


@dataclass
class TaskSubmission:
    """一次提交所需的全部信息。"""

    task_name: str
    task_type: str = "relax"                    # relax | ts_dimer | single_point | ...
    poscar_path: Path = Path("POSCAR")
    incar_template: Path | None = None          # None = 用远端已有的
    kpoints_template: Path | None = None
    extra_files: dict[str, Path] = field(default_factory=dict)  # 如 {"MODECAR": Path("MODECAR")}
    magmom_line: str | None = None              # 自动生成的 MAGMOM 行

    # qvasp 参数
    queue: str = "c-node"
    cores: int = 32
    vasp_version: str = "std"
    nodes: int = 1

    # 远程目录
    remote_preset: str = "relax"                # remote_dirs 里的 key


@dataclass
class PrepareResult:
    remote_job_dir: str
    review_dir: Path
    review_files: list[str]
    magmom_line: str | None
    stdout: str


@dataclass
class SubmitResult:
    remote_job_dir: str
    job_id: str
    stdout: str
    uploaded_files: list[str] = field(default_factory=list)


def prepare_task(
    sub: TaskSubmission,
    *,
    local_task_dir: Path,
    backend: RemoteBackend | None = None,
    cfg: RemoteConfig | None = None,
) -> PrepareResult:
    """准备任务：上传文件 → 远端建目录 → MAGMOM patch → 拉回 review。"""
    _validate_task_submission(sub)
    if cfg is None:
        cfg = RemoteConfig.load()
    if backend is None:
        backend = get_backend(cfg)

    remote_base = cfg.get_remote_dir(sub.remote_preset)
    remote_job_dir = str(PurePosixPath(remote_base) / sub.task_name)
    review_dir = local_task_dir / "submit_review"
    review_dir.mkdir(parents=True, exist_ok=True)

    # 1. 建目录
    timeout = cfg.remote_timeout_seconds
    _retry_runresult(lambda: backend.mkdir(remote_job_dir, timeout=timeout), "创建远端目录失败")

    # 2. 上传 POSCAR
    _retry_runresult(lambda: backend.upload(sub.poscar_path, remote_job_dir + "/POSCAR", timeout=timeout), "上传 POSCAR 失败")

    # 3. 上传 INCAR 模板
    if sub.incar_template and sub.incar_template.exists():
        _retry_runresult(lambda: backend.upload(sub.incar_template, remote_job_dir + "/INCAR", timeout=timeout), "上传 INCAR 失败")

    # 4. 上传 KPOINTS 模板
    if sub.kpoints_template and sub.kpoints_template.exists():
        _retry_runresult(lambda: backend.upload(sub.kpoints_template, remote_job_dir + "/KPOINTS", timeout=timeout), "上传 KPOINTS 失败")

    # 5. 上传附加文件（如 MODECAR）
    for remote_name, local_path in sub.extra_files.items():
        if local_path.exists():
            _retry_runresult(lambda local_path=local_path, remote_name=remote_name: backend.upload(local_path, remote_job_dir + f"/{remote_name}", timeout=timeout), f"上传 {remote_name} 失败")

    # 6. MAGMOM patch
    if sub.magmom_line:
        patch_cmd = (
            f"cd {shell_quote(remote_job_dir)} && "
            "rm -f INCAR.tmp && "
            "grep -Ev '^[[:space:]]*MAGMOM[[:space:]]*=' INCAR > INCAR.tmp 2>/dev/null || true && "
            f"printf '%s\\n' {shell_quote('MAGMOM = ' + sub.magmom_line)} >> INCAR.tmp && "
            "mv INCAR.tmp INCAR"
        )
        _retry_runresult(lambda: backend.run(patch_cmd, timeout=timeout), "MAGMOM patch 失败")

    # 7. 去 Windows 换行符
    normalize_cmd = (
        f"cd {shell_quote(remote_job_dir)} && "
        "(command -v dos2unix >/dev/null 2>&1 && dos2unix -q INCAR KPOINTS) "
        "|| sed -i 's/\\r$//' INCAR KPOINTS 2>/dev/null "
        "|| true"
    )
    _retry_runresult(lambda: backend.run(normalize_cmd, timeout=timeout), "规范化 INCAR/KPOINTS 换行符失败")

    # 8. 拉回 review 文件
    review_files = ["INCAR", "KPOINTS", "POSCAR"]
    for name in review_files:
        _retry_runresult(lambda name=name: backend.download(remote_job_dir + f"/{name}", review_dir, timeout=timeout), f"拉回 {name} 失败")

    # 也拉回附加文件供确认
    for remote_name in sub.extra_files:
        _retry_runresult(lambda remote_name=remote_name: backend.download(remote_job_dir + f"/{remote_name}", review_dir, timeout=timeout), f"拉回 {remote_name} 失败")
        review_files.append(remote_name)

    stdout_parts = [f"任务类型: {sub.task_type}", f"远端目录: {remote_job_dir}"]
    if sub.magmom_line:
        stdout_parts.append(f"MAGMOM: {sub.magmom_line}")
    if sub.extra_files:
        stdout_parts.append(f"附加文件: {', '.join(sub.extra_files.keys())}")

    result = PrepareResult(
        remote_job_dir=remote_job_dir,
        review_dir=review_dir,
        review_files=review_files,
        magmom_line=sub.magmom_line,
        stdout="\n".join(stdout_parts),
    )
    _write_submission_audit(
        local_task_dir,
        {
            "phase": "prepare",
            "task_name": sub.task_name,
            "task_type": sub.task_type,
            "remote_job_dir": remote_job_dir,
            "review_dir": str(review_dir),
            "review_files": review_files,
            "magmom_line": sub.magmom_line,
            "extra_files": sorted(sub.extra_files.keys()),
        },
    )
    return result


def confirm_submit(
    sub: TaskSubmission,
    *,
    review_dir: Path,
    backend: RemoteBackend | None = None,
    cfg: RemoteConfig | None = None,
) -> SubmitResult:
    """确认提交：上传 review 文件 → qvasp 提交。"""
    _validate_task_submission(sub)
    if cfg is None:
        cfg = RemoteConfig.load()
    if backend is None:
        backend = get_backend(cfg)
    timeout = cfg.remote_timeout_seconds

    remote_base = cfg.get_remote_dir(sub.remote_preset)
    remote_job_dir = str(PurePosixPath(remote_base) / sub.task_name)
    _validate_local_confirm_requirements(review_dir, required_extra=list(sub.extra_files.keys()))

    uploaded_files: list[str] = []
    # 上传确认后的文件
    for name in ["INCAR", "KPOINTS", "POSCAR"]:
        local_file = review_dir / name
        if local_file.exists():
            _retry_runresult(lambda local_file=local_file: backend.upload(local_file, remote_job_dir + "/", timeout=timeout), f"上传 {name} 失败")
            uploaded_files.append(name)
    for remote_name in sub.extra_files:
        local_file = review_dir / remote_name
        if local_file.exists():
            _retry_runresult(lambda local_file=local_file: backend.upload(local_file, remote_job_dir + "/", timeout=timeout), f"上传 {remote_name} 失败")
            uploaded_files.append(remote_name)

    # 构建 qvasp 命令
    qvasp_parts = [shell_quote(cfg.qvasp_path), "-n", str(int(sub.cores)), "-q", shell_quote(str(sub.queue)), "-v", shell_quote(str(sub.vasp_version)), "-N", str(int(sub.nodes))]
    qvasp_cmd = " ".join(qvasp_parts)

    result = _retry_runresult(lambda: backend.run(f"cd {shell_quote(remote_job_dir)} && {qvasp_cmd}", timeout=timeout), "qvasp 提交失败")

    stdout = result.stdout.strip()
    job_id = _parse_job_id(stdout)
    if not job_id:
        raise RuntimeError(f"提交完成但未解析到 job id:\n{stdout}")

    submit_result = SubmitResult(remote_job_dir=remote_job_dir, job_id=job_id, stdout=stdout, uploaded_files=uploaded_files)
    _write_submission_audit(
        review_dir.parent,
        {
            "phase": "submit",
            "task_name": sub.task_name,
            "task_type": sub.task_type,
            "remote_job_dir": remote_job_dir,
            "job_id": job_id,
            "uploaded_files": uploaded_files,
            "stdout": stdout,
        },
    )
    return submit_result


def prepare_and_submit(
    sub: TaskSubmission,
    *,
    local_task_dir: Path,
    auto_confirm: bool = False,
    backend: RemoteBackend | None = None,
    cfg: RemoteConfig | None = None,
) -> SubmitResult:
    """一步完成：prepare → (可选人工检查) → submit。"""
    if cfg is None:
        cfg = RemoteConfig.load()
    if backend is None:
        backend = get_backend(cfg)

    prep = prepare_task(sub, local_task_dir=local_task_dir, backend=backend, cfg=cfg)

    if not auto_confirm:
        print(f"\n请检查 {prep.review_dir} 中的文件，确认无误后按 Enter 继续提交...")
        input()

    return confirm_submit(sub, review_dir=prep.review_dir, backend=backend, cfg=cfg)


def _parse_job_id(text: str) -> str | None:
    patterns = [
        re.compile(r"Submitted batch job\s+(\d+)", re.IGNORECASE),
        re.compile(r"\bjob(?:_id)?\s*[:=]\s*(\d+)\b", re.IGNORECASE),
        re.compile(r"\bjob\s+(\d+)\s+(?:submitted|queued|started)\b", re.IGNORECASE),
    ]
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        for pattern in patterns:
            match = pattern.search(stripped)
            if match:
                return match.group(1)
        if re.fullmatch(r"\d{4,12}", stripped):
            LOGGER.warning("回退到纯数字行解析 job id：%s", stripped)
            return stripped
    return None


def _validate_task_submission(sub: TaskSubmission) -> None:
    task_name = sub.task_name.strip()
    if not task_name:
        raise ValueError("task_name 不能为空")
    if not TASK_NAME_PATTERN.fullmatch(task_name):
        raise ValueError(f"非法 task_name: {sub.task_name!r}")


def _validate_local_confirm_requirements(review_dir: Path, *, required_extra: list[str]) -> None:
    missing: list[str] = []
    for name in ["INCAR", "KPOINTS", "POSCAR", *required_extra]:
        if not (review_dir / name).exists():
            missing.append(str(review_dir / name))
    if missing:
        raise RuntimeError("待确认文件缺失: " + "；".join(missing))


def _retry_runresult(op, msg: str, *, attempts: int = DEFAULT_REMOTE_RETRY_ATTEMPTS) -> RunResult:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            result = op()
            result.check(msg)
            return result
        except Exception as exc:
            last_error = exc
            if attempt >= attempts:
                raise
            delay = 2 ** (attempt - 1)
            LOGGER.warning("%s，第 %d/%d 次失败，%.1fs 后重试: %s", msg, attempt, attempts, delay, exc)
            time.sleep(delay)
    raise RuntimeError(str(last_error) if last_error else msg)


def _write_submission_audit(local_task_dir: Path, payload: dict) -> None:
    audit_path = local_task_dir / "submission_audit.json"
    existing: list[dict] = []
    if audit_path.exists():
        try:
            existing = json.loads(audit_path.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                existing = [existing]
        except Exception:
            existing = []
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **payload,
    }
    existing.append(payload)
    audit_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
