from __future__ import annotations

"""Research workspace <-> cluster synchronization primitives."""

import re
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any

from dft_app.remote import SSHRemoteRunner

from .research_workspace import RESEARCH_ROOT, list_research_projects, resolve_research_project


class ResearchScopeError(ValueError):
    pass


def _safe_project(project: str | None) -> str | None:
    value = str(project or "").strip()
    if not value:
        return None
    resolved = resolve_research_project(value)
    if resolved:
        return resolved.slug
    raise ResearchScopeError(f"research 项目不存在或不在本地 research/ 下: {value}")


def _local_root(project: str | None) -> Path:
    slug = _safe_project(project)
    return RESEARCH_ROOT / slug if slug else RESEARCH_ROOT


def _remote_dir(runner: SSHRemoteRunner, project: str | None, explicit: str | None = None) -> str:
    if explicit:
        return explicit
    config = runner._load_config()
    base = f"/home/{config.user}/research"
    slug = _safe_project(project)
    return str(PurePosixPath(base) / slug) if slug else base


def _validate_rel(rel: str) -> PurePosixPath:
    path = PurePosixPath(rel)
    if path.is_absolute() or ".." in path.parts or not rel.strip():
        raise ValueError(f"不安全的相对路径: {rel}")
    return path


def research_workspace_diff(project: str | None = None, *, remote_research_dir: str | None = None) -> dict[str, Any]:
    try:
        runner = SSHRemoteRunner()
        local = _local_root(project)
        remote = _remote_dir(runner, project, remote_research_dir)
    except ResearchScopeError as exc:
        return {"status": "error", "message": str(exc), "available_projects": list_research_projects()}
    result = runner.research_status(local, remote_research_dir=remote)
    return {"status": result.status, "message": result.message, "project": _safe_project(project) or "", "local_root": str(local), "details": result.details}


def research_workspace_sync_to_cluster(
    project: str | None = None,
    *,
    remote_research_dir: str | None = None,
    apply: bool = False,
) -> dict[str, Any]:
    try:
        runner = SSHRemoteRunner()
        local = _local_root(project)
        remote = _remote_dir(runner, project, remote_research_dir)
    except ResearchScopeError as exc:
        return {"status": "error", "message": str(exc), "available_projects": list_research_projects()}
    result = runner.sync_research_to_remote(local, remote_research_dir=remote, dry_run=not apply)
    return {"status": result.status, "message": result.message, "project": _safe_project(project) or "", "local_root": str(local), "details": result.details}


def research_workspace_sync_from_cluster(
    project: str | None = None,
    *,
    remote_research_dir: str | None = None,
    apply: bool = False,
) -> dict[str, Any]:
    """Pull remote-only/differing files from cluster research into local research.

    Dry-run is default.  When applying, differing local files are backed up under
    ``.aether_pull_backups`` before overwrite.
    """

    try:
        runner = SSHRemoteRunner()
        local = _local_root(project)
        remote = _remote_dir(runner, project, remote_research_dir)
    except ResearchScopeError as exc:
        return {"status": "error", "message": str(exc), "available_projects": list_research_projects()}
    status = runner.research_status(local, remote_research_dir=remote)
    if status.status != "ok":
        return {"status": status.status, "message": status.message, "project": _safe_project(project) or "", "details": status.details}
    details = dict(status.details)
    to_pull = sorted(set(details.get("missing_local") or []) | set(details.get("differing") or []))
    if not to_pull:
        return {
            "status": "ok",
            "message": "无需反向同步：本地 research 已包含集群 research 当前版本。",
            "project": _safe_project(project) or "",
            "local_root": str(local),
            "details": {**details, "dry_run": not apply, "pulled": []},
        }
    if not apply:
        return {
            "status": "planned",
            "message": f"需要从集群拉取/覆盖 {len(to_pull)} 个 research 文件；apply=false 未修改本地。",
            "project": _safe_project(project) or "",
            "local_root": str(local),
            "details": {**details, "dry_run": True, "would_pull": to_pull},
        }

    config = runner._load_config()
    backend = runner._select_backend(config)
    tools_error = runner._ensure_local_tools(config, backend)
    if tools_error:
        return {"status": "blocked", "message": tools_error, "details": {"backend": backend}}
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    remote_tmp = str(PurePosixPath(remote).parent / ".aether_research_sync" / f"pull-{timestamp}.tar.gz")
    rel_args = " ".join(runner._quote(str(_validate_rel(rel))) for rel in to_pull)
    mkdir = runner._run_remote_command(
        config,
        f"mkdir -p {runner._quote(str(PurePosixPath(remote_tmp).parent))}",
        timeout=30,
        backend=backend,
    )
    if mkdir.returncode != 0:
        return {"status": "failed", "message": mkdir.stderr or mkdir.stdout or "远端临时目录创建失败。", "details": {"backend": backend}}
    pack = runner._run_remote_command(
        config,
        f"cd {runner._quote(remote)} && tar -czf {runner._quote(remote_tmp)} {rel_args}",
        timeout=120,
        backend=backend,
    )
    if pack.returncode != 0:
        return {"status": "failed", "message": pack.stderr or pack.stdout or "远端 research 打包失败。", "details": {"backend": backend, "remote_research_dir": remote}}
    pulled: list[str] = []
    backup_dir = local / ".aether_pull_backups" / timestamp
    try:
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "research-pull.tar.gz"
            runner._download_from_remote(config, remote_tmp, archive, timeout=180, backend=backend)
            local.mkdir(parents=True, exist_ok=True)
            for rel in to_pull:
                rel_path = Path(*_validate_rel(rel).parts)
                target = local / rel_path
                if target.exists():
                    backup = backup_dir / rel_path
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    backup.write_bytes(target.read_bytes())
            with tarfile.open(archive, "r:gz") as tar:
                safe_members = []
                for member in tar.getmembers():
                    _validate_rel(member.name)
                    if not (member.isfile() or member.isdir()):
                        raise ValueError(f"research 同步包含不安全成员类型: {member.name}")
                    safe_members.append(member)
                tar.extractall(local, members=safe_members)  # noqa: S202 - members validated above.
            pulled = to_pull
    except Exception as exc:
        return {"status": "failed", "message": f"本地反向同步失败: {exc}", "details": {"backend": backend, "pulled": pulled}}
    finally:
        runner._run_remote_command(config, f"rm -f {runner._quote(remote_tmp)}", timeout=30, backend=backend)
    return {
        "status": "synced",
        "message": f"已从集群 {remote} 拉取/覆盖 {len(pulled)} 个 research 文件。",
        "project": _safe_project(project) or "",
        "local_root": str(local),
        "details": {**details, "dry_run": False, "pulled": pulled, "backup_dir": str(backup_dir) if backup_dir.exists() else None},
    }


def research_workspace_pull_logs(project: str | None = None, *, run_id: str | None = None) -> dict[str, Any]:
    """Pull cluster outputs for a recorded run."""

    try:
        from dft_app.storage import RecordStore
    except Exception as exc:
        return {"status": "error", "message": str(exc)}
    store = RecordStore(Path.cwd())
    try:
        if run_id:
            run_root = store.resolve_run_root(run_id=run_id)
            record = store.load_run_record(run_root)
        else:
            candidates = store.list_runs(limit=50)
            if project:
                candidates = [rec for rec in candidates if str(rec.get("task_id") or "").startswith(str(project))]
            if not candidates:
                return {"status": "needs_run", "message": "没有找到可拉取的本地 run 记录。"}
            record = store.load_run_record(Path(candidates[0]["run_root"]))
    except Exception as exc:
        return {"status": "needs_run", "message": str(exc)}
    result = SSHRemoteRunner().fetch_outputs(record)
    store.save_run_record(record)
    return {"status": result.status, "message": result.message, "project": project or "", "run_id": record.run_id, "details": result.details}


def research_learning_capture(project: str, title: str, content: str, tags: list[str] | None = None) -> dict[str, Any]:
    paths = resolve_research_project(project)
    if paths is None:
        return {"status": "error", "message": f"research 项目不存在: {project}", "available_projects": list_research_projects()}
    clean_title = re.sub(r"[^\w\u4e00-\u9fff.-]+", "-", str(title or "").strip()).strip("-") or "untitled"
    learning_dir = paths.root / "Learning"
    learning_dir.mkdir(parents=True, exist_ok=True)
    path = learning_dir / f"{clean_title}.md"
    tag_line = " ".join(f"#{str(tag).strip()}" for tag in (tags or []) if str(tag).strip())
    text = "\n".join(
        [
            f"# {title.strip() or clean_title}",
            "",
            f"- Captured: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"- Tags: {tag_line or 'none'}",
            "",
            str(content or "").strip(),
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")
    return {"status": "ok", "project": paths.slug, "learning_path": str(path), "tags": tags or []}
