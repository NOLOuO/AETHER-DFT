from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dft_app.llm.key_store import load_api_keys
from dft_app.llm.provider_presets import build_provider_model_config

from .context_digests import (
    build_cluster_runtime_digest,
    build_job_watch_digest,
    build_research_workspace_digest,
    build_relevant_priors_digest,
)
from .model_catalog import load_model_catalog, resolve_effective_model_id, split_model_id
from .paths import PROJECT_ROOT
from .permissions import get_permission_mode, permission_mode_label
from .project_state import project_paths, read_project_context_digest
from .research_workspace import read_research_onboarding_context, resolve_research_project
from .runtime_harness.tool_registry import ToolRegistry
from .session_store import AetherSessionStore


@dataclass(frozen=True)
class PreloadSummary:
    payload: dict[str, Any]

    @property
    def status(self) -> str:
        return str(self.payload.get("status") or "ok")


def _key_configured(provider_id: str, config: dict[str, Any]) -> bool:
    env_name = str(config.get("api_key_env") or "")
    try:
        keys = load_api_keys(PROJECT_ROOT)
    except Exception:
        keys = {}
    return bool(keys.get(provider_id) or keys.get(env_name) or keys.get(env_name.upper()))


def build_preload_summary(*, project: str | None = None, probe_cluster: bool = False) -> PreloadSummary:
    """Collect the read-only context AETHER injects before model turns.

    This is intentionally cheap and mostly local. Live SSH probing is opt-in via
    probe_cluster so starting a chat does not unexpectedly touch the cluster.
    """

    model_id = resolve_effective_model_id()
    provider_id, model_name = split_model_id(model_id)
    model_config = build_provider_model_config(provider_id, model_name)
    session_store = AetherSessionStore()
    latest_session_id = session_store.latest_session_id(project=project)
    session_context = ""
    if latest_session_id:
        try:
            session_context = session_store.build_session_context(latest_session_id)
        except Exception:
            session_context = ""
    onboarding = read_research_onboarding_context(project, max_chars=1) if project else {
        "status": "not_requested",
        "project": project,
        "project_found": False,
        "files_read": [],
        "available_projects": [],
        "research_root": str(PROJECT_ROOT / "research"),
    }
    # Re-read without tiny truncation after collecting files/status; this keeps the
    # preload payload small while still reporting whether context exists.
    if project:
        onboarding = read_research_onboarding_context(project, max_chars=6000)
    research_paths = resolve_research_project(project)
    project_state_digest = read_project_context_digest(project) if project else ""
    research_digest = build_research_workspace_digest(project=project) if project else ""
    cluster_digest = build_cluster_runtime_digest(project=project)
    job_watch_digest = build_job_watch_digest(project=project)
    priors_digest = build_relevant_priors_digest(project=project, query=(session_context or project or ""))
    registry = ToolRegistry()
    try:
        discussion_tools = registry.openai_tool_schemas(interaction_mode="discussion")
        execution_tools = registry.openai_tool_schemas(interaction_mode="execution")
    except TypeError:
        discussion_tools = execution_tools = registry.openai_tool_schemas()

    cluster_probe: dict[str, Any] | None = None
    if probe_cluster:
        try:
            from dft_app.remote import SSHRemoteRunner

            result = SSHRemoteRunner().probe()
            cluster_probe = {"status": result.status, "message": result.message, "details": result.details}
        except Exception as exc:
            cluster_probe = {"status": "error", "message": str(exc), "details": {}}

    project_state_path = str(project_paths(project).state_md) if project else ""
    payload = {
        "status": "ok",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "workspace": str(PROJECT_ROOT),
        "model": {
            "model_id": model_id,
            "provider": provider_id,
            "model": model_name,
            "api_model": model_config.get("model"),
            "base_url": model_config.get("base_url"),
            "api_key_configured": _key_configured(provider_id, model_config),
            "context_window": model_config.get("context_window"),
            "available_model_count": len(load_model_catalog(PROJECT_ROOT)),
        },
        "permission": {"mode": get_permission_mode(), "label": permission_mode_label()},
        "project": {
            "slug": project,
            "project_state_path": project_state_path,
            "project_state_digest_chars": len(project_state_digest),
            "research_project_found": research_paths is not None,
            "research_project": research_paths.to_dict() if research_paths else None,
            "research_onboarding_status": onboarding.get("status"),
            "research_files_read": onboarding.get("files_read") or [],
            "available_research_projects": onboarding.get("available_projects") or [],
        },
        "session": {
            "latest_session_id": latest_session_id,
            "session_context_chars": len(session_context),
        },
        "prompt_preload": {
            "project_context_loaded": bool(project_state_digest),
            "research_onboarding_loaded": bool((onboarding.get("context") or "").strip()),
            "research_workspace_digest_loaded": bool(research_digest),
            "cluster_runtime_digest_loaded": bool(cluster_digest),
            "job_watch_digest_loaded": bool(job_watch_digest),
            "relevant_priors_loaded": bool(priors_digest),
            "discussion_tool_count": len(discussion_tools),
            "execution_tool_count": len(execution_tools),
        },
        "cluster": {
            "runtime_digest": cluster_digest,
            "job_watch_digest": job_watch_digest,
            "live_probe": cluster_probe,
        },
        "next_user_entrypoints": [
            f"aether-dft chat --project {project}" if project else "aether-dft chat --project <project>",
            f"aether-dft ask --project {project} \"继续当前课题，先判断下一步\"" if project else "aether-dft ask --project <project> \"继续当前课题\"",
            "aether-dft outcar find --limit 5",
            f"aether-dft outcar analyze --latest --project {project} --write-learning" if project else "aether-dft outcar analyze --latest --project <project> --write-learning",
        ],
    }
    return PreloadSummary(payload)


def format_preload_summary(summary: PreloadSummary) -> str:
    payload = summary.payload
    model = payload["model"]
    project = payload["project"]
    prompt = payload["prompt_preload"]
    lines = [
        "AETHER preload ready",
        f"- workspace: {payload['workspace']}",
        f"- model: {model['model_id']} api_key={'ok' if model['api_key_configured'] else 'missing'} ctx={model.get('context_window')}",
        f"- permission: {payload['permission']['mode']} / {payload['permission']['label']}",
        f"- project: {project.get('slug') or 'none'}",
    ]
    if project.get("slug"):
        lines.extend(
            [
                f"- project state chars: {project.get('project_state_digest_chars')}",
                f"- research project found: {project.get('research_project_found')}",
                f"- research files preloaded: {len(project.get('research_files_read') or [])}",
            ]
        )
        for file_path in (project.get("research_files_read") or [])[:6]:
            lines.append(f"  · {file_path}")
    lines.extend(
        [
            f"- latest session: {payload['session'].get('latest_session_id') or 'none'} ({payload['session'].get('session_context_chars')} chars)",
            "- prompt preload:",
            f"  · project_context={prompt['project_context_loaded']}",
            f"  · research_onboarding={prompt['research_onboarding_loaded']}",
            f"  · research_workspace_digest={prompt['research_workspace_digest_loaded']}",
            f"  · cluster_runtime_digest={prompt['cluster_runtime_digest_loaded']}",
            f"  · job_watch_digest={prompt['job_watch_digest_loaded']}",
            f"  · tools discussion/execution={prompt['discussion_tool_count']}/{prompt['execution_tool_count']}",
        ]
    )
    cluster_probe = payload.get("cluster", {}).get("live_probe")
    if cluster_probe:
        lines.append(f"- cluster probe: {cluster_probe.get('status')} — {cluster_probe.get('message')}")
    lines.append("- next:")
    for entry in payload.get("next_user_entrypoints") or []:
        lines.append(f"  · {entry}")
    return "\n".join(lines)
