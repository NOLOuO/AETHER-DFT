from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

from dft_app.llm.provider_presets import build_provider_model_config

from .model_catalog import format_model_table, resolve_effective_model_id, split_model_id
from .paths import CONFIG_DIR, PROJECT_ROOT, ensure_runtime_dir
from .prompt_engine import build_prompt_packet, render_prompt_packet_markdown
from .project_state import project_paths, read_project_context
from .permissions import get_permission_mode, permission_mode_label
from .session_store import AetherSessionStore
from .tool_registry import list_registered_tools


def build_context_payload(*, project: str | None = None) -> dict[str, Any]:
    model_id = resolve_effective_model_id()
    provider_id, model_name = split_model_id(model_id)
    model_config = build_provider_model_config(provider_id, model_name)
    prompt_packet = build_prompt_packet(project=project)
    session_store = AetherSessionStore()
    payload: dict[str, Any] = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "workspace": str(PROJECT_ROOT),
        "project": project,
        "model": {
            "model_id": model_id,
            "provider": provider_id,
            "model": model_name,
            "api_model": model_config.get("model"),
            "base_url": model_config.get("base_url"),
            "api_key_env": model_config.get("api_key_env"),
            "base_url_env": model_config.get("base_url_env"),
            "protocol": "openai-compatible",
        },
        "permission": {
            "mode": get_permission_mode(),
            "label": permission_mode_label(),
        },
        "prompt": prompt_packet["prompt"],
        "entrypoints": {
            "mainline": "aether-dft mainline [\"继续当前科研任务\"]",
            "chat": "aether-dft chat [--project <slug>] [--model provider:model]",
            "session": "aether-dft session list/show/resume",
            "tools": "aether-dft tools list/run",
            "project": "aether-dft project init/list/show/progress",
            "dft_mainline": "aether-dft run ... 或 aether-dft dft ...",
            "structure_io": "aether-dft structure convert <input> <output>",
            "explain": "aether-dft explain --run-root <path>",
        },
        "files": {
            "system_prompt": str(CONFIG_DIR / "system_prompt.md"),
            "permissions": str(CONFIG_DIR / "permissions.json"),
            "model_runtime": str(CONFIG_DIR / "model_runtime.json"),
            "project_state_md": str(project_paths(project).state_md) if project else "",
        },
    }
    payload["prompt_packet"] = prompt_packet
    payload["sessions"] = [item.to_dict() for item in session_store.list_sessions(project=project, limit=5)]
    payload["tools"] = list_registered_tools()
    latest_session_id = session_store.latest_session_id(project=project)
    if latest_session_id:
        payload["latest_session_id"] = latest_session_id
        try:
            payload["session_context"] = session_store.build_session_context(latest_session_id)
        except Exception:
            payload["session_context"] = ""
    if project:
        payload["project_context"] = read_project_context(project)
    return payload


def render_context_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# AETHER-DFT Context Snapshot",
        "",
        f"- Created: {payload['created_at']}",
        f"- Workspace: `{payload['workspace']}`",
        f"- Project: `{payload.get('project') or 'none'}`",
        "",
        "## Effective Model",
        "",
        "```json",
        json.dumps(payload["model"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Permission Mode",
        "",
        "```json",
        json.dumps(payload["permission"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Prompt Packet",
        "",
        render_prompt_packet_markdown(payload["prompt_packet"]),
        "",
        "## Model Catalog",
        "",
        "```text",
        format_model_table(),
        "```",
        "",
        "## Entrypoints",
        "",
    ]
    for name, command in payload["entrypoints"].items():
        lines.append(f"- **{name}**: `{command}`")
    lines.extend([
        "",
        "## Harness Tools",
        "",
    ])
    for tool in payload.get("tools", []):
        lines.append(f"- `{tool['name']}` — {tool['description']}")
    lines.extend([
        "",
        "## Recent Sessions",
        "",
    ])
    sessions = payload.get("sessions") or []
    if sessions:
        for session in sessions:
            lines.append(f"- `{session['session_id']}` turns={session['turn_count']} project={session.get('project') or 'none'}")
    else:
        lines.append("- none")
    lines.extend([
        "",
        "## System Prompt File",
        "",
        (CONFIG_DIR / "system_prompt.md").read_text(encoding="utf-8") if (CONFIG_DIR / "system_prompt.md").exists() else "",
        "",
    ])
    project_context = str(payload.get("project_context") or "").strip()
    if project_context:
        lines.extend(["## Project Context", "", project_context, ""])
    session_context = str(payload.get("session_context") or "").strip()
    if session_context:
        lines.extend(["## Latest Session Context", "", session_context, ""])
    return "\n".join(lines).rstrip() + "\n"


def write_context_snapshot(*, project: str | None = None) -> Path:
    payload = build_context_payload(project=project)
    context_dir = ensure_runtime_dir("context")
    suffix = f"-{project}" if project else ""
    path = context_dir / f"context{suffix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
    path.write_text(render_context_markdown(payload), encoding="utf-8")
    return path
