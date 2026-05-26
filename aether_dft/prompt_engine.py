from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from dft_app.llm.provider_presets import build_provider_model_config

from .model_catalog import format_model_table, resolve_effective_model_id, split_model_id
from .permissions import get_permission_mode, permission_mode_label, permission_policy_text
from .paths import CONFIG_DIR, PROJECT_ROOT, ensure_runtime_dir
from .prompt_sections import PromptSectionCompiler
from .project_state import read_project_context, read_project_context_digest
from .context_digests import build_cluster_runtime_digest, build_research_workspace_digest, build_relevant_priors_digest
from .research_workspace import read_research_onboarding_context

PROMPT_ASSETS_DIR = Path(__file__).resolve().parent / "prompt_assets"


def _read_optional(path: Path, *, max_chars: int = 12000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n...[truncated]"
    return text


def load_base_system_prompt() -> str:
    path = PROMPT_ASSETS_DIR / "system_chemistry.md"
    if not path.exists():
        path = CONFIG_DIR / "system_prompt.md"
    text = _read_optional(path).strip()
    if text:
        return text
    return (
        "# AETHER-DFT system prompt\n\n"
        "你是 AETHER-DFT，一个专门服务于计算化学 / DFT 的对话式科研合伙人。\n"
    )


def _digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12] if text else ""


def load_architecture_live_doc_snapshot(*, max_chars: int = 2400) -> dict[str, str]:
    """Read the root canonical live doc as volatile guidance only.

    `智能体架构.md` is intentionally not a stable prompt prefix and never
    replaces project execution state. Keep the payload short enough to act as
    a per-turn runtime hint that humans can edit between turns.
    """

    path = PROJECT_ROOT / "智能体架构.md"
    text = _read_optional(path, max_chars=max_chars * 2).strip()
    if not text:
        return {
            "architecture_live_doc": "",
            "architecture_live_doc_digest_text": "",
            "architecture_live_doc_digest": "",
            "architecture_live_doc_path": str(path),
        }
    headings = [line for line in text.splitlines() if line.startswith("#") or "Step 1" in line or "Step 2" in line]
    digest_text = "\n".join(headings).strip()
    if len(digest_text) < 400:
        digest_text = text[:max_chars].strip()
    else:
        digest_text = digest_text[:max_chars].strip()
    return {
        "architecture_live_doc": text,
        "architecture_live_doc_digest_text": digest_text,
        "architecture_live_doc_digest": _digest(text),
        "architecture_live_doc_path": str(path),
    }


def _tool_discovery_digest(*, max_tools: int = 80) -> str:
    """Summarize visible tools without making the prompt a schema dump."""

    try:
        from aether_dft.runtime_harness.tool_registry import list_registered_tools
    except Exception:
        return ""
    rows: list[str] = []
    for item in list_registered_tools()[:max_tools]:
        name = str(item.get("name") or "").strip()
        description = str(item.get("description") or "").strip()
        if not name:
            continue
        rows.append(f"- `{name}` — {description[:120]}")
    return "\n".join(rows)


def _runtime_data(*, project: str | None = None, session_context: str | None = None) -> dict[str, Any]:
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    model_id = resolve_effective_model_id()
    provider_id, model_name = split_model_id(model_id)
    model_config = build_provider_model_config(provider_id, model_name)
    permission_mode = get_permission_mode()
    project_context = read_project_context_digest(project) if project else ""
    research_context = read_research_onboarding_context(project, max_chars=7000)["context"] if project else ""
    if research_context:
        project_context = (project_context + "\n\n## Research workspace onboarding\n" + research_context).strip()
    tool_digest = _tool_discovery_digest()
    architecture = load_architecture_live_doc_snapshot()
    cluster_runtime_digest = build_cluster_runtime_digest(project=project)
    research_workspace_digest = build_research_workspace_digest(project=project) if project else ""
    relevant_priors_digest = build_relevant_priors_digest(project=project, query=session_context or project or "")
    response_contract = "\n".join(
        [
            "当前回复应服务科研推进，而不是泛泛解释。",
            "用户问概念、机理或方法时，可以清楚解释原理、边界和 trade-off。",
            "没有真实计算输出时不要编造能量、job id、收敛状态或文献结论。",
            "如果当前阶段可执行，优先给出最短可执行下一步。",
            "用户用自然语言提出任务时，你应自行选择和调用工具；不要要求用户记工具名或 slash command。",
            "若当前项目上下文存在，先续接已有进度，再补充新动作，不要从头重讲通用流程。",
            "如果需要外部资料或事实核对，明确说明需要查证，不要把推测说成事实。",
            "当得到有价值结论时，优先把它写回项目进展或知识库，而不是只停留在口头总结。",
        ]
    )
    return {
        "created_at": created_at,
        "workspace": str(PROJECT_ROOT),
        "project": project or "",
        "model_id": model_id,
        "provider_id": provider_id,
        "model_name": model_name,
        "model_context_window": str(model_config.get("context_window") or ""),
        "permission_mode": permission_mode,
        "permission_label": permission_mode_label(permission_mode),
        "permission_policy": permission_policy_text(permission_mode),
        "project_context": project_context,
        "project_context_digest": _digest(project_context),
        **architecture,
        "session_context": (session_context or "").strip(),
        "session_context_digest": _digest(session_context or ""),
        "tool_discovery_digest": tool_digest,
        "tool_discovery_digest_hash": _digest(tool_digest),
        "cluster_runtime_digest": cluster_runtime_digest,
        "cluster_runtime_digest_hash": _digest(cluster_runtime_digest),
        "research_workspace_digest": research_workspace_digest,
        "research_workspace_digest_hash": _digest(research_workspace_digest),
        "relevant_priors_digest": relevant_priors_digest,
        "relevant_priors_digest_hash": _digest(relevant_priors_digest),
        "response_contract": response_contract,
        "response_contract_digest": _digest(response_contract),
    }


def compile_system_prompt(*, project: str | None = None, session_context: str | None = None) -> dict[str, Any]:
    runtime_data = _runtime_data(project=project, session_context=session_context)
    return PromptSectionCompiler().build(runtime_data, fallback=load_base_system_prompt())


def build_prompt_packet(*, project: str | None = None, session_context: str | None = None) -> dict[str, Any]:
    model_id = resolve_effective_model_id()
    provider_id, model_name = split_model_id(model_id)
    model_config = build_provider_model_config(provider_id, model_name)
    permission_mode = get_permission_mode()
    base_prompt = load_base_system_prompt()
    project_context = read_project_context(project) if project else ""
    research_onboarding = read_research_onboarding_context(project, max_chars=12000) if project else {"context": "", "files_read": []}
    if research_onboarding.get("context"):
        project_context = (project_context + "\n\n## Research workspace onboarding\n" + str(research_onboarding["context"])).strip()
    architecture_live_doc = load_architecture_live_doc_snapshot()
    compiled = compile_system_prompt(project=project, session_context=session_context)
    runtime_contract = [
        "你是一个领域专属的 agent harness，不是通用闲聊助手。",
        "优先把用户意图转成可执行的科研任务，而不是停留在抽象建议。",
        "涉及事实、运行状态、计算结果时，优先调用本地工具和项目状态，不要凭空编造。",
        "如果项目上下文存在，就把它视为当前科研任务的长期记忆来源。",
        "在研究场景下，优先形成“讨论 -> 结构 -> 任务 -> 执行 -> 解释 -> 回写”的最小闭环。",
        "输出以中文为主；化学术语、命令、路径、接口字段保留必要英文。",
    ]
    tool_policy = [
        "工具负责确定性提取、结构化、执行与回写。",
        "模型负责科研判断、任务分解和结果解释。",
        "用户无需显式调用工具；模型根据意图主动调用。",
        "未知事实必须查证，不能猜。",
        "结果必须能写回项目状态或可追踪的运行记录。",
        "当结论或经验值得复用时，优先调用知识沉淀工具；当需要真实远程执行时，优先调用远程/执行工具。",
    ]
    compiled_system_prompt = str(compiled["prompt"])
    compile_projection = compiled["compile_projection"]
    payload: dict[str, Any] = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "workspace": str(PROJECT_ROOT),
        "project": project,
        "model": {
            "model_id": model_id,
            "provider": provider_id,
            "model": model_name,
            "context_window": model_config.get("context_window"),
        },
        "permission": {
            "mode": permission_mode,
            "label": permission_mode_label(permission_mode),
            "policy": permission_policy_text(permission_mode),
        },
        "prompt": {
            "base_prompt_path": str(PROMPT_ASSETS_DIR / "system_chemistry.md"),
            "base_prompt": base_prompt,
            "runtime_contract": runtime_contract,
            "tool_policy": tool_policy,
            "compiled_system_prompt": compiled_system_prompt,
            "layers": compiled["layers"],
            "source_map": compiled["source_map"],
            "compile_projection": compile_projection,
            "stable_layer_names": compile_projection.get("stable_layer_names", []),
            "volatile_layer_names": compile_projection.get("volatile_layer_names", []),
            "stable_prefix_text": compiled["stable_prefix_text"],
            "volatile_suffix_text": compiled["volatile_suffix_text"],
            "architecture_live_doc_path": architecture_live_doc["architecture_live_doc_path"],
            "architecture_live_doc_digest": architecture_live_doc["architecture_live_doc_digest"],
            "architecture_live_doc_digest_text": architecture_live_doc["architecture_live_doc_digest_text"],
            "research_onboarding_files": research_onboarding.get("files_read", []),
        },
        "project_context": project_context,
        "runtime": {
            "config_dir": str(CONFIG_DIR),
            "context_dir": str(ensure_runtime_dir("context")),
            "log_dir": str(ensure_runtime_dir("logs")),
            "session_dir": str(ensure_runtime_dir("sessions")),
        },
        "model_catalog": format_model_table(),
    }
    return payload


def render_compiled_system_prompt(*, project: str | None = None, session_context: str | None = None) -> str:
    return str(compile_system_prompt(project=project, session_context=session_context)["prompt"])


def render_prompt_packet_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# AETHER-DFT Prompt Packet",
        "",
        f"- Created: {payload['created_at']}",
        f"- Workspace: `{payload['workspace']}`",
        f"- Project: `{payload.get('project') or 'none'}`",
        "",
        "## Model",
        "",
        "```json",
        json.dumps(payload["model"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Permission",
        "",
        "```json",
        json.dumps(payload["permission"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Runtime Contract",
        "",
    ]
    for item in payload["prompt"]["runtime_contract"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Tool Policy", ""])
    for item in payload["prompt"]["tool_policy"]:
        lines.append(f"- {item}")
    lines.extend([
        "",
        "## Compiled System Prompt",
        "",
        payload["prompt"]["compiled_system_prompt"].rstrip(),
        "",
    ])
    if payload.get("project_context"):
        lines.extend(["## Project Context", "", str(payload["project_context"]).rstrip(), ""])
    projection = payload["prompt"].get("compile_projection") or {}
    lines.extend(
        [
            "## Prompt Layers",
            "",
            "```json",
            json.dumps(
                {
                    "compile_strategy": projection.get("compile_strategy"),
                    "stable_layer_names": projection.get("stable_layer_names"),
                    "volatile_layer_names": projection.get("volatile_layer_names"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            "```",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"
