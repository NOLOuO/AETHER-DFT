from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from dft_app.llm import DomesticCopilotLLM
from dft_app.llm.key_store import load_api_keys
from dft_app.llm.provider_presets import build_provider_model_config

from . import __version__
from .chat import ask_once
from .model_catalog import (
    format_model_table,
    load_model_catalog,
    resolve_effective_model_id,
    set_default_model,
    split_model_id,
)
from .permissions import get_permission_mode, permission_mode_label, set_permission_mode
from .project_state import append_progress, init_project, list_projects, load_project, project_paths, read_project_context

PROGRAM_NAME = "AETHER-DFT"
PROGRAM_COMMAND = "aether"


class Colors:
    RESET = "\033[0m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    BLUE = "\033[34m"


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def visible_len(text: str) -> int:
    return len(ANSI_RE.sub("", text))


def ensure_console_utf8() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def program_model_id() -> str:
    return resolve_effective_model_id()


def program_context_window() -> int | None:
    try:
        _, _, config = _current_model_config()
    except Exception:
        return None
    value = config.get("context_window")
    return int(value) if value else None


def print_banner() -> None:
    print(f"{PROGRAM_NAME} v{__version__}")
    print(f"model: {program_model_id()}")
    print("role: conversational DFT research partner")


def print_quick_start() -> None:
    print_banner()
    print()
    print("Usage:")
    print("  aether mainline [\"继续当前科研任务\"]")
    print("  aether chat \"继续当前科研任务\"")
    print("  aether model current")
    print("  aether project list")
    print("  aether recommend --project <slug>")
    print("  aether doctor")
    print("  aether ssh")
    print()
    print("Long form:")
    print("  aether-dft agent \"...\"")


def print_demo_home(run_root: str | None = None) -> None:
    run_root = run_root or r"runs\task_0a4a1ddd\run_a295c506"
    box_width = 58

    def line(text: str = "") -> None:
        raw = ANSI_RE.sub("", text)
        clipped = text if len(raw) <= box_width - 1 else raw[: box_width - 4] + "..."
        print(f"{Colors.DIM}│{Colors.RESET} {clipped}{' ' * max(0, box_width - 1 - visible_len(clipped))}{Colors.DIM}│{Colors.RESET}")

    print(f"{Colors.DIM}┌{'─' * box_width}┐{Colors.RESET}")
    line(f"{Colors.BOLD}{Colors.CYAN}Session Info{Colors.RESET}")
    print(f"{Colors.DIM}├{'─' * box_width}┤{Colors.RESET}")
    line(f"Program: {Colors.CYAN}{PROGRAM_NAME}{Colors.RESET}")
    line(f"Version: {Colors.GREEN}{__version__}{Colors.RESET}")
    line(f"Model: {Colors.YELLOW}{program_model_id()}{Colors.RESET}")
    line(f"Workspace: {Path.cwd()}")
    line(f"Cluster: {Colors.BLUE}SSH / SLURM configured{Colors.RESET}")
    print(f"{Colors.DIM}└{'─' * box_width}┘{Colors.RESET}")
    print()
    print(f"{Colors.DIM}Type {Colors.GREEN}/help{Colors.DIM} for help, {Colors.GREEN}/exit{Colors.DIM} to quit{Colors.RESET}")


def print_chat_home(*, session_id: str, project: str | None = None) -> None:
    box_width = 58

    def line(text: str = "") -> None:
        raw = ANSI_RE.sub("", text)
        clipped = text if len(raw) <= box_width - 1 else raw[: box_width - 4] + "..."
        print(f"{Colors.DIM}│{Colors.RESET} {clipped}{' ' * max(0, box_width - 1 - visible_len(clipped))}{Colors.DIM}│{Colors.RESET}")

    print(f"{Colors.DIM}┌{'─' * box_width}┐{Colors.RESET}")
    line(f"{Colors.BOLD}{Colors.CYAN}Session Info{Colors.RESET}")
    print(f"{Colors.DIM}├{'─' * box_width}┤{Colors.RESET}")
    line(f"Program: {Colors.CYAN}{PROGRAM_NAME}{Colors.RESET}")
    line(f"Version: {Colors.GREEN}{__version__}{Colors.RESET}")
    line(f"Model: {Colors.YELLOW}{program_model_id()}{Colors.RESET}")
    context_window = program_context_window()
    if context_window:
        line(f"Context: {context_window:,} tokens")
    line(f"Permission: {Colors.BLUE}{permission_mode_label()}{Colors.RESET}")
    line(f"Session: {session_id}")
    line(f"Project: {project or 'none'}")
    print(f"{Colors.DIM}└{'─' * box_width}┘{Colors.RESET}")


def print_chat_shortcuts() -> None:
    print("直接输入自然语言即可；模型会自己判断是否需要调用工具。")
    print("可选快捷：/status 状态；/context 上下文；/model 切换模型；/permission 切换权限；/exit 退出。")


def _shorten_inline(value: Any, *, limit: int = 160) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def print_chat_help() -> None:
    print(f"\n{Colors.BOLD}{Colors.CYAN}AETHER interactive chat{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 44}{Colors.RESET}")
    print("主流程：直接说科研目标、结构问题、计算方案或结果疑问；模型自行决定是否调用工具。")
    print("ask 权限模式下，写文件/提交作业/产生副作用时只会弹一次确认，再决定是否执行。")
    print(f"  {Colors.GREEN}/status{Colors.RESET}      当前 session/model/permission")
    print(f"  {Colors.GREEN}/context{Colors.RESET}     当前 1M context budget 与压缩状态")
    print(f"  {Colors.GREEN}/model{Colors.RESET}       查看或切换 deepseek/qwen")
    print(f"  {Colors.GREEN}/permission{Colors.RESET}  完全开发 / 需要用户同意")
    print(f"  {Colors.GREEN}/project{Colors.RESET}     当前项目状态")
    print(f"  {Colors.GREEN}/recommend{Colors.RESET}   推荐下一步科研任务")
    print(f"  {Colors.GREEN}/clear{Colors.RESET}       清屏")
    print(f"  {Colors.GREEN}/exit{Colors.RESET}        退出")
    print(f"{Colors.DIM}{'─' * 44}{Colors.RESET}\n")


def print_chat_status(*, session_store: Any, session_id: str, project: str | None) -> None:
    state = session_store.load_state(session_id)
    print_json(
        {
            "program": PROGRAM_NAME,
            "version": __version__,
            "model": program_model_id(),
            "context_window": program_context_window(),
            "permission": {"mode": get_permission_mode(), "label": permission_mode_label()},
            "session": {
                "id": session_id,
                "project": project or state.get("project"),
                "turn_count": state.get("turn_count"),
                "updated_at": state.get("updated_at"),
            },
        }
    )


def print_chat_context_status(*, session_store: Any, session_id: str) -> None:
    from .context_budget import current_context_window_tokens, usable_context_chars, usable_context_tokens

    state = session_store.load_state(session_id)
    session_context = session_store.build_session_context(session_id)
    print_json(
        {
            "model": program_model_id(),
            "model_context_window_tokens": current_context_window_tokens(),
            "usable_context_tokens": usable_context_tokens(),
            "usable_context_chars": usable_context_chars(),
            "current_session_context_chars": len(session_context),
            "compacted_turn_count": state.get("compacted_turn_count", 0),
            "has_compact_summary": bool(str(state.get("compact_summary") or "").strip()),
        }
    )


def make_chat_progress_printer() -> Any:
    def _printer(event: dict[str, Any]) -> None:
        kind = str(event.get("event") or "")
        if kind == "turn_start":
            print(f"{Colors.DIM}thinking with {event.get('model_id') or program_model_id()}...{Colors.RESET}")
        elif kind == "model_request":
            print(f"{Colors.DIM}↻ model step {event.get('step')}/{event.get('max_steps')}{Colors.RESET}")
        elif kind == "tool_start":
            args = _shorten_inline(event.get("arguments"), limit=180)
            print(f"{Colors.BLUE}↳ tool{Colors.RESET} {event.get('name')} {Colors.DIM}{args}{Colors.RESET}")
        elif kind == "tool_finish":
            status = event.get("status") or "done"
            persisted = event.get("persisted_output_path")
            suffix = f" {Colors.DIM}{persisted}{Colors.RESET}" if persisted else ""
            print(f"{Colors.GREEN}✓ tool{Colors.RESET} {event.get('name')} status={status}{suffix}")
        elif kind == "tool_permission_required":
            label = event.get("permission_label") or "需要用户同意"
            print(
                f"{Colors.YELLOW}! permission{Colors.RESET} {event.get('name')} ({label}) "
                f"{Colors.DIM}{_shorten_inline(event.get('message'), limit=140)}{Colors.RESET}"
            )
        elif kind == "tool_permission_granted":
            print(f"{Colors.GREEN}✓ permission{Colors.RESET} {event.get('name')} approved")
        elif kind == "tool_permission_denied":
            print(f"{Colors.RED}× permission{Colors.RESET} {event.get('name')} denied")

    return _printer


def make_permission_prompt_callback() -> Any:
    def _prompt(details: dict[str, Any]) -> bool:
        name = details.get("tool_name") or "tool"
        label = details.get("permission_label") or "需要用户同意"
        message = _shorten_inline(details.get("message"), limit=180)
        print(f"{Colors.YELLOW}[approval]{Colors.RESET} {name} ({label})")
        if message:
            print(f"  {Colors.DIM}{message}{Colors.RESET}")
        try:
            answer = input("Approve once? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        return answer in {"y", "yes", "ok", "是", "确认", "批准"}

    return _prompt


def print_turn_footer(record: dict[str, Any]) -> None:
    elapsed = record.get("elapsed_seconds")
    tool_executions = record.get("tool_executions") or []
    footer = f"[record] {record['record_path']}"
    if elapsed is not None:
        footer += f" | {elapsed}s"
    if tool_executions:
        footer += f" | tools={len(tool_executions)}"
    print(f"{Colors.DIM}{footer}{Colors.RESET}")


def handle_chat_model_command(line: str, args: argparse.Namespace) -> None:
    raw = line[len("/model") :].strip()
    if raw in {"", "current", "list"}:
        print(f"{Colors.CYAN}current model{Colors.RESET}: {Colors.YELLOW}{program_model_id()}{Colors.RESET}")
        print(format_model_table(load_model_catalog(Path.cwd()), program_model_id()))
        print("切换：/model deepseek:deepseek-v4-pro  或  /model bailian:qwen3.7-max")
        return
    if raw.startswith("set "):
        raw = raw[4:].strip()
    try:
        set_default_model(raw)
    except Exception as exc:
        print(f"{Colors.RED}model switch failed{Colors.RESET}: {exc}")
        return
    args.model = raw
    print(f"{Colors.GREEN}model switched{Colors.RESET}: {Colors.YELLOW}{raw}{Colors.RESET}")


def handle_chat_permission_command(line: str) -> None:
    raw = line[len("/permission") :].strip()
    if not raw:
        print(f"{Colors.CYAN}permission{Colors.RESET}: {Colors.YELLOW}{get_permission_mode()}{Colors.RESET} / {permission_mode_label()}")
        print("切换：/permission dev（完全开发） 或 /permission ask（需要用户同意）")
        return
    try:
        payload = set_permission_mode(raw)
    except Exception as exc:
        print(f"{Colors.RED}permission switch failed{Colors.RESET}: {exc}")
        return
    print(f"{Colors.GREEN}permission switched{Colors.RESET}: {payload['mode']} / {payload['label']}")


def print_demo_help() -> None:
    print(f"\n{Colors.BOLD}{Colors.CYAN}Commands{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 36}{Colors.RESET}")
    print(f"  {Colors.GREEN}/help{Colors.RESET}    show commands")
    print(f"  {Colors.GREEN}/model{Colors.RESET}   show program/model/version")
    print(f"  {Colors.GREEN}/ssh{Colors.RESET}     probe cluster")
    print(f"  {Colors.GREEN}/status{Colors.RESET}  show current adsorption workflow status")
    print(f"  {Colors.GREEN}/project{Colors.RESET} project state summary")
    print(f"  {Colors.GREEN}/recommend{Colors.RESET} next research step")
    print(f"  {Colors.GREEN}/clear{Colors.RESET}   clear screen")
    print(f"  {Colors.GREEN}/exit{Colors.RESET}    quit")
    print(f"{Colors.DIM}{'─' * 36}{Colors.RESET}\n")


def run_demo_repl(run_root: str | None = None) -> int:
    run_root = run_root or r"F:\AETHER-DFT\runs\task_0a4a1ddd\run_a295c506"
    print_demo_home(run_root)
    while True:
        try:
            line = input(f"\n{Colors.CYAN}aether>{Colors.RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line in {"/exit", "exit", "quit", ":q"}:
            return 0
        if line == "/help":
            print_demo_help()
            continue
        if line == "/clear":
            os.system("cls" if os.name == "nt" else "clear")
            print_demo_home(run_root)
            continue
        if line == "/model":
            print(f"{Colors.CYAN}{PROGRAM_NAME}{Colors.RESET} v{Colors.GREEN}{__version__}{Colors.RESET} | model: {Colors.YELLOW}{program_model_id()}{Colors.RESET}")
            continue
        if line == "/ssh":
            from dft_app.remote import SSHRemoteRunner

            print(f"{Colors.DIM}probing cluster...{Colors.RESET}")
            result = SSHRemoteRunner().probe()
            color = Colors.GREEN if result.status == "ok" else Colors.YELLOW
            print(f"{color}{result.status}{Colors.RESET}: {result.message}")
            probe = result.details.get("probe", {}) if isinstance(result.details, dict) else {}
            if probe:
                print(f"  hostname: {probe.get('hostname', '')}")
                print(f"  sbatch  : {probe.get('sbatch', '')}")
                print(f"  squeue  : {probe.get('squeue', '')}")
            continue
        if line == "/status":
            from dft_app.cli.main import execute_adsorption_workflow

            result = execute_adsorption_workflow(run_root=Path(run_root), status=True)
            status = result["workflow_status"]["status"]
            color = Colors.GREEN if status in {"prepared", "completed", "aggregated"} else Colors.YELLOW
            print(f"workflow: {color}{status}{Colors.RESET}")
            print(f"monitor_pending: {result['workflow_status'].get('monitor_pending', [])}")
            print(f"submit_ready    : {result['workflow_status'].get('submit_ready', [])}")
            continue
        if line == "/project":
            print_json({"projects": list_projects()})
            continue
        if line.startswith("/recommend"):
            from .recommendations import recommend_next_tasks

            focus = line[len("/recommend") :].strip() or None
            print_json({"recommendations": recommend_next_tasks(None, focus=focus)})
            continue

        print(f"{Colors.DIM}thinking with {program_model_id()}...{Colors.RESET}")
        from .agent import run_agent_once

        record = run_agent_once(
            line,
            max_tokens=1000,
            max_steps=4,
            allow_cluster_submit=False,
            permission_prompt_callback=make_permission_prompt_callback(),
        )
        print(record["response"])


def _current_model_config(model_id: str | None = None) -> tuple[str, str, dict[str, Any]]:
    provider_id, model_name = split_model_id(model_id or resolve_effective_model_id())
    return provider_id, model_name, build_provider_model_config(provider_id, model_name)


def doctor(args: argparse.Namespace) -> int:
    runtime = DomesticCopilotLLM(Path.cwd()).describe_runtime()
    provider_id, model_name, config = _current_model_config(args.model)
    api_keys = load_api_keys(Path.cwd())
    has_key = bool(str(api_keys.get(provider_id, "")).strip())
    if not has_key:
        import os

        has_key = bool(os.getenv(str(config["api_key_env"]), "").strip())
    has_base_url = bool(str(config.get("base_url", "") or "").strip())
    payload = {
        "program": {
            "name": PROGRAM_NAME,
            "command": PROGRAM_COMMAND,
            "version": __version__,
        },
        "runtime": runtime,
        "effective_model": {
            "model_id": f"{provider_id}:{model_name}",
            "provider": provider_id,
            "model": model_name,
            "api_model": config["model"],
            "context_window": config.get("context_window"),
            "base_url": config.get("base_url"),
            "api_key_env": config.get("api_key_env"),
            "base_url_env": config.get("base_url_env"),
            "openai_compatible": True,
            "api_key_configured": has_key,
            "base_url_configured": has_base_url,
        },
    }
    print("AETHER-DFT doctor")
    print_json(payload)
    if not has_base_url:
        print("WARN: 当前 OpenAI-compatible provider 未配置 base_url")
        return 1
    if not has_key:
        print("WARN: 当前模型 provider 未找到 API key；请检查 api_keys.local.json 或对应环境变量。")
        return 1
    print("OK: OpenAI-compatible runtime configured")
    return 0


def handle_models(args: argparse.Namespace) -> int:
    catalog = load_model_catalog(Path.cwd())
    current = resolve_effective_model_id()
    if args.json:
        print_json({"current_model_id": current, "models": [item.to_dict() for item in catalog.values()]})
    else:
        print(format_model_table(catalog, current))
        print("\nAETHER-DFT currently exposes only project-fit built-ins:")
        print("  deepseek:deepseek-v4-pro")
        print("  bailian:qwen3.7-max")
    return 0


def handle_model_current(args: argparse.Namespace) -> int:
    provider_id, model_name, config = _current_model_config()
    print_json({
        "model_id": f"{provider_id}:{model_name}",
        "provider": provider_id,
        "model": model_name,
        "api_model": config["model"],
        "context_window": config.get("context_window"),
        "base_url": config.get("base_url"),
        "api_key_env": config.get("api_key_env"),
        "base_url_env": config.get("base_url_env"),
    })
    return 0


def handle_model_set(args: argparse.Namespace) -> int:
    preferences = set_default_model(args.model_id)
    print_json({"status": "ok", "current_model_id": args.model_id, "preferences": preferences})
    return 0


def handle_mainline(args: argparse.Namespace) -> int:
    prompt = " ".join(getattr(args, "prompt", []) or []).strip()
    print(f"{Colors.BOLD}{Colors.CYAN}AETHER-DFT mainline{Colors.RESET}")
    print(f"{Colors.DIM}discussion -> plan -> structure -> recommend{Colors.RESET}")
    if prompt:
        print(f"{Colors.DIM}prompt: {prompt}{Colors.RESET}")
        return handle_chat(args)
    if hasattr(sys.stdin, "isatty") and sys.stdin.isatty():
        return handle_chat(args)
    if args.project:
        from .recommendations import recommend_next_tasks

        print_json(
            {
                "project": args.project,
                "recommendations": recommend_next_tasks(args.project, focus=getattr(args, "focus", None)),
            }
        )
    else:
        print_json(
            {
                "mainline": [
                    "1. 讨论课题并沉淀方案",
                    "2. 获取/生成/检查结构",
                    "3. 推荐下一步科研任务",
                ]
            }
        )
    return 0


def handle_project_init(args: argparse.Namespace) -> int:
    print_json(init_project(args.name, description=args.description or "", overwrite=args.overwrite))
    return 0


def handle_project_list(args: argparse.Namespace) -> int:
    print_json({"projects": list_projects()})
    return 0


def handle_project_show(args: argparse.Namespace) -> int:
    payload = load_project(args.slug)
    paths = project_paths(args.slug)
    if args.context:
        payload["context"] = read_project_context(args.slug)
    payload["paths"] = {k: str(v) for k, v in paths.__dict__.items() if k != "slug"}
    print_json(payload)
    return 0


def handle_project_progress(args: argparse.Namespace) -> int:
    path = append_progress(
        args.slug,
        completed=args.completed or [],
        blockers=args.blocker or [],
        next_steps=args.next_step or [],
    )
    print_json({"status": "ok", "progress_path": str(path)})
    return 0


def handle_task_plan(args: argparse.Namespace) -> int:
    from .task_bridge import create_task_plan

    prompt = " ".join(args.prompt).strip()
    envelope = create_task_plan(
        prompt,
        project=args.project,
        material=args.material,
        structure_path=args.structure_path,
        task_type=args.task_type,
        submit_profile=args.submit_profile,
        model_spec_path=args.model_spec_path,
        step2_manifest_path=args.step2_manifest_path,
        candidate_id=args.candidate_id,
        planner_mode=args.planner,
        persist=not args.no_persist,
    )
    print_json(envelope.to_dict())
    return 0


def handle_task_run(args: argparse.Namespace) -> int:
    from .task_bridge import run_dft_task

    prompt = " ".join(args.prompt).strip()
    if args.remote_submit:
        execution_mode = "remote_submit"
    elif args.submit:
        execution_mode = "submit"
    elif args.build:
        execution_mode = "build"
    else:
        execution_mode = "dry_run"
    result = run_dft_task(
        prompt,
        project=args.project,
        material=args.material,
        structure_path=args.structure_path,
        task_type=args.task_type,
        submit_profile=args.submit_profile,
        model_spec_path=args.model_spec_path,
        step2_manifest_path=args.step2_manifest_path,
        candidate_id=args.candidate_id,
        planner_mode=args.planner,
        execution_mode=execution_mode,
    )
    print_json(result)
    return 0 if result.get("status") in {"ok", "needs_confirmation"} else 1


def handle_task_list(args: argparse.Namespace) -> int:
    from .task_bridge import list_task_records

    print_json({"tasks": list_task_records(args.project)})
    return 0


def handle_kb_add(args: argparse.Namespace) -> int:
    from .knowledge import add_note

    content = args.text
    if args.file:
        content = Path(args.file).read_text(encoding="utf-8")
    note = add_note(args.project, args.title, content, tags=args.tag or [])
    print_json({"status": "ok", "note": note.to_dict()})
    return 0


def handle_kb_list(args: argparse.Namespace) -> int:
    from .knowledge import list_notes

    print_json({"notes": list_notes(args.project)})
    return 0


def handle_kb_search(args: argparse.Namespace) -> int:
    from .knowledge import search_notes

    print_json({"matches": search_notes(args.project, " ".join(args.query))})
    return 0


def handle_kb_show(args: argparse.Namespace) -> int:
    from .knowledge import show_note

    print_json(show_note(args.note, project=args.project))
    return 0


def handle_adsorption_plan(args: argparse.Namespace) -> int:
    from .adsorption import plan_adsorption_task

    prompt = " ".join(args.prompt).strip()
    plan = plan_adsorption_task(
        prompt,
        project=args.project,
        adsorbate=args.adsorbate,
        material=args.material,
        slab_path=args.slab_path,
        preferred_site=args.preferred_site,
        preferred_orientation=args.preferred_orientation,
        persist=not args.no_persist,
    )
    print_json(plan.to_dict())
    return 0


def handle_adsorption_build_slab(args: argparse.Namespace) -> int:
    from .adsorption import build_adsorption_slab

    result = build_adsorption_slab(
        material=args.material,
        output_dir=args.output_dir,
        structure_path=args.structure_path,
        mp_id=args.mp_id,
        source=args.source,
        miller_index=args.miller,
        supercell=args.supercell,
        min_slab_size=args.min_slab_size,
        min_vacuum_size=args.min_vacuum_size,
        fixed_bottom_layers=args.fixed_bottom_layers,
    )
    print_json(result.to_dict())
    return 0


def handle_adsorption_candidates(args: argparse.Namespace) -> int:
    from .adsorption import generate_adsorption_candidates

    prompt = args.prompt or f"{args.adsorbate} adsorption on {args.material}"
    result = generate_adsorption_candidates(
        slab_path=args.slab_path,
        adsorbate=args.adsorbate,
        material=args.material,
        prompt=prompt,
        project=args.project,
        output_dir=args.output_dir,
        task_id=args.task_id,
        candidate_height=args.candidate_height,
        max_sites_per_family=args.max_sites_per_family,
        preferred_site=args.preferred_site,
        preferred_orientation=args.preferred_orientation,
        vacancy_species=args.vacancy_species,
    )
    print_json(result)
    return 0 if result.get("status") == "ok" else 1


def handle_adsorption_pipeline(args: argparse.Namespace) -> int:
    from .adsorption import run_adsorption_pipeline

    prompt = args.prompt or f"{args.adsorbate} adsorption on {args.material}"
    result = run_adsorption_pipeline(
        material=args.material,
        adsorbate=args.adsorbate,
        output_dir=args.output_dir,
        prompt=prompt,
        project=args.project,
        structure_path=args.structure_path,
        mp_id=args.mp_id,
        source=args.source,
        miller_index=args.miller,
        supercell=args.supercell,
        candidate_height=args.candidate_height,
        max_sites_per_family=args.max_sites_per_family,
        preferred_site=args.preferred_site,
        preferred_orientation=args.preferred_orientation,
        vacancy_species=args.vacancy_species,
    )
    print_json(result)
    return 0 if result.get("status") == "ok" else 1


def handle_adsorption_full(args: argparse.Namespace) -> int:
    from .adsorption import run_adsorption_full_workflow

    prompt = args.prompt or f"计算 {args.adsorbate} 在 {args.material} 上的吸附能"
    result = run_adsorption_full_workflow(
        material=args.material,
        adsorbate=args.adsorbate,
        output_dir=args.output_dir,
        prompt=prompt,
        project=args.project,
        structure_path=args.structure_path,
        mp_id=args.mp_id,
        source=args.source,
        miller_index=args.miller,
        supercell=args.supercell,
        candidate_id=args.candidate_id,
        submit_profile=args.submit_profile,
        candidate_height=args.candidate_height,
        max_sites_per_family=args.max_sites_per_family,
        preferred_site=args.preferred_site,
        preferred_orientation=args.preferred_orientation,
        vacancy_species=args.vacancy_species,
    )
    print_json(result)
    return 0 if result.get("status") == "prepared" else 1


def handle_recommend(args: argparse.Namespace) -> int:
    from .recommendations import recommend_next_tasks

    print_json({"recommendations": recommend_next_tasks(args.project, focus=args.focus)})
    return 0


def handle_chat(args: argparse.Namespace) -> int:
    from .session_store import AetherSessionStore

    prompt = " ".join(args.prompt).strip()
    session_store = AetherSessionStore()
    session_id = args.session_id
    if args.resume:
        payload = session_store.resume_payload(session_id=session_id, project=args.project)
        if payload["status"] == "ok":
            session_id = payload["session_id"]
        elif session_id:
            print(f"错误: session 不存在: {session_id}")
            return 1
    if prompt and (args.task_plan or args.task_run):
        if args.task_run:
            from .task_bridge import run_dft_task

            result = run_dft_task(
                prompt,
                project=args.project,
                material=args.material,
                structure_path=args.structure_path,
                task_type=args.task_type,
                submit_profile=args.submit_profile,
                model_spec_path=getattr(args, "model_spec_path", None),
                step2_manifest_path=getattr(args, "step2_manifest_path", None),
                candidate_id=getattr(args, "candidate_id", None),
                planner_mode=args.planner,
                execution_mode="dry_run",
            )
            print_json(result)
            return 0 if result.get("status") in {"ok", "needs_confirmation"} else 1
        from .task_bridge import create_task_plan

        envelope = create_task_plan(
            prompt,
            project=args.project,
            material=args.material,
            structure_path=args.structure_path,
            task_type=args.task_type,
            submit_profile=args.submit_profile,
            model_spec_path=getattr(args, "model_spec_path", None),
            step2_manifest_path=getattr(args, "step2_manifest_path", None),
            candidate_id=getattr(args, "candidate_id", None),
            planner_mode=args.planner,
            persist=True,
        )
        print_json(envelope.to_dict())
        return 0

    if prompt:
        record = ask_once(
            prompt,
            project=args.project,
            model_id=args.model,
            max_tokens=args.max_tokens,
            max_steps=args.max_steps,
            session_id=session_id,
            permission_mode=get_permission_mode(),
            progress_callback=make_chat_progress_printer(),
            permission_prompt_callback=make_permission_prompt_callback(),
        )
        print(record["response"])
        print()
        print_turn_footer(record)
        next_steps = (record.get("progress") or {}).get("next_steps") or []
        if next_steps:
            print(f"[next] {next_steps[0]}")
        return 0

    if not session_id:
        session_id = session_store.start_session(project=args.project)
    print_chat_home(session_id=session_id, project=args.project)
    print_chat_shortcuts()
    if args.resume:
        payload = session_store.resume_payload(session_id=session_id, project=args.project)
        if payload["status"] == "ok" and payload["recent_turns"]:
            print("最近对话：")
            for turn in payload["recent_turns"][-3:]:
                record = turn.get("record", {})
                print(f"- user: {str(record.get('prompt') or '')[:80]}")
                print(f"  assistant: {str(record.get('response') or '')[:80]}")
    while True:
        try:
            line = input("aether> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if line in {"/exit", "exit", "quit", ":q"}:
            return 0
        if not line:
            continue
        if line == "/help":
            print_chat_help()
            continue
        if line == "/clear":
            os.system("cls" if os.name == "nt" else "clear")
            print_chat_home(session_id=session_id, project=args.project)
            print_chat_shortcuts()
            continue
        if line == "/status":
            print_chat_status(session_store=session_store, session_id=session_id, project=args.project)
            continue
        if line == "/context":
            print_chat_context_status(session_store=session_store, session_id=session_id)
            continue
        if line.startswith("/model"):
            handle_chat_model_command(line, args)
            continue
        if line.startswith("/permission"):
            handle_chat_permission_command(line)
            continue
        if line.startswith("/task "):
            from .task_bridge import create_task_plan

            envelope = create_task_plan(
                line[len("/task ") :],
                project=args.project,
                material=args.material,
                structure_path=args.structure_path,
                task_type=args.task_type,
                submit_profile=args.submit_profile,
                model_spec_path=getattr(args, "model_spec_path", None),
                step2_manifest_path=getattr(args, "step2_manifest_path", None),
                candidate_id=getattr(args, "candidate_id", None),
                planner_mode=args.planner,
                persist=True,
            )
            print_json(envelope.to_dict())
            continue
        if line.startswith("/run "):
            from .task_bridge import run_dft_task

            result = run_dft_task(
                line[len("/run ") :],
                project=args.project,
                material=args.material,
                structure_path=args.structure_path,
                task_type=args.task_type,
                submit_profile=args.submit_profile,
                model_spec_path=getattr(args, "model_spec_path", None),
                step2_manifest_path=getattr(args, "step2_manifest_path", None),
                candidate_id=getattr(args, "candidate_id", None),
                planner_mode=args.planner,
                execution_mode="dry_run",
            )
            print_json(result)
            continue
        if line.startswith("/adsorb "):
            from .adsorption import plan_adsorption_task

            plan = plan_adsorption_task(
                line[len("/adsorb ") :],
                project=args.project,
                adsorbate=args.adsorbate,
                material=args.material,
                slab_path=args.slab_path,
                preferred_site=args.preferred_site,
                preferred_orientation=args.preferred_orientation,
                persist=True,
            )
            print_json(plan.to_dict())
            continue
        if line == "/project":
            if not args.project:
                print("no project bound; restart with --project <slug>")
                continue
            print_json({"project": load_project(args.project), "context": read_project_context(args.project)})
            continue
        if line.startswith("/recommend"):
            from .recommendations import recommend_next_tasks

            focus = line[len("/recommend") :].strip() or None
            print_json({"recommendations": recommend_next_tasks(args.project, focus=focus)})
            continue
        record = ask_once(
            line,
            project=args.project,
            model_id=args.model,
            max_tokens=args.max_tokens,
            max_steps=args.max_steps,
            session_id=session_id,
            permission_mode=get_permission_mode(),
            progress_callback=make_chat_progress_printer(),
            permission_prompt_callback=make_permission_prompt_callback(),
        )
        print(record["response"])
        print_turn_footer(record)
        next_steps = (record.get("progress") or {}).get("next_steps") or []
        if next_steps:
            print(f"[next] {next_steps[0]}")


def handle_session_list(args: argparse.Namespace) -> int:
    from .session_store import AetherSessionStore

    store = AetherSessionStore()
    print_json({"sessions": [item.to_dict() for item in store.list_sessions(project=args.project, limit=args.limit)]})
    return 0


def handle_session_show(args: argparse.Namespace) -> int:
    from .session_store import AetherSessionStore

    store = AetherSessionStore()
    payload = store.resume_payload(session_id=args.session_id, limit=args.limit)
    print_json(payload)
    return 0 if payload["status"] == "ok" else 1


def handle_session_resume(args: argparse.Namespace) -> int:
    from .session_store import AetherSessionStore

    store = AetherSessionStore()
    payload = store.resume_payload(session_id=args.session_id, project=args.project, limit=args.limit)
    print_json(payload)
    return 0 if payload["status"] == "ok" else 1


def handle_tools_list(args: argparse.Namespace) -> int:
    from .tool_registry import list_registered_tools

    print_json({"tools": list_registered_tools()})
    return 0


def handle_tools_run(args: argparse.Namespace) -> int:
    from .agent_tools import parse_tool_arguments
    from .tool_registry import AetherToolRegistry

    arguments = parse_tool_arguments(args.arguments)
    registry = AetherToolRegistry(allow_cluster_submit=args.allow_cluster_submit)
    print_json(registry.run_tool(args.name, arguments))
    return 0


def handle_run(args: argparse.Namespace) -> int:
    try:
        from dft_app.cli.main import main as dft_main
    except ModuleNotFoundError as exc:
        print(f"错误: DFT 主线依赖未安装: {exc.name}；请先安装 pyproject.toml 中的科学计算依赖。")
        return 1

    return dft_main(["run", *args.dft_args])


def handle_dft(args: argparse.Namespace) -> int:
    try:
        from dft_app.cli.main import main as dft_main
    except ModuleNotFoundError as exc:
        print(f"错误: DFT 主线依赖未安装: {exc.name}；请先安装 pyproject.toml 中的科学计算依赖。")
        return 1

    return dft_main(args.dft_args)


def handle_structure_convert(args: argparse.Namespace) -> int:
    from dft_shared.structure_analyzer.io import convert_structure

    print_json(convert_structure(args.input, args.output, fmt=args.fmt))
    return 0


def handle_structure_tools(args: argparse.Namespace) -> int:
    from dft_shared.structure_analyzer.tool_registry import list_structure_tools

    print_json({"structure_tools": list_structure_tools()})
    return 0


def handle_context_snapshot(args: argparse.Namespace) -> int:
    from .context import write_context_snapshot

    path = write_context_snapshot(project=args.project)
    print_json({"status": "ok", "context_snapshot": str(path)})
    return 0


def handle_harness_preflight(args: argparse.Namespace) -> int:
    from .harness import preflight

    payload = preflight()
    print_json(payload)
    return 0 if payload.get("ok") else 1


def handle_cluster_import(args: argparse.Namespace) -> int:
    from dft_app.remote.config import write_local_cluster_profile

    payload = write_local_cluster_profile(
        source_ssh_config=Path(args.source),
        alias=args.alias,
        remote_base_dir=args.remote_base_dir,
    )
    print_json(payload)
    return 0


def handle_cluster_config(args: argparse.Namespace) -> int:
    from dft_app.remote import SSHRemoteRunner

    print_json({"status": "ok", "config": SSHRemoteRunner().describe_config()})
    return 0


def handle_cluster_probe(args: argparse.Namespace) -> int:
    from dft_app.remote import SSHRemoteRunner

    result = SSHRemoteRunner().probe()
    print_json({"status": result.status, "message": result.message, "details": result.details})
    return 0 if result.status in {"ok", "partial"} else 1


def handle_agent_run(args: argparse.Namespace) -> int:
    from .agent import run_agent_once

    prompt = " ".join(args.prompt).strip()
    record = run_agent_once(
        prompt,
        project=args.project,
        model_id=args.model,
        max_tokens=args.max_tokens,
        max_steps=args.max_steps,
        allow_cluster_submit=args.allow_cluster_submit,
        permission_prompt_callback=make_permission_prompt_callback(),
    )
    print(record["response"])
    print("\n[tool_executions]")
    print_json(record["tool_executions"])
    print(f"\n[record] {record['record_path']}")
    next_steps = (record.get("progress") or {}).get("next_steps") or []
    if next_steps:
        print(f"[next] {next_steps[0]}")
    return 0


def handle_demo(args: argparse.Namespace) -> int:
    if args.once:
        print_demo_home(args.run_root)
        return 0
    return run_demo_repl(args.run_root)


def handle_workflow_short(args: argparse.Namespace) -> int:
    from dft_app.cli.main import execute_adsorption_workflow

    action = args.short_command
    if action == "submit" and not args.yes:
        print("错误: 远程提交会真实 sbatch 作业；请显式加 --yes。")
        return 1
    result = execute_adsorption_workflow(
        run_root=Path(args.run_root),
        submit=action == "submit",
        status=action == "status",
        monitor=action == "monitor",
        fetch=action == "fetch",
        parse_analyze=action == "analyze",
        remote=action == "submit",
    )
    print_json(result)
    return 0


def handle_explain(args: argparse.Namespace) -> int:
    from dft_app.cli.main import main as dft_main

    forwarded = ["dft-tools-explain"]
    if args.run_id:
        forwarded.extend(["--run-id", args.run_id])
    if args.run_root:
        forwarded.extend(["--run-root", args.run_root])
    if args.base_url:
        forwarded.extend(["--base-url", args.base_url])
    if args.no_kb_ingest:
        forwarded.append("--no-kb-ingest")
    return dft_main(forwarded)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROGRAM_COMMAND,
        description=f"{PROGRAM_NAME} v{__version__} | model: {program_model_id()} | conversational DFT research partner.",
    )
    parser.add_argument("--version", action="version", version=f"{PROGRAM_NAME} {__version__} | model: {program_model_id()}")
    sub = parser.add_subparsers(dest="command")

    doctor_parser = sub.add_parser("doctor", help="检查程序名称、版本、模型运行时与项目底座。")
    doctor_parser.add_argument("--model", help="临时检查指定模型，格式 provider:model。")
    doctor_parser.set_defaults(func=doctor)

    models_parser = sub.add_parser("models", help="列出可用 OpenAI-compatible provider/model。")
    models_parser.add_argument("--json", action="store_true")
    models_parser.set_defaults(func=handle_models)

    demo_parser = sub.add_parser("demo", help="组会展示用极简首页，不提交、不联网。")
    demo_parser.add_argument("--run-root")
    demo_parser.add_argument("--once", action="store_true", help="只打印首页，不进入交互。")
    demo_parser.set_defaults(func=handle_demo)

    model_parser = sub.add_parser("model", help="查看或切换当前模型。")
    model_sub = model_parser.add_subparsers(dest="model_command")
    model_current = model_sub.add_parser("current", help="显示当前模型。")
    model_current.set_defaults(func=handle_model_current)
    model_set = model_sub.add_parser("set", help="设置默认模型，格式 provider:model。")
    model_set.add_argument("model_id")
    model_set.set_defaults(func=handle_model_set)

    project_parser = sub.add_parser("project", help="项目级科研状态容器。")
    project_sub = project_parser.add_subparsers(dest="project_command")
    project_init = project_sub.add_parser("init", help="创建项目容器。")
    project_init.add_argument("name")
    project_init.add_argument("--description", default="")
    project_init.add_argument("--overwrite", action="store_true")
    project_init.set_defaults(func=handle_project_init)
    project_list = project_sub.add_parser("list", help="列出项目。")
    project_list.set_defaults(func=handle_project_list)
    project_show = project_sub.add_parser("show", help="显示项目元数据/上下文。")
    project_show.add_argument("slug")
    project_show.add_argument("--context", action="store_true")
    project_show.set_defaults(func=handle_project_show)
    project_progress = project_sub.add_parser("progress", help="追加研究进展日期块。")
    project_progress.add_argument("slug")
    project_progress.add_argument("--completed", action="append")
    project_progress.add_argument("--blocker", action="append")
    project_progress.add_argument("--next-step", action="append")
    project_progress.set_defaults(func=handle_project_progress)

    task_parser = sub.add_parser("task", help="对话/自然语言 -> 结构化 DFT 任务 -> 主线执行。")
    task_sub = task_parser.add_subparsers(dest="task_command")
    task_plan = task_sub.add_parser("plan", help="生成并保存结构化 DFT 任务，不执行。")
    task_plan.add_argument("prompt", nargs="+")
    task_plan.add_argument("--project")
    task_plan.add_argument("--material")
    task_plan.add_argument("--structure-path")
    task_plan.add_argument("--task-type")
    task_plan.add_argument("--submit-profile")
    task_plan.add_argument("--model-spec-path", help="Step 2 产出的 model_spec.json；用于保留模型建模证据。")
    task_plan.add_argument("--step2-manifest-path", help="Step 2 候选/结构 manifest；用于 Step 3 lineage。")
    task_plan.add_argument("--candidate-id", help="Step 2 中被选中的候选结构 ID。")
    task_plan.add_argument("--planner", choices=["rule", "auto"], default="rule")
    task_plan.add_argument("--no-persist", action="store_true")
    task_plan.set_defaults(func=handle_task_plan)
    task_run = task_sub.add_parser("run", help="生成任务并进入 DFT 主线；默认 dry-run，不提交。")
    task_run.add_argument("prompt", nargs="+")
    task_run.add_argument("--project")
    task_run.add_argument("--material")
    task_run.add_argument("--structure-path")
    task_run.add_argument("--task-type")
    task_run.add_argument("--submit-profile")
    task_run.add_argument("--model-spec-path", help="Step 2 产出的 model_spec.json；用于保留模型建模证据。")
    task_run.add_argument("--step2-manifest-path", help="Step 2 候选/结构 manifest；用于 Step 3 lineage。")
    task_run.add_argument("--candidate-id", help="Step 2 中被选中的候选结构 ID。")
    task_run.add_argument("--planner", choices=["rule", "auto"], default="rule")
    task_run.add_argument("--build", action="store_true", help="真实生成工作区，但不提交。")
    task_run.add_argument("--submit", action="store_true", help="本地 Slurm submit；必须显式指定。")
    task_run.add_argument("--remote-submit", action="store_true", help="远程 Slurm submit；必须显式指定。")
    task_run.set_defaults(func=handle_task_run)
    task_list = task_sub.add_parser("list", help="列出任务记录。")
    task_list.add_argument("--project")
    task_list.set_defaults(func=handle_task_list)

    kb_parser = sub.add_parser("kb", help="项目知识库轻量入口。")
    kb_sub = kb_parser.add_subparsers(dest="kb_command")
    kb_add = kb_sub.add_parser("add", help="添加知识条目。")
    kb_add.add_argument("project")
    kb_add.add_argument("--title", required=True)
    kb_add.add_argument("--text", default="")
    kb_add.add_argument("--file")
    kb_add.add_argument("--tag", action="append")
    kb_add.set_defaults(func=handle_kb_add)
    kb_list = kb_sub.add_parser("list", help="列出知识条目。")
    kb_list.add_argument("project")
    kb_list.set_defaults(func=handle_kb_list)
    kb_search = kb_sub.add_parser("search", help="搜索知识条目。")
    kb_search.add_argument("project")
    kb_search.add_argument("query", nargs="+")
    kb_search.set_defaults(func=handle_kb_search)
    kb_show = kb_sub.add_parser("show", help="显示知识条目。")
    kb_show.add_argument("note")
    kb_show.add_argument("--project")
    kb_show.set_defaults(func=handle_kb_show)

    adsorption_parser = sub.add_parser("adsorption", help="第一个科研任务：吸附建模与候选构型生成。")
    adsorption_sub = adsorption_parser.add_subparsers(dest="adsorption_command")
    adsorption_plan = adsorption_sub.add_parser("plan", help="规划吸附任务；缺 slab/adsorbate 时给出下一步。")
    adsorption_plan.add_argument("prompt", nargs="+")
    adsorption_plan.add_argument("--project")
    adsorption_plan.add_argument("--adsorbate")
    adsorption_plan.add_argument("--material")
    adsorption_plan.add_argument("--slab-path")
    adsorption_plan.add_argument("--preferred-site")
    adsorption_plan.add_argument("--preferred-orientation")
    adsorption_plan.add_argument("--no-persist", action="store_true")
    adsorption_plan.set_defaults(func=handle_adsorption_plan)
    adsorption_build_slab = adsorption_sub.add_parser(
        "build-slab",
        help="首步建模：从本地结构、Materials Project（可选）或 ASE 元素兜底生成 slab POSCAR。",
    )
    adsorption_build_slab.add_argument("--material", required=True, help="材料/表面标签，例如 Pt(111)、Pt、Al2O3。")
    adsorption_build_slab.add_argument("--output-dir", required=True)
    adsorption_build_slab.add_argument("--structure-path", help="本地 bulk/slab 结构文件：CIF/POSCAR/CONTCAR/XSD 等。")
    adsorption_build_slab.add_argument("--mp-id", help="Materials Project ID，例如 mp-126；需要 MP_API_KEY。")
    adsorption_build_slab.add_argument(
        "--source",
        choices=["auto", "ase", "element", "builtin", "mp", "materials_project", "local"],
        default="auto",
        help="结构来源；默认 auto：本地/MP 参数优先，否则 ASE 单元素块体兜底。",
    )
    adsorption_build_slab.add_argument("--miller", nargs=3, type=int, metavar=("H", "K", "L"))
    adsorption_build_slab.add_argument("--supercell", nargs=3, type=int, default=[2, 2, 1], metavar=("A", "B", "C"))
    adsorption_build_slab.add_argument("--min-slab-size", type=float, default=8.0)
    adsorption_build_slab.add_argument("--min-vacuum-size", type=float, default=12.0)
    adsorption_build_slab.add_argument("--fixed-bottom-layers", type=int, default=2)
    adsorption_build_slab.set_defaults(func=handle_adsorption_build_slab)
    adsorption_candidates = adsorption_sub.add_parser("candidates", help="用 slab + adsorbate 生成吸附候选构型。")
    adsorption_candidates.add_argument("--slab-path", required=True)
    adsorption_candidates.add_argument("--adsorbate", required=True)
    adsorption_candidates.add_argument("--material", required=True)
    adsorption_candidates.add_argument("--prompt")
    adsorption_candidates.add_argument("--project")
    adsorption_candidates.add_argument("--output-dir")
    adsorption_candidates.add_argument("--task-id")
    adsorption_candidates.add_argument("--candidate-height", type=float, default=2.1)
    adsorption_candidates.add_argument("--max-sites-per-family", type=int, default=2)
    adsorption_candidates.add_argument("--preferred-site")
    adsorption_candidates.add_argument("--preferred-orientation")
    adsorption_candidates.add_argument("--vacancy-species")
    adsorption_candidates.set_defaults(func=handle_adsorption_candidates)
    adsorption_pipeline = adsorption_sub.add_parser(
        "pipeline",
        help="一键首步吸附建模：build-slab -> candidates（不提交 VASP）。",
    )
    adsorption_pipeline.add_argument("--material", required=True)
    adsorption_pipeline.add_argument("--adsorbate", required=True)
    adsorption_pipeline.add_argument("--output-dir", required=True)
    adsorption_pipeline.add_argument("--prompt")
    adsorption_pipeline.add_argument("--project")
    adsorption_pipeline.add_argument("--structure-path")
    adsorption_pipeline.add_argument("--mp-id")
    adsorption_pipeline.add_argument(
        "--source",
        choices=["auto", "ase", "element", "builtin", "mp", "materials_project", "local"],
        default="auto",
    )
    adsorption_pipeline.add_argument("--miller", nargs=3, type=int, metavar=("H", "K", "L"))
    adsorption_pipeline.add_argument("--supercell", nargs=3, type=int, default=[2, 2, 1], metavar=("A", "B", "C"))
    adsorption_pipeline.add_argument("--candidate-height", type=float, default=2.1)
    adsorption_pipeline.add_argument("--max-sites-per-family", type=int, default=2)
    adsorption_pipeline.add_argument("--preferred-site")
    adsorption_pipeline.add_argument("--preferred-orientation")
    adsorption_pipeline.add_argument("--vacancy-species")
    adsorption_pipeline.set_defaults(func=handle_adsorption_pipeline)
    adsorption_full = adsorption_sub.add_parser(
        "full",
        help="真实吸附全流程到 VASP 工作区：build-slab -> candidates -> select -> 三体系 workflow bundle。",
    )
    adsorption_full.add_argument("--material", required=True)
    adsorption_full.add_argument("--adsorbate", required=True)
    adsorption_full.add_argument("--output-dir", required=True)
    adsorption_full.add_argument("--prompt")
    adsorption_full.add_argument("--project")
    adsorption_full.add_argument("--structure-path")
    adsorption_full.add_argument("--mp-id")
    adsorption_full.add_argument(
        "--source",
        choices=["auto", "ase", "element", "builtin", "mp", "materials_project", "local"],
        default="auto",
    )
    adsorption_full.add_argument("--miller", nargs=3, type=int, metavar=("H", "K", "L"))
    adsorption_full.add_argument("--supercell", nargs=3, type=int, default=[2, 2, 1], metavar=("A", "B", "C"))
    adsorption_full.add_argument("--candidate-id", help="不指定时自动选择排序第一的 candidate。")
    adsorption_full.add_argument("--submit-profile")
    adsorption_full.add_argument("--candidate-height", type=float, default=2.1)
    adsorption_full.add_argument("--max-sites-per-family", type=int, default=2)
    adsorption_full.add_argument("--preferred-site")
    adsorption_full.add_argument("--preferred-orientation")
    adsorption_full.add_argument("--vacancy-species")
    adsorption_full.set_defaults(func=handle_adsorption_full)

    recommend_parser = sub.add_parser("recommend", help="基于项目状态/任务/知识库推荐下一步科研任务。")
    recommend_parser.add_argument("--project")
    recommend_parser.add_argument("--focus")
    recommend_parser.set_defaults(func=handle_recommend)

    session_parser = sub.add_parser("session", help="会话持久化与续接。")
    session_sub = session_parser.add_subparsers(dest="session_command")
    session_list = session_sub.add_parser("list", help="列出最近会话。")
    session_list.add_argument("--project")
    session_list.add_argument("--limit", type=int, default=20)
    session_list.set_defaults(func=handle_session_list)
    session_show = session_sub.add_parser("show", help="显示会话状态和最近 transcript。")
    session_show.add_argument("session_id")
    session_show.add_argument("--limit", type=int, default=20)
    session_show.set_defaults(func=handle_session_show)
    session_resume = session_sub.add_parser("resume", help="读取可续接的会话上下文。")
    session_resume.add_argument("session_id", nargs="?")
    session_resume.add_argument("--project")
    session_resume.add_argument("--limit", type=int, default=8)
    session_resume.set_defaults(func=handle_session_resume)

    tools_parser = sub.add_parser("tools", help="AETHER harness 工具注册表。")
    tools_sub = tools_parser.add_subparsers(dest="tools_command")
    tools_list = tools_sub.add_parser("list", help="列出模型可调用工具。")
    tools_list.set_defaults(func=handle_tools_list)
    tools_run = tools_sub.add_parser("run", help="直接运行一个注册工具。")
    tools_run.add_argument("name")
    tools_run.add_argument("--arguments", default="{}", help="JSON 参数对象。")
    tools_run.add_argument("--allow-cluster-submit", action="store_true")
    tools_run.set_defaults(func=handle_tools_run)

    chat_parser = sub.add_parser("chat", help="对话式科研合伙人入口；无 prompt 时进入 REPL。")
    chat_parser.add_argument("prompt", nargs="*")
    chat_parser.add_argument("--project")
    chat_parser.add_argument("--model", help="临时使用 provider:model。")
    chat_parser.add_argument("--max-tokens", type=int)
    chat_parser.add_argument("--max-steps", type=int, default=6)
    chat_parser.add_argument("--resume", action="store_true", help="续接最近或指定 session。")
    chat_parser.add_argument("--session-id", help="指定要续接/写入的 session id。")
    chat_parser.add_argument("--task-plan", action="store_true", help="把本轮输入转成结构化 DFT 任务，不普通聊天。")
    chat_parser.add_argument("--task-run", action="store_true", help="把本轮输入转成 DFT 任务并执行 dry-run。")
    chat_parser.add_argument("--material")
    chat_parser.add_argument("--structure-path")
    chat_parser.add_argument("--slab-path")
    chat_parser.add_argument("--adsorbate")
    chat_parser.add_argument("--preferred-site")
    chat_parser.add_argument("--preferred-orientation")
    chat_parser.add_argument("--task-type")
    chat_parser.add_argument("--submit-profile")
    chat_parser.add_argument("--model-spec-path")
    chat_parser.add_argument("--step2-manifest-path")
    chat_parser.add_argument("--candidate-id")
    chat_parser.add_argument("--planner", choices=["rule", "auto"], default="rule")
    chat_parser.set_defaults(func=handle_chat)

    mainline_parser = sub.add_parser("mainline", help="显式主线入口：讨论 → 方案 → 结构 → 推荐。")
    mainline_parser.add_argument("prompt", nargs="*")
    mainline_parser.add_argument("--project")
    mainline_parser.add_argument("--model", help="临时使用 provider:model。")
    mainline_parser.add_argument("--max-tokens", type=int)
    mainline_parser.add_argument("--max-steps", type=int, default=6)
    mainline_parser.add_argument("--resume", action="store_true", help="续接最近或指定 session。")
    mainline_parser.add_argument("--session-id", help="指定要续接/写入的 session id。")
    mainline_parser.add_argument("--task-plan", action="store_true", help="把本轮输入转成结构化 DFT 任务，不普通聊天。")
    mainline_parser.add_argument("--task-run", action="store_true", help="把本轮输入转成 DFT 任务并执行 dry-run。")
    mainline_parser.add_argument("--material")
    mainline_parser.add_argument("--structure-path")
    mainline_parser.add_argument("--slab-path")
    mainline_parser.add_argument("--adsorbate")
    mainline_parser.add_argument("--preferred-site")
    mainline_parser.add_argument("--preferred-orientation")
    mainline_parser.add_argument("--task-type")
    mainline_parser.add_argument("--submit-profile")
    mainline_parser.add_argument("--model-spec-path")
    mainline_parser.add_argument("--step2-manifest-path")
    mainline_parser.add_argument("--candidate-id")
    mainline_parser.add_argument("--planner", choices=["rule", "auto"], default="rule")
    mainline_parser.add_argument("--focus")
    mainline_parser.set_defaults(func=handle_mainline)

    run_parser = sub.add_parser("run", add_help=False, help="转入 DFT 主线 run 子命令。")
    run_parser.add_argument("dft_args", nargs=argparse.REMAINDER)
    run_parser.set_defaults(func=handle_run)

    dft_parser = sub.add_parser("dft", add_help=False, help="直接透传到底层 dft CLI。")
    dft_parser.add_argument("dft_args", nargs=argparse.REMAINDER)
    dft_parser.set_defaults(func=handle_dft)

    structure_parser = sub.add_parser("structure", help="结构 IO 工具。")
    structure_sub = structure_parser.add_subparsers(dest="structure_command")
    structure_convert = structure_sub.add_parser("convert", help="结构格式转换，例如 .xsd -> POSCAR 或 POSCAR -> .xsd。")
    structure_convert.add_argument("input")
    structure_convert.add_argument("output")
    structure_convert.add_argument("--fmt")
    structure_convert.set_defaults(func=handle_structure_convert)
    structure_tools = structure_sub.add_parser("tools", help="列出 AETHER-DFT 当前暴露的结构/DFT 工具。")
    structure_tools.set_defaults(func=handle_structure_tools)

    context_parser = sub.add_parser("context", help="Codex/Claude Code-like 上下文快照。")
    context_sub = context_parser.add_subparsers(dest="context_command")
    context_snapshot = context_sub.add_parser("snapshot", help="生成当前模型、prompt、项目状态与入口说明快照。")
    context_snapshot.add_argument("--project")
    context_snapshot.set_defaults(func=handle_context_snapshot)

    harness_parser = sub.add_parser("harness", help="运行时 harness / preflight 检查。")
    harness_sub = harness_parser.add_subparsers(dest="harness_command")
    harness_preflight = harness_sub.add_parser("preflight", help="检查 prompt/config/依赖/主线入口。")
    harness_preflight.set_defaults(func=handle_harness_preflight)

    cluster_parser = sub.add_parser("cluster", help="SSH/SLURM 集群配置、探测与导入。")
    cluster_sub = cluster_parser.add_subparsers(dest="cluster_command")
    cluster_import = cluster_sub.add_parser("import-ssh-config", help="复制本机 SSH config 到项目 .secrets，并设置默认集群 alias。")
    cluster_import.add_argument("--source", default=str(Path.home() / ".ssh" / "config"))
    cluster_import.add_argument("--alias", default="szhang")
    cluster_import.add_argument("--remote-base-dir", help="远程 run 根目录，默认 /home/<user>/aether-dft-runs。")
    cluster_import.set_defaults(func=handle_cluster_import)
    cluster_config = cluster_sub.add_parser("config", help="显示当前集群配置摘要，不暴露私钥/API key。")
    cluster_config.set_defaults(func=handle_cluster_config)
    cluster_probe = cluster_sub.add_parser("probe", help="真实 SSH 探测集群连通性和 sbatch/squeue/vasp_std。")
    cluster_probe.set_defaults(func=handle_cluster_probe)

    ssh_parser = sub.add_parser("ssh", help="简写：真实 SSH 探测集群。")
    ssh_parser.set_defaults(func=handle_cluster_probe)

    agent_parser = sub.add_parser("agent", help="让 qwen/OpenAI-compatible 模型通过工具调用 AETHER-DFT/集群。")
    agent_parser.add_argument("prompt", nargs="+")
    agent_parser.add_argument("--project")
    agent_parser.add_argument("--model", help="临时使用 provider:model，例如 deepseek:deepseek-v4-pro。")
    agent_parser.add_argument("--max-tokens", type=int)
    agent_parser.add_argument("--max-steps", type=int, default=6)
    agent_parser.add_argument(
        "--allow-cluster-submit",
        action="store_true",
        help="允许模型调用 adsorption_workflow_remote_submit 真正远程提交；不加则只会 blocked。",
    )
    agent_parser.set_defaults(func=handle_agent_run)

    ask_parser = sub.add_parser("ask", help="简写：让当前模型通过工具回答/操作。")
    ask_parser.add_argument("prompt", nargs="+")
    ask_parser.add_argument("--project")
    ask_parser.add_argument("--model", help="临时使用 provider:model，例如 deepseek:deepseek-v4-pro。")
    ask_parser.add_argument("--max-tokens", type=int)
    ask_parser.add_argument("--max-steps", type=int, default=6)
    ask_parser.add_argument("--allow-cluster-submit", action="store_true")
    ask_parser.set_defaults(func=handle_agent_run)

    for command_name, help_text in [
        ("status", "简写：查看吸附 workflow 状态。"),
        ("monitor", "简写：监控吸附 workflow 远程作业。"),
        ("fetch", "简写：拉取远程 VASP 输出。"),
        ("analyze", "简写：解析并汇总吸附能。"),
        ("submit", "简写：远程提交吸附 workflow；必须加 --yes。"),
    ]:
        short_parser = sub.add_parser(command_name, help=help_text)
        short_parser.add_argument("--run-root", required=True)
        if command_name == "submit":
            short_parser.add_argument("--yes", action="store_true", help="确认真实远程 sbatch 提交。")
        short_parser.set_defaults(func=handle_workflow_short, short_command=command_name)

    explain_parser = sub.add_parser("explain", help="对已有 DFT run 调用 dft_tools explain 桥。")
    explain_parser.add_argument("--run-id")
    explain_parser.add_argument("--run-root")
    explain_parser.add_argument("--base-url")
    explain_parser.add_argument("--no-kb-ingest", action="store_true")
    explain_parser.set_defaults(func=handle_explain)

    return parser


def main(argv: list[str] | None = None) -> int:
    ensure_console_utf8()
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if not raw_args:
        if hasattr(sys.stdin, "isatty") and sys.stdin.isatty():
            return handle_chat(
                argparse.Namespace(
                    prompt=[],
                    project=None,
                    model=None,
                    max_tokens=None,
                    max_steps=6,
                    resume=True,
                    session_id=None,
                    task_plan=False,
                    task_run=False,
                    material=None,
                    structure_path=None,
                    slab_path=None,
                    adsorbate=None,
                    preferred_site=None,
                    preferred_orientation=None,
                    task_type=None,
                    submit_profile=None,
                    planner="rule",
                )
            )
        print_quick_start()
        return 0
    if raw_args and raw_args[0] == "run":
        return handle_run(argparse.Namespace(dft_args=raw_args[1:]))
    if raw_args and raw_args[0] == "dft":
        return handle_dft(argparse.Namespace(dft_args=raw_args[1:]))
    parser = build_parser()
    args = parser.parse_args(raw_args)
    handler = getattr(args, "func", None)
    if handler is None:
        parser.print_help()
        return 0
    try:
        return int(handler(args))
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as exc:
        print(f"错误: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
