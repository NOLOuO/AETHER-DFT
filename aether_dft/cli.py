from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
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
    normalize_model_id,
    resolve_effective_model_id,
    set_default_model,
    split_model_id,
)
from .permissions import get_permission_mode, permission_mode_label, set_permission_mode
from .project_state import append_progress, init_project, list_projects, load_project, project_paths, read_project_context

PROGRAM_NAME = "AETHER-DFT"
PROGRAM_COMMAND = "aether"
TOP_LEVEL_COMMANDS = {
    "adsorption",
    "agent",
    "analyze",
    "ask",
    "chat",
    "cluster",
    "context",
    "demo",
    "dft",
    "doctor",
    "explain",
    "fetch",
    "harness",
    "kb",
    "mainline",
    "model",
    "models",
    "monitor",
    "outcar",
    "preload",
    "project",
    "recommend",
    "run",
    "session",
    "ssh",
    "status",
    "structure",
    "submit",
    "task",
    "tools",
}


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
    for stream_name in ("stdin", "stdout", "stderr"):
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


def active_model_id(args: argparse.Namespace | None = None) -> str:
    raw = getattr(args, "model", None) if args is not None else None
    if raw:
        return normalize_model_id(str(raw))
    return program_model_id()


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
    print("  aether                         # 进入持续交互式科研合伙人")
    print("  aether \"帮我看下现在该做什么\"  # 自然语言单轮；模型自行调用工具")
    print("  aether chat --resume           # 续接最近 session")
    print("  aether chat --model qwen       # 进入交互并切换模型")
    print("  aether mainline --resume       # 显式进入科研主线入口")
    print("  aether model current")
    print("  aether project list")
    print("  aether recommend --project <slug>")
    print("  aether preload --project <slug>")
    print("  aether outcar find --limit 5")
    print("  aether outcar analyze --latest --project <slug> --write-learning")
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


def print_chat_home(*, session_id: str, project: str | None = None, model_id: str | None = None) -> None:
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
    line(f"Model: {Colors.YELLOW}{model_id or program_model_id()}{Colors.RESET}")
    context_window = program_context_window()
    if context_window:
        line(f"Context: {context_window:,} tokens")
    line(f"Permission: {Colors.BLUE}{permission_mode_label()}{Colors.RESET}")
    line(f"Session: {session_id}")
    line(f"Project: {project or 'none'}")
    line("Preload: project/session/research injected each turn")
    print(f"{Colors.DIM}└{'─' * box_width}┘{Colors.RESET}")


def print_chat_shortcuts() -> None:
    print("直接输入自然语言即可；模型会自己判断是否需要调用工具。")
    print("输入 / 打开命令面板；也可直接用 /model、/project、/resume、/exit。")


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
    print(f"  {Colors.GREEN}/{Colors.RESET}            打开命令面板")
    print(f"  {Colors.GREEN}/status{Colors.RESET}      当前 session/model/permission")
    print(f"  {Colors.GREEN}/sessions{Colors.RESET}    最近会话列表")
    print(f"  {Colors.GREEN}/new{Colors.RESET}         新开当前项目的 session")
    print(f"  {Colors.GREEN}/resume{Colors.RESET}      打开会话续接选择器")
    print(f"  {Colors.GREEN}/preload{Colors.RESET}     模型本轮会预加载哪些设定")
    print(f"  {Colors.GREEN}/context{Colors.RESET}     当前 1M context budget 与压缩状态")
    print(f"  {Colors.GREEN}/model{Colors.RESET}       打开模型选择器")
    print(f"  {Colors.GREEN}/permission{Colors.RESET}  打开权限模式选择器")
    print(f"  {Colors.GREEN}/project{Colors.RESET}     打开项目选择器")
    print(f"  {Colors.GREEN}/recommend{Colors.RESET}   推荐下一步科研任务")
    print(f"  {Colors.GREEN}/clear{Colors.RESET}       清屏")
    print(f"  {Colors.GREEN}/exit{Colors.RESET}        退出")
    print(f"{Colors.DIM}{'─' * 44}{Colors.RESET}\n")


CHAT_COMMAND_PALETTE: list[tuple[str, str]] = [
    ("/model", "切换模型"),
    ("/project", "切换 research 课题项目"),
    ("/resume", "切换当前项目内的对话"),
    ("/new", "新开当前项目会话"),
    ("/status", "查看当前 session/model/project/permission"),
    ("/sessions", "列出当前 scope 的最近会话"),
    ("/permission", "切换权限模式"),
    ("/context", "查看上下文预算和压缩状态"),
    ("/preload", "查看本轮预加载设定"),
    ("/recommend", "根据项目状态推荐下一步"),
    ("/help", "查看帮助"),
    ("/clear", "清屏"),
    ("/exit", "退出"),
]


def handle_chat_command_palette() -> str | None:
    print(f"{Colors.CYAN}slash commands{Colors.RESET}:")
    for index, (command, description) in enumerate(CHAT_COMMAND_PALETTE, start=1):
        print(f"  {index}. {Colors.GREEN}{command:<11}{Colors.RESET} {description}")
    if not (hasattr(sys.stdin, "isatty") and sys.stdin.isatty()):
        print("非交互 stdin：请直接输入完整 slash command，例如 /model。")
        return None
    try:
        choice = input("command> ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    if not choice:
        print("command cancelled")
        return None
    if choice.isdigit() and 1 <= int(choice) <= len(CHAT_COMMAND_PALETTE):
        return CHAT_COMMAND_PALETTE[int(choice) - 1][0]
    if not choice.startswith("/"):
        choice = "/" + choice
    return choice


def print_chat_status(*, session_store: Any, session_id: str, project: str | None, args: argparse.Namespace | None = None) -> None:
    state = session_store.load_state(session_id)
    print_json(
        {
            "program": PROGRAM_NAME,
            "version": __version__,
            "model": active_model_id(args),
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


def print_chat_sessions(*, session_store: Any, project: str | None, limit: int = 10) -> None:
    sessions = session_store.list_sessions(project=project, limit=limit)
    if not sessions:
        print("没有可续接的 session。")
        return
    print(f"{Colors.CYAN}recent sessions{Colors.RESET}:")
    for item in sessions:
        project_label = item.project or "none"
        first = _shorten_inline(item.first_prompt, limit=56) or "empty"
        print(f"- {item.session_id}  project={project_label} turns={item.turn_count} updated={item.updated_at}")
        print(f"  {Colors.DIM}{first}{Colors.RESET}")


def print_resume_preview(payload: dict[str, Any]) -> None:
    state = payload.get("state") or {}
    print(
        f"{Colors.GREEN}resumed{Colors.RESET}: "
        f"{payload.get('session_id')} project={state.get('project') or 'none'} turns={state.get('turn_count') or 0}"
    )
    recent_turns = payload.get("recent_turns") or []
    if recent_turns:
        print("最近对话：")
        for turn in recent_turns[-3:]:
            record = turn.get("record", {})
            print(f"- user: {_shorten_inline(record.get('prompt'), limit=90)}")
            print(f"  assistant: {_shorten_inline(record.get('response'), limit=90)}")


def _resumable_sessions(session_store: Any, *, project: str | None, current_session_id: str, limit: int = 20) -> list[Any]:
    return [item for item in session_store.list_sessions(project=project, limit=limit) if item.session_id != current_session_id]


def _session_matches_query(item: Any, query: str) -> bool:
    text = " ".join(
        [
            str(getattr(item, "session_id", "") or ""),
            str(getattr(item, "project", "") or ""),
            str(getattr(item, "first_prompt", "") or ""),
            str(getattr(item, "last_response", "") or ""),
        ]
    ).lower()
    return query.lower() in text


def handle_chat_resume_command(line: str, args: argparse.Namespace, session_store: Any, current_session_id: str) -> str:
    raw = line[len("/resume") :].strip()
    scoped_sessions = _resumable_sessions(session_store, project=args.project, current_session_id=current_session_id, limit=20)
    if raw == "":
        sessions = scoped_sessions[:10]
        if not sessions:
            print("没有可续接的 session。")
            return current_session_id
        print(f"{Colors.CYAN}resume session{Colors.RESET}:")
        for index, item in enumerate(sessions, start=1):
            first = _shorten_inline(item.first_prompt, limit=64) or "empty"
            print(f"  {index}. {item.session_id}  project={item.project or 'none'} turns={item.turn_count}")
            print(f"     {Colors.DIM}{first}{Colors.RESET}")
        if not (hasattr(sys.stdin, "isatty") and sys.stdin.isatty()):
            print("非交互 stdin：请使用 /resume latest 或 /resume <session_id>。")
            return current_session_id
        try:
            choice = input("resume> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return current_session_id
        if not choice:
            print("session unchanged")
            return current_session_id
        if choice.isdigit() and 1 <= int(choice) <= len(sessions):
            raw = sessions[int(choice) - 1].session_id
        else:
            raw = choice
    elif raw == "latest":
        if scoped_sessions:
            raw = scoped_sessions[0].session_id
        else:
            print("没有可续接的 session。")
            return current_session_id
    elif not raw.startswith("session_"):
        matches = [item for item in scoped_sessions if _session_matches_query(item, raw)]
        if len(matches) == 1:
            raw = matches[0].session_id
        elif len(matches) > 1:
            print(f"{Colors.CYAN}resume matches for {raw!r}{Colors.RESET}:")
            for index, item in enumerate(matches[:10], start=1):
                first = _shorten_inline(item.first_prompt, limit=64) or "empty"
                print(f"  {index}. {item.session_id}  project={item.project or 'none'} turns={item.turn_count}")
                print(f"     {Colors.DIM}{first}{Colors.RESET}")
            if not (hasattr(sys.stdin, "isatty") and sys.stdin.isatty()):
                print("非交互 stdin：请使用 /resume <session_id>。")
                return current_session_id
            try:
                choice = input("resume> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return current_session_id
            if not choice:
                print("session unchanged")
                return current_session_id
            if choice.isdigit() and 1 <= int(choice) <= len(matches[:10]):
                raw = matches[int(choice) - 1].session_id
            else:
                raw = choice
        else:
            print(f"没有找到匹配 {raw!r} 的 session。用 /resume 打开选择器。")
            return current_session_id
    try:
        payload = session_store.resume_payload(session_id=raw)
    except Exception:
        payload = {"status": "missing"}
    if payload.get("status") != "ok":
        print("没有找到可续接的 session。用 /sessions 查看最近会话。")
        return current_session_id
    state = payload.get("state") or {}
    resumed_project = state.get("project")
    if resumed_project:
        args.project = str(resumed_project)
    print_resume_preview(payload)
    return str(payload["session_id"])


def handle_chat_project_command(line: str, args: argparse.Namespace, session_store: Any, current_session_id: str) -> str:
    raw = line[len("/project") :].strip()
    if raw in {"current", "show"}:
        if not args.project:
            print("当前没有绑定项目。输入 /project 打开项目选择器。")
            return current_session_id
        print_json({"project": load_project(args.project), "context": read_project_context(args.project)})
        return current_session_id
    if raw in {"", "list"}:
        projects = list_projects()
        if not projects:
            print("还没有项目。可以继续自然语言说明课题，或先用 project init 创建。")
            return current_session_id
        print(f"{Colors.CYAN}select project{Colors.RESET} (current: {Colors.YELLOW}{args.project or 'none'}{Colors.RESET})")
        for index, item in enumerate(projects, start=1):
            slug = str(item.get("slug") or item.get("name") or "")
            title = str(item.get("title") or item.get("name") or slug)
            marker = "*" if slug == args.project else " "
            print(f"  {index}. {marker} {slug}  {Colors.DIM}{_shorten_inline(title, limit=60)}{Colors.RESET}")
        if raw == "list" or not (hasattr(sys.stdin, "isatty") and sys.stdin.isatty()):
            print("非交互 stdin：请使用 /project <slug>；交互式输入 /project 后可选编号。")
            return current_session_id
        try:
            choice = input("project> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return current_session_id
        if not choice:
            print("project unchanged")
            return current_session_id
        if choice.isdigit() and 1 <= int(choice) <= len(projects):
            raw = str(projects[int(choice) - 1].get("slug") or projects[int(choice) - 1].get("name") or "")
        else:
            raw = choice
    if raw in {"none", "clear", "unbind"}:
        args.project = None
        session_id = session_store.start_session(project=None)
        print(f"{Colors.GREEN}project cleared{Colors.RESET}; new session: {session_id}")
        return session_id
    try:
        project = load_project(raw)
    except Exception as exc:
        print(f"{Colors.RED}project switch failed{Colors.RESET}: {exc}")
        return current_session_id
    slug = str(project.get("slug") or raw)
    args.project = slug
    session_id = session_store.latest_session_id(project=slug) or session_store.start_session(project=slug)
    print(f"{Colors.GREEN}project switched{Colors.RESET}: {Colors.YELLOW}{slug}{Colors.RESET}")
    payload = session_store.resume_payload(session_id=session_id)
    if payload.get("status") == "ok":
        print_resume_preview(payload)
    return session_id


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
    turn_started = {"t": 0.0}
    model_step_started: dict[int, float] = {}
    tool_started: dict[tuple[int, str], float] = {}

    def elapsed_since(start: float | None) -> str:
        if not start:
            return ""
        return f" ({time.perf_counter() - start:.1f}s)"

    def _printer(event: dict[str, Any]) -> None:
        kind = str(event.get("event") or "")
        if kind == "turn_start":
            turn_started["t"] = time.perf_counter()
            print(f"{Colors.DIM}thinking with {event.get('model_id') or program_model_id()}...{Colors.RESET}")
        elif kind == "model_request":
            step = int(event.get("step") or 0)
            model_step_started[step] = time.perf_counter()
            print(f"{Colors.DIM}↻ model step {event.get('step')}/{event.get('max_steps')}{Colors.RESET}")
        elif kind == "tool_start":
            step = int(event.get("step") or 0)
            name = str(event.get("name") or "")
            tool_started[(step, name)] = time.perf_counter()
            step_elapsed = elapsed_since(model_step_started.get(step))
            args = _shorten_inline(event.get("arguments"), limit=180)
            print(f"{Colors.BLUE}↳ tool{Colors.RESET} {event.get('name')}{Colors.DIM}{step_elapsed} {args}{Colors.RESET}")
        elif kind == "tool_finish":
            status = event.get("status") or "done"
            step = int(event.get("step") or 0)
            name = str(event.get("name") or "")
            tool_elapsed = elapsed_since(tool_started.get((step, name)))
            persisted = event.get("persisted_output_path")
            suffix = f" {Colors.DIM}{persisted}{Colors.RESET}" if persisted else ""
            print(f"{Colors.GREEN}✓ tool{Colors.RESET} {event.get('name')} status={status}{Colors.DIM}{tool_elapsed}{Colors.RESET}{suffix}")
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
        elif kind == "turn_interrupted":
            print(
                f"{Colors.YELLOW}interrupted{Colors.RESET}: partial trace will be saved"
                f"{Colors.DIM}{elapsed_since(turn_started.get('t'))}{Colors.RESET}"
            )

    return _printer


def make_stream_printer() -> tuple[Any, dict[str, bool]]:
    state = {"printed": False}

    def _printer(event: dict[str, Any]) -> None:
        if str(event.get("type") or "") != "content_delta":
            return
        delta = str(event.get("delta") or "")
        if not delta:
            return
        if not state["printed"]:
            print(f"{Colors.CYAN}assistant>{Colors.RESET} ", end="", flush=True)
        print(delta, end="", flush=True)
        state["printed"] = True

    return _printer, state


def print_streamed_or_final_response(record: dict[str, Any], stream_state: dict[str, bool]) -> None:
    if stream_state.get("printed"):
        print()
        return
    print(record["response"])


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
    current = active_model_id(args)
    catalog = load_model_catalog(Path.cwd())
    if raw in {"current", "list"}:
        print(f"{Colors.CYAN}current model{Colors.RESET}: {Colors.YELLOW}{current}{Colors.RESET}")
        print(format_model_table(catalog, current))
        return
    if raw == "":
        models = sorted(catalog)
        print(f"{Colors.CYAN}select model{Colors.RESET} (current: {Colors.YELLOW}{current}{Colors.RESET})")
        for index, model_id in enumerate(models, start=1):
            item = catalog[model_id]
            marker = "*" if model_id == current else " "
            status = "available" if item.available else f"missing key: {item.api_key_env}"
            print(f"  {index}. {marker} {model_id}  [{status}]")
        if not (hasattr(sys.stdin, "isatty") and sys.stdin.isatty()):
            print("非交互 stdin：输入 /model <编号或别名> 可直接切换。")
            return
        try:
            choice = input("model> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not choice:
            print("model unchanged")
            return
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            raw = models[int(choice) - 1]
        else:
            raw = choice
    if raw.startswith("set "):
        raw = raw[4:].strip()
    try:
        preferences = set_default_model(raw)
        normalized = str(preferences["global_default_model_id"])
    except Exception as exc:
        print(f"{Colors.RED}model switch failed{Colors.RESET}: {exc}")
        return
    args.model = normalized
    print(f"{Colors.GREEN}model switched{Colors.RESET}: {Colors.YELLOW}{normalized}{Colors.RESET}")


def handle_chat_permission_command(line: str) -> None:
    raw = line[len("/permission") :].strip()
    if not raw:
        current = get_permission_mode()
        options = [("dev", "完全开发：低风险可逆动作自动推进"), ("ask", "需要用户同意：写入/提交等副作用前确认")]
        print(f"{Colors.CYAN}select permission{Colors.RESET} (current: {Colors.YELLOW}{current} / {permission_mode_label()}{Colors.RESET})")
        for index, (mode, label) in enumerate(options, start=1):
            marker = "*" if mode == current else " "
            print(f"  {index}. {marker} {mode}  {Colors.DIM}{label}{Colors.RESET}")
        if not (hasattr(sys.stdin, "isatty") and sys.stdin.isatty()):
            print("非交互 stdin：请使用 /permission dev 或 /permission ask。")
            return
        try:
            choice = input("permission> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not choice:
            print("permission unchanged")
            return
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            raw = options[int(choice) - 1][0]
        else:
            raw = choice
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

        from .agent import run_agent_once

        stream_printer, stream_state = make_stream_printer()
        record = run_agent_once(
            line,
            max_tokens=1000,
            max_steps=4,
            allow_cluster_submit=False,
            progress_callback=make_chat_progress_printer(),
            permission_prompt_callback=make_permission_prompt_callback(),
            stream_callback=stream_printer,
        )
        print_streamed_or_final_response(record, stream_state)


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
        print("\nBuilt-ins are project-fit defaults; add more OpenAI-compatible models via config/model_providers.json.")
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


def handle_model_smoke(args: argparse.Namespace) -> int:
    from .agent import run_agent_once

    model_id = args.model or resolve_effective_model_id()
    project = args.project or "model-smoke-demo"
    prompt = (
        f"这是后端切换 smoke：必须先调用 project_state_read 读取 project={project}，"
        "然后用一句话总结。不要调用其他工具。"
    )
    record = run_agent_once(
        prompt,
        project=project,
        model_id=model_id,
        max_steps=args.max_steps,
        max_tokens=args.max_tokens,
        permission_mode="dev",
    )
    tool_names = [str(item.get("name") or "") for item in record.get("tool_executions", [])]
    payload = {
        "status": "ok" if "project_state_read" in tool_names else "failed",
        "model_id": model_id,
        "project": project,
        "required_tool": "project_state_read",
        "tool_names": tool_names,
        "finish_reason": record.get("finish_reason"),
        "response": record.get("response"),
        "record_path": record.get("record_path"),
    }
    print_json(payload)
    return 0 if payload["status"] == "ok" else 1


def handle_preload(args: argparse.Namespace) -> int:
    from .preload import build_preload_summary, format_preload_summary

    summary = build_preload_summary(project=args.project, probe_cluster=args.probe_cluster)
    if args.json:
        print_json(summary.payload)
    else:
        print(format_preload_summary(summary))
    return 0 if summary.status == "ok" else 1


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
            resumed_project = (payload.get("state") or {}).get("project")
            if not args.project and resumed_project:
                args.project = str(resumed_project)
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
        stream_printer, stream_state = make_stream_printer()
        try:
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
                stream_callback=stream_printer,
            )
        except Exception as exc:
            print(f"{Colors.RED}模型调用失败{Colors.RESET}: {_shorten_inline(str(exc), limit=360)}")
            print(f"{Colors.DIM}可以稍后重试，或临时切换同类 OpenAI-compatible 后端：aether chat --model qwen <你的问题>{Colors.RESET}")
            return 1
        print_streamed_or_final_response(record, stream_state)
        print()
        print_turn_footer(record)
        next_steps = (record.get("progress") or {}).get("next_steps") or []
        if next_steps:
            print(f"[next] {next_steps[0]}")
        return 0

    if not session_id:
        session_id = session_store.start_session(project=args.project)
    print_chat_home(session_id=session_id, project=args.project, model_id=active_model_id(args))
    print_chat_shortcuts()
    if args.resume:
        payload = session_store.resume_payload(session_id=session_id, project=args.project)
        if payload["status"] == "ok" and payload["recent_turns"]:
            print_resume_preview(payload)
    while True:
        try:
            model_short = active_model_id(args).split(":", 1)[-1]
            project_short = args.project or "no-project"
            line = input(f"aether[{project_short}|{model_short}]> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if line in {"/exit", "exit", "quit", ":q"}:
            return 0
        if not line:
            continue
        if line in {"/", "/commands"}:
            selected_command = handle_chat_command_palette()
            if not selected_command:
                continue
            line = selected_command
        if line == "/help":
            print_chat_help()
            continue
        if line == "/clear":
            os.system("cls" if os.name == "nt" else "clear")
            print_chat_home(session_id=session_id, project=args.project, model_id=active_model_id(args))
            print_chat_shortcuts()
            continue
        if line == "/status":
            print_chat_status(session_store=session_store, session_id=session_id, project=args.project, args=args)
            continue
        if line == "/sessions":
            print_chat_sessions(session_store=session_store, project=args.project)
            continue
        if line.startswith("/new"):
            raw_project = line[len("/new") :].strip()
            if raw_project:
                args.project = None if raw_project in {"none", "clear", "no-project"} else raw_project
            session_id = session_store.start_session(project=args.project)
            print(f"{Colors.GREEN}new session{Colors.RESET}: {session_id} project={args.project or 'none'}")
            continue
        if line.startswith("/resume"):
            session_id = handle_chat_resume_command(line, args, session_store, session_id)
            continue
        if line == "/preload":
            handle_preload(argparse.Namespace(project=args.project, probe_cluster=False, json=False))
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
        if line.startswith("/project"):
            session_id = handle_chat_project_command(line, args, session_store, session_id)
            continue
        if line.startswith("/recommend"):
            from .recommendations import recommend_next_tasks

            focus = line[len("/recommend") :].strip() or None
            print_json({"recommendations": recommend_next_tasks(args.project, focus=focus)})
            continue
        stream_printer, stream_state = make_stream_printer()
        try:
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
                stream_callback=stream_printer,
            )
        except Exception as exc:
            print(f"{Colors.RED}模型调用失败{Colors.RESET}: {_shorten_inline(str(exc), limit=360)}")
            print(f"{Colors.DIM}本 session 仍然保留；可 /model qwen 切换后继续，或稍后重试。{Colors.RESET}")
            continue
        print_streamed_or_final_response(record, stream_state)
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


def _print_outcar_table(outcars: list[dict[str, Any]]) -> None:
    if not outcars:
        print("没有找到 OUTCAR。")
        return
    print("最近 OUTCAR：")
    print()
    print(f"{'NO':<4} {'MODIFIED':<17} {'SIZE':>10}  PATH")
    for index, item in enumerate(outcars, start=1):
        size = item.get("size")
        size_text = str(size) if size is not None else "?"
        print(f"{index:<4} {str(item.get('modified') or ''):<17} {size_text:>10}  {item.get('path')}")
    print()
    print("分析最新一个：aether-dft outcar analyze --latest")
    print("分析指定路径：aether-dft outcar analyze /home/.../OUTCAR")


def handle_outcar_find(args: argparse.Namespace) -> int:
    from dft_app.remote import SSHRemoteRunner

    result = SSHRemoteRunner().find_remote_outcars(
        search_root=args.root,
        limit=args.limit,
        max_depth=args.max_depth,
    )
    payload = {"status": result.status, "message": result.message, "details": result.details}
    if args.json:
        print_json(payload)
    else:
        if result.status == "ok":
            _print_outcar_table(result.details.get("outcars") or [])
        else:
            print(f"{result.status}: {result.message}")
    return 0 if result.status == "ok" else 1


def _outcar_cache_dir(remote_outcar_path: str) -> Path:
    from .paths import ensure_runtime_dir

    parent_name = Path(str(remote_outcar_path).replace("\\", "/")).parent.name or "outcar"
    digest = hashlib.sha1(str(remote_outcar_path).encode("utf-8")).hexdigest()[:10]
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", parent_name).strip("_") or "outcar"
    return ensure_runtime_dir("remote_outcar_analysis", f"{slug}_{digest}")


def _write_outcar_learning(*, project: str, remote_outcar_path: str, local_target_dir: str, interpretation: dict[str, Any]) -> dict[str, Any]:
    from aether_dft.runtime_harness.tool_registry import ToolRegistry

    frequency = interpretation.get("frequency") or {}
    energy = interpretation.get("energy") or {}
    content = "\n".join(
        [
            f"# OUTCAR analysis: {Path(str(remote_outcar_path)).parent.name or 'remote run'}",
            "",
            f"Remote OUTCAR: `{remote_outcar_path}`",
            f"Local evidence copy: `{local_target_dir}`",
            "",
            "## Evidence",
            "",
            f"- Verdict: `{interpretation.get('verdict')}`",
            f"- Headline: {interpretation.get('headline')}",
            f"- Last TOTEN: `{energy.get('last_toten_ev')}` eV",
            f"- Frequency detected: `{frequency.get('detected', False)}`",
            f"- Real/imaginary modes: `{frequency.get('real_mode_count')}` / `{frequency.get('imaginary_mode_count')}`",
            "",
            "## Warnings",
            "",
            *[f"- {item}" for item in (interpretation.get("warnings") or ["none"])],
            "",
            "## Suggested next checks",
            "",
            *[f"- {item}" for item in (interpretation.get("suggestions") or ["none"])],
        ]
    )
    return ToolRegistry().run_tool(
        "research_learning_capture",
        {
            "project": project,
            "title": f"OUTCAR analysis {Path(str(remote_outcar_path)).parent.name or 'remote run'}",
            "content": content,
            "tags": ["OUTCAR", "analysis", "remote"],
        },
    )


def handle_outcar_analyze(args: argparse.Namespace) -> int:
    from aether_dft.result_insight import interpret_result
    from dft_app.remote import SSHRemoteRunner

    runner = SSHRemoteRunner()
    remote_outcar_path = str(args.remote_outcar or "").strip()
    if not remote_outcar_path:
        found = runner.find_remote_outcars(search_root=args.root, limit=1, max_depth=args.max_depth)
        if found.status != "ok" or not found.details.get("outcars"):
            print_json({"status": found.status, "message": found.message, "details": found.details})
            return 1
        remote_outcar_path = str(found.details["outcars"][0]["path"])

    local_target = Path(args.output_dir) if args.output_dir else _outcar_cache_dir(remote_outcar_path)
    pulled = runner.pull_remote_outcar_context(remote_outcar_path, local_target)
    if pulled.status != "synced":
        print_json({"status": pulled.status, "message": pulled.message, "details": pulled.details})
        return 1
    interpretation = interpret_result(local_target)
    learning = None
    if args.write_learning:
        if not args.project:
            print_json(
                {
                    "status": "failed",
                    "message": "--write-learning 需要同时提供 --project。",
                    "remote_outcar_path": remote_outcar_path,
                    "local_target_dir": str(local_target),
                    "interpretation": interpretation,
                }
            )
            return 1
        learning = _write_outcar_learning(
            project=args.project,
            remote_outcar_path=remote_outcar_path,
            local_target_dir=str(local_target),
            interpretation=interpretation,
        )

    payload = {
        "status": "ok",
        "remote_outcar_path": remote_outcar_path,
        "local_target_dir": str(local_target),
        "pulled": {"status": pulled.status, "message": pulled.message, "details": pulled.details},
        "interpretation": interpretation,
        "learning": learning,
    }
    if args.json:
        print_json(payload)
    else:
        print(f"OUTCAR: {remote_outcar_path}")
        print(f"local : {local_target}")
        print(f"verdict: {interpretation.get('verdict')}")
        print(f"headline: {interpretation.get('headline')}")
        energy = interpretation.get("energy") or {}
        print(f"last TOTEN: {energy.get('last_toten_ev')} eV")
        frequency = interpretation.get("frequency") or {}
        if frequency.get("detected"):
            print(
                "frequency: "
                f"real={frequency.get('real_mode_count')} "
                f"imaginary={frequency.get('imaginary_mode_count')} "
                f"min_real_THz={frequency.get('min_real_thz')}"
            )
        warnings = interpretation.get("warnings") or []
        if warnings:
            print("warnings:")
            for warning in warnings:
                print(f"  - {warning}")
        if learning:
            result = learning.get("result", {}) if isinstance(learning, dict) else {}
            print(f"learning: {result.get('learning_path')}")
    return 0


def handle_agent_run(args: argparse.Namespace) -> int:
    from .agent import run_agent_once

    prompt = " ".join(args.prompt).strip()
    stream_printer, stream_state = make_stream_printer()
    record = run_agent_once(
        prompt,
        project=args.project,
        model_id=args.model,
        max_tokens=args.max_tokens,
        max_steps=args.max_steps,
        allow_cluster_submit=args.allow_cluster_submit,
        progress_callback=make_chat_progress_printer(),
        permission_prompt_callback=make_permission_prompt_callback(),
        stream_callback=stream_printer,
    )
    print_streamed_or_final_response(record, stream_state)
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

    preload_parser = sub.add_parser("preload", help="显示启动时会预加载给模型的项目/会话/research/工具设定。")
    preload_parser.add_argument("--project", help="要绑定的 research/project slug，例如 MCH-Pt-Br。")
    preload_parser.add_argument("--probe-cluster", action="store_true", help="额外做一次真实 SSH 集群探测；默认不联网/不连集群。")
    preload_parser.add_argument("--json", action="store_true")
    preload_parser.set_defaults(func=handle_preload)

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
    model_smoke = model_sub.add_parser("smoke", help="真实调用当前/指定模型，验证工具调用后端。")
    model_smoke.add_argument("--model", help="临时使用 provider:model；默认当前模型。")
    model_smoke.add_argument("--project", default="model-smoke-demo")
    model_smoke.add_argument("--max-steps", type=int, default=4)
    model_smoke.add_argument("--max-tokens", type=int, default=1200)
    model_smoke.set_defaults(func=handle_model_smoke)

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
    chat_parser.add_argument("--model", help="临时使用模型；支持 qwen/deepseek 这类唯一别名或完整 provider:model。")
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
    mainline_parser.add_argument("--model", help="临时使用模型；支持 qwen/deepseek 这类唯一别名或完整 provider:model。")
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

    outcar_parser = sub.add_parser("outcar", help="查找/拉回/解释集群 OUTCAR。")
    outcar_sub = outcar_parser.add_subparsers(dest="outcar_command")
    outcar_find = outcar_sub.add_parser("find", help="只读查找集群最近 OUTCAR。")
    outcar_find.add_argument("--root", default="~/research", help="远端搜索根目录，默认 ~/research。")
    outcar_find.add_argument("--limit", type=int, default=20)
    outcar_find.add_argument("--max-depth", type=int, default=8)
    outcar_find.add_argument("--json", action="store_true")
    outcar_find.set_defaults(func=handle_outcar_find)
    outcar_analyze = outcar_sub.add_parser("analyze", help="拉回并解释 OUTCAR；不传路径时默认分析最新 OUTCAR。")
    outcar_analyze.add_argument("remote_outcar", nargs="?", help="远端 OUTCAR 绝对路径；省略则使用最新一个。")
    outcar_analyze.add_argument("--latest", action="store_true", help="显式使用最新 OUTCAR（默认行为）。")
    outcar_analyze.add_argument("--root", default="~/research", help="未指定路径时的远端搜索根目录。")
    outcar_analyze.add_argument("--max-depth", type=int, default=8)
    outcar_analyze.add_argument("--output-dir", help="本地证据保存目录；默认 .aether/runtime/remote_outcar_analysis/<slug>。")
    outcar_analyze.add_argument("--project", help="配合 --write-learning，把解释写回 research/<project>/Learning。")
    outcar_analyze.add_argument("--write-learning", action="store_true", help="把解释写回项目 Learning。")
    outcar_analyze.add_argument("--json", action="store_true")
    outcar_analyze.set_defaults(func=handle_outcar_analyze)

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
    if raw_args and raw_args[0] not in TOP_LEVEL_COMMANDS and not raw_args[0].startswith("-"):
        return handle_chat(
            argparse.Namespace(
                prompt=raw_args,
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
                model_spec_path=None,
                step2_manifest_path=None,
                candidate_id=None,
                planner="rule",
            )
        )
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
