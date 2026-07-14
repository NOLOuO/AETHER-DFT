from __future__ import annotations

import argparse
import copy
from datetime import datetime
import hashlib
import importlib.util
import json
import os
import re
import sys
import threading
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
from .paths import ensure_runtime_dir

PROGRAM_NAME = "AETHER-DFT"
PROGRAM_COMMAND = "aether"
AUTO_TURN_MIN_STEPS = 12
AUTO_TURN_MIN_TOKENS = 1400
AUTO_INITIAL_MAX_PASSES = 3
SUPPORTED_PYTHON_MAJOR = 3
SUPPORTED_PYTHON_MINORS = {12, 13}
TOP_LEVEL_COMMANDS = {
    "adsorption",
    "agent",
    "analyze",
    "ask",
    "auto",
    "benchmark",
    "chat",
    "cluster",
    "context",
    "demo",
    "dft",
    "doctor",
    "explain",
    "fetch",
    "followup",
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
    MAGENTA = "\033[35m"


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
    print(f"{Colors.BOLD}{Colors.CYAN}AETHER-DFT{Colors.RESET} v{Colors.GREEN}{__version__}{Colors.RESET}")
    print(f"Model: {Colors.YELLOW}{program_model_id()}{Colors.RESET}")
    print("Role: conversational DFT research partner")
    print()
    print("Start:")
    print("  aether                         # 打开交互式科研合伙人")
    print("  aether \"看看这个课题现在差什么证据\"")
    print("  aether chat --resume           # 明确续接最近 session")
    print()
    print("直接输入自然语言即可；模型会自己判断是否需要调用工具。")
    print("常用 slash command：/model、/project、/resume、/auto、/status、/exit。")
    print()
    print("Research shortcuts:")
    print("  aether auto status --project <slug>")
    print("  aether auto on \"研究目标\" --project <slug>")
    print("  aether outcar analyze --latest --project <slug>")
    print()
    print(f"{Colors.DIM}Advanced: aether chat --help-advanced | aether tools list | aether doctor{Colors.RESET}")


def print_chat_cli_help() -> None:
    print(f"{Colors.BOLD}{Colors.CYAN}aether chat{Colors.RESET} — conversational DFT partner")
    print()
    print("Just type natural language:")
    print("  aether chat --project MCH-Pt-Br")
    print("  aether chat --project MCH-Pt-Br \"看看现在离目标还差什么证据\"")
    print("  aether chat --resume")
    print()
    print("Inside chat:")
    print("  /model     switch DeepSeek/Qwen/other OpenAI-compatible backend")
    print("  /project   switch research project from research/.aether state")
    print("  /resume    choose a conversation in the current project")
    print("  /auto      toggle goal-driven autonomous research")
    print("  /status    current session/model/permission/project")
    print()
    print("Examples:")
    print("  /auto 验证 MCH 在 Br/Pt 上脱氢的最低能路径")
    print("  看看集群上哪些任务在跑，收敛怎么样")
    print("  根据 OUTCAR 判断这个候选是否可以进入下一轮")
    print()
    print(f"{Colors.DIM}Need every internal flag? Use: aether chat --help-advanced{Colors.RESET}")


def print_demo_home(run_root: str | None = None) -> None:
    run_root = run_root or r"runs\task_0a4a1ddd\run_a295c506"
    print(f"{Colors.BOLD}{Colors.CYAN}AETHER-DFT{Colors.RESET} {Colors.DIM}demo session{Colors.RESET}")
    print(
        f"{Colors.DIM}model{Colors.RESET} {Colors.YELLOW}{program_model_id()}{Colors.RESET}  "
        f"{Colors.DIM}workspace{Colors.RESET} {Path.cwd()}"
    )
    print(f"{Colors.DIM}cluster{Colors.RESET} {Colors.BLUE}SSH / SLURM configured{Colors.RESET}")
    print(f"{Colors.DIM}sample run{Colors.RESET} {run_root}")
    print(f"{Colors.DIM}Type {Colors.GREEN}/help{Colors.DIM} for help, {Colors.GREEN}/exit{Colors.DIM} to quit{Colors.RESET}")


def print_chat_home(*, session_id: str, project: str | None = None, model_id: str | None = None) -> None:
    print(f"{Colors.BOLD}{Colors.CYAN}AETHER-DFT{Colors.RESET} {Colors.DIM}v{__version__}{Colors.RESET}")
    print(
        f"{Colors.DIM}model{Colors.RESET} {Colors.YELLOW}{model_id or program_model_id()}{Colors.RESET}  "
        f"{Colors.DIM}project{Colors.RESET} {Colors.MAGENTA}{project or 'none'}{Colors.RESET}"
    )
    print(
        f"{Colors.DIM}session{Colors.RESET} {Colors.BLUE}{session_id}{Colors.RESET}  "
        f"{Colors.DIM}permission{Colors.RESET} {Colors.GREEN}{permission_mode_label()}{Colors.RESET}"
    )
    context_window = program_context_window()
    if context_window:
        print(f"{Colors.DIM}context{Colors.RESET} {context_window:,} tokens")
    print(f"{Colors.DIM}preload{Colors.RESET} project + session + research memory")


def print_chat_shortcuts() -> None:
    print(f"{Colors.DIM}直接输入自然语言；{Colors.GREEN}/{Colors.DIM} 打开命令面板。{Colors.GREEN}/status{Colors.DIM} 查看状态，{Colors.GREEN}/exit{Colors.DIM} 退出。{Colors.RESET}")


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
    print(f"  {Colors.GREEN}/continue{Colors.RESET}    继续上次失败/未完成的用户输入")
    print(f"  {Colors.GREEN}/history{Colors.RESET}     搜索/查看当前 session 对话历史")
    print(f"  {Colors.GREEN}/rename{Colors.RESET}      重命名当前 session")
    print(f"  {Colors.GREEN}/preload{Colors.RESET}     模型本轮会预加载哪些设定")
    print(f"  {Colors.GREEN}/context{Colors.RESET}     当前 1M context budget 与压缩状态")
    print(f"  {Colors.GREEN}/compact{Colors.RESET}     手动压缩旧对话上下文，保留完整 transcript")
    print(f"  {Colors.GREEN}/model{Colors.RESET}       打开模型选择器")
    print(f"  {Colors.GREEN}/permission{Colors.RESET}  打开权限模式选择器")
    print(f"  {Colors.GREEN}/project{Colors.RESET}     打开项目选择器")
    print(f"  {Colors.GREEN}/auto{Colors.RESET}        开关目标驱动自动科研模式")
    print(f"  {Colors.GREEN}/recommend{Colors.RESET}   推荐下一步科研任务")
    print(f"  {Colors.GREEN}/clear{Colors.RESET}       清屏")
    print(f"  {Colors.GREEN}/exit{Colors.RESET}        退出")
    print(f"{Colors.DIM}{'─' * 44}{Colors.RESET}\n")


CHAT_COMMAND_PALETTE: list[tuple[str, str]] = [
    ("/model", "切换模型"),
    ("/project", "切换 research 课题项目"),
    ("/auto", "开关目标驱动自动科研模式"),
    ("/resume", "切换当前项目内的对话"),
    ("/continue", "继续上次失败/未完成输入"),
    ("/history", "搜索/查看当前 session 历史"),
    ("/rename", "重命名当前 session"),
    ("/new", "新开当前项目会话"),
    ("/status", "查看当前 session/model/project/permission"),
    ("/sessions", "列出当前 scope 的最近会话"),
    ("/permission", "切换权限模式"),
    ("/context", "查看上下文预算和压缩状态"),
    ("/compact", "压缩旧对话上下文"),
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


def _chat_status_payload(*, session_store: Any, session_id: str, project: str | None, args: argparse.Namespace | None = None) -> dict[str, Any]:
    state = session_store.load_state(session_id)
    project_ref = None
    if hasattr(session_store, "project_session_reference_path"):
        ref_path = session_store.project_session_reference_path(session_id)
        project_ref = str(ref_path) if ref_path else None
    auto = {}
    try:
        from .auto_mode import auto_mode_status

        auto = auto_mode_status(project=project or state.get("project"), include_due=False).get("state") or {}
    except Exception:
        auto = {}
    return {
        "program": PROGRAM_NAME,
        "version": __version__,
        "model": active_model_id(args),
        "context_window": program_context_window(),
        "permission": {"mode": get_permission_mode(), "label": permission_mode_label()},
        "auto": {
            "enabled": bool(auto.get("enabled")),
            "research_goal": auto.get("research_goal") or "",
            "status": auto.get("status") or "",
            "monitor_interval_hours": auto.get("monitor_interval_hours"),
            "daily_report_time": auto.get("daily_report_time"),
            "allow_cluster_submit": bool(auto.get("allow_cluster_submit")),
        },
        "session": {
            "id": session_id,
            "project": project or state.get("project"),
            "title": state.get("title"),
            "turn_count": state.get("turn_count"),
            "updated_at": state.get("updated_at"),
            "pending_turn": state.get("pending_turn") if isinstance(state.get("pending_turn"), dict) else None,
            "project_session_ref": project_ref,
        },
    }


def print_chat_status(
    *,
    session_store: Any,
    session_id: str,
    project: str | None,
    args: argparse.Namespace | None = None,
    json_output: bool = False,
) -> None:
    payload = _chat_status_payload(session_store=session_store, session_id=session_id, project=project, args=args)
    if json_output:
        print_json(payload)
        return
    session = payload["session"]
    auto = payload["auto"]
    permission = payload["permission"]
    print(f"{Colors.BOLD}{Colors.CYAN}AETHER status{Colors.RESET}")
    print(f"  model      : {Colors.YELLOW}{payload['model']}{Colors.RESET} ({payload['context_window']:,} ctx)")
    print(f"  project    : {session.get('project') or 'none'}")
    print(f"  session    : {session.get('id')}  turns={session.get('turn_count') or 0}")
    print(f"  title      : {_shorten_inline(session.get('title'), limit=90) or 'New research chat'}")
    print(f"  permission : {permission.get('label')} ({permission.get('mode')})")
    auto_label = "ON" if auto.get("enabled") else "OFF"
    print(f"  auto       : {auto_label}  status={auto.get('status') or 'idle'}  submit={'allowed' if auto.get('allow_cluster_submit') else 'off'}")
    goal = str(auto.get("research_goal") or "").strip()
    if goal:
        print(f"  goal       : {_shorten_inline(goal, limit=110)}")
    pending = session.get("pending_turn") if isinstance(session.get("pending_turn"), dict) else None
    if pending:
        print(f"  {Colors.YELLOW}pending{Colors.RESET}    : {_shorten_inline(pending.get('prompt'), limit=110)}")
        print(f"               use /continue to retry")
    if session.get("project_session_ref"):
        print(f"  ref        : {session.get('project_session_ref')}")
    print(f"{Colors.DIM}  json       : /status --json{Colors.RESET}")


def print_chat_sessions(*, session_store: Any, project: str | None, limit: int = 10) -> None:
    sessions = session_store.list_sessions(project=project, limit=limit)
    if not sessions:
        print("没有可续接的 session。")
        return
    print(f"{Colors.CYAN}recent sessions{Colors.RESET}:")
    for item in sessions:
        project_label = item.project or "none"
        title = _shorten_inline(getattr(item, "title", "") or item.first_prompt, limit=64) or "New research chat"
        first = _shorten_inline(item.first_prompt, limit=56) or "empty"
        print(f"- {title}  {Colors.DIM}{item.session_id}{Colors.RESET}")
        print(f"  project={project_label} turns={item.turn_count} updated={item.updated_at}")
        print(f"  {Colors.DIM}{first}{Colors.RESET}")
        if getattr(item, "pending_turn_status", ""):
            print(
                f"  {Colors.YELLOW}pending={item.pending_turn_status}{Colors.RESET}: "
                f"{_shorten_inline(getattr(item, 'pending_prompt', ''), limit=72)}"
            )


def print_resume_preview(payload: dict[str, Any]) -> None:
    state = payload.get("state") or {}
    print(
        f"{Colors.GREEN}resumed{Colors.RESET}: "
        f"{state.get('title') or payload.get('session_id')} "
        f"({payload.get('session_id')}) project={state.get('project') or 'none'} turns={state.get('turn_count') or 0}"
    )
    recent_turns = payload.get("recent_turns") or []
    if recent_turns:
        print("最近对话：")
        for turn in recent_turns[-3:]:
            record = turn.get("record", {})
            print(f"- user: {_shorten_inline(record.get('prompt'), limit=90)}")
            print(f"  assistant: {_shorten_inline(record.get('response'), limit=90)}")
    pending = state.get("pending_turn")
    if isinstance(pending, dict) and str(pending.get("prompt") or "").strip():
        print(
            f"{Colors.YELLOW}未完成输入{Colors.RESET} ({pending.get('status') or 'pending'}): "
            f"{_shorten_inline(pending.get('prompt'), limit=100)}"
        )
        print(f"{Colors.DIM}输入 /continue 可继续这条输入。{Colors.RESET}")


def _resumable_sessions(session_store: Any, *, project: str | None, current_session_id: str, limit: int = 20) -> list[Any]:
    return [item for item in session_store.list_sessions(project=project, limit=limit) if item.session_id != current_session_id]


def _session_matches_query(item: Any, query: str) -> bool:
    text = " ".join(
        [
            str(getattr(item, "session_id", "") or ""),
            str(getattr(item, "project", "") or ""),
            str(getattr(item, "title", "") or ""),
            str(getattr(item, "first_prompt", "") or ""),
            str(getattr(item, "last_response", "") or ""),
        ]
    ).lower()
    return query.lower() in text


def _print_resume_options(sessions: list[Any], *, heading: str) -> None:
    print(f"{Colors.CYAN}{heading}{Colors.RESET}:")
    for index, item in enumerate(sessions, start=1):
        title = _shorten_inline(getattr(item, "title", "") or item.first_prompt, limit=64) or "New research chat"
        first = _shorten_inline(item.first_prompt, limit=64) or "empty"
        last = _shorten_inline(getattr(item, "last_response", "") or "", limit=76)
        print(f"  {index}. {title}  {Colors.DIM}{item.session_id}{Colors.RESET}")
        print(f"     project={item.project or 'none'} turns={item.turn_count} updated={getattr(item, 'updated_at', '')}")
        print(f"     {Colors.DIM}{first}{Colors.RESET}")
        if last:
            print(f"     last: {Colors.DIM}{last}{Colors.RESET}")
        if getattr(item, "pending_turn_status", ""):
            print(
                f"     {Colors.YELLOW}pending={item.pending_turn_status}{Colors.RESET}: "
                f"{_shorten_inline(getattr(item, 'pending_prompt', ''), limit=72)}"
            )


def _rank_resume_matches(session_store: Any, raw: str, scoped_sessions: list[Any], *, project: str | None, current_session_id: str) -> tuple[list[Any], dict[str, Any]]:
    lexical_matches = [item for item in scoped_sessions if _session_matches_query(item, raw)]
    if len(lexical_matches) == 1:
        return lexical_matches, {"selection_method": "lexical_exact", "selection_error": ""}
    if hasattr(session_store, "rank_sessions"):
        try:
            ranked = session_store.rank_sessions(
                query=raw,
                project=project,
                exclude_session_id=current_session_id,
                limit=50,
                max_results=10,
            )
            hits = ranked.get("matches") if isinstance(ranked, dict) else []
            ranked_matches = [hit.summary for hit in hits if hasattr(hit, "summary")]
            if ranked_matches:
                return ranked_matches, ranked if isinstance(ranked, dict) else {}
            if lexical_matches:
                return lexical_matches, ranked if isinstance(ranked, dict) else {}
        except Exception as exc:
            if lexical_matches:
                return lexical_matches, {"selection_method": "lexical_fallback", "selection_error": str(exc)}
            return [], {"selection_method": "semantic_failed", "selection_error": str(exc)}
    return lexical_matches, {"selection_method": "lexical", "selection_error": ""}


def handle_chat_resume_command(line: str, args: argparse.Namespace, session_store: Any, current_session_id: str) -> str:
    raw = line[len("/resume") :].strip()
    all_scope = False
    if raw in {"all", "--all"}:
        all_scope = True
        raw = ""
    elif raw.startswith("all "):
        all_scope = True
        raw = raw[len("all ") :].strip()
    elif raw.startswith("--all "):
        all_scope = True
        raw = raw[len("--all ") :].strip()
    scope_project = None if all_scope else args.project
    scoped_sessions = _resumable_sessions(session_store, project=scope_project, current_session_id=current_session_id, limit=50)
    if raw == "":
        sessions = scoped_sessions[:10]
        if not sessions:
            scope = "全部项目" if all_scope or not args.project else f"当前项目 {args.project}"
            print(f"{scope} 没有可续接的 session。")
            if args.project and not all_scope:
                print(f"{Colors.DIM}需要跨项目查找时输入 /resume all。{Colors.RESET}")
            return current_session_id
        heading = "resume session (all projects)" if all_scope else "resume session"
        _print_resume_options(sessions, heading=heading)
        if not (hasattr(sys.stdin, "isatty") and sys.stdin.isatty()):
            print("非交互 stdin：请使用 /resume latest、/resume all latest 或 /resume <session_id>。")
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
        matches, match_meta = _rank_resume_matches(
            session_store,
            raw,
            scoped_sessions,
            project=scope_project,
            current_session_id=current_session_id,
        )
        if len(matches) == 1:
            method = str(match_meta.get("selection_method") or "")
            if method.startswith("semantic"):
                print(f"{Colors.DIM}semantic resume match: {method}{Colors.RESET}")
            raw = matches[0].session_id
        elif len(matches) > 1:
            method = str(match_meta.get("selection_method") or "matches")
            _print_resume_options(matches[:10], heading=f"resume matches for {raw!r} ({method})")
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
            suffix = "；跨项目查找可用 /resume all <query>" if args.project and not all_scope else ""
            print(f"没有找到匹配 {raw!r} 的 session。用 /resume 打开选择器{suffix}。")
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
    project_slug = raw
    try:
        project = load_project(project_slug)
    except Exception as exc:
        query = raw.lower()
        matches = [
            item
            for item in list_projects()
            if query
            and query
            in " ".join(
                [
                    str(item.get("slug") or ""),
                    str(item.get("name") or ""),
                    str(item.get("title") or ""),
                    str(item.get("description") or ""),
                    str(item.get("source") or ""),
                ]
            ).lower()
        ]
        if len(matches) == 1:
            project_slug = str(matches[0].get("slug") or matches[0].get("name") or raw)
            project = load_project(project_slug)
        elif len(matches) > 1:
            print(f"{Colors.CYAN}project matches for {raw!r}{Colors.RESET}:")
            for index, item in enumerate(matches[:10], start=1):
                slug = str(item.get("slug") or item.get("name") or "")
                title = str(item.get("title") or item.get("name") or slug)
                source = str(item.get("source") or "")
                print(f"  {index}. {slug}  {Colors.DIM}{_shorten_inline(title, limit=56)} [{source}]{Colors.RESET}")
            if not (hasattr(sys.stdin, "isatty") and sys.stdin.isatty()):
                print("非交互 stdin：请使用 /project <完整 slug>。")
                return current_session_id
            try:
                choice = input("project> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return current_session_id
            if not choice:
                print("project unchanged")
                return current_session_id
            if choice.isdigit() and 1 <= int(choice) <= len(matches[:10]):
                project_slug = str(matches[int(choice) - 1].get("slug") or matches[int(choice) - 1].get("name") or raw)
            else:
                project_slug = choice
            try:
                project = load_project(project_slug)
            except Exception as nested_exc:
                print(f"{Colors.RED}project switch failed{Colors.RESET}: {nested_exc}")
                return current_session_id
        else:
            print(f"{Colors.RED}project switch failed{Colors.RESET}: {exc}")
            return current_session_id
    slug = str(project.get("slug") or project_slug)
    args.project = slug
    session_id = session_store.latest_session_id(project=slug) or session_store.start_session(project=slug)
    print(f"{Colors.GREEN}project switched{Colors.RESET}: {Colors.YELLOW}{slug}{Colors.RESET}")
    payload = session_store.resume_payload(session_id=session_id)
    if payload.get("status") == "ok":
        print_resume_preview(payload)
    return session_id


def print_chat_context_status(*, session_store: Any, session_id: str) -> None:
    from .context_budget import context_budget, current_context_window_tokens, usable_context_chars, usable_context_tokens

    state = session_store.load_state(session_id)
    session_context = session_store.build_session_context(session_id)
    budget = context_budget()
    used_chars = len(session_context)
    usable_chars = usable_context_chars()
    usage_ratio = used_chars / usable_chars if usable_chars else 0.0
    analysis = {}
    if hasattr(session_store, "analyze_context"):
        try:
            analysis = session_store.analyze_context(session_id)
        except Exception as exc:
            analysis = {"status": "error", "message": str(exc)}
    print_json(
        {
            "model": program_model_id(),
            "model_context_window_tokens": current_context_window_tokens(),
            "usable_context_tokens": usable_context_tokens(),
            "usable_context_chars": usable_chars,
            "auto_compact_threshold_chars": budget.auto_compact_chars,
            "auto_compact_ratio": budget.auto_compact_ratio,
            "guard_threshold_chars": budget.guard_chars,
            "guard_ratio": budget.guard_ratio,
            "current_session_context_chars": used_chars,
            "context_usage_percent": round(usage_ratio * 100, 2),
            "compacted_turn_count": state.get("compacted_turn_count", 0),
            "has_compact_summary": bool(str(state.get("compact_summary") or "").strip()),
            "last_compact": {
                "at": state.get("last_compacted_at"),
                "trigger": state.get("last_compact_trigger"),
                "reason": state.get("last_compact_reason"),
                "stats": state.get("last_compact_stats"),
            },
            "context_analysis": analysis,
            "suggestions": _context_suggestions(
                usage_ratio=usage_ratio,
                has_compact_summary=bool(str(state.get("compact_summary") or "").strip()),
            ),
        }
    )


def _context_suggestions(*, usage_ratio: float, has_compact_summary: bool) -> list[str]:
    suggestions: list[str] = []
    if usage_ratio >= 0.80:
        suggestions.append("上下文接近上限：建议先执行 /compact，再继续长对话。")
    elif usage_ratio >= 0.55:
        suggestions.append("上下文已过半：如果接下来要做长工具链，可考虑 /compact 预先整理。")
    if has_compact_summary:
        suggestions.append("当前 session 已有 compact summary；/resume 会继续带入该摘要。")
    if not suggestions:
        suggestions.append("上下文充足；可以继续自然语言对话。")
    return suggestions


def print_chat_resume_hint(*, session_id: str | None, project: str | None) -> None:
    if not session_id:
        return
    project_arg = f" --project {project}" if project else ""
    print(
        f"{Colors.DIM}Resume this session with:{Colors.RESET}\n"
        f"  {Colors.GREEN}aether chat --resume --session-id {session_id}{project_arg}{Colors.RESET}"
    )


def handle_chat_compact_command(line: str, *, session_store: Any, session_id: str) -> None:
    raw = line[len("/compact") :].strip()
    keep_recent = 12
    if raw:
        try:
            keep_recent = max(1, min(int(raw), 200))
        except ValueError:
            print("用法：/compact [保留最近轮数]，例如 /compact 12")
            return
    if not hasattr(session_store, "compact_session"):
        print("当前 session store 不支持 compact。")
        return
    result = session_store.compact_session(session_id, keep_recent=keep_recent, trigger="manual")
    if result.get("status") == "ok":
        print(
            f"{Colors.GREEN}compact complete{Colors.RESET}: "
            f"compacted={result.get('compacted_turn_count')} keep_recent={result.get('keep_recent')} "
            f"summary_chars={result.get('compact_summary_chars')}"
        )
    else:
        print(
            f"{Colors.YELLOW}compact skipped{Colors.RESET}: "
            f"{result.get('reason')} turns={result.get('turn_count')} keep_recent={result.get('keep_recent')}"
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
            parallel = " ⇉" if event.get("parallel") else ""
            print(f"{Colors.BLUE}↳ tool{parallel}{Colors.RESET} {event.get('name')}{Colors.DIM}{step_elapsed} {args}{Colors.RESET}")
        elif kind == "tool_parallel_start":
            names = ", ".join(str(item) for item in (event.get("names") or []))
            if len(names) > 120:
                names = names[:117] + "..."
            print(f"{Colors.BLUE}⇉ parallel tools{Colors.RESET} x{event.get('count') or 0} {Colors.DIM}{names}{Colors.RESET}")
        elif kind == "tool_progress":
            elapsed = event.get("elapsed_seconds")
            print(
                f"{Colors.DIM}… tool {event.get('name')} still running"
                f" ({elapsed}s): {_shorten_inline(event.get('message'), limit=110)}{Colors.RESET}"
            )
        elif kind == "tool_finish":
            status = event.get("status") or "done"
            step = int(event.get("step") or 0)
            name = str(event.get("name") or "")
            tool_elapsed = elapsed_since(tool_started.get((step, name)))
            persisted = event.get("persisted_output_path")
            compact = " compact" if event.get("microcompacted") else ""
            suffix = f" {Colors.DIM}{persisted}{Colors.RESET}" if persisted else ""
            print(f"{Colors.GREEN}✓ tool{Colors.RESET} {event.get('name')} status={status}{Colors.DIM}{tool_elapsed}{Colors.RESET}{suffix}")
            if compact:
                print(f"{Colors.DIM}  ↳ large tool output was microcompacted for the model; full payload is persisted above{Colors.RESET}")
        elif kind == "token_guard_finalize":
            ratio = float(event.get("usage_ratio") or 0.0) * 100
            print(
                f"{Colors.YELLOW}context guard{Colors.RESET}: {ratio:.1f}% budget used; "
                "stopping further tool calls and asking the model to summarize"
            )
        elif kind == "session_auto_compacted":
            print(
                f"{Colors.YELLOW}auto-compact{Colors.RESET}: "
                f"compacted={event.get('compacted_turn_count')} "
                f"summary_chars={event.get('compact_summary_chars')} "
                f"before={event.get('approx_chars_before')} "
                f"threshold={event.get('auto_compact_threshold_chars')}"
            )
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


def run_chat_model_turn(
    prompt: str,
    *,
    args: argparse.Namespace,
    session_store: Any,
    session_id: str | None,
    failure_hint: str,
) -> tuple[bool, str]:
    session_id = session_store.ensure_session(session_id=session_id, project=args.project, first_prompt=prompt)
    if hasattr(session_store, "record_pending_turn"):
        session_store.record_pending_turn(
            session_id,
            prompt=prompt,
            project=args.project,
            model_id=active_model_id(args),
            status="in_progress",
        )
    stream_printer, stream_state = make_stream_printer()
    interactive_cli_callbacks = bool(getattr(args, "auto_interactive_questions", True))
    allow_cluster_submit = False
    try:
        from .auto_mode import load_auto_state

        auto = load_auto_state(args.project)
        allow_cluster_submit = bool(auto.get("enabled") and auto.get("allow_cluster_submit"))
    except Exception:
        allow_cluster_submit = False
    try:
        record = ask_once(
            prompt,
            project=args.project,
            model_id=args.model,
            max_tokens=args.max_tokens,
            max_steps=args.max_steps,
            allow_cluster_submit=allow_cluster_submit,
            session_id=session_id,
            permission_mode=get_permission_mode(),
            progress_callback=make_chat_progress_printer(),
            permission_prompt_callback=make_permission_prompt_callback() if interactive_cli_callbacks else None,
            stream_callback=stream_printer,
            human_question_callback=answer_auto_human_question_from_cli
            if interactive_cli_callbacks
            else None,
        )
    except Exception as exc:
        if hasattr(session_store, "mark_pending_turn_failed"):
            session_store.mark_pending_turn_failed(session_id, error=str(exc))
        print(f"{Colors.RED}模型调用失败{Colors.RESET}: {_shorten_inline(str(exc), limit=360)}")
        print(f"{Colors.DIM}{failure_hint}；这条输入已保存在当前 session，修复后可输入 /continue 重试。{Colors.RESET}")
        return False, session_id
    interrupted = record.get("finish_reason") == "user_interrupted"
    if hasattr(session_store, "clear_pending_turn") and not interrupted:
        session_store.clear_pending_turn(session_id)
    print_streamed_or_final_response(record, stream_state)
    print_turn_footer(record)
    if interrupted:
        print(f"{Colors.YELLOW}本轮已中断{Colors.RESET}；输入 /continue 可继续重试这条请求。")
        return False, session_id
    next_steps = (record.get("progress") or {}).get("next_steps") or []
    if next_steps:
        print(f"[next] {next_steps[0]}")
    return True, session_id


def handle_chat_continue_command(args: argparse.Namespace, session_store: Any, session_id: str) -> tuple[bool, str]:
    if not hasattr(session_store, "pending_turn"):
        print("当前 session 没有可继续的未完成输入。")
        return True, session_id
    pending = session_store.pending_turn(session_id)
    if not pending:
        print("当前 session 没有可继续的未完成输入。")
        return True, session_id
    pending_project = pending.get("project")
    if pending_project:
        args.project = str(pending_project)
    prompt = str(pending.get("prompt") or "").strip()
    print(f"{Colors.CYAN}continue>{Colors.RESET} {_shorten_inline(prompt, limit=140)}")
    return run_chat_model_turn(
        prompt,
        args=args,
        session_store=session_store,
        session_id=session_id,
        failure_hint="本 session 仍然保留",
    )


def handle_chat_history_command(line: str, *, session_store: Any, session_id: str) -> None:
    raw = line[len("/history") :].strip()
    limit = 12
    query = raw
    match = re.search(r"(?:^|\s)--limit\s+(\d+)(?:\s|$)", raw)
    if match:
        limit = max(1, min(int(match.group(1)), 100))
        query = (raw[: match.start()] + " " + raw[match.end() :]).strip()
    if hasattr(session_store, "search_transcript"):
        turns = session_store.search_transcript(session_id, query=query, limit=limit)
    else:
        turns = session_store.read_transcript(session_id, limit=limit)
    if not turns:
        print("当前 session 没有匹配的历史。")
        return
    heading = "history" if not query else f"history matches for {query!r}"
    print(f"{Colors.CYAN}{heading}{Colors.RESET}:")
    for turn in turns:
        record = turn.get("record") or {}
        timestamp = str(turn.get("timestamp") or "")
        print(f"- {Colors.DIM}{timestamp}{Colors.RESET} user: {_shorten_inline(record.get('prompt'), limit=110)}")
        response = _shorten_inline(record.get("response"), limit=120)
        if response:
            print(f"  assistant: {response}")
        tool_names = [str(item.get("name") or "") for item in record.get("tool_executions") or [] if item.get("name")]
        if tool_names:
            print(f"  tools: {', '.join(tool_names[:8])}")


def handle_chat_rename_command(line: str, *, session_store: Any, session_id: str) -> None:
    title = line[len("/rename") :].strip()
    if not title:
        if not (hasattr(sys.stdin, "isatty") and sys.stdin.isatty()):
            print("用法：/rename <新标题>")
            return
        try:
            title = input("title> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
    if not title:
        print("rename cancelled")
        return
    if not hasattr(session_store, "rename_session"):
        print("当前 session store 不支持 rename。")
        return
    try:
        state = session_store.rename_session(session_id, title)
    except Exception as exc:
        print(f"{Colors.RED}rename failed{Colors.RESET}: {exc}")
        return
    print(f"{Colors.GREEN}renamed{Colors.RESET}: {state.get('title')}")


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
        normalized_candidate = normalize_model_id(raw, Path.cwd())
        candidate = catalog.get(normalized_candidate)
        if candidate is not None and not candidate.available:
            print(
                f"{Colors.RED}model switch blocked{Colors.RESET}: "
                f"{normalized_candidate} 缺少 API key（{candidate.api_key_env}）。请先配置 key，或选择 available 模型。"
            )
            return
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


def print_auto_preview(payload: dict[str, Any]) -> None:
    state = payload.get("state") or {}
    if not state:
        print_json(payload)
        return
    enabled = bool(state.get("enabled"))
    label = f"{Colors.GREEN}ON{Colors.RESET}" if enabled else f"{Colors.YELLOW}OFF{Colors.RESET}"
    print(f"{Colors.BOLD}{Colors.CYAN}/auto{Colors.RESET}: {label}")
    print(f"  project : {state.get('project') or 'none'}")
    print(f"  goal    : {_shorten_inline(state.get('research_goal'), limit=150) or 'none'}")
    print(f"  phase   : {state.get('current_phase') or state.get('status') or 'idle'}")
    print(f"  rounds  : {state.get('iteration_count') or 0}")
    print(f"  monitor : every {state.get('monitor_interval_hours')}h | daily report {state.get('daily_report_time')}")
    submit = f"{Colors.GREEN}allowed{Colors.RESET}" if state.get("allow_cluster_submit") else f"{Colors.YELLOW}off{Colors.RESET} — AI can prepare jobs and ask before submit"
    print(f"  cluster : {submit}")
    print(
        f"  policy  : literature={'on' if state.get('allow_literature_search') else 'off'}, "
        f"structure={'on' if state.get('allow_structure_build') else 'off'}, "
        f"writeback={'on' if state.get('allow_research_writeback') else 'off'}"
    )
    criteria = [str(item) for item in (state.get("success_criteria") or []) if str(item).strip()]
    if criteria:
        print(f"  {Colors.CYAN}success criteria{Colors.RESET}:")
        for item in criteria[:5]:
            print(f"   - {_shorten_inline(item, limit=150)}")
    audit = state.get("convergence_audit") if isinstance(state.get("convergence_audit"), dict) else {}
    if audit:
        verdict = audit.get("verdict") or "none"
        print(f"  {Colors.CYAN}convergence audit{Colors.RESET}: {verdict}")
        completed = [str(item) for item in (audit.get("completed_items") or []) if str(item).strip()]
        missing = [str(item) for item in (audit.get("missing_evidence") or []) if str(item).strip()]
        if completed:
            print("   completed:")
            for item in completed[:4]:
                print(f"    ✓ {_shorten_inline(item, limit=145)}")
        if missing:
            print("   missing evidence:")
            for item in missing[:4]:
                print(f"    ! {_shorten_inline(item, limit=145)}")
        next_focus = str(audit.get("next_focus") or "").strip()
        if next_focus:
            print(f"   next: {_shorten_inline(next_focus, limit=150)}")
    due = ((payload.get("due_followups") or {}).get("followups") or []) if isinstance(payload.get("due_followups"), dict) else []
    scheduled = ((payload.get("scheduled_followups") or {}).get("followups") or []) if isinstance(payload.get("scheduled_followups"), dict) else []
    campaigns = ((payload.get("active_campaigns") or {}).get("campaigns") or []) if isinstance(payload.get("active_campaigns"), dict) else []
    if due or scheduled or campaigns:
        print(f"  {Colors.CYAN}DFT board{Colors.RESET}: due={len(due)} scheduled={len(scheduled)} campaigns={len(campaigns)}")
        for item in due[:3]:
            print(f"   due now: {item.get('title') or item.get('id')}")
    daemon = payload.get("daemon") if isinstance(payload.get("daemon"), dict) else {}
    if daemon:
        daemon_status = daemon.get("status") or "unknown"
        lock_process = daemon.get("lock_process") if isinstance(daemon.get("lock_process"), dict) else {}
        process_status = lock_process.get("status") if lock_process else ""
        suffix = f" / process={process_status}" if process_status else ""
        print(f"  {Colors.CYAN}daemon{Colors.RESET}: {daemon_status}{suffix}")
        if daemon_status == "stale_lock":
            print(f"   {Colors.YELLOW}stale lock{Colors.RESET}: {daemon.get('lock_path')}")
    questions = state.get("human_questions") or []
    if questions:
        print(f"  {Colors.YELLOW}questions for human{Colors.RESET}:")
        for question in questions[:5]:
            print(f"   - {question}")
    if not enabled and not state.get("research_goal"):
        project_hint = f" --project {state.get('project')}" if state.get("project") else ""
        print(f"{Colors.DIM}  next: /auto <研究目标>  或  aether auto \"研究目标\"{project_hint}{Colors.RESET}")


def answer_auto_human_question_from_cli(payload: dict[str, Any]) -> dict[str, Any]:
    """Ask one model-authored /auto question in the terminal and persist answer."""

    from .auto_mode import answer_auto_human_question

    question_record = payload.get("question") if isinstance(payload.get("question"), dict) else {}
    project = str(payload.get("project") or question_record.get("project") or "").strip() or None
    question_id = str(payload.get("question_id") or question_record.get("id") or "").strip() or None
    question = str(question_record.get("question") or payload.get("question_text") or payload.get("question") or "").strip()
    if not question:
        return {"status": "error", "message": "auto_human_question 缺少 question。"}
    print(f"\n{Colors.CYAN}[auto question]{Colors.RESET}")
    why = str(question_record.get("why_needed") or payload.get("why_needed") or "").strip()
    boundary = str(question_record.get("decision_boundary") or payload.get("decision_boundary") or "").strip()
    if why:
        print(f"  why: {why}")
    if boundary:
        print(f"  decision: {boundary}")
    evidence = question_record.get("evidence_refs") or payload.get("evidence_refs") or []
    if evidence:
        print(f"  evidence: {', '.join(str(item) for item in evidence[:5])}")
    print(f"  Q: {Colors.BOLD}{question}{Colors.RESET}")
    options = [str(item).strip() for item in (question_record.get("options") or payload.get("options") or []) if str(item).strip()]
    for index, option in enumerate(options, start=1):
        print(f"   {index}. {option}")
    default = str(question_record.get("default_if_unanswered") or payload.get("default_if_unanswered") or "").strip()
    if default:
        print(f"  default if blank: {default}")
    try:
        raw = input("answer> ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return {"status": "pending_human_answer", "question": question_record, "message": "用户暂未回答；问题保持 pending。"}
    if raw.isdigit() and options and 1 <= int(raw) <= len(options):
        raw = options[int(raw) - 1]
    if not raw and default:
        raw = default
    if not raw:
        return {"status": "pending_human_answer", "question": question_record, "message": "空答案未记录；问题保持 pending。"}
    result = answer_auto_human_question(project=project, question_id=question_id, answer=raw, source="cli")
    if result.get("status") == "answered":
        print(f"{Colors.GREEN}[auto]{Colors.RESET} human answer recorded; AI can continue from it.")
    return result


def handle_pending_auto_question(
    args: argparse.Namespace,
    *,
    session_store: Any,
    session_ref: dict[str, str | None],
) -> bool:
    """Surface a queued /auto question before accepting the next user command."""

    from .auto_mode import latest_pending_auto_human_question

    pending = latest_pending_auto_human_question(project=args.project)
    if not pending:
        return False
    result = answer_auto_human_question_from_cli({"project": args.project, "question": pending, "question_id": pending.get("id")})
    if result.get("status") != "answered":
        return False
    follow = run_auto_due_once(
        args=args,
        session_store=session_store,
        session_ref=session_ref,
        quiet=False,
        interactive_questions=True,
    )
    return bool(follow.get("ran"))


def auto_turn_args(args: argparse.Namespace) -> argparse.Namespace:
    """Return a non-mutating args copy with enough budget for /auto work.

    A normal user chat can be intentionally short, but an autonomous pass must
    have enough tool steps to inspect state, act, and write a checkpoint.
    Otherwise `/auto` feels broken: it starts, uses a few tools, then stops
    before persisting progress.
    """

    auto_args = copy.copy(args)
    try:
        auto_args.max_steps = max(int(getattr(auto_args, "max_steps", 0) or 0), AUTO_TURN_MIN_STEPS)
    except (TypeError, ValueError):
        auto_args.max_steps = AUTO_TURN_MIN_STEPS
    try:
        current_tokens = int(getattr(auto_args, "max_tokens", 0) or 0)
    except (TypeError, ValueError):
        current_tokens = 0
    auto_args.max_tokens = max(current_tokens, AUTO_TURN_MIN_TOKENS) if current_tokens else AUTO_TURN_MIN_TOKENS
    return auto_args


def _latest_session_record(session_store: Any, session_id: str | None) -> dict[str, Any]:
    if not session_id or not hasattr(session_store, "read_transcript"):
        return {}
    try:
        rows = session_store.read_transcript(session_id, limit=1)
    except Exception:
        return {}
    if not rows:
        return {}
    record = rows[-1].get("record") if isinstance(rows[-1], dict) else None
    return record if isinstance(record, dict) else {}


def _auto_fallback_checkpoint(
    *,
    project: str | None,
    session_id: str | None,
    record: dict[str, Any],
) -> dict[str, Any]:
    from .auto_mode import checkpoint_auto_mode

    tools = [item for item in (record.get("tool_executions") or []) if isinstance(item, dict)]
    if not tools:
        return {"status": "skipped", "reason": "no_tool_evidence"}
    names = [str(item.get("name") or "tool") for item in tools if item.get("name")]
    evidence_refs = []
    if record.get("record_path"):
        evidence_refs.append(str(record.get("record_path")))
    if session_id:
        evidence_refs.append(f"session:{session_id}")
    response = _shorten_inline(record.get("response"), limit=180)
    observation = (
        f"Auto pass executed {len(tools)} tool(s): {', '.join(names[:12])}."
        + (f" Last model response: {response}" if response else "")
    )
    return checkpoint_auto_mode(
        project=project,
        status="active",
        observation=observation,
        decision=(
            "Harness fallback checkpoint: model advanced the project with tool evidence but did not call "
            "auto_mode_checkpoint before the turn ended."
        ),
        evidence_refs=evidence_refs,
        next_focus="Inspect campaign/project status and continue from the latest persisted tool evidence.",
        human_questions=[],
    )


def _auto_record_has_completion_signal(record: dict[str, Any]) -> bool:
    """Return whether a tool-evidence turn is safe to consume a due intent.

    A pass that only inspected state or built an intermediate slab should be
    checkpointed but kept due so /auto can immediately continue.  Consuming the
    due item is reserved for turns that persisted a decision/candidate/result,
    submitted/updated work, wrote progress, or otherwise reached a durable
    handoff.
    """

    tool_names = {str(item.get("name") or "") for item in (record.get("tool_executions") or []) if isinstance(item, dict)}
    completion_tools = {
        "auto_campaign_register_candidates",
        "auto_campaign_update_candidate",
        "auto_campaign_prune_plan",
        "cluster_remote_submit",
        "cluster_remote_monitor",
        "cluster_remote_fetch",
        "project_progress_append",
        "research_progress_append",
        "research_learning_capture",
        "candidate_outcome_record",
        "auto_human_question",
        "auto_mode_convergence_audit",
    }
    return bool(tool_names & completion_tools)


def _auto_register_manifest_if_present(*, project: str | None, record: dict[str, Any]) -> dict[str, Any]:
    tools = [item for item in (record.get("tool_executions") or []) if isinstance(item, dict)]
    if any(str(item.get("name") or "") == "auto_campaign_register_candidates" for item in tools):
        return {"status": "skipped", "reason": "already_registered"}
    manifest_path = ""
    for item in reversed(tools):
        if str(item.get("name") or "") != "adsorption_candidate_manifest_compose":
            continue
        result = item.get("result") if isinstance(item.get("result"), dict) else {}
        if str(result.get("status") or "") in {"ok", "composed"}:
            manifest_path = str(result.get("manifest_json") or result.get("manifest_path") or "").strip()
            if manifest_path:
                break
    if not manifest_path:
        return {"status": "skipped", "reason": "no_manifest"}
    try:
        from .auto_campaign import list_campaigns, register_candidates

        campaign_id = ""
        for item in tools:
            if str(item.get("name") or "") != "auto_campaign_start":
                continue
            result = item.get("result") if isinstance(item.get("result"), dict) else {}
            campaign = result.get("campaign") if isinstance(result.get("campaign"), dict) else {}
            campaign_id = str(result.get("campaign_id") or campaign.get("campaign_id") or "").strip()
            if campaign_id:
                break
        if not campaign_id and project:
            campaigns = list_campaigns(project=str(project), include_closed=False, limit=1).get("campaigns") or []
            if campaigns:
                campaign_id = str(campaigns[0].get("campaign_id") or "").strip()
        if not project or not campaign_id:
            return {"status": "skipped", "reason": "missing_project_or_campaign", "manifest_path": manifest_path}
        return register_candidates(
            project=str(project),
            campaign_id=campaign_id,
            candidates=[],
            source_manifest_path=manifest_path,
            note="Harness auto-registered candidates from manifest evidence after model turn ended before campaign registration.",
        )
    except Exception as exc:
        return {"status": "error", "message": str(exc), "manifest_path": manifest_path}


def _auto_checkpoint_fingerprint(checkpoint: dict[str, Any]) -> str:
    """Return a stable fingerprint for checkpoint change detection.

    ``updated_at`` is second-resolution, so two legitimate checkpoints written in
    the same second can otherwise look unchanged.  Fingerprinting the whole
    payload keeps the due-loop decision tied to actual state, not wall-clock
    granularity.
    """

    try:
        return json.dumps(checkpoint or {}, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        return repr(checkpoint)


def run_auto_due_once(
    *,
    args: argparse.Namespace,
    session_store: Any,
    session_ref: dict[str, str | None],
    now: str | None = None,
    turn_runner: Any | None = None,
    quiet: bool = False,
    interactive_questions: bool = True,
) -> dict[str, Any]:
    """Run one due autonomous model turn, if the durable queue says one is due."""

    ensure_console_utf8()

    from .auto_mode import collect_due_auto_intents, complete_due_auto_intents, load_auto_state

    plan = collect_due_auto_intents(project=args.project, now=now, limit=5)
    if not plan.get("should_run"):
        return {"status": plan.get("status") or "idle", "ran": False, "plan": plan}
    before_checkpoint = load_auto_state(args.project).get("last_checkpoint") or {}
    before_checkpoint_fingerprint = _auto_checkpoint_fingerprint(before_checkpoint)
    if not quiet:
        titles = ", ".join(str(item.get("title") or item.get("id")) for item in (plan.get("followups") or [])[:3])
        print(f"\n{Colors.CYAN}[auto]{Colors.RESET} due: {titles or 'scheduled research work'}")
    runner = turn_runner or run_chat_model_turn
    runner_args = auto_turn_args(args)
    runner_args.auto_interactive_questions = bool(interactive_questions)
    ok, new_session_id = runner(
        str(plan.get("prompt") or ""),
        args=runner_args,
        session_store=session_store,
        session_id=session_ref.get("id"),
        failure_hint="auto 后台推进未完成；状态和 follow-up 已保留，下次到期会继续。",
    )
    if new_session_id:
        session_ref["id"] = new_session_id
    if not ok:
        return {"status": "error", "ran": True, "completed": False, "plan": plan}
    latest_record = _latest_session_record(session_store, session_ref.get("id"))
    auto_register = _auto_register_manifest_if_present(project=args.project, record=latest_record)
    if auto_register.get("status") == "ok":
        latest_record = copy.deepcopy(latest_record)
        latest_record.setdefault("tool_executions", []).append(
            {"name": "auto_campaign_register_candidates", "result": auto_register}
        )
    after_checkpoint = load_auto_state(args.project).get("last_checkpoint") or {}
    after_checkpoint_fingerprint = _auto_checkpoint_fingerprint(after_checkpoint)
    checkpoint_changed = bool(after_checkpoint) and after_checkpoint_fingerprint != before_checkpoint_fingerprint
    completion_signal = _auto_record_has_completion_signal(latest_record)
    if not checkpoint_changed:
        fallback = _auto_fallback_checkpoint(project=args.project, session_id=session_ref.get("id"), record=latest_record)
        if fallback.get("status") == "ok":
            if not completion_signal:
                if not quiet:
                    print(f"{Colors.DIM}[auto] wrote partial fallback checkpoint; keeping due intent open for immediate continuation.{Colors.RESET}")
                return {
                    "status": "partial_checkpoint",
                    "ran": True,
                    "completed": False,
                    "plan": plan,
                    "fallback_checkpoint": fallback,
                    "message": "已写入部分进展 checkpoint，但本轮还没有候选/结果/提交/进展写回等完成信号；due intent 保留以便继续推进。",
                }
            completed = complete_due_auto_intents(
                project=args.project,
                followup_ids=plan.get("followup_ids") or [],
                note="Processed by /auto autonomous loop with harness fallback checkpoint.",
                reschedule=True,
            )
            if not quiet:
                print(f"{Colors.DIM}[auto] wrote fallback checkpoint from executed tool evidence.{Colors.RESET}")
            return {
                "status": "ok",
                "ran": True,
                "completed": True,
                "plan": plan,
                "completion": completed,
                "fallback_checkpoint": fallback,
            }
        if not quiet:
            print(
                f"{Colors.YELLOW}[auto] model turn ended without auto_mode_checkpoint; "
                f"leaving due intent open so the next auto pass can continue.{Colors.RESET}"
            )
        return {
            "status": "needs_checkpoint",
            "ran": True,
            "completed": False,
            "plan": plan,
            "message": "模型本轮没有写入 auto_mode_checkpoint；未消费 due follow-up，后续会继续推进。",
        }
    if not completion_signal:
        if not quiet:
            print(
                f"{Colors.DIM}[auto] model wrote a partial checkpoint; "
                f"keeping due intent open for the next autonomous pass.{Colors.RESET}"
            )
        return {
            "status": "partial_checkpoint",
            "ran": True,
            "completed": False,
            "plan": plan,
            "message": "模型写入了 checkpoint，但本轮还没有候选/结果/提交/进展写回等完成信号；due intent 保留继续推进。",
        }
    completed = complete_due_auto_intents(
        project=args.project,
        followup_ids=plan.get("followup_ids") or [],
        note="Processed by /auto autonomous loop.",
        reschedule=True,
    )
    return {"status": "ok", "ran": True, "completed": True, "plan": plan, "completion": completed}


def start_auto_background_loop(
    *,
    args: argparse.Namespace,
    session_store: Any,
    session_ref: dict[str, str | None],
) -> threading.Event:
    """Start the in-process /auto scheduler for interactive chat.

    The durable follow-up queue is the source of truth; this thread merely
    wakes up periodically while the chat UI is open and lets the model handle
    any due research work.  No manual user command is required.
    """

    stop_event = threading.Event()
    wake_event = threading.Event()
    setattr(args, "_auto_wake_event", wake_event)
    try:
        interval = max(10.0, float(os.getenv("AETHER_AUTO_LOOP_INTERVAL_SECONDS", "300")))
    except ValueError:
        interval = 300.0
    lock = threading.Lock()

    def worker() -> None:
        # First wait keeps chat startup snappy and avoids firing immediately
        # after the user toggles /auto on; due work is handled by schedule time.
        while not stop_event.is_set():
            wake_event.wait(interval)
            wake_event.clear()
            if stop_event.is_set():
                break
            if lock.locked():
                continue
            with lock:
                try:
                    run_auto_due_once(args=args, session_store=session_store, session_ref=session_ref, interactive_questions=False)
                except Exception as exc:  # defensive UI guard; details are visible to the user.
                    print(f"\n{Colors.YELLOW}[auto] background loop paused after error: {exc}{Colors.RESET}")

    thread = threading.Thread(target=worker, name="aether-auto-loop", daemon=True)
    thread.start()
    return stop_event


def handle_chat_auto_command(line: str, args: argparse.Namespace, session_store: Any, session_id: str) -> tuple[bool, str]:
    from .auto_mode import auto_mode_status, configure_auto_mode, infer_research_goal

    raw = line[len("/auto") :].strip()
    if raw in {"off", "stop", "pause", "disable"}:
        result = configure_auto_mode(project=args.project, enabled=False)
        print_auto_preview(result)
        return True, session_id
    if raw == "on":
        raw = ""
    if raw in {"status", "show", "list"}:
        print_auto_preview(_auto_status_with_daemon(args.project))
        return True, session_id
    if raw in {"tick", "run", "continue"}:
        print("不用手动推进：/auto 开启后会根据项目 follow-up 到期时间由后台自动让模型工作。")
        return True, session_id
    goal = raw
    if not goal:
        current = auto_mode_status(project=args.project, include_due=True)
        state = current.get("state") or {}
        if state.get("enabled"):
            result = configure_auto_mode(project=args.project, enabled=False)
            print_auto_preview(result)
            return True, session_id
        goal = str(state.get("research_goal") or "").strip()
        inferred = {}
        if not goal:
            inferred = infer_research_goal(project=args.project, session_store=session_store, session_id=session_id)
            goal = str(inferred.get("goal") or "").strip()
            if goal:
                print(f"{Colors.DIM}auto goal inferred from {inferred.get('source')}: {_shorten_inline(goal, limit=120)}{Colors.RESET}")
        if hasattr(sys.stdin, "isatty") and sys.stdin.isatty():
            if not goal:
                try:
                    goal = input("research goal> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    return True, session_id
        if not goal:
            print("没有从当前项目或对话中找到明确研究目标。请先说明目标，或输入 /auto <目标>。")
            return True, session_id
    result = configure_auto_mode(
        project=args.project,
        enabled=True,
        research_goal=goal,
        monitor_interval_hours=4,
        daily_report_time="18:00",
        allow_literature_search=True,
        allow_structure_build=True,
        # /auto is a switch, not an implicit permission escalation.  Keep the
        # interactive path aligned with `aether auto on`: cluster submission is
        # disabled unless the user explicitly enables it through the CLI/tool
        # permission path.
        allow_cluster_submit=False,
        allow_research_writeback=True,
        reset_questions=True,
    )
    print_auto_preview(result)
    print(f"{Colors.DIM}/auto 已作为开关开启：已创建初始推进、周期 monitor 和 daily-report follow-up；后台会按目标自动推进。{Colors.RESET}")
    if result.get("status") == "ok":
        wake_event = getattr(args, "_auto_wake_event", None)
        if wake_event is not None and hasattr(wake_event, "set"):
            wake_event.set()
            print(f"{Colors.DIM}后台已被唤醒；如 AI 需要人类判断，会直接在 CLI 里提出一个问题。{Colors.RESET}")
    return True, session_id


def _current_model_config(model_id: str | None = None) -> tuple[str, str, dict[str, Any]]:
    provider_id, model_name = split_model_id(model_id or resolve_effective_model_id())
    return provider_id, model_name, build_provider_model_config(provider_id, model_name)


def python_version_supported() -> bool:
    return sys.version_info.major == SUPPORTED_PYTHON_MAJOR and sys.version_info.minor in SUPPORTED_PYTHON_MINORS


def doctor(args: argparse.Namespace) -> int:
    runtime = DomesticCopilotLLM(Path.cwd()).describe_runtime()
    provider_id, model_name, config = _current_model_config(args.model)
    api_keys = load_api_keys(Path.cwd())
    has_key = bool(str(api_keys.get(provider_id, "")).strip())
    if not has_key:
        has_key = bool(os.getenv(str(config["api_key_env"]), "").strip())
    has_base_url = bool(str(config.get("base_url", "") or "").strip())
    python_ok = python_version_supported()
    venv_root = Path.cwd() / ".venv"
    dependency_modules = {
        "aether_dft": "aether_dft",
        "ase": "ase.io",
        "openai": "openai",
        "pymatgen": "pymatgen",
        "rdkit": "rdkit",
    }
    dependencies = {
        label: {"module": module, "available": importlib.util.find_spec(module) is not None}
        for label, module in dependency_modules.items()
    }
    dependencies_ok = all(item["available"] for item in dependencies.values())
    payload = {
        "program": {
            "name": PROGRAM_NAME,
            "command": PROGRAM_COMMAND,
            "version": __version__,
        },
        "python": {
            "required": "3.12.x or 3.13.x",
            "current": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "ok": python_ok,
        },
        "project_venv": {
            "path": str(venv_root),
            "exists": venv_root.exists(),
            "ready_marker": (venv_root / ".aether_ready").exists(),
            "python": str(venv_root / "Scripts" / "python.exe"),
        },
        "dependencies": dependencies,
        "runtime": runtime,
        "cache_policy": {
            "pip_cache_dir": os.getenv("PIP_CACHE_DIR"),
            "xdg_cache_home": os.getenv("XDG_CACHE_HOME"),
            "mpl_config_dir": os.getenv("MPLCONFIGDIR"),
            "temp": os.getenv("TEMP"),
            "tmp": os.getenv("TMP"),
            "python_no_user_site": os.getenv("PYTHONNOUSERSITE"),
        },
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
    if getattr(args, "json", False):
        print_json(payload)
    else:
        print("AETHER-DFT doctor")
        print_json(payload)
    if not python_ok:
        if not getattr(args, "json", False):
            print("ERROR: AETHER requires Python 3.12.x or 3.13.x. Please install a supported Python and recreate the project .venv.")
        return 1
    if not dependencies_ok:
        missing = ", ".join(label for label, info in dependencies.items() if not info["available"])
        if not getattr(args, "json", False):
            print(f"ERROR: Missing required Python packages: {missing}. Re-run aether.cmd after fixing installation.")
        return 1
    if not has_base_url:
        if not getattr(args, "json", False):
            print("WARN: 当前 OpenAI-compatible provider 未配置 base_url")
        return 1
    if not has_key:
        if not getattr(args, "json", False):
            print("WARN: 当前模型 provider 未找到 API key；请检查 api_keys.local.json 或对应环境变量。")
        return 1
    if not getattr(args, "json", False):
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
    print(f"{Colors.DIM}evidence-led research chat; no fixed workflow. Use / for command palette in REPL.{Colors.RESET}")
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
                "entrypoint": [
                    "直接进入 aether 交互式 REPL，用自然语言说明当前科研目标。",
                    "模型会按证据决定是否读 research、查结构、调用建模/集群工具。",
                    "需要切项目、模型、会话时输入 / 打开命令面板。",
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
        ok, session_id = run_chat_model_turn(
            prompt,
            args=args,
            session_store=session_store,
            session_id=session_id,
            failure_hint="可以稍后重试，或用 /model 切换同类 OpenAI-compatible 后端",
        )
        return 0 if ok else 1

    if not session_id:
        session_id = session_store.start_session(project=args.project)
    print_chat_home(session_id=session_id, project=args.project, model_id=active_model_id(args))
    print_chat_shortcuts()
    if args.resume:
        payload = session_store.resume_payload(session_id=session_id, project=args.project)
        if payload["status"] == "ok" and payload["recent_turns"]:
            print_resume_preview(payload)
    session_ref: dict[str, str | None] = {"id": session_id}
    auto_stop_event = start_auto_background_loop(args=args, session_store=session_store, session_ref=session_ref)
    while True:
        if handle_pending_auto_question(args, session_store=session_store, session_ref=session_ref):
            session_id = session_ref.get("id") or session_id
            continue
        try:
            model_short = active_model_id(args).split(":", 1)[-1]
            project_short = args.project or "no-project"
            prompt = (
                f"{Colors.CYAN}aether{Colors.RESET}"
                f"{Colors.DIM}[{Colors.MAGENTA}{project_short}{Colors.DIM}|{Colors.YELLOW}{model_short}{Colors.DIM}]"
                f"{Colors.RESET} › "
            )
            line = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            auto_stop_event.set()
            print_chat_resume_hint(session_id=session_id, project=args.project)
            return 0
        if line in {"/exit", "exit", "quit", ":q"}:
            auto_stop_event.set()
            print_chat_resume_hint(session_id=session_id, project=args.project)
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
        if line.startswith("/status"):
            raw_status = line[len("/status") :].strip()
            print_chat_status(
                session_store=session_store,
                session_id=session_id,
                project=args.project,
                args=args,
                json_output=raw_status == "--json",
            )
            continue
        if line == "/sessions":
            print_chat_sessions(session_store=session_store, project=args.project)
            continue
        if line.startswith("/new"):
            raw_project = line[len("/new") :].strip()
            if raw_project:
                args.project = None if raw_project in {"none", "clear", "no-project"} else raw_project
            session_id = session_store.start_session(project=args.project)
            session_ref["id"] = session_id
            print(f"{Colors.GREEN}new session{Colors.RESET}: {session_id} project={args.project or 'none'}")
            continue
        if line.startswith("/resume"):
            session_id = handle_chat_resume_command(line, args, session_store, session_id)
            session_ref["id"] = session_id
            continue
        if line == "/continue":
            _, session_id = handle_chat_continue_command(args, session_store, session_id)
            session_ref["id"] = session_id
            continue
        if line.startswith("/history"):
            handle_chat_history_command(line, session_store=session_store, session_id=session_id)
            continue
        if line.startswith("/rename"):
            handle_chat_rename_command(line, session_store=session_store, session_id=session_id)
            continue
        if line == "/preload":
            handle_preload(argparse.Namespace(project=args.project, probe_cluster=False, json=False))
            continue
        if line == "/context":
            print_chat_context_status(session_store=session_store, session_id=session_id)
            continue
        if line.startswith("/compact"):
            handle_chat_compact_command(line, session_store=session_store, session_id=session_id)
            continue
        if line.startswith("/model"):
            handle_chat_model_command(line, args)
            continue
        if line.startswith("/auto"):
            _, session_id = handle_chat_auto_command(line, args, session_store, session_id)
            session_ref["id"] = session_id
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
            session_ref["id"] = session_id
            continue
        if line.startswith("/recommend"):
            from .recommendations import recommend_next_tasks

            focus = line[len("/recommend") :].strip() or None
            print_json({"recommendations": recommend_next_tasks(args.project, focus=focus)})
            continue
        _, session_id = run_chat_model_turn(
            line,
            args=args,
            session_store=session_store,
            session_id=session_id,
            failure_hint="本 session 仍然保留；输入 /model 打开模型选择器后继续，或稍后重试",
        )
        session_ref["id"] = session_id


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


def handle_followup_schedule(args: argparse.Namespace) -> int:
    from .followups import schedule_followup

    result = schedule_followup(
        project=args.project,
        prompt=args.prompt,
        due_at=args.due_at,
        interval_minutes=args.interval_minutes,
        title=args.title,
        related_job_id=args.job_id,
        related_run_id=args.run_id,
    )
    print_json(result)
    return 0 if result.get("status") == "ok" else 1


def handle_followup_list(args: argparse.Namespace) -> int:
    from .followups import list_followups

    print_json(list_followups(project=args.project, include_done=args.include_done, limit=args.limit))
    return 0


def handle_followup_due(args: argparse.Namespace) -> int:
    from .followups import due_followups

    print_json(due_followups(project=args.project, now=args.now, limit=args.limit))
    return 0


def handle_followup_complete(args: argparse.Namespace) -> int:
    from .followups import complete_followup

    result = complete_followup(
        args.followup_id,
        project=args.project,
        status=args.status,
        note=args.note,
        reschedule=not args.no_reschedule,
    )
    print_json(result)
    return 0 if result.get("status") in {"ok", "rescheduled"} else 1


def handle_auto_status(args: argparse.Namespace) -> int:
    payload = _auto_status_with_daemon(args.project)
    if getattr(args, "json", False):
        print_json(payload)
    else:
        print_auto_preview(payload)
    return 0


def _auto_status_with_daemon(project: str | None) -> dict[str, Any]:
    from .auto_mode import auto_mode_status

    payload = auto_mode_status(project=project, include_due=True)
    payload["daemon"] = _auto_daemon_status(project, event_limit=3)
    return payload


def _prepare_auto_launcher_turn_args(args: argparse.Namespace) -> argparse.Namespace:
    turn_args = copy.copy(args)
    defaults = {
        "model": None,
        "max_tokens": None,
        "max_steps": 6,
    }
    for name, value in defaults.items():
        if not hasattr(turn_args, name):
            setattr(turn_args, name, value)
    return turn_args


def run_auto_initial_due_from_launcher(args: argparse.Namespace) -> dict[str, Any]:
    """Run the just-scheduled first /auto pass from the top-level launcher."""

    from .session_store import AetherSessionStore

    turn_args = auto_turn_args(_prepare_auto_launcher_turn_args(args))
    session_store = AetherSessionStore()
    resumed = session_store.resume_payload(project=args.project)
    session_ref = {"id": resumed.get("session_id") if resumed.get("status") == "ok" else None}
    return run_auto_due_once(
        args=turn_args,
        session_store=session_store,
        session_ref=session_ref,
        quiet=False,
        interactive_questions=True,
    )


def maybe_start_auto_after_enable(args: argparse.Namespace, result: dict[str, Any]) -> dict[str, Any]:
    if result.get("status") != "ok":
        return {"status": "skipped", "reason": "configure_failed"}
    if getattr(args, "json", False):
        return {"status": "skipped", "reason": "json_mode"}
    if getattr(args, "no_start", False):
        print(f"{Colors.DIM}  auto start skipped (--no-start); scheduled follow-up remains due.{Colors.RESET}")
        return {"status": "skipped", "reason": "no_start"}
    print(f"{Colors.CYAN}[auto]{Colors.RESET} starting first autonomous pass now…")
    return run_auto_initial_due_from_launcher(args)


def configure_auto_enabled_from_args(args: argparse.Namespace, *, goal: str) -> dict[str, Any]:
    from .auto_mode import configure_auto_mode

    return configure_auto_mode(
        project=args.project,
        enabled=True,
        research_goal=goal,
        monitor_interval_hours=args.monitor_interval_hours,
        daily_report_time=args.daily_report_time,
        allow_cluster_submit=args.allow_cluster_submit,
        allow_structure_build=not args.no_structure_build,
        allow_literature_search=not args.no_literature,
        allow_research_writeback=not args.no_writeback,
        reset_questions=True,
    )


def print_auto_configured_and_maybe_start(args: argparse.Namespace, result: dict[str, Any]) -> int:
    if getattr(args, "json", False):
        print_json(result)
        return 0 if result.get("status") == "ok" else 1
    print_auto_preview(result)
    if result.get("status") != "ok":
        return 1
    print(
        f"{Colors.DIM}  scheduled: initial advance now, "
        f"monitor every {result['state'].get('monitor_interval_hours')}h, "
        f"daily report {result['state'].get('daily_report_time')}{Colors.RESET}"
    )
    started = maybe_start_auto_after_enable(args, result)
    return 1 if started.get("status") == "error" else 0


def handle_auto_switch(args: argparse.Namespace) -> int:
    """Top-level ``aether auto`` switch.

    The interactive slash command already treats /auto as a switch.  This keeps
    the non-interactive launcher aligned with that mental model while retaining
    ``auto on/status/off`` as explicit scripting aliases.
    """

    from .auto_mode import auto_mode_status, configure_auto_mode, infer_research_goal
    from .session_store import AetherSessionStore

    payload = auto_mode_status(project=args.project, include_due=True)
    state = payload.get("state") or {}
    if state.get("enabled"):
        result = configure_auto_mode(project=args.project, enabled=False)
        if getattr(args, "json", False):
            print_json(result)
        else:
            print_auto_preview(result)
        return 0 if result.get("status") == "ok" else 1

    goal = str(state.get("research_goal") or "").strip()
    inferred: dict[str, Any] = {}
    if not goal:
        inferred = infer_research_goal(project=args.project, session_store=AetherSessionStore())
        goal = str(inferred.get("goal") or "").strip()
    if not goal and hasattr(sys.stdin, "isatty") and sys.stdin.isatty():
        try:
            goal = input("research goal> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            goal = ""
    if not goal:
        result = {
            "status": "needs_goal",
            "message": "没有从当前项目或会话中找到明确研究目标。请直接输入：aether auto \"<研究目标>\"",
            "state": state,
        }
        if getattr(args, "json", False):
            print_json(result)
        else:
            print_auto_preview(payload)
            print(f"{Colors.YELLOW}{result['message']}{Colors.RESET}")
        return 1
    if inferred.get("source") and not getattr(args, "json", False):
        print(f"{Colors.DIM}auto goal inferred from {inferred.get('source')}: {_shorten_inline(goal, limit=120)}{Colors.RESET}")
    return print_auto_configured_and_maybe_start(args, configure_auto_enabled_from_args(args, goal=goal))


def handle_auto_on(args: argparse.Namespace) -> int:
    return print_auto_configured_and_maybe_start(args, configure_auto_enabled_from_args(args, goal=args.goal))


def handle_auto_off(args: argparse.Namespace) -> int:
    from .auto_mode import configure_auto_mode

    result = configure_auto_mode(project=args.project, enabled=False)
    if getattr(args, "json", False):
        print_json(result)
    else:
        print_auto_preview(result)
    return 0 if result.get("status") == "ok" else 1


def _auto_daemon_scope(project: str | None) -> str:
    text = str(project or "default").strip() or "default"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)[:80] or "default"


def _auto_daemon_paths(project: str | None) -> dict[str, Path]:
    root = ensure_runtime_dir("auto_daemon")
    scope = _auto_daemon_scope(project)
    return {
        "root": root,
        "lock": root / f"{scope}.lock.json",
        "log": root / f"{scope}.jsonl",
    }


def _json_default(value: Any) -> str:
    try:
        return str(value)
    except Exception:
        return repr(value)


def _append_daemon_event(path: Path, event: dict[str, Any]) -> None:
    payload = {"at": datetime.now().astimezone().isoformat(timespec="seconds"), **event}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")


def _read_daemon_lock(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _daemon_pid_status(pid: Any) -> dict[str, Any]:
    """Best-effort liveness check for a daemon lock PID.

    The lock file is the coordination primitive; PID probing is only a product
    diagnostic so users can tell a genuinely running daemon from a stale lock.
    """

    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        return {"status": "unknown", "pid": pid, "message": "lock has no valid pid"}
    if pid_int <= 0:
        return {"status": "unknown", "pid": pid_int, "message": "lock pid is not positive"}
    if pid_int == os.getpid():
        return {"status": "running", "pid": pid_int, "message": "current process"}
    if os.name == "nt":
        try:
            import ctypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid_int)
            if handle:
                kernel32.CloseHandle(handle)
                return {"status": "running", "pid": pid_int, "message": "process exists"}
            error_code = ctypes.get_last_error()
            if error_code == 5:
                return {"status": "running", "pid": pid_int, "message": "process exists but access is denied"}
            return {"status": "stale", "pid": pid_int, "message": f"process not found or inaccessible ({error_code})"}
        except Exception as exc:
            return {"status": "unknown", "pid": pid_int, "message": f"pid probe failed: {exc}"}
    try:
        os.kill(pid_int, 0)
        return {"status": "running", "pid": pid_int, "message": "process exists"}
    except ProcessLookupError:
        return {"status": "stale", "pid": pid_int, "message": "process not found"}
    except PermissionError:
        return {"status": "running", "pid": pid_int, "message": "process exists but access is denied"}
    except Exception as exc:
        return {"status": "unknown", "pid": pid_int, "message": f"pid probe failed: {exc}"}


def _tail_jsonl(path: Path, *, limit: int = 5) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-max(1, limit) :]
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in lines:
        try:
            data = json.loads(line)
            if isinstance(data, dict):
                rows.append(data)
        except json.JSONDecodeError:
            rows.append({"event": "unparseable_log_line", "raw": line[:300]})
    return rows


def _auto_daemon_status(project: str | None, *, event_limit: int = 5) -> dict[str, Any]:
    paths = _auto_daemon_paths(project)
    lock_path = paths["lock"]
    log_path = paths["log"]
    lock = _read_daemon_lock(lock_path) if lock_path.exists() else {}
    lock_process = _daemon_pid_status(lock.get("pid")) if lock else {}
    events = _tail_jsonl(log_path, limit=event_limit)
    last_event = events[-1] if events else {}
    if lock:
        status = "stale_lock" if lock_process.get("status") == "stale" else "locked"
    elif last_event.get("event") == "daemon_stop":
        status = "stopped"
    elif log_path.exists():
        status = "no_lock_with_log"
    else:
        status = "not_started"
    return {
        "status": status,
        "project": project or "",
        "lock_exists": lock_path.exists(),
        "lock_path": str(lock_path),
        "lock": lock,
        "lock_process": lock_process,
        "log_exists": log_path.exists(),
        "log_path": str(log_path),
        "log_size_bytes": log_path.stat().st_size if log_path.exists() else 0,
        "recent_events": events,
    }


def _acquire_auto_daemon_lock(*, project: str | None, force: bool = False) -> dict[str, Any]:
    paths = _auto_daemon_paths(project)
    lock_path = paths["lock"]
    if force and lock_path.exists():
        try:
            lock_path.unlink()
        except OSError as exc:
            return {"status": "error", "message": f"无法清理旧 daemon lock: {exc}", **paths}
    payload = {
        "project": project or "",
        "pid": os.getpid(),
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "lock_path": str(lock_path),
    }
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(str(lock_path), flags)
    except FileExistsError:
        existing = _read_daemon_lock(lock_path)
        process = _daemon_pid_status(existing.get("pid")) if existing else {}
        status = "stale_lock" if process.get("status") == "stale" else "locked"
        return {
            "status": status,
            "message": "auto daemon lock already exists",
            "existing": existing,
            "lock_process": process,
            **paths,
        }
    except OSError as exc:
        return {"status": "error", "message": str(exc), **paths}
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    return {"status": "ok", "lock": payload, **paths}


def _release_auto_daemon_lock(lock_result: dict[str, Any]) -> None:
    if lock_result.get("status") != "ok":
        return
    lock_path = lock_result.get("lock")
    path = lock_result.get("lock_path") or (lock_path.get("lock_path") if isinstance(lock_path, dict) else "")
    if not path:
        path_obj = lock_result.get("lock")
        if isinstance(path_obj, Path):
            path = str(path_obj)
    try:
        Path(str(path)).unlink(missing_ok=True)
    except Exception:
        pass


def handle_auto_daemon(args: argparse.Namespace) -> int:
    """Run the durable /auto follow-up queue as a long-lived foreground worker.

    This is intentionally a scheduler shell, not a second research workflow.
    The follow-up queue and auto-mode state remain the source of truth; the
    daemon only wakes up, lets the model handle due intents, and sleeps again.
    """

    from .auto_mode import auto_mode_status
    from .session_store import AetherSessionStore

    ensure_console_utf8()
    if bool(getattr(args, "status", False)):
        payload = _auto_daemon_status(args.project, event_limit=int(getattr(args, "event_limit", 5) or 5))
        if getattr(args, "json", False):
            print_json(payload)
        else:
            print(f"{Colors.BOLD}{Colors.CYAN}auto daemon{Colors.RESET}: {payload['status']}")
            print(f"  project : {payload.get('project') or 'default'}")
            print(f"  lock    : {payload['lock_path']} ({'present' if payload['lock_exists'] else 'absent'})")
            if payload.get("lock"):
                print(f"  pid     : {payload['lock'].get('pid')}")
                if payload.get("lock_process"):
                    print(f"  process : {payload['lock_process'].get('status')} ({payload['lock_process'].get('message')})")
                print(f"  started : {payload['lock'].get('started_at')}")
            print(f"  log     : {payload['log_path']} ({payload['log_size_bytes']} bytes)")
            for event in payload.get("recent_events") or []:
                print(f"  - {event.get('at', '?')} {event.get('event', '?')} {event.get('result', '')}")
        return 0
    lock_result = _acquire_auto_daemon_lock(project=args.project, force=bool(getattr(args, "force_lock", False)))
    log_path = Path(str(lock_result.get("log") or _auto_daemon_paths(args.project)["log"]))
    if lock_result.get("status") != "ok":
        payload = {
            "status": lock_result.get("status"),
            "message": lock_result.get("message"),
            "existing": lock_result.get("existing"),
            "lock_process": lock_result.get("lock_process"),
            "lock_path": str(lock_result.get("lock") or ""),
            "log_path": str(log_path),
        }
        _append_daemon_event(log_path, {"event": "daemon_lock_failed", **payload})
        if getattr(args, "json", False):
            print_json(payload)
        else:
            print(f"{Colors.YELLOW}[auto daemon]{Colors.RESET} {payload['message'] or payload['status']}")
            if payload.get("existing"):
                print(f"{Colors.DIM}existing: {payload['existing']}{Colors.RESET}")
            if payload.get("lock_process"):
                print(f"{Colors.DIM}process: {payload['lock_process']}{Colors.RESET}")
            print(f"{Colors.DIM}Use --force-lock only after confirming no other daemon is running or the lock is stale.{Colors.RESET}")
        return 1
    session_store = AetherSessionStore()
    resumed = session_store.resume_payload(project=args.project)
    session_ref = {"id": resumed.get("session_id") if resumed.get("status") == "ok" else None}
    try:
        interval = max(1.0, float(args.interval_seconds))
    except (TypeError, ValueError):
        interval = 300.0
    max_cycles = int(args.max_cycles) if getattr(args, "max_cycles", None) is not None else None
    cycle = 0
    last_result: dict[str, Any] = {"status": "not_started"}
    _append_daemon_event(
        log_path,
        {
            "event": "daemon_start",
            "project": args.project or "",
            "pid": os.getpid(),
            "interval_seconds": interval,
            "session_id": session_ref.get("id"),
        },
    )
    try:
        if not getattr(args, "json", False):
            print(
                f"{Colors.CYAN}[auto daemon]{Colors.RESET} project={args.project or 'default'} "
                f"interval={interval:g}s log={log_path}; Ctrl+C to stop"
            )
        while True:
            cycle += 1
            try:
                status = auto_mode_status(project=args.project, include_due=True)
                state = status.get("state") or {}
                if not state.get("enabled"):
                    last_result = {"status": "auto_disabled", "ran": False, "cycle": cycle, "state": state}
                    _append_daemon_event(log_path, {"event": "cycle_result", "result": last_result})
                    if getattr(args, "json", False):
                        print_json(last_result)
                    else:
                        print(f"{Colors.YELLOW}[auto daemon]{Colors.RESET} /auto is off; enable it with /auto or `aether auto \"<goal>\"`.")
                    return 0
                pending_questions = [str(item).strip() for item in state.get("human_questions") or [] if str(item).strip()]
                if pending_questions:
                    last_result = {
                        "status": "waiting_for_human",
                        "ran": False,
                        "cycle": cycle,
                        "questions": pending_questions,
                    }
                    _append_daemon_event(log_path, {"event": "cycle_result", "result": last_result})
                    if getattr(args, "json", False):
                        print_json(last_result)
                    else:
                        print(f"{Colors.YELLOW}[auto daemon]{Colors.RESET} waiting for human answer:")
                        for question in pending_questions[:3]:
                            print(f"  - {question}")
                    return 0
                turn_args = auto_turn_args(_prepare_auto_launcher_turn_args(args))
                last_result = run_auto_due_once(
                    args=turn_args,
                    session_store=session_store,
                    session_ref=session_ref,
                    quiet=bool(getattr(args, "quiet", False) or getattr(args, "json", False)),
                    interactive_questions=not bool(getattr(args, "no_interactive_questions", False)),
                )
                last_result["cycle"] = cycle
                _append_daemon_event(log_path, {"event": "cycle_result", "result": last_result})
            except Exception as exc:
                last_result = {"status": "error", "ran": False, "cycle": cycle, "message": str(exc)}
                _append_daemon_event(log_path, {"event": "cycle_error", "result": last_result})
                if getattr(args, "once", False) or (max_cycles is not None and cycle >= max_cycles):
                    if getattr(args, "json", False):
                        print_json(last_result)
                    else:
                        print(f"{Colors.RED}[auto daemon]{Colors.RESET} error: {_shorten_inline(str(exc), limit=240)}")
                    return 1
                if not getattr(args, "quiet", False) and not getattr(args, "json", False):
                    print(f"{Colors.YELLOW}[auto daemon]{Colors.RESET} cycle error; will retry: {_shorten_inline(str(exc), limit=180)}")
            if getattr(args, "json", False):
                print_json(last_result)
            elif last_result.get("ran"):
                print(
                    f"{Colors.CYAN}[auto daemon]{Colors.RESET} cycle={cycle} "
                    f"status={last_result.get('status')} completed={last_result.get('completed')}"
                )
            elif not getattr(args, "quiet", False):
                due_count = (status.get("due_followups") or {}).get("count", 0) if "status" in locals() else "unknown"
                print(f"{Colors.DIM}[auto daemon] cycle={cycle} idle; due={due_count}{Colors.RESET}")
            if getattr(args, "once", False) or (max_cycles is not None and cycle >= max_cycles):
                return 1 if last_result.get("status") == "error" else 0
            try:
                time.sleep(interval)
            except KeyboardInterrupt:
                if not getattr(args, "json", False):
                    print(f"\n{Colors.DIM}[auto daemon] stopped by user{Colors.RESET}")
                _append_daemon_event(log_path, {"event": "daemon_stopped", "reason": "keyboard_interrupt", "cycle": cycle})
                return 0
    finally:
        _append_daemon_event(log_path, {"event": "daemon_stop", "project": args.project or "", "pid": os.getpid(), "last_result": last_result})
        _release_auto_daemon_lock(lock_result)


def handle_tools_list(args: argparse.Namespace) -> int:
    from .tool_registry import list_registered_tools

    print_json({"tools": list_registered_tools()})
    return 0


def handle_tools_run(args: argparse.Namespace) -> int:
    from .agent_tools import parse_tool_arguments
    from .tool_registry import AetherToolRegistry

    raw_arguments = args.arguments
    if getattr(args, "arguments_file", None):
        try:
            raw_arguments = Path(args.arguments_file).read_text(encoding="utf-8")
        except OSError as exc:
            print_json({"status": "error", "message": f"无法读取 --arguments-file: {exc}"})
            return 1
    try:
        arguments = parse_tool_arguments(raw_arguments)
    except ValueError as exc:
        print_json({"status": "error", "message": str(exc)})
        return 1
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


def handle_harness_real_case(args: argparse.Namespace) -> int:
    from .real_case_validation import run_real_case_validation

    stream_printer, stream_state = make_stream_printer()
    payload = run_real_case_validation(
        project=args.project,
        model_id=args.model,
        cluster_alias=args.cluster_alias,
        include_outcar=args.include_outcar,
        max_steps=args.max_steps,
        max_tokens=args.max_tokens,
        progress_callback=None if args.json else make_chat_progress_printer(),
        stream_callback=None if args.json else stream_printer,
    )
    if args.json:
        print_json(payload)
    else:
        if stream_state.get("printed"):
            print()
        status = payload.get("status")
        color = Colors.GREEN if status == "ok" else (Colors.YELLOW if status == "incomplete" else Colors.RED)
        print(f"{color}real-case validation: {status}{Colors.RESET}")
        print(f"project: {payload.get('project')} | model: {payload.get('model_id') or '(current)'}")
        print(f"tools: {', '.join(payload.get('tool_names') or []) or '(none)'}")
        if payload.get("missing_evidence"):
            print(f"{Colors.YELLOW}missing evidence{Colors.RESET}: {', '.join(payload['missing_evidence'])}")
        print(f"record: {payload.get('record_path') or '(none)'}")
        print(f"report: {payload.get('report_path')}")
        if payload.get("response"):
            print()
            print(payload["response"])
    return 0 if payload.get("status") == "ok" else 1


def handle_cluster_import(args: argparse.Namespace) -> int:
    from dft_app.remote.config import write_local_cluster_profile

    payload = write_local_cluster_profile(
        source_ssh_config=Path(args.source),
        alias=args.alias,
        remote_base_dir=args.remote_base_dir,
    )
    print_json(payload)
    return 0


def handle_cluster_list(args: argparse.Namespace) -> int:
    from dft_app.remote.config import list_local_cluster_profiles

    payload = list_local_cluster_profiles()
    if getattr(args, "json", False):
        print_json(payload)
        return 0 if payload.get("status") == "ok" else 1
    print(f"{Colors.CYAN}AETHER project clusters{Colors.RESET}")
    print(f"ssh_config: {payload.get('ssh_config_path') or '(not imported)'}")
    print(f"active: {Colors.YELLOW}{payload.get('active_alias') or '(none)'}{Colors.RESET}")
    clusters = payload.get("clusters") or []
    if not clusters:
        print("项目内还没有可识别集群。运行：aether cluster import-ssh-config --source <你的 SSH config 路径> --alias <Host 别名>")
        return 1
    for item in clusters:
        alias = str(item.get("alias") or "")
        marker = "*" if alias == payload.get("active_alias") else " "
        dup = f" duplicate {item.get('occurrence')}/{item.get('duplicate_count')}" if item.get("duplicate_count", 1) > 1 else ""
        print(
            f"  {marker} {alias:<16} {item.get('user')}@{item.get('hostname')}:{item.get('port')} "
            f"identity={'yes' if item.get('identityfile_configured') else 'no'}{Colors.DIM}{dup}{Colors.RESET}"
        )
    return 0


def handle_cluster_use(args: argparse.Namespace) -> int:
    from dft_app.remote.config import use_local_cluster_profile

    payload = use_local_cluster_profile(args.alias, remote_base_dir=args.remote_base_dir)
    print_json(payload)
    return 0


def handle_cluster_config(args: argparse.Namespace) -> int:
    from dft_app.remote import SSHRemoteRunner

    print_json({"status": "ok", "config": SSHRemoteRunner().describe_config()})
    return 0


def handle_cluster_probe(args: argparse.Namespace) -> int:
    from dft_app.remote import SSHRemoteRunner
    from dft_app.remote.config import config_for_local_cluster_alias

    alias = str(getattr(args, "alias", "") or "").strip()
    runner = SSHRemoteRunner(config=config_for_local_cluster_alias(alias)) if alias else SSHRemoteRunner()
    result = runner.probe()
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
    print_demo_home(args.run_root)
    print()
    print(f"{Colors.DIM}Demo is display-only. Start the real Codex-like chat with: {Colors.GREEN}aether{Colors.RESET}")
    return 0


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


def handle_research_benchmark(args: argparse.Namespace) -> int:
    from .research_benchmark import (
        benchmark_case_records_digest,
        build_benchmark_manifest,
        experiment_matrix_summary,
        load_jsonl,
        recorded_case_suite,
        select_benchmark_cases,
        reference_ablation_traces,
        reference_traces,
        score_benchmark,
        write_benchmark_report,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = args.variants or ["aether_full"]
    input_traces = load_jsonl(args.input) if args.input else []
    suite_records = (
        recorded_case_suite(input_traces)
        if args.input
        else [case.to_dict() for case in select_benchmark_cases(args.suite, args.case_ids)]
    )
    suite_path = output_dir / "case_suite.json"
    suite_path.write_text(
        json.dumps(suite_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if args.plan_only:
        print_json(
            experiment_matrix_summary(
                suite=args.suite,
                model_ids=[args.live_model] if args.live_model else [],
                variants=variants,
                repeats=args.repeats,
                max_steps=args.max_steps,
                case_timeout_seconds=args.case_timeout_seconds,
                shard_count=args.shard_count,
                case_ids=args.case_ids,
            )
        )
        return 0
    traces = []
    traces_path = output_dir / "traces.jsonl"
    if args.resume and traces_path.exists():
        traces.extend(load_jsonl(traces_path))
    completed_episode_keys = {
        str(trace.get("episode_key") or "") for trace in traces if str(trace.get("episode_key") or "")
    }
    if args.reference_fixtures:
        traces = reference_traces() + reference_ablation_traces()
    elif args.input:
        traces.extend(input_traces)
    else:
        from .research_benchmark_live import run_live_research_benchmark

        traces.extend(
            run_live_research_benchmark(
                model_id=args.live_model,
                output_dir=output_dir / "live" / args.live_model.replace(":", "_"),
                case_ids=args.case_ids,
                max_steps=args.max_steps,
                max_tokens=args.max_tokens,
                case_timeout_seconds=args.case_timeout_seconds,
                variant_names=args.variants,
                repeats=args.repeats,
                suite=args.suite,
                completed_episode_keys=completed_episode_keys,
                shard_index=args.shard_index,
                shard_count=args.shard_count,
            )
        )
    result = score_benchmark(traces)
    traces_path.write_text(
        "".join(json.dumps(trace, ensure_ascii=False) + "\n" for trace in traces),
        encoding="utf-8",
    )
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = write_benchmark_report(result, output_dir / "report.md")
    manifest_path = output_dir / "run_manifest.json"
    manifest_arguments = dict(vars(args))
    if args.input:
        manifest_arguments["suite"] = "recorded_input"
    manifest_payload = build_benchmark_manifest(
        arguments=manifest_arguments,
        source_paths=[
            "aether_dft/research_benchmark.py",
            "aether_dft/research_benchmark_live.py",
            "aether_dft/scientific_state.py",
            "aether_dft/runtime_harness/core.py",
        ],
    )
    manifest_payload["suite_sha256"] = benchmark_case_records_digest(suite_records)
    if args.input:
        manifest_payload["recorded_input_sha256"] = hashlib.sha256(Path(args.input).read_bytes()).hexdigest()
    manifest_path.write_text(
        json.dumps(
            manifest_payload,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print_json(
        {
            "status": "ok",
            "reference_fixtures": bool(args.reference_fixtures),
            "traces_path": str(traces_path),
            "results_path": str(results_path),
            "report_path": str(report_path),
            "manifest_path": str(manifest_path),
            "case_suite_path": str(suite_path),
            "variants": result["variants"],
        }
    )
    return 0


def add_auto_enable_options(parser: argparse.ArgumentParser, *, include_goal: bool = False) -> None:
    if include_goal:
        parser.add_argument("goal")
    parser.add_argument("--project")
    parser.add_argument("--monitor-interval-hours", type=int, default=4)
    parser.add_argument("--daily-report-time", default="18:00")
    parser.add_argument("--allow-cluster-submit", action="store_true", help="允许 auto turn 调用真实 cluster submit；仍受 permission 模式约束。")
    parser.add_argument("--no-structure-build", action="store_true")
    parser.add_argument("--no-literature", action="store_true")
    parser.add_argument("--no-writeback", action="store_true")
    parser.add_argument("--no-start", action="store_true", help="只开启/切换状态，不立即运行首轮 autonomous pass。")
    parser.add_argument("--model", help="首轮 autonomous pass 临时使用的模型；支持 qwen/deepseek 或完整模型 ID。")
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--json", action="store_true", help="输出原始 JSON，供脚本/测试使用。")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROGRAM_COMMAND,
        description=f"{PROGRAM_NAME} v{__version__} | model: {program_model_id()} | conversational DFT research partner.",
    )
    parser.add_argument("--version", action="version", version=f"{PROGRAM_NAME} {__version__} | model: {program_model_id()}")
    sub = parser.add_subparsers(dest="command")

    doctor_parser = sub.add_parser("doctor", help="检查程序名称、版本、模型运行时与项目底座。")
    doctor_parser.add_argument("--model", help="临时检查指定模型；支持 qwen/deepseek 或完整模型 ID。")
    doctor_parser.add_argument("--json", action="store_true", help="只输出机器可读 JSON。")
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
    demo_parser.set_defaults(func=handle_demo)

    benchmark_parser = sub.add_parser("benchmark", help="长期科研连续性、证据和安全边界评估。")
    benchmark_sub = benchmark_parser.add_subparsers(dest="benchmark_command", required=True)
    research_benchmark = benchmark_sub.add_parser("research", help="评估记录的 long-horizon research-agent traces。")
    source = research_benchmark.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="真实 agent episode JSONL。")
    source.add_argument("--reference-fixtures", action="store_true", help="只跑确定性工程 fixtures，不代表真实模型结果。")
    source.add_argument("--live-model", help="运行真实模型与模拟工具的多阶段 benchmark，例如 deepseek:deepseek-v4-pro。")
    research_benchmark.add_argument("--case", action="append", dest="case_ids")
    research_benchmark.add_argument("--variant", action="append", dest="variants")
    research_benchmark.add_argument("--repeats", type=int, default=1)
    research_benchmark.add_argument("--suite", choices=("pilot", "parameterized"), default="pilot")
    research_benchmark.add_argument("--resume", action="store_true")
    research_benchmark.add_argument("--plan-only", action="store_true")
    research_benchmark.add_argument("--shard-count", type=int, default=1)
    research_benchmark.add_argument("--shard-index", type=int, default=0)
    research_benchmark.add_argument("--max-steps", type=int, default=8)
    research_benchmark.add_argument("--max-tokens", type=int, default=1000)
    research_benchmark.add_argument("--case-timeout-seconds", type=float, default=600.0)
    research_benchmark.add_argument("--output-dir", default=".aether/benchmarks/latest")
    research_benchmark.set_defaults(func=handle_research_benchmark)

    model_parser = sub.add_parser("model", help="查看或切换当前模型。")
    model_sub = model_parser.add_subparsers(dest="model_command")
    model_current = model_sub.add_parser("current", help="显示当前模型。")
    model_current.set_defaults(func=handle_model_current)
    model_list = model_sub.add_parser("list", help="列出可用模型；等同于顶层 models。")
    model_list.add_argument("--json", action="store_true")
    model_list.set_defaults(func=handle_models)
    model_set = model_sub.add_parser("set", help="设置默认模型；支持 qwen/deepseek 或完整模型 ID。")
    model_set.add_argument("model_id")
    model_set.set_defaults(func=handle_model_set)
    model_smoke = model_sub.add_parser("smoke", help="真实调用当前/指定模型，验证工具调用后端。")
    model_smoke.add_argument("--model", help="临时使用模型；支持 qwen/deepseek 或完整模型 ID；默认当前模型。")
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

    followup_parser = sub.add_parser("followup", help="项目科研 follow-up / 提醒队列。")
    followup_sub = followup_parser.add_subparsers(dest="followup_command")
    followup_schedule = followup_sub.add_parser("schedule", help="写入一个未来科研检查意图。")
    followup_schedule.add_argument("prompt")
    followup_schedule.add_argument("--project")
    followup_schedule.add_argument("--due-at", dest="due_at", help="ISO 时间，例如 2026-06-18T09:00:00+08:00")
    followup_schedule.add_argument("--interval-minutes", type=int)
    followup_schedule.add_argument("--title")
    followup_schedule.add_argument("--job-id")
    followup_schedule.add_argument("--run-id")
    followup_schedule.set_defaults(func=handle_followup_schedule)
    followup_list = followup_sub.add_parser("list", help="列出未完成 follow-up。")
    followup_list.add_argument("--project")
    followup_list.add_argument("--include-done", action="store_true")
    followup_list.add_argument("--limit", type=int, default=50)
    followup_list.set_defaults(func=handle_followup_list)
    followup_due = followup_sub.add_parser("due", help="列出已到期 follow-up。")
    followup_due.add_argument("--project")
    followup_due.add_argument("--now")
    followup_due.add_argument("--limit", type=int, default=20)
    followup_due.set_defaults(func=handle_followup_due)
    followup_complete = followup_sub.add_parser("complete", help="完成/取消一个 follow-up。")
    followup_complete.add_argument("followup_id")
    followup_complete.add_argument("--project")
    followup_complete.add_argument("--status", default="done")
    followup_complete.add_argument("--note")
    followup_complete.add_argument("--no-reschedule", action="store_true")
    followup_complete.set_defaults(func=handle_followup_complete)

    auto_parser = sub.add_parser(
        "auto",
        help="目标驱动自动科研模式；无子命令时像 /auto 一样作为开关，或直接跟研究目标。",
        description=(
            "目标驱动自动科研模式。常用：aether auto --project MCH-Pt-Br；"
            "aether auto \"验证 MCH 在 Br/Pt 上脱氢最低能路径\" --project MCH-Pt-Br；"
            "旧式脚本入口 auto status/on/off 仍可用。"
        ),
    )
    add_auto_enable_options(auto_parser)
    auto_parser.set_defaults(func=handle_auto_switch)
    auto_parser.epilog = (
        "Shortcut: if the word after 'auto' is not status/on/off, AETHER treats the rest as the research goal. "
        "Example: aether auto \"筛选 CO/Pt(111) 最稳定吸附构型\" --project demo"
    )
    auto_sub = auto_parser.add_subparsers(dest="auto_command", metavar="[status|on|off|daemon]")
    auto_status = auto_sub.add_parser("status", help="查看 /auto 状态。")
    auto_status.add_argument("--project")
    auto_status.add_argument("--json", action="store_true", help="输出原始 JSON，供脚本/测试使用。")
    auto_status.set_defaults(func=handle_auto_status)
    auto_on = auto_sub.add_parser("on", help="开启 /auto 并设置明确研究目标。")
    add_auto_enable_options(auto_on, include_goal=True)
    auto_on.set_defaults(func=handle_auto_on)
    auto_off = auto_sub.add_parser("off", help="关闭 /auto。")
    auto_off.add_argument("--project")
    auto_off.add_argument("--json", action="store_true", help="输出原始 JSON，供脚本/测试使用。")
    auto_off.set_defaults(func=handle_auto_off)
    auto_daemon = auto_sub.add_parser(
        "daemon",
        help="长期运行 /auto follow-up worker；用于每隔几小时监控任务和日报。",
        description=(
            "Foreground worker for /auto. It does not define a fixed research pipeline; "
            "it repeatedly checks the durable follow-up queue and lets the model handle due evidence work."
        ),
    )
    auto_daemon.add_argument("--project")
    auto_daemon.add_argument("--interval-seconds", type=float, default=float(os.getenv("AETHER_AUTO_DAEMON_INTERVAL_SECONDS", "300")))
    auto_daemon.add_argument("--once", action="store_true", help="只检查/推进一次后退出，适合任务计划器调用。")
    auto_daemon.add_argument("--max-cycles", type=int, help="最多循环次数；测试或受控运行使用。")
    auto_daemon.add_argument("--quiet", action="store_true", help="减少心跳输出。")
    auto_daemon.add_argument("--json", action="store_true", help="每轮输出 JSON。")
    auto_daemon.add_argument("--status", action="store_true", help="只查看 daemon lock/log 状态，不启动 worker。")
    auto_daemon.add_argument("--event-limit", type=int, default=5, help="--status 显示的最近事件数量。")
    auto_daemon.add_argument("--force-lock", action="store_true", help="强制清理已有 lock 后启动；仅确认旧进程已退出时使用。")
    auto_daemon.add_argument("--no-interactive-questions", action="store_true", help="后台运行时不要在终端提问；遇到人类问题则保持 pending。")
    auto_daemon.set_defaults(func=handle_auto_daemon)
    tools_parser = sub.add_parser("tools", help="AETHER harness 工具注册表。")
    tools_sub = tools_parser.add_subparsers(dest="tools_command")
    tools_list = tools_sub.add_parser("list", help="列出模型可调用工具。")
    tools_list.set_defaults(func=handle_tools_list)
    tools_run = tools_sub.add_parser("run", help="直接运行一个注册工具。")
    tools_run.add_argument("name")
    tools_run.add_argument("--arguments", default="{}", help="JSON 参数对象。")
    tools_run.add_argument("--arguments-file", help="从 JSON 文件读取参数对象，避免 Windows/PowerShell 引号问题。")
    tools_run.add_argument("--allow-cluster-submit", action="store_true")
    tools_run.set_defaults(func=handle_tools_run)

    chat_parser = sub.add_parser("chat", help="对话式科研合伙人入口；无 prompt 时进入 REPL。")
    chat_parser.add_argument("prompt", nargs="*")
    chat_parser.add_argument("--project")
    chat_parser.add_argument("--model", help="临时使用模型；支持 qwen/deepseek 这类唯一别名或完整模型 ID。")
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

    mainline_parser = sub.add_parser("mainline", help="兼容入口：模型主导科研回合（推荐直接运行 aether-dft 进入 REPL）。")
    mainline_parser.add_argument("prompt", nargs="*")
    mainline_parser.add_argument("--project")
    mainline_parser.add_argument("--model", help="临时使用模型；支持 qwen/deepseek 这类唯一别名或完整模型 ID。")
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
    harness_real_case = harness_sub.add_parser(
        "real-case",
        help="真实模型主导的只读课题验收：检查项目上下文、集群配置/队列，可选 OUTCAR 证据。",
    )
    harness_real_case.add_argument("--project", required=True, help="research/project slug，例如 MCH-Pt-Br。")
    harness_real_case.add_argument("--model", help="临时使用模型；支持 qwen/deepseek 或完整模型 ID。")
    harness_real_case.add_argument("--cluster-alias", help="优先让模型使用的项目内 SSH Host alias。")
    harness_real_case.add_argument("--include-outcar", action="store_true", help="要求模型尝试只读 OUTCAR/结果证据。")
    harness_real_case.add_argument("--max-steps", type=int, default=6)
    harness_real_case.add_argument("--max-tokens", type=int, default=1400)
    harness_real_case.add_argument("--json", action="store_true")
    harness_real_case.set_defaults(func=handle_harness_real_case)

    cluster_parser = sub.add_parser("cluster", help="SSH/SLURM 集群配置、探测与导入。")
    cluster_sub = cluster_parser.add_subparsers(dest="cluster_command")
    cluster_import = cluster_sub.add_parser("import-ssh-config", help="复制本机 SSH config 到项目 .secrets，并设置默认集群 alias。")
    cluster_import.add_argument("--source", default=str(Path.home() / ".ssh" / "config"))
    cluster_import.add_argument("--alias", required=True)
    cluster_import.add_argument("--remote-base-dir", help="远程 run 根目录，默认 /home/<user>/aether-dft-runs。")
    cluster_import.set_defaults(func=handle_cluster_import)
    cluster_list = cluster_sub.add_parser("list", help="列出项目内 SSH config 可识别的集群 Host。")
    cluster_list.add_argument("--json", action="store_true")
    cluster_list.set_defaults(func=handle_cluster_list)
    cluster_use = cluster_sub.add_parser("use", help="选择项目内 SSH config 的 active 集群 alias。")
    cluster_use.add_argument("alias")
    cluster_use.add_argument("--remote-base-dir", help="远程 run 根目录，默认 /home/<user>/aether-dft-runs。")
    cluster_use.set_defaults(func=handle_cluster_use)
    cluster_config = cluster_sub.add_parser("config", help="显示当前集群配置摘要，不暴露私钥/API key。")
    cluster_config.set_defaults(func=handle_cluster_config)
    cluster_probe = cluster_sub.add_parser("probe", help="真实 SSH 探测集群连通性和 sbatch/squeue/vasp_std。")
    cluster_probe.add_argument("--alias", help="不切换 active 集群，直接探测项目内某个 SSH Host alias。")
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
    ssh_parser.add_argument("--alias", help="不切换 active 集群，直接探测项目内某个 SSH Host alias。")
    ssh_parser.set_defaults(func=handle_cluster_probe)

    agent_parser = sub.add_parser("agent", help="让 qwen/OpenAI-compatible 模型通过工具调用 AETHER-DFT/集群。")
    agent_parser.add_argument("prompt", nargs="+")
    agent_parser.add_argument("--project")
    agent_parser.add_argument("--model", help="临时使用模型；支持 qwen/deepseek 或完整模型 ID。")
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
    ask_parser.add_argument("--model", help="临时使用模型；支持 qwen/deepseek 或完整模型 ID。")
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


def normalize_auto_argv(raw_args: list[str]) -> list[str]:
    """Normalize human-friendly ``aether auto <goal>`` into the legacy parser.

    This is a CLI affordance, not a research workflow: it only maps the user's
    natural entrypoint to the existing ``auto on`` command so all durable /auto
    behavior remains model/tool driven.
    """

    if not raw_args or raw_args[0] != "auto":
        return raw_args
    if len(raw_args) == 1:
        return raw_args
    first = raw_args[1]
    explicit = {"status", "on", "off", "daemon", "-h", "--help"}
    if first in explicit or first.startswith("-"):
        return raw_args
    goal_tokens: list[str] = []
    rest: list[str] = []
    in_rest = False
    for token in raw_args[1:]:
        if not in_rest and token.startswith("-"):
            in_rest = True
        if in_rest:
            rest.append(token)
        else:
            goal_tokens.append(token)
    goal = " ".join(goal_tokens).strip()
    if not goal:
        return raw_args
    return ["auto", "on", goal, *rest]


def main(argv: list[str] | None = None) -> int:
    ensure_console_utf8()
    raw_args = normalize_auto_argv(list(sys.argv[1:] if argv is None else argv))
    if not raw_args:
        if hasattr(sys.stdin, "isatty") and sys.stdin.isatty():
            return handle_chat(
                argparse.Namespace(
                    prompt=[],
                    project=None,
                    model=None,
                    max_tokens=None,
                    max_steps=6,
                    resume=False,
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
    if raw_args[:2] == ["chat", "--help"]:
        print_chat_cli_help()
        return 0
    if raw_args[:2] == ["chat", "--help-advanced"]:
        raw_args = ["chat", "--help", *raw_args[2:]]
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
