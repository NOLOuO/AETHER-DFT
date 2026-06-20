from __future__ import annotations

from typing import Any, Callable

from .agent import run_agent_once


def load_system_prompt() -> str:
    from .prompt_engine import render_compiled_system_prompt

    return render_compiled_system_prompt()


def build_messages(user_text: str, *, project: str | None = None) -> list[dict[str, str]]:
    from .prompt_engine import render_compiled_system_prompt

    system = load_system_prompt() if not project else render_compiled_system_prompt(project=project)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_text},
    ]


def ask_once(
    prompt: str,
    *,
    project: str | None = None,
    model_id: str | None = None,
    max_tokens: int | None = None,
    max_steps: int = 6,
    allow_cluster_submit: bool = False,
    session_id: str | None = None,
    permission_mode: str | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    permission_prompt_callback: Callable[[dict[str, Any]], bool] | None = None,
    stream_callback: Callable[[dict[str, Any]], None] | None = None,
    human_question_callback: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    record = run_agent_once(
        prompt,
        project=project,
        model_id=model_id,
        max_tokens=max_tokens,
        max_steps=max_steps,
        allow_cluster_submit=allow_cluster_submit,
        session_id=session_id,
        permission_mode=permission_mode,
        progress_callback=progress_callback,
        permission_prompt_callback=permission_prompt_callback,
        stream_callback=stream_callback,
        human_question_callback=human_question_callback,
    )
    return record
