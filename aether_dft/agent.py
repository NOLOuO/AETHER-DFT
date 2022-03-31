from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from aether_dft.runtime_harness.core import AgentHarness
from aether_dft.model_catalog import resolve_effective_model_id, split_model_id
from aether_dft.research_loop import summarize_research_turn
from dft_app.llm import DomesticCopilotLLM as _DomesticCopilotLLM


# Compatibility hook for tests and older callers that monkeypatch the module
# attribute directly.
DomesticCopilotLLM = _DomesticCopilotLLM


class _ModuleAdapter:
    def __init__(self, model_id: str | None = None):
        self.model_id = model_id or resolve_effective_model_id()
        provider_id, model_name = split_model_id(self.model_id)
        self.runtime = type("Runtime", (), {"model_id": f"{provider_id}:{model_name}"})()
        self.llm = DomesticCopilotLLM(Path.cwd())

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = "auto",
        max_tokens: int | None = None,
        stream_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        provider_id, model_name = split_model_id(self.model_id)
        if tools and hasattr(self.llm, "call_messages_with_tools"):
            kwargs: dict[str, Any] = {}
            if stream_callback is not None:
                kwargs["stream_callback"] = stream_callback
            return self.llm.call_messages_with_tools(
                messages,
                tools=tools,
                provider_id=provider_id,
                model_id=model_name,
                max_tokens=max_tokens,
                tool_choice=tool_choice,
                **kwargs,
            )
        kwargs = {}
        if stream_callback is not None:
            kwargs["stream_callback"] = stream_callback
        return self.llm.call_messages_inline(
            messages,
            provider_id=provider_id,
            model_id=model_name,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )


def run_agent_once(
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
) -> dict[str, Any]:
    """Run one Codex-like AETHER harness turn.

    Kept as the CLI-compatible entry point while the implementation now lives
    in ``aether_dft/runtime_harness/core.py`` as required by ``智能体架构.md``.
    """

    harness = AgentHarness(
        adapter=_ModuleAdapter(model_id),
        allow_cluster_submit=allow_cluster_submit,
        permission_mode=permission_mode,
    )
    record = harness.run_turn(
        prompt,
        project=project,
        session_id=session_id,
        max_tokens=max_tokens,
        max_steps=max_steps,
        progress_callback=progress_callback,
        permission_prompt_callback=permission_prompt_callback,
        stream_callback=stream_callback,
    )
    return summarize_research_turn(record, project=project)
