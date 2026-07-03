from pathlib import Path
import json

import pytest

from aether_dft.model_catalog import load_model_catalog, normalize_model_id, split_model_id
from dft_app.llm import DomesticCopilotLLM
from dft_app.llm.llm_client import (
    _chat_tools_to_responses_tools,
    _messages_to_responses_payload,
    call_openai_compatible_result,
)
from dft_app.llm.provider_presets import build_provider_model_config


def test_aether_default_model_is_deepseek_v4_pro(monkeypatch, tmp_path):
    monkeypatch.delenv("AETHER_DFT_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("AETHER_DFT_LLM_MODEL", raising=False)
    monkeypatch.delenv("SEMI_DFT_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("SEMI_DFT_LLM_MODEL", raising=False)
    runtime = DomesticCopilotLLM(tmp_path).describe_runtime()
    assert runtime["default_model"] == {"provider": "deepseek", "model": "deepseek-v4-pro"}


def test_deepseek_v4_pro_uses_thinking_mode_and_1m_context():
    config = build_provider_model_config("deepseek", "deepseek-v4-pro")
    assert config["model"] == "deepseek-v4-pro"
    assert config["base_url"] == "https://api.deepseek.com"
    assert config["api_key_env"] == "DEEPSEEK_API_KEY"
    assert config["context_window"] == 1_000_000
    assert config["extra_body"]["thinking"]["type"] == "enabled"
    assert config["reasoning_effort"] == "max"


def test_qwen37_uses_dashscope_beijing_openai_compatible_endpoint():
    config = build_provider_model_config("bailian", "qwen3.7-max")
    assert config["model"] == "qwen3.7-max"
    assert config["base_url"] == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert config["api_key_env"] == "DASHSCOPE_API_KEY"
    assert config["context_window"] == 1_000_000


def test_model_catalog_only_lists_project_fit_models():
    catalog = load_model_catalog(Path.cwd())
    assert set(catalog) == {"deepseek:deepseek-v4-pro", "bailian:qwen3.7-max"}


def test_model_aliases_resolve_through_catalog_without_provider_specific_paths():
    assert normalize_model_id("qwen") == "bailian:qwen3.7-max"
    assert normalize_model_id("deepseek") == "deepseek:deepseek-v4-pro"
    assert split_model_id("qwen") == ("bailian", "qwen3.7-max")


def test_external_model_provider_config_extends_catalog(monkeypatch, tmp_path):
    config_path = tmp_path / "model_providers.json"
    config_path.write_text(
        json.dumps(
            {
                "providers": {
                    "localai": {
                        "label": "Local AI",
                        "base_url": "http://localhost:8000/v1",
                        "api_key_env": "LOCALAI_API_KEY",
                        "timeout_seconds": 30,
                        "max_tokens": 512,
                        "models": [
                            {
                                "id": "research-model",
                                "label": "Research Model",
                                "api_model": "research-model",
                                "context_window": 64000,
                            }
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AETHER_MODEL_PROVIDERS_PATH", str(config_path))
    monkeypatch.setenv("LOCALAI_API_KEY", "present")

    config = build_provider_model_config("localai", "research-model")
    catalog = load_model_catalog(tmp_path)

    assert config["base_url"] == "http://localhost:8000/v1"
    assert config["model"] == "research-model"
    assert catalog["localai:research-model"].available is True
    assert catalog["localai:research-model"].context_window == 64000


def test_domestic_runtime_reads_aether_model_preferences(monkeypatch, tmp_path):
    monkeypatch.delenv("AETHER_DFT_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("AETHER_DFT_LLM_MODEL", raising=False)
    monkeypatch.delenv("SEMI_DFT_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("SEMI_DFT_LLM_MODEL", raising=False)
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    (runtime_dir / "model-preferences.json").write_text(
        '{"global_default_model_id": "bailian:qwen3.7-max"}',
        encoding="utf-8",
    )
    assert DomesticCopilotLLM(tmp_path).resolve_default_model() == ("bailian", "qwen3.7-max")


def test_unsupported_model_is_rejected():
    with pytest.raises(KeyError):
        build_provider_model_config("deepseek", "deepseek-chat")


def test_responses_payload_conversion_preserves_tool_calls():
    instructions, conversation = _messages_to_responses_payload(
        [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "project_state_read", "arguments": "{\"project\":\"demo\"}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "{\"status\":\"ok\"}"},
        ]
    )

    assert instructions == "system prompt"
    assert conversation[0] == {"role": "user", "content": "hello"}
    assert conversation[1]["type"] == "function_call"
    assert conversation[1]["name"] == "project_state_read"
    assert conversation[2]["type"] == "function_call_output"
    assert conversation[2]["call_id"] == "call_1"


def test_responses_tool_schema_is_flattened():
    tools = _chat_tools_to_responses_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "project_state_read",
                    "description": "read project state",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
    )

    assert tools == [
        {
            "type": "function",
            "name": "project_state_read",
            "description": "read project state",
            "parameters": {"type": "object", "properties": {}},
        }
    ]


def test_chat_completions_streaming_reconstructs_content(monkeypatch):
    captured: dict[str, object] = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured["kwargs"] = kwargs
            return iter(
                [
                    {"choices": [{"delta": {"content": "Hello "}}]},
                    {"choices": [{"delta": {"content": "world"}, "finish_reason": "stop"}]},
                ]
            )

    class FakeClient:
        chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr("dft_app.llm.llm_client._build_openai_client", lambda *args, **kwargs: FakeClient())
    events: list[dict[str, object]] = []

    result = call_openai_compatible_result(
        "deepseek",
        "deepseek-v4-pro",
        "fake-key",
        [{"role": "user", "content": "hello"}],
        stream_callback=events.append,
    )

    assert captured["kwargs"]["stream"] is True
    assert result["content"] == "Hello world"
    assert result["finish_reason"] == "stop"
    assert [event["delta"] for event in events] == ["Hello ", "world"]


def test_chat_completions_streaming_degrades_when_only_reasoning_arrives(monkeypatch):
    class FakeCompletions:
        def create(self, **kwargs):
            return iter(
                [
                    {"choices": [{"delta": {"reasoning_content": "先思考"}}]},
                    {"choices": [{"delta": {"reasoning_content": "但未输出正文"}, "finish_reason": "length"}]},
                ]
            )

    class FakeClient:
        chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr("dft_app.llm.llm_client._build_openai_client", lambda *args, **kwargs: FakeClient())
    events: list[dict[str, object]] = []

    result = call_openai_compatible_result(
        "deepseek",
        "deepseek-v4-pro",
        "fake-key",
        [{"role": "user", "content": "hello"}],
        stream_callback=events.append,
    )

    assert result["finish_reason"] == "length"
    assert "只返回了 reasoning_content" in result["content"]
    assert result["reasoning_content"] == "先思考但未输出正文"
    assert [event["type"] for event in events] == ["reasoning_delta", "reasoning_delta"]


def test_chat_completions_empty_message_degrades_without_throwing(monkeypatch):
    class FakeCompletion:
        def model_dump(self):
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "", "tool_calls": []},
                    }
                ]
            }

    class FakeCompletions:
        def create(self, **kwargs):
            return FakeCompletion()

    class FakeClient:
        chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr("dft_app.llm.llm_client._build_openai_client", lambda *args, **kwargs: FakeClient())

    result = call_openai_compatible_result(
        "deepseek",
        "deepseek-v4-pro",
        "fake-key",
        [{"role": "user", "content": "hello"}],
    )

    assert result["finish_reason"] == "stop"
    assert "模型未返回可展示正文" in result["content"]


def test_qwen_tools_use_same_chat_completions_backend(monkeypatch):
    captured: dict[str, object] = {}

    class FakeCompletion:
        def model_dump(self):
            return {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_qwen",
                                    "type": "function",
                                    "function": {"name": "project_state_read", "arguments": "{\"project\":\"demo\"}"},
                                }
                            ],
                        },
                    }
                ]
            }

    class FakeCompletions:
        def create(self, **kwargs):
            captured["kwargs"] = kwargs
            return FakeCompletion()

    class FakeResponses:
        def create(self, **kwargs):  # pragma: no cover - should never be called
            raise AssertionError("Qwen tools must use the unified Chat Completions backend")

    class FakeClient:
        chat = type("Chat", (), {"completions": FakeCompletions()})()
        responses = FakeResponses()

    monkeypatch.setattr("dft_app.llm.llm_client._build_openai_client", lambda *args, **kwargs: FakeClient())

    result = call_openai_compatible_result(
        "bailian",
        "qwen3.7-max",
        "fake-key",
        [{"role": "user", "content": "read state"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "project_state_read",
                    "description": "read",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )

    assert captured["kwargs"]["model"] == "qwen3.7-max"
    assert captured["kwargs"]["messages"] == [{"role": "user", "content": "read state"}]
    assert captured["kwargs"]["tools"][0]["function"]["name"] == "project_state_read"
    assert captured["kwargs"]["tool_choice"] == "auto"
    assert "input" not in captured["kwargs"]
    assert result["tool_calls"][0]["id"] == "call_qwen"


def test_chat_completions_streaming_reconstructs_tool_call(monkeypatch):
    class FakeCompletions:
        def create(self, **kwargs):
            return iter(
                [
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_1",
                                            "type": "function",
                                            "function": {"name": "project_state_read", "arguments": "{\"project\""},
                                        }
                                    ]
                                }
                            }
                        ]
                    },
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "function": {"arguments": ":\"demo\"}"},
                                        }
                                    ]
                                },
                                "finish_reason": "tool_calls",
                            }
                        ]
                    },
                ]
            )

    class FakeClient:
        chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr("dft_app.llm.llm_client._build_openai_client", lambda *args, **kwargs: FakeClient())

    result = call_openai_compatible_result(
        "deepseek",
        "deepseek-v4-pro",
        "fake-key",
        [{"role": "user", "content": "read state"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "project_state_read",
                    "description": "read",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        stream_callback=lambda _event: None,
    )

    assert result["finish_reason"] == "tool_calls"
    assert result["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "project_state_read", "arguments": "{\"project\":\"demo\"}"},
        }
    ]
