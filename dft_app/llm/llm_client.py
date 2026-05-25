from __future__ import annotations

import json
import os
from typing import Any

from .provider_presets import build_provider_model_config


def extract_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") in {"text", "output_text"} and item.get("text"):
                chunks.append(str(item["text"]))
        return "\n".join(chunks).strip()
    return ""


def extract_reasoning_text(reasoning_details: Any) -> str:
    if not isinstance(reasoning_details, list):
        return ""
    chunks: list[str] = []
    for item in reasoning_details:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if text:
            chunks.append(str(text))
    return "\n".join(chunks).strip()


def maybe_strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def format_error_payload(raw: str) -> str:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if isinstance(payload, dict):
        if isinstance(payload.get("error"), dict) and payload["error"].get("message"):
            return str(payload["error"]["message"])
        if payload.get("message"):
            return str(payload["message"])
    return raw


def _content_to_text(content: Any) -> str:
    return maybe_strip_markdown_fence(extract_message_text(content))


def _messages_to_responses_payload(messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    instructions_parts: list[str] = []
    conversation: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role") or "").strip()
        content = message.get("content")
        if role == "system":
            text = _content_to_text(content)
            if text:
                instructions_parts.append(text)
            continue
        if role == "assistant" and message.get("tool_calls"):
            text = _content_to_text(content)
            if text:
                conversation.append({"role": "assistant", "content": text})
            for raw_call in message.get("tool_calls") or []:
                call_id = str(raw_call.get("id") or raw_call.get("tool_call_id") or "")
                function = raw_call.get("function") or {}
                name = str(function.get("name") or raw_call.get("name") or "").strip()
                arguments_raw = function.get("arguments", raw_call.get("arguments"))
                if isinstance(arguments_raw, dict):
                    arguments = json.dumps(arguments_raw, ensure_ascii=False)
                else:
                    arguments = str(arguments_raw or "{}")
                if not name:
                    continue
                conversation.append(
                    {
                        "type": "function_call",
                        "name": name,
                        "arguments": arguments,
                        "call_id": call_id or name,
                    }
                )
            continue
        if role == "tool":
            call_id = str(
                message.get("tool_call_id")
                or message.get("id")
                or message.get("call_id")
                or ""
            )
            conversation.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": _content_to_text(content),
                }
            )
            continue
        text = _content_to_text(content)
        if text:
            conversation.append({"role": role or "user", "content": text})
    instructions = "\n\n".join(part for part in instructions_parts if part).strip() or None
    return instructions, conversation


def _chat_tools_to_responses_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") != "function":
            continue
        if isinstance(tool.get("function"), dict):
            function = tool["function"]
            normalized.append(
                {
                    "type": "function",
                    "name": str(function.get("name") or "").strip(),
                    "description": str(function.get("description") or ""),
                    "parameters": function.get("parameters") or {"type": "object", "properties": {}},
                }
            )
    return [item for item in normalized if item.get("name")]


def _build_openai_client(api_key: str, base_url: str, timeout: int) -> Any:
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("缺少 openai Python 包，请先安装 openai>=1.57") from exc

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=float(timeout),
    )


def _resolve_api_key(config: dict[str, Any], api_key: str) -> str:
    resolved_key = (api_key or "").strip() or os.getenv(str(config["api_key_env"]), "").strip()
    if not resolved_key:
        raise RuntimeError(f"{config['label']} API Key 未填写")
    return resolved_key


def call_openai_compatible_responses_result(
    provider_id: str,
    model_id: str,
    api_key: str,
    messages: list[dict[str, Any]],
    max_tokens: int | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    config = build_provider_model_config(provider_id, model_id)
    base_url = str(config.get("base_url", "") or "").strip().rstrip("/")
    if not base_url:
        base_url_env = str(config.get("base_url_env", "") or "").strip()
        hint = f"；请设置 {base_url_env}" if base_url_env else ""
        raise RuntimeError(f"{config['label']} OpenAI-compatible base_url 未配置{hint}")

    resolved_key = _resolve_api_key(config, api_key)
    instructions, input_messages = _messages_to_responses_payload(messages)
    request_kwargs: dict[str, Any] = {
        "model": config["model"],
        "input": input_messages,
        "max_output_tokens": max_tokens or config.get("max_tokens", 1600),
    }
    if instructions:
        request_kwargs["instructions"] = instructions
    response_tools = _chat_tools_to_responses_tools(tools or [])
    if response_tools:
        request_kwargs["tools"] = response_tools

    client = _build_openai_client(
        resolved_key,
        base_url,
        int(config.get("timeout_seconds", 180)),
    )

    try:
        completion = client.responses.create(**request_kwargs)
    except Exception as exc:
        raise RuntimeError(f"{config['label']} 接口调用失败: {format_error_payload(str(exc))}") from exc

    data = completion.model_dump() if hasattr(completion, "model_dump") else json.loads(completion.model_dump_json())
    output_items = data.get("output") or []
    tool_calls: list[dict[str, Any]] = []
    for item in output_items:
        if not isinstance(item, dict) or item.get("type") != "function_call":
            continue
        tool_calls.append(
            {
                "id": str(item.get("call_id") or item.get("id") or ""),
                "type": "function",
                "function": {
                    "name": str(item.get("name") or ""),
                    "arguments": str(item.get("arguments") or "{}"),
                },
            }
        )
    content = maybe_strip_markdown_fence(str(getattr(completion, "output_text", "") or "").strip())
    if not content and output_items:
        message_chunks: list[str] = []
        for item in output_items:
            if isinstance(item, dict) and item.get("type") == "message":
                for block in item.get("content") or []:
                    if isinstance(block, dict) and block.get("type") in {"text", "output_text"} and block.get("text"):
                        message_chunks.append(str(block["text"]))
        content = maybe_strip_markdown_fence("\n".join(message_chunks).strip())
    if not content and output_items:
        reasoning_chunks: list[str] = []
        for item in output_items:
            if isinstance(item, dict) and item.get("type") == "reasoning":
                for block in item.get("summary") or []:
                    if isinstance(block, dict) and block.get("text"):
                        reasoning_chunks.append(str(block["text"]))
        content = maybe_strip_markdown_fence("\n".join(reasoning_chunks).strip())
    if not content and not tool_calls:
        status = str(data.get("status") or "unknown")
        content = f"模型未返回正文（status={status}）；可能是输出 token 预算不足或 provider 返回空内容。请提高 max_tokens 或继续追问。"
    return {
        "content": content,
        "finish_reason": "tool_calls" if tool_calls else "stop",
        "tool_calls": tool_calls,
        "raw": data,
    }


def call_openai_compatible_result(
    provider_id: str,
    model_id: str,
    api_key: str,
    messages: list[dict[str, str]],
    max_tokens: int | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = build_provider_model_config(provider_id, model_id)
    base_url = str(config.get("base_url", "") or "").strip().rstrip("/")
    if not base_url:
        base_url_env = str(config.get("base_url_env", "") or "").strip()
        hint = f"；请设置 {base_url_env}" if base_url_env else ""
        raise RuntimeError(f"{config['label']} OpenAI-compatible base_url 未配置{hint}")

    resolved_key = _resolve_api_key(config, api_key)

    if tools and str(config.get("provider_id") or "").lower() == "bailian" and str(config.get("model") or "").startswith("qwen3"):
        return call_openai_compatible_responses_result(
            provider_id,
            model_id,
            resolved_key,
            messages,
            max_tokens=max_tokens,
            tools=tools,
        )

    client = _build_openai_client(
        resolved_key,
        base_url,
        int(config.get("timeout_seconds", 180)),
    )

    body: dict[str, Any] = {
        "model": config["model"],
        "messages": messages,
        "max_tokens": max_tokens or config.get("max_tokens", 1600),
    }
    if config.get("supports_temperature") and config.get("default_temperature") is not None:
        body["temperature"] = config["default_temperature"]
    if config.get("supports_top_p") and config.get("default_top_p") is not None:
        body["top_p"] = config["default_top_p"]
    if config.get("reasoning_effort"):
        body["reasoning_effort"] = config["reasoning_effort"]
    extra_body = dict(config.get("extra_body") or {})

    request_kwargs = dict(body)
    if extra_body:
        request_kwargs["extra_body"] = extra_body
    if tools:
        request_kwargs["tools"] = tools
        request_kwargs["tool_choice"] = tool_choice or "auto"

    try:
        completion = client.chat.completions.create(**request_kwargs)
    except Exception as exc:
        raise RuntimeError(f"{config['label']} 接口调用失败: {format_error_payload(str(exc))}") from exc

    data = completion.model_dump() if hasattr(completion, "model_dump") else json.loads(completion.model_dump_json())
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("模型接口未返回 choices")

    choice = choices[0]
    message = choice.get("message") or {}
    tool_calls = message.get("tool_calls") or []
    reasoning_content = extract_message_text(message.get("reasoning_content"))
    content = maybe_strip_markdown_fence(extract_message_text(message.get("content")))
    if not content:
        content = maybe_strip_markdown_fence(
            extract_reasoning_text(message.get("reasoning_details"))
        )
    if not content and not tool_calls:
        raise RuntimeError("模型接口未返回可展示内容")
    return {
        "content": content,
        "reasoning_content": reasoning_content,
        "finish_reason": choice.get("finish_reason"),
        "tool_calls": tool_calls,
        "raw": data,
    }


def call_openai_compatible(
    provider_id: str,
    model_id: str,
    api_key: str,
    messages: list[dict[str, str]],
    max_tokens: int | None = None,
) -> str:
    result = call_openai_compatible_result(provider_id, model_id, api_key, messages, max_tokens=max_tokens)
    return str(result["content"])
