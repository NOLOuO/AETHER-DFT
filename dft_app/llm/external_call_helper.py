from __future__ import annotations

import json
import sys
from pathlib import Path

from .domestic_copilot_adapter import DomesticCopilotLLM


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        if not isinstance(payload, dict):
            raise RuntimeError("helper 输入必须是 JSON 对象")

        app_root_raw = str(payload.get("app_root") or "").strip()
        llm = DomesticCopilotLLM(Path(app_root_raw) if app_root_raw else None)
        result = llm.call_messages_inline(
            payload.get("messages") or [],
            provider_id=(
                str(payload.get("provider_id")).strip()
                if payload.get("provider_id")
                else None
            ),
            model_id=(
                str(payload.get("model_id")).strip()
                if payload.get("model_id")
                else None
            ),
            max_tokens=(
                int(payload["max_tokens"])
                if payload.get("max_tokens") is not None
                else None
            ),
        )
        sys.stdout.write(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as exc:
        sys.stderr.write(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
