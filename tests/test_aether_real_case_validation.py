from __future__ import annotations

from pathlib import Path
from typing import Any

from aether_dft import real_case_validation


def _runtime_dir(tmp_path: Path):
    def _ensure(*parts: str) -> Path:
        path = tmp_path.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path

    return _ensure


def test_real_case_validation_records_model_led_readonly_evidence(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(real_case_validation, "ensure_runtime_dir", _runtime_dir(tmp_path))
    permission_seen: list[dict[str, Any]] = []

    def fake_ask(prompt: str, **kwargs: Any) -> dict[str, Any]:
        assert "真实课题验收" in prompt
        assert "不要写文件" in prompt
        assert "cluster_alias=szhang" in prompt
        assert kwargs["project"] == "MCH-Pt-Br"
        assert kwargs["permission_mode"] == "ask"
        kwargs["permission_prompt_callback"]({"tool_name": "research_progress_append", "permission_label": "写入"})
        return {
            "model_id": "fake:model",
            "finish_reason": "stop",
            "record_path": str(tmp_path / "transcript.jsonl"),
            "response": "已完成只读验收。",
            "tool_executions": [
                {"name": "project_state_read", "result": {"status": "ok"}},
                {"name": "cluster_profile_list", "result": {"status": "ok"}},
                {"name": "cluster_probe", "result": {"status": "ok"}},
                {"name": "cluster_my_jobs", "result": {"status": "ok"}},
            ],
        }

    result = real_case_validation.run_real_case_validation(
        project="MCH-Pt-Br",
        model_id="fake:model",
        cluster_alias="szhang",
        ask_fn=fake_ask,
        permission_prompt_callback=permission_seen.append,
    )

    assert result["status"] == "ok"
    assert result["evidence"]["project_context"] is True
    assert result["evidence"]["cluster_profile"] is True
    assert result["evidence"]["cluster_live"] is True
    assert result["denied_permissions"][0]["tool_name"] == "research_progress_append"
    assert permission_seen
    assert Path(result["report_path"]).exists()


def test_real_case_validation_marks_missing_outcar_when_requested(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(real_case_validation, "ensure_runtime_dir", _runtime_dir(tmp_path))

    def fake_ask(prompt: str, **kwargs: Any) -> dict[str, Any]:
        assert "OUTCAR" in prompt
        return {
            "model_id": "fake:model",
            "finish_reason": "stop",
            "response": "没有找到 OUTCAR 路径。",
            "tool_executions": [
                {"name": "project_state_read", "result": {"status": "ok"}},
                {"name": "cluster_profile_list", "result": {"status": "ok"}},
                {"name": "cluster_probe", "result": {"status": "ok"}},
            ],
        }

    result = real_case_validation.run_real_case_validation(
        project="MCH-Pt-Br",
        include_outcar=True,
        ask_fn=fake_ask,
    )

    assert result["status"] == "incomplete"
    assert result["missing_evidence"] == ["outcar_read"]
