from __future__ import annotations

from pathlib import Path
from typing import Any

from .llm_client import call_llm_race


def generate_incar_advice(
    *,
    task_type: str,
    incar_content: str,
    structure_path: str | Path | None = None,
) -> dict[str, Any]:
    structure_info = _extract_structure_info(structure_path)
    messages = [
        {
            "role": "system",
            "content": (
                "你是 VASP INCAR 参数顾问。"
                "请基于任务类型、结构信息和当前 INCAR，仅对 INCAR 参数给出建议。"
                "重点关注 EDIFFG、NSW、ENCUT、ISMEAR、SIGMA、IBRION、POTIM、ISPIN、MAGMOM、LREAL、PREC、EDIFF。"
                "不要建议修改 POSCAR、KPOINTS、POTCAR 或目录结构。"
                "如果当前参数基本合理，就明确说明“无需明显修改”。"
                "请用中文、简洁、按条列输出。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"任务类型: {task_type}\n"
                f"结构信息: {structure_info}\n\n"
                f"当前 INCAR:\n```\n{incar_content}\n```"
            ),
        },
    ]
    result = call_llm_race(messages, max_tokens=1200, timeout=30)
    return {
        "provider": result["provider"],
        "model": result["model"],
        "advice": result["content"],
        "usage": result.get("usage") or {},
    }


def _extract_structure_info(structure_path: str | Path | None) -> str:
    if structure_path is None:
        return "无"
    path = Path(structure_path)
    if not path.exists():
        return f"结构文件不存在: {path}"
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:
        return f"读取结构文件失败: {exc}"
    if len(lines) >= 7:
        symbols = lines[5].strip()
        counts = lines[6].strip()
        return f"文件={path.name}; 元素={symbols}; 数量={counts}"
    return f"文件={path.name}; 结构文件内容不足，无法提取元素/数量"
