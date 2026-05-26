from __future__ import annotations

"""General-purpose research partner primitives.

These helpers deliberately stay small and dependency-free.  When a live
connector is unavailable they return an honest, structured "not connected"
result instead of fabricating search or vision evidence.
"""

import json
import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any


def web_search(query: str, *, max_results: int = 5, live: bool | None = None) -> dict[str, Any]:
    """Return a safe web-search request envelope.

    The production harness may route this to an MCP/browser connector.  The
    local fallback is intentionally non-authoritative: it gives the model
    reproducible query URLs and requires a live connector before treating
    external facts as evidence.
    """

    query = " ".join(str(query or "").split())
    if not query:
        return {"status": "error", "message": "query 不能为空。"}
    max_results = max(1, min(int(max_results or 5), 10))
    should_live = bool(live) or os.getenv("AETHER_ENABLE_LIVE_WEB_SEARCH", "").lower() in {"1", "true", "yes"}
    encoded = urllib.parse.quote_plus(query)
    if not should_live:
        return {
            "status": "ok",
            "mode": "connector_required",
            "query": query,
            "max_results": max_results,
            "results": [],
            "search_urls": {
                "google_scholar": f"https://scholar.google.com/scholar?q={encoded}",
                "google": f"https://www.google.com/search?q={encoded}",
                "bing": f"https://www.bing.com/search?q={encoded}",
            },
            "guidance": "当前本地 harness 没有接入 live web connector；不要把这个空结果当成事实，只能把 query_urls 交给可联网检索层。",
        }

    # Best-effort arXiv-style Atom parsing for pages that expose Atom/XML.
    return {
        "status": "ok",
        "mode": "live_connector_not_configured",
        "query": query,
        "max_results": max_results,
        "results": [],
        "guidance": "已请求 live=true，但仓库内未绑定通用网页搜索 MCP；请由外层连接器接管。",
    }


def literature_search(query: str, *, max_results: int = 5, source: str = "arxiv", live: bool | None = None) -> dict[str, Any]:
    """Search literature in a connector-friendly way with an arXiv fallback."""

    query = " ".join(str(query or "").split())
    if not query:
        return {"status": "error", "message": "query 不能为空。"}
    max_results = max(1, min(int(max_results or 5), 20))
    source = str(source or "arxiv").lower()
    encoded = urllib.parse.quote_plus(query)
    should_live = bool(live) or os.getenv("AETHER_ENABLE_LIVE_LITERATURE_SEARCH", "").lower() in {"1", "true", "yes"}
    query_urls = {
        "arxiv": f"https://export.arxiv.org/api/query?search_query=all:{encoded}&start=0&max_results={max_results}",
        "semantic_scholar": f"https://www.semanticscholar.org/search?q={encoded}&sort=relevance",
        "google_scholar": f"https://scholar.google.com/scholar?q={encoded}",
    }
    if not should_live:
        return {
            "status": "ok",
            "mode": "connector_required",
            "query": query,
            "source": source,
            "results": [],
            "query_urls": query_urls,
            "guidance": "未启用 live 文献检索；模型必须调用外部检索/浏览连接器或让用户提供文献，不能凭空引用。",
        }
    if source != "arxiv":
        return {
            "status": "ok",
            "mode": "live_connector_not_configured",
            "query": query,
            "source": source,
            "results": [],
            "query_urls": query_urls,
            "guidance": f"{source} 需要外层文献 connector；本地只内置 arXiv Atom fallback。",
        }
    try:
        with urllib.request.urlopen(query_urls["arxiv"], timeout=8) as response:  # nosec B310 - opt-in live query
            raw = response.read(600_000)
    except Exception as exc:
        return {
            "status": "error",
            "mode": "live_arxiv_failed",
            "query": query,
            "message": str(exc),
            "query_urls": query_urls,
        }
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(raw)
    results: list[dict[str, Any]] = []
    for entry in root.findall("atom:entry", ns)[:max_results]:
        authors = [node.findtext("atom:name", default="", namespaces=ns) for node in entry.findall("atom:author", ns)]
        results.append(
            {
                "title": " ".join((entry.findtext("atom:title", default="", namespaces=ns) or "").split()),
                "summary": " ".join((entry.findtext("atom:summary", default="", namespaces=ns) or "").split())[:900],
                "url": entry.findtext("atom:id", default="", namespaces=ns),
                "published": entry.findtext("atom:published", default="", namespaces=ns),
                "authors": [item for item in authors if item],
            }
        )
    return {"status": "ok", "mode": "live_arxiv", "query": query, "source": source, "results": results, "query_urls": query_urls}


def chemistry_compute(operation: str, **kwargs: Any) -> dict[str, Any]:
    """Small deterministic chemistry calculator for discussion-time checks.

    The public tool remains model-guided rather than a fixed workflow: callers
    may use the older ``operation=...`` vocabulary or the newer
    ``mode=convert|boltzmann|gibbs|tst_rate|kBT`` vocabulary.  The richer
    implementation lives in :mod:`aether_dft.chemistry_compute`; this wrapper
    preserves old result keys so existing prompts/tests do not break.
    """

    op = str(operation or kwargs.get("mode") or "").strip()
    op_norm = op.lower()
    params = dict(kwargs)
    params.pop("mode", None)

    def enhanced(mode: str, **extra: Any) -> dict[str, Any]:
        from .chemistry_compute import compute

        result = compute(mode, **extra)
        result.setdefault("operation", op_norm)
        return result

    try:
        if op_norm in {"convert"}:
            return enhanced("convert", **params)
        if op_norm in {"kjmol_to_ev", "kj_mol_to_ev"}:
            result = enhanced("convert", value=params["value"], from_unit="kJ/mol", to_unit="eV")
            if result.get("status") == "ok":
                result["ev"] = result.get("result")
            return result
        if op_norm in {"ev_to_kjmol", "ev_to_kj_mol"}:
            result = enhanced("convert", value=params["value"], from_unit="eV", to_unit="kJ/mol")
            if result.get("status") == "ok":
                result["kj_mol"] = result.get("result")
            return result
        if op_norm in {"boltzmann", "boltzmann_population"}:
            energies = params.get("energies")
            if energies is None:
                energies = params.get("energies_ev")
            result = enhanced(
                "boltzmann",
                energies=energies or [],
                temperature_k=params.get("temperature_k") or 298.15,
                energy_unit=params.get("energy_unit") or "eV",
                reference_energy=params.get("reference_energy"),
            )
            if result.get("status") == "ok":
                result["populations"] = result.get("result")
                if str(params.get("energy_unit") or "eV").lower() == "ev":
                    result["relative_energies_ev"] = [
                        float(e) - min(float(item) for item in (energies or [])) for e in (energies or [])
                    ]
            return result
        if op_norm in {"tst_rate", "eyring_rate"}:
            activation_energy = params.get("activation_energy", params.get("barrier_ev"))
            result = enhanced(
                "tst_rate",
                activation_energy=activation_energy,
                temperature_k=params.get("temperature_k") or 298.15,
                energy_unit=params.get("energy_unit") or "eV",
                prefactor_hz=params.get("prefactor_hz"),
                transmission_coefficient=params.get("transmission_coefficient") or 1.0,
            )
            if result.get("status") == "ok":
                result["rate_s^-1"] = result.get("result")
                result["barrier_ev"] = result.get("activation_energy_ev")
            return result
        if op_norm in {"delta_g", "gibbs"}:
            result = enhanced(
                "gibbs",
                enthalpy=params.get("enthalpy", params.get("delta_h_ev", 0.0)),
                entropy=params.get("entropy", params.get("delta_s_ev_k", 0.0)),
                temperature_k=params.get("temperature_k") or 298.15,
                enthalpy_unit=params.get("enthalpy_unit") or "eV",
                entropy_unit=params.get("entropy_unit") or "eV/K",
            )
            if result.get("status") == "ok":
                result["delta_g_ev"] = result.get("result")
            return result
        if op_norm in {"kbt", "kbt()"}:
            return enhanced("kBT", temperature_k=params.get("temperature_k") or 298.15, unit=params.get("unit") or "eV")
    except (KeyError, TypeError, ValueError) as exc:
        return {"status": "error", "operation": op_norm, "message": str(exc)}
    return {
        "status": "error",
        "operation": op_norm,
        "message": "未知 operation/mode；支持 convert、kjmol_to_ev、ev_to_kjmol、boltzmann_population/boltzmann、tst_rate、delta_g/gibbs、kBT。",
    }


def image_understand(image_path: str, *, prompt: str = "") -> dict[str, Any]:
    """Connector envelope for image understanding, with file-level sanity checks."""

    if not str(image_path or "").strip():
        return {"status": "error", "message": "image_path 不能为空。"}
    path = Path(image_path)
    if not path.exists() or not path.is_file():
        return {"status": "missing", "image_path": str(path), "message": "图片文件不存在或不是文件。"}
    header = path.read_bytes()[:32]
    kind = "unknown"
    if header.startswith(b"\x89PNG"):
        kind = "png"
    elif header.startswith(b"\xff\xd8"):
        kind = "jpeg"
    elif header[:6] in {b"GIF87a", b"GIF89a"}:
        kind = "gif"
    return {
        "status": "ok",
        "mode": "vision_connector_required",
        "image_path": str(path),
        "bytes": path.stat().st_size,
        "detected_format": kind,
        "prompt": prompt,
        "guidance": "本地工具只验证图片可读；真正图像理解需外层 vision/MCP connector。不要伪造视觉结论。",
    }


def _safe_persist_path(persist_path: str | None, *, project: str | None = None) -> Path | None:
    """Scope optional model-written snapshots to AETHER runtime/project storage."""

    if not persist_path:
        return None
    from .paths import ensure_runtime_dir
    from .project_state import project_paths

    target = Path(persist_path).expanduser().resolve()
    allowed_roots = [ensure_runtime_dir("discussion_snapshots").resolve()]
    if project:
        allowed_roots.append((project_paths(project).root / "discussion_snapshots").resolve())
    if not any(target == root or root in target.parents for root in allowed_roots):
        raise ValueError(
            "persist_path 超出允许范围；只能写入 runtime/discussion_snapshots 或项目 discussion_snapshots 目录。"
        )
    return target


def discussion_state_snapshot(
    *,
    project: str | None = None,
    goal: str = "",
    title: str = "",
    summary: str = "",
    consensus: list[str] | None = None,
    known_facts: list[str] | None = None,
    open_questions: list[str] | None = None,
    next_steps: list[str] | None = None,
    tags: list[str] | None = None,
    persist_path: str | None = None,
    write_to_project_state: bool = False,
) -> dict[str, Any]:
    """Let the model checkpoint a conversation without imposing a fixed workflow."""

    if title or summary or consensus or tags or write_to_project_state:
        from .discussion_snapshot import capture_discussion_snapshot

        merged_consensus = [*(consensus or []), *(known_facts or [])]
        result = capture_discussion_snapshot(
            project=project,
            title=title or goal or "讨论快照",
            summary=summary or goal,
            consensus=merged_consensus,
            open_questions=open_questions,
            next_steps=next_steps,
            tags=tags,
            write_to_project_state=write_to_project_state,
        )
        if persist_path and result.get("status") in {"ok", "warning"}:
            try:
                path = _safe_persist_path(persist_path, project=project)
                assert path is not None
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(result.get("snapshot") or result, ensure_ascii=False, indent=2), encoding="utf-8")
                result["persisted_path"] = str(path)
            except Exception as exc:
                result = {**result, "status": "warning", "message": f"快照已保存，但 persist_path 被拒绝: {exc}"}
        return result

    snapshot = {
        "status": "ok",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "project": project or "",
        "goal": goal.strip(),
        "known_facts": known_facts or [],
        "open_questions": open_questions or [],
        "next_steps": next_steps or [],
    }
    if persist_path:
        try:
            path = _safe_persist_path(persist_path, project=project)
            assert path is not None
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
            snapshot["persisted_path"] = str(path)
        except Exception as exc:
            snapshot["status"] = "warning"
            snapshot["message"] = f"persist_path 被拒绝: {exc}"
    return snapshot
