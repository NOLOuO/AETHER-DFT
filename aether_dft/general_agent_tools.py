from __future__ import annotations

"""General-purpose research partner primitives.

These helpers deliberately stay small and dependency-free.  When a live
connector is unavailable they return an honest, structured "not connected"
result instead of fabricating search or vision evidence.
"""

import json
import math
import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any


EV_PER_KJ_MOL = 1.0 / 96.4853321233
KB_EV = 8.617333262145e-5
H_EV_S = 4.135667696e-15


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
    """Small deterministic chemistry calculator for discussion-time checks."""

    op = str(operation or "").strip().lower()
    try:
        if op in {"kjmol_to_ev", "kj_mol_to_ev"}:
            value = float(kwargs["value"])
            return {"status": "ok", "operation": op, "input": value, "ev": value * EV_PER_KJ_MOL}
        if op in {"ev_to_kjmol", "ev_to_kj_mol"}:
            value = float(kwargs["value"])
            return {"status": "ok", "operation": op, "input": value, "kj_mol": value / EV_PER_KJ_MOL}
        if op == "boltzmann_population":
            energies = [float(item) for item in kwargs.get("energies_ev", [])]
            temperature = float(kwargs.get("temperature_k") or 298.15)
            if not energies:
                return {"status": "error", "message": "energies_ev 不能为空。"}
            emin = min(energies)
            weights = [math.exp(-(e - emin) / (KB_EV * temperature)) for e in energies]
            total = sum(weights)
            return {
                "status": "ok",
                "operation": op,
                "temperature_k": temperature,
                "relative_energies_ev": [e - emin for e in energies],
                "populations": [w / total for w in weights],
            }
        if op in {"tst_rate", "eyring_rate"}:
            barrier_ev = float(kwargs["barrier_ev"])
            temperature = float(kwargs.get("temperature_k") or 298.15)
            prefactor = KB_EV * temperature / H_EV_S
            rate = prefactor * math.exp(-barrier_ev / (KB_EV * temperature))
            return {"status": "ok", "operation": op, "barrier_ev": barrier_ev, "temperature_k": temperature, "rate_s^-1": rate}
        if op in {"delta_g", "gibbs"}:
            delta_h_ev = float(kwargs.get("delta_h_ev", 0.0))
            delta_s_ev_k = float(kwargs.get("delta_s_ev_k", 0.0))
            temperature = float(kwargs.get("temperature_k") or 298.15)
            return {"status": "ok", "operation": op, "delta_g_ev": delta_h_ev - temperature * delta_s_ev_k, "temperature_k": temperature}
    except (KeyError, TypeError, ValueError) as exc:
        return {"status": "error", "operation": op, "message": str(exc)}
    return {
        "status": "error",
        "operation": op,
        "message": "未知 operation；支持 kjmol_to_ev、ev_to_kjmol、boltzmann_population、tst_rate、delta_g。",
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


def discussion_state_snapshot(
    *,
    project: str | None = None,
    goal: str = "",
    known_facts: list[str] | None = None,
    open_questions: list[str] | None = None,
    next_steps: list[str] | None = None,
    persist_path: str | None = None,
) -> dict[str, Any]:
    """Let the model checkpoint a conversation without imposing a fixed workflow."""

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
        path = Path(persist_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        snapshot["persisted_path"] = str(path)
    return snapshot
