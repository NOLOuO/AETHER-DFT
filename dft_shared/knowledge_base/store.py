"""知识库存储引擎。

职责：CRUD + 去重 + 索引重建。
数据目录可配置，默认在项目根目录下 .aether/knowledge_data/。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import TaskRecord


class KnowledgeStore:
    """知识库存储实例——所有工具共享同一个实例即可互通。"""

    def __init__(self, data_dir: Path | str | None = None):
        if data_dir is None:
            data_dir = Path(__file__).resolve().parents[2] / ".aether" / "knowledge_data"
        self.data_dir = Path(data_dir)
        self.history_path = self.data_dir / "run_history.jsonl"
        self.snapshots_dir = self.data_dir / "task_snapshots"
        self.index_path = self.data_dir / "index.json"
        self.failure_library_path = self.data_dir / "failure_library.json"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

    # ---- 写入 ----

    def ingest(self, record: TaskRecord) -> dict[str, Any]:
        """写入一条任务记录。去重：signature 相同则跳过追加，但更新快照。"""
        self.ensure_dirs()

        record.parsed_at = datetime.now().astimezone().isoformat(timespec="seconds")
        record.signature = self._build_signature(record)

        latest = self.load_snapshot(record.task_name)
        is_new = latest is None or latest.get("signature") != record.signature
        record.version = int(latest.get("version", 0)) + 1 if latest and is_new else (int(latest.get("version", 1)) if latest else 1)

        if is_new:
            with self.history_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

        # 始终更新快照
        snapshot_path = self.snapshots_dir / f"{record.task_name}.latest.json"
        snapshot_path.write_text(
            json.dumps(record.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        self._rebuild_index()

        return {
            "appended": is_new,
            "version": record.version,
            "signature": record.signature,
            "snapshot_path": str(snapshot_path),
        }

    # ---- 读取 ----

    def load_snapshot(self, task_name: str) -> dict[str, Any] | None:
        path = self.snapshots_dir / f"{task_name}.latest.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def load_all_snapshots(
        self,
        *,
        task_type: str | None = None,
        status: str | None = None,
        tags: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """加载所有快照，可选按 task_type / status / tags 过滤。"""
        if not self.snapshots_dir.exists():
            return []
        results = []
        for path in sorted(self.snapshots_dir.glob("*.latest.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if task_type and record.get("task_type") != task_type:
                continue
            if status and record.get("status") != status:
                continue
            if tags:
                record_tags = set(record.get("tags") or [])
                if not set(tags).issubset(record_tags):
                    continue
            results.append(record)
        return results

    def load_successful(self, **kwargs) -> list[dict[str, Any]]:
        """加载所有成功的任务记录。"""
        return self.load_all_snapshots(status="success", **kwargs)

    def load_failed(self, **kwargs) -> list[dict[str, Any]]:
        """加载所有失败的任务记录。"""
        return self.load_all_snapshots(status="failed", **kwargs)

    def load_history(self) -> list[dict[str, Any]]:
        if not self.history_path.exists():
            return []
        records = []
        for line in self.history_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
        return records

    def load_index(self) -> dict[str, Any]:
        if not self.index_path.exists():
            return {}
        try:
            return json.loads(self.index_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    # ---- 查询 ----

    def query(
        self,
        *,
        task_type: str | None = None,
        status: str | None = None,
        tags: list[str] | None = None,
        formula_contains: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """通用查询接口。"""
        results = self.load_all_snapshots(task_type=task_type, status=status, tags=tags)
        if formula_contains:
            results = [
                r for r in results
                if formula_contains.lower() in str(r.get("structure_context", {}).get("reduced_formula", "")).lower()
            ]
        return results[:limit]

    def get_failure_library(self) -> dict[str, Any]:
        """获取失败原因聚合库。"""
        if self.failure_library_path.exists():
            try:
                return json.loads(self.failure_library_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return self._rebuild_failure_library()

    # ---- 内部方法 ----

    def _build_signature(self, record: TaskRecord) -> str:
        payload = {
            "task_name": record.task_name,
            "task_type": record.task_type,
            "completed": record.completed,
            "converged": record.converged,
            "status": record.status,
            "total_energy": _round(record.total_energy),
            "max_force": _round(record.max_force),
            "ionic_steps": record.ionic_steps,
        }
        content = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _rebuild_index(self) -> Path:
        snapshots = self.load_all_snapshots()
        index = {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "total_records": len(snapshots),
            "by_task_type": _count_field(snapshots, "task_type"),
            "by_status": _count_field(snapshots, "status"),
            "by_source_tool": _count_field(snapshots, "source_tool"),
            "tag_counts": _count_tags(snapshots),
        }
        self.index_path.write_text(
            json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self._rebuild_failure_library()
        return self.index_path

    def _rebuild_failure_library(self) -> dict[str, Any]:
        records = self.load_all_snapshots(status="failed")
        buckets: dict[str, dict] = {}
        for r in records:
            reason = str(r.get("failure_reason") or "").strip()
            if not reason:
                continue
            b = buckets.setdefault(reason, {"reason": reason, "count": 0, "tasks": [], "actions": {}})
            b["count"] += 1
            b["tasks"].append(r.get("task_name"))
            for action in r.get("recommended_actions") or []:
                b["actions"][action] = b["actions"].get(action, 0) + 1
        rows = sorted(buckets.values(), key=lambda x: -x["count"])
        payload = {"failure_reasons": rows}
        self.failure_library_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return payload


# ---- 默认单例 ----

_default_store: KnowledgeStore | None = None


def get_default_store() -> KnowledgeStore:
    global _default_store
    if _default_store is None:
        _default_store = KnowledgeStore()
    return _default_store


# ---- 工具函数 ----

def _round(v: float | None) -> float | None:
    return round(v, 8) if v is not None else None


def _count_field(records: list[dict], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in records:
        val = str(r.get(field, "unknown"))
        counts[val] = counts.get(val, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def _count_tags(records: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in records:
        for tag in r.get("tags") or []:
            counts[tag] = counts.get(tag, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
