"""共享知识库模块。

所有 dft_tools 子工具通过这个模块读写知识库。

用法::

    from dft_shared.knowledge_base import KnowledgeStore, TaskRecord, find_similar

    store = KnowledgeStore()  # 或指定路径: KnowledgeStore("/path/to/kb")

    # 写入
    record = TaskRecord(task_name="C7H14-re", task_type="relax", status="success", ...)
    store.ingest(record)

    # 查询
    results = store.query(task_type="relax", status="success")

    # 匹配相似案例
    matches = find_similar(record, store=store, pool_status="success", limit=5)
"""

from .matcher import find_similar
from .models import MatchResult, TaskRecord
from .store import KnowledgeStore, get_default_store

__all__ = [
    "KnowledgeStore",
    "TaskRecord",
    "MatchResult",
    "find_similar",
    "get_default_store",
]
