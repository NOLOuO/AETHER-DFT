"""DFT Tools 共享基础设施：知识库、结构分析、远程连接、结果解释 contract。"""

__version__ = "0.2.0"

from .analysis_dossier import (
    EvidenceBundle,
    EvidenceGap,
    InterpretationRequest,
    InterpretationResult,
    KnowledgeBackflowPayload,
)
from .llm_advisor import generate_incar_advice
from .poscar_writer import write_poscar
from .result_interpreter import (
    build_knowledge_backflow_payload,
    interpret_result,
    render_interpretation_markdown,
)

__all__ = [
    "EvidenceBundle",
    "EvidenceGap",
    "InterpretationRequest",
    "InterpretationResult",
    "KnowledgeBackflowPayload",
    "generate_incar_advice",
    "interpret_result",
    "build_knowledge_backflow_payload",
    "render_interpretation_markdown",
    "write_poscar",
]
