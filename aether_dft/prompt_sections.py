from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Iterable

from .paths import PROJECT_ROOT

PROMPT_ASSETS_DIR = Path(__file__).resolve().parent / "prompt_assets"


class SafeFormatDict(dict):
    """Format helper that leaves missing runtime fields empty."""

    def __missing__(self, key: str) -> str:
        return ""


@dataclass(frozen=True)
class PromptSection:
    name: str
    content: str
    kind: str
    cache_scope: str
    layer_class: str
    invalidation_rule: tuple[str, ...]


class PromptSectionCompiler:
    """Composable prompt compiler adapted from the reference agent harness.

    AETHER needs the same engineering shape as a Code/Codex-like agent:
    stable role/policy layers plus volatile runtime/project/tool context.  The
    compiler keeps those layers inspectable so prompt changes do not become a
    single untestable string blob.
    """

    DEFAULT_STATIC_ORDER = (
        "base_role",
        "research_partner",
        "general_agent_voice",
        "computational_chemistry_workflow",
        "research_execution_loop",
        "evidence_contract",
        "tool_policy",
        "structure_modeling",
        "cluster_execution",
        "cluster_realtime_disposition",
        "research_workspace_habit",
        "adsorption_authoring",
    )
    DEFAULT_RUNTIME_ORDER = (
        "runtime_context",
        "architecture_live_doc",
        "project_context",
        "cluster_runtime_digest",
        "job_watch_digest",
        "research_workspace_digest",
        "relevant_priors_digest",
        "session_context",
        "tool_discovery",
        "response_contract",
    )

    def __init__(
        self,
        section_dir: str | Path | None = None,
        *,
        static_order: Iterable[str] | None = None,
        runtime_order: Iterable[str] | None = None,
    ):
        self.section_dir = Path(section_dir) if section_dir else PROMPT_ASSETS_DIR / "sections"
        self.static_order = tuple(static_order or self.DEFAULT_STATIC_ORDER)
        self.runtime_order = tuple(runtime_order or self.DEFAULT_RUNTIME_ORDER)

    def _section_path(self, name: str) -> Path:
        return self.section_dir / f"{name}.md"

    def _descriptor(self, name: str) -> dict[str, Any]:
        if name in self.DEFAULT_STATIC_ORDER:
            return {
                "kind": "static",
                "cache_scope": "stable_prefix",
                "layer_class": "policy" if name in {"evidence_contract", "tool_policy", "structure_modeling", "cluster_execution", "cluster_realtime_disposition", "research_workspace_habit", "adsorption_authoring"} else "role",
                "invalidation_rule": ("section_file",),
            }
        if name == "runtime_context":
            return {
                "kind": "dynamic",
                "cache_scope": "volatile_suffix",
                "layer_class": "runtime",
                "invalidation_rule": ("workspace", "model_id", "created_at"),
            }
        if name == "project_context":
            return {
                "kind": "dynamic",
                "cache_scope": "volatile_suffix",
                "layer_class": "project_state",
                "invalidation_rule": ("project", "project_context_digest"),
            }
        if name == "architecture_live_doc":
            return {
                "kind": "dynamic",
                "cache_scope": "volatile_suffix",
                "layer_class": "live_architecture",
                "invalidation_rule": ("architecture_live_doc_digest",),
            }
        if name == "session_context":
            return {
                "kind": "dynamic",
                "cache_scope": "volatile_suffix",
                "layer_class": "session",
                "invalidation_rule": ("session_context_digest",),
            }
        if name == "cluster_runtime_digest":
            return {
                "kind": "dynamic",
                "cache_scope": "volatile_suffix",
                "layer_class": "cluster_runtime",
                "invalidation_rule": ("cluster_runtime_digest_hash",),
            }
        if name == "job_watch_digest":
            return {
                "kind": "dynamic",
                "cache_scope": "volatile_suffix",
                "layer_class": "cluster_runtime",
                "invalidation_rule": ("job_watch_digest_hash",),
            }
        if name == "research_workspace_digest":
            return {
                "kind": "dynamic",
                "cache_scope": "volatile_suffix",
                "layer_class": "research_workspace",
                "invalidation_rule": ("research_workspace_digest_hash",),
            }
        if name == "relevant_priors_digest":
            return {
                "kind": "dynamic",
                "cache_scope": "volatile_suffix",
                "layer_class": "knowledge_priors",
                "invalidation_rule": ("relevant_priors_digest_hash",),
            }
        if name == "tool_discovery":
            return {
                "kind": "dynamic",
                "cache_scope": "volatile_suffix",
                "layer_class": "tooling",
                "invalidation_rule": ("tool_discovery_digest",),
            }
        return {
            "kind": "dynamic",
            "cache_scope": "volatile_suffix",
            "layer_class": "response",
            "invalidation_rule": ("response_contract_digest",),
        }

    def load_section(self, name: str) -> PromptSection | None:
        path = self._section_path(name)
        if not path.exists():
            return None
        descriptor = self._descriptor(name)
        return PromptSection(
            name=name,
            content=path.read_text(encoding="utf-8"),
            kind=str(descriptor["kind"]),
            cache_scope=str(descriptor["cache_scope"]),
            layer_class=str(descriptor["layer_class"]),
            invalidation_rule=tuple(str(item) for item in descriptor["invalidation_rule"]),
        )

    @staticmethod
    def _digest(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12] if text else ""

    def render_section(self, name: str, runtime_data: dict[str, Any]) -> str:
        section = self.load_section(name)
        if section is None:
            return ""
        rendered = section.content.format_map(SafeFormatDict(runtime_data)).strip()
        if not rendered:
            return ""
        # Dynamic layers should disappear when their payload is empty instead
        # of leaving headings with no evidence behind.
        empty_guards = {
            "architecture_live_doc": "architecture_live_doc_digest",
            "project_context": "project_context",
            "session_context": "session_context",
            "job_watch_digest": "job_watch_digest",
            "tool_discovery": "tool_discovery_digest",
        }
        guard = empty_guards.get(name)
        if guard and not str(runtime_data.get(guard) or "").strip():
            return ""
        return rendered

    def build(self, runtime_data: dict[str, Any], *, fallback: str = "") -> dict[str, Any]:
        normalized = SafeFormatDict(runtime_data)
        layers: list[dict[str, Any]] = []
        pieces: list[str] = []
        for order, name in enumerate((*self.static_order, *self.runtime_order), start=1):
            rendered = self.render_section(name, normalized)
            section = self.load_section(name)
            descriptor = self._descriptor(name)
            layer = {
                "name": name,
                "order": order,
                "included": bool(rendered),
                "kind": str(descriptor["kind"]),
                "cache_scope": str(descriptor["cache_scope"]),
                "layer_class": str(descriptor["layer_class"]),
                "invalidation_rule": list(descriptor["invalidation_rule"]),
                "char_count": len(rendered),
                "content_digest": self._digest(rendered),
                "source": {
                    "type": "file" if section else "missing",
                    "path": str(self._section_path(name)) if section else "",
                },
                "rendered_text": rendered,
            }
            layers.append(layer)
            if rendered:
                pieces.append(rendered)

        if not pieces and fallback.strip():
            rendered = fallback.format_map(normalized).strip()
            layers.append(
                {
                    "name": "fallback",
                    "order": len(layers) + 1,
                    "included": bool(rendered),
                    "kind": "fallback",
                    "cache_scope": "volatile_suffix",
                    "layer_class": "fallback",
                    "invalidation_rule": [],
                    "char_count": len(rendered),
                    "content_digest": self._digest(rendered),
                    "source": {"type": "fallback", "path": ""},
                    "rendered_text": rendered,
                }
            )
            if rendered:
                pieces.append(rendered)

        stable_layers = [item for item in layers if item["included"] and item["cache_scope"] == "stable_prefix"]
        volatile_layers = [item for item in layers if item["included"] and item["cache_scope"] != "stable_prefix"]
        prompt = "\n\n".join(pieces).rstrip() + ("\n" if pieces else "")
        stable_prefix_text = "\n\n".join(str(item["rendered_text"]) for item in stable_layers).strip()
        volatile_suffix_text = "\n\n".join(str(item["rendered_text"]) for item in volatile_layers).strip()
        return {
            "prompt": prompt,
            "layers": layers,
            "source_map": layers,
            "stable_prefix_text": stable_prefix_text,
            "volatile_suffix_text": volatile_suffix_text,
            "compile_projection": {
                "compile_strategy": "aether_section_compiler",
                "stable_layer_names": [item["name"] for item in stable_layers],
                "volatile_layer_names": [item["name"] for item in volatile_layers],
                "stable_prefix_char_count": len(stable_prefix_text),
                "volatile_suffix_char_count": len(volatile_suffix_text),
                "cache_breakpoints": [
                    {
                        "layer": item["name"],
                        "reason": "runtime/project/session/tool context changes per turn",
                        "invalidation_rule": item["invalidation_rule"],
                    }
                    for item in volatile_layers
                ],
            },
        }
