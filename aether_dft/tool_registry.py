from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aether_dft.runtime_harness.tool_registry import ToolRegistry, list_registered_tools as _list_registered_tools


@dataclass(frozen=True)
class RegisteredTool:
    name: str
    description: str
    parameters: dict[str, Any]
    source: str = "tools"
    read_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "source": self.source,
            "read_only": self.read_only,
        }


class AetherToolRegistry:
    """Compatibility facade over the root ``harness.tool_registry`` surface."""

    def __init__(self, *, allow_cluster_submit: bool = False, permission_mode: str | None = None):
        self.registry = ToolRegistry(allow_cluster_submit=allow_cluster_submit, permission_mode=permission_mode)

    def list_tools(self) -> list[RegisteredTool]:
        return [RegisteredTool(**item) for item in self.registry.list_tools()]

    def openai_tool_schemas(self) -> list[dict[str, Any]]:
        return self.registry.openai_tool_schemas()

    def run_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.registry.run_tool(name, arguments or {})


def list_registered_tools() -> list[dict[str, Any]]:
    return _list_registered_tools()
