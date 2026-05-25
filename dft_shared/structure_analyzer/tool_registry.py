from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class StructureTool:
    name: str
    status: str
    command: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


STRUCTURE_TOOLS: tuple[StructureTool, ...] = (
    StructureTool("xsd_to_poscar", "implemented", "aether-dft structure convert input.xsd POSCAR", "Materials Studio XSD 转 VASP POSCAR，保留可识别的固定原子约束。"),
    StructureTool("poscar_to_xsd", "implemented-if-ase-supports-xsd", "aether-dft structure convert POSCAR output.xsd --fmt xsd", "POSCAR/CONTCAR 转 XSD，依赖 ASE 当前安装是否支持 xsd writer。"),
    StructureTool("structure_to_json", "implemented", "python -m dft_shared.structure_analyzer.serializer", "Atoms/结构转 JSON/XYZ，供 3Dmol.js 渲染。"),
    StructureTool("structure_resolve", "implemented", "AETHER harness tool: structure_resolve", "统一解析本地 POSCAR/CIF/XSD、MP mp-id 或 ASE 单元素 bulk。"),
    StructureTool("structure_supercell", "implemented", "AETHER harness tool: structure_supercell", "生成 supercell 扩胞结构。"),
    StructureTool("structure_build_slab", "implemented", "AETHER harness tool: structure_build_slab", "按 miller/supercell/vacuum/fixed layers 构建 slab。"),
    StructureTool("structure_add_adsorbate", "implemented", "AETHER harness tool: structure_add_adsorbate", "在 slab 顶层或指定坐标添加吸附物初猜。"),
    StructureTool("structure_defect", "implemented", "AETHER harness tool: structure_defect", "统一 vacancy / substitution dopant 缺陷入口。"),
    StructureTool("structure_add_vacancy", "implemented", "AETHER harness tool: structure_add_vacancy", "删除指定元素原子，生成 vacancy/缺陷结构。"),
    StructureTool("structure_add_dopant", "implemented", "AETHER harness tool: structure_add_dopant", "替换指定原子，生成 dopant/掺杂结构。"),
    StructureTool("structure_sanity_check", "implemented", "AETHER harness tool: structure_sanity_check", "检查最短距离、真空层估算、原子数和物种。"),
    StructureTool("bond_analyzer", "implemented", "dft_shared.structure_analyzer.bond_analyzer", "结构键连分析。"),
    StructureTool("structure_comparator", "implemented", "dft_shared.structure_analyzer.comparator", "比较结构变化、位移和路径差异。"),
    StructureTool("poscar_writer", "implemented", "dft_shared.poscar_writer", "写出 VASP POSCAR。"),
    StructureTool("adsorption_candidates", "implemented", "aether-dft dft adsorption-candidates ...", "吸附构型候选生成。"),
    StructureTool("adsorption_select", "implemented", "aether-dft dft adsorption-select ...", "候选构型选择并交给 builder/submit 主线。"),
    StructureTool("workflow_bundle", "implemented", "aether-dft dft adsorption-workflow ...", "clean slab / isolated adsorbate / adsorbed system 三子任务束。"),
    StructureTool("result_explain_bridge", "implemented", "aether-dft explain --run-root <path>", "把已有 run 结果转入 dft_tools explain / 知识回流桥。"),
)


def list_structure_tools() -> list[dict[str, Any]]:
    return [tool.to_dict() for tool in STRUCTURE_TOOLS]
