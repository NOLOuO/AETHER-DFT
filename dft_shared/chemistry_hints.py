"""吸附物化学提示：让模型在生成候选前不必靠训练记忆做猜测。

仅提供"百科式"原型信息（anchor、binding motif、几何），不替模型决策。
"""

from __future__ import annotations

from typing import Any

try:
    from ase.build import molecule as ase_molecule
except Exception:  # pragma: no cover
    ase_molecule = None

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except Exception:  # pragma: no cover
    Chem = None
    AllChem = None


_CURATED: dict[str, dict[str, Any]] = {
    "H": {
        "summary": "氢原子吸附；常见 hollow / bridge 占优。",
        "anchor_candidates": [{"element": "H", "rationale": "单原子"}],
        "binding_motifs": [
            {"label": "hollow", "rationale": "在 fcc/hcp 金属常见，最稳定", "preferred_site_family": "hollow"},
            {"label": "bridge", "rationale": "次稳，常作敏感性对比", "preferred_site_family": "bridge"},
        ],
        "geometry": {"shape": "atom", "size_angstrom": 0.5},
        "default_orientation": "upright",
        "typical_height_angstrom": 1.0,
    },
    "H2O": {
        "summary": "水分子；O 上 lone pair 主导，O-down 吸附为主。",
        "anchor_candidates": [
            {"element": "O", "rationale": "O 上两对 lone pair，给体配位"},
        ],
        "binding_motifs": [
            {"label": "atop O-down", "rationale": "Pt/Pd/Cu/Ru 上经典 motif", "preferred_site_family": "ontop"},
            {"label": "tilted O-down", "rationale": "二级网格 / H-bond 网络存在时", "preferred_site_family": "ontop"},
        ],
        "geometry": {"shape": "bent", "size_angstrom": 1.5, "hoh_angle_deg": 104.5},
        "default_orientation": "upright",
        "typical_height_angstrom": 2.1,
    },
    "OH": {
        "summary": "羟基；O 端为 anchor，常见 bridge / atop 共存。",
        "anchor_candidates": [{"element": "O", "rationale": "O 端有 unpaired electron"}],
        "binding_motifs": [
            {"label": "bridge O-down", "rationale": "贵金属表面常见", "preferred_site_family": "bridge"},
            {"label": "atop O-down", "rationale": "弱吸附面或氧化物常见", "preferred_site_family": "ontop"},
        ],
        "geometry": {"shape": "linear", "size_angstrom": 1.0},
        "default_orientation": "upright",
        "typical_height_angstrom": 1.9,
    },
    "CO": {
        "summary": "一氧化碳；C 端为 anchor，atop > bridge >> hollow（Pt/Ru/Pd）。",
        "anchor_candidates": [
            {"element": "C", "rationale": "C 端 lone pair → 5σ donation + 2π* back-donation"},
        ],
        "binding_motifs": [
            {"label": "atop C-down upright", "rationale": "Blyholder 模型经典 motif", "preferred_site_family": "ontop"},
            {"label": "bridge C-down", "rationale": "Pd 系明显，Pt 系次稳", "preferred_site_family": "bridge"},
        ],
        "geometry": {"shape": "linear", "size_angstrom": 1.13},
        "default_orientation": "upright",
        "typical_height_angstrom": 1.9,
    },
    "CO2": {
        "summary": "二氧化碳；活化前几乎不吸附；活化后弯曲，O 或 C 与表面成键。",
        "anchor_candidates": [
            {"element": "C", "rationale": "活化后 C 弯曲下行结合"},
            {"element": "O", "rationale": "线性物理吸附时 O 略微指向表面"},
        ],
        "binding_motifs": [
            {"label": "physisorption flat", "rationale": "弱吸附面常见", "preferred_site_family": "ontop"},
            {"label": "bent CO2(-) bridge", "rationale": "Cu/Ag 活化态", "preferred_site_family": "bridge"},
        ],
        "geometry": {"shape": "linear", "size_angstrom": 2.32},
        "default_orientation": "flat",
        "typical_height_angstrom": 2.5,
    },
    "NH3": {
        "summary": "氨；N 上 lone pair；atop N-down 几乎默认。",
        "anchor_candidates": [{"element": "N", "rationale": "N 上 lone pair 给体"}],
        "binding_motifs": [
            {"label": "atop N-down upright", "rationale": "贵金属面经典 motif", "preferred_site_family": "ontop"},
        ],
        "geometry": {"shape": "pyramidal", "size_angstrom": 1.5},
        "default_orientation": "upright",
        "typical_height_angstrom": 2.1,
    },
    "CH4": {
        "summary": "甲烷；物理吸附为主；C-H 解离后才转化学吸附。",
        "anchor_candidates": [{"element": "C", "rationale": "无明显 lone pair；接触 anchor 一般是 C"}],
        "binding_motifs": [
            {"label": "physisorption C-down", "rationale": "弱吸附", "preferred_site_family": "ontop"},
        ],
        "geometry": {"shape": "tetrahedral", "size_angstrom": 1.5},
        "default_orientation": "upright",
        "typical_height_angstrom": 3.0,
    },
    "O": {
        "summary": "氧原子；fcc/hcp hollow 占优。",
        "anchor_candidates": [{"element": "O", "rationale": "单原子"}],
        "binding_motifs": [
            {"label": "fcc hollow", "rationale": "Pt/Cu 等典型偏好", "preferred_site_family": "hollow"},
            {"label": "bridge", "rationale": "次稳", "preferred_site_family": "bridge"},
        ],
        "geometry": {"shape": "atom", "size_angstrom": 0.6},
        "default_orientation": "upright",
        "typical_height_angstrom": 1.4,
    },
    "N": {
        "summary": "氮原子；hollow 占优。",
        "anchor_candidates": [{"element": "N", "rationale": "单原子"}],
        "binding_motifs": [
            {"label": "fcc hollow", "rationale": "过渡金属常见", "preferred_site_family": "hollow"},
        ],
        "geometry": {"shape": "atom", "size_angstrom": 0.7},
        "default_orientation": "upright",
        "typical_height_angstrom": 1.5,
    },
    "CH3OH": {
        "summary": "甲醇；O 端吸附为主，C-H/O-H 解离决定中间体。",
        "anchor_candidates": [{"element": "O", "rationale": "O 上 lone pair"}],
        "binding_motifs": [
            {"label": "atop O-down", "rationale": "贵金属面常见", "preferred_site_family": "ontop"},
        ],
        "geometry": {"shape": "tetrahedral", "size_angstrom": 2.0},
        "default_orientation": "upright",
        "typical_height_angstrom": 2.2,
    },
}

_ANCHOR_PRIORITY = ["O", "N", "S", "P", "C", "H"]


def _normalize_token(token: str) -> str:
    cleaned = token.strip()
    upper = cleaned.upper()
    aliases = {"WATER": "H2O", "METHANOL": "CH3OH", "AMMONIA": "NH3", "METHANE": "CH4"}
    return aliases.get(upper, cleaned)


def _ase_inferred_hint(token: str) -> dict[str, Any] | None:
    if ase_molecule is None:
        return None
    candidates = [token, token.upper(), token.lower()]
    for candidate in candidates:
        try:
            atoms = ase_molecule(candidate)
        except Exception:
            continue
        symbols = [atom.symbol for atom in atoms]
        if not symbols:
            continue
        anchor = next(
            (symbol for symbol in _ANCHOR_PRIORITY if symbol in symbols),
            symbols[0],
        )
        positions = atoms.get_positions()
        size = float(positions.max(axis=0).max() - positions.min(axis=0).min()) if len(positions) else 1.0
        return {
            "source": "ase_inferred",
            "summary": f"ASE 内置分子 {candidate}；anchor 用启发式优先级推断（O>N>S>P>C>H）。",
            "anchor_candidates": [
                {
                    "element": anchor,
                    "rationale": "启发式：取常见给体优先级最高的元素；模型应结合 lone pair / 配位偏好再确认。",
                }
            ],
            "binding_motifs": [
                {
                    "label": f"atop {anchor}-down",
                    "rationale": "通用先验，需结合表面再细化",
                    "preferred_site_family": "ontop",
                }
            ],
            "geometry": {
                "shape": "ase_inferred",
                "size_angstrom": round(max(size, 1.0), 2),
                "atom_count": len(symbols),
                "composition": {symbol: symbols.count(symbol) for symbol in set(symbols)},
            },
            "default_orientation": "upright",
            "typical_height_angstrom": 2.0 if len(symbols) > 3 else 1.8,
        }
    return None


def _rdkit_inferred_hint(token: str) -> dict[str, Any] | None:
    if Chem is None or AllChem is None:
        return None
    molecule = Chem.MolFromSmiles(token)
    if molecule is None:
        return None
    molecule = Chem.AddHs(molecule)
    if AllChem.EmbedMolecule(molecule, randomSeed=7) != 0:
        return None
    AllChem.UFFOptimizeMolecule(molecule)
    symbols = [atom.GetSymbol() for atom in molecule.GetAtoms()]
    anchor = next(
        (symbol for symbol in _ANCHOR_PRIORITY if symbol in symbols),
        symbols[0],
    )
    conformer = molecule.GetConformer()
    coords = []
    for idx in range(molecule.GetNumAtoms()):
        point = conformer.GetAtomPosition(idx)
        coords.append((point.x, point.y, point.z))
    if coords:
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        zs = [c[2] for c in coords]
        size = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
    else:
        size = 1.0
    return {
        "source": "rdkit_smiles",
        "summary": f"SMILES `{token}` 解析自 RDKit；anchor 用启发式优先级推断。",
        "anchor_candidates": [
            {
                "element": anchor,
                "rationale": "启发式：取常见给体优先级最高的元素；模型必须自己确认。",
            }
        ],
        "binding_motifs": [
            {
                "label": f"atop {anchor}-down",
                "rationale": "通用先验；分子较大时建议先看 flat / tilted",
                "preferred_site_family": "ontop",
            }
        ],
        "geometry": {
            "shape": "rdkit_optimized",
            "size_angstrom": round(max(float(size), 1.0), 2),
            "atom_count": len(symbols),
            "composition": {symbol: symbols.count(symbol) for symbol in set(symbols)},
        },
        "default_orientation": "upright",
        "typical_height_angstrom": 2.3 if len(symbols) > 6 else 2.0,
    }


def get_adsorbate_chemistry_hint(adsorbate: str) -> dict[str, Any]:
    """返回吸附物的化学先验，包括 anchor、binding motif 与几何尺寸。

    查找顺序：curated → ASE 分子 → RDKit SMILES → 报错。
    """
    if not adsorbate or not adsorbate.strip():
        raise ValueError("adsorbate 不能为空。")
    token = _normalize_token(adsorbate)
    if token in _CURATED:
        payload = dict(_CURATED[token])
        payload.update(
            {
                "status": "ok",
                "source": "curated",
                "adsorbate": token,
                "input": adsorbate,
            }
        )
        payload["guidance"] = (
            "把 anchor_candidates[0].element 作为 structure_add_adsorbate.anchor_symbol；"
            "把 binding_motifs[i].preferred_site_family 与 structure_enumerate_sites 的 site_family 对齐；"
            "把 typical_height_angstrom 作为 structure_add_adsorbate.height 的起点。"
        )
        return payload

    inferred = _ase_inferred_hint(token) or _rdkit_inferred_hint(token)
    if inferred is None:
        return {
            "status": "unknown",
            "adsorbate": token,
            "input": adsorbate,
            "message": (
                "无法识别该吸附物。请使用 ASE 支持的分子名（H2O/CO/NH3 等）、curated 列表中的名字或可解析 SMILES；"
                "也可以直接提供分子文件路径给 structure_add_adsorbate。"
            ),
        }
    inferred.update(
        {
            "status": "ok",
            "adsorbate": token,
            "input": adsorbate,
            "guidance": (
                "本提示来自启发式推断，化学正确性需要模型自己判断；"
                "如果你对 anchor 元素或 binding motif 有更强依据，直接覆写即可。"
            ),
        }
    )
    return inferred


def list_curated_adsorbates() -> list[str]:
    return sorted(_CURATED.keys())
