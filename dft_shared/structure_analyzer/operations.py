from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_structure(path: str | Path):
    from pymatgen.core import Structure

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"结构文件不存在: {source}")
    if source.suffix.lower() == ".xsd":
        from dft_shared.structure_analyzer.io import xsd_to_poscar
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            converted = Path(tmp) / "POSCAR"
            xsd_to_poscar(source, converted)
            return Structure.from_file(converted)
    return Structure.from_file(source)


def _write_structure(structure: Any, output_path: str | Path) -> Path:
    from pymatgen.io.vasp import Poscar

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = target.suffix.lower()
    if suffix == ".cif":
        structure.to(fmt="cif", filename=str(target))
    else:
        Poscar(structure).write_file(target)
    return target


def _structure_summary(structure: Any) -> dict[str, Any]:
    return {
        "formula": structure.composition.reduced_formula,
        "atom_count": len(structure),
        "lattice_abc": [float(item) for item in structure.lattice.abc],
        "lattice_angles": [float(item) for item in structure.lattice.angles],
        "species": sorted({site.specie.symbol for site in structure}),
    }


def resolve_structure(
    *,
    input_path: str | None = None,
    material: str | None = None,
    mp_id: str | None = None,
    output_path: str | None = None,
    source: str = "auto",
) -> dict[str, Any]:
    if input_path:
        structure = _load_structure(input_path)
        resolved_source = "local"
        source_detail = str(Path(input_path))
    elif mp_id or source.lower() in {"mp", "materials_project"}:
        from dft_app.llm.key_store import resolve_api_key
        from mp_api.client import MPRester

        api_key = resolve_api_key(
            Path.cwd(),
            aliases=("materials_project", "mp"),
            env_names=("MP_API_KEY", "MATERIALS_PROJECT_API_KEY"),
        )
        if not api_key:
            raise RuntimeError("未找到 Materials Project API key；请提供本地结构或设置 MP_API_KEY。")
        with MPRester(api_key) as rester:
            if not mp_id:
                raise ValueError("source=mp 时必须提供 mp_id，避免自动选错结构。")
            structure = rester.get_structure_by_material_id(mp_id)
        resolved_source = "materials_project"
        source_detail = str(mp_id)
    elif material:
        from ase.build import bulk
        from pymatgen.core import Composition
        from pymatgen.io.ase import AseAtomsAdaptor

        composition = Composition(str(material).split("(")[0].strip())
        if len(composition.elements) != 1:
            raise ValueError("自动 ASE bulk 兜底只支持单元素；多元素请提供 input_path 或 mp_id。")
        atoms = bulk(composition.elements[0].symbol)
        structure = AseAtomsAdaptor.get_structure(atoms)
        resolved_source = "ase_bulk"
        source_detail = composition.elements[0].symbol
    else:
        raise ValueError("resolve_structure 需要 input_path、material 或 mp_id。")

    written = None
    if output_path:
        written = str(_write_structure(structure, output_path))
    return {
        "status": "ok",
        "source": resolved_source,
        "source_detail": source_detail,
        "output_path": written,
        "summary": _structure_summary(structure),
    }


def make_supercell(input_path: str, output_path: str, scaling_matrix: list[int] | tuple[int, int, int]) -> dict[str, Any]:
    if len(scaling_matrix) != 3:
        raise ValueError("scaling_matrix 必须是三个整数，例如 [2,2,1]。")
    structure = _load_structure(input_path)
    structure.make_supercell([int(item) for item in scaling_matrix])
    target = _write_structure(structure, output_path)
    return {"status": "ok", "output_path": str(target), "scaling_matrix": list(scaling_matrix), "summary": _structure_summary(structure)}


def add_adsorbate(
    *,
    slab_path: str,
    adsorbate: str,
    output_path: str,
    height: float = 2.0,
    site_index: int | None = None,
    cart_coords: list[float] | None = None,
    orientation: str = "upright",
    anchor_atom_index: int | None = None,
    anchor_symbol: str | None = None,
    coords_mode: str = "cartesian",
    fixed_bottom_layers: int = 2,
) -> dict[str, Any]:
    from dft_app.modeling.adsorption_candidate_generator import (
        AdsorbateStructureFactory,
        AdsorptionCandidateGenerator,
        apply_selective_dynamics,
    )
    from pymatgen.io.ase import AseAtomsAdaptor

    slab = _load_structure(slab_path)
    slab_atoms = AseAtomsAdaptor.get_atoms(slab)
    ads_atoms = AdsorbateStructureFactory().load_adsorbate_atoms(adsorbate).copy()
    if anchor_atom_index is not None:
        anchor_index = int(anchor_atom_index)
        if anchor_index < 0 or anchor_index >= len(ads_atoms):
            raise ValueError(f"anchor_atom_index 越界: {anchor_index}; 吸附物原子数 {len(ads_atoms)}")
    elif anchor_symbol:
        matches = [idx for idx, atom in enumerate(ads_atoms) if atom.symbol.lower() == anchor_symbol.lower()]
        if not matches:
            raise ValueError(f"吸附物 {adsorbate} 中找不到 anchor_symbol={anchor_symbol}")
        anchor_index = matches[0]
    else:
        anchor_index = AdsorptionCandidateGenerator._preferred_anchor_index(ads_atoms)
    AdsorptionCandidateGenerator._orient_adsorbate(ads_atoms, anchor_index=anchor_index, orientation_label=orientation)
    if cart_coords is not None:
        if len(cart_coords) != 3:
            raise ValueError("cart_coords 必须是三个浮点数。")
        if coords_mode.lower() not in {"cartesian", "cart"}:
            raise ValueError("当前只支持 coords_mode=cartesian；分数坐标请先转换为笛卡尔坐标。")
        target = np.array(cart_coords, dtype=float)
    else:
        top_indexes = sorted(range(len(slab)), key=lambda idx: float(slab[idx].coords[2]), reverse=True)
        requested_site_index = int(site_index or 0)
        if requested_site_index < 0 or requested_site_index >= len(top_indexes):
            raise ValueError(f"site_index 越界: {requested_site_index}; 可用 0..{len(top_indexes)-1}（0-based 顶层原子序号）")
        selected = top_indexes[requested_site_index]
        target = np.array(slab[selected].coords, dtype=float)
        target[2] = max(float(site.coords[2]) for site in slab) + float(height)
    ads_atoms.translate(target - np.array(ads_atoms[anchor_index].position, dtype=float))
    combined = slab_atoms.copy()
    combined.extend(ads_atoms)
    structure = AseAtomsAdaptor.get_structure(combined)
    if fixed_bottom_layers > 0:
        structure = apply_selective_dynamics(
            structure,
            slab_atom_count=len(slab_atoms),
            fixed_bottom_layers=int(fixed_bottom_layers),
        )
    target_path = _write_structure(structure, output_path)
    return {
        "status": "ok",
        "output_path": str(target_path),
        "adsorbate": adsorbate,
        "anchor_index": anchor_index,
        "anchor_symbol": ads_atoms[anchor_index].symbol,
        "site_index_semantics": "0-based index into atoms sorted by descending z when cart_coords is absent",
        "target_cart_coords": target.tolist(),
        "slab_atom_count": len(slab_atoms),
        "fixed_bottom_layers": int(fixed_bottom_layers),
        "summary": _structure_summary(structure),
        "sanity": sanity_check(str(target_path)),
    }


def add_vacancy(
    *,
    input_path: str,
    output_path: str,
    species: str,
    index: int | None = None,
    surface_only: bool = True,
) -> dict[str, Any]:
    structure = _load_structure(input_path)
    matches = [idx for idx, site in enumerate(structure) if site.specie.symbol.lower() == species.lower()]
    if not matches:
        raise ValueError(f"结构中找不到物种 {species}")
    if index is not None:
        selected = int(index)
        if selected not in matches:
            raise ValueError(f"index={selected} 不是 {species} 原子。")
    elif surface_only:
        selected = max(matches, key=lambda idx: float(structure[idx].coords[2]))
    else:
        selected = matches[0]
    mutated = structure.copy()
    removed = {"index": selected, "species": structure[selected].specie.symbol, "coords": [float(x) for x in structure[selected].coords]}
    mutated.remove_sites([selected])
    target = _write_structure(mutated, output_path)
    return {"status": "ok", "output_path": str(target), "removed_site": removed, "summary": _structure_summary(mutated)}


def add_dopant(
    *,
    input_path: str,
    output_path: str,
    dopant: str,
    species: str | None = None,
    index: int | None = None,
    surface_only: bool = False,
) -> dict[str, Any]:
    structure = _load_structure(input_path)
    if index is None:
        if not species:
            raise ValueError("未提供 index 时必须提供要替换的 species。")
        matches = [idx for idx, site in enumerate(structure) if site.specie.symbol.lower() == species.lower()]
        if not matches:
            raise ValueError(f"结构中找不到可替换物种 {species}")
        selected = max(matches, key=lambda idx: float(structure[idx].coords[2])) if surface_only else matches[0]
    else:
        selected = int(index)
    original = structure[selected].specie.symbol
    mutated = structure.copy()
    mutated.replace(selected, dopant)
    target = _write_structure(mutated, output_path)
    return {
        "status": "ok",
        "output_path": str(target),
        "replacement": {"index": selected, "from": original, "to": dopant},
        "summary": _structure_summary(mutated),
    }


def enumerate_adsorption_sites(
    slab_path: str,
    *,
    max_sites_per_family: int = 4,
    top_layer_tolerance: float = 0.75,
    nearest_neighbors: int = 3,
) -> dict[str, Any]:
    """枚举 slab 的吸附位点，供模型自主选择候选。

    返回结构：
    - ``top_layer_atoms``: 顶层原子摘要（按 z 降序），含 ``atom_index``（0-based，可作
      ``structure_add_adsorbate.site_index``）、``element``、``cart_coords``。
    - ``sites``: 全部位点（ontop/bridge/hollow ...），含 ``site_id``、``site_family``、
      ``cart_coords``、最近邻顶层原子列表。
    - ``site_families``: 每个 family 出现次数，便于模型快速决定要不要展开某 family。
    - ``slab_top_z``: 顶层 z 坐标参考。
    """
    structure = _load_structure(slab_path)
    from pymatgen.analysis.adsorption import AdsorbateSiteFinder
    finder = AdsorbateSiteFinder(structure)
    raw_sites = finder.find_adsorption_sites()

    top_z = max(float(site.coords[2]) for site in structure)
    top_layer = [
        {
            "atom_index": idx,
            "element": site.specie.symbol,
            "cart_coords": [float(value) for value in site.coords],
            "z_offset_from_top": float(site.coords[2]) - top_z,
        }
        for idx, site in enumerate(structure)
        if top_z - float(site.coords[2]) <= top_layer_tolerance
    ]
    top_layer.sort(key=lambda item: item["cart_coords"][2], reverse=True)

    sites: list[dict[str, Any]] = []
    site_family_counts: dict[str, int] = {}
    for family, coords_list in raw_sites.items():
        if family == "all":
            continue
        family_str = str(family)
        for index, coords in enumerate(coords_list[:max_sites_per_family], start=1):
            target = np.array(coords, dtype=float)
            neighbours = sorted(
                top_layer,
                key=lambda atom, t=target: float(
                    np.linalg.norm(np.array(atom["cart_coords"], dtype=float)[:2] - t[:2])
                ),
            )[: max(1, nearest_neighbors)]
            site_id = f"{family_str.lower()}-{index:02d}"
            sites.append(
                {
                    "site_id": site_id,
                    "site_family": family_str,
                    "site_index_within_family": index,
                    "cart_coords": [float(value) for value in target],
                    "nearest_top_atoms": [
                        {
                            "atom_index": atom["atom_index"],
                            "element": atom["element"],
                            "in_plane_distance": float(
                                np.linalg.norm(
                                    np.array(atom["cart_coords"], dtype=float)[:2] - target[:2]
                                )
                            ),
                        }
                        for atom in neighbours
                    ],
                }
            )
            site_family_counts[family_str] = site_family_counts.get(family_str, 0) + 1

    # AdsorbateSiteFinder may symmetry-prune ontop sites to a single representative
    # on high-symmetry slabs. For model-authored candidate generation we expose up
    # to max_sites_per_family concrete top-atom ontop coordinates, then let
    # slab_surface_inspect / adsorption_candidate_plan decide which ones are
    # symmetry-equivalent enough to prune.
    existing_ontop_coords = [
        np.array(site["cart_coords"], dtype=float)
        for site in sites
        if str(site["site_family"]).lower() == "ontop"
    ]
    ontop_count = site_family_counts.get("ontop", 0)
    for atom in top_layer:
        if ontop_count >= max_sites_per_family:
            break
        coords = np.array(atom["cart_coords"], dtype=float)
        if any(float(np.linalg.norm(coords[:2] - existing[:2])) < 1e-3 for existing in existing_ontop_coords):
            continue
        ontop_count += 1
        existing_ontop_coords.append(coords)
        sites.append(
            {
                "site_id": f"ontop-{ontop_count:02d}",
                "site_family": "ontop",
                "site_index_within_family": ontop_count,
                "cart_coords": [float(value) for value in coords],
                "nearest_top_atoms": [
                    {
                        "atom_index": atom["atom_index"],
                        "element": atom["element"],
                        "in_plane_distance": 0.0,
                    }
                ],
                "note": "concrete top-atom ontop site exposed after symmetry-pruned AdsorbateSiteFinder output",
            }
        )
        site_family_counts["ontop"] = ontop_count

    return {
        "status": "ok",
        "slab_path": str(slab_path),
        "summary": _structure_summary(structure),
        "slab_top_z": top_z,
        "top_layer_atoms": top_layer,
        "site_families": site_family_counts,
        "sites": sites,
        "guidance": (
            "把 cart_coords 传给 structure_add_adsorbate 的 cart_coords 参数，可精确放置吸附物；"
            "也可用 site_index（顶层原子 0-based 序号）走简化路径。"
            "建议每个候选生成后跑 structure_sanity_check。"
        ),
    }


def inspect_slab_surface(
    slab_path: str,
    *,
    top_layer_tolerance: float = 0.75,
    second_layer_tolerance: float = 2.5,
    neighbor_radius: float = 3.5,
    symprec: float = 0.05,
) -> dict[str, Any]:
    """报告 slab 表面化学环境，让模型在生成候选前先看懂表面。

    返回：
    - ``top_layer``: 顶层原子完整摘要（含配位数、最近邻、对称等价分组 id）
    - ``second_layer``: 第二层原子摘要（便于识别 stepped / 合金面）
    - ``symmetry_groups``: 顶层对称等价分组（同 group 的原子等价，只需保留一个）
    - ``surface_composition``: 顶层 / 第二层元素分布
    - ``lattice``: 晶格摘要
    - ``guidance``: 给模型的使用提示
    """
    structure = _load_structure(slab_path)
    if len(structure) == 0:
        raise ValueError("slab 为空。")
    top_z = max(float(site.coords[2]) for site in structure)

    top_indices = [
        idx for idx, site in enumerate(structure)
        if top_z - float(site.coords[2]) <= top_layer_tolerance
    ]
    second_indices = [
        idx for idx, site in enumerate(structure)
        if top_layer_tolerance < (top_z - float(site.coords[2])) <= second_layer_tolerance
    ]

    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        analyzer = SpacegroupAnalyzer(structure, symprec=symprec)
        equivalent_sets = analyzer.get_symmetry_dataset().equivalent_atoms.tolist()
        spacegroup = analyzer.get_space_group_symbol()
    except Exception:
        equivalent_sets = list(range(len(structure)))
        spacegroup = None

    sym_group_remap: dict[int, int] = {}
    next_group_id = 0
    for site_index in top_indices:
        canonical = int(equivalent_sets[site_index])
        if canonical not in sym_group_remap:
            sym_group_remap[canonical] = next_group_id
            next_group_id += 1

    def _layer_atom_summary(indices: list[int]) -> list[dict[str, Any]]:
        atoms: list[dict[str, Any]] = []
        for idx in sorted(indices, key=lambda i: -float(structure[i].coords[2])):
            site = structure[idx]
            neighbours: list[dict[str, Any]] = []
            try:
                raw_neighbours = structure.get_neighbors(site, neighbor_radius)
            except Exception:
                raw_neighbours = []
            for neighbour in sorted(raw_neighbours, key=lambda n: float(n.nn_distance))[:8]:
                neighbour_index = getattr(neighbour, "index", None)
                neighbours.append(
                    {
                        "neighbor_index": int(neighbour_index) if neighbour_index is not None else None,
                        "element": neighbour.specie.symbol,
                        "distance": float(neighbour.nn_distance),
                        "z_offset": float(neighbour.coords[2]) - float(site.coords[2]),
                    }
                )
            canonical = int(equivalent_sets[idx])
            atoms.append(
                {
                    "atom_index": idx,
                    "element": site.specie.symbol,
                    "cart_coords": [float(v) for v in site.coords],
                    "z_offset_from_top": float(site.coords[2]) - top_z,
                    "coordination_number": len(raw_neighbours),
                    "nearest_neighbors": neighbours,
                    "symmetry_group_id": sym_group_remap.get(canonical, -1),
                    "symmetry_canonical_atom_index": canonical,
                }
            )
        return atoms

    top_layer = _layer_atom_summary(top_indices)
    second_layer = _layer_atom_summary(second_indices)

    def _composition(atoms: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for atom in atoms:
            counts[atom["element"]] = counts.get(atom["element"], 0) + 1
        return counts

    symmetry_groups: list[dict[str, Any]] = []
    by_group: dict[int, list[dict[str, Any]]] = {}
    for atom in top_layer:
        by_group.setdefault(atom["symmetry_group_id"], []).append(atom)
    for group_id in sorted(by_group):
        members = by_group[group_id]
        symmetry_groups.append(
            {
                "group_id": group_id,
                "representative_atom_index": members[0]["atom_index"],
                "representative_element": members[0]["element"],
                "member_atom_indices": [atom["atom_index"] for atom in members],
                "multiplicity": len(members),
            }
        )

    return {
        "status": "ok",
        "slab_path": str(slab_path),
        "summary": _structure_summary(structure),
        "lattice": {
            "abc": [float(v) for v in structure.lattice.abc],
            "angles": [float(v) for v in structure.lattice.angles],
        },
        "spacegroup": spacegroup,
        "slab_top_z": top_z,
        "top_layer": top_layer,
        "second_layer": second_layer,
        "symmetry_groups": symmetry_groups,
        "surface_composition": {
            "top_layer": _composition(top_layer),
            "second_layer": _composition(second_layer),
        },
        "guidance": (
            "对称等价的顶层原子只需保留一个作为候选 anchor (见 symmetry_groups.representative_atom_index)；"
            "高配位数原子通常是 ontop 强结合位；低配位数顶层原子（台阶/边角）通常对小分子结合更强。"
            "把发现写进 adsorption_candidate_plan.rationale，再决定要展开哪些 site_family。"
        ),
    }


def candidate_quality_score(
    *,
    slab_path: str,
    candidate_path: str,
    adsorbate: str | None = None,
    anchor_symbol: str | None = None,
    top_layer_tolerance: float = 0.75,
    min_anchor_surface_distance: float = 1.1,
    max_anchor_surface_distance: float = 3.5,
    min_adsorbate_slab_distance: float = 0.65,
) -> dict[str, Any]:
    """对模型自主生成的吸附候选做轻量几何自检。

    这是 Phase 4 的确定性 guardrail：不替模型决定科学优先级，只回答这个
    POSCAR 是否像一个可送去预优化/DFT 的初猜。默认假设 candidate 由
    ``slab + adsorbate`` 顺序拼接而成（``structure_add_adsorbate`` 与黑盒候选生成器
    都满足这个约定）。
    """
    slab = _load_structure(slab_path)
    candidate = _load_structure(candidate_path)
    if len(slab) == 0:
        raise ValueError("slab 为空，无法评分。")
    if len(candidate) <= len(slab):
        return {
            "status": "warning",
            "verdict": "reject",
            "score": {"total": 0.0, "breakdown": {"has_adsorbate": 0.0}},
            "issues": ["candidate 原子数不多于 slab，未检测到吸附物原子。"],
            "measurements": {"slab_atom_count": len(slab), "candidate_atom_count": len(candidate)},
        }

    slab_indices = list(range(len(slab)))
    ads_indices = list(range(len(slab), len(candidate)))
    top_z = max(float(site.coords[2]) for site in slab)
    top_layer_indices = [
        idx for idx, site in enumerate(slab)
        if top_z - float(site.coords[2]) <= top_layer_tolerance
    ] or [max(slab_indices, key=lambda idx: float(slab[idx].coords[2]))]

    ads_symbols = [candidate[idx].specie.symbol for idx in ads_indices]
    anchor_candidates = [
        idx for idx in ads_indices
        if anchor_symbol and candidate[idx].specie.symbol.lower() == anchor_symbol.lower()
    ]
    anchor_index = anchor_candidates[0] if anchor_candidates else min(ads_indices, key=lambda idx: float(candidate[idx].coords[2]))
    anchor_coords = np.array(candidate[anchor_index].coords, dtype=float)

    top_distances = [
        float(np.linalg.norm(anchor_coords - np.array(slab[idx].coords, dtype=float)))
        for idx in top_layer_indices
    ]
    anchor_surface_distance = min(top_distances) if top_distances else None
    all_ads_slab_distances = [
        float(np.linalg.norm(np.array(candidate[a].coords, dtype=float) - np.array(slab[s].coords, dtype=float)))
        for a in ads_indices
        for s in slab_indices
    ]
    min_ads_slab_distance = min(all_ads_slab_distances) if all_ads_slab_distances else None

    expected_adsorbate_atoms = None
    if adsorbate:
        try:
            from dft_app.modeling.adsorption_candidate_generator import AdsorbateStructureFactory

            expected_adsorbate_atoms = len(AdsorbateStructureFactory().load_adsorbate_atoms(adsorbate))
        except Exception:
            expected_adsorbate_atoms = None

    internal_bond_count = 0
    if len(ads_indices) > 1:
        from ase.data import atomic_numbers, covalent_radii

        for pos_i, idx_i in enumerate(ads_indices):
            symbol_i = candidate[idx_i].specie.symbol
            radius_i = float(covalent_radii[atomic_numbers.get(symbol_i, 0)] or 0.7)
            for idx_j in ads_indices[pos_i + 1:]:
                symbol_j = candidate[idx_j].specie.symbol
                radius_j = float(covalent_radii[atomic_numbers.get(symbol_j, 0)] or 0.7)
                distance = float(
                    np.linalg.norm(
                        np.array(candidate[idx_i].coords, dtype=float) - np.array(candidate[idx_j].coords, dtype=float)
                    )
                )
                if distance <= 1.35 * (radius_i + radius_j):
                    internal_bond_count += 1

    issues: list[str] = []
    if expected_adsorbate_atoms is not None and expected_adsorbate_atoms != len(ads_indices):
        issues.append(
            f"吸附物原子数不匹配：期望 {expected_adsorbate_atoms}，candidate 中检测到 {len(ads_indices)}。"
        )
    if anchor_symbol and not anchor_candidates:
        issues.append(f"candidate 吸附物片段中未找到 anchor_symbol={anchor_symbol}，已回退到最低 z 原子。")
    if anchor_surface_distance is not None and anchor_surface_distance < min_anchor_surface_distance:
        issues.append(
            f"anchor-surface 距离 {anchor_surface_distance:.2f} Å 过近，低于 {min_anchor_surface_distance:.2f} Å。"
        )
    if anchor_surface_distance is not None and anchor_surface_distance > max_anchor_surface_distance:
        issues.append(
            f"anchor-surface 距离 {anchor_surface_distance:.2f} Å 过远，疑似 floating adsorbate。"
        )
    if min_ads_slab_distance is not None and min_ads_slab_distance < min_adsorbate_slab_distance:
        issues.append(
            f"吸附物-slab 最短距离 {min_ads_slab_distance:.2f} Å 过近，疑似原子重叠。"
        )
    if len(ads_indices) > 1 and internal_bond_count == 0:
        issues.append("吸附物内部未检测到合理共价键，可能被拆散或 atom order 不符合 slab+adsorbate 约定。")

    if anchor_surface_distance is None:
        anchor_distance_score = 0.0
    elif anchor_surface_distance < min_anchor_surface_distance:
        anchor_distance_score = max(0.0, anchor_surface_distance / min_anchor_surface_distance)
    elif anchor_surface_distance > max_anchor_surface_distance:
        anchor_distance_score = max(
            0.0,
            1.0 - (anchor_surface_distance - max_anchor_surface_distance) / max_anchor_surface_distance,
        )
    else:
        anchor_distance_score = 1.0
    overlap_score = 0.0 if (min_ads_slab_distance is not None and min_ads_slab_distance < min_adsorbate_slab_distance) else 1.0
    integrity_score = 0.0 if (len(ads_indices) > 1 and internal_bond_count == 0) else 1.0
    count_score = 1.0 if expected_adsorbate_atoms in {None, len(ads_indices)} else 0.5
    total = round(
        0.40 * anchor_distance_score
        + 0.25 * overlap_score
        + 0.25 * integrity_score
        + 0.10 * count_score,
        3,
    )
    if total >= 0.75 and not issues:
        verdict = "pass"
    elif total >= 0.45:
        verdict = "retry"
    else:
        verdict = "reject"

    return {
        "status": "ok" if not issues else "warning",
        "verdict": verdict,
        "score": {
            "total": total,
            "breakdown": {
                "anchor_distance": round(anchor_distance_score, 3),
                "no_overlap": overlap_score,
                "adsorbate_integrity": integrity_score,
                "adsorbate_atom_count": count_score,
            },
        },
        "issues": issues,
        "measurements": {
            "slab_atom_count": len(slab),
            "candidate_atom_count": len(candidate),
            "adsorbate_atom_count": len(ads_indices),
            "expected_adsorbate_atom_count": expected_adsorbate_atoms,
            "adsorbate_symbols": ads_symbols,
            "anchor_candidate_index": anchor_index,
            "anchor_symbol": candidate[anchor_index].specie.symbol,
            "anchor_surface_distance": anchor_surface_distance,
            "min_adsorbate_slab_distance": min_ads_slab_distance,
            "internal_adsorbate_bond_count": internal_bond_count,
            "top_layer_atom_indices": top_layer_indices,
        },
        "guidance": (
            "verdict=retry/reject 时回到 structure_add_adsorbate 调整 height、cart_coords、orientation "
            "或 anchor_symbol；score < 0.5 不要写入最终 manifest。"
        ),
    }


def structure_relax_short(
    *,
    input_path: str,
    output_path: str,
    calculator: str = "emt",
    max_steps: int = 20,
    fmax: float = 0.2,
    trajectory_path: str | None = None,
) -> dict[str, Any]:
    """Run a real short local geometry relaxation when an ASE calculator is available.

    This is a lightweight pre-screening primitive, not a replacement for DFT.
    It never fabricates success: unsupported calculators/elements return
    ``status='unavailable'`` or ``status='failed'`` with the exception message.
    """
    from ase.io import read as ase_read, write as ase_write
    from ase.optimize import LBFGS

    calc_name = (calculator or "emt").strip().lower()
    if calc_name != "emt":
        return {
            "status": "unavailable",
            "calculator": calc_name,
            "message": "当前无新增依赖前提下只支持 ASE EMT；MACE/GFN-FF 需安装外部 calculator 后再接入。",
        }

    atoms = ase_read(input_path)
    try:
        from ase.calculators.emt import EMT

        atoms.calc = EMT()
        initial_energy = float(atoms.get_potential_energy())
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        traj = str(trajectory_path) if trajectory_path else None
        opt = LBFGS(atoms, trajectory=traj, logfile=None)
        opt.run(fmax=float(fmax), steps=max(1, int(max_steps)))
        final_energy = float(atoms.get_potential_energy())
        ase_write(target, atoms, format="vasp", direct=True, vasp5=True)
        return {
            "status": "ok",
            "calculator": "emt",
            "input_path": str(input_path),
            "output_path": str(target),
            "trajectory_path": traj,
            "max_steps": int(max_steps),
            "fmax": float(fmax),
            "initial_energy_ev": initial_energy,
            "final_energy_ev": final_energy,
            "energy_delta_ev": final_energy - initial_energy,
            "converged": bool(opt.converged()),
            "boundary": "ASE EMT 短程预优化，仅作几何筛查；不能替代 VASP/DFT。",
        }
    except Exception as exc:
        return {
            "status": "failed",
            "calculator": "emt",
            "input_path": str(input_path),
            "output_path": str(output_path),
            "message": str(exc),
            "boundary": "未生成可信 relaxed 结构；不要把该结果当作成功预优化。",
        }


def enumerate_defect_sites(
    structure_path: str,
    *,
    species: str | None = None,
    surface_only: bool = True,
    top_layer_tolerance: float = 0.75,
    max_sites: int = 12,
) -> dict[str, Any]:
    """枚举 vacancy / substitution 可操作的原子位点。

    这是 Phase 6 的缺陷主线原语：只列出候选位点与推荐 mode，不替模型决定
    应该删哪个原子或掺哪个元素。
    """
    structure = _load_structure(structure_path)
    if len(structure) == 0:
        raise ValueError("structure 为空。")
    top_z = max(float(site.coords[2]) for site in structure)
    selected: list[dict[str, Any]] = []
    for idx, site in enumerate(structure):
        element = site.specie.symbol
        if species and element.lower() != species.lower():
            continue
        z_offset = float(site.coords[2]) - top_z
        is_surface = top_z - float(site.coords[2]) <= top_layer_tolerance
        if surface_only and not is_surface:
            continue
        selected.append(
            {
                "site_id": f"{element.lower()}-{idx:03d}",
                "atom_index": idx,
                "element": element,
                "cart_coords": [float(v) for v in site.coords],
                "z_offset_from_top": z_offset,
                "is_surface": is_surface,
                "suggested_modes": ["vacancy", "substitution"],
                "reason_prompt": "模型需结合价态/配位/先验说明为什么选择该缺陷位点。",
            }
        )

    selected.sort(key=lambda item: (not item["is_surface"], item["element"], item["atom_index"]))
    limited = selected[: max(1, int(max_sites))]
    return {
        "status": "ok",
        "structure_path": str(structure_path),
        "summary": _structure_summary(structure),
        "filters": {
            "species": species,
            "surface_only": surface_only,
            "top_layer_tolerance": top_layer_tolerance,
            "max_sites": max_sites,
        },
        "candidate_count": len(limited),
        "candidates": limited,
        "guidance": (
            "先把候选写入 defect/adsorption_candidate_plan 的 rationale；"
            "再用 structure_defect(mode=vacancy/substitution, index=atom_index) 生成具体结构。"
        ),
    }


def interpolate_ts_midpoint_candidates(
    *,
    initial_path: str,
    final_path: str,
    output_dir: str,
    n_images: int = 3,
) -> dict[str, Any]:
    """基于 IS/FS 线性插值生成 TS/NEB 中间构型初猜。

    不执行 NEB/Dimer，也不声称找到了过渡态；只产出可审查的几何初猜。
    """
    from pymatgen.io.vasp import Poscar

    initial = _load_structure(initial_path)
    final = _load_structure(final_path)
    if len(initial) != len(final):
        raise ValueError("initial/final 原子数不同，不能做一一对应线性插值。")
    if [site.specie.symbol for site in initial] != [site.specie.symbol for site in final]:
        raise ValueError("initial/final 原子顺序或元素不同，不能安全插值。")
    count = max(1, int(n_images))
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    initial_coords = np.array([site.coords for site in initial], dtype=float)
    final_coords = np.array([site.coords for site in final], dtype=float)
    images: list[dict[str, Any]] = []
    for image_index in range(1, count + 1):
        fraction = image_index / (count + 1)
        interpolated = initial.copy()
        new_coords = (1.0 - fraction) * initial_coords + fraction * final_coords
        for atom_index, coords in enumerate(new_coords):
            interpolated.replace(atom_index, initial[atom_index].specie, coords=coords, coords_are_cartesian=True)
        image_dir = target_dir / f"image_{image_index:02d}"
        image_dir.mkdir(parents=True, exist_ok=True)
        poscar_path = image_dir / "POSCAR"
        Poscar(interpolated).write_file(poscar_path)
        images.append(
            {
                "image_id": f"image_{image_index:02d}",
                "fraction": fraction,
                "poscar_path": str(poscar_path),
                "summary": _structure_summary(interpolated),
            }
        )
    manifest = {
        "status": "ok",
        "initial_path": str(initial_path),
        "final_path": str(final_path),
        "n_images": count,
        "images": images,
        "boundary": "线性插值初猜；未执行 NEB/Dimer，也不代表真实 TS。",
    }
    (target_dir / "ts_midpoint_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def sanity_check(
    structure_path: str,
    *,
    min_distance: float = 0.65,
    min_vacuum: float = 6.0,
    vacuum_axis: str = "c",
) -> dict[str, Any]:
    structure = _load_structure(structure_path)
    matrix = structure.distance_matrix
    min_pair: dict[str, Any] | None = None
    if len(structure) >= 2:
        masked = matrix + np.eye(len(structure)) * 9999.0
        flat_index = int(np.argmin(masked))
        i, j = divmod(flat_index, len(structure))
        min_pair = {
            "i": i,
            "j": j,
            "species": [structure[i].specie.symbol, structure[j].specie.symbol],
            "distance": float(masked[i, j]),
        }
    axis = {"a": 0, "b": 1, "c": 2, "x": 0, "y": 1, "z": 2}.get(vacuum_axis.lower())
    if axis is None:
        raise ValueError("vacuum_axis 必须是 a/b/c 或 x/y/z。")
    axis_values = [float(site.coords[axis]) for site in structure]
    axis_span = max(axis_values) - min(axis_values) if axis_values else 0.0
    lengths = [float(item) for item in structure.lattice.abc]
    axis_length = lengths[axis]
    issues: list[str] = []
    if min_pair and min_pair["distance"] < min_distance:
        issues.append(f"最短距离 {min_pair['distance']:.3f} Å 小于阈值 {min_distance:.3f} Å")
    if axis_length > 0 and axis_length - axis_span < min_vacuum:
        issues.append(f"沿 {vacuum_axis} 方向估算真空/空隙可能不足 {min_vacuum:.1f} Å；若这是 slab，请人工确认真空层。")
    return {
        "status": "ok" if not issues else "warning",
        "structure_path": str(structure_path),
        "summary": _structure_summary(structure),
        "min_pair": min_pair,
        "vacuum_axis": vacuum_axis,
        "axis_span": axis_span,
        "axis_length": axis_length,
        "estimated_empty_axis": axis_length - axis_span,
        "z_span": max([float(site.coords[2]) for site in structure]) - min([float(site.coords[2]) for site in structure]) if len(structure) else 0.0,
        "c_length": float(structure.lattice.c),
        "estimated_empty_c": float(structure.lattice.c) - (max([float(site.coords[2]) for site in structure]) - min([float(site.coords[2]) for site in structure]) if len(structure) else 0.0),
        "issues": issues,
    }
