from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from ase import Atoms
from ase.build import molecule as ase_molecule
from ase.io import read as ase_read
from pymatgen.core import Structure
from pymatgen.core.adsorption import AdsorbateSiteFinder
from pymatgen.io.ase import AseAtomsAdaptor

from .adsorption_models import AdsorptionCandidate
from .candidate_ranker import (
    AdsorptionCandidateRanker,
    AdsorptionRankingContext,
    minimum_intergroup_distance,
)

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except Exception:  # pragma: no cover - optional dependency
    Chem = None
    AllChem = None

def apply_selective_dynamics(
    structure: Structure,
    slab_atom_count: int,
    fixed_bottom_layers: int = 2,
    layer_tolerance: float = 0.5,
) -> Structure:
    """为 slab 结构自动添加 selective dynamics：冻结底部 N 层原子。

    对 combined (slab+adsorbate) 结构，slab 部分前 slab_atom_count 个原子按 z
    坐标分层，底部 fixed_bottom_layers 层标记为 [F, F, F]，其余（含 adsorbate）
    标记为 [T, T, T]。对纯 slab 结构，slab_atom_count 设为 len(structure) 即可。
    """
    if fixed_bottom_layers <= 0 or slab_atom_count <= 0:
        return structure

    slab_z = sorted(
        {round(float(structure[i].coords[2]), 4) for i in range(min(slab_atom_count, len(structure)))}
    )

    # 按 tolerance 聚类为层
    layers: list[float] = []
    for z in slab_z:
        if not layers or abs(z - layers[-1]) > layer_tolerance:
            layers.append(z)
        else:
            layers[-1] = (layers[-1] + z) / 2.0  # 更新为层平均值

    frozen_z_max = layers[min(fixed_bottom_layers, len(layers)) - 1] + layer_tolerance / 2.0

    flags: list[list[bool]] = []
    for i, site in enumerate(structure):
        if i < slab_atom_count and float(site.coords[2]) <= frozen_z_max:
            flags.append([False, False, False])
        else:
            flags.append([True, True, True])

    new_structure = structure.copy()
    new_structure.add_site_property("selective_dynamics", flags)
    return new_structure


@dataclass
class AdsorptionGenerationRequest:
    slab_structure: Structure
    adsorbate_source: str
    task_id: str
    material_name: str
    source_prompt: str
    defect_recipe: dict[str, Any] | None = None
    preferred_site: str | None = None
    preferred_orientation: str | None = None
    candidate_height: float | None = None
    max_sites_per_family: int = 2
    orientation_labels: tuple[str, ...] = ("upright", "flat", "tilted")
    fixed_bottom_layers: int = 2


class AdsorptionCandidateGenerator:
    """Generate first-version adsorption candidates from a slab + adsorbate input."""

    def __init__(self) -> None:
        self._ranker = AdsorptionCandidateRanker()
        self._factory = AdsorbateStructureFactory()

    def generate(self, request: AdsorptionGenerationRequest) -> list[AdsorptionCandidate]:
        working_slab = request.slab_structure.copy()
        defect_label = None
        if request.defect_recipe:
            working_slab, defect_label = self._apply_defect_recipe(
                working_slab,
                request.defect_recipe,
            )

        slab_atoms = AseAtomsAdaptor.get_atoms(working_slab)
        adsorbate_atoms = self._factory.load_adsorbate_atoms(request.adsorbate_source)
        anchor_index = self._preferred_anchor_index(adsorbate_atoms)
        anchor_symbol = adsorbate_atoms[anchor_index].symbol

        # 吸附高度：用户显式指定时使用指定值，否则根据分子大小自适应
        effective_height = (
            request.candidate_height
            if request.candidate_height is not None
            else self._estimate_height(adsorbate_atoms, anchor_index)
        )

        surface_sites = self._enumerate_surface_sites(
            working_slab,
            max_sites_per_family=request.max_sites_per_family,
        )
        slab_atom_count = len(slab_atoms)

        candidates: list[AdsorptionCandidate] = []
        for site in surface_sites:
            for orientation_label in request.orientation_labels:
                combined_structure, metadata = self._build_candidate_structure(
                    working_slab=working_slab,
                    slab_atoms=slab_atoms,
                    adsorbate_atoms=adsorbate_atoms,
                    anchor_index=anchor_index,
                    height=effective_height,
                    site=site,
                    orientation_label=orientation_label,
                )
                # 自动添加 selective dynamics：冻结底部 slab 层
                combined_structure = apply_selective_dynamics(
                    combined_structure,
                    slab_atom_count=slab_atom_count,
                    fixed_bottom_layers=request.fixed_bottom_layers,
                )
                candidate_id = self._candidate_id(site["family"], site["index"], orientation_label)
                candidates.append(
                    AdsorptionCandidate(
                        candidate_id=candidate_id,
                        site_family=site["family"],
                        site_label=site["label"],
                        orientation_label=orientation_label,
                        anchor_symbol=anchor_symbol,
                        height=effective_height,
                        defect_label=defect_label,
                        structure=combined_structure,
                        metadata={
                            **metadata,
                            "task_id": request.task_id,
                            "material_name": request.material_name,
                            "source_prompt": request.source_prompt,
                            "adsorbate_source": request.adsorbate_source,
                            "defect_recipe": request.defect_recipe,
                            "fixed_bottom_layers": request.fixed_bottom_layers,
                        },
                    )
                )

        return self._ranker.rank(
            candidates,
            context=AdsorptionRankingContext(
                preferred_site=request.preferred_site,
                preferred_orientation=request.preferred_orientation,
                target_height=effective_height,
            ),
        )

    def _enumerate_surface_sites(
        self,
        slab_structure: Structure,
        *,
        max_sites_per_family: int,
    ) -> list[dict[str, Any]]:
        site_records: list[dict[str, Any]] = []
        finder = AdsorbateSiteFinder(slab_structure)
        sites = finder.find_adsorption_sites()
        for family, coords_list in sites.items():
            if family == "all":
                continue
            for index, coords in enumerate(coords_list[:max_sites_per_family], start=1):
                site_records.append(
                    {
                        "family": str(family),
                        "index": index,
                        "label": f"{family}-{index:02d}",
                        "cart_coords": np.array(coords, dtype=float),
                    }
                )
        if site_records:
            return site_records

        # fallback: use top-surface atoms when AdsorbateSiteFinder yields nothing
        z_coords = [site.z for site in slab_structure]
        z_cutoff = max(z_coords) - 1.0
        top_sites = [site for site in slab_structure if site.z >= z_cutoff]
        for index, site in enumerate(top_sites[: max(1, max_sites_per_family)], start=1):
            site_records.append(
                {
                    "family": "top",
                    "index": index,
                    "label": f"top-{index:02d}",
                    "cart_coords": np.array(site.coords, dtype=float),
                }
            )
        return site_records

    def _build_candidate_structure(
        self,
        *,
        working_slab: Structure,
        slab_atoms: Atoms,
        adsorbate_atoms: Atoms,
        anchor_index: int,
        height: float,
        site: dict[str, Any],
        orientation_label: str,
    ) -> tuple[Structure, dict[str, Any]]:
        ads_copy = adsorbate_atoms.copy()
        self._orient_adsorbate(ads_copy, anchor_index=anchor_index, orientation_label=orientation_label)

        slab_top_z = max(atom.position[2] for atom in slab_atoms)
        target = np.array(site["cart_coords"], dtype=float)
        target[2] = slab_top_z + height
        anchor_position = np.array(ads_copy[anchor_index].position, dtype=float)
        ads_copy.translate(target - anchor_position)

        combined_atoms = slab_atoms.copy()
        combined_atoms.extend(ads_copy)
        combined_structure = AseAtomsAdaptor.get_structure(combined_atoms)
        clearance = minimum_intergroup_distance(combined_structure, len(slab_atoms))
        ads_centroid_z = float(np.mean([atom.position[2] for atom in ads_copy])) if len(ads_copy) else slab_top_z
        metadata = {
            "site_cart_coords": target.tolist(),
            "site_family": site["family"],
            "site_label": site["label"],
            "orientation_label": orientation_label,
            "minimum_clearance": clearance,
            "adsorbate_centroid_z": ads_centroid_z,
            "slab_top_z": float(slab_top_z),
            "slab_atom_count": len(slab_atoms),
            "adsorbate_atom_count": len(ads_copy),
        }
        return combined_structure, metadata

    @staticmethod
    def _candidate_id(site_family: str, site_index: int, orientation_label: str) -> str:
        normalized_family = site_family.lower().replace(" ", "_")
        normalized_orientation = orientation_label.lower().replace(" ", "_")
        return f"{normalized_family}_{site_index:02d}_{normalized_orientation}"

    @staticmethod
    def _preferred_anchor_index(atoms: Atoms) -> int:
        preferred = ["O", "N", "S", "P", "C"]
        for symbol in preferred:
            for index, atom in enumerate(atoms):
                if atom.symbol == symbol:
                    return index
        return 0

    @staticmethod
    def _estimate_height(adsorbate_atoms: Atoms, anchor_index: int) -> float:
        """根据吸附物大小自适应估计初始吸附高度（A）。"""
        n = len(adsorbate_atoms)
        if n == 1:
            symbol = adsorbate_atoms[anchor_index].symbol
            return 1.0 if symbol == "H" else 1.5
        if n <= 3:
            return 1.8
        if n <= 8:
            return 2.1
        return 2.5

    @staticmethod
    def _orient_adsorbate(
        atoms: Atoms,
        *,
        anchor_index: int,
        orientation_label: str,
    ) -> None:
        positions = atoms.get_positions()
        anchor = positions[anchor_index].copy()
        positions -= anchor
        atoms.set_positions(positions)
        if orientation_label == "upright":
            return
        if orientation_label == "flat":
            atoms.rotate(90.0, "x", center=(0.0, 0.0, 0.0))
            return
        if orientation_label == "tilted":
            atoms.rotate(45.0, "x", center=(0.0, 0.0, 0.0))
            return
        if orientation_label == "inverted":
            atoms.rotate(180.0, "x", center=(0.0, 0.0, 0.0))

    @staticmethod
    def _apply_defect_recipe(
        structure: Structure,
        defect_recipe: dict[str, Any],
    ) -> tuple[Structure, str | None]:
        recipe = dict(defect_recipe)
        if str(recipe.get("mode") or "").lower() != "vacancy":
            return structure, None
        species = recipe.get("species") or recipe.get("site")
        if not species:
            return structure, "vacancy-unspecified"
        matching_indexes = [
            index
            for index, site in enumerate(structure)
            if site.specie.symbol.lower() == str(species).lower()
        ]
        if not matching_indexes:
            return structure, f"vacancy-{species}-missing"
        if recipe.get("surface_only", True):
            selected = max(matching_indexes, key=lambda idx: float(structure[idx].coords[2]))
        else:
            selected = matching_indexes[0]
        mutated = structure.copy()
        mutated.remove_sites([selected])
        return mutated, f"vacancy-{species}@{selected}"

class AdsorbateStructureFactory:
    """Load adsorbate structures from names, files, or simple SMILES and box them when needed."""

    def load_adsorbate_atoms(self, adsorbate_source: str) -> Atoms:
        source_path = Path(adsorbate_source)
        if source_path.exists():
            return self._load_adsorbate_from_file(source_path)

        token = adsorbate_source.strip()
        for candidate in self._ase_name_candidates(token):
            try:
                return ase_molecule(candidate)
            except Exception:
                continue

        rdkit_atoms = self._load_adsorbate_from_rdkit(token)
        if rdkit_atoms is not None:
            return rdkit_atoms

        raise ValueError(
            f"无法识别吸附物来源: {adsorbate_source}。请提供 ASE 支持的分子名、分子文件路径或可解析的 SMILES。"
        )

    def build_boxed_structure(
        self,
        adsorbate_source: str,
        *,
        box_lengths: tuple[float, float, float] = (20.0, 20.0, 20.0),
    ) -> Structure:
        atoms = self.load_adsorbate_atoms(adsorbate_source)
        atoms = atoms.copy()
        atoms.center(vacuum=0.0)
        atoms.set_cell(box_lengths)
        atoms.center()
        return AseAtomsAdaptor.get_structure(atoms)

    def _load_adsorbate_from_file(self, source_path: Path) -> Atoms:
        suffix = source_path.suffix.lower()
        if suffix in {".mol", ".sdf", ".mol2", ".pdb"}:
            rdkit_atoms = self._load_adsorbate_from_rdkit_file(source_path)
            if rdkit_atoms is not None:
                return rdkit_atoms
        return ase_read(source_path)

    def _load_adsorbate_from_rdkit(self, smiles: str) -> Atoms | None:
        if Chem is None or AllChem is None:
            return None
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return None
        molecule = Chem.AddHs(molecule)
        status = AllChem.EmbedMolecule(molecule, randomSeed=7)
        if status != 0:
            return None
        AllChem.UFFOptimizeMolecule(molecule)
        return self._rdkit_mol_to_atoms(molecule)

    def _load_adsorbate_from_rdkit_file(self, source_path: Path) -> Atoms | None:
        if Chem is None or AllChem is None:
            return None
        suffix = source_path.suffix.lower()
        molecule = None
        if suffix == ".mol":
            molecule = Chem.MolFromMolFile(str(source_path), removeHs=False)
        elif suffix == ".sdf":
            supplier = Chem.SDMolSupplier(str(source_path), removeHs=False)
            molecule = supplier[0] if supplier else None
        elif suffix == ".mol2":
            molecule = Chem.MolFromMol2File(str(source_path), removeHs=False)
        elif suffix == ".pdb":
            molecule = Chem.MolFromPDBFile(str(source_path), removeHs=False)
        if molecule is None:
            return None
        if molecule.GetNumConformers() == 0:
            molecule = Chem.AddHs(molecule, addCoords=True)
            if AllChem.EmbedMolecule(molecule, randomSeed=7) != 0:
                return None
            AllChem.UFFOptimizeMolecule(molecule)
        return self._rdkit_mol_to_atoms(molecule)

    @staticmethod
    def _rdkit_mol_to_atoms(molecule: Any) -> Atoms:
        conformer = molecule.GetConformer()
        symbols = [atom.GetSymbol() for atom in molecule.GetAtoms()]
        coords = []
        for index in range(molecule.GetNumAtoms()):
            point = conformer.GetAtomPosition(index)
            coords.append((point.x, point.y, point.z))
        return Atoms(symbols=symbols, positions=coords)

    @staticmethod
    def _ase_name_candidates(token: str) -> Iterable[str]:
        mapping = {
            "OH": ["OH", "OH2"],
            "H2O": ["H2O", "water"],
            "CO": ["CO"],
            "CO2": ["CO2"],
            "NH3": ["NH3"],
            "CH4": ["CH4"],
            "H": ["H"],
        }
        token_upper = token.upper()
        if token_upper in mapping:
            return mapping[token_upper]
        return [token, token_upper, token.lower()]
