from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import io
import json
import re
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from .project_state import append_progress, project_paths
from .task_bridge import create_task_plan
from dft_app.llm.key_store import resolve_api_key


@dataclass(frozen=True)
class AdsorptionPlan:
    task_id: str
    prompt: str
    project: str | None
    adsorbate: str | None
    material: str | None
    slab_path: str | None
    readiness: str
    missing_inputs: list[str]
    next_research_tasks: list[str]
    dft_entrypoints: dict[str, str]
    task_record: dict[str, Any]
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SlabBuildResult:
    status: str
    material: str
    source: str
    source_detail: str | None
    miller_index: tuple[int, int, int]
    slab_path: str
    output_dir: str
    metadata_path: str
    atom_count: int
    formula: str
    next_research_tasks: list[str]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _infer_adsorbate(prompt: str) -> str | None:
    formula_matches = re.findall(r"\b(H2O|OH|CO2|CO|NH3|NO2|NO|O2|N2|H2|CH4|CH3OH)\b", prompt, flags=re.IGNORECASE)
    if formula_matches:
        return formula_matches[-1]
    zh_match = re.search(r"(.+?)在.+?(?:表面|slab|面)上.*(?:吸附|adsorb)", prompt, flags=re.IGNORECASE)
    if zh_match:
        value = zh_match.group(1).strip(" ，,。")
        return value or None
    en_match = re.search(r"(?:adsorption|adsorb)\s+(?:of\s+)?(.+?)\s+on\s+", prompt, flags=re.IGNORECASE)
    if en_match:
        return en_match.group(1).strip() or None
    return None


def _infer_miller_from_material(material: str | None) -> tuple[int, int, int] | None:
    if not material:
        return None
    match = re.search(r"\((\d)\s*(\d)\s*(\d)\)", material)
    if match:
        return tuple(int(item) for item in match.groups())  # type: ignore[return-value]
    match = re.search(r"\b([0-3])\s+([0-3])\s+([0-3])\b", material)
    if match:
        return tuple(int(item) for item in match.groups())  # type: ignore[return-value]
    return None


def _clean_material_formula(material: str | None) -> str:
    text = (material or "").strip()
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.split(r"[/@,，\s]+", text)[0]
    return text.strip()


def _load_bulk_from_ase(material: str) -> tuple[Any, dict[str, Any]]:
    """Build an elemental bulk structure with ASE as a no-API fallback."""
    try:
        from ase.build import bulk
        from pymatgen.core import Composition, Element
        from pymatgen.io.ase import AseAtomsAdaptor
    except Exception as exc:  # pragma: no cover - dependency is declared, but keep honest errors.
        raise RuntimeError(f"ASE/pymatgen 兜底建模依赖不可用: {exc}") from exc

    formula = _clean_material_formula(material)
    if not formula:
        raise ValueError("无法从 material 推断元素；请提供 --structure-path、--mp-id 或类似 Pt(111) 的材料名。")
    composition = Composition(formula)
    if len(composition.elements) != 1:
        raise ValueError(
            f"ASE 兜底当前只自动构建单元素块体，收到 {formula}；"
            "多元素/氧化物请提供 --structure-path 或 --mp-id。"
        )
    symbol = composition.elements[0].symbol
    Element(symbol)  # validate
    atoms = bulk(symbol)
    structure = AseAtomsAdaptor.get_structure(atoms)
    return structure, {
        "input_format": "ase_bulk",
        "element": symbol,
        "ase_formula": atoms.get_chemical_formula(),
        "note": "ASE reference bulk fallback; verify lattice parameter before production DFT.",
    }


def _load_bulk_from_mp(mp_id: str | None, material: str | None, api_key: str | None = None) -> tuple[Any, dict[str, Any]]:
    resolved_key = api_key or resolve_api_key(
        Path.cwd(),
        aliases=("materials_project", "mp"),
        env_names=("MP_API_KEY", "MATERIALS_PROJECT_API_KEY"),
    )
    if not resolved_key:
        raise RuntimeError("未找到 Materials Project API key；请设置 MP_API_KEY/MATERIALS_PROJECT_API_KEY 或改用本地结构/ASE 兜底。")
    try:
        from mp_api.client import MPRester
    except Exception as exc:  # pragma: no cover - dependency is declared, but keep honest errors.
        raise RuntimeError(f"mp-api 不可用，无法调用 Materials Project: {exc}") from exc

    with MPRester(resolved_key) as rester:
        if mp_id:
            structure = rester.get_structure_by_material_id(mp_id)
            return structure, {"input_format": "materials_project", "mp_id": mp_id}

        formula = _clean_material_formula(material)
        if not formula:
            raise ValueError("使用 Materials Project 自动搜索时必须提供 --material 公式或 --mp-id。")
        try:
            docs = rester.materials.summary.search(
                formula=formula,
                fields=["material_id", "energy_above_hull"],
            )
        except Exception as exc:
            raise RuntimeError(f"Materials Project 公式搜索失败；建议改用明确 --mp-id。错误: {exc}") from exc
        if not docs:
            raise RuntimeError(f"Materials Project 未找到公式 {formula} 的候选结构；请改用 --mp-id 或本地结构。")
        docs = sorted(docs, key=lambda item: float(getattr(item, "energy_above_hull", 9999) or 0.0))
        selected = docs[0]
        selected_mp_id = str(selected.material_id)
        structure = rester.get_structure_by_material_id(selected_mp_id)
        return structure, {
            "input_format": "materials_project",
            "mp_id": selected_mp_id,
            "formula_query": formula,
            "selection_rule": "lowest energy_above_hull from summary.search",
        }


def _load_bulk_structure(
    *,
    material: str,
    structure_path: str | None,
    mp_id: str | None,
    source: str,
    mp_api_key: str | None,
) -> tuple[Any, str, str | None, dict[str, Any]]:
    normalized_source = source.lower()
    if structure_path:
        from dft_app.builder.structure_resolver import StructureResolver
        from dft_app.models import StructureSource, TaskType
        from dft_app.models.experiment_spec import ExperimentSpec

        spec = ExperimentSpec(
            task_id=f"slab_{uuid4().hex[:8]}",
            task_type=TaskType.RELAX,
            material_name=material,
            source_prompt=f"Build slab for {material}",
            structure_source=StructureSource.LOCAL_FILE,
            structure_path=structure_path,
        )
        resolved = StructureResolver().resolve(spec)
        if resolved.structure is None:
            raise RuntimeError(resolved.message)
        return resolved.structure, "local", structure_path, resolved.metadata or {}

    if normalized_source in {"mp", "materials_project"} or mp_id:
        structure, metadata = _load_bulk_from_mp(mp_id, material, api_key=mp_api_key)
        return structure, "materials_project", metadata.get("mp_id") or mp_id, metadata

    if normalized_source in {"auto", "ase", "element", "builtin"}:
        structure, metadata = _load_bulk_from_ase(material)
        return structure, "ase", metadata.get("element"), metadata

    raise ValueError(f"未知结构来源 source={source!r}；可选 auto/local/mp/ase。")


def _parse_triplet(values: list[int] | tuple[int, int, int] | None, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    if values is None:
        return fallback
    if len(values) != 3:
        raise ValueError("miller/supercell 必须是三个整数。")
    return tuple(int(item) for item in values)  # type: ignore[return-value]


def build_adsorption_slab(
    *,
    material: str,
    output_dir: str,
    structure_path: str | None = None,
    mp_id: str | None = None,
    source: str = "auto",
    miller_index: tuple[int, int, int] | list[int] | None = None,
    supercell: tuple[int, int, int] | list[int] = (2, 2, 1),
    min_slab_size: float = 8.0,
    min_vacuum_size: float = 12.0,
    fixed_bottom_layers: int = 2,
    center_slab: bool = True,
    mp_api_key: str | None = None,
) -> SlabBuildResult:
    """Build the first slab POSCAR for an adsorption task.

    Source order is intentionally broad:
    - local structure file if ``structure_path`` is supplied;
    - Materials Project if ``source=mp``/``mp_id`` and an API key exists;
    - ASE elemental bulk fallback for names such as ``Pt`` or ``Pt(111)``.
    """
    if not material.strip():
        raise ValueError("material 不能为空，例如 Pt(111)、Pt 或 Al2O3。")
    inferred_miller = _infer_miller_from_material(material) or (1, 1, 1)
    resolved_miller = _parse_triplet(miller_index, inferred_miller)
    resolved_supercell = _parse_triplet(supercell, (2, 2, 1))
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    bulk_structure, resolved_source, source_detail, source_metadata = _load_bulk_structure(
        material=material,
        structure_path=structure_path,
        mp_id=mp_id,
        source=source,
        mp_api_key=mp_api_key,
    )

    from dft_app.modeling.adsorption_candidate_generator import apply_selective_dynamics
    from pymatgen.core.surface import SlabGenerator
    from pymatgen.io.vasp import Poscar

    generator = SlabGenerator(
        bulk_structure,
        resolved_miller,
        min_slab_size=min_slab_size,
        min_vacuum_size=min_vacuum_size,
        center_slab=center_slab,
    )
    slabs = generator.get_slabs(symmetrize=False)
    if not slabs:
        raise RuntimeError(f"无法为 {material} 的 {resolved_miller} 面生成 slab；请换 miller 或提供手工 slab。")
    slab = slabs[0].get_orthogonal_c_slab()
    if resolved_supercell != (1, 1, 1):
        slab.make_supercell(resolved_supercell)
    slab = apply_selective_dynamics(slab, slab_atom_count=len(slab), fixed_bottom_layers=fixed_bottom_layers)

    slab_path = output_root / "POSCAR"
    Poscar(slab).write_file(slab_path)
    metadata = {
        "status": "ok",
        "material": material,
        "source": resolved_source,
        "source_detail": source_detail,
        "source_metadata": source_metadata,
        "miller_index": list(resolved_miller),
        "supercell": list(resolved_supercell),
        "min_slab_size": min_slab_size,
        "min_vacuum_size": min_vacuum_size,
        "fixed_bottom_layers": fixed_bottom_layers,
        "center_slab": center_slab,
        "slab_path": str(slab_path),
        "atom_count": len(slab),
        "formula": slab.composition.reduced_formula,
        "created_at": _now(),
    }
    metadata_path = output_root / "slab_build.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    next_tasks = [
        f"检查 `{slab_path}` 的层数、真空层和固定层；ASE/MP 自动结构进入生产计算前应确认晶格常数。",
        "运行 adsorption candidates 生成吸附位点/取向候选。",
        "选择候选后进入 clean slab / isolated adsorbate / adsorbed system 三子任务吸附能流程。",
    ]
    return SlabBuildResult(
        status="ok",
        material=material,
        source=resolved_source,
        source_detail=source_detail,
        miller_index=resolved_miller,
        slab_path=str(slab_path),
        output_dir=str(output_root),
        metadata_path=str(metadata_path),
        atom_count=len(slab),
        formula=slab.composition.reduced_formula,
        next_research_tasks=next_tasks,
        metadata=metadata,
    )


def plan_adsorption_task(
    prompt: str,
    *,
    project: str | None = None,
    adsorbate: str | None = None,
    material: str | None = None,
    slab_path: str | None = None,
    preferred_site: str | None = None,
    preferred_orientation: str | None = None,
    persist: bool = True,
) -> AdsorptionPlan:
    normalized_prompt = prompt.strip()
    if not normalized_prompt:
        raise ValueError("吸附任务 prompt 不能为空")
    resolved_adsorbate = adsorbate or _infer_adsorbate(normalized_prompt)
    missing: list[str] = []
    if not resolved_adsorbate:
        missing.append("adsorbate")
    if not slab_path:
        missing.append("slab_path")
    if not material:
        missing.append("material")
    readiness = "ready_for_candidates" if not missing else "needs_confirmation"
    task_envelope = create_task_plan(
        normalized_prompt,
        project=project,
        material=material,
        structure_path=slab_path,
        task_type="relax_scf",
        planner_mode="rule",
        persist=persist,
    )
    dft_entrypoints = {
        "build_slab": (
            "aether-dft adsorption build-slab "
            "--material <MATERIAL_OR_SURFACE> --output-dir <SLAB_DIR> "
            "[--source ase|mp|auto] [--structure-path bulk.cif] [--mp-id mp-...]"
        ),
        "candidate_generation": (
            "aether-dft adsorption candidates "
            "--slab-path <POSCAR> --adsorbate <ADSORBATE> --material <MATERIAL>"
        ),
        "select_candidate": "aether-dft dft adsorption-select --manifest <candidate_manifest.json> --candidate-id <ID> --material <MATERIAL> --prompt <PROMPT>",
        "workflow": "aether-dft dft adsorption-workflow --run-root <RUN_ROOT> --status",
    }
    next_tasks = recommend_adsorption_next_steps(
        readiness=readiness,
        missing_inputs=missing,
        adsorbate=resolved_adsorbate,
        material=material,
        slab_path=slab_path,
        preferred_site=preferred_site,
        preferred_orientation=preferred_orientation,
    )
    plan = AdsorptionPlan(
        task_id=task_envelope.task_id,
        prompt=normalized_prompt,
        project=project,
        adsorbate=resolved_adsorbate,
        material=material,
        slab_path=slab_path,
        readiness=readiness,
        missing_inputs=missing,
        next_research_tasks=next_tasks,
        dft_entrypoints=dft_entrypoints,
        task_record=task_envelope.to_dict(),
        created_at=_now(),
    )
    if persist and project:
        append_progress(
            project,
            completed=[f"已建立吸附任务 `{task_envelope.task_id}`，readiness={readiness}。"],
            blockers=[f"缺少 {item}" for item in missing],
            next_steps=next_tasks[:3],
        )
    return plan


def recommend_adsorption_next_steps(
    *,
    readiness: str,
    missing_inputs: list[str],
    adsorbate: str | None,
    material: str | None,
    slab_path: str | None,
    preferred_site: str | None = None,
    preferred_orientation: str | None = None,
) -> list[str]:
    steps: list[str] = []
    if "slab_path" in missing_inputs:
        build_hint = "先运行 adsorption build-slab 从本地结构、MP（可选）或 ASE 元素兜底生成 slab POSCAR。"
        if material and _clean_material_formula(material):
            build_hint = (
                f"先运行 `aether-dft adsorption build-slab --material \"{material}\" --output-dir <SLAB_DIR>` "
                "生成 slab POSCAR；也可传 --structure-path 或 --mp-id。"
            )
        steps.append(build_hint)
    if "adsorbate" in missing_inputs:
        steps.append("明确吸附物种，例如 H2O、CO、OH 或分子文件路径。")
    if "material" in missing_inputs:
        steps.append("给出材料/表面标签，例如 Pt(111)、Ru/Al2O3 或自定义 slab 名称。")
    if readiness == "ready_for_candidates":
        site = preferred_site or "ontop/bridge/hollow 多位点"
        orientation = preferred_orientation or "upright/flat/tilted 多取向"
        steps.extend(
            [
                f"生成 {adsorbate} 在 {material} slab 上的吸附候选，覆盖 {site} 与 {orientation}。",
                "选择 top-ranked candidate 后运行 adsorption-select，进入 clean slab / isolated adsorbate / adsorbed system 三子任务工作流。",
                "完成 VASP 后运行 adsorption-workflow --parse-analyze 汇总 E_ads，并把结果写入项目知识库。",
            ]
        )
    steps.append("如果结果分散，下一轮推荐做位点/覆盖度/取向敏感性对比。")
    return steps


def generate_adsorption_candidates(
    *,
    slab_path: str,
    adsorbate: str,
    material: str,
    prompt: str,
    project: str | None = None,
    output_dir: str | None = None,
    task_id: str | None = None,
    candidate_height: float = 2.1,
    max_sites_per_family: int = 2,
    preferred_site: str | None = None,
    preferred_orientation: str | None = None,
    vacancy_species: str | None = None,
) -> dict[str, Any]:
    if not Path(slab_path).exists():
        raise FileNotFoundError(f"slab_path 不存在: {slab_path}")
    resolved_task_id = task_id or f"ads_{uuid4().hex[:8]}"
    if output_dir is None:
        if project:
            output_root = project_paths(project).runs / "adsorption_candidates" / resolved_task_id
        else:
            output_root = Path("runtime") / "adsorption_candidates" / resolved_task_id
    else:
        output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    from dft_app.cli.main import main as dft_main

    args = [
        "adsorption-candidates",
        "--slab-path",
        slab_path,
        "--adsorbate",
        adsorbate,
        "--material",
        material,
        "--prompt",
        prompt,
        "--output-dir",
        str(output_root),
        "--task-id",
        resolved_task_id,
        "--candidate-height",
        str(candidate_height),
        "--max-sites-per-family",
        str(max_sites_per_family),
    ]
    if preferred_site:
        args.extend(["--preferred-site", preferred_site])
    if preferred_orientation:
        args.extend(["--preferred-orientation", preferred_orientation])
    if vacancy_species:
        args.extend(["--vacancy-species", vacancy_species])

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        exit_code = dft_main(args)
    stdout = buffer.getvalue().strip()
    parsed: dict[str, Any] | None = None
    if stdout:
        try:
            parsed = json.loads(stdout)
        except json.JSONDecodeError:
            parsed = None
    result = {
        "status": "ok" if exit_code == 0 else "failed",
        "exit_code": exit_code,
        "task_id": resolved_task_id,
        "output_dir": str(output_root),
        "stdout": stdout,
        "result": parsed,
        "next_research_tasks": recommend_adsorption_next_steps(
            readiness="ready_for_candidates",
            missing_inputs=[],
            adsorbate=adsorbate,
            material=material,
            slab_path=slab_path,
            preferred_site=preferred_site,
            preferred_orientation=preferred_orientation,
        ),
    }
    if project:
        append_progress(
            project,
            completed=[f"已为吸附任务 `{resolved_task_id}` 生成候选构型，退出码 {exit_code}。"],
            blockers=[] if exit_code == 0 else ["候选构型生成失败，需查看 stdout。"],
            next_steps=result["next_research_tasks"][:3],
        )
    return result


def run_adsorption_pipeline(
    *,
    material: str,
    adsorbate: str,
    output_dir: str,
    prompt: str | None = None,
    project: str | None = None,
    structure_path: str | None = None,
    mp_id: str | None = None,
    source: str = "auto",
    miller_index: tuple[int, int, int] | list[int] | None = None,
    supercell: tuple[int, int, int] | list[int] = (2, 2, 1),
    candidate_height: float = 2.1,
    max_sites_per_family: int = 2,
    preferred_site: str | None = None,
    preferred_orientation: str | None = None,
    vacancy_species: str | None = None,
) -> dict[str, Any]:
    output_root = Path(output_dir)
    slab_result = build_adsorption_slab(
        material=material,
        output_dir=str(output_root / "slab"),
        structure_path=structure_path,
        mp_id=mp_id,
        source=source,
        miller_index=miller_index,
        supercell=supercell,
    )
    run_prompt = prompt or f"{adsorbate} adsorption on {material}"
    candidate_result = generate_adsorption_candidates(
        slab_path=slab_result.slab_path,
        adsorbate=adsorbate,
        material=material,
        prompt=run_prompt,
        project=project,
        output_dir=str(output_root / "candidates"),
        candidate_height=candidate_height,
        max_sites_per_family=max_sites_per_family,
        preferred_site=preferred_site,
        preferred_orientation=preferred_orientation,
        vacancy_species=vacancy_species,
    )
    payload = {
        "status": "ok" if candidate_result.get("status") == "ok" else "failed",
        "slab": slab_result.to_dict(),
        "candidates": candidate_result,
        "next_research_tasks": [
            *slab_result.next_research_tasks[:2],
            *candidate_result.get("next_research_tasks", [])[:3],
        ],
    }
    (output_root / "adsorption_pipeline.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def run_adsorption_full_workflow(
    *,
    material: str,
    adsorbate: str,
    output_dir: str,
    prompt: str | None = None,
    project: str | None = None,
    structure_path: str | None = None,
    mp_id: str | None = None,
    source: str = "auto",
    miller_index: tuple[int, int, int] | list[int] | None = None,
    supercell: tuple[int, int, int] | list[int] = (2, 2, 1),
    candidate_id: str | None = None,
    submit_profile: str | None = None,
    candidate_height: float = 2.1,
    max_sites_per_family: int = 2,
    preferred_site: str | None = None,
    preferred_orientation: str | None = None,
    vacancy_species: str | None = None,
) -> dict[str, Any]:
    """Run the adsorption workflow through VASP workspace materialization.

    This intentionally stops before claiming DFT energies: it prepares the three
    real VASP input workspaces and returns workflow status. Submission, monitor,
    fetch and parse/analyze remain separate external-compute steps.
    """
    output_root = Path(output_dir)
    run_prompt = prompt or f"计算 {adsorbate} 在 {material} 上的吸附能"
    slab_result = build_adsorption_slab(
        material=material,
        output_dir=str(output_root / "slab"),
        structure_path=structure_path,
        mp_id=mp_id,
        source=source,
        miller_index=miller_index,
        supercell=supercell,
    )
    candidate_result = generate_adsorption_candidates(
        slab_path=slab_result.slab_path,
        adsorbate=adsorbate,
        material=material,
        prompt=run_prompt,
        project=project,
        output_dir=str(output_root / "candidates"),
        candidate_height=candidate_height,
        max_sites_per_family=max_sites_per_family,
        preferred_site=preferred_site,
        preferred_orientation=preferred_orientation,
        vacancy_species=vacancy_species,
    )
    parsed_candidates = candidate_result.get("result") or {}
    if candidate_result.get("status") != "ok" or parsed_candidates.get("status") != "generated":
        return {
            "status": "failed",
            "reason": "候选构型生成失败，无法继续选择和生成 workflow bundle。",
            "slab": slab_result.to_dict(),
            "candidates": candidate_result,
        }
    selected_id = candidate_id
    if not selected_id:
        top_candidates = parsed_candidates.get("top_candidates") or []
        if not top_candidates:
            return {
                "status": "failed",
                "reason": "候选构型清单为空，无法自动选择 candidate。",
                "slab": slab_result.to_dict(),
                "candidates": candidate_result,
            }
        selected_id = str(top_candidates[0]["candidate_id"])

    from dft_app.cli.main import (
        STORE,
        collect_adsorption_workflow_status,
        create_complex_run_record,
        create_plan_result,
        maybe_materialize_adsorption_workflow_bundle,
        maybe_select_adsorption_candidate,
    )

    workflow_args = SimpleNamespace(
        prompt=run_prompt,
        task_type=None,
        material=material,
        structure_path=slab_result.slab_path,
        submit_profile=submit_profile,
        dry_run=False,
        submit=False,
        remote=False,
        selected_candidate_id=selected_id,
    )
    planning_result = create_plan_result(workflow_args)
    if planning_result.spec is not None:
        return {
            "status": "failed",
            "reason": "planner 未将该任务识别为 complex adsorption workflow；请检查 prompt 是否为吸附能任务。",
            "slab": slab_result.to_dict(),
            "candidates": candidate_result,
            "plan": planning_result.plan.to_dict(),
            "spec": planning_result.spec.to_dict(),
        }
    run_record = create_complex_run_record(planning_result.plan)
    selection_result = maybe_select_adsorption_candidate(
        args=workflow_args,
        plan=planning_result.plan,
        run_record=run_record,
        adsorption_candidate_result=parsed_candidates,
    )
    workflow_bundle = maybe_materialize_adsorption_workflow_bundle(
        args=workflow_args,
        plan=planning_result.plan,
        run_record=run_record,
        selected_candidate_result=selection_result,
    )
    STORE.save_run_record(run_record)
    workflow_status = None
    if workflow_bundle and workflow_bundle.get("status") == "prepared":
        workflow_status = collect_adsorption_workflow_status(Path(run_record.run_root))

    payload = {
        "status": "prepared" if workflow_status and workflow_status.get("status") == "prepared" else "failed",
        "slab": slab_result.to_dict(),
        "candidates": {
            **candidate_result,
            "stdout": "<captured>",
        },
        "selected_candidate_id": selected_id,
        "selection": selection_result,
        "workflow_bundle": workflow_bundle,
        "workflow_status": workflow_status,
        "run_root": run_record.run_root,
        "next_commands": {
            "status": f"aether-dft dft adsorption-workflow --run-root {run_record.run_root} --status",
            "submit": f"aether-dft dft adsorption-workflow --run-root {run_record.run_root} --submit",
            "parse_after_outputs": f"aether-dft dft adsorption-workflow --run-root {run_record.run_root} --parse-analyze",
        },
        "honest_boundary": "已真实生成三体系 VASP 输入工作区；尚未运行 VASP/SLURM，因此没有真实吸附能数值。",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "adsorption_full_workflow.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload
