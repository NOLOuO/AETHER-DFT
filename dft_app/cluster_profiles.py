from __future__ import annotations

from dataclasses import dataclass, field

from dft_app.models.experiment_spec import ExperimentSpec


@dataclass(frozen=True)
class SubmitProfile:
    name: str
    partition: str
    nodes: int
    ntasks_per_node: int
    walltime: str
    memory_per_cpu: str
    vasp_variant: str = "std"
    oneapi_source: str = "source /share/intel/2024/oneapi/setvars.sh --intel64"
    module_load_commands: list[str] = field(
        default_factory=lambda: ["module load vasp/6.4.2_cpu_dftd4 gcc/12.1.0"]
    )
    run_command_template: str = "mpirun vasp_{vasp_variant} > vasp.out"


SUBMIT_PROFILES: dict[str, SubmitProfile] = {
    "c32": SubmitProfile(
        name="c32",
        partition="c-node",
        nodes=1,
        ntasks_per_node=32,
        walltime="24:00:00",
        memory_per_cpu="3000",
    ),
    "b96": SubmitProfile(
        name="b96",
        partition="b-node",
        nodes=1,
        ntasks_per_node=96,
        walltime="24:00:00",
        memory_per_cpu="3000",
    ),
}


def resolve_submit_profile(spec: ExperimentSpec) -> SubmitProfile:
    profile_name = spec.submit_profile or "c32"
    if profile_name not in SUBMIT_PROFILES:
        supported = ", ".join(sorted(SUBMIT_PROFILES))
        raise ValueError(f"不支持的 submit_profile: {profile_name}。当前支持: {supported}")
    return SUBMIT_PROFILES[profile_name]


def infer_submit_profile_from_prompt(prompt: str) -> str | None:
    prompt_lower = prompt.lower()
    if "b-node" in prompt_lower or "96核" in prompt_lower or "96 核" in prompt_lower:
        return "b96"
    if "c-node" in prompt_lower or "32核" in prompt_lower or "32 核" in prompt_lower:
        return "c32"
    return None
