"""通用计算化学计算器：Boltzmann 分布 / Gibbs / TST 速率 / 单位换算。

设计原则：
- 纯函数，无副作用，无外部依赖。
- 拒绝荒谬输入（负温度、空能量列表、零或负的 prefactor），但给出清晰的错误信息。
- 每个 mode 返回 ``status`` + ``result`` + ``guidance``，让模型直接拿来回答用户。
"""

from __future__ import annotations

import math
from typing import Any

# 物理常数（CODATA 2018）
KB_EV_PER_K = 8.617333262e-5            # Boltzmann，eV/K
KB_J_PER_K = 1.380649e-23               # Boltzmann，J/K
H_EV_S = 4.135667696e-15                # Planck，eV·s
H_J_S = 6.62607015e-34                  # Planck，J·s
NA = 6.02214076e23                      # Avogadro
EV_TO_J = 1.602176634e-19
EV_TO_KJ_PER_MOL = 96.48533212331
EV_TO_KCAL_PER_MOL = 23.06054783
EV_TO_HARTREE = 1.0 / 27.211386245988

# 反向因子由正向倒数生成，便于矩阵化
_UNIT_TO_EV: dict[str, float] = {
    "ev": 1.0,
    "kj/mol": 1.0 / EV_TO_KJ_PER_MOL,
    "kj_per_mol": 1.0 / EV_TO_KJ_PER_MOL,
    "kcal/mol": 1.0 / EV_TO_KCAL_PER_MOL,
    "kcal_per_mol": 1.0 / EV_TO_KCAL_PER_MOL,
    "hartree": 1.0 / EV_TO_HARTREE,
    "j": 1.0 / EV_TO_J,
    "joule": 1.0 / EV_TO_J,
}


def _normalize_unit(unit: str) -> str:
    return str(unit or "").strip().lower().replace(" ", "")


def _eV_to_unit(value_ev: float, unit: str) -> float:
    norm = _normalize_unit(unit)
    if norm not in _UNIT_TO_EV:
        raise ValueError(f"不支持的单位: {unit!r}；支持 {sorted(_UNIT_TO_EV)}")
    return value_ev / _UNIT_TO_EV[norm]


def _unit_to_eV(value: float, unit: str) -> float:
    norm = _normalize_unit(unit)
    if norm not in _UNIT_TO_EV:
        raise ValueError(f"不支持的单位: {unit!r}；支持 {sorted(_UNIT_TO_EV)}")
    return value * _UNIT_TO_EV[norm]


def convert(value: float, *, from_unit: str, to_unit: str) -> dict[str, Any]:
    """能量单位换算。"""
    value_ev = _unit_to_eV(float(value), from_unit)
    converted = _eV_to_unit(value_ev, to_unit)
    return {
        "status": "ok",
        "mode": "convert",
        "result": converted,
        "input": {"value": value, "from_unit": from_unit, "to_unit": to_unit},
        "value_ev": value_ev,
        "guidance": f"{value} {from_unit} = {converted:.6g} {to_unit}",
    }


def boltzmann_populations(
    energies: list[float],
    *,
    temperature_k: float,
    energy_unit: str = "eV",
    reference_energy: float | None = None,
) -> dict[str, Any]:
    """给定一组能量与温度，返回 Boltzmann 占据率。

    ``reference_energy`` 默认取最小值，避免溢出。返回归一化后的概率分布。
    """
    if not energies:
        raise ValueError("energies 不能为空。")
    if temperature_k <= 0:
        raise ValueError(f"temperature_k 必须 > 0，收到 {temperature_k}。")
    energies_ev = [_unit_to_eV(float(e), energy_unit) for e in energies]
    ref = float(reference_energy) if reference_energy is not None else min(energies_ev)
    kBT = KB_EV_PER_K * float(temperature_k)
    factors = [math.exp(-(e - ref) / kBT) for e in energies_ev]
    total = sum(factors)
    if total <= 0:
        raise ValueError("Boltzmann 分布权重和为 0；输入是否合理？")
    populations = [f / total for f in factors]
    most_populated = max(range(len(populations)), key=lambda i: populations[i])
    return {
        "status": "ok",
        "mode": "boltzmann",
        "result": populations,
        "input": {
            "energies": energies,
            "energy_unit": energy_unit,
            "temperature_k": temperature_k,
            "reference_energy_ev": ref,
        },
        "kBT_ev": kBT,
        "most_populated_index": most_populated,
        "most_populated_fraction": populations[most_populated],
        "guidance": (
            f"在 {temperature_k} K 下，最稳态（index {most_populated}）占 {populations[most_populated]*100:.2f}%；"
            "差距 ≥ 5kBT 时通常可以只算最稳态。"
        ),
    }


def gibbs_free_energy(
    *,
    enthalpy: float,
    entropy: float,
    temperature_k: float,
    enthalpy_unit: str = "eV",
    entropy_unit: str = "eV/K",
) -> dict[str, Any]:
    """G = H - T·S；H 默认 eV，S 默认 eV/K。

    支持 entropy_unit ∈ {"eV/K", "J/(mol·K)", "kJ/(mol·K)", "kcal/(mol·K)"}。
    """
    if temperature_k <= 0:
        raise ValueError(f"temperature_k 必须 > 0，收到 {temperature_k}。")
    h_ev = _unit_to_eV(float(enthalpy), enthalpy_unit)
    norm_s = _normalize_unit(entropy_unit).replace("(", "").replace(")", "").replace("·", "")
    if norm_s in {"ev/k"}:
        s_ev_per_k = float(entropy)
    elif norm_s in {"j/molk", "j/mol/k"}:
        # J/(mol·K) → eV/K：除以 NA 转每分子，再 J→eV
        s_ev_per_k = float(entropy) / NA / EV_TO_J
    elif norm_s in {"kj/molk", "kj/mol/k"}:
        s_ev_per_k = float(entropy) * 1000.0 / NA / EV_TO_J
    elif norm_s in {"kcal/molk", "kcal/mol/k"}:
        s_ev_per_k = float(entropy) * 4184.0 / NA / EV_TO_J
    else:
        raise ValueError(
            f"不支持的 entropy_unit: {entropy_unit!r}；支持 eV/K / J/(mol·K) / kJ/(mol·K) / kcal/(mol·K)"
        )
    g_ev = h_ev - float(temperature_k) * s_ev_per_k
    return {
        "status": "ok",
        "mode": "gibbs",
        "result": g_ev,
        "input": {
            "enthalpy": enthalpy,
            "enthalpy_unit": enthalpy_unit,
            "entropy": entropy,
            "entropy_unit": entropy_unit,
            "temperature_k": temperature_k,
        },
        "enthalpy_ev": h_ev,
        "entropy_ev_per_k": s_ev_per_k,
        "ts_correction_ev": float(temperature_k) * s_ev_per_k,
        "guidance": (
            f"G = H - TS = {h_ev:.4f} - {temperature_k}·{s_ev_per_k:.6e} = {g_ev:.4f} eV"
        ),
    }


def transition_state_rate(
    *,
    activation_energy: float,
    temperature_k: float,
    energy_unit: str = "eV",
    prefactor_hz: float | None = None,
    transmission_coefficient: float = 1.0,
) -> dict[str, Any]:
    """k = κ · A · exp(-Ea / kBT)；A 默认取 kBT/h（Eyring 形式）。

    返回 rate (1/s)、半衰期、Arrhenius 形式与 Eyring 形式对比。
    """
    if temperature_k <= 0:
        raise ValueError(f"temperature_k 必须 > 0，收到 {temperature_k}。")
    if transmission_coefficient <= 0:
        raise ValueError("transmission_coefficient 必须 > 0。")
    ea_ev = _unit_to_eV(float(activation_energy), energy_unit)
    if ea_ev < 0:
        raise ValueError("activation_energy 必须 ≥ 0（TST 假设鞍点高于反应物）。")
    kBT_ev = KB_EV_PER_K * float(temperature_k)
    eyring_prefactor = kBT_ev / H_EV_S  # 1/s
    A = float(prefactor_hz) if prefactor_hz is not None else eyring_prefactor
    if A <= 0:
        raise ValueError("prefactor_hz 必须 > 0。")
    exponent = -ea_ev / kBT_ev
    rate = transmission_coefficient * A * math.exp(exponent)
    half_life_s = math.log(2) / rate if rate > 0 else math.inf
    return {
        "status": "ok",
        "mode": "tst_rate",
        "result": rate,
        "input": {
            "activation_energy": activation_energy,
            "energy_unit": energy_unit,
            "temperature_k": temperature_k,
            "prefactor_hz": prefactor_hz,
            "transmission_coefficient": transmission_coefficient,
        },
        "activation_energy_ev": ea_ev,
        "kBT_ev": kBT_ev,
        "eyring_prefactor_hz": eyring_prefactor,
        "effective_prefactor_hz": A,
        "exponent": exponent,
        "half_life_s": half_life_s,
        "guidance": (
            f"k = {transmission_coefficient}·{A:.3e}·exp(-{ea_ev:.3f}/{kBT_ev:.4f}) = {rate:.3e} /s；"
            f"半衰期 ≈ {half_life_s:.3e} s。"
            "kBT/h 在 300 K 下 ≈ 6.25e12 /s 是经典 Eyring 上限。"
        ),
    }


def kBT(temperature_k: float, *, unit: str = "eV") -> dict[str, Any]:
    """快捷查 kBT 在指定单位的数值，方便模型判断"差距 ≥ 5kBT"等问题。"""
    if temperature_k <= 0:
        raise ValueError(f"temperature_k 必须 > 0，收到 {temperature_k}。")
    val_ev = KB_EV_PER_K * float(temperature_k)
    out = _eV_to_unit(val_ev, unit)
    return {
        "status": "ok",
        "mode": "kBT",
        "result": out,
        "input": {"temperature_k": temperature_k, "unit": unit},
        "kBT_ev": val_ev,
        "guidance": f"kBT @ {temperature_k} K = {val_ev:.6f} eV = {out:.6g} {unit}",
    }


def compute(mode: str, **params: Any) -> dict[str, Any]:
    """模式分发入口。

    支持的 mode：
    - "convert": value, from_unit, to_unit
    - "boltzmann": energies, temperature_k, energy_unit?, reference_energy?
    - "gibbs": enthalpy, entropy, temperature_k, enthalpy_unit?, entropy_unit?
    - "tst_rate": activation_energy, temperature_k, energy_unit?, prefactor_hz?, transmission_coefficient?
    - "kBT": temperature_k, unit?
    """
    mode = str(mode or "").strip().lower()
    try:
        if mode == "convert":
            return convert(
                float(params["value"]),
                from_unit=str(params["from_unit"]),
                to_unit=str(params["to_unit"]),
            )
        if mode == "boltzmann":
            return boltzmann_populations(
                [float(x) for x in params["energies"]],
                temperature_k=float(params["temperature_k"]),
                energy_unit=str(params.get("energy_unit") or "eV"),
                reference_energy=(float(params["reference_energy"]) if params.get("reference_energy") is not None else None),
            )
        if mode == "gibbs":
            return gibbs_free_energy(
                enthalpy=float(params["enthalpy"]),
                entropy=float(params["entropy"]),
                temperature_k=float(params["temperature_k"]),
                enthalpy_unit=str(params.get("enthalpy_unit") or "eV"),
                entropy_unit=str(params.get("entropy_unit") or "eV/K"),
            )
        if mode == "tst_rate":
            return transition_state_rate(
                activation_energy=float(params["activation_energy"]),
                temperature_k=float(params["temperature_k"]),
                energy_unit=str(params.get("energy_unit") or "eV"),
                prefactor_hz=(float(params["prefactor_hz"]) if params.get("prefactor_hz") is not None else None),
                transmission_coefficient=float(params.get("transmission_coefficient") or 1.0),
            )
        if mode == "kBT" or mode == "kbt":
            return kBT(
                float(params["temperature_k"]),
                unit=str(params.get("unit") or "eV"),
            )
    except KeyError as exc:
        return {"status": "error", "mode": mode, "message": f"缺少参数 {exc}"}
    except ValueError as exc:
        return {"status": "error", "mode": mode, "message": str(exc)}
    return {
        "status": "error",
        "mode": mode,
        "message": f"未知 mode: {mode!r}；支持 convert / boltzmann / gibbs / tst_rate / kBT。",
    }
