#!/usr/bin/env python3
"""
Mean-field microkinetic model for MCH <-> toluene on Pt / modified Pt.

Design target: a compact, DFT-energy-table driven MKM that follows the two local
MCH dehydrogenation papers closely enough for first production comparisons:
- mean-field ODE: dtheta_i/dt = sum_j nu_ij r_j(theta)
- unactivated adsorption from kinetic gas theory
- surface reaction rate constants from TST
- adsorption/desorption thermodynamic consistency
- multi-site occupancy separated from the C7 carbon-pool coverage cap
- temperature scan, reaction orders, apparent Ea and DRC diagnostics

Input energies in inputs/example_network.csv are placeholders. Replace them with DFT
free-energy barriers / reaction free energies before interpreting the rates.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.integrate import solve_ivp

KB_EV = 8.617333262145e-5  # eV/K
KB_SI = 1.380649e-23       # J/K
H_SI = 6.62607015e-34      # J s
AMU_KG = 1.66053906660e-27
BAR_PA = 1.0e5
ANG2_M2 = 1.0e-20
P_STD_BAR = 1.0
EPS = 1.0e-30
RATE_FLOOR = 1.0e-20

TERM_RE = re.compile(r"^\s*(?:(\d+(?:\.\d+)?)\s*)?(.+?)\s*$")


@dataclass(frozen=True)
class Step:
    step_id: str
    kind: str
    reactants: Dict[str, float]
    products: Dict[str, float]
    gaf_eV: float | None
    gar_eV: float | None
    dg_eV: float | None
    sticking: float
    mass_amu: float | None
    site_area_A2: float
    tof_label: str
    notes: str


@dataclass(frozen=True)
class Species:
    name: str
    phase: str
    site_size: float
    molar_mass_amu: float | None
    carbon_pool: bool


@dataclass(frozen=True)
class ModelSettings:
    T: float
    pressures: Dict[str, float]
    t_final: float
    steady_tol: float
    max_extend: int
    carbon_pool_cap: float | None
    adsorption_dg_state: str
    carbon_pool_exponent: float


@dataclass(frozen=True)
class Solution:
    T: float
    pressures: Dict[str, float]
    y: np.ndarray
    residual: float
    t_end: float
    converged: bool
    idx: Dict[str, int]
    site_sizes: np.ndarray
    carbon_pool_mask: np.ndarray
    k_pairs: List[Tuple[float, float]]
    rate_rows: List[Tuple[Step, float, float, float, float, float]]
    theta_vacant: float
    theta_carbon_pool: float
    carbon_pool_factor: float
    observables: Dict[str, float]


def parse_bool(text: str | None, default: bool = False) -> bool:
    if text is None or str(text).strip() == "":
        return default
    return str(text).strip().lower() in {"1", "true", "yes", "y", "on"}


def is_gas(name: str) -> bool:
    return name.endswith("_g") or name.endswith("(g)")


def is_vacant(name: str) -> bool:
    return name == "*"


def is_surface(name: str) -> bool:
    return ("*" in name) and not is_vacant(name) and not is_gas(name)


def infer_carbon_pool(name: str) -> bool:
    return is_surface(name) and name.startswith("C7")


def parse_float(value: str | None, default: float | None = None) -> float | None:
    if value is None:
        return default
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "na", "-"}:
        return default
    return float(text)


def parse_side(side: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    side = side.strip()
    if not side or side == "0":
        return out
    for raw in side.split("+"):
        token = raw.strip()
        if not token:
            continue
        m = TERM_RE.match(token)
        if not m:
            raise ValueError(f"Cannot parse stoichiometric term: {token!r}")
        coeff = float(m.group(1)) if m.group(1) else 1.0
        species = m.group(2).strip()
        if not species:
            raise ValueError(f"Missing species in term: {token!r}")
        out[species] = out.get(species, 0.0) + coeff
    return out


def parse_equation(equation: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    if "<->" in equation:
        left, right = equation.split("<->", 1)
    elif "->" in equation:
        left, right = equation.split("->", 1)
    elif "=" in equation:
        left, right = equation.split("=", 1)
    else:
        raise ValueError(f"Equation must contain <-> or ->: {equation}")
    return parse_side(left), parse_side(right)


def read_species(path: Path | None) -> Dict[str, Species]:
    species: Dict[str, Species] = {}
    if path is None:
        return species
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            name = row["species"].strip()
            phase = row.get("phase", "surface" if is_surface(name) else "gas" if is_gas(name) else "site").strip()
            site_size = parse_float(row.get("site_size"), 1.0) or 1.0
            mass = parse_float(row.get("molar_mass_amu"), None)
            carbon_pool = parse_bool(row.get("carbon_pool"), infer_carbon_pool(name))
            species[name] = Species(name=name, phase=phase, site_size=site_size, molar_mass_amu=mass, carbon_pool=carbon_pool)
    return species


def read_steps(path: Path) -> List[Step]:
    steps: List[Step] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            reactants, products = parse_equation(row["equation"])
            steps.append(
                Step(
                    step_id=row.get("step_id", f"R{len(steps)+1}").strip(),
                    kind=row.get("type", "surface").strip().lower(),
                    reactants=reactants,
                    products=products,
                    gaf_eV=parse_float(row.get("Gaf_eV"), None),
                    gar_eV=parse_float(row.get("Gar_eV"), None),
                    dg_eV=parse_float(row.get("dG_eV"), None),
                    sticking=parse_float(row.get("sticking"), 1.0) or 1.0,
                    mass_amu=parse_float(row.get("mass_amu"), None),
                    site_area_A2=parse_float(row.get("site_area_A2"), 10.0) or 10.0,
                    tof_label=row.get("tof_label", "").strip(),
                    notes=row.get("notes", "").strip(),
                )
            )
    return steps




def validate_inputs(steps: Sequence[Step], species_db: Mapping[str, Species]) -> None:
    known = set(species_db) | {"*"}
    errors: List[str] = []
    valid_types = {"surface", "ads", "adsorption", "dissociative_adsorption"}
    for step in steps:
        if step.kind not in valid_types:
            errors.append(f"{step.step_id}: unsupported type {step.kind!r}")
        names = set(step.reactants) | set(step.products)
        for name in names:
            if name not in known and not is_gas(name) and not is_vacant(name):
                errors.append(f"{step.step_id}: species {name!r} is not declared in species CSV")
        has_gas = any(is_gas(name) for name in names)
        if step.kind in {"ads", "adsorption", "dissociative_adsorption"}:
            if not has_gas:
                errors.append(f"{step.step_id}: adsorption step has no gas species")
            if step.dg_eV is None and step.gar_eV is None:
                errors.append(f"{step.step_id}: adsorption step needs dG_eV or Gar_eV for desorption")
            try:
                infer_gas_mass(step, species_db)
            except ValueError as exc:
                errors.append(str(exc))
        elif step.kind == "surface":
            if step.gaf_eV is None:
                errors.append(f"{step.step_id}: surface step needs Gaf_eV")
            if has_gas:
                errors.append(f"{step.step_id}: gas species in non-adsorption surface step is not supported by this first-pass model")
            if step.gar_eV is None and step.dg_eV is None:
                errors.append(f"{step.step_id}: reversible surface step should provide Gar_eV or dG_eV")
    for sp in species_db.values():
        if sp.site_size < 0:
            errors.append(f"{sp.name}: site_size must be non-negative")
    if errors:
        raise ValueError("Input validation failed:\n" + "\n".join(f"- {e}" for e in errors))


def collect_surface_species(steps: Iterable[Step], species_db: Mapping[str, Species]) -> List[str]:
    names = set()
    for step in steps:
        for name in list(step.reactants) + list(step.products):
            if is_surface(name):
                names.add(name)
    names.update(name for name, sp in species_db.items() if sp.phase == "surface" and is_surface(name))
    return sorted(names)


def infer_gas_mass(step: Step, species_db: Mapping[str, Species]) -> float:
    if step.mass_amu is not None:
        return step.mass_amu
    for name in list(step.reactants) + list(step.products):
        if is_gas(name) and name in species_db and species_db[name].molar_mass_amu:
            return float(species_db[name].molar_mass_amu)  # type: ignore[arg-type]
    raise ValueError(f"{step.step_id}: adsorption step needs mass_amu or gas species mass in species CSV")


def pressure_bar_for_species(name: str, pressures: Mapping[str, float]) -> float:
    variants = [name, name.replace("(g)", "_g"), name.replace("_g", ""), name.replace("_g", "(g)")]
    for key in variants:
        if key in pressures:
            return pressures[key]
    return 1.0


def adsorption_pressure_bar(step: Step, pressures: Mapping[str, float]) -> float:
    for name in step.reactants:
        if is_gas(name):
            return pressure_bar_for_species(name, pressures)
    for name in step.products:
        if is_gas(name):
            return pressure_bar_for_species(name, pressures)
    return 1.0


def k_ads_at_pressure(step: Step, T: float, species_db: Mapping[str, Species], p_bar: float) -> float:
    p_pa = p_bar * BAR_PA
    area_m2 = step.site_area_A2 * ANG2_M2
    mass_kg = infer_gas_mass(step, species_db) * AMU_KG
    return step.sticking * p_pa * area_m2 / math.sqrt(2.0 * math.pi * mass_kg * KB_SI * T)


def k_tst(barrier_eV: float, T: float) -> float:
    return (KB_SI * T / H_SI) * math.exp(-barrier_eV / (KB_EV * T))


def rate_constants(
    step: Step,
    T: float,
    pressures: Mapping[str, float],
    species_db: Mapping[str, Species],
    adsorption_dg_state: str,
) -> Tuple[float, float]:
    if step.kind in {"ads", "adsorption", "dissociative_adsorption"}:
        current_p = adsorption_pressure_bar(step, pressures)
        kf = k_ads_at_pressure(step, T, species_db, current_p)
    else:
        if step.gaf_eV is None:
            raise ValueError(f"{step.step_id}: surface step needs Gaf_eV")
        kf = k_tst(step.gaf_eV, T)

    if step.gar_eV is not None:
        kr = k_tst(step.gar_eV, T)
    elif step.dg_eV is not None:
        # dG_eV is for the written forward direction.
        # adsorption_dg_state=standard: dG is referenced to 1 bar gas, so k_des is pressure-independent.
        # adsorption_dg_state=current: dG already includes the current gas chemical potential.
        if step.kind in {"ads", "adsorption", "dissociative_adsorption"} and adsorption_dg_state == "standard":
            k_ref = k_ads_at_pressure(step, T, species_db, P_STD_BAR)
            kr = k_ref * math.exp(step.dg_eV / (KB_EV * T))
        else:
            kr = kf * math.exp(step.dg_eV / (KB_EV * T))
    else:
        kr = 0.0
    return kf, kr


def carbon_pool_coeff(stoich: Mapping[str, float], species_db: Mapping[str, Species]) -> float:
    total = 0.0
    for name, coeff in stoich.items():
        sp = species_db.get(name)
        if (sp and sp.carbon_pool) or (sp is None and infer_carbon_pool(name)):
            total += coeff
    return total


def carbon_pool_delta(step: Step, species_db: Mapping[str, Species]) -> float:
    return carbon_pool_coeff(step.products, species_db) - carbon_pool_coeff(step.reactants, species_db)


def build_rhs(
    steps: List[Step],
    surf_species: List[str],
    species_db: Mapping[str, Species],
    settings: ModelSettings,
    k_pairs_override: Sequence[Tuple[float, float]] | None = None,
):
    idx = {s: i for i, s in enumerate(surf_species)}
    site_sizes = np.array([species_db.get(s, Species(s, "surface", 1.0, None, infer_carbon_pool(s))).site_size for s in surf_species], dtype=float)
    carbon_pool_mask = np.array([species_db.get(s, Species(s, "surface", 1.0, None, infer_carbon_pool(s))).carbon_pool for s in surf_species], dtype=bool)
    if k_pairs_override is None:
        k_pairs = [rate_constants(step, settings.T, settings.pressures, species_db, settings.adsorption_dg_state) for step in steps]
    else:
        k_pairs = list(k_pairs_override)
    step_pool_delta = [carbon_pool_delta(step, species_db) for step in steps]

    def theta_vac(y: np.ndarray) -> float:
        yy = np.clip(y, 0.0, None)
        return max(EPS, 1.0 - float(np.dot(site_sizes, yy)))

    def theta_carbon_pool(y: np.ndarray) -> float:
        yy = np.clip(y, 0.0, None)
        if not carbon_pool_mask.any():
            return 0.0
        return float(np.sum(yy[carbon_pool_mask]))

    def carbon_factor(y: np.ndarray) -> float:
        if settings.carbon_pool_cap is None or settings.carbon_pool_cap <= 0.0:
            return 1.0
        pool = theta_carbon_pool(y)
        # Smooth approximation to max(0, 1 - theta_C7/cap). Hard clipping makes
        # stiff ODE solvers fail near the cap; this keeps the cap numerically stable.
        x = 1.0 - pool / settings.carbon_pool_cap
        sharpness = 50.0 * max(settings.carbon_pool_exponent, EPS)
        z = sharpness * x
        if z > 50.0:
            available = x
        elif z < -50.0:
            available = math.exp(z) / sharpness
        else:
            available = math.log1p(math.exp(z)) / sharpness
        return max(EPS, available)

    def activity_product(stoich: Mapping[str, float], y: np.ndarray, vac: float) -> float:
        value = 1.0
        yy = np.clip(y, 0.0, None)
        for name, coeff in stoich.items():
            if is_gas(name):
                # For adsorption rows, gas pressure is already folded into k_ads.
                value *= pressure_bar_for_species(name, settings.pressures) ** 0.0
            elif is_vacant(name):
                value *= vac ** coeff
            elif name in idx:
                value *= max(float(yy[idx[name]]), EPS) ** coeff
        return value

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        vac = theta_vac(y)
        pool_factor = carbon_factor(y)
        dy = np.zeros_like(y)
        for step, (kf, kr), pool_delta in zip(steps, k_pairs, step_pool_delta):
            # Carbon-pool cap only restricts the direction that increases total C7* coverage.
            # Internal C7* rearrangement/dehydrogenation (delta=0) and desorption/depletion are not blocked.
            forward_scale = pool_factor if pool_delta > 0.0 else 1.0
            reverse_scale = pool_factor if pool_delta < 0.0 else 1.0
            rf = forward_scale * kf * activity_product(step.reactants, y, vac)
            rr = reverse_scale * kr * activity_product(step.products, y, vac)
            rnet = rf - rr
            for name, coeff in step.products.items():
                if name in idx:
                    dy[idx[name]] += coeff * rnet
            for name, coeff in step.reactants.items():
                if name in idx:
                    dy[idx[name]] -= coeff * rnet
        return dy

    return rhs, idx, site_sizes, carbon_pool_mask, k_pairs, step_pool_delta, theta_vac, theta_carbon_pool, carbon_factor


def integrate_to_steady(rhs, y0: np.ndarray, t_final: float, steady_tol: float, max_extend: int):
    t0 = 0.0
    y = y0.copy()
    sol = None
    for _ in range(max_extend + 1):
        sol = solve_ivp(
            rhs,
            (t0, t_final),
            y,
            method="BDF",
            rtol=1e-8,
            atol=1e-14,
            max_step=max(t_final / 100.0, 1e-12),
        )
        if not sol.success:
            raise RuntimeError(sol.message)
        y = np.clip(sol.y[:, -1], 0.0, None)
        residual = float(np.max(np.abs(rhs(sol.t[-1], y)))) if y.size else 0.0
        if residual < steady_tol:
            return y, residual, sol.t[-1], True
        t0 = sol.t[-1]
        t_final *= 10.0
    assert sol is not None
    residual = float(np.max(np.abs(rhs(sol.t[-1], y)))) if y.size else 0.0
    return y, residual, sol.t[-1], False


def compute_step_rates(
    steps: List[Step],
    k_pairs: Sequence[Tuple[float, float]],
    y: np.ndarray,
    idx: Mapping[str, int],
    site_sizes: np.ndarray,
    pressures: Mapping[str, float],
    step_pool_delta: Sequence[float],
    pool_factor: float,
):
    vac = max(EPS, 1.0 - float(np.dot(site_sizes, np.clip(y, 0.0, None))))

    def prod(stoich: Mapping[str, float]) -> float:
        value = 1.0
        for name, coeff in stoich.items():
            if is_vacant(name):
                value *= vac ** coeff
            elif name in idx:
                value *= max(float(y[idx[name]]), EPS) ** coeff
            elif is_gas(name):
                value *= pressure_bar_for_species(name, pressures) ** 0.0
        return value

    rows = []
    for step, (kf, kr), pool_delta in zip(steps, k_pairs, step_pool_delta):
        forward_scale = pool_factor if pool_delta > 0.0 else 1.0
        reverse_scale = pool_factor if pool_delta < 0.0 else 1.0
        rf = forward_scale * kf * prod(step.reactants)
        rr = reverse_scale * kr * prod(step.products)
        rows.append((step, kf, kr, rf, rr, rf - rr))
    return vac, rows


def compute_observables(rate_rows: Sequence[Tuple[Step, float, float, float, float, float]]) -> Dict[str, float]:
    toluene_des = 0.0
    h2_des = 0.0
    mch_consumption = 0.0
    deh_fluxes: List[float] = []
    for step, _kf, _kr, _rf, _rr, rn in rate_rows:
        label = step.tof_label.lower()
        sid = step.step_id.lower()
        has_gas = any(is_gas(name) for name in step.reactants) or any(is_gas(name) for name in step.products)
        if has_gas and ("tol" in label or "toluene" in label or "c7h8" in sid):
            # Adsorption row written gas + sites <-> adsorbate: negative rn is desorption.
            toluene_des += max(0.0, -rn)
        if has_gas and ("h2" in label or "h2" in sid):
            h2_des += max(0.0, -rn)
        if has_gas and ("mch" in label or "c7h14" in sid):
            # Positive rn is net MCH adsorption/consumption for gas + sites -> adsorbate convention.
            mch_consumption += max(0.0, rn)
        if label.startswith("deh") or "deh" in sid:
            deh_fluxes.append(rn)
    net_dehydrogenation = float(np.mean(deh_fluxes)) if deh_fluxes else float("nan")
    if deh_fluxes:
        deh_min = float(np.min(deh_fluxes))
        deh_max = float(np.max(deh_fluxes))
        deh_span = deh_max - deh_min
    else:
        deh_min = deh_max = deh_span = float("nan")
    return {
        "net_dehydrogenation": net_dehydrogenation,
        "toluene_desorption": toluene_des,
        "h2_desorption": h2_des,
        "mch_consumption": mch_consumption,
        "deh_flux_min": deh_min,
        "deh_flux_max": deh_max,
        "deh_flux_span": deh_span,
    }


def solve_model(
    steps: List[Step],
    surf_species: List[str],
    species_db: Mapping[str, Species],
    settings: ModelSettings,
    y0: np.ndarray | None = None,
    k_pairs_override: Sequence[Tuple[float, float]] | None = None,
) -> Solution:
    rhs, idx, site_sizes, carbon_pool_mask, k_pairs, step_pool_delta, theta_vac, theta_carbon_pool, carbon_factor = build_rhs(
        steps, surf_species, species_db, settings, k_pairs_override=k_pairs_override
    )
    if y0 is None:
        y0 = np.zeros(len(surf_species), dtype=float)
    y, residual, t_end, converged = integrate_to_steady(rhs, y0, settings.t_final, settings.steady_tol, settings.max_extend)
    pool_factor = carbon_factor(y)
    vac, rate_rows = compute_step_rates(steps, k_pairs, y, idx, site_sizes, settings.pressures, step_pool_delta, pool_factor)
    observables = compute_observables(rate_rows)
    return Solution(
        T=settings.T,
        pressures=dict(settings.pressures),
        y=y,
        residual=residual,
        t_end=t_end,
        converged=converged,
        idx=idx,
        site_sizes=site_sizes,
        carbon_pool_mask=carbon_pool_mask,
        k_pairs=k_pairs,
        rate_rows=rate_rows,
        theta_vacant=vac,
        theta_carbon_pool=theta_carbon_pool(y),
        carbon_pool_factor=pool_factor,
        observables=observables,
    )


def finite_drc(
    steps: List[Step],
    surf_species: List[str],
    species_db: Mapping[str, Species],
    settings: ModelSettings,
    base: Solution,
    target: str,
    perturb: float,
) -> List[List[object]]:
    base_rate = base.observables.get(target, float("nan"))
    rows: List[List[object]] = []
    if not np.isfinite(base_rate) or abs(base_rate) < RATE_FLOOR:
        for step in steps:
            rows.append([step.step_id, step.tof_label, target, "nan", base_rate, "base target ~0"])
        return rows
    factor = 1.0 + perturb
    for i, step in enumerate(steps):
        kp = list(base.k_pairs)
        # Keeping Ki fixed means scaling forward and reverse rate constants together.
        kp[i] = (kp[i][0] * factor, kp[i][1] * factor)
        try:
            sol = solve_model(steps, surf_species, species_db, settings, k_pairs_override=kp)
            new_rate = sol.observables.get(target, float("nan"))
            if not np.isfinite(new_rate):
                drc = float("nan")
                note = "nonfinite perturbed target"
            else:
                drc = (new_rate - base_rate) / (base_rate * perturb)
                note = ""
        except Exception as exc:
            new_rate = float("nan")
            drc = float("nan")
            note = f"solver_fail: {exc}"
        rows.append([step.step_id, step.tof_label, target, f"{drc:.12e}", f"{new_rate:.12e}", note])
    return rows


def reaction_orders(
    steps: List[Step],
    surf_species: List[str],
    species_db: Mapping[str, Species],
    settings: ModelSettings,
    base: Solution,
    target: str,
    pressure_factor: float,
) -> List[List[object]]:
    base_rate = base.observables.get(target, float("nan"))
    rows: List[List[object]] = []
    gas_keys = ["C7H14_g", "H2_g", "C7H8_g"]
    if not np.isfinite(base_rate) or abs(base_rate) < RATE_FLOOR:
        for key in gas_keys:
            if key in settings.pressures:
                rows.append([key, target, "nan", f"{base_rate:.12e}", "nan", pressure_factor])
        return rows
    for key in gas_keys:
        if key not in settings.pressures:
            continue
        p2 = dict(settings.pressures)
        p2[key] *= pressure_factor
        s2 = ModelSettings(
            T=settings.T,
            pressures=p2,
            t_final=settings.t_final,
            steady_tol=settings.steady_tol,
            max_extend=settings.max_extend,
            carbon_pool_cap=settings.carbon_pool_cap,
            adsorption_dg_state=settings.adsorption_dg_state,
            carbon_pool_exponent=settings.carbon_pool_exponent,
        )
        try:
            sol = solve_model(steps, surf_species, species_db, s2)
            new_rate = sol.observables.get(target, float("nan"))
            note = ""
            if abs(base_rate) < RATE_FLOOR or abs(new_rate) < RATE_FLOOR or not np.isfinite(base_rate) or not np.isfinite(new_rate):
                order = float("nan")
            else:
                order = math.log(abs(new_rate / base_rate)) / math.log(pressure_factor)
        except Exception as exc:
            new_rate = float("nan")
            order = float("nan")
            note = f"solver_fail: {exc}"
        rows.append([key, target, f"{order:.12e}", f"{base_rate:.12e}", f"{new_rate:.12e}", pressure_factor, note])
    return rows


def temperature_grid(spec: str) -> List[float]:
    parts = [float(x) for x in re.split(r"[:,]", spec) if x.strip()]
    if len(parts) != 3:
        raise ValueError("--T-range expects START:STOP:STEP, e.g. 573:923:25")
    start, stop, step = parts
    if step <= 0:
        raise ValueError("T-range step must be positive")
    vals = []
    T = start
    while T <= stop + 1.0e-9:
        vals.append(T)
        T += step
    return vals


def apparent_ea_from_scan(rows: List[Dict[str, float]], target: str) -> None:
    if len(rows) < 3:
        for r in rows:
            r[f"Eapp_{target}_eV"] = float("nan")
            r[f"rate_sign_change_{target}"] = 0.0
        return
    temps = np.array([r["T_K"] for r in rows], dtype=float)
    signed = np.array([r[target] for r in rows], dtype=float)
    rates = np.abs(signed)
    safe = np.where(rates > RATE_FLOOR, rates, np.nan)
    ln_r = np.log(safe)
    dln_dT = np.gradient(ln_r, temps, edge_order=1)
    eapp = KB_EV * temps * temps * dln_dT
    signs = np.sign(signed)
    for i, (r, ea) in enumerate(zip(rows, eapp)):
        sign_change = False
        for j in (i - 1, i + 1):
            if 0 <= j < len(signs) and signs[i] != 0 and signs[j] != 0 and signs[i] != signs[j]:
                sign_change = True
        r[f"rate_sign_change_{target}"] = 1.0 if sign_change else 0.0
        r[f"Eapp_{target}_eV"] = float(ea) if np.isfinite(ea) and not sign_change else float("nan")


def write_csv(path: Path, header: List[str], rows: Iterable[Iterable[object]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow(row)


def write_solution_outputs(prefix: Path, surf_species: List[str], sol: Solution, carbon_pool_cap: float | None) -> Tuple[Path, Path, Path]:
    cov_path = prefix.with_suffix(".coverages.csv")
    rates_path = prefix.with_suffix(".rates.csv")
    summary_path = prefix.with_suffix(".summary.csv")
    write_csv(
        cov_path,
        ["species", "theta_ML", "site_size", "occupied_sites_ML", "carbon_pool"],
        (
            [
                s,
                f"{sol.y[sol.idx[s]]:.12e}",
                f"{sol.site_sizes[sol.idx[s]]:.6g}",
                f"{sol.site_sizes[sol.idx[s]] * sol.y[sol.idx[s]]:.12e}",
                bool(sol.carbon_pool_mask[sol.idx[s]]),
            ]
            for s in surf_species
        ),
    )
    write_csv(
        rates_path,
        ["step_id", "tof_label", "kf", "kr", "r_forward", "r_reverse", "r_net", "equation", "notes"],
        (
            [
                step.step_id,
                step.tof_label,
                f"{kf:.12e}",
                f"{kr:.12e}",
                f"{rf:.12e}",
                f"{rr:.12e}",
                f"{rn:.12e}",
                format_equation(step.reactants, step.products),
                step.notes,
            ]
            for step, kf, kr, rf, rr, rn in sol.rate_rows
        ),
    )
    cap_violation = 0.0 if carbon_pool_cap is None else max(0.0, sol.theta_carbon_pool - carbon_pool_cap)
    summary_rows = [
        ["T_K", f"{sol.T:.12e}"],
        ["p_C7H14_g_bar", f"{sol.pressures.get('C7H14_g', float('nan')):.12e}"],
        ["p_H2_g_bar", f"{sol.pressures.get('H2_g', float('nan')):.12e}"],
        ["p_C7H8_g_bar", f"{sol.pressures.get('C7H8_g', float('nan')):.12e}"],
        ["converged", int(sol.converged)],
        ["max_abs_dtheta_dt", f"{sol.residual:.12e}"],
        ["theta_vacant_sites", f"{sol.theta_vacant:.12e}"],
        ["occupied_sites", f"{1.0 - sol.theta_vacant:.12e}"],
        ["theta_carbon_pool", f"{sol.theta_carbon_pool:.12e}"],
        ["carbon_pool_cap", "" if carbon_pool_cap is None else f"{carbon_pool_cap:.12e}"],
        ["carbon_pool_violation", f"{cap_violation:.12e}"],
        ["carbon_pool_factor", f"{sol.carbon_pool_factor:.12e}"],
    ]
    summary_rows.extend([key, f"{value:.12e}"] for key, value in sol.observables.items())
    write_csv(summary_path, ["metric", "value"], summary_rows)
    return cov_path, rates_path, summary_path


def parse_pressures(args: argparse.Namespace) -> Dict[str, float]:
    p = {
        "C7H14_g": args.p_mch,
        "C7H8_g": args.p_tol,
        "H2_g": args.p_h2,
        "C7H14": args.p_mch,
        "C7H8": args.p_tol,
        "H2": args.p_h2,
    }
    for item in args.pressure:
        if "=" not in item:
            raise ValueError(f"--pressure expects name=bar, got {item!r}")
        name, value = item.split("=", 1)
        p[name.strip()] = float(value)
    return p


def format_side(stoich: Mapping[str, float]) -> str:
    if not stoich:
        return "0"
    parts = []
    for name, coeff in stoich.items():
        if abs(coeff - 1.0) < 1e-12:
            parts.append(name)
        else:
            parts.append(f"{coeff:g}{name}")
    return " + ".join(parts)


def format_equation(reactants: Mapping[str, float], products: Mapping[str, float]) -> str:
    return f"{format_side(reactants)} <-> {format_side(products)}"


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Mean-field MKM for MCH/toluene on Pt")
    ap.add_argument("--network", required=True, type=Path, help="CSV elementary-step network")
    ap.add_argument("--species", type=Path, help="CSV species table with site_size, mass and carbon_pool")
    ap.add_argument("--T", type=float, default=623.0, help="temperature / K")
    ap.add_argument("--T-range", help="temperature scan START:STOP:STEP, e.g. 573:923:25")
    ap.add_argument("--p-mch", type=float, default=1.0, help="C7H14 pressure / bar")
    ap.add_argument("--p-h2", type=float, default=0.15, help="H2 pressure / bar")
    ap.add_argument("--p-tol", type=float, default=0.10, help="C7H8 pressure / bar")
    ap.add_argument("--pressure", action="append", default=[], help="extra gas pressure, name=bar")
    ap.add_argument("--t-final", type=float, default=1.0e3, help="initial integration horizon / s")
    ap.add_argument("--steady-tol", type=float, default=1.0e-8, help="max |dtheta/dt| steady criterion")
    ap.add_argument("--max-extend", type=int, default=4, help="extend t_final by x10 this many times")
    ap.add_argument("--out-prefix", type=Path, default=Path("mkm_out"), help="output prefix")
    ap.add_argument("--carbon-pool-cap", type=float, default=0.11, help="sum theta(C7*) cap in ML; <=0 disables")
    ap.add_argument("--carbon-pool-exponent", type=float, default=1.0, help="exponent for carbon cap availability factor")
    ap.add_argument("--adsorption-dg-state", choices=["standard", "current"], default="standard", help="interpret adsorption dG as 1 bar standard-state or current-p dG")
    ap.add_argument("--drc", action="store_true", help="compute degree of rate control by finite difference")
    ap.add_argument("--drc-target", default="net_dehydrogenation", choices=["net_dehydrogenation", "toluene_desorption", "h2_desorption", "mch_consumption"], help="observable used for DRC")
    ap.add_argument("--drc-perturb", type=float, default=0.10, help="fractional perturbation for DRC; paper uses 0.10")
    ap.add_argument("--reaction-orders", action="store_true", help="compute finite-difference reaction orders at reference T")
    ap.add_argument("--pressure-perturb", type=float, default=1.12, help="pressure factor for reaction order; paper uses 12%% increase")
    args = ap.parse_args(argv)

    species_db = read_species(args.species)
    steps = read_steps(args.network)
    validate_inputs(steps, species_db)
    surf_species = collect_surface_species(steps, species_db)
    pressures = parse_pressures(args)
    carbon_pool_cap = args.carbon_pool_cap if args.carbon_pool_cap > 0 else None

    base_settings = ModelSettings(
        T=args.T,
        pressures=pressures,
        t_final=args.t_final,
        steady_tol=args.steady_tol,
        max_extend=args.max_extend,
        carbon_pool_cap=carbon_pool_cap,
        adsorption_dg_state=args.adsorption_dg_state,
        carbon_pool_exponent=args.carbon_pool_exponent,
    )

    base = solve_model(steps, surf_species, species_db, base_settings)
    cov_path, rates_path, summary_path = write_solution_outputs(args.out_prefix, surf_species, base, carbon_pool_cap)

    if args.drc:
        drc_rows = finite_drc(steps, surf_species, species_db, base_settings, base, args.drc_target, args.drc_perturb)
        drc_path = args.out_prefix.with_suffix(".drc.csv")
        write_csv(drc_path, ["step_id", "tof_label", "target", "drc", "perturbed_target_rate", "notes"], drc_rows)
    else:
        drc_path = None

    if args.reaction_orders:
        order_rows = reaction_orders(steps, surf_species, species_db, base_settings, base, args.drc_target, args.pressure_perturb)
        orders_path = args.out_prefix.with_suffix(".reaction_orders.csv")
        write_csv(orders_path, ["gas", "target", "reaction_order", "base_rate", "perturbed_rate", "pressure_factor", "notes"], order_rows)
    else:
        orders_path = None

    scan_path = None
    if args.T_range:
        scan_rows: List[Dict[str, float]] = []
        y0 = np.zeros(len(surf_species), dtype=float)
        for T in temperature_grid(args.T_range):
            settings = ModelSettings(
                T=T,
                pressures=pressures,
                t_final=args.t_final,
                steady_tol=args.steady_tol,
                max_extend=args.max_extend,
                carbon_pool_cap=carbon_pool_cap,
                adsorption_dg_state=args.adsorption_dg_state,
                carbon_pool_exponent=args.carbon_pool_exponent,
            )
            try:
                sol = solve_model(steps, surf_species, species_db, settings, y0=y0)
                row = {
                    "T_K": T,
                    "converged": float(sol.converged),
                    "max_abs_dtheta_dt": sol.residual,
                    "theta_vacant_sites": sol.theta_vacant,
                    "theta_carbon_pool": sol.theta_carbon_pool,
                    "carbon_pool_factor": sol.carbon_pool_factor,
                    "carbon_pool_violation": 0.0 if carbon_pool_cap is None else max(0.0, sol.theta_carbon_pool - carbon_pool_cap),
                    "notes": "",
                    **sol.observables,
                }
            except Exception as exc:
                row = {
                    "T_K": T,
                    "converged": 0.0,
                    "max_abs_dtheta_dt": float("nan"),
                    "theta_vacant_sites": float("nan"),
                    "theta_carbon_pool": float("nan"),
                    "carbon_pool_factor": float("nan"),
                    "carbon_pool_violation": float("nan"),
                    "notes": f"solver_fail: {exc}",
                    "net_dehydrogenation": float("nan"),
                    "toluene_desorption": float("nan"),
                    "h2_desorption": float("nan"),
                    "mch_consumption": float("nan"),
                    "deh_flux_min": float("nan"),
                    "deh_flux_max": float("nan"),
                    "deh_flux_span": float("nan"),
                }
            scan_rows.append(row)
        apparent_ea_from_scan(scan_rows, args.drc_target)
        scan_path = args.out_prefix.with_suffix(".tof_vs_T.csv")
        headers = list(scan_rows[0].keys()) if scan_rows else []
        write_csv(scan_path, headers, ([r.get(h, "") for h in headers] for r in scan_rows))

    print(f"T = {args.T:.2f} K")
    print("pressures_bar = " + ", ".join(f"{k}:{v:g}" for k, v in sorted(pressures.items()) if k.endswith("_g")))
    print(f"adsorption_dg_state = {args.adsorption_dg_state}")
    cap_violation = 0.0 if carbon_pool_cap is None else max(0.0, base.theta_carbon_pool - carbon_pool_cap)
    print(f"carbon_pool_cap = {carbon_pool_cap} ML ; theta_carbon_pool = {base.theta_carbon_pool:.12e} ML ; violation = {cap_violation:.12e} ML ; factor = {base.carbon_pool_factor:.12e}")
    print(f"steady_converged = {base.converged} ; t_end = {base.t_end:.3e} s ; max_abs_dtheta_dt = {base.residual:.3e} ML/s")
    print(f"theta_vacant_sites = {base.theta_vacant:.12e} ML")
    print(f"occupied_sites = {1.0 - base.theta_vacant:.12e} ML")
    print("observables:")
    for key, value in base.observables.items():
        print(f"  {key:22s} = {value:.12e} s^-1 site^-1")
    print("top_coverages:")
    for s in sorted(surf_species, key=lambda name: base.y[base.idx[name]], reverse=True)[:12]:
        print(f"  {s:12s} theta={base.y[base.idx[s]]:.6e} occupied={base.site_sizes[base.idx[s]] * base.y[base.idx[s]]:.6e}")
    print(f"wrote {cov_path}")
    print(f"wrote {rates_path}")
    print(f"wrote {summary_path}")
    if drc_path:
        print(f"wrote {drc_path}")
    if orders_path:
        print(f"wrote {orders_path}")
    if scan_path:
        print(f"wrote {scan_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
