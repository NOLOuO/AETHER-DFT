from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def _run(args: list[str], *, input_text: str | None = None, timeout: int = 180) -> dict[str, Any]:
    started = time.time()
    proc = subprocess.run(
        [PY, "-m", "aether_dft", *args],
        cwd=ROOT,
        input=input_text,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout,
    )
    return {
        "argv": ["python", "-m", "aether_dft", *args],
        "returncode": proc.returncode,
        "seconds": round(time.time() - started, 3),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _must(step: str, result: dict[str, Any], checks: list[str]) -> dict[str, Any]:
    ok = result["returncode"] == 0 and all(token in result["stdout"] for token in checks)
    result["step"] = step
    result["ok"] = ok
    if not ok:
        raise RuntimeError(f"{step} failed: rc={result['returncode']} missing={checks}\nSTDOUT:\n{result['stdout']}\nSTDERR:\n{result['stderr']}")
    return result


def _json_stdout(result: dict[str, Any]) -> dict[str, Any]:
    return json.loads(result["stdout"])


def _write_args(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run real AETHER acceptance checks across model, REPL, structure, DFT input, cluster, and OUTCAR reading.")
    parser.add_argument("--project", default="MCH-Pt-Br")
    parser.add_argument("--work-dir", default=".aether/runtime/acceptance")
    parser.add_argument("--run-llm", action="store_true", help="Call live model APIs for DeepSeek tool-use checks.")
    parser.add_argument("--run-qwen", action="store_true", help="Also call Qwen/Bailian live API.")
    parser.add_argument("--allow-cluster-submit", action="store_true", help="Submit one test SLURM job, then cancel only that job in finally.")
    parser.add_argument("--outcar-root", help="Remote directory containing OUTCAR for read-only real result parsing.")
    args = parser.parse_args(argv)

    work_dir = (ROOT / args.work_dir / time.strftime("e2e-%Y%m%d-%H%M%S")).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    report: list[dict[str, Any]] = []
    submitted_job_id: str | None = None

    try:
        report.append(_must("preload_project", _run(["preload", "--project", args.project, "--json"], timeout=90), ["\"status\": \"ok\""]))
        report.append(_must("model_catalog", _run(["model", "list", "--json"], timeout=60), ["current_model_id", "models"]))
        report.append(_must(
            "interactive_repl_slash_commands",
            _run(["chat"], input_text=f"/project {args.project}\n/status\n/exit\n", timeout=90),
            ["AETHER", "session", args.project],
        ))

        if args.run_llm:
            report.append(_must(
                "deepseek_model_tool_use",
                _run([
                    "chat", "--model", "deepseek", "--max-steps", "2", "--max-tokens", "360",
                    "只读真实工具调用测试：请调用 aether_discover_tools 查询 project_memory 类工具，然后一句话说明。不要写文件，不要提交任务。",
                ], timeout=180),
                ["tool", "aether_discover_tools"],
            ))
        if args.run_qwen:
            report.append(_must(
                "qwen_model_tool_use",
                _run([
                    "chat", "--model", "qwen", "--max-steps", "2", "--max-tokens", "360",
                    "只读真实工具调用测试：请调用 aether_discover_tools 查询 project_memory 类工具，然后一句话说明。不要写文件，不要提交任务。",
                ], timeout=180),
                ["tool", "aether_discover_tools"],
            ))

        slab_dir = work_dir / "slab"
        candidates_dir = work_dir / "candidates"
        report.append(_must(
            "step2_build_slab",
            _run(["adsorption", "build-slab", "--material", "Pt", "--output-dir", str(slab_dir), "--source", "ase", "--miller", "1", "1", "1", "--supercell", "2", "2", "1", "--min-vacuum-size", "12"], timeout=180),
            ["POSCAR", "atom_count"],
        ))
        report.append(_must(
            "step2_candidates",
            _run(["adsorption", "candidates", "--slab-path", str(slab_dir / "POSCAR"), "--adsorbate", "H2O", "--material", "Pt", "--project", "acceptance-smoke", "--output-dir", str(candidates_dir), "--max-sites-per-family", "1"], timeout=180),
            ["candidate_count", "candidate_manifest"],
        ))
        candidate_poscar = candidates_dir / "candidates" / "ontop_01_upright" / "POSCAR"
        task = _run([
            "task", "run", "对 H2O/Pt(111) ontop_01_upright 做一次快速 relaxation 输入准备",
            "--project", args.project,
            "--material", "Pt",
            "--structure-path", str(candidate_poscar),
            "--task-type", "relax",
            "--submit-profile", "c32",
            "--build",
        ], timeout=180)
        report.append(_must("step3_build_inputs", task, ["run_root", "build_result", "ready"]))
        task_payload = _json_stdout(task)
        run_root = task_payload["build_result"]["run_root"]

        preflight_args = _write_args(work_dir / "preflight.args.json", {"run_root": run_root, "project": args.project, "task_type": "relax", "require_potcar": False})
        report.append(_must("step3_preflight", _run(["tools", "run", "vasp_input_preflight_check", "--arguments-file", str(preflight_args)], timeout=90), ["\"status\": \"ready\"", "POTCAR.mapping.json"]))
        report.append(_must("cluster_probe", _run(["cluster", "probe"], timeout=90), ["\"status\": \"ok\"", "squeue"]))

        if args.allow_cluster_submit:
            submit_args = _write_args(work_dir / "submit.args.json", {"run_root": run_root})
            submit = _run(["tools", "run", "cluster_remote_submit", "--allow-cluster-submit", "--arguments-file", str(submit_args)], timeout=180)
            report.append(_must("cluster_submit", submit, ["submitted", "job_id", "remote_run_root"]))
            submit_result = _json_stdout(submit)["result"]
            submitted_job_id = (
                submit_result.get("job_id")
                or (submit_result.get("details") or {}).get("job_id")
                or (re.search(r"job_id=(\d+)", submit["stdout"] or "") or re.search(r"Submitted batch job\s+(\d+)", submit["stdout"] or "") or [None, None])[1]
            )
            if not submitted_job_id:
                raise RuntimeError("cluster_submit did not return job_id")
            status_args = _write_args(work_dir / "status.args.json", {"job_id": submitted_job_id})
            report.append(_must("cluster_status_submitted_job", _run(["tools", "run", "cluster_job_status_brief", "--arguments-file", str(status_args)], timeout=90), [submitted_job_id]))

        if args.outcar_root:
            outcar_args = _write_args(work_dir / "outcar.args.json", {"remote_run_root": args.outcar_root})
            report.append(_must("real_outcar_partial_parse", _run(["tools", "run", "cluster_job_partial_outcar", "--arguments-file", str(outcar_args)], timeout=90), ["last_toten_ev", "last_ionic_step"]))

    finally:
        if submitted_job_id:
            cancel_args = _write_args(work_dir / "cancel.args.json", {"job_id": submitted_job_id})
            cancel = _run(["tools", "run", "cluster_job_cancel", "--arguments-file", str(cancel_args)], timeout=90)
            cancel["step"] = "cluster_cancel_submitted_job"
            cancel["ok"] = cancel["returncode"] == 0 and submitted_job_id in cancel["stdout"]
            report.append(cancel)
        report_path = work_dir / "real_acceptance_report.json"
        report_path.write_text(json.dumps({"status": "ok" if all(item.get("ok") for item in report) else "failed", "work_dir": str(work_dir), "steps": report}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"report_path": str(report_path), "steps": [{"step": item.get("step"), "ok": item.get("ok"), "seconds": item.get("seconds")} for item in report]}, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
