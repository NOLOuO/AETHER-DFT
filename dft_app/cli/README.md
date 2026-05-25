# CLI

当前 CLI 已经打通到 `build` 和基础 `runner`：

- `dft run --dry-run`
- `dft run --status`
- `dft run --reset`
- `dft step <phase>`
- `dft list`
- `dft report <run_id>`

当前行为补充：

- 简单标准任务：
  - `dft run` 会继续走 `build -> submit`
- 复杂组合任务：
  - `dft run` 会生成 workflow scaffold
  - 当前不会直接进入真实计算提交

当前版本已经支持：

1. 自然语言生成 `ExperimentSpec`
2. 创建任务工作目录
3. 生成真实 `POSCAR` / `INCAR` / `KPOINTS`
4. 记录 `RunRecord`
5. 尝试执行 `step submit`
6. 在缺少 `sbatch`、`squeue`、`sacct` 时给出明确阻塞信息
7. 执行 `step parse` 并生成 `parsed_result.json`
8. 通过 `report` 汇总任务定义、构建信息、提交信息和解析结果
9. 执行 `step analyze` 并生成 `analysis_summary.json` 与 Markdown 报告
10. 执行 `step export` 并生成交付目录、`export_manifest.json` 与 zip 包
11. 通过 `--remote` 执行 SSH 远程提交与远程监控
12. 通过 `fetch` 手动拉回远程输出
13. 通过 `--submit-profile` 选择集群提交模板
14. 通过 `adsorption-workflow --status` 汇总三个吸附子任务的整体状态与下一步建议
15. 通过 `adsorption-workflow --monitor` / `--fetch` 协调三个吸附子任务的远程轮询与结果拉回闭环
16. 通过 `adsorption-workflow --parse-analyze` 在三个子任务结果齐全后生成吸附能聚合结果与 Markdown 报告

推荐的吸附工作流闭环：

1. `dft adsorption-workflow --run-root <complex_run_root> --status`
2. `dft adsorption-workflow --run-root <complex_run_root> --submit [--remote]`
3. `dft adsorption-workflow --run-root <complex_run_root> --monitor`
4. 远程完成后执行 `dft adsorption-workflow --run-root <complex_run_root> --fetch`
5. `dft adsorption-workflow --run-root <complex_run_root> --parse-analyze`

关键产物：

- 主 run: `metadata/adsorption_workflow_status.json`
- 子任务: `metadata/remote_monitor_summary.json` / `metadata/remote_fetch_summary.json`
- 聚合结果: `metadata/adsorption_energy_result.json` 与 `report/adsorption_energy_report.md`

当前仍未完成：

1. 真实集群环境联调

当前内置 `submit_profile`：

1. `c32`
2. `b96`
