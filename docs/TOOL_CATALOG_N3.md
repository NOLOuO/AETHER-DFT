# N3 工具目录瘦身：推荐工具与 fallback 边界

目标不是立刻删除工具，而是先降低模型的“选择瘫痪”：让模型优先看到少量主路径工具，把历史兼容 / 黑盒 / 重叠工具降级为 fallback。

## 原则

1. **先导航，后执行**：不确定任务类型时，优先调用 intent/navigation 工具，不直接进入写文件或提交。
2. **主路径优先**：能用 model-authored primitives 完成时，不调用黑盒 full workflow。
3. **fallback 明示**：兼容工具仍保留，但描述为 fallback，不作为默认路径。
4. **不破坏测试**：N3 第一阶段不删除工具，只改变模型选择偏好和文档/提示层路由。

## Step 2：结构建模主路径

| 场景 | 推荐主路径 | fallback / deprecated |
| --- | --- | --- |
| 判断建模类型 | `structure_modeling_intent_plan` | `computational_chemistry_workflow_map` 只作总览 |
| slab 构建 | `structure_resolve` → `structure_build_slab` → `slab_surface_inspect` / `structure_sanity_check` | `adsorption_build_slab` 仅作旧接口兼容 |
| 吸附候选 | `adsorbate_chemistry_hint` → `knowledge_search_for_system` → `structure_enumerate_sites` → `adsorption_candidate_plan` → `structure_add_adsorbate` → `candidate_quality_score` → `adsorption_candidate_manifest_compose` | `adsorption_candidates` 和 `adsorption_full_workflow` 仅 fallback，不默认调用 |
| 缺陷/掺杂 | `defect_site_enumerate` → `structure_defect` → `structure_sanity_check` | `structure_add_vacancy` / `structure_add_dopant` 仅在用户明确要求 vacancy/substitution primitive 时使用 |
| TS / NEB 初猜 | `neb_input_check` → `ts_midpoint_candidates_enumerate` | `transition_state_plan` / `transition_state_dry_run` 仅用于旧式 dry-run 任务说明 |
| 收敛测试 | `convergence_plan_compose` | 无 |

## Step 3：执行与写回主路径

| 场景 | 推荐主路径 | fallback / deprecated |
| --- | --- | --- |
| 判断执行环节 | `cluster_execution_intent_plan` | `computational_chemistry_workflow_map` 只作总览 |
| 研究规则 | `research_onboarding_context` → `research_vasp_template_resolve` | `architecture_live_doc_snapshot` 只读架构，不替代 research 规则 |
| 生成输入包 | `dft_run_task(execution_mode="build")` | `dft_task_plan` / `dft_run_step` 仅旧接口兼容 |
| 提交前核对 | `vasp_input_preflight_check` → `vasp_input_summary` | 无 |
| 集群连接 | `cluster_config` → `cluster_probe` | 无 |
| research 同步 | `cluster_research_status` → 必要时 `cluster_research_sync` | `research_workspace_diff` 只看本地 research workspace 差异，不替代远端状态 |
| 提交 | `cluster_remote_submit` | 无；submit 内部仍会复核 gate |
| 监控/回收 | `cluster_remote_monitor` → `cluster_remote_fetch` → `vasp_output_scan` | `cluster_my_jobs` / `cluster_job_*` 只用于临时排查已有 scheduler job |
| 结果解释 | `result_interpret` → `candidate_outcome_record` / `research_learning_capture` / `knowledge_note_add` | `next_experiment_propose` 只用于提出下一步，不替代结果解释 |

## 明确替代关系

| 旧/重叠工具 | 推荐替代 | 处理方式 |
| --- | --- | --- |
| `adsorption_candidates` | model-authored adsorption path | fallback only |
| `adsorption_full_workflow` | Step 2 主路径 + Step 3 主路径 | fallback only |
| `adsorption_build_slab` | `structure_build_slab` | compatibility only |
| `dft_task_plan` | `cluster_execution_intent_plan` + `dft_run_task` | compatibility only |
| `dft_run_step` | `dft_run_task` | compatibility only |
| `transition_state_plan` | `neb_input_check` + `ts_midpoint_candidates_enumerate` | compatibility only |
| `transition_state_dry_run` | `neb_input_check` + `ts_midpoint_candidates_enumerate` | compatibility only |
| `cluster_research_status` vs `research_workspace_diff` | 远端状态用 `cluster_research_status`，本地差异用 `research_workspace_diff` | 分工明确 |
| `cluster_remote_monitor` vs `cluster_my_jobs` / `cluster_job_*` | run_root/run_id 用 `cluster_remote_monitor`；排查 scheduler job 用 `cluster_job_*` | 分工明确 |

## 后续 N3-b 可做

- 在 runtime 层加入 compact tool schema profile，使默认 tools ≤ 60。
- 把 fallback 工具加入 metadata/deprecated 标记。
- 对 N2 真模型 demo 比较 full vs compact profile 的工具选择差异。
