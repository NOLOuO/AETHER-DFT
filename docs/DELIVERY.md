# AETHER-DFT 开发版交付说明

本文记录当前开发版 MVP 的真实能力、运行入口和交付验证方式。AETHER-DFT 的定位不是固定脚本，而是计算化学 / DFT 的对话式科研合伙人：模型根据项目状态和证据自主选择工具完成讨论、建模、集群执行、结果解释与研究回写。

## 已跑通的主线

0. **统一模型后端**
   - DeepSeek 与 Qwen 都通过同一个 OpenAI-compatible Chat Completions 工具调用路径接入，切换模型只改变 provider/model 配置，不改变 harness 业务流程。
   - Responses-compatible 解析保留为低层兼容 helper，但不再作为 Qwen3 tools 默认路径。
   - 真实 CLI smoke（2026-06-06）已验证 `deepseek:deepseek-v4-pro` 与 `bailian:qwen3.7-max` 都能调用同一个 `project_state_read` 工具路径：`aether-dft model smoke --model ...`。

1. **讨论与研究判断**
   - `web_search` / `literature_search` 在无 live connector 时返回 `connector_required`，不伪造外部事实。
   - `chemistry_compute` 支持 `convert`、`boltzmann`、`gibbs`、`tst_rate`、`kBT`，也兼容旧 `operation=...` 参数。
   - `discussion_state_snapshot` 可写 Markdown/JSON 快照，作为长对话 anchor。
   - `project_continuity_digest` 在每轮开始时汇总项目状态、research、KB、近期 run 和最近结果；它只给证据地图，不规定流程。
   - `research_cycle_checkpoint` 把阶段性科研判断、证据引用、blocker、下一步写入项目 checkpoint / progress / state。
   - `evidence_claim_audit` 检查准备写出的科学 claim 是否带证据引用；无证据 claim 必须降级为假设或下一步。

2. **Step 2 结构建模**
   - 模型通过 `structure_modeling_intent_plan` 获取导航，但不是固定流程。
   - 最近一次手动真实 API 验证（2026-05-26，本地运行态 artifact 未纳入 git）中，模型已自主构建 H2O/Pt(111) slab、添加吸附物、做 quality/sanity check 并 compose manifest。

3. **Step 3 集群输入与提交门控**
   - 模型通过 `cluster_execution_intent_plan` 学会：先读 research 模板，再 build 输入包，再 preflight，再 probe/submit。
   - 最近一次手动真实 Step 3 冒烟（2026-05-26，本地运行态 artifact 未纳入 git）已完成 build/preflight/remote submit/cancel/fetch：实际提交到 Slurm 后立即 `scancel`，不影响现有任务。
   - 真实 VASP 生产提交前仍应确认 POTCAR 来源、research 项目模板、队列/账号策略。

4. **Step 4 实时集群检查**
   - `cluster_job_status_brief`
   - `cluster_my_jobs`
   - `cluster_job_tail_log`
   - `cluster_job_partial_outcar`
   - `cluster_job_progress_estimate`

5. **Step 5 结果解释**
   - `result_interpret` 诚实区分 `no_outputs`、partial、not converged、converged。
   - 已用集群 `~/research/MCH-Pt-Br/.../Hstar_2Br-H_freq_Honly_20260601/OUTCAR` 做真实解析回归：频率任务正常结束、3 个实频、0 个虚频，verdict=`frequency_finished_no_imaginary_modes`。
   - 新增用户入口：`aether-dft outcar find` 可只读列出集群 `~/research` 下最近 OUTCAR；`aether-dft outcar analyze --latest --project <project> --write-learning` 会拉回 OUTCAR/OSZICAR/CONTCAR/POSCAR、解释结果并写入项目 Learning。
   - `next_experiment_propose` 给少量下一步科研动作，而不是扩成固定程序。

6. **Step 6 research/ 回写与同步**
   - `research_workspace_diff`
   - `research_workspace_sync_to_cluster`
   - `research_workspace_sync_from_cluster`
   - `research_workspace_pull_logs`
   - `research_learning_capture`
   - `research_cycle_checkpoint`

## 常用入口

```powershell
D:/miniconda3/Scripts/activate
conda activate p312env

# 交互式科研合伙人
aether chat --project <project>

# 单轮 agent harness
aether agent "继续当前 DFT 课题，先看项目状态再决定下一步" --project <project>

# 工具/模型自检
aether doctor
aether models
aether model smoke --model deepseek:deepseek-v4-pro
aether model smoke --model bailian:qwen3.7-max
aether cluster probe
aether outcar find --limit 5
aether outcar analyze --latest --project MCH-Pt-Br --write-learning
```

## 真实 API 测试

默认测试跳过真实 API。需要本地有 `api_keys.local.json` 或环境变量，并显式开启：

```powershell
D:/miniconda3/Scripts/activate
conda activate p312env
$env:AETHER_RUN_LLM_TESTS='1'
python -m pytest tests/test_llm_authored_adsorption_e2e.py -q -s
```

最近一次手动验证（2026-06-06）：`1 passed`，模型 `deepseek:deepseek-v4-pro` 能真实调用工具生成 adsorption candidate plan。该测试默认跳过，只有显式设置 `AETHER_RUN_LLM_TESTS=1` 才会访问真实 API。

## 全量回归

```powershell
D:/miniconda3/Scripts/activate
conda activate p312env
python -m pytest -q
```

本轮回归结果：`239 passed, 1 skipped`。

## 真实模型后端 smoke

最近一次手动执行（2026-06-06）：

- `aether-dft model smoke --model deepseek:deepseek-v4-pro --project model-smoke-demo` → `status=ok`，工具链：`project_state_read`
- `aether-dft model smoke --model bailian:qwen3.7-max --project model-smoke-demo` → `status=ok`，工具链：`project_state_read`
- DeepSeek streaming smoke 使用本地 key 成功访问真实 API；该模型在 thinking 模式下返回了 `reasoning_content` 流但未在 1200 token 内返回最终 `content`，程序现在诚实降级为“只收到 reasoning_content，需提高 max_tokens 或调低 thinking”，不再抛异常。

## 真实 OUTCAR 解析记录

最近一次手动执行（2026-06-06）：

- 命令：`aether-dft outcar analyze --latest --project MCH-Pt-Br --write-learning --json`
- 远端来源：`/home/szhang/research/MCH-Pt-Br/MKM_actual_data_package_20260605/2Br/freq_tasks/Hstar_2Br-H_freq_Honly_20260601/OUTCAR`
- 本地证据副本：`.aether/runtime/remote_outcar_analysis/Hstar_2Br-H_freq_Honly_20260601_b5e0389a91/`
- 已拉回证据：`OUTCAR`、`OSZICAR`、`CONTCAR`、`POSCAR`
- 解释结果：`frequency_finished_no_imaginary_modes`，最后 `TOTEN=-640.56417145 eV`，3 个实频、0 个虚频；POSCAR/CONTCAR 位移检查显示吸附几何稳定，最大位移约 `0.015 Å`。
- 已写回 Learning：`research/MCH-Pt-Br/Learning/OUTCAR-analysis-Hstar_2Br-H_freq_Honly_20260601.md`

## 真实 Step 3 冒烟验证记录

最近一次手动执行（2026-06-06）：

- build 输入包：`dft_run_task(execution_mode="build")`
- preflight：`vasp_input_preflight_check(require_potcar=False)` → `ready`
- 远程提交：`cluster_remote_submit` → Slurm job `99160` submitted
- 立即取消：`scancel <job_id>`，随后 `squeue -j <job_id>` 为空
- 本轮未执行回拉；取消后通过 `squeue -j 99160` 确认队列为空。

为避免消耗集群资源，冒烟测试把生成的 `job.slurm` 运行命令替换为短 `sleep`；它验证的是上传、sbatch 与取消链路，不代表生产 VASP 计算完成。

## 交付前注意

- `.aether/`、`.omx/`、`.secrets/`、`api_keys.local.json` 都是本地运行态/密钥，不提交。
- 生产提交前必须确认：
  - research 项目模板存在且 hash/规则已复核；
  - `POTCAR` 或 `POTCAR.mapping.json` 对应集群赝势库；
  - preflight 无 blocker；
  - `cluster_probe` 成功；
  - 用户明确允许真实提交。
- `behavior_audit` 后 harness 会强制进入自然语言回复，避免模型无限连续调工具。
- 长期项目续接建议每轮优先读 `project_continuity_digest`，阶段性决定用 `research_cycle_checkpoint` 落盘；这两者是“科研状态锚点”，不是固定程序。
