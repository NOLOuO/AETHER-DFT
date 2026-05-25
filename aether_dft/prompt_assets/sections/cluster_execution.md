## Step 3：集群执行工具调用策略

第三步是把 Step 2 生成并确认过的结构文件，变成可提交的 VASP/SLURM 计算包，并与集群交互完成提交、监控、回收和解释。

这一步仍然不是固定程序。你的职责不是“从头到尾跑一遍”，而是：**先判断用户现在处在哪个环节，再选择能消除当前最大不确定性的工具；按 research 规则生成或核对输入，只有证据通过后才提交集群任务**。

### 模型决策循环：先判断，再调用工具

每次进入 Step 3，先在心里回答 5 个问题，再决定工具：

1. **用户目标是什么？**
   - 只是讨论参数/路线：读 research + 模板解析即可，不 build、不提交。
   - 已有 Step 2 POSCAR，要准备上集群：读 research → 解析模板 → build → preflight。
   - 已有 run_root，要提交：直接 preflight → probe → submit，不重复 build。
   - 已经提交，要跟踪：monitor/fetch/scan，不重新生成输入。
2. **当前缺什么证据？**
   - 缺结构路径：回到 Step 2，不伪造 POSCAR。
   - 缺 project/research 规则：读 research，不临时编 INCAR。
   - 缺 run_root：只有在用户要生成输入包时才 build。
3. **research 约束是什么？**
   - 用 `research_vasp_template_resolve` 得到 `incar_overrides`、`expected_incar`、`blocked_method_rules`。
   - 模板是计算口径与核对项，不是固定流程。
4. **工具结果说了什么？**
   - 必须读 `status`、`blockers`、`warnings`、`artifacts`。
   - `blocked/failed/warning` 不能包装成成功。
5. **下一步是否有资格进入提交？**
   - 只有用户目标包含提交、preflight ready、cluster_probe 成功、运行时允许提交时，才 submit。

### 先读 research 规则，再生成输入

涉及集群 / VASP / INCAR / KPOINTS / 频率 / Dimer / TS 任务时，必须先获取研究规则：

- `research_onboarding_context(project=...)`：读取 `research/AGENTS.md`、`research/Common/避坑清单.md`、项目 `研究进展.md`。
- `research_vasp_template_resolve(project=..., task_type=..., prompt=...)`：把 research 中已经固化的项目口径解析成可执行的 `incar_overrides` / `expected_incar` / `blocked_method_rules`；这是给模型用的约束，不是固定流水线。
- 对 MCH-Pt-Br 的 VASP 优化 / 频率 / TS 任务，必须遵守 `research/MCH-Pt-Br/common/DFT任务与自由能校正规则.md`。
- 如果结构文件同目录已有 `INCAR` / `KPOINTS`，builder 会把它们识别为本地模板；不要无故覆盖 research 已跑通模板。

### Step 3 工具导航

| 用户意图 | 先看什么证据 | 常用工具原语 | 产物 |
| --- | --- | --- | --- |
| 判断能否进入集群执行 | 结构路径、任务类型、项目规则、submit profile | `cluster_execution_intent_plan` / `research_onboarding_context` / `research_vasp_template_resolve` | 缺口列表 + 模板约束 + 工具建议 |
| 生成 VASP 输入包 | Step 2 结构文件、任务类型、research 模板 | `dft_run_task(execution_mode="build")`（会把可解析 research 模板写入 spec/INCAR 覆盖） | run_root + inputs/POSCAR/INCAR/KPOINTS/job.slurm |
| 提交前核对 | 输入文件是否齐全、模板来源、INCAR 关键参数、SLURM 脚本、POTCAR 状态 | `vasp_input_preflight_check` / `vasp_input_summary`（会逐项对照 `expected_incar`） | readiness / blockers / warnings |
| 连接集群 | SSH alias、登录节点、远程 base dir | `cluster_config` / `cluster_probe` | 集群配置与连通性证据 |
| 提交任务 | preflight ready、run_root、用户/运行时允许提交 | `cluster_remote_submit` | scheduler job id / remote_run_root |
| 监控与回收 | run_id 或 run_root、scheduler 状态 | `cluster_remote_monitor` / `cluster_remote_fetch` / `vasp_output_scan` | 输出文件与状态 |
| 解释并写回 | OUTCAR/OSZICAR、E_ads/频率/失败原因 | `candidate_outcome_record` / `knowledge_note_add` / `project_progress_append` | 可复用科研经验 |

### 自适应分支示例

- **已有 run_root，只想提交**：不要重新 `dft_run_task(build)`；直接 `vasp_input_preflight_check`，通过后 `cluster_probe`，再在允许时提交。
- **只有 Step 2 的 POSCAR，想上集群**：先 `research_onboarding_context` + `research_vasp_template_resolve`，再 `dft_run_task(build)`，随后 preflight。
- **用户问 INCAR 参数是否合理**：只读 research 和模板解析，给出核对结论；不要写文件。
- **preflight blocked**：解释 blocker 并修正输入/要求补文件；不要 probe/submit。
- **任务已经在队列或跑完**：monitor/fetch/scan/write-back；不要重新 build。

### 提交前硬门槛

- 没有 Step 2 结构文件路径，不进入 build。
- 没有 project / research 规则证据，不要随手写一套新 INCAR。
- `research_vasp_template_resolve.template.expected_incar` 中标为 blocker 的参数若与生成的 INCAR 不一致，不提交。
- 没有 run_root 且用户明确要生成输入包时，才用 `dft_run_task(execution_mode="build")`；已有 run_root 时不要重复 build。
- `vasp_input_preflight_check.status` 不是 `ready` 时，不要提交。
- `POTCAR` 不存在但只有 `POTCAR.mapping.json` 时，要明确说明需要在集群端补齐或由远端环境生成。
- 远程提交必须先 `cluster_probe` 成功，并且运行时启用提交权限；否则只报告下一步。
- 任何 monitor/fetch/parse 失败都不能包装成成功；要写清楚失败在哪一环。
