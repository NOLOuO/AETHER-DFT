## Step 3：集群执行工具调用策略

第三步是把 Step 2 生成并确认过的结构文件，变成可提交的 VASP/SLURM 计算包，并与集群交互完成提交、监控、回收和解释。

这一步仍然不是固定程序。你的职责是：**按 research 里的规则和模板生成输入，核对证据，只有核对通过后才提交集群任务**。

### 先读 research 规则，再生成输入

涉及集群 / VASP / INCAR / KPOINTS / 频率 / Dimer / TS 任务时，必须先获取研究规则：

- `research_onboarding_context(project=...)`：读取 `research/AGENTS.md`、`research/Common/避坑清单.md`、项目 `研究进展.md`。
- 对 MCH-Pt-Br 的 VASP 优化 / 频率 / TS 任务，必须遵守 `research/MCH-Pt-Br/common/DFT任务与自由能校正规则.md`。
- 如果结构文件同目录已有 `INCAR` / `KPOINTS`，builder 会把它们识别为本地模板；不要无故覆盖 research 已跑通模板。

### Step 3 工具导航

| 用户意图 | 先看什么证据 | 常用工具原语 | 产物 |
| --- | --- | --- | --- |
| 判断能否进入集群执行 | 结构路径、任务类型、项目规则、submit profile | `cluster_execution_intent_plan` / `research_onboarding_context` | 缺口列表 + 工具建议 |
| 生成 VASP 输入包 | Step 2 结构文件、任务类型、research 模板 | `dft_run_task(execution_mode="build")` | run_root + inputs/POSCAR/INCAR/KPOINTS/job.slurm |
| 提交前核对 | 输入文件是否齐全、模板来源、INCAR 关键参数、SLURM 脚本、POTCAR 状态 | `vasp_input_preflight_check` / `vasp_input_summary` | readiness / blockers / warnings |
| 连接集群 | SSH alias、登录节点、远程 base dir | `cluster_config` / `cluster_probe` | 集群配置与连通性证据 |
| 提交任务 | preflight ready、run_root、用户/运行时允许提交 | `cluster_remote_submit` | scheduler job id / remote_run_root |
| 监控与回收 | run_id 或 run_root、scheduler 状态 | `cluster_remote_monitor` / `cluster_remote_fetch` / `vasp_output_scan` | 输出文件与状态 |
| 解释并写回 | OUTCAR/OSZICAR、E_ads/频率/失败原因 | `candidate_outcome_record` / `knowledge_note_add` / `project_progress_append` | 可复用科研经验 |

### 提交前硬门槛

- 没有 Step 2 结构文件路径，不进入 build。
- 没有 project / research 规则证据，不要随手写一套新 INCAR。
- `dft_run_task` 默认先用 `execution_mode="build"` 生成本地输入包；不要直接 remote submit。
- `vasp_input_preflight_check.status` 不是 `ready` 时，不要提交。
- `POTCAR` 不存在但只有 `POTCAR.mapping.json` 时，要明确说明需要在集群端补齐或由远端环境生成。
- 远程提交必须先 `cluster_probe` 成功，并且运行时启用提交权限；否则只报告下一步。
- 任何 monitor/fetch/parse 失败都不能包装成成功；要写清楚失败在哪一环。
