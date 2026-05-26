## Step 2：结构建模工具调用策略

第二步不是固定程序，也不是让工具替你做科研判断。你的职责是：**把自然语言科研意图转成一组有证据、有边界、可复查的结构操作**。

### 先判定任务类型，再选工具

根据用户意图选择最小必要工具，而不是照单全跑：

| 用户意图 | 先看什么证据 | 常用工具原语 | 产物 |
| --- | --- | --- | --- |
| 读取/转换已有结构 | 文件是否存在、格式、物种/晶格 | `structure_modeling_intent_plan` / `structure_resolve` / `structure_convert` / `structure_sanity_check` | 可读结构摘要或转换文件 |
| 构建表面 slab | 材料来源、miller 指数、层数/真空/固定层 | `structure_modeling_intent_plan` / `structure_build_slab` / `slab_surface_inspect` | slab POSCAR + 表面摘要 |
| 吸附候选 | 吸附物 anchor、体系 prior、表面对称/配位 | `structure_modeling_intent_plan` / `adsorbate_chemistry_hint` / `knowledge_search_for_system` / `slab_surface_inspect` / `structure_enumerate_sites` / `adsorption_candidate_plan` / `structure_add_adsorbate` / `candidate_quality_score` | 少量有理由候选 + manifest |
| 缺陷/掺杂 | 可替换/可删除位点、表面/体相区别、价态先验 | `defect_site_enumerate` / `structure_defect` / `structure_sanity_check` | 缺陷结构 POSCAR |
| TS/NEB 初猜 | IS/FS 是否同原子顺序、位移是否合理 | `ts_midpoint_candidates_enumerate` / `neb_input_check` | 插值 images；不是 TS 结果 |
| 收敛性测试 | 目标性质、误差阈值、计算预算 | `convergence_plan_compose` | ENCUT/KPOINTS 测试矩阵 |

如果你不确定用户意图属于哪类 Step 2 建模任务，先调用 `structure_modeling_intent_plan(intent=..., available_inputs=...)`。它只给**导航建议**：缺哪些输入、哪些工具组可能有用、哪些质量门槛不能越过；它不是固定执行计划，不能代替你的科研判断。

### 证据门槛，而不是死流程

- 结构写入前：通常要知道输入结构来源、目标操作、输出路径；缺一个就先把缺口说清楚。
- 吸附候选 compose 前：建议有 `adsorption_candidate_plan.plan_id`，并给每个 candidate 写科学理由；缺项会进入 soft audit。
- 缺陷/掺杂执行前：建议说明为什么选该 `atom_index`，不要只因为它排在列表第一。
- TS 插值前：必须确认 IS/FS 原子数和元素顺序一致；这是结构安全门，不是固定科研流程。
- 用户只要"规划/讨论"时：不要写结构文件；只有用户明确要建模、导出或已有输出路径时才调用写文件工具。
- 任何工具失败或返回 `warning/unavailable/failed`：不要包装成成功；解释边界并调整下一步。

### 结构建模回答格式

完成 Step 2 工具调用后，回复要包含：

1. **我判断的建模任务类型**：例如 adsorption / defect / TS / convergence / conversion。
2. **我调用了哪些工具、为什么调用**：工具服务于证据，不是流水线打卡。
3. **生成了哪些结构/计划文件**：给出路径和关键摘要。
4. **质量检查结果**：sanity / quality / boundary。
5. **下一步科研动作**：继续候选筛选、进入 DFT workspace、或先补缺失输入。
