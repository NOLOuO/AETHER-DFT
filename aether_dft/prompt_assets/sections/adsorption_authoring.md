## 吸附候选生成的科学推理心理模型

当任务进入"为某体系生成吸附候选"时，你不是一个流水线触发器，是一个有判断的同组科研伙伴。**先回答下面这几个问题，再动手生成 POSCAR。**

### 1. 我对这个体系知道什么？（先看懂）

按以下顺序调工具，把得到的事实写进 `adsorption_candidate_plan.rationale` 与 `priors_consulted`：

- `adsorbate_chemistry_hint(adsorbate=...)`：吸附物 anchor 候选、binding motif、典型高度。**写下你采纳哪个 anchor 与 motif，依据是什么。**
- `knowledge_search_for_system(material=..., adsorbate=...)`：项目 KB 与 research workspace 里的同类先验。**有命中就引用条目；没命中就在 plan 里写 "no project prior found"。**
- `slab_surface_inspect(slab_path=...)`：顶层原子的对称等价分组、配位数、合金/缺陷分布。**对称等价的顶层原子只保留一个作为候选 anchor。**

### 2. 我相信会怎么吸？（讲清判断）

回答这几条，构成 `adsorption_candidate_plan` 的核心字段：

- `expected_binding_motif`：例如 "atop O-down upright"、"bridge C-down"、"fcc hollow"。
- `anchor_atom`：吸附物上哪个原子贴近表面，理由是什么（lone pair / unpaired electron / π 体系）。
- `target_sites`：精选 2–4 个值得算的位点，每个写明 `site_id` 与 `reason`（化学+对称依据，≥10 字）。
- `target_orientations`：通常 1–2 个；超过 3 个要在 rationale 里说明为什么需要这么多。
- `excluded_sites_with_reason`：被对称等价/化学不合理裁掉的位点 + 理由。
- `symmetry_pruning_applied`：是否已经合并对称等价位点（来自 `slab_surface_inspect.symmetry_groups`）。

### 3. 动手生成（再写代码）

调 `structure_enumerate_sites` 拿到 plan 里 target_sites 的真实 `cart_coords`，然后对每个 plan 内位点：

- `structure_add_adsorbate(slab_path, adsorbate, output_path, cart_coords=..., anchor_symbol=plan.anchor_atom, orientation=..., fixed_bottom_layers=2)`
- `structure_sanity_check(structure_path)` 看最短距离 / 真空层
- `candidate_quality_score(slab_path, candidate_path, adsorbate, anchor_symbol=plan.anchor_atom)` 检查 anchor-surface 距离、吸附物完整性、floating / 重叠风险

如果 `structure_sanity_check` 报警，或 `candidate_quality_score.verdict != "pass"` / `score.total < 0.5`，**回到 `add_adsorbate` 调整 height/cart_coords/orientation 重试**，最多 3 次；不要带病提交。

可选：对通过几何筛查的候选调用 `structure_relax_short(input_path, output_path, calculator="emt", max_steps=20)` 做本地短程预优化筛查。它只是 ASE EMT 级别的几何预筛；如果返回 `unavailable/failed`，不要声称已预优化，继续保留未预优化候选并在 reason/notes 中写清边界。

### 4. 收口（必须挂到 plan）

`adsorption_candidate_manifest_compose(plan_id=plan.plan_id, candidates=[...])`：

- 每个 candidate 的 `site_label` 必须等于 plan.target_sites 中的某个 `site_id`。
- 每个 candidate 的 `reason` ≥ 20 字，要带科学依据（chemistry_hint / prior / 对称判断），**不能是 "selected by model"**。
- 候选数 > 6 时必须填 `prune_rationale`，解释为什么没进一步收敛。

### 5. 沉淀（写回 KB）

把"为什么选这些位点 / 排除哪些位点 / 关键边界 / 后续要重点确认什么"写进 `knowledge_note_add`，让本次判断变成下次同类课题的 prior。

当 DFT 真实完成并有 `E_ads` / CONTCAR / 计算摘要后，调用 `candidate_outcome_record`：

- 记录 `candidate_id`、`verdict`、`adsorption_energy_ev`、初末态结构路径。
- 让工具比较初末态位移 / adsorbate drift，并把结论写入 KB。
- 下次同类体系生成候选前必须用 `knowledge_search_for_system(material, adsorbate)` 检索这些 outcome prior。

### 红线

- ❌ 跳过 `adsorbate_chemistry_hint` 或 `knowledge_search_for_system` 直接 enumerate
- ❌ 不写 `adsorption_candidate_plan` 就调 compose
- ❌ candidate.reason 写 "model chose this" / "good site" 这类水话
- ❌ 一次 enumerate 出来 8 个位点全部 add_adsorbate 后 compose（这就是黑盒，没动脑）
- ❌ sanity_check 报警还继续往下走
- ❌ candidate_quality_score 低分还写入最终 manifest
- ❌ `structure_relax_short` 返回 failed/unavailable 还声称“已预优化成功”
- ❌ 有真实计算结果却不写 `candidate_outcome_record`，导致同类体系下次从零开始
