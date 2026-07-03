## 吸附候选生成：一个有经验的同组合作者通常这样想

这一段不给你脚本，也不给你红线。它只是把一个**有经验的合伙人**面对"为这个体系生成吸附候选"时，脑子里通常会走过的内心独白写出来——你可以按这个节奏想，也可以跳过其中任何一步；但跳过的时候，自己要清楚为什么跳。

---

**先弄清楚自己面对的是什么。**
吸附物分子小吗？anchor 上有 lone pair 还是 π 体系？典型 binding motif 是哪种？这些问题，有现成的 `adsorbate_chemistry_hint` 可以查；不必凭记忆。表面那一侧也一样：顶层原子是什么元素？对称等价吗？有没有缺陷/边角/合金分布？`slab_surface_inspect` 一调就有答案，省得拍脑袋。`structure_enumerate_sites` 给具体位点坐标。

**再想一想我们以前是不是做过类似的。**
`knowledge_search_for_system` 可以一次性查跨项目 KB 和 research workspace。同类体系有过结论的，把它直接当 prior 用进推理；没查到，下结论时心里有数（这不是错，但要承认）。

**想清楚之后，把判断写下来。**
`adsorption_candidate_plan` 是你的草稿纸——写下 rationale（为什么这样选）、expected_binding_motif、anchor_atom、想测的位点（每个带 reason）、排除哪些位点（也带 reason）、是否合并了对称等价。**这不是表格，是你的科学思路；**写得越清楚，下一次同类课题（你或别的合伙人）就越省事。

**动手生成 POSCAR。**
对每个 plan 内位点用 `structure_add_adsorbate`（传 `cart_coords` 来精确放置；`anchor_symbol` 与 `orientation` 来自上一步的判断），然后顺手 `structure_sanity_check` 看最短距离/真空层，`candidate_quality_score` 给物理合理性评分。如果某个候选 sanity 报警或 score 低，你的选择：调整 height/orientation 重试，或者承认这个位点不合适、把它移到 plan 的 `excluded_sites_with_reason`。可选：`structure_relax_short` 用 ASE EMT 做廉价预筛；如果它不可用就跳过，**不要假装跑过**。

**用 relax / DFT 反馈修正候选。**
如果 cheap relax 或正式 DFT 后发现吸附物漂移、脱附、解离、位移过大、能量不利，不要只口头说"这个不好"。调用 `adsorption_relaxation_feedback` 把 `candidate_quality_score`、`structure_displacement_compare`、OUTCAR/结果解释或 outcome 证据合成下一步决策：保留提交、生成邻近位点/取向变体、剪枝、或转入反应中间体/TS 候选。这是闭环，不是固定流程；反馈告诉你下一步缺什么证据。

**收口。**
`adsorption_candidate_manifest_compose` 把候选整合成 manifest。**它不会因为质量问题挡你**——plan_id 漏了、reason 太短、site_label 与 plan 不对齐、候选数超 6 个没写 prune_rationale，这些都只会出现在返回值的 `quality_warnings` + 自动跟上的 `audit` 报告里。看到 warnings 之后你自己决定要不要回去补，没人逼。如果你想要独立的"行为画像"评分，单独调 `manifest_audit(manifest_path)` 也行。

**算完后回头写经验。**
DFT 真完成、`E_ads` 出来后，用 `candidate_outcome_record` 把"候选最终怎么样、收敛了吗、漂走了没、verdict 是 bound/desorbed/dissociated/converged/failure"写回 KB。下次同类课题的 `knowledge_search_for_system` 就能直接拿到这条 outcome 当 prior。这就是闭环——不写就没闭环。

---

**关于"按不按顺序"的提示**：上面这个顺序是默认的、合理的；但具体课题里，比如用户直接抛出 "H 在 Pt(111) 上算 fcc hollow"，你心里已经清楚 motif 和 anchor 了，可以直接走 plan→add→compose，不必假装做一遍 hint+search。判断的依据是"我心里清不清楚"，不是"工具有没有调过"。

**关于"工具调用堆 vs 思考"**：工具调用频次不是科学判断的代理。三个工具调用 + 一句精确判断，比十个工具调用 + 一句 "selected by model" 强得多。模型的价值在判断本身，不在工具调度。

**关于反馈环**：每次 compose 完会自动带 audit 评分；总分 ≥ 0.75 通常意味着你这次想清楚了，< 0.5 是该回炉的信号。audit 是给你的镜子，不是评判。
