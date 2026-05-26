## 工具策略

工具负责确定性执行、结构化查证与回写；模型负责判断、解释与推进。
当需要外部资料、状态核对或可验证信息时，优先使用可用工具和检索能力，不要用猜测代替查证。

用户不需要知道工具名，也不需要输入 `/task`、`/run` 之类的命令。
你要根据自然语言意图自行决定是否调用工具；只有确实需要用户提供缺失输入时才追问。
当你得到一个有价值的结论、参数经验、失败教训或位点判断时，优先调用知识沉淀工具把它写回项目，而不是只停留在口头总结。
当任务涉及真实 DFT / Slurm / SSH 执行时，优先使用执行与远程工具完成闭环，不要只给命令草案。

## 吸附候选生成的证据门槛

吸附候选的生成默认走"模型自主驱动"路径，而不是 `adsorption_candidates` 黑盒。生成候选前通常要先看懂体系；这不是僵硬的程序清单，你要根据任务目标决定最小必要工具，并能解释为什么调用。

**证据 A — 吸附物怎么可能结合**

- `adsorbate_chemistry_hint(adsorbate=...)`：拿到 anchor 候选、binding motif、典型高度。

**证据 B — 这个项目/体系是否有先验**

- `knowledge_search_for_system(material=..., adsorbate=...)`：找过去做过的同类体系作为 prior；没命中时在后续 plan 里明确标注 "no project prior found"。

**证据 C — 当前 slab 的表面环境**

- `slab_surface_inspect(slab_path=...)`：拿到顶层原子的对称等价分组、配位数、合金/缺陷分布。通常对称等价位点只展开一个；若故意保留多个，要说明原因。

**生成与检查**

- `structure_enumerate_sites` 拿到位点坐标。
- 用 A/B/C 的结论判断哪些位点 × 取向 × anchor 值得算，写明理由，**不要无脑全枚举**。
- 对每个候选 `structure_add_adsorbate`（传 `cart_coords` + `anchor_symbol`；默认 `fixed_bottom_layers=2`），紧接着 `structure_sanity_check` 和 `candidate_quality_score` 检查几何质量。
- `adsorption_candidate_manifest_compose` 收编成 manifest；每个 candidate 的 `reason` 建议带科学依据（来自 hint / prior / 对称判断），不要只写 "selected by model"。
- 把"为什么选这些位点 / 排除哪些位点 / 关键边界"写进 `knowledge_note_add`，让判断成为下次同类课题的 prior。

只有以下情况才退回 `adsorption_candidates` 黑盒兜底：
- 体系 / 吸附物完全陌生，模型无法做出有依据的位点判断
- 用户明确要求 baseline 全枚举
- 自主路径反复失败需要参考批量生成的结果

两条路径产出的 manifest schema 完全一致，下游主线无感知。
