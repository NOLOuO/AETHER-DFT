## research/ 工作区习惯

`research/` 是长期科研记忆，不是附件目录。重要节点要自然回写，并与集群 `~/research` 保持一致：

- 有新参数经验、失败教训、位点选择理由、模板约束时，用 `research_learning_capture` 写入项目 `Learning/`。
- 每轮开始或不确定“我们做到哪了”时，用 `project_continuity_digest` 读取项目状态、research、知识库、近期 run 和最近结果；它是证据地图，不是固定流程。
- 做出阶段性科研判断时，用 `research_cycle_checkpoint` 记录 goal / current_decision / evidence_refs / blockers / next_steps，方便下次会话从同一科学状态继续。
- 写科学 claim 前，必要时用 `evidence_claim_audit` 确认每条结论都有 evidence_refs；没有证据的内容只能写成假设或下一步。
- 本地与集群差异先用 `research_workspace_diff` 看摘要；需要统一时再选 `research_workspace_sync_to_cluster` 或 `research_workspace_sync_from_cluster`。
- 默认不要覆盖另一端：sync 工具默认 dry-run / planned；只有用户明确要应用或当前权限已允许时才 `apply=true`。
- 跑完或回拉输出后，用 `research_workspace_pull_logs` / `cluster_remote_fetch` 把证据带回本地，再做 `result_interpret`。
- 记录要写“为什么这么判断”，不要只写“已完成”。
