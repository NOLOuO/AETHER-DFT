# AETHER-DFT

AETHER-DFT 是面向计算化学 / DFT 的对话式科研合伙人骨架。

当前阶段先完成“能接力开发、能跑主线”的最小产品底座：

- DFT 执行主线：搬自 `semi_auto_dft` 的 `planner -> builder -> runner -> parser -> analyzer -> export`
- 对话式科研合伙人：模型可自然聊天、主动调用工具、续接会话、压缩上下文、推荐下一步科研任务
- 共享工具层：搬入 `dft_shared`，优先支持结构分析、POSCAR writer、远程后端与结果解释基础设施
- 研究闭环：支持课题讨论、结构操作、任务规划、真实执行、结果解释、知识沉淀与进展回写
- 默认模型：`deepseek / deepseek-v4-pro`
- 可切换模型：`bailian / qwen3.7-max`
- 运行数据根：`.aether/`（projects、knowledge_base、runtime、runs、cache）
- API Key：默认不复制密钥，只按顺序只读查找：
  1. 当前项目 `api_keys.local.json`
  2. `AETHER_DFT_API_KEYS_PATHS` 环境变量指定的分号分隔路径
  3. `F:\agents\My-Agent\api_keys.local.json`
  4. `F:\agents\api_keys.local.json`
  5. `F:\_\DFTauto\research-copilot\api_keys.local.json`

## 仓库怎么读

- 结构地图：`docs/ARCHITECTURE.md`
- 根目录共识文档：`智能体架构.md`
- 工作区约束：`AGENTS.md`
- 避坑清单：`research/Common/避坑清单.md`

## 最常用入口

```powershell
aether
aether mainline
python -m aether_dft
aether chat "继续当前科研任务"
aether model current
aether model set deepseek:deepseek-v4-pro
aether permission ask
aether recommend --project demo
aether adsorption plan "计算 H2O 在 Pt(111) 上的吸附" --adsorbate H2O --material Pt(111)
```

## 默认环境

- 默认模型：`deepseek / deepseek-v4-pro`
- 交互方式：自然语言优先；模型自己决定是否调用工具，不要求用户记工具名
- 会话能力：支持 `resume`、上下文压缩与项目级持续回写
- 权限模式：`dev / 完全开发` 与 `ask / 需要用户同意`
- Python/pytest/pip：先 `D:/miniconda3/Scripts/activate`，再 `conda activate p312env`
- 结构/执行能力：`dft_app/` + `dft_shared/`
- 编排/提示/工具：`aether_dft/runtime_harness/`、`aether_dft/prompt_assets/`、`aether_dft/tool_specs/`
- 项目状态：`.aether/projects/`、`.aether/knowledge_base/`、`.aether/runtime/`、`.aether/runs/`

## 当前开发版交付状态

- M11-M17 主线已完成：模型可按证据自主选择工具，不再依赖固定流程。
- 已补长期科研续接能力：`project_continuity_digest` 汇总证据状态，`research_cycle_checkpoint` 落盘阶段性判断，`evidence_claim_audit` 防止无证据结论。
- 最近一次手动真实 API 验证（2026-05-26）：`deepseek:deepseek-v4-pro` 能调用工具完成 H2O/Pt(111) 候选建模计划。
- 最近一次手动真实 Step 3 冒烟（2026-05-26）：build → preflight → remote submit → cancel → fetch，提交后立即取消。
- 详细交付说明见 [`docs/DELIVERY.md`](docs/DELIVERY.md)。
