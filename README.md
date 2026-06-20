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

发布版不会硬编码个人工作区路径；如需复用已有密钥文件，请显式设置 `AETHER_DFT_API_KEYS_PATHS`。

## 仓库怎么读

- 结构地图：`docs/ARCHITECTURE.md`
- 根目录共识文档：`智能体架构.md`
- 工作区约束：`AGENTS.md`
- 避坑清单：`research/Common/避坑清单.md`

## 最常用入口

```powershell
.\aether.cmd                    # 第一次/双击启动：自动创建项目 .venv 并进入对话
aether                         # 进入持续交互式科研合伙人
aether "帮我看一下现在该做什么"  # 单轮自然语言；模型自己决定是否调用工具
aether project list            # 查看 research/ 课题项目
aether preload --project MCH-Pt-Br
aether adsorption plan "计算 H2O 在 Pt(111) 上的吸附" --adsorbate H2O --material Pt(111)
```

进入 `aether` 之后，直接输入自然语言。需要切状态时输入 `/` 打开命令面板：

```text
/model       切换模型
/project     切换 research 课题项目
/resume      切换当前项目里的对话
/new         新开当前项目会话
/permission  切换权限模式
```

## 默认环境

- 默认模型：`deepseek / deepseek-v4-pro`
- 交互方式：自然语言优先；模型自己决定是否调用工具，不要求用户记工具名
- 会话能力：支持 `resume`、上下文压缩与项目级持续回写
- 权限模式：`dev / 完全开发` 与 `ask / 需要用户同意`
- Python/pytest/pip：发布入口使用项目内 `.venv/`；第一次运行 `aether.cmd` 会自动创建和安装
- 结构/执行能力：`dft_app/` + `dft_shared/`
- 编排/提示/工具：`aether_dft/runtime_harness/`、`aether_dft/prompt_assets/`、`aether_dft/tool_specs/`
- 项目状态：用户面对的是 `research/<project>/`；`.aether/` 保存会话、运行记录、索引和兼容元数据。
- 对话存储：完整 transcript 在 `.aether/runtime/sessions/<session_id>/transcript.jsonl`；绑定 research 课题时，会同步写一个轻量索引到 `research/<project>/.aether/sessions/`，用于 `/resume` 和人工追踪对应关系。

## 当前开发版交付状态

- M11-M17 主线已完成：模型可按证据自主选择工具，不再依赖固定流程。
- 已补长期科研续接能力：`project_continuity_digest` 汇总证据状态，`research_cycle_checkpoint` 落盘阶段性判断，`evidence_claim_audit` 防止无证据结论。
- 最近一次手动真实交互验证（2026-06-14）：REPL `/` 命令面板、`/model`、`/project`、`/resume` 可用；`qwen` 后端可真实对话和调用只读工具。
- 最近一次手动真实 Step 3 冒烟（2026-05-26）：build → preflight → remote submit → cancel → fetch，提交后立即取消。
- 详细交付说明见 [`docs/DELIVERY.md`](docs/DELIVERY.md)。
