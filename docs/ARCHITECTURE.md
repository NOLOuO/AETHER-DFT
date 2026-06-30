# AETHER-DFT Architecture

AETHER-DFT 采用类似 Codex / Claude Code 的分层 harness 结构：模型只负责推理，项目本身负责工具、状态、记忆、任务和可验证执行。

## 目录分层

仓库按四层理解最清楚：

1. **产品外壳**：`aether_dft/`
2. **执行核心**：`dft_app/`、`dft_shared/`
3. **编排与提示层**：`aether_dft/runtime_harness/`、`aether_dft/prompt_assets/`
4. **状态与接力层**：`.aether/`、`.omx/`

```text
aether_dft/       产品外壳：CLI、doctor、chat/agent、mainline 入口
config/           可提交配置；不放密钥
dft_app/          DFT 执行主线：planner -> builder -> runner -> parser -> analyzer -> export
dft_shared/       共享 DFT 工具：结构分析、POSCAR writer、远程后端、知识库基础设施
aether_dft/runtime_harness/  prompt/context 组装、模型循环、会话编排
aether_dft/prompt_assets/    system prompt 与 prompt sections
aether_dft/tool_specs/       旧工具规范保留为内部实现细节
research/         项目约束、避坑清单、方法论
docs/             架构、迁移和接力说明
.aether/          运行数据根目录：projects / knowledge_base / runtime / runs / cache
.omx/             计划、状态、上下文与运行时控制
tests/            回归测试
api_keys.local.json  本地密钥；不提交
```

## LLM runtime

默认模型：

```text
provider = bailian
model    = qwen3.7-max
base_url = https://dashscope.aliyuncs.com/compatible-mode/v1
api_key  = DASHSCOPE_API_KEY 或 api_keys.local.json 中的 bailian
```

调用方式固定使用 OpenAI SDK 的兼容接口：

```python
from openai import OpenAI

client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
client.chat.completions.create(
    model="qwen3.7-max",
    messages=[{"role": "user", "content": "..."}],
)
```

## 搬运原则

1. 不整仓硬拼；按能力层搬运。
2. DFT 主线优先从 `semi_auto_dft` 继承。
3. 结构/结果工具优先从 `dft_tools/dft_shared` 继承。
4. 后续项目状态、知识库和长期记忆统一沉淀为 AETHER 自有的发布版数据模型，不依赖个人外部项目。
5. 对话式外壳统一收敛到 `aether-dft`：持续 REPL、命令面板、项目/会话/模型选择器。

## 下一阶段接口目标

- `python -m aether_dft`：标准包入口，等价于 `aether`。
- `aether-dft` / `python -m aether_dft`：默认进入持续 REPL；输入 `/` 打开命令面板。
- `aether-dft doctor`：检查本地运行时。
- `aether-dft "自然语言科研问题"`：一次性模型主导回合；模型按证据选择工具。
- `aether-dft project ...`：创建/进入项目。
- `aether-dft run ...`：把自然语言研究任务转入 DFT 主线。
- `dft ...`：保留底层 DFT 自动化 CLI。

## Harness / Context 层

AETHER-DFT 现在有一层轻量 Codex/Claude Code-like harness：

- `aether_dft/context.py`：把 system prompt、当前 OpenAI-compatible 模型、项目状态、主线入口渲染为运行时上下文快照。
- `aether_dft/harness.py`：做 preflight、权限检查事件记录，事件写入 `.aether/runtime/logs/harness-events.jsonl`。
- `aether_dft/model_catalog.py`：模型选择不锁死厂商；`compatible:<model>` 可接任意 OpenAI-compatible base_url。
- `aether_dft/project_state.py`：项目级状态、研究进展和知识库目录分离。

这层只做运行时编排，不吞掉底层 DFT 主线；底层仍由 `dft_app` 负责 planner → builder → runner → parser → analyzer。

## Task Bridge / Knowledge 层

- `aether_dft/task_bridge.py`：把自然语言科研意图转换成 DFT task envelope，包含 plan/spec/readiness/推荐 DFT 命令；可写入 `.aether/projects/<slug>/tasks/` 并回写 `research_progress.md`。
- `aether_dft/research_loop.py`：把一轮对话后的结果沉淀为项目进展，补上 next steps / blockers / recommendations。
- `aether_dft/knowledge.py`：项目知识库轻量入口，保存到 `.aether/knowledge_base/<slug>/notes/`，支持 add/list/search/show。
- `aether-dft chat --task-plan/--task-run` 与 REPL 中的 `/task`、`/run`、`/recommend` 让“对话入口”本身可以进入 DFT 任务闭环并持续推进。

执行模式分级：

1. `dry_run`：默认，仅规划和打印主线计划。
2. `build`：生成 DFT 工作区，不提交。
3. `submit`：本地 Slurm 提交，必须显式指定。
4. `remote_submit`：远程 Slurm 提交，必须显式指定。

## Adsorption First Task

吸附是 AETHER-DFT 的第一条真实科研主线：

- `aether_dft/adsorption.py`：吸附任务规划、缺失输入诊断、候选构型生成、下一步任务建议。
- `aether_dft/recommendations.py`：基于项目上下文、任务记录和知识库的持续科研推荐。
- `aether-dft adsorption plan`：在缺 slab/adsorbate/material 时不假装能算，而是给出 concrete missing inputs。
- `aether-dft adsorption candidates`：复用 `dft_app` 的吸附候选生成器，产出 manifest、candidate POSCAR/CIF。
- `chat` REPL `/adsorb` 与 `/recommend`：让对话入口具备“科研合伙人持续推进”能力。

