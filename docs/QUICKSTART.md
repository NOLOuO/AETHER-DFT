# AETHER-DFT Quickstart

目标：从一个干净工作区开始，跑通 **对话式计算化学合伙人** 的最小体验：持续交互式 REPL、research 课题切换、会话续接、真实模型工具调用 dry-run，以及可选的集群探测。

> 默认路径不会提交集群任务。真实 LLM/API 和真实集群提交都需要显式开关。

## 1. 安装与环境

普通用户不需要手动激活 Conda，也不需要记 Python 模块路径。第一次启动直接双击：

```powershell
cd <仓库根目录>
.\aether.cmd
```

第一次启动会自动创建项目内虚拟环境 `.venv/`、安装依赖，并注册全局 `aether` 命令。之后可直接：

```powershell
aether                 # 续接最近科研对话
aether --new           # 新开对话
aether "看看现在该做什么"
```

检查运行时：

```powershell
aether doctor
aether models
aether preload --project MCH-Pt-Br
aether model smoke --model deepseek:deepseek-v4-pro
aether model smoke --model bailian:qwen3.7-max
```

API key 可放在 `api_keys.local.json`，或使用环境变量：

- DeepSeek: `DEEPSEEK_API_KEY`
- 阿里百炼 / Qwen: `DASHSCOPE_API_KEY`

新增 OpenAI-compatible 模型时，优先复制 `config/model_providers.example.json` 为 `config/model_providers.json` 后改 provider/model/base_url/key env；不需要改 agent harness。

`preload` 是正式对话前的“启动态检查”：它不调用模型，默认也不连集群，只告诉你下一轮模型会预加载哪些设定：

- 当前模型、API key 是否可用、context window；
- 绑定 project 的 `.aether/projects/<slug>/` 长期状态；
- `research/AGENTS.md`、`research/Common/避坑清单.md`、项目 `研究进展.md` 和项目 common 规则；
- 最近 session 摘要、research workspace digest、cluster runtime digest；
- discussion / execution 两种模式会暴露多少工具。

## 2. 进入交互式科研合伙人

```powershell
aether
```

启动后直接输入自然语言即可。需要切换运行状态时输入 `/` 打开命令面板：

```text
aether[no-project|deepseek-v4-pro]> /
slash commands:
  1. /model       切换模型
  2. /project     切换 research 课题项目
  3. /resume      切换当前项目内的对话
  4. /new         新开当前项目会话
  5. /status      查看当前状态
```

## 3. 选择 research 课题项目

`/project` 对应的是 `research/<project>/` 里的真实科研项目，而不是内部 `.aether/projects` 容器。

```text
aether[no-project|deepseek-v4-pro]> /project
select project:
  1. MCH-Pt-Br
  2. MSR-Ru-Al2O3
```

`/resume` 只在当前 project 范围内切换对话：

```text
aether[MCH-Pt-Br|qwen3.7-max]> /resume
resume session:
  1. session_xxx  project=MCH-Pt-Br turns=4
```

完整对话 transcript 存在 `.aether/runtime/sessions/<session_id>/transcript.jsonl`。
为了让课题目录也能看见对应关系，AETHER 会在
`research/<project>/.aether/sessions/` 写入轻量索引；这里不是科研正文，
不会污染 `研究进展.md`。

## 4. 真实模型工具调用 smoke test（可选）

显式开启后才会访问外部模型 API：

```powershell
$env:AETHER_RUN_LLM_TESTS='1'
python -m pytest tests/test_llm_authored_adsorption_e2e.py -q -s
```

最近一次人工验证：`1 passed`。真实 DeepSeek 调用了：

1. `adsorbate_chemistry_hint`
2. `knowledge_search_for_system`
3. `adsorption_candidate_plan`

这验证的是“模型知道如何调用工具形成吸附候选推理 plan”，不是固定流水线。

## 5. 对话式 dry-run

```powershell
aether "只做规划：为 H2O/Pt(111) 生成吸附候选前，先说明该查哪些证据和工具"
```

CLI 会显示工具调用进度：

```text
thinking with deepseek:deepseek-v4-pro...
↻ model step 1/6
↳ tool adsorbate_chemistry_hint (0.8s) {"adsorbate":"H2O"}
✓ tool adsorbate_chemistry_hint status=ok (0.2s)
assistant> 根据已有证据，下一步应先比较 top / bridge / hollow 吸附候选...
```

工具选择阶段仍等待模型返回完整 JSON；无工具/最终回复阶段会边生成边显示，避免用户长时间只看到静态 “thinking”。

## 6. 集群探测与真实提交边界

只读探测：

```powershell
aether cluster config
aether cluster probe
```

在 REPL 中可以直接自然语言询问状态，模型会按需要调用只读集群工具：

```text
aether[MCH-Pt-Br|qwen3.7-max]> 看一下我现在集群上这些任务怎么样了，先只读。
```

查找并解释已有 OUTCAR（只读访问集群，只把证据复制到本地）：

```powershell
aether outcar find --limit 5
aether outcar analyze --latest --project MCH-Pt-Br --write-learning
```

`outcar analyze` 会把 `OUTCAR` 以及同目录下存在的 `OSZICAR` / `CONTCAR` / `POSCAR` 拉到
`.aether/runtime/remote_outcar_analysis/<run>_<hash>/`，再做本地解析；`--write-learning` 会把结论写回
`research/<project>/Learning/`。它不会提交、取消或修改集群任务。

真实提交必须显式启用对应 submit 命令/权限。测试提交时请使用短 `sleep` job，并只取消本轮返回的 job id；不要对 `squeue --me` 里的其他作业批量操作。

## 7. 常用命令

```powershell
aether
aether doctor
aether models
aether preload --project MCH-Pt-Br
aether model smoke --model deepseek:deepseek-v4-pro
aether model smoke --model bailian:qwen3.7-max
aether outcar find --limit 5
aether outcar analyze --latest --project MCH-Pt-Br --write-learning
aether project list
aether session list
aether tools list
aether "继续分析当前课题"
```

## 8. 测试策略

日常快速回归：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_aether_fast_path.py tests/test_harness_architecture.py tests/test_aether_cluster_realtime.py -q
```

维护者也可以使用自己的开发环境运行同样命令；发布入口本身不依赖 Conda。

```powershell
python -m pytest tests/test_aether_fast_path.py tests/test_harness_architecture.py tests/test_aether_cluster_realtime.py -q
```

提交前全量回归：

```powershell
python -m pytest -q
```

当前全量大约 4 分钟，主要耗时来自 ASE / pymatgen / spglib / adsorption workflow 真实结构处理。
