# AETHER-DFT Quickstart

目标：从一个干净工作区开始，跑通 **对话式计算化学合伙人** 的最小体验：秒级状态查询、真实模型工具调用 dry-run、以及可选的集群探测。

> 默认路径不会提交集群任务。真实 LLM/API 和真实集群提交都需要显式开关。

## 1. 安装与环境

```powershell
cd F:\AETHER-DFT
D:/miniconda3/Scripts/activate
conda activate p312env
python -m pip install -e .
```

检查运行时：

```powershell
aether-dft doctor
aether-dft models
```

API key 可放在 `api_keys.local.json`，或使用环境变量：

- DeepSeek: `DEEPSEEK_API_KEY`
- 阿里百炼 / Qwen: `DASHSCOPE_API_KEY`

## 2. 创建一个项目

```powershell
aether-dft project init h2o-pt111 --description "H2O adsorption on Pt(111)"
aether-dft project list
```

项目状态会落在 `.aether/projects/<slug>/`，用于跨会话续接。

## 3. 秒级 fast-path：不用等模型

这些高频查询会直接走工具，不经过 LLM：

```powershell
aether-dft 看看怎么样了
aether-dft job 99160 怎么样
aether-dft 99160 收敛了吗
aether-dft 我有哪些项目
aether-dft 切到 deepseek
```

典型输出：

```text
当前队列：2 个作业（fast-path，未调用 LLM）

JOBID      STATE      ELAPSED    NODE         NAME / REASON
12345      RUNNING    1:23       c001         relax_a
12346      PENDING    0:00                    freq_b (Priority)
```

如果 fast-path 没命中，顶层自然语言会自动回退到普通对话入口：

```powershell
aether-dft "讨论一下 H2O 在 Pt(111) 上应该先算哪些吸附构型"
```

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
aether-dft chat --project h2o-pt111 "只做规划：为 H2O/Pt(111) 生成吸附候选前，先说明该查哪些证据和工具"
```

CLI 会显示工具调用进度：

```text
thinking with deepseek:deepseek-v4-pro...
↻ model step 1/6
↳ tool adsorbate_chemistry_hint {"adsorbate":"H2O"}
✓ tool adsorbate_chemistry_hint status=ok
```

## 6. 集群探测与真实提交边界

只读探测：

```powershell
aether-dft cluster config
aether-dft cluster probe
aether-dft 看看怎么样了
```

真实提交必须显式启用对应 submit 命令/权限。测试提交时请使用短 `sleep` job，并只取消本轮返回的 job id；不要对 `squeue --me` 里的其他作业批量操作。

## 7. 常用命令

```powershell
aether-dft doctor
aether-dft models
aether-dft model set deepseek:deepseek-v4-pro
aether-dft model set bailian:qwen3.7-max
aether-dft project list
aether-dft session list
aether-dft tools list
aether-dft chat --project h2o-pt111 "继续"
```

## 8. 测试策略

日常快速回归：

```powershell
python -m pytest tests/test_aether_fast_path.py tests/test_harness_architecture.py tests/test_aether_cluster_realtime.py -q
```

提交前全量回归：

```powershell
python -m pytest -q
```

当前全量大约 4 分钟，主要耗时来自 ASE / pymatgen / spglib / adsorption workflow 真实结构处理。
