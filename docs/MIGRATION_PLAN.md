# AETHER-DFT 迁移计划

## 已搬入

- `dft_app/`：DFT 执行主线模块，承载 planner / builder / runner / parser / analyzer / export。
- `dft_shared/`：共享结构/结果工具层。
- `tests/upstream_auto_dft/`：来自 auto_dft 的回归测试，作为后续适配锚点。

## 已适配

- 默认 summary / planner 模型切换为 `bailian:qwen3.7-max`。
- `dft_app.llm.key_store` 只读取当前项目 `api_keys.local.json` 与显式环境变量路径，不复制密钥、不硬编码个人工作区。
- 新增 `aether-dft doctor` 作为最小外壳入口。

## 待搬入

1. `research-copilot/core/project.py` 的项目容器和状态目录。
2. `research-copilot/agent/research_automation` 的 harvest / knowledge_base 流。
3. `dft_tools` 的 `.xsd <-> POSCAR` 明确工具入口。
4. My-Agent 的 `.my-agent` 风格运行时和 workflow command 外壳。
