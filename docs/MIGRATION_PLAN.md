# AETHER-DFT 迁移计划

## 已搬入

- `dft_app/`：来自 `F:\_\DFTauto\DFT\auto_dft\dft_app`，作为 DFT 执行主线。
- `dft_shared/`：来自 `F:\_\DFTauto\DFT\dft_tools\dft_shared`，作为共享结构/结果工具层。
- `tests/upstream_auto_dft/`：来自 auto_dft 的回归测试，作为后续适配锚点。

## 已适配

- 默认 summary / planner 模型切换为 `bailian:qwen3.7-max`。
- `dft_app.llm.key_store` 增加只读参考 key 路径查找，不复制密钥。
- 新增 `aether-dft doctor` 作为最小外壳入口。

## 待搬入

1. `research-copilot/core/project.py` 的项目容器和状态目录。
2. `research-copilot/agent/research_automation` 的 harvest / knowledge_base 流。
3. `dft_tools` 的 `.xsd <-> POSCAR` 明确工具入口。
4. My-Agent 的 `.my-agent` 风格运行时和 workflow command 外壳。
