# dft-shared

`dft_shared` 是 `dft_tools` 仓库中的共享基础设施包，提供：

- 共享计算化学知识、结构分析与结果解释 contract
- 共享 workflow 配置读取
- 可被 AETHER-DFT 对话式 agent 复用的轻量工具模块

集群连接、远程提交、POTCAR materialization 与提交前证据门槛已统一迁移到
`dft_app.remote` / `dft_app.submission_gate`。`dft_shared` 不再保留可直接提交作业的
旧入口，避免绕过当前安全检查。

## 安装

推荐从包含各 DFT 工具包的仓库根目录统一安装：

```powershell
cd <your-dft-tools-repo>
pip install -e ./dft_shared -e ./xsd_mace_preopt -e ./mace_ts_search -e ./dft_web
```

如果只需要共享基础设施，也可以单独安装：

```powershell
cd <your-dft-tools-repo>/dft_shared
pip install -e .
```
