# dft-shared

`dft_shared` 是 `dft_tools` 仓库中的共享基础设施包，提供：

- 远程连接后端（OpenSSH / WinSCP）
- 集群提交流程抽象
- 共享 workflow 配置读取
- 结构分析与知识库辅助模块

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
