# AETHER-DFT

AETHER-DFT 是面向计算化学 / DFT 的对话式科研合伙人。它不是固定流水线脚本；用户用自然语言给出研究目标，模型基于项目状态、research 规则、结构文件、集群状态和计算结果，自主选择工具推进：查文献、建结构、生成 VASP 输入、提交/监控集群任务、解析 OUTCAR、沉淀项目经验。

## 产品形态：CLI-first

AETHER-DFT 的前端就是 **终端 CLI**，不是 Web 页面。交互体验目标接近 Codex / Claude Code：用户直接输入自然语言，少数控制动作通过 slash command 完成。

核心入口：

```powershell
.\aether.cmd
```

进入 REPL 后支持 `/model`、`/project`、`/resume`、`/auto`、`/status`、`/continue`、`/exit`。详细 CLI 产品设计见 `docs/CLI_FRONTEND_DESIGN.md`。

## 当前能力

- **交互式科研对话**：进入持续 REPL 后直接输入自然语言；模型自己判断是否需要调用工具。
- **项目级记忆**：会话、运行记录、知识笔记和 research 课题状态保存在项目内 `.aether/` 与 `research/<project>/`。
- **单一主模型**：默认使用 DeepSeek；Qwen/Bailian 只作为统一 OpenAI-compatible 接口下的可选兼容后端，
  不维护第二套 prompt、工具 schema 或业务流程。
- **DFT 工具闭环**：支持结构建模、吸附候选生成、VASP 输入检查、集群提交门控、实时查询、OUTCAR/OSZICAR 解析和结果回写；三子任务 adsorption workflow 的提交、监控、回拉、解析与吸附能汇总也可由同一交互模型自主调用。
- **证据优先**：模型必须区分本地记录、实时集群查询和真实计算输出；没有证据时只能提出下一步查询。
- **项目内环境**：首次启动自动创建 `.venv`，依赖安装到仓库内，不污染系统 Python 或 Conda 环境。

## 快速启动

前提：本机已有 Python **3.12 或 3.13**。AETHER 会用这个 Python 创建项目内 `.venv`；不会把依赖装进原 Python 环境。

### Windows 双击启动

在仓库根目录双击：

```text
aether.cmd
```

首次启动会自动：

1. 创建 `<仓库根目录>\.venv`
2. 使用 `pyproject.toml` 安装 AETHER-DFT 和运行依赖
3. 把 pip/cache/temp 放到项目内 `.cache/` 与 `.tmp/`
4. 进入交互式对话

全局命令不是默认写入；如需把 `aether` 注册到 PowerShell Profile，显式运行 `./aether.ps1 -InstallCommand`。

### PowerShell 启动

```powershell
cd <仓库根目录>
.\aether.cmd
```

常用命令：

```powershell
.\aether.cmd doctor --json          # 检查 Python、依赖、模型、缓存策略
.\aether.cmd                        # 续接最近对话并进入 REPL
.\aether.cmd --new                  # 新开对话
.\aether.cmd "看看当前项目下一步该做什么"  # 单轮自然语言
```

## 交互式用法

进入 REPL 后直接输入自然语言，例如：

```text
帮我看一下 MCH-Pt-Br 这个课题现在卡在哪里，下一步应该查什么证据。
根据已有结构生成几个 MCH 在 Pt 上的吸附候选，先不要提交集群。
看一下我集群上的任务跑到哪了，只读查询，不要取消任务。
```

Slash 命令：

```text
/model      打开模型选择；输入编号切换
/project    切换 research/ 下的课题项目
/resume     切换当前项目里的历史对话
/new        新开当前项目会话
/auto       开关自动科研推进模式
/status     查看当前模型、权限、项目、auto 状态
/exit       退出
```

`/auto` 是项目级自动推进开关。开启后模型会围绕当前研究目标持续收敛：证据不足时主动问人类，能推进时自动查证、建模、提交门控、监控和复盘。真实集群提交仍受权限和安全门控约束。

## API Key

AETHER 不提交密钥文件。默认只读查找：

1. 当前项目 `api_keys.local.json`
2. `AETHER_DFT_API_KEYS_PATHS` 环境变量指定的分号分隔路径
3. 进程环境变量，例如 `DEEPSEEK_API_KEY`、`DASHSCOPE_API_KEY`

示例 `api_keys.local.json`：

```json
{
  "deepseek": {"api_key": "..."},
  "bailian": {"api_key": "..."}
}
```

## 集群与 research 工作区

- 本地 research 课题位于 `research/<project>/`。
- 集群连接配置由项目内 cluster/profile 工具读取；模型根据自然语言和配置选择合适集群。
- 真实提交前必须通过输入文件 preflight、research 模板核对、集群 probe 和权限门控。
- 查询“跑到哪了”时，模型应调用只读工具（如 `cluster_my_jobs`、`cluster_job_status_brief`、`cluster_job_partial_outcar`），不能用本地记录冒充实时队列证据。

更多操作见：

- `docs/QUICKSTART.md`
- `docs/DELIVERY.md`
- `docs/ARCHITECTURE.md`

## 开发与验证

推荐使用项目 `.venv`：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall aether_dft dft_app dft_shared scripts tests
.\.venv\Scripts\python.exe -m ruff check aether_dft dft_app scripts tests
```

如果是新机器，可先运行 `aether.cmd` 完成 bootstrap，再执行上面的命令。

## 发布状态

当前版本适合内部试用和同组协作。公开发布前仍建议至少完成一次真实小体系的完整闭环验证：生成输入、提交集群、等待 VASP 完成、fetch/parse、写回 Learning。

## 许可证

MIT，见 `LICENSE`。
