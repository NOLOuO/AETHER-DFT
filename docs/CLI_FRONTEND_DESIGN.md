# AETHER-DFT CLI Frontend Design

AETHER-DFT 的产品前端就是 **终端 CLI**。不做 Web 前端，不做 Electron 壳，不把科研流程拆成按钮式固定向导。CLI 的目标是让用户像使用 Codex / Claude Code 一样，用自然语言持续推进一个计算化学课题；Slash command 只负责切换状态和处理少数显式控制动作。

## 1. 产品定位

CLI 是科研合伙人的交互界面，不是传统命令行工具集合。

用户主要做三件事：

1. 直接输入自然语言研究目标、观察、问题或约束。
2. 在少数状态切换场景使用 slash command。
3. 回答 AI 主动提出的人类判断问题。

模型负责：

1. 判断当前需要什么证据。
2. 自主选择工具查文献、读 research、建结构、检查集群、解析结果。
3. 在证据不足、权限不足、分支代价高或目标不清时向人类提问。
4. 把进展写回项目状态，而不是要求用户手动拼接流程。

## 2. 启动体验

默认入口：

```powershell
.\aether.cmd
```

启动后必须直接进入可用 REPL：

```text
┌──────────────────────────────────────────────────────────┐
│ Session Info                                             │
├──────────────────────────────────────────────────────────┤
│ Program: AETHER-DFT                                      │
│ Version: 0.1.0                                           │
│ Model: deepseek:deepseek-v4-pro                          │
│ Context: 1,000,000 tokens                                │
│ Permission: 完全开发                                     │
│ Session: session_xxx                                     │
│ Project: MCH-Pt-Br                                       │
│ Preload: project/session/research injected each turn     │
└──────────────────────────────────────────────────────────┘
直接输入自然语言即可；模型会自己判断是否需要调用工具。
输入 / 打开命令面板；也可直接用 /model、/project、/resume、/exit。
```

启动页必须回答用户的四个问题：

- 当前用哪个模型？
- 当前在哪个 research 项目？
- 是否续接了哪个 session？
- 模型是否已经预加载项目状态？

不要在启动页塞长说明。长说明放 `/help` 或文档。

## 3. Prompt 行

格式：

```text
aether[MCH-Pt-Br|deepseek-v4-pro]>
```

规则：

- 第一段是 research project slug；没有项目时显示 `no-project`。
- 第二段是当前模型短名；模型全 ID 在 `/status` 里看。
- Prompt 行不显示长路径，避免压迫感。

## 4. Slash command 原则

Slash command 是控制面，不是科研流程。

| 命令 | 目的 | 交互要求 |
|---|---|---|
| `/` | 打开命令面板 | 编号选择，像 Codex / Claude Code |
| `/model` | 切换 OpenAI-compatible 后端 | 无参数时打开选择器；不能切到缺 key 模型 |
| `/project` | 切换 `research/<project>/` | 无参数时打开项目选择器 |
| `/resume` | 切换当前项目里的会话 | 默认限定当前 project；跨项目用 `/resume all` |
| `/new` | 新开当前项目会话 | 不删除旧会话 |
| `/status` | 人类可读状态面板 | 默认不是 JSON |
| `/status --json` | 机器可读状态 | 给测试和脚本用 |
| `/continue` | 重试失败或中断的用户输入 | 必须保留 pending prompt |
| `/history` | 搜索当前 session 历史 | 不替代 research 记忆 |
| `/compact` | 手动压缩旧上下文 | 完整 transcript 仍保留 |
| `/permission` | 切换权限模式 | 不能绕过显式人类确认 |
| `/auto` | 目标驱动自动科研开关 | 是开关，不是 tick 命令 |
| `/exit` | 退出 | 打印 resume 命令 |

禁止把科研步骤做成固定 slash 命令链，例如 `/step1 literature`、`/step2 build`。科研推进应由模型根据目标和证据动态决定。

## 5. `/model` 设计

用户输入：

```text
/model
```

CLI 输出：

```text
select model:
  1. deepseek:deepseek-v4-pro  available  ctx=1,000,000
  2. bailian:qwen3.7-max       available  ctx=1,000,000
  3. minimax:...               missing key
```

要求：

- DeepSeek、Qwen、后续新增模型都走同一套 OpenAI-compatible backend。
- `/model` 无参数时打开选择器；这符合 Codex / Claude Code 心智。
- `/model qwen` 这类直接参数只作为高级快捷方式，不是主路径。
- 缺 API key 的模型不能切换成功，必须显示缺哪个 env。

## 6. `/project` 设计

`/project` 对应 `research/<project>/`，不是内部 `.aether/projects`。

用户输入：

```text
/project
```

CLI 输出：

```text
select project:
  1. MCH-Pt-Br       MCH on Br/Pt
  2. MSR-Ru-Al2O3    methane steam reforming
```

切换后：

```text
project switched: MCH-Pt-Br
```

要求：

- 支持模糊匹配，但不要猜错时静默切换。
- 切换 project 后，新 session 默认绑定该 project。
- `/resume` 默认只显示当前 project 的会话。

## 7. `/resume` 设计

用户输入：

```text
/resume
```

CLI 输出：

```text
resume session:
  1. MCH-Pt-Br mechanism discussion      session_xxx  turns=12
  2. adsorption candidate cleanup        session_yyy  turns=5
```

要求：

- 默认限定当前 project，避免误续接别的课题。
- 支持 `/resume all` 跨项目。
- 如果 transcript 有损坏，CLI 应提示已恢复/跳过坏片段，而不是崩溃。
- `/continue` 处理的是失败/中断的当前 prompt，不等同于 `/resume`。

## 8. `/status` 设计

默认输出人类可读面板：

```text
AETHER status
  model      : deepseek:deepseek-v4-pro (1,000,000 ctx)
  project    : MCH-Pt-Br
  session    : session_xxx  turns=4
  title      : MCH-Pt-Br 反应路径讨论
  permission : 完全开发 (dev)
  auto       : ON  status=monitoring  submit=off
  pending    : 继续检查当前课题
               use /continue to retry
  ref        : research\MCH-Pt-Br\.aether\sessions\session_xxx.json
  json       : /status --json
```

规则：

- 默认不输出 JSON。
- JSON 只通过 `/status --json`。
- pending / auto / permission 必须一眼可见。

## 9. `/auto` 设计

`/auto` 是项目级目标驱动开关。

用户输入：

```text
/auto
```

含义：

- 如果已有明确 research goal，则开启。
- 如果没有明确 goal，从当前 project/session 尝试推断。
- 推断不到时，问用户一个具体问题，而不是要求用户填一堆参数。

用户也可以直接：

```text
/auto 验证 MCH 在 Br/Pt 上脱氢的最低能路径，并找出速控步骤
```

要求：

- `/auto tick` 不是主交互；用户不应该手动推进轮次。
- 后台 follow-up queue 决定何时推进。
- AI 可以准备结构、检查输入、查询集群、写 checkpoint。
- 真实集群提交不能由模型自授权；必须有 CLI/人类明确授权。
- 遇到目标不清、昂贵分支、权限、不可逆动作时，AI 必须在 CLI 里问人。

## 10. AI 问人类问题

当模型需要人类判断时，CLI 直接显示问题：

```text
AI needs your decision
Project : MCH-Pt-Br
Why     : 两个吸附构型会消耗不同的集群资源，现有证据无法判断优先级。

Question:
  这轮优先验证低覆盖度单分子吸附，还是直接做高覆盖度 Br 共吸附？

Options:
  1. 低覆盖度单分子吸附
  2. Br 共吸附
  3. 先只生成两套输入，不提交

Your answer>
```

规则：

- 每次只问一个问题。
- 问题必须说明为什么需要人类。
- 如果有选项，选项要短。
- 用户可以自由文本回答。
- 回答要写回 auto question record，后续模型必须能读取。

## 11. 工具执行显示

长工具不能静默。

基本格式：

```text
thinking with deepseek:deepseek-v4-pro...
↻ model step 1/15
↳ tool cluster_my_jobs ... 0.0s
  still running: SSH query 2.1s
✓ tool cluster_my_jobs status=ok (2.4s)
```

并发只读工具：

```text
↳ parallel tools: cluster_my_jobs, cluster_job_status_brief
```

规则：

- 只读工具可以并发。
- 修改状态/文件/集群的工具必须受权限模式控制。
- 取消集群 job 永远需要人类显式确认。

## 12. 错误恢复

模型/API/网络失败：

```text
模型调用失败: temporary provider timeout
本 session 仍然保留；这条输入已保存，修复后可输入 /continue 重试。
```

用户 Ctrl+C：

```text
本轮已中断；输入 /continue 可继续重试这条请求。
```

要求：

- 失败/中断不能丢用户 prompt。
- `/status` 必须显示 pending。
- `/continue` 只能重试 pending prompt，不应编造新输入。

## 13. 非目标

不做：

- Web dashboard。
- 固定按钮式科研流程。
- 把每个科研步骤做成用户必须记住的命令。
- 把外部私人项目路径作为产品运行依赖。

可以以后做：

- 更好的 TUI 渲染。
- 后台 auto daemon 的系统托盘/Windows service 包装。
- 结构预览图导出。

这些都不能改变 CLI-first 主产品形态。

