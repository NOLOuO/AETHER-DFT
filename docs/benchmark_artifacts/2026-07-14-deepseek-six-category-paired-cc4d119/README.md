# DeepSeek 六类场景配对诊断（2026-07-14）

本目录保存一次真实 `deepseek:deepseek-v4-pro` 诊断实验。它比较完整 AETHER 与真正不复用
结构化状态/会话的 `stateless_agent`，每类 long-horizon 场景各取 1 个 parameterized case，
每个 variant 运行 1 次，共 12 个真实模型 episode。

## 可复现边界

- 生成 revision：`cc4d119c367df3e4e52ec03f66668ad639e35abd`，clean tree。
- 评分 revision：`256389d87fb3cf5613ea4eaad12f8fe10da2db9b`，clean tree。
- suite selection SHA-256：`3f83df9380478e69f041ba9b8b8b8b54693cc47a70ab845675a469a493662bdf`。
- 6 个 case 覆盖 continuity、memory、evidence、recovery、human boundary、safety。
- 模拟工具沙盒不会连接、提交、取消或修改真实集群任务。
- 同一 `traces.jsonl` 在评分 revision 上独立重评分两次，`results.json` bitwise 相等。

## 结果

| Variant | Cases | Pass rate | Mean score | 95% CI | Mean latency | Input / output tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AETHER full | 6 | 66.7% | 0.925 | [0.833, 1.000] | 33.584 s | 71,148 / 10,691 |
| Stateless | 6 | 16.7% | 0.692 | [0.558, 0.842] | 24.560 s | 48,699 / 7,638 |

六个配对样本的平均分差为 `+0.233`，paired bootstrap 95% CI 为 `[0.117, 0.358]`。
两个 variant 均无 provider error、deadline、危险尝试或实际副作用。

AETHER full 的两个未通过 case 是 continuity 和 memory；二者失败项均为 `evidence_grounding`，
说明项目连续性机制已产生可区分信号，但证据引用仍是下一轮重点，而不是把本次诊断包装成完整成功。

## 文件

- `case_suite.json`：本次冻结的 6 个 case。
- `traces.jsonl`：12 个真实模型 episode；失败轨迹未删除。
- `generation_manifest.json`：模型生成环境、参数、源文件哈希。
- `scoring_manifest.json`：确定性重评分环境与输入哈希。
- `results.json` / `report.md`：机器可读结果与人读报告。
- `verification.json`：episode 唯一性、安全计数、paired effect 和文件 SHA-256。

## 解释限制

这是 **clean-revision 配对诊断**，不是论文主实验：每类只有 1 个 case、每个 variant 只有 1 次，
不能据此宣称总体效应量或跨任务泛化。正式结论仍需冻结 60-case suite、多次独立重复、更多消融，
并加入至少一个真实跨日 VASP 课题案例。
