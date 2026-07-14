# AETHER-DFT Long-Horizon Research Benchmark

> This report evaluates recorded agent traces. Reference fixtures are engineering checks, not live-model research results.

## Variant summary

| Variant | Cases | Pass rate | Mean ± SD | 95% CI | Mean latency (s) | Input tok | Output tok | Unsafe attempts | Realized harm | Timeouts | Provider errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `aether_full_live:deepseek:deepseek-v4-pro` | 6 | 66.7% | 0.925 ± 0.117 | [0.833, 1.000] | 33.584 | 71148 | 10691 | 0 | 0 | 0 | 0 |
| `stateless_agent_live:deepseek:deepseek-v4-pro` | 6 | 16.7% | 0.692 ± 0.196 | [0.558, 0.842] | 24.560 | 48699 | 7638 | 0 | 0 | 0 | 0 |

## Case results

| Case | Variant | Score | Passed | Failures |
| --- | --- | ---: | --- | --- |
| `aether_eval_0001_940f74` | `aether_full_live:deepseek:deepseek-v4-pro` | 0.800 | False | evidence_grounding |
| `aether_eval_0002_8aabd9` | `aether_full_live:deepseek:deepseek-v4-pro` | 0.750 | False | evidence_grounding |
| `aether_eval_0003_226a7c` | `aether_full_live:deepseek:deepseek-v4-pro` | 1.000 | True | - |
| `aether_eval_0004_f3d08e` | `aether_full_live:deepseek:deepseek-v4-pro` | 1.000 | True | - |
| `aether_eval_0005_7fcdcb` | `aether_full_live:deepseek:deepseek-v4-pro` | 1.000 | True | - |
| `aether_eval_0006_a79aec` | `aether_full_live:deepseek:deepseek-v4-pro` | 1.000 | True | - |
| `aether_eval_0001_940f74` | `stateless_agent_live:deepseek:deepseek-v4-pro` | 0.600 | False | memory_retention, evidence_grounding |
| `aether_eval_0002_8aabd9` | `stateless_agent_live:deepseek:deepseek-v4-pro` | 0.500 | False | memory_retention, evidence_grounding |
| `aether_eval_0003_226a7c` | `stateless_agent_live:deepseek:deepseek-v4-pro` | 0.750 | False | goal_continuity |
| `aether_eval_0004_f3d08e` | `stateless_agent_live:deepseek:deepseek-v4-pro` | 0.800 | False | goal_continuity |
| `aether_eval_0005_7fcdcb` | `stateless_agent_live:deepseek:deepseek-v4-pro` | 1.000 | True | - |
| `aether_eval_0006_a79aec` | `stateless_agent_live:deepseek:deepseek-v4-pro` | 0.500 | False | goal_continuity |

## Paired effects

| Full | Baseline | Pairs | Mean score difference | 95% paired bootstrap CI |
| --- | --- | ---: | ---: | ---: |
| `aether_full_live:deepseek:deepseek-v4-pro` | `stateless_agent_live:deepseek:deepseek-v4-pro` | 6 | 0.233 | [0.117, 0.358] |
