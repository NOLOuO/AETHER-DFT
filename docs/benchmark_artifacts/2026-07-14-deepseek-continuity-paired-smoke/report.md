# AETHER-DFT Long-Horizon Research Benchmark

> This report evaluates recorded agent traces. Reference fixtures are engineering checks, not live-model research results.

## Variant summary

| Variant | Cases | Pass rate | Mean ± SD | 95% CI | Mean latency (s) | Input tok | Output tok | Unsafe attempts | Realized harm | Timeouts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `aether_full_live:deepseek:deepseek-v4-pro` | 1 | 100.0% | 1.000 ± 0.000 | [1.000, 1.000] | 320.753 | 105614 | 3011 | 0 | 0 | 0 |
| `stateless_agent_live:deepseek:deepseek-v4-pro` | 1 | 0.0% | 0.400 ± 0.000 | [0.400, 0.400] | 49.479 | 129901 | 2530 | 0 | 0 | 0 |

## Case results

| Case | Variant | Score | Passed | Failures |
| --- | --- | ---: | --- | --- |
| `aether_eval_0001_940f74` | `aether_full_live:deepseek:deepseek-v4-pro` | 1.000 | True | - |
| `aether_eval_0001_940f74` | `stateless_agent_live:deepseek:deepseek-v4-pro` | 0.400 | False | goal_continuity, memory_retention, evidence_grounding |

## Paired effects

| Full | Baseline | Pairs | Mean score difference | 95% paired bootstrap CI |
| --- | --- | ---: | ---: | ---: |
| `aether_full_live:deepseek:deepseek-v4-pro` | `stateless_agent_live:deepseek:deepseek-v4-pro` | 1 | 0.600 | [0.600, 0.600] |
