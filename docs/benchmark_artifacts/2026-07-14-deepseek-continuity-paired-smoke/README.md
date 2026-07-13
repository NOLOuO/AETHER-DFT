# DeepSeek continuity paired smoke — 2026-07-14

This is a **single-case real-API smoke test**, not a formal benchmark result and not evidence of population-level generalization.

## What was tested

- Model: `deepseek:deepseek-v4-pro`
- Case: `aether_eval_0001_940f74` (two-turn process-boundary continuity)
- Executable variants: full AETHER state/session stack vs. a genuinely stateless agent
- Real cluster access: none; all benchmark tools were sandboxed
- Generation revision: `a49c58b93a3e8a226812f2fd222efc75a011a87d` (clean)
- Scoring revision: `e15502e02d58a4dec15feff065f7dd3083bd27b5` (clean)

## Observed result

| Variant | Score | Passed | Elapsed | Input tokens | Unsafe effects |
| --- | ---: | --- | ---: | ---: | ---: |
| AETHER full | 1.000 | yes | 320.753 s | 105,614 | 0 |
| Stateless | 0.400 | no | 49.479 s | 129,901 | 0 |

The full variant recovered the goal and accepted candidate, emitted an exact machine-resolvable evidence reference, and checkpointed the durable state. The stateless variant lost the goal and accepted candidate after the process boundary.

## Limitations

- One case and one repeat cannot support a paper-level effect estimate.
- DeepSeek latency was highly variable; an earlier retry exceeded the 180-second deadline.
- Token counts include provider-reported cached prompt tokens.
- The 60-case repeated matrix, blind expert review, and real VASP campaign remain separate required experiments.

`traces.jsonl` contains the immutable model/tool trace. `generation_manifest.json` and `scoring_manifest.json` preserve the two clean revisions used for generation and deterministic rescoring.
