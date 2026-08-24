# Campus v2.2 Generalization Report

AI Quality Gate is deterministic and does not replace Human Gate. Blind and stress sets are holdout-only.

- v2.1 -> v2.2 average: 85.66 -> 90.11
- v2.1 -> v2.2 good/close/bad: {'good': 20, 'close': 77, 'bad': 3} -> {'good': 74, 'close': 25, 'bad': 1}
- Blind 300 average: 91.13
- Stress 100 critical errors: 0
- Retrieval selected: reranked (R@1 0.77, R@3 0.8667, MRR 0.8111, false match 0.23)
- Knowledge: 518 sources / 1096 chunks
- Human-AI agreement: 0.7 -> 0.9
- ChatGPT gap proxy (no external comparison): {'directness': 91.0, 'completeness': 86.5, 'specificity': 89.14, 'actionability': 93.72, 'grounding': 94.84, 'student_tool_usefulness': 100.0}
- Remaining local-rubric gaps: ['completeness', 'specificity']
- Human review required: 16
- Standard 50M needed: YES
- Production Gate: FAIL
- Beta recommended: NO
- Production/Render/Vercel/Release changed: NO
- Automatic training: NO
