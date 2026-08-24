# UniPilot Campus v2.1 evaluation

本番v0.4、Standard 50M、Render、Vercel、Releaseには変更を加えていない。外部AI/APIはOFF。
既存adversarial 300件とblind 2000件はtest専用で、学習・閾値選択には使用していない。
別作成のadversarial train 1,500件とvalidation 300件は分離し、validation accuracyは100%。

## Campus v2 vs v2.1 — same blind 2000

| Metric | Campus v2 | Campus v2.1 | Gate |
|---|---:|---:|---:|
| Determinate Category Accuracy | 99.82% | 99.82% | ≥97% |
| Ambiguous Handling Accuracy | 90.00% | 100.00% | ≥97% |
| Overall Routing Success | 98.35% | 99.85% | ≥95% |
| Action Accuracy | 92.40% | 99.35% | ≥95% |
| Adversarial Category Accuracy | 77.00% | 100.00% | ≥95% |
| Correctness | 96.20% | 99.20% | ≥92% |
| Relevance | 98.15% | 99.65% | ≥92% |
| Hallucination | 2.15% | 0.65% | ≤1% |
| Completion | 100.00% | 100.00% | ≥99% |
| Natural Japanese | 100.00% | 100.00% | ≥99% |
| Actionable | 4.668 | 4.676 | ≥4.5 |
| Retrieval Recall@1 | 75.47% | 97.06% | ≥90% |
| Retrieval Recall@3 | 77.59% | 97.06% | ≥95% |
| Retrieval MRR | 0.765 | 0.971 | ≥0.92 |
| False FAQ Match | — | 0.00% | ≤2% |
| Router P95 | 8.988 ms | 8.273 ms | <20 ms |
| FAQ P95 | 26.276 ms | 16.307 ms | <50 ms |
| Tool P95 | 29.705 ms | 16.905 ms | <50 ms |
| Total P95 | 26.599 ms | 15.204 ms | — |
| Peak RAM | — | 368.12 MB | <450 MB |

## Real Student Set 500

100件ずつ: very short / colloquial / correction / normal / compound。既存データとの正規化完全一致は0件。

| Metric | Campus v2 | Campus v2.1 |
|---|---:|---:|
| Category / Routing | 29.80% | 98.40% |
| Action | 33.60% | 96.40% |
| Correctness | 29.80% | 98.40% |
| Multi-intent Recall | 36.00% | 96.50% |
| Actionable | 4.597 | 4.624 |

## Retrieval — independent test 338

Selected: `category_filtered` / threshold `0.145` (validation only).

| Recall@1 | Recall@3 | MRR | false FAQ | P95 |
|---:|---:|---:|---:|---:|
| 97.06% | 97.06% | 0.971 | 0.00% | 3.532 ms |

Retrieval failures:

- `campus-v21-retrieval-test-match-0227` WRONG_FAQ: study_planの次の行動
- `campus-v21-retrieval-test-match-0132` FALSE_NO_MATCH: deadline_organizer、確認すること
- `campus-v21-retrieval-test-match-0125` WRONG_FAQ: gpaの次の行動
- `campus-v21-retrieval-test-match-0158` WRONG_FAQ: absence_emailの次の行動
- `campus-v21-retrieval-test-match-0015` FALSE_NO_MATCH: grade_simulator、確認すること
- `campus-v21-retrieval-test-match-0230` WRONG_FAQ: report_outlineの次の行動
- `campus-v21-retrieval-test-match-0124` WRONG_FAQ: lateness_emailってどう進める

## Remaining blind failures (top 10)

- `campus-v2-blind-colloquial-0045` OTHER: tuition → tuition / FAQ → FAQ / margin 1.591
- `campus-v2-blind-colloquial-0050` TOOL_SELECTION: general → general / CLARIFY → MODEL / margin 1.814
- `campus-v2-blind-colloquial-0101` TOOL_SELECTION: gpa → gpa / FAQ → TOOL / margin 1.984
- `campus-v2-blind-colloquial-0153` OTHER: report_outline → report_outline / TOOL → TOOL / margin 2.023
- `campus-v2-blind-colloquial-0325` OTHER: tuition → tuition / FAQ → FAQ / margin 1.591
- `campus-v2-blind-colloquial-0330` TOOL_SELECTION: general → general / CLARIFY → MODEL / margin 1.814
- `campus-v2-blind-colloquial-0381` TOOL_SELECTION: gpa → gpa / FAQ → TOOL / margin 1.984
- `campus-v2-blind-colloquial-0433` OTHER: report_outline → report_outline / TOOL → TOOL / margin 2.023
- `campus-v2-blind-normal-0016` TOOL_SELECTION: gpa → gpa / FAQ → TOOL / margin 1.505
- `campus-v2-blind-normal-0037` OTHER: tuition → tuition / FAQ → FAQ / margin 1.631

Failure reasons: AMBIGUOUS=0, NEGATION=2, CONTRAST=0, SHORT_QUERY=0, CATEGORY_COLLISION=0, RETRIEVAL_FAILURE=0, TOOL_SELECTION=6, MULTI_INTENT=3, UNKNOWN=0, OTHER=17

## Remaining adversarial failures (top 10)

- `campus-v2-adversarial-0044` NEGATION: gpa → gpa / FAQ → TOOL
- `campus-v2-adversarial-0078` CONTRAST: gpa → gpa / FAQ → TOOL
- `campus-v2-adversarial-0180` NEGATION: gpa → gpa / FAQ → TOOL
- `campus-v2-adversarial-0214` CONTRAST: gpa → gpa / FAQ → TOOL

## Correctness bottleneck

- routing/correct_route: n=1997, correct answer=99.35%
- routing/wrong_route: n=3, correct answer=0.00%
- retrieval/correct: n=710, correct answer=98.17%
- retrieval/wrong: n=4, correct answer=100.00%
- retrieval/no_retrieval: n=1211, correct answer=99.75%
- retrieval/not_applicable: n=75, correct answer=100.00%

Correctness低下16件の最大要因は、正しいroute後のanswer-level検証/hallucination 13件。route誤りは3件で、wrong retrieval 4件はこのtestでは直接のCorrectness低下を起こしていない。正しいroute時のanswer correctnessが99%超のため、Standard 50Mは現段階では不要。

## Gate / human evaluation

- Automatic gate: PASS
- Human 100: PENDING
- Final gate: STOP
- RAM peak: 368.12 MB (<450 MB)
- ChatGPT/Gemini比較: 外部APIを使わず、`/campus-v21-eval`で同一質問のUI結果を手入力する。
- Human gateは未採点を合格扱いしない。本番v0.4を維持する。

## Decision

自動ゲートは合格。Human 100が未完了のため総合ゲートはSTOP。Campus v2.1は本番昇格しない。
