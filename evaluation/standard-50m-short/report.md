# UniPilot Standard 50M Short Validation

Campus v2.3と本番v0.4を保持し、外部AI/APIなしで短時間検証だけを実施した。

## 結果

- TOEIC既知重大誤回答: 3 -> 0
- 採用候補: 45,445,120 parameters / vocab 2048 / context 512
- 100step: train loss 6.6906, validation loss 6.1171, 可視文章 0/8、Gate FAIL
- 500step: 未実行（100step文章成立条件を満たさないため）

## Independent Blind 200

| Axis | Campus v2.3 | Standard step100 | Delta |
|---|---:|---:|---:|
| correctness | 93.00% | 93.00% | +0.00pt |
| relevance | 91.44% | 82.80% | -8.64pt |
| completeness | 91.00% | 71.00% | -20.00pt |
| specificity | 88.98% | 73.00% | -15.98pt |
| naturalness | 93.00% | 91.00% | -2.00pt |
| actionable | 92.40% | 64.00% | -28.40pt |

- Campus: critical 0, hallucination 0.00%, unsupported 0.00%, first response 0.070s, peak RSS 515.2MB
- Standard: raw generation success 0.00%, validator fallback 100.00%, first token 0.339s, 41.64 tok/s, peak RSS 603.9MB

## 判定

Standard 50M継続価値: **NO**。次の学習step: **なし（step100で停止）**。500/1000/2000/5000 step計画は作成しない。
