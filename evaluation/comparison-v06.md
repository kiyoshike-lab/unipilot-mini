# UniPilot Mini v0.6 evaluation

All quality rows below use the same 300 held-out prompts, deterministic generation, 64 generated-token cap, and the transparent 12-dimension proxy in `evaluate_v06.py`. Automated scores are not human scores.

| Metric | v0.4 production | v0.5 replay-500 | v0.6 best-2000 |
|---|---:|---:|---:|
| Parameters | 19,814,784 | 19,814,784 | 19,814,784 |
| Vocabulary | 512 | 512 | 512 |
| Context | 256 | 256 | 256 |
| Conversation/instruction rows | 8,000 | 149 | 798 |
| v0.6 validation loss | 6.1157 | 4.2144 | 4.2036 |
| Natural Japanese pass | 99.67% | 96.67% | 98.33% |
| Relevance pass | 8.67% | 4.00% | 10.33% |
| Accuracy proxy pass | 4.67% | 4.00% | 10.33% |
| Category pass | 47.33% | 52.67% | 54.33% |
| Keyword hit | 46.00% | 50.67% | 51.67% |
| Hallucination proxy | 27.33% | 1.67% | 0.00% |
| Unnecessary information | 27.33% | 9.00% | 0.00% |
| Completion | 46.00% | 78.67% | 91.00% |
| EOS | 38.67% | 76.00% | 91.67% |
| Runaway | 61.33% | 24.00% | 8.33% |
| Repetition | 0.08% | 1.86% | 0.21% |
| Automated overall / 5 | 3.417 | 3.818 | 3.952 |
| Human spot check / 5 | not rerun | not rerun | 1.6 (10 items) |
| Local optimized mean tok/s | 57.08 | 57.37 | 57.25 |
| Local first-token probe | 0.054 s | 0.054 s | 0.053 s |
| Local 20-token probe | 0.247 s | 0.243 s | 0.249 s |
| Local 64-token probe | 1.857 s | 1.909 s | 1.874 s |
| Local peak RSS | 311.66 MB | 322.33 MB | 310.40 MB |
| Inference checkpoint | 75.64 MB | 75.64 MB | 75.64 MB |

## Decision

The automatic best v0.6 checkpoint is step 2000. It passes naturalness, completion, unnecessary-information, repetition, and EOS targets, but fails relevance (10.33% vs 85%), keyword (51.67% vs 80%), category (54.33% vs 80%), and runaway (8.33% vs 1%). The representative human audit also fails 4/5. Production therefore remains v0.4; no API, Render, Vercel, or production checkpoint reference was changed.

v0.6 is safer than v0.4 but often selects a fluent answer template from the wrong category. For example, it answers GPA with attendance guidance and an email request with an exam plan. This is far from ChatGPT/Gemini-level semantic routing and generalization. No ChatGPT/Gemini output was fetched or used.

## Candidate conclusions

- v0.5 failure: 149 new rows, 20% v0.4 replay, learning rate 1e-5, and clipping on nearly every update caused a narrow distribution/style shift without enough ability replay. On the common 300-question suite, v0.5 relevance is 4.0%, below v0.4's 8.67%, despite its lower validation loss. Lower loss was therefore an overfitting signal, not a production-quality signal.
- Model size sanity: 20M/50.8M/99.0M/199.1M all complete one optimizer step. Estimated FP32 inference weights are 75.6/193.9/377.5/759.3 MB; measured training RSS is 606/1,097/1,840/3,408 MB. Only 20M has a realistic Render Free 512 MB inference envelope.
- Vocabulary: 1024 reduces held-out tokens/character from 1.188 to 0.871; 2048 reaches 0.824. The 4096 request reaches only 2812 tokens on the deliberately small reviewed corpus and gives no useful compression beyond 2048. Mini stays at 512 for exact v0.4 compatibility. A future Standard scratch model can test 1024.
- Context: 256 fits 84% of measured sequences; 512 fits 100% and costs only 0.375 MB in position weights, but attention score memory is 4x and the v0.4 position checkpoint is not shape-compatible. Mini stays at 256; Standard should test 512.
- Dataset: the reviewed core is 798 rows, not the 5,000 target. It contains 300 negative-to-corrected examples and 130 subject-genericized, answer-deduplicated v0.4 replay rows. Mechanical duplication was rejected. Reaching 5,000 requires new reviewed semantic cases, especially category boundaries, not paraphrase padding.
- Curriculum: A filtered/generalized replay, B stable knowledge instruction, C direct instruction, D multi-constraint, E hallucination correction. At the final mix A remains 55% to limit forgetting.
- Knowledge/RAG: 18 attributed CC BY-SA public articles stay separate from instruction data. Local BM25 gets 6/6 top-3 hits on a tiny offline probe. It is not production-enabled and must not answer current university rules without current official verification.
- Mini/Standard: Mini remains 19.8M, vocab 512, context 256. Standard candidate is about 50.8M, vocab 1024, context 512, but is not feasible on the current 512 MB Render service without a different deployment tier and a fresh quality training run.

## Next bounded experiment

Author and review category-balanced cases until each target category has at least 100 semantically distinct answers; add explicit near-neighbor routing contrasts (GPA vs credits vs attendance, email vs report, citation vs copyright); keep genericized replay; measure unclipped gradient distributions before choosing a clip threshold; then rerun only the 500/1000/2000 gates. Do not train 5000/10000 on the current 798-row core.
