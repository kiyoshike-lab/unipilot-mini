# UniPilot Standard v0.8 evaluation

## Outcome

Standard v0.8 is implemented as a new, opt-in series. Mini checkpoints, the default API path, Render, Vercel, and GitHub Release configuration are unchanged. No external LLM API, external pretrained model, or external embedding API was used.

The bounded experiment stops at Stage A step 100. The 44.92M scratch model learns a natural opening phrase and EOS but scores only 0.57% relevance, 0% correctness, and 0% completion on the independent 528-question set. Training does not continue to 500 or later steps.

## Independent-blind comparison

All grounded rows below use the new 528-question blind set, not the v0.7 FAQ-family evaluation. The older 33-category labels are mapped to Mini's 21-category taxonomy where necessary.

| Metric | Mini v0.4 raw | Mini v0.4 Grounded | Mini v0.7 Grounded | Standard v0.8 A-100 |
|---|---:|---:|---:|---:|
| Parameters | 19,814,784 | 19,814,784 | 19,814,784 | 44,920,832 |
| Vocab / context | 512 / 256 | 512 / 256 | 512 / 256 | 1024 / 512 |
| Blind questions | 528 | 528 | 528 | 528 |
| Relevance | 15.91% | 53.98% | 53.98% | 0.57% |
| Correctness | 1.52% | 8.33% | 8.33% | 0.00% |
| Category accuracy | 10.23% mapped | 10.23% mapped | 10.23% mapped | 12.12% |
| Natural Japanese | 87.50% | 100.00% | 100.00% | 100.00% |
| Completion | 87.12% | 100.00% | 100.00% | 0.00% |
| Effective EOS | 53.79% | 99.81% | 98.48% | 100.00% |
| Hallucination proxy | 25.95% | 0.00% | 0.00% | 0.00% |
| Human score | not run | not run | not run | unscored 100-item form |
| Local raw generation | 57.08 tok/s | 57.08 tok/s | 52.94 tok/s | 48.12 tok/s |
| First token | 0.054 s | 0.054 s | 0.053 s | 0.079 s raw / 0.212 s pipeline mean |
| Peak RSS | 311.66 MiB | 311.66 MiB | 321.70 MiB | 408.92 MiB raw / 523.71 MiB full blind |
| Inference checkpoint | 75.64 MiB | 75.64 MiB | 75.64 MiB | 171.43 MiB |

The raw v0.4 control uses the same category router and prompt wrapper as the grounded run, but disables retrieval, validation, and fallback. Grounding raises relevance by 38.07 percentage points and removes the automatic hallucination flags on this blind set. Mini v0.4 and v0.7 then select the same reviewed v0.7 FAQ answer on 73.30% of blind questions, which explains their identical grounded relevance and correctness. Their old same-family 300-question scores did not generalize to new scenarios.

## Architecture selection

| Candidate | Structure | Parameters | Generation | First token | Training RSS | FP32 checkpoint | KV cache 512 |
|---|---|---:|---:|---:|---:|---:|---:|
| A | 14×512, 8 heads | 44.92M | 44.33 tok/s | 0.105 s | 1020.84 MiB | 171.41 MiB | 28.0 MiB |
| B | 16×512, 8 heads | 51.23M | 35.15 tok/s | 0.143 s | 1109.84 MiB | 195.47 MiB | 32.0 MiB |
| C | 14×576, 9 heads | 56.73M | 39.63 tok/s | 0.126 s | 1201.85 MiB | 216.45 MiB | 31.5 MiB |

Every candidate has a natural 64-dimensional head. Candidate A is selected because it is the fastest and smallest model while still providing more than twice Mini's capacity. There is no trained evidence yet that B or C improves quality enough to justify their cost.

Context 256/512/1024 full-forward probes take 0.271/0.635/1.543 seconds. FP32 KV cache is 14/28/56 MiB, and relative attention work is 1x/4x/16x. Context 512 is selected.

## Tokenizer

| Vocab | Tokens/Japanese character | Mean blind prompt+keypoint tokens | Mean university-term tokens | Probe generation |
|---|---:|---:|---:|---:|
| 512 | 1.269 | 140.92 | 9.67 | 358.43 tok/s |
| 1024 | 0.922 | 102.34 | 7.92 | 361.24 tok/s |
| 2048 | 0.733 | 81.36 | 6.33 | 319.39 tok/s |

Vocab 1024 is selected. It materially improves compression without the 2048 speed regression and adds only 1 MiB of tied embedding parameters over 512. The tokenizer is trained from project data and includes `<CONTEXT>` as a special token.

## Data audit

- 33 categories, 10 concerns, and 10 constraints create 3,300 semantic scenario cells.
- 9,900 instruction rows, 3,300 corrected rows, 6,000 conversation rows, 1,000 compound rows, and 2,000 general-Japanese rows.
- 774 knowledge documents: 330 Standard project documents plus 444 inherited v0.7 documents with their original source/license fields.
- 338 retrieval-development questions are used for method selection. Separately, 528 blind questions contain 132 each for simple, medium, hard, and compound; exact train-question and knowledge-title overlap is zero.
- 100 manual ChatGPT/Gemini comparison questions contain expected key points and a scoring rubric; no API calls are made.

These counts have an explicit limitation: concern/constraint inventories are independently authored, but rows are programmatically composed and have not received row-by-row human review. They are not 3,300 independently written answers.

Length modes and caps are implemented, but the sampled training answers miss their requested ranges: short/normal/detailed average 14.75/34.72/83.31 tokens versus targets 40–80/100–200/200–400. This is another long-training blocker.

## Retrieval

On the independent blind set:

| Method | Recall@1 | Recall@3 | MRR@10 | Category retrieval@1 | Classifier accuracy | Mean latency |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 8.14% | 15.53% | 14.47% | 14.02% | 12.12% | 22.51 ms |
| TF-IDF | 4.55% | 8.33% | 8.06% | 10.61% | 12.12% | 31.72 ms |
| Keyword | 10.98% | 18.56% | 17.21% | 11.74% | 12.12% | 24.64 ms |
| Hybrid | 8.14% | 15.72% | 13.96% | 14.58% | 12.12% | 33.90 ms |

Method selection uses a separate 338-question validation set: TF-IDF wins there with Recall@3 49.11%, so it is frozen before reading final blind metrics. Its blind Recall@3 then falls to 8.33%. Keyword happens to score higher on blind, but switching to it would contaminate the benchmark; the failure is retained.

## Training, performance, streaming, and INT8

Stage A step 100 reaches training loss 4.7358 and validation loss 2.8331 with no NaN/Inf, but clipping occurs on 100/100 updates. The corrected 528-question evaluation returns 0.57% relevance, 0% correctness/completion, and 100% fallback, so later stages are not run.

FP32 optimized inference averages 48.12 tok/s, first token is 0.079–0.148 seconds, and peak RSS is 408.92 MiB. End-to-end retrieval plus real incremental streaming has a 0.212-second mean first event and emits token snapshots before final validation. The failed candidate is never enabled in production.

Dynamic INT8 reduces serialized state size by 73.19% and preserves the single fixed-probe output, but it slows the generation probe to 68.26% of FP32 and does not reduce measured process RSS in the conversion process. It is not adopted; quality must first pass in FP32.

## Product decision

Standard v0.8 is not close to ChatGPT/Gemini quality in any of the eight requested university domains. No domain has a human score because the 100-question form is intentionally unscored. The current result cannot determine whether 50M capacity is sufficient: inadequate scratch pretraining, repetitive structured data, weak blind retrieval, and length mismatch dominate before model size can be isolated.

Do not move to 100M yet. First replace template-heavy rows with at least 30,000 individually reviewed instructions, 5,000 corrected examples, 10,000 conversations, and 3,000 genuinely compound cases; add a substantially larger, license-tracked Japanese language corpus; rewrite answer lengths; and build retrieval documents that cover new concepts without copying blind questions. Then retrain 45M through the 100/500/1000 gates. Compare 50–100M only after the 45M run demonstrates learning rather than memorized openings.

**Promotion decision: REJECT.** Production remains Mini v0.4. No push, Render/Vercel deployment, Release update, or production checkpoint switch is authorized or performed.
