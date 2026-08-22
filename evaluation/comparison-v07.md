# UniPilot Mini v0.7 evaluation

v0.7 implements an opt-in, fully local five-stage path: 21-category BM25 classification, local BM25 retrieval, bounded `<CONTEXT>` prompt construction, the 19,814,784-parameter Mini model, and a deterministic answer validator. No external AI API, external pretrained model, or scraped training corpus is used. The production default remains v0.4; v0.7 only activates when `UNIPILOT_PIPELINE_VERSION=v0.7` is explicitly set.

## Same-300 comparison

The raw v0.4 and v0.6 rows reproduce the existing common 300-prompt evaluation. The RAG rows use the mapped version of those same 300 prompts. These are deterministic proxy metrics, not blind human scores.

| Metric | A: v0.4 raw | B: v0.6 raw | C: v0.4 + BM25 | D: v0.7 + BM25 |
|---|---:|---:|---:|---:|
| Parameters | 19,814,784 | 19,814,784 | 19,814,784 | 19,814,784 |
| Vocabulary / context | 512 / 256 | 512 / 256 | 512 / 256 | 512 / 256 |
| Dataset | v0.4 clean 8,000 | v0.6 reviewed 798 | v0.4 + v0.7 knowledge | v0.7 2,100 direct + 840 corrected |
| Category accuracy | 47.33% | 54.33% | 100.00% | 100.00% |
| Relevance | 8.67% | 10.33% | 96.00% | 96.00% |
| Correctness | 4.67% | 10.33% | 96.00% | 96.00% |
| Hallucination | 27.33% | 0.00% | 0.00% | 0.00% |
| Completion | 46.00% | 91.00% | 100.00% | 100.00% |
| Effective EOS | 38.67% | 91.67% | 100.00% | 100.00% |
| Natural Japanese | 99.67% | 98.33% | 100.00% | 100.00% |
| Human spot check | not rerun | 1.6/5 | not scored | 4.6/5 (10, non-blind) |
| Local raw generation | 57.08 tok/s | 57.25 tok/s | 57.08 tok/s | 52.94 tok/s |
| Local first-token probe | 0.054 s | 0.053 s | 0.054 s | 0.053 s |
| Local peak RSS | 311.66 MB | 310.40 MB | 290.20 MB fast path | 321.70 MB raw generation |
| Inference checkpoint | 75.64 MB | 75.64 MB | v0.4, 75.64 MB | v0.7, 75.64 MB |

The v0.7 automatic target gate passes only on FAQ-covered questions where retrieval supplies a reviewed final answer. The 10-item human score is a representative, non-blinded spot check of those reviewed FAQ answers; the generated 300-item human form is deliberately left unscored. It must not be represented as a full independent human evaluation.

## Classifier and retrieval selection

| Classifier | Accuracy | Mean latency |
|---|---:|---:|
| Rules | 84.00% | 0.013 ms |
| BM25 | 100.00% | 1.704 ms |
| character TF-IDF | 97.67% | 0.194 ms |
| Hybrid | 95.67% | 0.218 ms |

BM25 is selected because it has the highest same-300 accuracy and remains well below model-generation latency. The result is optimistic: classifier training examples and the fixed evaluation share the same 84 reviewed semantic seed families, so a truly novel blind set is still required.

Retrieval top-k 1, 3, and 5 all obtain 100% category-document hit and 96% expected-keyword hit. Top-k 1 is selected because added documents provide no quality gain and consume the 256-token context budget.

## Dataset and knowledge audit

- 420 natural FAQ question phrasings from 84 manually reviewed semantic answer seeds.
- 2,100 direct-answer rows, split by semantic family with no family leakage.
- 840 corrected rows: 140 each for wrong category, invented subject, unrelated advice, university-specific hallucination, incomplete answer, and excessive generic response.
- 444 knowledge documents: 420 reviewed FAQ documents, 18 attributed Wikipedia documents, and 6 project-authored summaries pointing to official sources.
- All knowledge rows include `id`, `title`, `text`, `category`, `source`, `source_url`, `license`, and `retrieved_at`.
- The FAQ has 336 repeated answer texts because five question phrasings share one reviewed answer. Pair duplicates are zero. Thus 2,100 is the training-row count, not 2,100 semantically independent answers.

Wikipedia content retains CC BY-SA 4.0 attribution. Official-source entries are original CC0 summaries and do not copy source text. The source URLs point to the [Ministry of Education syllabus material](https://www.mext.go.jp/a_menu/koutou/daigaku/04052801/003.htm), [MEXT study-support information](https://www.mext.go.jp/kyufu/qa/qa_university.html), [Ministry of Health student part-time guidance](https://www.check-roudou.mhlw.go.jp/parttime/), [JASSO scholarship guidance](https://www.jasso.go.jp/shogakukin/saiyochu/siori/index.html), and the [Digital Agency open-data terms](https://www.digital.go.jp/resources/open_data). No indiscriminate scraping was performed.

## Context, candidate, and validator experiments

Full retrieved prompts average 270 tokens (p95 302). Context 256 fits 24% without truncation; 512 and 1024 fit 100%. Relative attention memory is 1x / 4x / 16x, while learned-position parameters add approximately 0 / 0.375 / 1.125 MB. Mini remains at 256 for v0.4 checkpoint compatibility and the 512 MB deployment envelope; it explicitly truncates retrieved context while reserving 48 answer tokens. A future Standard model should use 512.

On 21 category-balanced prompts, 1 / 2 / 3 model candidates take 0.139 / 0.268 / 0.394 seconds per question and produce the same grounded-selection and fallback result. One candidate is selected; multiple generation is rejected because latency grows linearly without measured quality gain.

The validator is material, but the ablation also exposes the model limit:

| v0.7 condition | Relevance | Correctness | Completion | Natural | Mean time | Raw tok/s |
|---|---:|---:|---:|---:|---:|---:|
| FAQ fast path + validator | 96.00% | 96.00% | 100.00% | 100.00% | 0.0039 s | n/a |
| Forced model + validator/selection | 96.00% | 96.00% | 99.67% | 100.00% | 0.3535 s | 21.08 |
| Forced model, no validator/grounded selection | 3.33% | 1.00% | 30.33% | 33.67% | 0.3375 s | 21.54 |

Fallback use is 0% on the same 300 because reviewed grounded answers validate successfully. For retrieval misses or invalid candidates, category-specific safe fallbacks remain available; the implementation does not route every answer to a fallback.

## Training and performance

v0.7 continues the project-owned v0.4 step-2000 checkpoint for 500 steps with learning rate 1e-6, 30% generalized v0.6 replay, EOS weight 1.5, and gradient clipping at 5.0. Final training loss is 3.7377, validation loss 5.5634, with no NaN/Inf. Training stopped at the bounded 500-step checkpoint because the trained model gives no quality advantage over v0.4 + retrieval.

The inference-only checkpoint is 75.6393 MiB and removes `optimizer_state`. Its SHA-256 is `59eebc29d191428b001d5a9e0925602cdce55954e64fc90f4d7373f9982d446e`. Raw local generation averages 52.94 tok/s, with 0.053 s first token and 321.70 MB peak RSS. The 20-token probe takes 0.295 s and the 64-token probe 1.870 s, although raw long-form text remains incoherent. Context-forced generation averages about 21 tok/s; the reviewed-FAQ fast path returns in about 3.9 ms because it validates and returns retrieved final text without model generation.

No v0.7 Render deployment or production switch was made, so there is no honest v0.7 Render Free measurement. Reporting a local number as Render performance would be misleading.

## Capacity and product decision

Within FAQ-covered university-life questions, retrieval makes answers direct and reliable enough to approach the useful shape of a large assistant. It does not approach ChatGPT/Gemini on novel paraphrases, compositional questions, reasoning, general knowledge, or unsupported facts. No ChatGPT/Gemini output was fetched for training or evaluation.

The no-validator ablation, incoherent raw 64-token output, and lack of improvement over the v0.4 base show that the 20M generator is a major constraint outside reviewed retrieval coverage. The product split should therefore be:

- **UniPilot Mini:** 19.8M, vocab 512, context 256, retrieval-first, one candidate, suitable for the current 512 MB service.
- **UniPilot Standard:** initially 50M, then at most 100M if justified; vocab 1024, context 512, a deployment tier above Render Free, and a fresh tokenizer/checkpoint rather than an in-place Mini shape change.

Before another training run, expand from 84 to at least 2,100 semantically distinct reviewed answers (100 per category), retain at least 1,000 category-boundary corrections, and create a separately authored blind evaluation set. Add near-neighbor contrasts such as GPA/credit/attendance, email/report, citation/copyright, and exam/study. Do not count mechanical paraphrases as new knowledge.

## Promotion decision

**REJECT PRODUCTION PROMOTION.** D meets the automated same-300 thresholds but is not clearly better than C: v0.4 + BM25 has the same 96% relevance/correctness and slightly lower latency in this run. The v0.7 model itself is worse than the retrieved answer and adds no demonstrated production value. Production remains v0.4, v0.7 is opt-in only, no Render/Vercel setting or checkpoint reference is changed, and no push is performed.
