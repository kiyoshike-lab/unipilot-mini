# UniPilot Foundation v1.2 Training Dynamics Investigation

外部LLM/API、Final Blind、Production/Campus/Webには触れず、20M Foundationの学習核だけを検証した。

## Core audits

- Random baseline: 8.317766
- Step 100 validation loss / perplexity: 7.165496 / 1294.00
- Random baseline比改善: 13.85%
- Causal shift: PASS
- Causal mask: PASS
- Loss: PASS
- Gradient health: PASS
- Weight update: PASS
- EOS sanity: PASS

## Tiny overfit

- 1_document: step 50, train loss 0.2332, exact continuation 100.0%, EOS 100.0%
- 10_documents: step 400, train loss 0.1178, exact continuation 82.2%, EOS 100.0%
- 100_documents: step 300, train loss 6.6395, exact continuation 4.6%, EOS 0.0%

## Learning-rate sweep

- 3e-05: train 7.4600, validation 7.5301, grad 1.6745, diverged False
- 1e-04: train 6.9951, validation 7.1389, grad 1.4062, diverged False
- 3e-04: train 6.9598, validation 7.1574, grad 1.3639, diverged False
- 6e-04: train 6.9593, validation 7.1796, grad 1.3327, diverged False

## Tokenizer short training

- vocab 2048: parameters 19,512,828, validation 6.6578, normalized improvement 12.68%, 549.6 tok/s, RAM 1188.7MB
- vocab 4096: parameters 19,514,880, validation 7.1218, normalized improvement 14.38%, 566.5 tok/s, RAM 1170.0MB
- Recommended vocab: 4096

## Token frequency / decoding audit

- EOS: 10,012 / 33,402,759 (0.02997%)
- Original 51,200-token runの期待EOS観測数: 15.35
- Used vocabulary: 4,028 / 4,096
- Weight tying: True
- Clean step 100 decoding（Level 1 / repetition）:
  - greedy_no_penalty: 0% / 98.4%
  - temperature_0.7_untruncated: 10% / 53.0%
  - temperature_1.0_untruncated: 0% / 29.1%
  - temperature_0.7_topk40_topp0.9: 30% / 23.2%
  - temperature_0.8_topk40_penalty1.1: 70% / 64.2%
- Greedy repetitionはcontext 16/64/128でも解消せず、samplingで低下するが、100stepではLevel 2〜3が成立しない。主因は未学習であり、decoding調整だけでは代替しない。

## Verification

- Resume: PASS
- Tokenizer roundtrip: PASS
- Checkpoint integrity: PASS
- Final Blind SHA256: `fa7912d58ce251bb10b513f59793bb8ca6c0023b4fe08d1c040b8ccbfe49845b`（内容未使用）

## Decision

- TRAINING CORE: **PASS**
- Best LR: 0.0001
- Full Clean 250step: **YES**（未実行）
- Corpus追加: **NO**
- Architecture変更: **NO**
- 500step、46M、Final Blindは未実行。
