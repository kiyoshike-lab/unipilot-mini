# UniPilot Foundation v1.3 — Full Clean 250-Step Report

Clean Foundation v1.1 corpusとvocab 4096 tokenizerだけを使い、20M Baseをscratchから250step学習した。

## Training curve

| Step | Train loss | Validation | PPL | LR | Tokens | Corpus | tok/s | RAM MB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | — | 8.3745 | 4335.0 | 5.00e-06 | 0 | 0.0000% | — | 399.1 |
| 50 | 7.6483 | 7.4866 | 1783.9 | 9.65e-05 | 25,600 | 0.0766% | 888.1 | 868.8 |
| 100 | 7.2534 | 7.2180 | 1363.7 | 7.62e-05 | 51,200 | 0.1533% | 871.0 | 870.7 |
| 150 | 7.1990 | 7.1435 | 1265.9 | 4.64e-05 | 76,800 | 0.2299% | 874.9 | 870.7 |
| 200 | 7.0476 | 7.0932 | 1203.7 | 2.05e-05 | 102,400 | 0.3066% | 868.3 | 870.7 |
| 250 | 7.0528 | 7.0727 | 1179.3 | 1.00e-05 | 128,000 | 0.3832% | 855.2 | 870.7 |

## Generation (50 fixed completion prompts)

| Step | Mode | Valid | Natural | Semantic | Completion | EOS | Runaway | Repetition | L0/L1/L2/L3/L4/L5 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | greedy_no_penalty | 34% | 0% | 0% | 0% | 0% | 100% | 60.7% | 34%/0%/0%/0%/0%/0% |
| 0 | sampling_t07_topk40_topp09_no_penalty | 0% | 0% | 0% | 0% | 0% | 100% | 3.3% | 0%/0%/0%/0%/0%/0% |
| 50 | greedy_no_penalty | 100% | 0% | 0% | 0% | 0% | 100% | 98.4% | 100%/0%/0%/0%/0%/0% |
| 50 | sampling_t07_topk40_topp09_no_penalty | 100% | 8% | 8% | 2% | 0% | 100% | 21.0% | 100%/8%/8%/8%/0%/0% |
| 100 | greedy_no_penalty | 100% | 0% | 0% | 0% | 0% | 100% | 98.4% | 100%/0%/0%/0%/0%/0% |
| 100 | sampling_t07_topk40_topp09_no_penalty | 98% | 4% | 4% | 4% | 0% | 100% | 5.5% | 98%/4%/4%/0%/0%/0% |
| 150 | greedy_no_penalty | 100% | 0% | 0% | 0% | 0% | 100% | 64.6% | 100%/0%/0%/0%/0%/0% |
| 150 | sampling_t07_topk40_topp09_no_penalty | 94% | 0% | 0% | 0% | 0% | 100% | 15.9% | 94%/0%/0%/0%/0%/0% |
| 200 | greedy_no_penalty | 98% | 0% | 0% | 0% | 0% | 100% | 21.6% | 98%/0%/0%/0%/0%/0% |
| 200 | sampling_t07_topk40_topp09_no_penalty | 98% | 0% | 0% | 0% | 0% | 100% | 1.0% | 98%/0%/0%/0%/0%/0% |
| 250 | greedy_no_penalty | 98% | 0% | 0% | 0% | 0% | 100% | 29.5% | 98%/0%/0%/0%/0%/0% |
| 250 | sampling_t07_topk40_topp09_no_penalty | 92% | 0% | 0% | 0% | 0% | 100% | 14.0% | 92%/0%/0%/0%/0%/0% |

Base評価はrepetition penaltyなし。samplingはtemperature 0.7 / top-k 40 / top-p 0.9。

## Step 250 sampling observation

| Mode | Valid | Natural | Semantic | Completion | EOS | Runaway | Repetition |
|---|---:|---:|---:|---:|---:|---:|---:|
| sampling_t07_untruncated_no_penalty | 8% | 0% | 0% | 0% | 0% | 100% | 30.9% |
| sampling_t10_untruncated_no_penalty | 0% | 0% | 0% | 0% | 0% | 100% | 0.2% |
| sampling_t10_topk40_topp09_no_penalty | 94% | 8% | 8% | 4% | 0% | 100% | 37.5% |

## EOS / Base knowledge probes

| Step | EOS probability after document | Knowledge keyword Greedy/Sampling | Natural Greedy/Sampling |
|---:|---:|---:|---:|
| 0 | 0.000137 | 10%/10% | 0%/0% |
| 50 | 0.000087 | 0%/0% | 0%/0% |
| 100 | 0.000134 | 0%/0% | 0%/20% |
| 150 | 0.000225 | 0%/0% | 0%/0% |
| 200 | 0.000179 | 0%/0% | 0%/0% |
| 250 | 0.000201 | 0%/0% | 0%/10% |

## Findings

- Best validation: step 250 / 7.072652
- Step 250 processed: 128,000 tokens / 0.3832% / 0.003832 epoch
- 日本語の立ち上がり: NO
- 意味文の増加: NO
- Step 100から改善: NO
- Validation gap at 250: 0.019883

## Gate

- 250step Gate: **INVESTIGATE**
- 500stepへ進むべきか: **NO**
- Corpus拡張が今必要か: **NO**
- Model変更が必要か: **NO**
- 理由: lossは下降したが、Level 1〜2の増加、step 100比改善、または文字健全性のいずれかが250step継続条件に届かなかった。

## Integrity / protection

- Checkpoints: PASS
- Resume: PASS
- Tokenizer roundtrip: PASS
- Final Blind SHA256: `fa7912d58ce251bb10b513f59793bb8ca6c0023b4fe08d1c040b8ccbfe49845b`（内容未使用）
- 500/1000step、46M、Campus pretraining、instruction tuning、DPOは未実行。
- Production v0.4、Campus v2.3、Render、Vercel、Releaseは変更していない。
