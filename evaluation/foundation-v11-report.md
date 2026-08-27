# UniPilot Foundation v1.1 Clean Corpus Reconstruction

## Corpus: Dirty v1.0 vs Clean v1.1

| Metric | Dirty v1.0 | Clean v1.1 |
|---|---:|---:|
| Documents / unique | 11,499 | 10,315 |
| Characters | 50,209,324 | 46,233,086 |
| Tokens | 37,076,815 | 34,271,050 |
| Semantic duplicates excluded | 106 | 69 |
| Any markup residue documents | 3,114 | 0 |
| `[[` / `]]` residue documents | 72 / 2,830 | 0 / 0 |
| Template residue documents | 252 | 0 |
| HTML residue documents | 19 | 0 |
| Table residue documents | 4 | 0 |
| File/Image residue documents | 70 | 0 |
| Integration exclusions | 108 | 70 |

Clean split tokens: train 33,402,759, validation 466,818, test 401,473.

## Tokenizer

| Metric | Dirty v1.0 | Clean v1.1 |
|---|---:|---:|
| Selected vocab | 4096 | 4096 |
| Tokens / character | 0.743962 | 0.747081 |
| Roundtrip | 100.00% | 100.00% |
| Encoding chars/s | not recorded | 1,327,011 |
| Decoding tokens/s | not recorded | 5,293,254 |

Special IDs are PAD=0, BOS=1, EOS=2, UNK=3, USER=4, ASSISTANT=5, SYSTEM=6. Normal text never emitted EOS, and packed Train has exactly 10,012 BOS/EOS boundaries with no PAD/UNK/dialogue special tokens.

## Resume reproducibility

Scratch→40 and scratch→20→resume→40 are bitwise identical: loss, weights, optimizer, scheduler, and sampler all have maximum difference 0. RNG and sampler states are persisted. Resume Integrity: **PASS**.

## Dirty vs Clean 100 steps

| Metric | Dirty v1.0 | Clean v1.1 |
|---|---:|---:|
| Train loss | 7.2147 | 7.1717 |
| Validation loss | 7.2826 | 7.1655 |
| Tokens processed | 51,200 | 51,200 |
| Corpus fraction | 0.14% | 0.15% |
| Tokens/s | 878.44 | 868.46 |
| Peak RAM MB | 853.16 | 841.62 |

## Generation: fixed 50 prompts

| Greedy metric | Dirty v1.0 | Clean v1.1 |
|---|---:|---:|
| Character validity | 100.00% | 100.00% |
| Natural Japanese | 0.00% | 0.00% |
| Semantic coherence | 0.00% | 0.00% |
| Completion | 8.00% | 0.00% |
| EOS | 0.00% | 0.00% |
| Runaway | 100.00% | 100.00% |
| Repetition | 43.02% | 44.29% |

| Sampling metric | Dirty v1.0 | Clean v1.1 |
|---|---:|---:|
| Character validity | 100.00% | 58.00% |
| Natural Japanese | 60.00% | 36.00% |
| Semantic coherence | 16.00% | 10.00% |
| Completion | 34.00% | 44.00% |
| EOS | 0.00% | 0.00% |
| Runaway | 100.00% | 100.00% |
| Repetition | 21.71% | 34.48% |

## Gate

- Corpus Quality: **PASS**
- Tokenizer: **PASS**
- Resume Integrity: **PASS**
- Clean 100step: **INVESTIGATE**
- Clean 500step: **NO**

Loss and validation improve normally, but generation remains too immature for the next training stage. No 500-step run, Campus tuning, DPO, preference training, external AI, production change, push, or deploy was performed.
