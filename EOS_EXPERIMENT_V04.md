# EOS Weight Experiment v0.4

全実験はv0.3 Stage B step 2,000、同一Clean Stage C、seed 42、500 step、max new tokens 64で実施した。

| EOS weight | Loss | EOS | Runaway | Meaningful | Keyword | Category | JP | Repetition | Avg chars | Too short |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.4615 | 84.67% | 15.33% | 26.33% | 12.78% | 11.33% | 98.98% | 0.73% | 38.07 | 0% |
| 1.5 | 0.4614 | 84.33% | 15.67% | **28.00%** | 11.33% | 10.00% | 99.18% | 0.89% | 38.22 | 0% |
| 2.0 | 0.4616 | 81.00% | 19.00% | 26.67% | 12.11% | 11.33% | 99.20% | 0.77% | 37.81 | 0% |

Meaningfulを最優先して1.5を選択した。2.0はEOSを改善せず、1.0はMeaningfulが低かった。全weightでToo-short 0%のため過剰EOSは観測されなかった。
