# UniPilot Foundation v1.8 Reference Cross-Check Report

## 判定

- Architecture Gate: **SYNTHETIC_BENCHMARK_ISSUE**
- Full 256kへ進む: **NO**
- Depth-scaled init正式候補: **NO**
- 正式Foundation architecture: Currentのまま（本番・Campus・Final Blind内容は未変更）

## Reference architecture / fairness

Referenceは`torch.nn.MultiheadAttention(batch_first=True)`を使う独立decoderで、Pre-LN、Final LN、learned absolute position、GELU、causal mask、tied LM headです。custom attention/residual blockは共有していません。

| Model | Formal params (vocab4096/context512) | Synthetic params (vocab256/context80) | Layers | Hidden | Heads | FFN |
|---|---:|---:|---:|---:|---:|---:|
| Current | 19,514,880 | 17,874,432 | 10 | 384 | 6 | 1536 |
| Depth-init | 19,514,880 | 17,874,432 | 10 | 384 | 6 | 1536 |
| Reference MHA | 19,514,880 | 17,874,432 | 10 | 384 | 6 | 1536 |

Reference correctness: causal leakage、position sensitivity、gradient flow、tiny overfit、EOS sanity、parameter parityを全件PASS。

比較差分はattention実装（custom fused QKV/明示softmax 対 `nn.MultiheadAttention`）と、Currentだけのresidual出力初期値です。Depth-initとReferenceの初期値、dropout、bias、norm位置、residual式、embedding、position、tied LM headは一致させました。

## Synthetic LR pilot

| LR | Current | Depth-init | Reference | Cross-model mean |
|---:|---:|---:|---:|---:|
| 0.0003 | 0.1226 | 0.1242 | 0.1165 | 0.1211 |
| 0.0009 | 0.0717 | 0.0824 | 0.0817 | 0.0786 |
| 0.003 | 0.0036 | 0.0036 | 0.0036 | 0.0036 |

採用LRは3モデル共通の3e-4。AdamW、betas=(0.9,0.95)、eps=1e-8、weight decay=0.01、batch=16、clip=1.0で統一。これはcapacity診断専用であり、Foundation pretraining LRには転用しません。

## Synthetic learning curves

Key列はshort/medium/longの最小accuracy。chanceはpairsごとに50/25/12.5/6.25%。Copy/Position/Long/Numericは1/32=3.125%、Symbolicは1/4=25%、Contextは1/4=25%を目安とします。

| Model | Budget | Copy 4/8/16/32/64 | Key 2/4/8/16 | Numeric | Symbolic | Long | Context | Position min | Gate |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| Current | 0% | 0.0%/0.0%/0.0%/0.0%/0.0% | 0.0%/0.0%/0.0%/0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | FAIL |
| Current | 10% | 92.2%/10.9%/7.8%/0.0%/3.1% | 29.7%/14.1%/6.2%/6.2% | 1.6% | 26.6% | 100.0% | 100.0% | 10.9% | FAIL |
| Current | 25% | 100.0%/87.5%/7.8%/4.7%/9.4% | 29.7%/25.0%/4.7%/4.7% | 0.0% | 50.0% | 100.0% | 100.0% | 6.2% | FAIL |
| Current | 50% | 100.0%/100.0%/68.8%/14.1%/14.1% | 39.1%/18.8%/10.9%/3.1% | 17.2% | 100.0% | 100.0% | 100.0% | 57.8% | FAIL |
| Current | 100% | 100.0%/100.0%/96.9%/42.2%/31.2% | 45.3%/18.8%/7.8%/3.1% | 18.8% | 100.0% | 100.0% | 100.0% | 100.0% | FAIL |
| Current | 200% | 100.0%/100.0%/98.4%/53.1%/34.4% | 40.6%/15.6%/6.2%/1.6% | 35.9% | 100.0% | 100.0% | 100.0% | 100.0% | FAIL |
| Current | 400% | 100.0%/100.0%/100.0%/68.8%/40.6% | 40.6%/20.3%/6.2%/1.6% | 54.7% | 100.0% | 100.0% | 100.0% | 100.0% | FAIL |
| Depth-init | 0% | 0.0%/0.0%/0.0%/0.0%/0.0% | 0.0%/0.0%/0.0%/0.0% | 0.0% | 3.1% | 0.0% | 0.0% | 0.0% | FAIL |
| Depth-init | 10% | 18.8%/12.5%/10.9%/1.6%/7.8% | 26.6%/14.1%/9.4%/7.8% | 3.1% | 23.4% | 100.0% | 100.0% | 6.2% | FAIL |
| Depth-init | 25% | 100.0%/45.3%/7.8%/6.2%/3.1% | 21.9%/9.4%/0.0%/4.7% | 0.0% | 14.1% | 100.0% | 100.0% | 4.7% | FAIL |
| Depth-init | 50% | 100.0%/100.0%/57.8%/15.6%/7.8% | 40.6%/14.1%/9.4%/3.1% | 21.9% | 100.0% | 100.0% | 100.0% | 56.2% | FAIL |
| Depth-init | 100% | 100.0%/100.0%/95.3%/40.6%/34.4% | 31.2%/20.3%/6.2%/3.1% | 29.7% | 100.0% | 100.0% | 100.0% | 90.6% | FAIL |
| Depth-init | 200% | 100.0%/100.0%/100.0%/53.1%/37.5% | 35.9%/14.1%/9.4%/3.1% | 28.1% | 100.0% | 100.0% | 100.0% | 100.0% | FAIL |
| Depth-init | 400% | 100.0%/100.0%/100.0%/71.9%/48.4% | 45.3%/10.9%/7.8%/0.0% | 40.6% | 100.0% | 100.0% | 100.0% | 100.0% | FAIL |
| Reference MHA | 0% | 0.0%/0.0%/0.0%/0.0%/0.0% | 0.0%/0.0%/0.0%/0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | FAIL |
| Reference MHA | 10% | 15.6%/9.4%/6.2%/1.6%/3.1% | 25.0%/7.8%/6.2%/3.1% | 3.1% | 18.8% | 100.0% | 100.0% | 4.7% | FAIL |
| Reference MHA | 25% | 96.9%/53.1%/9.4%/4.7%/6.2% | 32.8%/17.2%/7.8%/3.1% | 0.0% | 17.2% | 100.0% | 100.0% | 1.6% | FAIL |
| Reference MHA | 50% | 100.0%/100.0%/62.5%/14.1%/20.3% | 40.6%/9.4%/7.8%/9.4% | 10.9% | 100.0% | 100.0% | 100.0% | 62.5% | FAIL |
| Reference MHA | 100% | 100.0%/100.0%/98.4%/50.0%/35.9% | 45.3%/18.8%/7.8%/3.1% | 17.2% | 100.0% | 100.0% | 100.0% | 96.9% | FAIL |
| Reference MHA | 200% | 100.0%/100.0%/100.0%/53.1%/37.5% | 35.9%/18.8%/12.5%/3.1% | 34.4% | 100.0% | 100.0% | 100.0% | 100.0% | FAIL |
| Reference MHA | 400% | 100.0%/100.0%/100.0%/87.5%/54.7% | 48.4%/17.2%/7.8%/3.1% | 50.0% | 100.0% | 100.0% | 100.0% | 100.0% | FAIL |

全cellのloss、Context shuffled/removed、Pattern全種、exact/template/leakage監査はsummary JSONに収録しています。

3モデルすべてでexact train/test overlap=0、answer leakage=0です。Templateはtask定義として意図的に共有し、token instanceは分離しています。

## Final attention audit（全layer/head平均）

| Model | Entropy | Max prob | Top-3 mass | Correct K+V mass | Correct rank | Q RMS | K RMS | scaled logit std | Margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Current | 0.0735 | 0.9025 | 0.9948 | 0.1339 | 10.49 | 3.3509 | 1.8821 | 31.5583 | -39.5343 |
| Depth-init | 0.0842 | 0.8817 | 0.9933 | 0.0800 | 9.94 | 3.4278 | 1.9900 | 32.3882 | -41.2010 |
| Reference MHA | 0.0894 | 0.8774 | 0.9908 | 0.1373 | 9.68 | 3.2257 | 1.9278 | 31.9977 | -36.9956 |

step 0〜400%の各layer/head値と、pairs×distance全12cellのfinal値はsummary JSON内の生データに保持しています。

最終attentionはuniformではなく強くselective（entropy 0.074〜0.089、max prob 0.877〜0.903）ですが、4-pair mediumの正解K+V massは0.080〜0.137（chance 0.071）、平均rank 9.68〜10.49、marginは全モデルで大幅な負値です。つまりattentionを絞れないのではなく、正しいkey/value関係へ選択を向けるsupervision/curriculumが不足しています。

## Japanese diagnostic（同一128k tokens）

| Model | Tokens | Loss | Top-1 | Top-5 | Top-10 | Correct prob | Punc mass | Context | Layer9 RMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Current | 0 | 8.4219 | 0.04% | 0.10% | 0.13% | 0.0002 | 0.00% | 13.1837 | 0.6522 |
| Current | 32,768 | 7.0778 | 8.02% | 13.29% | 17.29% | 0.0271 | 95.69% | 1.7840 | 3.3022 |
| Current | 65,536 | 6.9165 | 8.73% | 13.51% | 17.26% | 0.0411 | 95.73% | 1.7972 | 4.3965 |
| Current | 131,072 | 6.7512 | 9.70% | 15.06% | 18.49% | 0.0544 | 72.56% | 7.2796 | 3.5979 |
| Depth-init | 0 | 8.3962 | 0.00% | 0.09% | 0.16% | 0.0002 | 0.56% | 8.0553 | 0.1325 |
| Depth-init | 32,768 | 7.0286 | 8.31% | 13.35% | 17.44% | 0.0329 | 95.35% | 1.0140 | 1.5052 |
| Depth-init | 65,536 | 6.8260 | 9.31% | 14.55% | 18.51% | 0.0477 | 57.86% | 3.2242 | 1.5535 |
| Depth-init | 131,072 | 6.6565 | 10.55% | 15.91% | 19.52% | 0.0588 | 48.06% | 4.9486 | 1.6507 |
| Reference MHA | 0 | 8.4003 | 0.04% | 0.13% | 0.27% | 0.0002 | 1.66% | 7.5770 | 0.1376 |
| Reference MHA | 32,768 | 7.0264 | 8.23% | 13.61% | 17.58% | 0.0348 | 95.19% | 0.9420 | 1.5426 |
| Reference MHA | 65,536 | 6.8215 | 9.14% | 14.26% | 18.49% | 0.0481 | 61.41% | 3.6396 | 1.5953 |
| Reference MHA | 131,072 | 6.6399 | 10.14% | 15.92% | 19.71% | 0.0588 | 43.57% | 6.5794 | 1.6934 |

Frequency bucketごとのTop-1/5/10と正解token確率はsummary JSONに収録しています。

## 原因分析

- Custom、Depth-init、独立Referenceの全てが400%でも同じKey retrieval失敗を再現したため、custom attention固有のfatal defectではありません。
- Symbolicは3モデルとも100%へ収束した一方、atomic IDを使うnumericは40〜55%でした。Tokenizer分割ではなく、数列規則推定の難度とtask/variant当たりのsupervision密度の問題です。
- Copy 4/8/16、Long Range、Context control、Positionは全モデルでPASSし、context capacityそのものの欠如も否定されます。
- よって今回の失敗はarchitectureではなく、複数難度を1/6 task schedule内で疎に混ぜ、最終answer tokenだけを教師にしたSynthetic benchmark/training setupに帰属します。次はarchitectureを変えず、Key Lookup単独curriculumと中間関係supervisionを隔離検証すべきです。

## Integrity / protection

- Final Blind SHA256: `fa7912d58ce251bb10b513f59793bb8ca6c0023b4fe08d1c040b8ccbfe49845b` (content unopened, MATCH=True)
- Synthetic train/test exact overlap、answer leakageは各モデルreport参照。
- 3モデルのcheckpointはmodel/optimizer stateを含みstrict reload PASS。
- Full 256k、46M、Tokenizer、Campus、本番、push、deployは未実施。

