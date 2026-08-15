# UniPilot Mini v0.3 Training Report

## 実行結果

19,814,784 parameters、11 layers、6 heads、context 256、vocab 512を変更せず、scratchから5,000 stepを実行した。Stage Aを0–1,000、Stage Bを1,000–2,000、Stage Cを2,000–5,000にした。Stage Aのvalidation悪化とStage Bでのcatastrophic forgettingを観測したため、当初候補より早くStageを切り替え、BではAを15%、CではAを10%・Bを20% replayした。

| Step | Stage | train loss | stage val | general val | university val | conversation val | tokens/s | RAM MB |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | A | 0.3668 | 3.5816 | 3.6198 | 7.2600 | 7.0800 | 392.63 | 668.16 |
| 2,000 | B | 0.2763 | 0.2846 | 3.6964 | 0.2553 | 4.0867 | 695.06 | 784.24 |
| 2,500 | C | 0.7694 | 0.4433 | 3.5251 | 0.2255 | 0.5421 | 186.86 | 822.04 |
| 5,000 | C | 0.2099 | 0.1357 | 3.7348 | 0.1179 | 0.1540 | 280.81 | 838.65 |

ログ点平均は約0.25秒/step、468.04 training tokens/sec、最大CPU RAM 854.95MB。CUDA未使用のためVRAMは該当なし。最終checkpointは237,947,117 bytes（約226.9 MiB）。gradient clipping累計3,316、NaN 0、Inf 0。各checkpointにモデル、tokenizer、dataset、stage、step、seed、optimizer、scheduler、generation設定、Git commitを含むmanifestを保存した。

## 長期学習の見積り

0.25秒/stepの純学習時間からの概算で、10kは約42分、20kは約84分、30kは約126分、50kは約210分。5kからの追加分はそれぞれ約21、63、105、189分で、validation・checkpoint・300問生成の時間は別途必要である。長期用configと10k/20k/30k/50kのbatを用意したが、5kのEOS 16.33%、repetition 8.39%、日本語比率95.70%が停止条件に触れるため実行していない。

## Best checkpoint

validation lossと自動semantic relevanceはいずれも `stage-c/checkpoint-step-5000.pt`。Human scoreは未採点なのでbest human checkpointは未選択。最終選択はlossだけでなく、意味評価を優先した。
