# UniPilot Mini v0.4 Training Report

Architectureは19,814,784 parameters、11 layers、6 heads、context 256、512 vocabのまま。v0.3 Stage B checkpointからClean Stage Cをassistant-only lossで学習し、EOS weight 1.5を採用した。2,000 stepで停止し、それ以上は実行していない。

| Step | train loss | weighted val loss | Generation val loss | PPL | Training tokens/s | RAM MB |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 0.5813 | 0.5099 | 0.4614 | 1.59 | 116.12* | 897.20* |
| 1,000 | 0.1774 | 0.2581 | 0.2082 | 1.23 | 313.90 | 751.38 |
| 2,000 | 0.1905 | 0.1126 | 0.1016 | 1.11 | 280.88 | 781.42 |

`*` 500 stepは3実験並列の競合を含む。NaN 0、Inf 0。最終checkpointは237,946,605 bytes（約226.9MiB）。各checkpoint manifestにexperiment ID、base checkpoint、dataset/evaluation version、EOS weight、step、seed、generation設定、Git commitを保存した。
