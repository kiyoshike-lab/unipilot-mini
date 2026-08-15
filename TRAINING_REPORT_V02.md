# TRAINING REPORT V0.2

## 実行環境

- CPU: Intel Core i7-9700F、8 cores / 8 threads
- RAM: 15.9GB
- GPU: NVIDIA GeForce RTX 2070 SUPER
- PyTorch: 2.12.0+cpu（CUDA unavailableのためCPU学習）
- 実行batch size: 1、gradient accumulation: 1
- Assistant-response loss masking: ON
- Optimizer: AdamW、weight decay 0.1、gradient clip 1.0
- 初期learning rate: 3e-4、warmup + staged cosine decay
- Seed: 42

## モデル

- Decoder-only Transformer
- Parameters: 19,814,784
- Layers: 11 / Heads: 6 / Context: 256
- Embedding: 384 / FFN: 1536 / Dropout: 0.1
- Tokenizer: v0.2 Byte-level BPE 512語彙
- 重み: 完全なランダム初期化（学習済みモデル不使用）

## 段階学習結果

| Stage | Train loss | Validation loss | PPL | Final step time | Training tokens/sec | Memory | Checkpoint size |
|---|---:|---:|---:|---:|---:|---:|---:|
| v0.2-100 | 5.4025 | 5.5939 | 268.78 | 618.74ms | 111.52 | 635MB | 237,945,159 bytes |
| v0.2-500 | 2.7204 | 2.7265 | 15.28 | 441.46ms | 552.72 | 713MB | 237,945,159 bytes |
| v0.2-1000 | 2.2731 | 2.4060 | 11.09 | 429.91ms | 537.32 | 744MB | 237,945,709 bytes |

1000 step時点でもValidation lossは悪化しておらず、明白な過学習は検出されませんでした。tokens/secはAssistant mask後のloss対象token数を分子とするため、sample長によって変動します。

各段階は前段のmodel/optimizer stateから再開しました。再開時は保存learning rateから次段階末へcosine decayし、learning rateの跳ね上がりを防いでいます。1000 stepを超える長時間学習は実行していません。
