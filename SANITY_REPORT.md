# UniPilot Mini 実動確認レポート

実行日: 2026-08-15（Windows、完全ローカル）

## 環境

- CPU: Intel Core i7-9700F（8 cores / 8 threads）
- RAM: 15.9 GB
- GPU: NVIDIA GeForce RTX 2070 SUPER（Windows APIの報告値はVRAM 4.0 GB）
- PyTorch: 2.12.0+cpu、CUDA利用不可（現在の環境にはCPU版PyTorchが導入済み）
- Python 3.14.5 / Node.js 24.16.0

## モデルとTokenizer

- v0.1設定: Decoder-only Transformer、19,814,784 parameters、11 layers、6 heads、embedding 384、FFN 1536、context 256
- 実動sanityモデル: 128,768 parameters、2 layers、4 heads、embedding 64、FFN 256、context 64
- Tokenizer: 自作UTF-8 byte-level BPE
- 語彙数: sanity 384（特殊token 7個を含む）
- 重み: PyTorchの乱数から初期化。事前学習済み重みなし

## データ

- UniPilot用に自作した大学生活会話: 700件（14カテゴリ、各50件）
- 自作一般日本語文: 210件
- ライセンス: CC0-1.0
- 外部サイトから収集した文章: 0件

## 100-step sanity学習

- Device: CPU
- Training loss（100 step平均）: 4.421895
- Validation loss: 学習前 5.968301 → 学習後 3.161597
- 独立評価loss: 3.148535
- Perplexity: 23.301903
- Training speed: 20,527.83 tokens/sec
- 平均step time: 20.60 ms
- 平均generation speed: 417.17 tokens/sec（40 token、3 prompt）
- Checkpoint: `checkpoints/sanity-100/checkpoint-step-100.pt`
- Checkpoint size: 1,580,903 bytes（約1.51 MiB）
- モデルload時RSS増分: 約4.92 MiB（Pythonプロセス全体は約207 MiB）

生成例（100 stepのみのため品質は低い）:

```text
入力: 明日試験です
出力例: ことははにやるとずを分けったい。
```

日本語らしい断片の生成、EOS停止、checkpoint保存・再読込、CPU推論を確認しました。意味の整合した回答はまだ安定しません。

## 検証結果

- pytest: 7 passed
- Tokenizer: 日本語encode/decode完全往復、save/load成功
- Model: forward、shape、loss、causal mask成功
- Training: loss低下、checkpoint保存・再開情報の読込成功
- Inference: greedyおよびsampling生成成功
- FastAPI: 実サーバーを起動し `/health`、`/model-info`、`/chat` がHTTP 200
- Next.js: production build成功
- npm audit: 0 vulnerabilities
- 禁止依存scan: Transformers、OpenAI、Anthropic、Ollama等のimportなし
- 外部AI API通信: なし

## 現在できないこと

- 事実保証、大学固有制度の正確な回答、長文理解、複雑な推論
- 100-step sanityモデルによる自然で一貫した会話
- 現在のPyTorch環境でのCUDA学習（CUDA版PyTorchの導入が必要）
- v0.1 19.8Mモデルの本格学習（設定・コマンドは用意済みだが、長時間学習は未実行）

## 次の改善

1. 権利確認済みで重複の少ない自然な日本語コーパスを増やし、通常文の事前学習を行う。
2. 大学ごとの制度を混同しないデータ設計と、訓練に未使用の評価セットを追加する。
3. CUDA版PyTorchを導入し、v0.1を十分なstep数で学習して固定質問を比較する。
4. repetition、事実性、安全性、人手評価を継続記録する。

100Mへはembedding 640、16 layers、10 heads、FFN 2560付近を出発点に、語彙増加、gradient accumulation/checkpointing、より大規模なライセンス確認済みデータ、GPUメモリ増強が必要です。
