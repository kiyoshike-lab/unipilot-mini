# UniPilot Mini

UniPilot Miniは、大学生活に関する短い日本語対話を目的にしたDecoder-only言語モデルです。OpenAIなどの外部AI APIや、他社の学習済みLLMを呼び出すアプリではありません。独自Tokenizer、ランダム初期化したTransformer、自作データを使い、ローカルPCだけで学習・推論します。

> v0.1は教育・研究用の小型モデルです。ChatGPT級の知識や正確性はなく、重要な履修・健康・制度上の判断は必ず大学の公式情報で確認してください。

## 「ゼロから」の定義

- UTF-8 byte-level BPE Tokenizerをこのリポジトリで実装し、データからmergeを学習
- Multi-Head Attention、causal mask、FFN、残差接続をPyTorchで直接実装
- 重みはランダム初期化し、同梱または利用者が許可したデータだけから学習
- Hugging Face Transformers、既存LLM、事前学習済み重みは不使用
- PyTorch、NumPyなどの数学・GPU計算用ライブラリは使用

使用していないもの: OpenAI / ChatGPT API / Claude / Gemini / Llama / Qwen / Gemma / Phi / Mistral / Ollama / LM Studio / その他の学習済みLLM・外部推論サービス。

## 構成

`model/` はTransformer、`tokenizer/` は独自BPE、`training/` はnext-token学習、`inference/` は生成、`evaluation/` はloss/perplexity/固定質問、`api/` はFastAPI、`web/` はNext.js UIです。`data/SOURCES.md` にデータの出典・ライセンスを記録します。

v0.1設定は context 256、embedding 384、11 layers、6 heads、FFN 1536で約20M parametersです。`configs/sanity.json` はCPU確認用の約0.12Mモデルです。両方ともTokenizerの実語彙数を起動時に反映します。

## Windowsセットアップ

Python 3.11〜3.13を推奨します。PowerShellで次を実行してください。

```powershell
cd path\to\unipilot-mini
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
cd web
npm install
cd ..
```

NVIDIA GPUを使う場合は、PyTorch公式手順に従って環境に合うCUDA版PyTorchを入れてください。`torch.cuda.is_available()` が `True` なら `--device auto --mixed-precision` で自動利用します。現在のCPU版でもsanityモデルは動きます。

## 最短の動作確認

```powershell
.\train-sanity.bat
python -m inference.generate --checkpoint checkpoints/sanity-100/checkpoint-step-100.pt --prompt "明日試験です" --max-new-tokens 40
python -m evaluation.evaluate --checkpoint checkpoints/sanity-100/checkpoint-step-100.pt --dataset data/conversations
pytest -q
```

学習は入力を1 tokenずらしたCrossEntropyLoss（paddingは `-100` で無視）です。AdamW、warmup + cosine decay、gradient clipping、CUDA mixed precision、validation split、CSV履歴、resume可能なcheckpointを実装しています。

通常文で事前学習してから会話データへ追加学習する例です。2段階目は1段階目のcheckpointを `--resume` へ渡します。

```powershell
python -m training.train --config configs/v0.1.json --dataset data/general_japanese --epochs 5 --output-dir checkpoints/pretrain
python -m training.train --config configs/v0.1.json --dataset data/conversations --epochs 10 --resume checkpoints/pretrain/checkpoint-step-N.pt --output-dir checkpoints/v0.1
```

## v0.1本学習

`train-mini.bat` を実行します。16GB RAMではbatch size 2〜4から始めてください。RTX 2070 SUPERでもVRAMの実容量・CUDA環境によっては不足するため、その場合は `embedding_dim`、`n_layers`、`context_length`、batch sizeを下げます。長時間学習の前にsanity実行を完了してください。

個別実行例:

```powershell
python -m scripts.generate_dataset
python -m tokenizer.train_tokenizer --input "data/**/*.jsonl" "data/**/*.txt" --vocab-size 512
python -m training.train --config configs/v0.1.json --dataset data/conversations --epochs 10 --batch-size 4 --learning-rate 0.0003 --mixed-precision --output-dir checkpoints/v0.1
python -m training.train --config configs/v0.1.json --dataset data/conversations --epochs 20 --resume checkpoints/v0.1/checkpoint-step-1000.pt --output-dir checkpoints/v0.1
```

## 生成とチャット

```powershell
python -m inference.generate --checkpoint checkpoints/v0.1/checkpoint-step-N.pt --prompt "大学の課題が終わらない"
python chat.py --checkpoint checkpoints/v0.1/checkpoint-step-N.pt
```

生成はgreedy（`--temperature 0`）、temperature、top-k、top-p、repetition penalty、EOS、最大token数に対応します。対話を終えるには `exit`、`quit` または `終了` と入力します。

## Local APIとWeb UI

ターミナルを二つ開き、先に `.\start-api.bat`、次に `.\start-web.bat` を実行し、`http://localhost:3000` を開きます。APIは `127.0.0.1:8000` のみで待ち受けます。

- `GET /health`
- `GET /model-info`
- `POST /generate`
- `POST /chat`

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/chat -Method Post -ContentType application/json -Body '{"prompt":"明日試験です","max_new_tokens":40,"temperature":0.8}'
```

## データと権利

`scripts/generate_dataset.py` はこのプロジェクト用の短い会話700件と一般文210件を決定的に生成します。同梱内容はCC0-1.0です。スクレイピングした文章はありません。データを追加するときは、本人が作成・明示提供したもの、パブリックドメイン、または学習利用可能なライセンスのものに限り、`data/SOURCES.md` へ名称、作者/URL、ライセンス、取得日を記録してください。

会話テンプレート:

```text
<BOS><USER>
明日試験なんだけど何したらいい？
<ASSISTANT>
まず試験範囲を整理しましょう。
<EOS>
```

## 評価と可視化

`evaluation.evaluate` はvalidation loss、perplexity、固定6質問の生成結果と生成速度をJSON保存します。学習履歴にはstep timeとtokens/secも入ります。

```powershell
python -m evaluation.evaluate --checkpoint checkpoints/v0.1/checkpoint-step-N.pt --output evaluation/results-v0.1.json
python evaluation/plot_losses.py --history checkpoints/v0.1/training_history.csv --output evaluation/loss-v0.1.png
python -m scripts.benchmark --checkpoint checkpoints/v0.1/checkpoint-step-N.pt --history checkpoints/v0.1/training_history.csv
```

## 拡張

100Mへは設定だけでなく、学習データの質と量、validation分離、GPUメモリを増やす必要があります。例としてembedding 640、layers 16、heads 10、FFN 2560（語彙・weight tyingにより約85〜100M）を出発点にし、gradient accumulationとcheckpointingを追加します。300M/1Bでは単一の一般向けPCを超えるため、分散学習、より大規模で権利確認済みのコーパス、長期評価が必要です。

## 制約

小規模な合成データだけでは一般知識、事実性、長文理解、複雑な推論は身につきません。出力が不自然、反復的、または誤っている可能性があります。改善は学習済みモデルの流用ではなく、権利確認済みデータの拡充、重複削除、応答品質評価、モデル規模と学習stepの段階的増加で行います。
