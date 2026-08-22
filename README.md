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

## v0.2: データ・評価・段階学習

v0.2ではモデル構造を変えず、学習データを50,000件へ拡張しました。大学データ30,000件、一般日本語20,000件です。課題、試験、勉強計画、単位、教授メール、履修、出席、レポート、プレゼン、大学生活、予定管理へ分け、90/5/5でTrain/Validation/Testを分離しています。template familyごとにsplitを固定するため、同じ派生系列がtestへ漏れません。

```powershell
python -m scripts.generate_dataset_v02
python -m scripts.check_dataset
python -m tokenizer.train_tokenizer_v02
python -m scripts.analyze_tokenizer --tokenizer tokenizer/vocab-v02-512.json --output evaluation/tokenizer-analysis-v02-512.json
```

既存384語彙はsanity/v0.1 checkpoint互換用に保持しています。測定では384語彙が1.57 tokens/文字、追加512語彙が0.85 tokens/文字だったため、v0.2は512語彙を使用します。UNKはいずれも0です。1024/2048候補は `scripts/compare_tokenizers.py` で同条件比較できます。

### 学習stage

`training.train_v02` は通常文、大学説明文、会話を40/30/30でweighted samplingでき、会話では既定でAssistant回答部分だけをloss対象にします。

```powershell
# 段階実行
.\train-v02-100.bat
.\train-v02-500.bat
.\train-v02-1000.bat

# 個別stage（長時間学習向け）
python -m training.train_v02 --stage general --max-steps 1000 --max-records 0 --output-dir checkpoints/v02-general
python -m training.train_v02 --stage university --max-steps 2000 --max-records 0 --resume checkpoints/v02-general/checkpoint-step-1000.pt --output-dir checkpoints/v02-university
python -m training.train_v02 --stage conversation --max-steps 3000 --max-records 0 --resume checkpoints/v02-university/checkpoint-step-2000.pt --output-dir checkpoints/v02-conversation
```

`--max-records 0` は45,000件すべてを使います。既定の段階確認では初期化時間を抑えるため最大2,000 recordを読みます。CUDA OOM時はエラーに安全なbatch sizeを表示します。RTX 2070 SUPERではCUDA版PyTorchを導入し、batch 1〜2とgradient accumulation 4〜8を出発点にしてください。

### 固定評価と比較

固定評価300問は学習に使いません。loss/PPLに加え、非空回答、EOS、反復率、平均長、keyword relevance、日本語文字率を計測します。

```powershell
python -m evaluation.evaluate_v02 --checkpoint checkpoints/unipilot-v02-step-1000/checkpoint-step-1000.pt --output evaluation/results-v02-1000.json
python -m evaluation.compare_checkpoints evaluation/results-v02-100.json evaluation/results-v02-500.json evaluation/results-v02-1000.json
python evaluation/plot_v02.py checkpoints/unipilot-v02-step-100/training_log.csv checkpoints/unipilot-v02-step-500/training_log.csv checkpoints/unipilot-v02-step-1000/training_log.csv
```

1000 stepまでの実測ではlossは改善しましたが、自然な回答品質にはまだ達していません。詳細は `DATASET_REPORT_V02.md`、`TRAINING_REPORT_V02.md`、`EVALUATION_REPORT_V02.md` を参照してください。モデルを50M/100Mへ拡大する前に、19.8Mモデルを全データでより長く学習することを推奨します。

### v0.2 API

既存endpointに `GET /checkpoints`、`POST /model/load`、`GET /evaluation/latest` を追加しました。model切替は `UNIPILOT_DEV_MODE=1` のときだけ有効で、`checkpoints/` 外のファイルは読み込めません。Webの「比較」ページからローカルcheckpointを選べます。本番相当ではdeveloper modeを設定せず、起動時に指定した1モデルだけを使ってください。

## v0.3: 意味対応を重視したCurriculum Learning

v0.3は19,814,784 parameters、11 layers、6 heads、context 256、512 vocabを固定し、データ品質、A/B/C curriculum、Stage Cのassistant-only loss、EOS検査、semantic評価だけを追加した版です。外部AIや事前学習済みモデルは使いません。

### 1. データ準備

```powershell
.\prepare-v03.bat
```

42,000件をStage A General Japanese 20,000、Stage B University Text 10,000、Stage C University Conversation 12,000へ分けます。validatorは重複、split leak、format、intent、会話EOS、品質scoreを検査します。

### 2. Stage A / B / C

```powershell
.\train-v03-stage-a.bat
.\train-v03-stage-b.bat
.\train-v03-stage-c.bat
```

各batは直前stageのcheckpointをresumeし、1,000、2,000、5,000 stepで保存します。scratch実験とv0.2継続実験は `--initialization scratch-v03` / `--initialization resume-v02` で区別できます。学習済み5kから長期実験を行うコマンドは `train-v03-10000.bat`、`train-v03-20000.bat`、`train-v03-30000.bat`、`train-v03-50000.bat` です。ただし現在の5kモデルはEOS・反復の停止条件を満たさないため、データ改善前の実行は推奨しません。

### 3. 評価

```powershell
.\evaluate-v03.bat
python -m evaluation.build_v03_artifacts
```

固定300問でexpected/category/forbidden keyword、0–100 relevance、category accuracy、meaningful response、EOS、runaway、反復、日本語比率、broken text、メール構造を測定します。人手評価は `evaluation/human-eval-v03.json` の50件へ0–4を入力できます。

### 4. チャットとWeb

```powershell
.\chat-v03.bat
.\start-api.bat
.\start-web.bat
```

開発者ページには最終評価、training monitor、v0.2/v0.3比較を表示します。APIには `GET /training/latest` と `GET /evaluation/comparison` もあります。最終5kモデルは意味的関連性が改善しましたが、日本語の混線、反復、途中終了が残る研究用checkpointです。

## 制約

小規模な合成データだけでは一般知識、事実性、長文理解、複雑な推論は身につきません。出力が不自然、反復的、または誤っている可能性があります。改善は学習済みモデルの流用ではなく、権利確認済みデータの拡充、重複削除、応答品質評価、モデル規模と学習stepの段階的増加で行います。

## CPU推論とRender

推論はKV cache、`torch.inference_mode()`、checkpointのmmapロードを使用します。学習checkpointを削除せず、Render向けにoptimizerを除いた同一重みの推論専用版を作れます。

```powershell
python -m scripts.export_inference_checkpoint --input checkpoints/v04-eos15/checkpoint-step-2000.pt --output checkpoints/v04-eos15/unipilot-mini-v04-inference.pt
```

Renderではbuild commandを`pip install -r requirements-prod.txt`、start commandを`uvicorn api.main:app --host 0.0.0.0 --port $PORT --workers 1`とし、`UNIPILOT_CPU_THREADS=1`を設定してください。配布先から推論専用checkpointを取得する場合は、`UNIPILOT_CHECKPOINT`にそのパスを指定します。`POST /chat`は互換維持され、`POST /chat/stream`はNDJSONで累積回答を逐次返します。

## v0.4: Clean ConversationとEOS学習

v0.4はモデル構造と512 vocabを固定し、Stage B checkpointから8,000件のClean Stage Cだけを最大2,000 step学習します。

```powershell
.\prepare-v04.bat
.\train-v04.bat
.\evaluate-v04.bat
.\chat-v04.bat
```

CleanデータはEOS・自然終止100%、pair重複0、カテゴリ混線0、最大opening 7.44%です。推奨設定はEOS weight 1.5、temperature 0.7、top-k 40、top-p 0.9、repetition penalty 1.1、max new tokens 96です。開発者ページではv0.3/v0.4比較と人手評価50問の0–4採点ができます。人間が採点するまではHuman Scoreを`PENDING`として扱います。
