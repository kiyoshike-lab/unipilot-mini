# Campus v2.1 (opt-in evaluation mode)

Campus v2.1はv0.4モデルを変更せず、曖昧質問、否定・対比表現、FAQ検索だけを改善する評価用pipelineです。Standard 50M学習、外部LLM/API、本番設定変更は行いません。

## Reproduce

```powershell
python -m scripts.build_campus_v21_data
python -m evaluation.tune_campus_v21_clarification
python -m evaluation.evaluate_campus_v21_retrieval
python -m evaluation.evaluate_campus_v21
python -m evaluation.analyze_campus_v21_errors
python -m evaluation.report_campus_v21
```

Clarification thresholdは`clarification-validation-1200`、retrieval thresholdはretrieval validation 152件だけで選択します。既存blind 2000件とadversarial 300件はtest専用です。

## Opt in locally

```powershell
$env:UNIPILOT_PIPELINE_VERSION = "campus-v2.1"
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

人手評価は`http://localhost:3000/campus-v21-eval`で行います。ChatGPT/Geminiの回答は各UIで同じ質問を実行し、手入力します。外部APIは使いません。

自動ゲート合格だけでは本番昇格できません。Human 100が完了して基準を満たすまで、本番v0.4を維持します。
