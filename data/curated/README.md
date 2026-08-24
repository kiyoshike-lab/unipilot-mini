# Human-approved answer memory

`POST /ai-review/campus` で人間が「採用」した改善回答だけが
`human-approved-answers.jsonl` に保存されます。

このファイルは学習候補のメモリです。学習・checkpoint更新・本番反映には自動で使いません。
採用データを次世代学習へ使う場合は、別途、人間による内容・出典・ライセンス確認が必要です。
