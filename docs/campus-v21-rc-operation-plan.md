# Campus v2.1 RC: Human Gate後の運用設計

状態: **INACTIVE — Human Production GateがPASSするまで実行禁止**

評価対象はcommit `0dc18789be28613a8c651cfefde63fb659ee2019`です。Human Evaluation中はRouter、FAQ、Tool、Retrieval、Composer、回答Validatorを変更しません。現行本番v0.4、Render、Vercel、GitHub Releaseにも変更を加えません。

## 本番移行手順（Gate PASS後のみ）

1. `evaluation/campus-v21-rc-human-report.json`のGateが`PASS`で、100問・全Pairwise・全問題チェックが完了していることを確認する。
2. RC manifestの回答ロジックSHA256とcheckpoint/tokenizer SHA256を再検証する。
3. 別の明示的な承認作業で、Renderの`UNIPILOT_PIPELINE_VERSION`だけを`campus-v2.1`へ変更する。モデルはv0.4 step 2000のまま維持する。
4. `/health`、`/model-info`、`/chat`、`/chat/stream`と専門Toolをsmoke testする。
5. エラー率または重大な誤回答が基準を超えた場合は、pipeline versionを直前値へ戻す。評価RCをその場で修正しない。

この文書は手順だけを定義し、push、deploy、本番切替を実行しません。

## 段階Beta（Gate PASS後のみ）

| 段階 | 人数 | 昇格条件 |
|---|---:|---|
| Closed | 5–10人 | 重大誤回答0、個人情報収集なし、主要Tool完了を確認 |
| Limited | 30人 | 👎理由と再質問率を確認し、停止条件を超えない |
| Expanded | 100人 | 専門カテゴリの正確性、Tool完了率、p95応答時間を確認 |
| General | 一般 | 各段階の人間レビュー完了と別途承認 |

停止条件は、重大誤回答または大学制度の誤断定が1%超、Validator fallbackの急増、Tool計算の再現不能、HTTPエラーの継続です。停止時はv0.4構成へ戻し、Betaデータをv2.2候補として分析します。

## 収集する匿名集計

- 👍 / 👎
- 役に立ったか、正しかったか
- 再質問の有無（内容ではなく回数）
- Tool開始・完了、入力不足、コピー操作
- 応答時間、Streaming first event、fallback
- Router category、action、confidence band
- pipeline version、RC manifest hash、回答hash

氏名、学籍番号、大学名、メール本文、質問・回答の生テキスト、IPアドレスは評価指標として保存しません。自由記述を設ける場合は送信前に「個人情報を書かない」旨を表示し、短期保存・削除期限を別途定めます。

## 軽量Feedback UI設計

回答下に`👍`と`👎`だけを常時表示します。`👍`は匿名event idと回答hashだけを送信します。`👎`を選んだ場合だけ、次の単一選択を表示します。

- 回答が違う
- 質問に答えていない
- 分かりにくい
- 大学によって違う
- その他

送信payloadは`event_id`、時刻、pipeline version、category/action/confidence band、tool名、回答hash、選択理由、応答時間、再質問回数に限定します。Feedbackは非同期で、失敗しても回答を再生成しません。Router・Tool・FAQの選択、回答本文、Human Evaluationデータには影響させません。

## 次期候補の判定

- Campus v2.2: Human Gateの失敗理由、hallucination 13候補、`toeic_plan → study_plan` 3件、Retrieval 7件を人間が確認してから必要最小限を決める。
- Standard 50M: Router/Tool/RAGの人間Correctnessが基準以上で、MODELを使う質問だけが4.2未満になった場合のみ再開候補とする。それまでは停止を維持する。
