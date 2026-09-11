# PHASE53 — 実施済み成果と停止地点

2026-09-11 JST。Fresh pool構築・admissibility判断・Stage5 Web実装・ローカルQA、および承認済みTrack C generation policy studyは完了。**PHASE53全体はPARTIAL**：Fresh holdoutは文書数不足、Confirmatory評価は未実施で、Formal LRは未解決です。新学習は一切開始していません。

## ML結果

| 項目 | 結果 |
|---|---|
| Resolver / Z root | PASS、`Z:\AI\unipilot-mini\checkpoints`。各ML processへ明示設定、19 manifest destinations一致 |
| Checkpoint integrity | 9件SHA一致。同一bytesに対するPHASE51 strict/model/optimizer/scheduler/sampler/Python・NumPy・PyTorch・CUDA RNG PASSを再利用 |
| Final Blind | 内容未開封、SHAのみ一致：`fa7912d58ce251bb10b513f59793bb8ca6c0023b4fe08d1c040b8ccbfe49845b` |
| Training cutoff | 最大取得時刻2026-08-27T13:21:24.057451+00:00。revision最大2026-08-24。latest URLのためdump日付自体はUNKNOWN |
| Fresh source / license | Wikimedia公式Japanese Wikipedia Action APIの新規ページ初期revision。CC BY-SA4.0、page/revision/attribution/cleaning/hashを記録 |
| Fresh documents / tokens | 固定500候補枠で129 /174885。網羅的取得の最大数とは主張しない。general偏重、math/language supportなし |
| Exact / near duplicate exclusions | exact0、normalized0、near0、known page identity0。既知11916 text fieldsと比較。未検出paraphraseや外部履歴に限界あり |
| Fresh Holdout Gate | `FRESH_HOLDOUT_TOO_SMALL`。250文書条件を緩和しない |
| Diagnostic / Confirmatory / Reserve | 77/25/27文書、100094/36960/37831tokens。全未scored・未consumed。Reserveは封印、以後hash確認のみ |
| Confirmatory prereg SHA | `e22b3043ff5cdfdc1d80ba0f4ee9171c990d26053296484517bc474592b52649`（未登録理由artifact） |
| Confirmatory Gate | `CONFIRMATORY_DATA_INVALID`。実測モデルFAILではなく、適格データ不足 |
| 5e-5 /7.5e-5 metrics、EOS、Sampling | すべてfresh確認はNOT_RUN。過去5e-5比較優位は独立confirmatory証拠ではない |
| LR status | `FORMAL_LR_UNRESOLVED` |
| Generation policy / best safe setting | 既存PHASE51固定prompt 24件、C/B seed42、CUDA FP32で固定grid 20 runsを実施。best safe settingは`NONE`。greedy runawayは両arm 100%、非greedy最良でもC 95.83% / B 97.92% |
| Attractor Gate | `GENERATION_POLICY_UNSAFE`。登録済み品質screenを全settingがFAIL。MODEL_FIXED / staging安全性を主張しない |
| New training / canonical /20M / Base | すべてNO |

Fresh本文・取得rawは `Z:\AI\unipilot-mini\data\phase53-fresh`、Git外。checkpointとは別の保存先。COPY/MOVE/DELETE/OVERWRITE禁止対象のcheckpoint操作はすべて0。既存の正式成果物・PHASE42 READY・Final Blindを変更しない。

詳細：`evaluation/foundation-v42-confirmatory-report.md`、`evaluation/foundation-v42-generation-policy-report.md`、`evaluation/phase53/final-qa.json`。

## Product / QA

Academic OS Stage5 PARTIAL。Feature9 **Beta**：Timetable / Attendance / Assignments / local Save/Clear/JSON Export/Import、Timezone/DST・重複・期限計算のローカルQA PASS。Today's UniPilot **Foundation**：保存済みユーザー入力からのみ次の授業・今日/近い期限・欠席注意を表示。架空予定・大学公式出席ルールを生成しない。

Feature10 **Beta（Template限定）**：教授メールの7用途・3tone、未入力placeholder、編集可能preview、古いpreviewのCopy無効化、Copy/ClearがPASS。名前・日付・理由・約束を補完しない。AI生成とは表示せず、API/mail連携・自動送信OFF。`frameEmailFacts`は未接続の将来契約のみ。

Feature1–8 regression PASS。pytest505 passed＋Web framing契約1 passed、Web unit45 passed、npm lint/build PASS（21 static outputs）。Browserは11routes×5幅55件＋入力済み35件＝90件、機能確認もPASS。幅360/390/768/1024/1440、labels/keyboard/focus/aria-live/error association/44px/semantic tables・lists/reduced motionを確認。全支援技術の認証ではありません。

Demo PASS、Live Preview **NOT_TESTED**。Stage5の認証済みセッション・deploymentは未検証。過去Previewの結果を流用してLive PASSとしない。bounded exact-origin CORS・Vercel保護維持、外部AI OFF。

## Gitと再開方法

開始HEAD/originは `aacbc4c276066472d4e1bcb976b35fcac63fde2d`、branch `foundation-research`。指定のresearch /planner /emailの3commitに分離。実SHA/push結果は最終応答またはGit履歴を参照。

PHASE53はローカル全QA PASSならpush可と明記しているため、Live未検証のみを理由にpushを禁止しない。Track Cは完了したが、generation policyは安全ではない。main、Render/Vercel Production、Foundation production promotionは変更しない。保護4・READY5はSHA維持、stage禁止。checkpoint binary・large downloaded corpus・huge rawもstage禁止。既存storage関連dirty/deletionには触れない。

停止地点：Track Cは完了。loop-stop replayはrunawayをC greedyで25.00%、B greedyで33.33%へ下げた一方、triggered outputのEOS率は0%で、正常なmath/list/definition対照も未登録のため誤停止率を推定できない。安全候補のstagingは行わない。以後はFresh Holdout Gateの適格化と、必要なら事前登録済み正常対照を含む独立decoder studyを新規に承認してから扱う。Fresh acquisitionや封印Reserveを最初からやり直さない。
