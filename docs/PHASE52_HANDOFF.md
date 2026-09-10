# PHASE52 — 最終記録 / 残る承認ゲート

2026-09-11 JST。研究監査・Stage4実装・ローカルQAは完了。認証済みPreviewのLive QAはNOT_TESTEDのため、PHASE52全体の完了とpushは保留。最初からやり直す必要はありません。

## ML

| 項目 | 結果 |
|---|---|
| Resolver Gate / Z root | PASS。各ML processで `UNIPILOT_CHECKPOINT_ROOT=Z:\AI\unipilot-mini\checkpoints` を明示。19 manifest destinations一致 |
| Checkpoint integrity | 対象9件SHA256一致。PHASE51のstrict/model/optimizer/scheduler/sampler・permutation/Python・NumPy・PyTorch・CUDA RNG PASSを同一SHAで再利用。今回の新規reloadとは主張しない |
| Checkpoint操作 | COPY/MOVE/DELETE/OVERWRITEすべて0。正式checkpointを変更しない |
| Final Blind | 内容未開封、SHAのみ一致：`fa7912d58ce251bb10b513f59793bb8ca6c0023b4fe08d1c040b8ccbfe49845b` |
| Fresh data audit | 75候補を監査。Foundation v1.1 testも過去completion評価・失敗診断で使用済み。その他候補には利用履歴・独立性の証明不足 |
| Confirmatory availability | `PROVENANCE_UNCERTAIN`。適格セットのpath/SHAなし。将来の独立管理holdout・human評価設計を保存 |
| Confirmatory Gate | `CONFIRMATORY_DATA_INVALID`＝適格データ未確立で未実施。モデルの測定FAILではない |
| EOS / Core / Sampling confirmatory | すべてNOT_RUN。既存testをfreshとして再利用していない |
| Attractor root cause | `MIXED_OR_UNKNOWN`。既存900 greedy tracesはrunaway100%。全時系列、最大32token前の確率・相対logit差、上位20cycleのtrain頻度を保存。相関を因果とは扱わない |
| Decoder mitigation | 既存temperature0.7観測のみ。新規調整実験なし、モデル修復・Beta安全性の主張なし |
| Attractor Gate | `ATTRACTOR_CAUSE_STILL_UNKNOWN` |
| 5e-5 evidence | PHASE51でLM/Top-k/Middle/Core/Supported Tail/Contextは7.5e-5より優位。ただしEOS seed variance・Core seed2026・Sampling prompt-local回帰が未解決 |
| 7.5e-5 evidence | 比較優位の独立確認なし。既存の追加gate不合格も保持。新しい合格基準で救済しない |
| Formal LR Candidate Gate | `FORMAL_LR_UNRESOLVED` |
| New inference / training / canonical / 20M / Foundation Base | すべてNO |

詳細は `evaluation/foundation-v41-confirmatory-report.md` と `evaluation/foundation-v41-attractor-report.md`。巨大raw2件はローカルのみ、summaryにpath/SHAを記録。checkpointと元データは保持。

## Product / QA

| 項目 | 結果 |
|---|---|
| Academic OS | Stage4 PARTIAL、Feature7 Foundation、Feature8 Beta。特定大学制度のValidated/Completeではない |
| GPA deterministic engine | `/gpa`。Current/What-if/Target、必要将来平均、上限超過Impossible、0分母、単位・評価除外、再履修3policy、丸め分離、tab Save/Clear・local JSON export。LLM算術なし |
| DegreeRule schema | `/degree`。大学/学部/学科/年度/version/有効期間/カテゴリ/必要単位/course・group constraints/source。TEST UNIVERSITY架空fixtureのみ |
| Source / conflict | 出典不足Unknown。version/year競合は自動混合せず停止。実在大学の卒業可否を断定しない |
| Feature1–6 regression | ローカルDemoのTutor/Materials/Exam/Report/Research/Citation PASS。Evidence Ledger・exact span・claim trace・偽metadata拒否を保持 |
| Mobile | 360/390/768/1024/1440、9画面45件＋入力済み20件PASS。狭幅の9項目ナビは横スクロール |
| Accessibility | label/keyboard/focus/aria-live/44px target/error association/semantic table/reduced motion確認PASS。完全な支援技術認証ではない |
| pytest | 501 passed、0 failed、5 warnings。別途Web framing契約1 passed。既存の生成物5件はテスト前のbytesに復元 |
| Web unit / npm lint / npm build | 31 passed / PASS / PASS（19 static outputs） |
| Preview Live QA | NOT_TESTED。既知の歴史的PHASE49 Previewはログインへredirect、Render health/OPTIONSはtimeout。Stage4のLive成功やcompiled API URLを未確認 |
| CORS / protection | bounded exact-origin設計とPreview保護を保持。wildcard化・auth bypassなし。ログインページのヘッダをAPI CORS成功と混同しない |
| External AI API | OFF。GPA/単位データの外部送信なし |

証跡：`evaluation/phase52/final-qa.json`、`web/qa/phase52/`。テスト画像のOnline/回答はDemo fixtureであり、Live品質証明ではありません。

## Git / 安全性 / 再開条件

開始HEAD/origin：`ab40aac8335a5cab2d0f858d4b3831f947d065a6`。branch：`foundation-research`。

指定の3commitに分離（実際のSHAは最終応答または `git log -3 --oneline` で確認）：

1. `research: add phase 52 confirmatory and attractor audit`
2. `web: add deterministic gpa simulator`
3. `web: add degree rule foundation`

Protected4、PHASE42 READY5はSHA一致、変更/stageなし。checkpoint binary・巨大rawはstageなし。既存の別作業storage関連dirty/deletionは保持し、今回のcommitには含めない。main merge/push、Render/Vercel Production deploy/config変更なし。

Pushは「全QA PASS後のみ」の条件により保留。認証済みLive QAを完了するか、ユーザーがローカルQA PASSをpush条件とすることを明示承認した場合のみ、保護対象・branch・originを再確認して `git push origin foundation-research`。未承認の新学習や本番deployは行わない。

Liveの短い確認手順は `PREVIEW_LIVE_QA.md`。新しいPreviewを使う場合はStage4のdeployment SHAを確認する。今回の歴史的URLをStage4環境と偽らない。
