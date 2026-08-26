# Foundation v1.0 Human Feedback Plan

- Base学習中はReward model、DPO、Preference learningを実行しない。
- Base Gate通過後、同一promptに対する2回答のblind比較をまず100件集める。
- 評価者間の一致率を確認し、曖昧なrubricを修正してから300件、最終的に500件へ増やす。
- Preference UIは既存機能を維持し、回答、選好、理由、モデル/checkpoint、prompt hashを保存する。
- Human Approvedへ入れるのは明示的に承認された回答だけとし、自動評価結果を承認として扱わない。
- 100〜500比較が蓄積し、品質監査と汚染検査を通過するまでDPOを開始しない。
