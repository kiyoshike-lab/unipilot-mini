# UniPilot Foundation v0.9 Data Design

- Base: CC BY-SA 4.0の日本語Wikipedia本文のみをlanguage modeling形式で学習する。
- Campus: project-authored CC0の安定した大学生活知識だけを追加pretrainingする。
- Instruction: project-authored品質確認済みQ&AとHuman ◎を別stageで学習する。
- RAG-only: 政府・大学公式の制度、数字、期限はweightへ入れずsource metadata付きで検索する。
- Human: 明示的に◎の回答だけをapproved replayへ入れる。AI改善案や△は自動採用しない。
- Evaluation: validation 200と未開封final Blind 1,000を学習前に固定し、trainとの近似重複を除外する。

Instruction 30,000目標に対する未達: 18156件。近似文の水増しでは埋めず、Human reviewと新しい意味単位を追加する。
