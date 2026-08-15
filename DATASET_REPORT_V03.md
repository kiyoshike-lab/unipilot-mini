# UniPilot Mini v0.3 Dataset Report

## 結論

`unipilot-dataset-v03` は外部AIを使わないローカル規則生成で再構成した。50,000件の維持を優先せず、42,000件を採用した。元の目標件数との差は8,000件、生成後のvalidator除外は0件である。これは既存50,000行から8,000行を個別に低品質判定したという意味ではなく、品質基準を満たす構成だけを42,000件生成・採用した結果である。

| Stage | 用途 | 件数 | 学習対象 |
|---|---|---:|---|
| A | General Japanese | 20,000 | 全tokenのnext-token prediction |
| B | University text | 10,000 | 全tokenのnext-token prediction |
| C | University conversation | 12,000 | ASSISTANT回答とEOSのみ |
| 合計 |  | 42,000 |  |

Splitはtrain 37,800、validation 2,100、test 2,100。Stage Cには10 intentを付与し、全回答を `<ASSISTANT> ... <EOS>` として扱える形式にした。特定大学の制度を断定せず、情報不足時は時間割、シラバス、試験案内などの確認を促す回答を含む。教授メールは欠席、遅刻、課題提出遅れ、試験・単位相談、面談、質問を含む。

## 品質検査

| 指標 | 結果 |
|---|---:|
| exact duplicates | 0 |
| template-family split leaks | 0 |
| invalid samples | 0 |
| conversation EOS valid | 100% |
| unique conversation answers | 13.42% |
| average length score | 99.90% |
| average diversity score | 87.08% |
| average keyword alignment | 52.38% |

各行に `length_score`、`diversity_score`、`keyword_alignment`、`format_valid` を付与した。固定評価は10カテゴリ各30問、合計300問で、各問にcategory、intent、expected/forbidden keywordsを持たせた。人手評価用50問は採点前のまま保存している。

## 残るデータ課題

会話回答のユニーク率13.42%は小型モデル向けの一貫性には寄与するが、表現の画一化と文断片の混線を招いた可能性が高い。次版では件数追加より、短く一意な回答、カテゴリ間で重ならない表現、EOS直前の自然な結びを増やすべきである。
