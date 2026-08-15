# UniPilot Mini v0.4 Evaluation Report

BestはEOS weight 1.5、step 2,000、max new tokens 96、temperature 0.7、top-k 40、top-p 0.9、repetition penalty 1.1。固定`unipilot-eval-v03-300`を変更せず評価した。

| Metric | Target | Result | Status |
|---|---:|---:|:---:|
| EOS | >=80% | 99.67% | PASS |
| Runaway | <=20% | 0.33% | PASS |
| Meaningful | >=40% | 52.00% | PASS |
| Keyword | >=30% | 35.56% | PASS |
| Category | >=30% | 28.67% | MISS |
| Japanese | >=98% | 98.33% | PASS |
| Repetition | <=3% | 0.25% | PASS |
| Too short | — | 0% | PASS |
| Broken generation | — | 0% | PASS |

Validation loss 0.1016、PPL 1.11、平均57.77 token / 42.96文字、generation 24.85 tokens/sec。broken byte 0%、invalid sequence 0%、symbol noise 0.13%。Human evaluationは50問のUIと永続化APIを実装したが、人間が未採点なので`PENDING`。

64 tokenではMeaningful 47.67%に対してEOS 75.67%だった。96 tokenでは自然なEOSまで到達でき、EOS 99.67%・Meaningful 52%となった。128は50問探索で96と同等だったため、短い96を採用した。現在最大の問題はCategory Accuracy 28.67%と、Email structure 3.33%である。

v0.5は **A: Clean Stage Cをさらに改善** を推奨する。カテゴリ固有語彙、とくにEMAIL/通常相談の分離とanswer重複削減を優先する。19.8M長期学習と50M拡張はまだ推奨しない。
