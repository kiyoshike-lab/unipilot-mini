# UniPilot Mini v0.3 / v0.4 生成比較

先頭10件は指定された重要質問、残り20件は固定評価の各カテゴリ2件です。

| Metric | v0.3-5000 | v0.4-2000 |
|---|---:|---:|
| EOS | 16.33% | 99.67% |
| Runaway | 83.67% | 0.33% |
| Meaningful | 29.00% | 52.00% |
| Keyword | 23.22% | 35.56% |
| Category | 18.33% | 28.67% |
| Japanese | 95.70% | 98.33% |
| Repetition | 8.39% | 0.25% |

## 1. 明日試験なんだけど何したらいい？

Expected category: `exam`  
Expected keywords: 試験, 範囲, 復習

v0.3:

件名：状況を整理しましょう。まご連絡
○○○先生
お世話になっております。○○学の〉です。ご連絡
○先生
お世話になっております。○○学部の〈学生氏名〉です。授業に遅れる見込み

v0.4:

無理なく進めるなら、統計学の試験範囲を確認し、残り1日なら苦手な部分と重要問題に絞って復習しましょう。

Automatic metrics: relevance 23.5 → 100.0; meaningful False → True; EOS False → True.

## 2. 課題が3つあるんだけどどれからやればいい？

Expected category: `assignment`  
Expected keywords: 課題, 締切, 提出

v0.3:

件名：状況を整理しましょう。件名：授ってあります。○○○先生
お世話になっております。○○学部の〉です。本日の話になっております。○学部の〈学生氏名〉です。授業を欠席いたします。直前の連絡とな

v0.4:

無理なく進めるなら、線形代数の課題は締切と作業量を比べ、期限が近いものから提出準備を進めるのがおすすめです。

Automatic metrics: relevance 24.0 → 100.0; meaningful False → True; EOS False → True.

## 3. 教授に欠席メールを送りたい

Expected category: `email`  
Expected keywords: 件名, 先生, 連絡

v0.3:

件名トってずーレレートってロいのトって牉ロのログートってトは、資料集め、資料集め、構成、執筆、引用確認、推敲に分けます。残り8日なら、まず構成を作るのがおすすす

v0.4:

最初の一歩は、締切、重要度、残り作業を並べ、今日終える一つを決めて予定に入れてください。

Automatic metrics: relevance 40.0 → 25.0; meaningful True → False; EOS False → True.

## 4. 単位が心配

Expected category: `credit`  
Expected keywords: 単位, シラバス, 教務

v0.3:

件ります。件名＼授業欠大学ご連絡


○○○先生
お世話の』っております。○○先生
お世話になっております。○○学部の〈学生氏名〉です。ご相談しましましたく、あり時間をあ

v0.4:

締切を確認すると、情報科学は25分だけ勉強し、最後に要点を復習する計画にすると続けやすいです。

Automatic metrics: relevance 23.8 → 25.0; meaningful False → False; EOS False → True.

## 5. レポートが終わらない

Expected category: `report`  
Expected keywords: レポート, 構成, 引用

v0.3:

件名トって畏い手順に分けると進めると進めると進めやすい手順に分けると進めやすいです。ログログラミングの出席条件は授業によって異なります。シラバスと出席記録を確認し、欠席の扱いが不明なら担当教員へ相談してください。分

v0.4:

最初の一歩は、締切を確認すると、重要度、残り作業を並べ、今日終える一つを決めて予定に入れてください。

Automatic metrics: relevance 25.0 → 25.0; meaningful False → False; EOS False → True.

## 6. 履修をどう決めればいい？

Expected category: `registration`  
Expected keywords: 履修, 必修, シラバス

v0.3:

件名：状況を整理し手順に分けると進めるお試験時間の情報が登録されていないため、現在は分かりません。時間割や大学の案内を確認してください。

v0.4:

無理なく進めるなら、法学の履修は必修区分、時間割、シラバス、卒業要件を順に確認して決めてください。

Automatic metrics: relevance 25.0 → 100.0; meaningful False → True; EOS True → True.

## 7. 空きコマ何したらいい？

Expected category: `study`  
Expected keywords: 勉強, 復習, 計画

v0.3:

件名＼状況を整理しましょう。ト業を決学のレロい手順に分けると進めると進めやすいです。ポータ分けると進めやすいです。情報科学のレポートは、問いの確認、資料集め、構成、執筆、引用確認、推敲に分けます。残り8日なら、ま

v0.4:

無理なく進めるなら、線形代数のレポートは問い、資料、構成を先に決め、本文を書いた後に引用を確認してください。

Automatic metrics: relevance 25.0 → 25.0; meaningful False → False; EOS False → True.

## 8. 明日の試験って何時？

Expected category: `general`  
Expected keywords: 確認, 情報, 分かりません

v0.3:

形のご連絡


○○先生
○○先生
お世談のお世話になっております。○○先生
お世話になっております。○○学部の〈学生氏名〉です。授業を欠席いたく、面質

v0.4:

今できることは、締切を確認すると、情報科学の課題は締切と作業量を比べ、期限が近いものから提出準備を進めるのがおすすめです。

Automatic metrics: relevance 23.5 → 65.0; meaningful False → True; EOS False → True.

## 9. 今日何したらいい？

Expected category: `schedule`  
Expected keywords: 予定, 締切, 優先

v0.3:

を大心理学の大学に分けると進めると進めやすいです。短い手順に分けると進めやすいです。哲学の課題は、提出締切と形式を先に確認します。残り8日なら、未完了の作業を分け、今日の提出準備から始めると進めやすいです。無理な場合

v0.4:

締切を確認すると、締切を確認すると、重要度、残り作業を並べ、今日終える一つを決めて予定に入れてください。

Automatic metrics: relevance 40.0 → 85.0; meaningful True → True; EOS False → True.

## 10. 出席が少なくて心配

Expected category: `attendance`  
Expected keywords: 出席, 欠席, シラバス

v0.3:

今できることを大学を大学を大学ず状況を整理しましょう。日本史の出席条件は授業によって異なります。シラバスと出席記録を確認し、欠席の扱いが不明なら担当教員へ相談してください。無理な場合は担当者へ早めに

v0.4:

落ち着いて、締切を確認すると、線形代数の単位条件は断定できません。シラバスを確認し、不明なら担当教員か教務へ相談してください。

Automatic metrics: relevance 100.0 → 40.0; meaningful True → True; EOS False → True.

## 11. 課題が3つあるんだけど、どれからやればいい？

Expected category: `assignment`  
Expected keywords: 課題, 締切, 提出

v0.3:

件名＼状況を整理しましょう。微ります。件名：授業内容についてのご連絡

○○先生
お世話になっております。○○学部の〈学生氏名〉です。授業について込みです。日の授業に遅容についてです。氏出れるな時間の

v0.4:

おすすめは、統計学の課題は締切と作業量を比べ、期限が近いものから提出準備を進めるのがおすすめです。

Automatic metrics: relevance 24.5 → 100.0; meaningful False → True; EOS False → True.

## 12. 課題が3つあるんだけど、どれからやればいい？ 科目は生物学で、残り8日です。

Expected category: `assignment`  
Expected keywords: 課題, 締切, 提出

v0.3:

焦らことをず、確認から始めましょう。日本史の課題は、提出締切と形式を先に確認します。残り8日なら、未完了の作業を分け、今日の提出準備から始めると進めやすいです。分からない点は早めに大学へ確認すると安心です。研究

v0.4:

線形代数の課題は締切と作業量を比べ、期限が近いものから提出準備を進めるのがおすすめです。

Automatic metrics: relevance 100.0 → 100.0; meaningful True → True; EOS False → True.

## 13. 欠席が増えて出席状況が不安です。

Expected category: `attendance`  
Expected keywords: 出席, 欠席, シラバス

v0.3:

まず状況を整理しましょう。焦らず状況を整理しましょう。教質問い手順に分の出勉学の出席条件は授業によって異なります。シラバスと出席記録を確認し、欠席の扱いが不明なら担当教員へ相談してください。最後に条件をもう一度確認してください。

v0.4:

その状況では、線形代数の単位条件は断定できません。シラバスを確認し、不明なら担当教員か教務へ相談してください。

Automatic metrics: relevance 100.0 → 40.0; meaningful True → True; EOS True → True.

## 14. 欠席が増えて出席状況が不安です。 科目は生物学で、残り8日です。

Expected category: `attendance`  
Expected keywords: 出席, 欠席, シラバス

v0.3:

焦らのの情報が登の情報が登録されていないため、現在は分かりません。時間割や大学の勉強は、今日の目標を一つに絞り、短い集中と休憩を繰り返します。最後に復習し、次回の計画を一行で

v0.4:

試験範囲を確認し、残り1日なら苦手な部分と重要問題に絞って復習しましょう。

Automatic metrics: relevance 25.0 → 25.0; meaningful False → False; EOS False → True.

## 15. 必修科目の単位が心配です。

Expected category: `credit`  
Expected keywords: 単位, シラバス, 教務

v0.3:

件名＼大○です。件名：授業欠席い手順に分けると進めると進めると進のご連絡
○○先生
お世話になっております。○○学験範囲を確認し、名〉です。授業啰名〉です。授業を欠

v0.4:

落ち着いて、締切を確認すると、生物学の単位条件は断定できません。シラバスを確認し、不明なら担当教員か教務へ相談してください。

Automatic metrics: relevance 24.0 → 100.0; meaningful False → True; EOS False → True.

## 16. 必修科目の単位が心配です。 科目はプログラミングで、残り8日です。

Expected category: `credit`  
Expected keywords: 単位, シラバス, 教務

v0.3:

今できるずートってします。件名：遅を整理しましょう。トって物ロいのず、資料集め、構成、執筆中中と休憩を入修に分けると進めやすいです。終わった項目を記録すると進捗が分かります。

v0.4:

試験範囲を確認し、残り1日なら苦手な部分と重要問題に絞って復習しましょう。

Automatic metrics: relevance 25.0 → 25.0; meaningful False → False; EOS True → True.

## 17. 教授に今日休むとメールしたいです。

Expected category: `email`  
Expected keywords: 件名, 先生, 連絡

v0.3:

件名トってってって短い手順に分けると進めると進めやすい手順に分けると進めやすいです。ポータ分位条件は明業や大学によって異なります。シラバスで評価方法を確認し、不明点は担当教員や教務へ相談してください。無理な

v0.4:

短く整理すると、心理学のレポートは問い、資料、構成を先に決め、本文を書いた後に引用を確認してください。

Automatic metrics: relevance 40.0 → 25.0; meaningful True → False; EOS False → True.

## 18. 教授に今日休むとメールしたいです。 科目は情報科学で、残り8日です。

Expected category: `email`  
Expected keywords: 件名, 先生, 連絡

v0.3:

短い手順に分けると進めるず状況を整理しましょう。ロい手順に分けると進めやすいです。物理学のレポートは、問いの確認、資料集め、構成、執筆、引用確認、推敲に分けます。残り1日なら、まず構成を作るのがおすすめです。分からない

v0.4:

本史の試験範囲を確認し、残り8日なら苦手な部分と重要問題に絞って復習しましょう。

Automatic metrics: relevance 25.0 → 5.0; meaningful False → False; EOS False → True.

## 19. 明日試験だけど、まだ何もしていません。

Expected category: `exam`  
Expected keywords: 試験, 範囲, 復習

v0.3:

件名に分の情報が登の情報が登録されていないため、現在は分かりません。時間割や大学の試験範囲を確認し、基礎、問題演習、苦手部分の復習という順で勉強します。残り1日に合わせて、一

v0.4:

件名：席条件は授業ご連絡
○先生
お世試験範囲を確認し、残り1日なら苦手な部分と重要問題に絞って復習しましょう。

Automatic metrics: relevance 100.0 → 99.9; meaningful True → True; EOS False → True.

## 20. 明日試験だけど、まだ何もしていません。 科目は心理学で、残り8日です。

Expected category: `exam`  
Expected keywords: 試験, 範囲, 復習

v0.3:

今できることを順番に分けると進めやすいです。心理学の課題は、提出締切と形式を先に確認します。残り8日なら、未完了の作業を分け、今日の提出準備から始めると進めやすいです。分からない点は早めに大学へ確認すると安心です。

v0.4:

本史の試験範囲を確認し、残り1日なら苦手な部分と重要問題に絞って復習しましょう。

Automatic metrics: relevance 25.0 → 100.0; meaningful False → True; EOS True → True.

## 21. 明日の試験は何時ですか？

Expected category: `general`  
Expected keywords: 確認, 情報, 分かりません

v0.3:

報がご連絡
○○○先生
お世話がおります。○○○先生
お世話になっております。○○学部の〈学生氏名〉です。授業啰名〉です。授業内容について質問がご質問が

v0.4:

締切を確認する順番は、締切、重要度、残り作業を並べ、今日終える一つを決めて予定に入れてください。

Automatic metrics: relevance 23.4 → 40.0; meaningful False → True; EOS False → True.

## 22. 明日の試験は何時ですか？ 科目は日本史で、残り8日です。

Expected category: `general`  
Expected keywords: 確認, 情報, 分かりません

v0.3:

本の情報が登の情報が登録されていないため、現在は分かりません。時間割や大学の案内を確認してください。

v0.4:

試験範囲を確認し、残り1日なら苦手な部分と重要問題に絞って復習しましょう。

Automatic metrics: relevance 100.0 → 40.0; meaningful True → True; EOS True → True.

## 23. 履修科目をどう選べばいい？

Expected category: `registration`  
Expected keywords: 履修, 必修, シラバス

v0.3:

件名：状況を整理しましょう。件名：授ってお世話めましょう。心理学の出件名：授業欠ります。○○先生
お世話になっております。○○学部の〈学生氏名〉です。本日の授業内容について込ご相談しましたく、

v0.4:

時間が少ないないなら、基礎演習は25分だけ勉強し、最後に要点を復習する計画にすると続けやすいです。

Automatic metrics: relevance 24.3 → 25.0; meaningful False → False; EOS False → True.

## 24. 履修科目をどう選べばいい？ 科目は教育学で、残り8日です。

Expected category: `registration`  
Expected keywords: 履修, 必修, シラバス

v0.3:

今できることをことを順番になって番に分けると進めやすいです。情報科学の試験範囲を確認し、基礎、問題演習、苦手部分の復習という順で勉強します。残り1日に合わせて、一日ごとの範囲を決めると安心です。予定には

v0.4:

史の履修は必修区分、時間割、シラバス、卒業要件を順に確認して決めてください。

Automatic metrics: relevance 25.0 → 100.0; meaningful False → True; EOS False → True.

## 25. レポートの構成が決まりません。

Expected category: `report`  
Expected keywords: レポート, 構成, 引用

v0.3:

件名レロいのートってートって牉トってンタ分けると進めると進めやすい手順に分けると進めやすいです。ログミングの試験範囲を確認し、基礎、問題演習、苦手部分の復習という順で勉強します。残り1日に

v0.4:

無理なく進めるなら、線形代数のレポートは問い、資料、構成を先に決め、本文を書いた後に引用を確認してください。

Automatic metrics: relevance 25.0 → 100.0; meaningful False → True; EOS False → True.

## 26. レポートの構成が決まりません。 科目はプログラミングで、残り8日です。

Expected category: `report`  
Expected keywords: レポート, 構成, 引用

v0.3:

件名よって短い手順に分けると進めやすい手順に分けると進めやすいです。ポータ分析の確認、資料集め、構成、執筆、引用確認、推敲に分けます。残り1日なら、まず構成を作るのがおすすめです。無理な場合は担

v0.4:

本席記録を確認し、り1日なら苦手な部分と重要問題に絞って復習しましょう。

Automatic metrics: relevance 85.0 → 25.0; meaningful True → False; EOS False → True.

## 27. 締切と試験が重なっています。今日の優先順位は？

Expected category: `schedule`  
Expected keywords: 予定, 締切, 優先

v0.3:


報に分けると進めると進めると進めやすい手順に分けると進めやすいです。統計学の課題は、提出締切と形式を先に確認します。残り1日なら、未完了の作業を分け、今日の提出準備から始めると進めやすいです。分からない点は早めに大学へ確認す

v0.4:

無理なく進めるなら、経済学の試験範囲を確認し、残り1日なら苦手な部分と重要問題に絞って復習しましょう。

Automatic metrics: relevance 40.0 → 25.0; meaningful True → False; EOS False → True.

## 28. 締切と試験が重なっています。今日の優先順位は？ 科目は社会学で、残り8日です。

Expected category: `schedule`  
Expected keywords: 予定, 締切, 優先

v0.3:

件名：授業授業大学ご連絡

○○先生
お世話になっております。○○学部の〈学生氏名〉です。授業を欠席いたします。直について質問出れる見りまこ。おすいです。分からないとないです。無理

v0.4:

線形代数の課題は締切と作業量を比べ、期限が近いものから提出準備を進めるのがおすめです。

Automatic metrics: relevance 24.5 → 40.0; meaningful False → True; EOS False → True.

## 29. 試験勉強の計画を立てたいです。

Expected category: `study`  
Expected keywords: 勉強, 復習, 計画

v0.3:

件名：授業に分けると進めると進めると進めやすいです。質問名：授業内容についてのご相談
○先生
お世話になっております。○○学部の〈学生氏名〉です。授業内容について質問がご相談しましましましましま

v0.4:

締切を確認すると、統計学の試験範囲を確認し、残り8日なら苦手な部分と重要問題に絞って復習しましょう。

Automatic metrics: relevance 24.6 → 40.0; meaningful False → True; EOS False → True.

## 30. 試験勉強の計画を立てたいです。 科目は情報科学で、残り8日です。

Expected category: `study`  
Expected keywords: 勉強, 復習, 計画

v0.3:

件名：授業授業欠席のご連絡
○○先生
お世話になっております。○○先生
お世話になっております。○学部の〈学生氏名〉です。本日の授業に遅れる見質内れる見迷します。

v0.4:

発表は経授業になってごとに異ります。シラバスと出席記録を確認し、担当教員へ相談してください。

Automatic metrics: relevance 24.2 → 25.0; meaningful False → False; EOS False → True.
