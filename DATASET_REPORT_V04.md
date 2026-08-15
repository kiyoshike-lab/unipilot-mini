# UniPilot Mini v0.4 Dataset Report

Dataset versionは`unipilot-clean-conversation-v04`。ローカル規則生成のみで8,000件を作成し、train 7,200 / validation 400 / test 400へtemplate family単位で分離した。

| Category | Samples |
|---|---:|
| Assignment | 1,100 |
| Exam | 1,100 |
| Study | 1,000 |
| Credit | 700 |
| Email | 900 |
| Registration | 600 |
| Attendance | 600 |
| Report | 500 |
| Presentation | 400 |
| Schedule | 600 |
| General | 300 |
| Unknown | 200 |

平均回答長53.78文字、中央値50文字。EOS coverage 100%、EOS直前自然終止100%、最大opening 7.44%、pair exact duplicate 0、near duplicate 0（空白・句読点正規化による完全一致）、template-family leak 0、category contamination 0%、broken sample 0。Assistant回答の完全一致は6,974件あるため、表現多様性にはなお改善余地がある。

IntentはASK_PRIORITY 1,100、ASK_EXAM_PLAN 1,100、ASK_STUDY_PLAN 1,000、ASK_EMAIL 900、ASK_CREDIT_RISK 700、ASK_REGISTRATION/ASK_ABSENCE/ASK_DAILY_PLAN各600、ASK_REPORT 500、ASK_PRESENTATION 400、ASK_GENERAL_ADVICE 300、ASK_UNKNOWN_INFORMATION 200。
