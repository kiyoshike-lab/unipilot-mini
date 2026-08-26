# UniPilot Foundation v1.0 Report

## Corpus

- Documents: 11,499 (unique 11,499)
- Characters: 50,209,324
- Tokens: 37,076,815 (train 36,028,718 / validation 581,767 / test 466,330)
- Sources: Wikipedia 26,108,513 tokens; Wikibooks 10,820,688 tokens; API supplement 147,614 tokens
- License: CC BY-SA 4.0 with per-article attribution and revision metadata
- Semantic duplicates excluded: 106
- Holdout contamination excluded: 0; maximum similarity 0.4286

## Tokenizer

- Selected: Foundation-only byte BPE 4096, trained from scratch on Base train text
- 1024 / 2048 / 4096 tokens per character: 1.0340 / 0.8662 / 0.7440
- 4096 improvement over 2048: 14.12%

## Model comparison at 50 steps

| Model | Parameters | Validation loss | Train tok/s | RAM MB |
|---|---:|---:|---:|---:|
| 20m | 19,514,880 | 7.3443 | 867.00 | 808.84 |
| 30m | 31,036,544 | 7.3552 | 595.42 | 1055.45 |
| 46m | 46,755,840 | 7.2127 | 308.15 | 3020.13 |


Selected sanity model: 20M (19,514,880 parameters, vocab 4096, context 512).

## Learning curve and Base Gate

- 100 steps: loss 7.2147, validation 7.2826, natural Japanese 0%
- 500 steps: loss 6.7218, validation 6.8688, natural Japanese 0%
- Base 100: natural 0%, relevance 0%, completion 0%, runaway 96%
- Base Gate: FAIL

## Decision

Corpus licensing, contamination control, stage separation, and the monotonic loss curve are healthy. Generated Japanese is not established, so Campus/Instruction/DPO must not start. Continue only the 20M Stage-A checkpoint to 1000 as the next sanity point. Do not use 46M yet.
