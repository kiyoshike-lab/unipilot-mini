# Foundation v2.9 Generation-Lag Diagnostic

## Scope and protection

Read-only PHASE 40 diagnosis of the three official 15.360M checkpoints. No training, checkpoint mutation, corpus change, production decoding change, Render, or Vercel action was performed. Checkpoint SHA-256 values were equal before and after diagnostics.

## Audit verdict

| Check | Result |
|---|---:|
| Generation implementation bug | NO |
| Special-token bug | NO |
| Packing boundary issue | NO |
| Train/eval mismatch | NO |
| GPU regression | NO |

Tokenizer special IDs: `{'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3, '<USER>': 4, '<ASSISTANT>': 5, '<SYSTEM>': 6}`. The 100-case visible-text BPE roundtrip rate is 89.0%; EOS visibility and stripping both pass at 100%.

## EOS and boundary evidence

Train EOS exposure: 10,012 EOS over 10,012 documents (1.000/document; 3336.3 tokens/EOS). Packed BOS is preceded by EOS at 100.0%; boundary loss is not observed.

At 500 held-out document ends per seed, mean P(EOS) is 0.0093; EOS Top-1/Top-5/Top-10 are 0.0%/37.0%/63.7%. Full numbers, historical trajectory, and non-EOS boundary metrics are in the JSON summary.

## Generation dynamics

Across 100 fixed held-out prefixes × three 15.360M seeds (128-token greedy traces): runaway is 100.0%, repetition-1 is 0.925, Naturalness is 2.3%, and Semantic coherence is 1.0%. FIRST_BREAK: なし.

Median detected loop onset is 20.0; taxonomy and first-loop transitions are stored in the trace artifact. Candidate recovery on 44 loop onsets is PARTIALLY_RECOVERABLE.

## Conclusion and gate

Primary cause: `EOS_LEARNING_LAG`. Secondary cause: `GREEDY_ATTRACTOR`. The diagnostic gate is **DECODING_LAYER_CAN_MITIGATE_BUT_BASE_NOT_READY**. Continue to 20M GPU: **NO**. Foundation Base completion remains **NO**.

Thermal diagnostic maximum was 56.0°C; throttling evidence: NO.
