# Foundation v3.0 EOS correction experiment

Experimental only; no official checkpoint was promoted or overwritten.

Seed 42 control and EOS weights 1.25, 1.5, and 2.0 were continued for 256k tokens from the identical 15.360M optimizer-resume state. Weight 1.5 was selected for confirmation: terminal P(EOS) was 0.01302 (control 0.00760), while non-terminal P(EOS) remained 0.000447. Weight 2.0 raised non-terminal P(EOS) more sharply and was classified too aggressive.

Weight 1.5 confirmation terminal P(EOS): seed 42 0.01302, seed 123 0.01449, seed 2026 0.01164. Greedy runaway remained 100% in all confirmations; this is **EOS_FIXED_BUT_GREEDY_ATTRACTOR_REMAINS**. No premature EOS Top-1 was observed, validation loss remained approximately 4.429, and experimental checkpoints strictly reload with baseline SHA unchanged.

Foundation Base completion remains **NO**. No formal 20M continuation is authorized.
