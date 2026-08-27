# UniPilot Foundation v1.0 1000-step Preflight

## Result

Preflight: **FAIL**. The 500-to-1000 training run was not started.

## Blocking findings

1. The packed stream has correct `BOS document EOS BOS document EOS` boundaries, but the
   cleaned source corpus still contains residual MediaWiki markup. Eight of the first 20
   audited documents contain `[[` or `]]`. Across 11,499 documents, 2,839 contain a wiki-link
   signal, 252 contain a template signal, 19 contain an HTML-tag signal, and four contain a
   wiki-table signal. The current image/link regular expression stops at the first closing
   bracket and leaves nested caption markup behind.
2. The step-500 checkpoint contains model state, 124 optimizer state entries, configuration,
   and global step, but predates RNG persistence. It has no Python/PyTorch random state. The
   scheduler is stateless and can be reconstructed from global step; exact dropout-RNG resume
   still requires a tested legacy recovery or a clean restart.

## EOS and packing

- Special token IDs: PAD 0, BOS 1, EOS 2, UNK 3.
- Train documents/BOS/EOS: 11,143 / 11,143 / 11,143.
- Inter-document `EOS -> BOS` transitions: 11,142 (all boundaries).
- EOS targets covered by full 512-token training blocks: 11,142 / 11,143 (99.991%).
- EOS is a normal cross-entropy target; only `-100` is ignored.
- First 20 source documents round-trip exactly through the tokenizer and match the packed
  prefix token-for-token.

## Protected data

The Final Blind file was hashed as bytes only and was not parsed. Its SHA256 remains
`fa7912d58ce251bb10b513f59793bb8ca6c0023b4fe08d1c040b8ccbfe49845b`.

## Required next action

Fix nested MediaWiki link/image removal, rebuild and re-audit the Base corpus and tokenizer,
then restart the Base sanity run from scratch. Continuing the existing step-500 model on a
different cleaned corpus would not be a controlled 500-to-1000 comparison.
