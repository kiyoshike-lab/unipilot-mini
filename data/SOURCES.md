# Data sources

All bundled examples were authored algorithmically for UniPilot Mini and are dedicated to CC0-1.0. No scraped or copyrighted corpus is included. Add future dataset name, URL/author, license, and retrieval date here before training.

## UniPilot Mini v0.2

The 50,000 v0.2 samples were created locally by `scripts/generate_dataset_v02.py` on 2026-08-15. They combine hand-written Japanese templates with synthetic, non-personal scenario variables. Source: UniPilot project original. License: CC0-1.0. No external AI service, scraped corpus, real student/professor name, student number, email address, or phone number is included.

## UniPilot Mini v0.5

The v0.5 instruction conversations are project-authored and dedicated to CC0-1.0. Selected Japanese Wikipedia introductions are fetched through the official MediaWiki Action API by `scripts/collect_wikimedia_v05.py`, stored separately under `data/v05/knowledge`, and are not included in the short v0.5 conversation fine-tuning run. Wikipedia text is attributed per record to Wikipedia contributors with article history, revision, retrieval time, and CC BY-SA 4.0 metadata. See `evaluation/wikimedia-collection-v05.json` and <https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use>.
