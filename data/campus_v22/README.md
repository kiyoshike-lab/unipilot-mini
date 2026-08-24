# UniPilot Campus v2.2 data boundary

Campus v2.2 is an opt-in knowledge/RAG expansion. It does not replace the
Campus v2.1 router, tools, production v0.4 model, or their evaluation assets.

The data classes remain physically or logically separate:

- `knowledge/`: reusable external factual sources with full provenance.
- `faq/`: reviewed, project-authored FAQ referenced from Campus v2.
- `instruction/`: router/instruction assets; never indexed as factual evidence.
- `conversation/`: conversation training assets; never indexed as knowledge.
- `corrected/`: correction/replay assets; never indexed as knowledge.
- `rag_index/`: generated index manifest only; the index is built locally.
- `benchmarks/`: independent v2.2 evaluation prompts and human review queue.

Run `python scripts/update_knowledge.py` to refresh the licensed corpus. A page
without an enabled registry entry and an explicit reusable license is excluded.
