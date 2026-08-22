from __future__ import annotations

import json
from pathlib import Path
import time

from pipeline.classifier import BM25CategoryClassifier
from pipeline.rag import KnowledgeRetriever


def load_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    prompts = json.loads(Path("evaluation/fixed_prompts_v07.json").read_text(encoding="utf-8"))
    classifier = BM25CategoryClassifier(load_jsonl("data/v07/classifier/train.jsonl"))
    retriever = KnowledgeRetriever.from_jsonl("data/v07/knowledge/documents.jsonl")
    results = []
    for top_k in (1, 3, 5):
        rows = []
        started = time.perf_counter()
        for item in prompts:
            category = classifier.predict(item["prompt"])[0]
            documents = retriever.retrieve(item["prompt"], category, top_k)
            expected = item.get("expected_keywords", [])
            text = " ".join(document.get("answer") or document["text"] for document in documents)
            rows.append({"id": item["id"], "category_correct": category == item["category"],
                         "retrieved_category_hit": any(document["category"] == item["category"] for document in documents),
                         "keyword_hit": not expected or any(word in text for word in expected),
                         "document_ids": [document["id"] for document in documents]})
        elapsed = time.perf_counter() - started
        results.append({"top_k": top_k, "questions": len(rows),
                        "category_accuracy": sum(row["category_correct"] for row in rows) / len(rows),
                        "retrieved_category_hit_rate": sum(row["retrieved_category_hit"] for row in rows) / len(rows),
                        "retrieved_keyword_hit_rate": sum(row["keyword_hit"] for row in rows) / len(rows),
                        "mean_classifier_plus_retrieval_ms": elapsed * 1000 / len(rows), "rows": rows})
    selected = max(results, key=lambda row: (row["retrieved_keyword_hit_rate"], -row["mean_classifier_plus_retrieval_ms"]))
    report = {"knowledge_documents": len(retriever.documents), "results": results, "selected_top_k": selected["top_k"],
              "selection": "Highest keyword hit, then lowest latency", "external_ai_api": "OFF"}
    Path("evaluation/retrieval-benchmark-v07.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"knowledge_documents": report["knowledge_documents"], "selected_top_k": report["selected_top_k"],
                      "results": [{key: value for key, value in row.items() if key != "rows"} for row in results]},
                     ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
