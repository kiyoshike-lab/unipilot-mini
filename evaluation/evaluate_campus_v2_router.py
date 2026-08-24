from __future__ import annotations

from collections import Counter
import argparse
import gc
import json
from pathlib import Path
import statistics
from time import perf_counter

from pipeline.campus_retrieval import load_jsonl
from pipeline.campus_router import CampusBM25Router, CampusRuleRouter, CampusTfidfRouter
from pipeline.campus_router_v2 import CampusRouterV2, CampusSklearnRouter


ROOT = Path(__file__).resolve().parents[1]


def rss_mb() -> float | None:
    try:
        import os
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024**2
    except (ImportError, OSError):
        pass
    try:
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        counters = Counters(); counters.cb = ctypes.sizeof(counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return counters.WorkingSetSize / 1024**2
    except (AttributeError, OSError):
        pass
    return None


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def load_split(name: str) -> list[dict]:
    paths = {
        "dev": ROOT / "data" / "campus_v2" / "router" / "dev.json",
        "blind": ROOT / "data" / "campus_v2" / "blind" / "evaluation-2000.json",
        "adversarial": ROOT / "data" / "campus_v2" / "adversarial" / "negation-300.json",
    }
    return json.loads(paths[name].read_text(encoding="utf-8"))


def instantiate(name: str, train: list[dict]):
    if name == "rules": return CampusRuleRouter()
    if name == "bm25": return CampusBM25Router(train)
    if name == "tfidf_char_centroid": return CampusTfidfRouter(train)
    if name == "char_ngram_svm": return CampusSklearnRouter(train, "char_svm")
    if name == "word_ngram_svm": return CampusSklearnRouter(train, "word_svm")
    if name == "logistic_regression": return CampusSklearnRouter(train, "logistic")
    if name == "linear_svm": return CampusSklearnRouter(train, "char_svm")
    if name == "naive_bayes": return CampusSklearnRouter(train, "naive_bayes")
    if name == "hierarchical_svm": return CampusSklearnRouter(train, "char_svm", hierarchical=True)
    if name == "hierarchical_hybrid": return CampusRouterV2(train)
    raise ValueError(name)


def evaluate(name: str, train: list[dict], rows: list[dict]) -> dict:
    before = rss_mb(); fit_started = perf_counter(); router = instantiate(name, train)
    fit_seconds = perf_counter() - fit_started; after = rss_mb()
    latencies, correct, action_correct, multi_correct = [], 0, 0, 0
    bands, failures = Counter(), []
    for item in rows:
        question = item.get("question") or item["prompt"]
        started = perf_counter()
        if name == "hierarchical_hybrid":
            decision = router.decide(question)
            predicted, confidence = decision.primary, decision.confidence
            action, intents, band = decision.action, set(decision.intents), decision.confidence_band
        else:
            predicted, confidence, _ = router.predict(question)
            action, intents, band = None, {predicted}, None
        latency = (perf_counter() - started) * 1000
        latencies.append(latency)
        is_correct = predicted == item["category"]
        correct += is_correct
        if action is not None:
            action_correct += action == item["expected_action"]
        expected_intents = set(item.get("intent_labels", [item["category"]]))
        multi_match = expected_intents.issubset(intents)
        multi_correct += multi_match
        if band: bands[band] += 1
        if not is_correct and len(failures) < 100:
            failures.append({"id": item["id"], "question": question, "expected": item["category"],
                             "predicted": predicted, "confidence": confidence})
    result = {
        "method": name, "split": args.split, "questions": len(rows), "accuracy": correct / len(rows),
        "action_accuracy": action_correct / len(rows) if name == "hierarchical_hybrid" else None,
        "multi_intent_recall": multi_correct / len(rows) if name == "hierarchical_hybrid" else None,
        "fit_seconds": fit_seconds, "mean_latency_ms": statistics.fmean(latencies),
        "p95_latency_ms": percentile(latencies, .95), "p99_latency_ms": percentile(latencies, .99),
        "rss_before_mb": before, "rss_after_fit_mb": after,
        "rss_delta_mb": None if before is None or after is None else after - before,
        "confidence_bands": dict(bands), "sample_failures": failures,
    }
    del router; gc.collect()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "blind", "adversarial"), default="dev")
    parser.add_argument("--selected-only", action="store_true")
    args = parser.parse_args()
    train = load_jsonl(ROOT / "data" / "campus_v2" / "router" / "train.jsonl")
    rows = load_split(args.split)
    names = ["hierarchical_hybrid"] if args.selected_only else [
        "rules", "bm25", "tfidf_char_centroid", "char_ngram_svm", "word_ngram_svm",
        "logistic_regression", "linear_svm", "naive_bayes", "hierarchical_svm", "hierarchical_hybrid",
    ]
    results = []
    for method in names:
        result = evaluate(method, train, rows); results.append(result)
        print(method, round(result["accuracy"], 4), round(result["p95_latency_ms"], 3))
    eligible = [item for item in results if item["p95_latency_ms"] < 20]
    selected = max(eligible, key=lambda item: (item["accuracy"], item["method"] == "hierarchical_hybrid"))["method"] if eligible else None
    output = {"split": args.split, "train": len(train), "selection_policy": "highest dev accuracy with P95 below 20ms; ties prefer multi-intent/action-capable hierarchical hybrid",
              "selected": selected if args.split == "dev" else "hierarchical_hybrid (frozen on dev)",
              "results": results, "external_ai_api": "OFF"}
    path = ROOT / "evaluation" / f"campus-v2-router-{args.split}.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
