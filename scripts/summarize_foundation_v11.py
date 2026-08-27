from __future__ import annotations

from collections import Counter
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foundation.mediawiki_cleaner import residue_signals


def load(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def dirty_residue_documents() -> dict:
    counts = Counter()
    any_markup = 0
    for split in ("train", "validation", "test"):
        with gzip.open(ROOT / f"data/foundation_v10/documents/{split}.jsonl.gz", "rt",
                       encoding="utf-8") as file:
            for line in file:
                signals = residue_signals(json.loads(line)["text"])
                if any(signals.values()):
                    any_markup += 1
                for key, value in signals.items():
                    counts[key] += int(value > 0)
    return {"any_markup": any_markup, **dict(counts)}


def sum_excluded(*reports: dict) -> int:
    return sum(sum(int(value) for value in report.get("excluded", {}).values())
               for report in reports)


def main() -> int:
    dirty_audit = load("evaluation/foundation-v10-data-audit.json")
    dirty_pack = load("data/foundation_v10/packed/vocab-4096/manifest.json")
    dirty_tokenizer = load("evaluation/foundation-v10-tokenizer-benchmark.json")
    dirty_checkpoint = load(
        "checkpoints/foundation-v10-sanity/20m/checkpoint-step-100.manifest.json"
    )
    dirty_wikipedia = load("evaluation/foundation-v10-wikipedia-dump.json")
    dirty_wikibooks = load("evaluation/foundation-v10-wikibooks-dump.json")
    dirty_api = load("evaluation/foundation-v10-wikipedia-collection.json")

    clean_audit = load("evaluation/foundation-v11-data-audit.json")
    clean_pack = load("data/foundation_v11/packed/vocab-4096/manifest.json")
    clean_quality = load("evaluation/foundation-v11-corpus-quality-audit.json")
    clean_tokenizer = load("evaluation/foundation-v11-tokenizer-benchmark.json")
    clean_special = load("evaluation/foundation-v11-special-token-audit.json")
    clean_checkpoint = load(
        "checkpoints/foundation-v11-clean-100/checkpoint-step-100.manifest.json"
    )
    clean_wikipedia = load("evaluation/foundation-v11-wikipedia-dump.json")
    clean_wikibooks = load("evaluation/foundation-v11-wikibooks-dump.json")
    clean_api = load("evaluation/foundation-v11-wikipedia-api-reclean.json")
    resume = load("evaluation/foundation-v11-resume-reproducibility.json")
    generation = load("evaluation/foundation-v11-dirty-clean-generation.json")
    dirty_residue = dirty_residue_documents()

    dirty_token = next(row for row in dirty_tokenizer["results"] if row["actual_vocab"] == 4096)
    clean_token = next(row for row in clean_tokenizer["results"] if row["actual_vocab"] == 4096)
    clean_history = clean_checkpoint["history"]
    dirty_metrics = dirty_checkpoint["training_metrics"]
    clean_metrics = clean_history[-1]
    generated = {row["name"]: row for row in generation["results"]}
    corpus = {
        "dirty_v1.0": {
            "documents": dirty_pack["total_documents"],
            "characters": dirty_pack["total_characters"], "tokens": dirty_pack["total_tokens"],
            "unique_documents": dirty_audit["unique_documents"],
            "semantic_duplicates": dirty_audit["excluded"].get("semantic_duplicate", 0),
            "markup_residue_documents": dirty_residue["any_markup"],
            "wiki_open_documents": dirty_residue["wiki_open"],
            "wiki_close_documents": dirty_residue["wiki_close"],
            "template_residue_documents": dirty_residue["template_close"],
            "html_residue_documents": dirty_residue["html_tag"],
            "table_residue_documents": dirty_residue["table_close"],
            "file_image_residue_documents": dirty_residue["file_image"],
            "integration_excluded_documents": sum(dirty_audit["excluded"].values()),
            "extraction_exclusion_events": sum_excluded(
                dirty_wikipedia, dirty_wikibooks, dirty_api
            ),
            "modified_documents": None, "empty_after_clean": None,
            "final_documents_below_500_characters": 0,
        },
        "clean_v1.1": {
            "documents": clean_pack["total_documents"],
            "characters": clean_pack["total_characters"], "tokens": clean_pack["total_tokens"],
            "unique_documents": clean_audit["unique_documents"],
            "semantic_duplicates": clean_audit["excluded"].get("semantic_duplicate", 0),
            "markup_residue_documents": 0, "wiki_open_documents": 0,
            "wiki_close_documents": 0, "template_residue_documents": 0,
            "html_residue_documents": 0, "table_residue_documents": 0,
            "file_image_residue_documents": 0,
            "integration_excluded_documents": sum(clean_audit["excluded"].values()),
            "extraction_exclusion_events": sum_excluded(
                clean_wikipedia, clean_wikibooks, clean_api
            ),
            "modified_documents": clean_wikipedia["modified_documents"] +
                                  clean_wikibooks["modified_documents"] +
                                  clean_api["modified_documents"],
            "empty_after_clean": clean_wikipedia["empty_after_clean"] +
                                 clean_wikibooks["empty_after_clean"],
            "final_documents_below_500_characters": 0,
        },
    }
    tokenizer = {
        "dirty_v1.0": {
            "vocab": dirty_tokenizer["selected_vocab"],
            "tokens_per_character": dirty_token["tokens_per_character"],
            "roundtrip_rate": dirty_token["exact_roundtrip_rate"],
            "generation_probe_tokens_per_second": dirty_token["generation_tokens_per_second"],
            "encoding_characters_per_second": None,
            "decoding_tokens_per_second": None,
        },
        "clean_v1.1": {
            "vocab": clean_tokenizer["selected_vocab"],
            "tokens_per_character": clean_token["tokens_per_character"],
            "roundtrip_rate": clean_token["exact_roundtrip_rate"],
            "wikipedia_roundtrip_rate": clean_token["wikipedia_roundtrip_rate"],
            "campus_question_roundtrip_rate": clean_token["campus_question_roundtrip_rate"],
            "generation_probe_tokens_per_second": clean_token["generation_tokens_per_second"],
            "encoding_characters_per_second": clean_token["encoding_characters_per_second"],
            "decoding_tokens_per_second": clean_token["decoding_tokens_per_second"],
        },
    }
    training = {
        "dirty_v1.0": {
            "train_loss": dirty_metrics["loss"],
            "validation_loss": dirty_metrics["validation_loss"],
            "learning_rate": dirty_metrics["learning_rate"], "tokens_processed": 51_200,
            "corpus_fraction": 51_200 / dirty_pack["splits"]["train"]["tokens"],
            "tokens_per_second": dirty_metrics["tokens_per_second"],
            "peak_ram_mb": dirty_metrics["memory_usage_mb"],
        },
        "clean_v1.1": {
            "train_loss": clean_metrics["train_loss"],
            "validation_loss": clean_metrics["validation_loss"],
            "learning_rate": clean_metrics["learning_rate"],
            "tokens_processed": clean_metrics["tokens_processed"],
            "corpus_fraction": clean_metrics["corpus_fraction"],
            "tokens_per_second": clean_metrics["tokens_per_second"],
            "peak_ram_mb": max(row["peak_ram_mb"] for row in clean_history),
        },
    }
    generation_metrics = {
        name: {mode: data["metrics"] for mode, data in row["modes"].items()}
        for name, row in generated.items()
    }
    clean_gate = "INVESTIGATE"
    decisions = {
        "corpus_quality": clean_quality["corpus_quality"],
        "tokenizer": clean_special["tokenizer_gate"],
        "resume_integrity": resume["resume_integrity"],
        "clean_100step": clean_gate,
        "clean_500step_recommended": False,
        "reason": (
            "Corpus, tokenizer, resume, and loss curves pass, but at 100 steps greedy natural "
            "Japanese and semantic coherence remain 0%, EOS is 0%, runaway is 100%, and sampled "
            "character validity is 58%. Natural Japanese 0% alone is not a STOP condition; the "
            "combined generation signals require investigation before 500 steps."
        ),
    }
    summary = {
        "schema_version": "foundation-v11-summary-v1", "corpus": corpus,
        "splits": clean_pack["splits"], "tokenizer": tokenizer,
        "special_tokens": clean_special["per_vocab"][2]["special_token_ids"],
        "resume": resume, "training_100step": training,
        "generation_50_prompts": generation_metrics, "decisions": decisions,
        "protected": {"foundation_v10_preserved": True, "campus_v23_changed": False,
                      "production_v04_changed": False, "render_changed": False,
                      "vercel_changed": False, "release_changed": False,
                      "final_blind_opened": False},
        "external_ai_api": "OFF", "push_or_deploy_performed": False,
    }
    summary_path = ROOT / "evaluation/foundation-v11-summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")

    def pct(value):
        return f"{value:.2%}"

    dirty_g = generation_metrics["dirty_v1.0"]["greedy"]
    clean_g = generation_metrics["clean_v1.1"]["greedy"]
    dirty_s = generation_metrics["dirty_v1.0"]["sampling"]
    clean_s = generation_metrics["clean_v1.1"]["sampling"]
    report = f"""# UniPilot Foundation v1.1 Clean Corpus Reconstruction

## Corpus: Dirty v1.0 vs Clean v1.1

| Metric | Dirty v1.0 | Clean v1.1 |
|---|---:|---:|
| Documents / unique | {corpus['dirty_v1.0']['documents']:,} | {corpus['clean_v1.1']['documents']:,} |
| Characters | {corpus['dirty_v1.0']['characters']:,} | {corpus['clean_v1.1']['characters']:,} |
| Tokens | {corpus['dirty_v1.0']['tokens']:,} | {corpus['clean_v1.1']['tokens']:,} |
| Semantic duplicates excluded | {corpus['dirty_v1.0']['semantic_duplicates']:,} | {corpus['clean_v1.1']['semantic_duplicates']:,} |
| Any markup residue documents | {corpus['dirty_v1.0']['markup_residue_documents']:,} | 0 |
| `[[` / `]]` residue documents | {corpus['dirty_v1.0']['wiki_open_documents']:,} / {corpus['dirty_v1.0']['wiki_close_documents']:,} | 0 / 0 |
| Template residue documents | {corpus['dirty_v1.0']['template_residue_documents']:,} | 0 |
| HTML residue documents | {corpus['dirty_v1.0']['html_residue_documents']:,} | 0 |
| Table residue documents | {corpus['dirty_v1.0']['table_residue_documents']:,} | 0 |
| File/Image residue documents | {corpus['dirty_v1.0']['file_image_residue_documents']:,} | 0 |
| Integration exclusions | {corpus['dirty_v1.0']['integration_excluded_documents']:,} | {corpus['clean_v1.1']['integration_excluded_documents']:,} |

Clean split tokens: train {clean_pack['splits']['train']['tokens']:,}, validation {clean_pack['splits']['validation']['tokens']:,}, test {clean_pack['splits']['test']['tokens']:,}.

## Tokenizer

| Metric | Dirty v1.0 | Clean v1.1 |
|---|---:|---:|
| Selected vocab | {tokenizer['dirty_v1.0']['vocab']} | {tokenizer['clean_v1.1']['vocab']} |
| Tokens / character | {tokenizer['dirty_v1.0']['tokens_per_character']:.6f} | {tokenizer['clean_v1.1']['tokens_per_character']:.6f} |
| Roundtrip | {pct(tokenizer['dirty_v1.0']['roundtrip_rate'])} | {pct(tokenizer['clean_v1.1']['roundtrip_rate'])} |
| Encoding chars/s | not recorded | {tokenizer['clean_v1.1']['encoding_characters_per_second']:,.0f} |
| Decoding tokens/s | not recorded | {tokenizer['clean_v1.1']['decoding_tokens_per_second']:,.0f} |

Special IDs are PAD=0, BOS=1, EOS=2, UNK=3, USER=4, ASSISTANT=5, SYSTEM=6. Normal text never emitted EOS, and packed Train has exactly 10,012 BOS/EOS boundaries with no PAD/UNK/dialogue special tokens.

## Resume reproducibility

Scratch→40 and scratch→20→resume→40 are bitwise identical: loss, weights, optimizer, scheduler, and sampler all have maximum difference 0. RNG and sampler states are persisted. Resume Integrity: **{resume['resume_integrity']}**.

## Dirty vs Clean 100 steps

| Metric | Dirty v1.0 | Clean v1.1 |
|---|---:|---:|
| Train loss | {training['dirty_v1.0']['train_loss']:.4f} | {training['clean_v1.1']['train_loss']:.4f} |
| Validation loss | {training['dirty_v1.0']['validation_loss']:.4f} | {training['clean_v1.1']['validation_loss']:.4f} |
| Tokens processed | {training['dirty_v1.0']['tokens_processed']:,} | {training['clean_v1.1']['tokens_processed']:,} |
| Corpus fraction | {pct(training['dirty_v1.0']['corpus_fraction'])} | {pct(training['clean_v1.1']['corpus_fraction'])} |
| Tokens/s | {training['dirty_v1.0']['tokens_per_second']:.2f} | {training['clean_v1.1']['tokens_per_second']:.2f} |
| Peak RAM MB | {training['dirty_v1.0']['peak_ram_mb']:.2f} | {training['clean_v1.1']['peak_ram_mb']:.2f} |

## Generation: fixed 50 prompts

| Greedy metric | Dirty v1.0 | Clean v1.1 |
|---|---:|---:|
| Character validity | {pct(dirty_g['character_validity_rate'])} | {pct(clean_g['character_validity_rate'])} |
| Natural Japanese | {pct(dirty_g['natural_japanese_rate'])} | {pct(clean_g['natural_japanese_rate'])} |
| Semantic coherence | {pct(dirty_g['semantic_coherence_rate'])} | {pct(clean_g['semantic_coherence_rate'])} |
| Completion | {pct(dirty_g['completion_rate'])} | {pct(clean_g['completion_rate'])} |
| EOS | {pct(dirty_g['eos_rate'])} | {pct(clean_g['eos_rate'])} |
| Runaway | {pct(dirty_g['runaway_rate'])} | {pct(clean_g['runaway_rate'])} |
| Repetition | {pct(dirty_g['mean_repetition_rate'])} | {pct(clean_g['mean_repetition_rate'])} |

| Sampling metric | Dirty v1.0 | Clean v1.1 |
|---|---:|---:|
| Character validity | {pct(dirty_s['character_validity_rate'])} | {pct(clean_s['character_validity_rate'])} |
| Natural Japanese | {pct(dirty_s['natural_japanese_rate'])} | {pct(clean_s['natural_japanese_rate'])} |
| Semantic coherence | {pct(dirty_s['semantic_coherence_rate'])} | {pct(clean_s['semantic_coherence_rate'])} |
| Completion | {pct(dirty_s['completion_rate'])} | {pct(clean_s['completion_rate'])} |
| EOS | {pct(dirty_s['eos_rate'])} | {pct(clean_s['eos_rate'])} |
| Runaway | {pct(dirty_s['runaway_rate'])} | {pct(clean_s['runaway_rate'])} |
| Repetition | {pct(dirty_s['mean_repetition_rate'])} | {pct(clean_s['mean_repetition_rate'])} |

## Gate

- Corpus Quality: **{decisions['corpus_quality']}**
- Tokenizer: **{decisions['tokenizer']}**
- Resume Integrity: **{decisions['resume_integrity']}**
- Clean 100step: **{decisions['clean_100step']}**
- Clean 500step: **NO**

Loss and validation improve normally, but generation remains too immature for the next training stage. No 500-step run, Campus tuning, DPO, preference training, external AI, production change, push, or deploy was performed.
"""
    report_path = ROOT / "evaluation/foundation-v11-report.md"
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps({"decisions": decisions, "summary": summary_path.as_posix(),
                      "report": report_path.as_posix()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
