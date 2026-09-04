from __future__ import annotations
import json, statistics as s
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
load=lambda p:json.loads((ROOT/p).read_text(encoding="utf-8"))
seeds=(42,123,2026); milestones=(5_120_000,6_144_000,7_168_000,8_704_000,10_240_000)
runs=[load(f"evaluation/foundation-v27-runs/current-seed-{x}.json") for x in seeds]
def row(r,t): return next(x for x in r["training"]["history"] if x["tokens_processed"]==t)
curve=[]
for t in milestones:
 rows=[row(r,t) for r in runs]; v=[x["validation"] for x in rows]; curve.append({"tokens":t,"loss":s.fmean(x["loss"] for x in v),"top_1":s.fmean(x["top_1_accuracy"] for x in v),"top_5":s.fmean(x["top_5_accuracy"] for x in v),"top_10":s.fmean(x["top_10_accuracy"] for x in v),"corpus_exposure_percent":100*t/33402759})
diag={t:load(f"evaluation/foundation-v23-generation-diagnostics-{t}.json") for t in milestones}
g={t:diag[t]["decoding_comparison"]["temperature_0.7"] for t in milestones}; gr={t:diag[t]["validation_document_prefix"]["metrics"]["free_running"] for t in milestones}
final=curve[-1]; finalg=g[10_240_000]; finalgr=gr[10_240_000]
gate="CONTINUE_15M_GENERATION_LAG" if finalg["semantic_local_syntax_proxy"]>.32 and finalgr["runaway_rate"]>=.9 else "SEMANTIC_PLATEAU_INVESTIGATE"
out={"schema":"foundation-v27-summary-v1","phase":38,"training_curve":curve,"sampling":{str(t):{"naturalness":g[t]["natural_japanese_proxy"],"semantic":g[t]["semantic_local_syntax_proxy"]} for t in milestones},"greedy":{str(t):{"repetition_1":gr[t]["ngram_repetition"]["1"],"runaway":gr[t]["runaway_rate"]} for t in milestones},"intermediate_gate":"CONTINUE_TO_10M","final_gate":gate,"language_emergence":"PARTIAL","base_maturing":True,"foundation_base_complete":False,"checkpoint_integrity":load("evaluation/foundation-v27-checkpoint-verification.json"),"final_blind":{"sha256":"fa7912d58ce251bb10b513f59793bb8ca6c0023b4fe08d1c040b8ccbfe49845b","content_opened":False}}
(ROOT/"evaluation/foundation-v27-summary.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
(ROOT/"evaluation/foundation-v27-report.md").write_text(f"# Foundation v2.7\n\nFinal Gate: **{gate}**\n\n10.240M loss {final['loss']:.4f}; Top-1/5/10 {final['top_1']:.2%}/{final['top_5']:.2%}/{final['top_10']:.2%}. Sampling naturalness/semantic {finalg['natural_japanese_proxy']:.0%}/{finalg['semantic_local_syntax_proxy']:.0%}; greedy repetition-1/runaway {finalgr['ngram_repetition']['1']:.3f}/{finalgr['runaway_rate']:.0%}.\n",encoding="utf-8")
print(json.dumps({"gate":gate,"loss":final["loss"]}))
