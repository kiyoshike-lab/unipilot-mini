"""Verify every PHASE 38 checkpoint can strictly reload and has resume state."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from foundation.diagnostic_transformer_v17 import DiagnosticConfigV17, DiagnosticTransformerV17
from training.train_foundation_v21_ab import file_sha256
SEEDS=(42,123,2026); TOKENS=(6_144_000,7_168_000,8_704_000,10_240_000)
rows=[]
for seed in SEEDS:
 for tokens in TOKENS:
  p=ROOT/f"checkpoints/foundation-v26-current/current/seed-{seed}/checkpoint-tokens-{tokens}.pt"; d=torch.load(p,map_location="cpu",weights_only=False); m=DiagnosticTransformerV17(DiagnosticConfigV17(**d["config"])); m.load_state_dict(d["model_state"],strict=True)
  checks={"strict_reload":True,"optimizer":bool(d.get("optimizer_state",{}).get("state")),"rng":set(d.get("random_state",{}))=={"python","numpy","torch_cpu"},"tokens":int(d.get("tokens_processed",0))==tokens,"sha256":bool(file_sha256(p))}; rows.append({"seed":seed,"tokens":tokens,"path":p.relative_to(ROOT).as_posix(),"checks":checks,"pass":all(checks.values())})
out={"schema":"foundation-v27-checkpoint-verification-v1","phase":38,"expected_checkpoints":12,"verified_checkpoints":len(rows),"integrity_pass":len(rows)==12 and all(r["pass"] for r in rows),"rows":rows}
(ROOT/"evaluation/foundation-v27-checkpoint-verification.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps({"verified":len(rows),"integrity_pass":out["integrity_pass"]}))
