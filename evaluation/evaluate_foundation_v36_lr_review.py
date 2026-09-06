"""Fixed-population CPU audit and unchanged Phase 46 evaluations for PHASE 47."""
from __future__ import annotations
import argparse
import hashlib
import math
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from training.run_foundation_v36_lr_review import EVAL, SEEDS, checkpoint, official, read_json, write_json, fingerprint
from evaluation import evaluate_foundation_v35_short_gate as v35
from foundation.base_tokenizer import FoundationTokenizer
from foundation.diagnostic_transformer_v17 import DiagnosticConfigV17, DiagnosticTransformerV17
from training.train_foundation_v21_ab import frequency_ranks, file_sha256

BUCKETS = ("top_1_percent", "top_5_percent_excluding_top_1", "top_20_percent_excluding_top_5", "middle_20_to_80_percent", "rare_bottom_20_percent")


def bucket_masks(targets, ranks):
    edges = [0] + [math.ceil(len(ranks)*p) for p in (.01,.05,.2,.8)] + [len(ranks)]
    return {name: (ranks[targets] >= low) & (ranks[targets] < high) for name, low, high in zip(BUCKETS, edges[:-1], edges[1:])}


def probability_metrics(probabilities):
    a = np.asarray(probabilities, dtype=np.float64)
    if not a.size:
        return {"sample_count": 0, "mean": None, "median": None, "q25": None, "q75": None, "ce": None}
    return {"sample_count": len(a), "mean": float(a.mean()), "median": float(np.median(a)),
            "q25": float(np.percentile(a,25)), "q75": float(np.percentile(a,75)),
            "ce": float(-np.log(a.clip(1e-30)).mean())}


def load_model(path):
    p = torch.load(path, map_location="cpu", weights_only=False)
    m = DiagnosticTransformerV17(DiagnosticConfigV17(**p["config"]))
    m.load_state_dict(p["model_state"], strict=True); m.eval()
    return p, m


@torch.inference_mode()
def frequency_probe(model, validation, ranks):
    assigned, tops, truth = [], [], []
    for start in range(0,8192,512):
        a = torch.from_numpy(np.asarray(validation[start:start+513], dtype=np.int64).copy())
        logits, _ = model(a[:-1][None])
        z = logits[0].float()
        assigned.extend(z.softmax(-1).gather(1,a[1:,None]).squeeze(1).tolist())
        tops.extend(z.topk(10,-1).indices.tolist()); truth.extend(a[1:].tolist())
    ids, probs, top = np.asarray(truth), np.asarray(assigned), np.asarray(tops)
    result = {"population_sha256": fingerprint(ids), "rank_sha256": fingerprint(ranks), "buckets": {}}
    for name, mask in bucket_masks(ids,ranks).items():
        result["buckets"][name] = {
            **probability_metrics(probs[mask]),
            **{f"top{k}": float((top[mask,:k] == ids[mask,None]).any(-1).mean()) for k in (1,5,10)},
            "positions": (np.flatnonzero(mask)+1).tolist(), "token_ids": ids[mask].tolist(),
            "probabilities": probs[mask].tolist(), "per_position_ce": (-np.log(probs[mask].clip(1e-30))).tolist(),
        }
    return result


def token_class(tok, token_id):
    if token_id in tok.special_to_id.values(): return "special_token"
    text = tok.decode([token_id]).strip()
    if not text or '\ufffd' in text: return "other_or_partial_byte_fragment"
    if text.isnumeric(): return "number"
    if all(unicodedata.category(c).startswith('P') for c in text): return "punctuation"
    if all(unicodedata.category(c).startswith('S') for c in text): return "symbol"
    if all(ord(c)<128 and c.isalpha() for c in text): return "Latin"
    if any('\u3040'<=c<='\u9fff' for c in text): return "Japanese_character_or_subword"
    return "other"


def concentration(before, after):
    assert before["positions"] == after["positions"] and before["token_ids"] == after["token_ids"]
    delta = np.asarray(after["per_position_ce"]) - before["per_position_ce"]
    contributions = {}
    for token, change in zip(before["token_ids"], delta):
        contributions[token] = contributions.get(token,0.) + float(change)
    ordered = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
    positive = sum(max(0,v) for _,v in ordered)
    return {"ce_delta": float(delta.mean()), "fraction_positions_worse": float(np.mean(delta>0)),
            "median_position_delta": float(np.median(delta)),
            "top10_positive_contribution_share": sum(max(0,v) for _,v in ordered[:10])/positive if positive else 0.,
            "unique_tokens": len(contributions),
            "top50": [{"token_id": int(t), "ce_sum_delta": v, "mean_ce_contribution": v/len(delta)} for t,v in ordered if v > 0][:50]}


def frequency_classification(details, evaluator_match=True):
    if not evaluator_match: return "EVALUATOR_ARTIFACT"
    material = [d for d in details if d["ce_delta"] > .05]
    if not material: return "NORMAL_VARIANCE"
    if len(material) < len(details): return "SEED_LOCAL_VARIANCE"
    if all(d["top10_positive_contribution_share"] >= .5 for d in material): return "OUTLIER_DRIVEN"
    return "TRUE_RARE_REGRESSION"


def audit():
    tok = FoundationTokenizer.load(ROOT/"tokenizer/foundation-v11-base-4096.json")
    train = np.memmap(ROOT/"data/foundation_v11/packed/vocab-4096/train.bin",dtype=np.uint16,mode="r")
    val = np.memmap(ROOT/"data/foundation_v11/packed/vocab-4096/validation.bin",dtype=np.uint16,mode="r")
    ranks = frequency_ranks(train,tok.vocab_size)
    pairs, trajectory, deterministic, matches = {}, {}, None, []
    for seed in SEEDS:
        pair = []
        training_counts = np.zeros(tok.vocab_size, dtype=np.int64)
        for final in (False,True):
            path=official(seed, final); payload, model=load_model(path)
            if not final:
                for block in payload['permutation'][payload['update']:payload['update']+500]:
                    start=int(block)*512
                    training_counts += np.bincount(train[start+1:start+513],minlength=tok.vocab_size)
            probe=frequency_probe(model,val,ranks)
            if seed==42 and not final:
                repeated=frequency_probe(model,val,ranks)
                deterministic=fingerprint(probe)==fingerprint(repeated)
            stage='gate1' if final else 'baseline'
            reference=read_json(ROOT/f"evaluation/phase46/{stage}/seed-{seed}.json")
            errors={name: abs(probe['buckets'][name]['ce']-reference['validation']['frequency_buckets'][name]['cross_entropy']) for name in BUCKETS}
            matches.append(max(errors.values())<1e-6)
            probe.update({"checkpoint_sha256":file_sha256(path),"phase46_ce_absolute_errors":errors})
            write_json(EVAL/f"audit/{stage}-seed-{seed}.json",probe)
            pair.append(probe)
        details=concentration(pair[0]['buckets'][BUCKETS[-1]],pair[1]['buckets'][BUCKETS[-1]])
        for row in details['top50']:
            row.update({"text":tok.decode([row['token_id']]),"raw_token":tok.backend.id_to_token(row['token_id']),"taxonomy":token_class(tok,row['token_id']),"training_interval_occurrences":int(training_counts[row['token_id']])})
        rare_ids=np.flatnonzero(ranks>=math.ceil(tok.vocab_size*.8))
        probe_ids=np.unique(pair[0]['buckets'][BUCKETS[-1]]['token_ids'])
        details['training_interval_exposure']={
            'all_tokens':int(training_counts.sum()), 'rare_occurrences':int(training_counts[rare_ids].sum()),
            'probe_rare_token_types_absent':int(np.sum(training_counts[probe_ids]==0)),
            'probe_rare_token_types':len(probe_ids),
            'note':'Observed exposure counts, not a causal test of corpus composition.'}
        details['taxonomy_top50_counts']=dict(Counter(r['taxonomy'] for r in details['top50']))
        details['baseline']=probability_metrics(pair[0]['buckets'][BUCKETS[-1]]['probabilities'])
        details['final']=probability_metrics(pair[1]['buckets'][BUCKETS[-1]]['probabilities'])
        pairs[str(seed)]=details
        trajectory[str(seed)]={}
        for tokens,phase,stage in ((15360000,44,'baseline'),(15616000,44,'gate1'),(15872000,46,'baseline'),(16128000,46,'gate1')):
            old=read_json(ROOT/f"evaluation/phase{phase}/{stage}/seed-{seed}.json")
            trajectory[str(seed)][str(tokens)]={name:old['validation']['frequency_buckets'][name] for name in BUCKETS[-2:]}
        print(f"Rare audit seed {seed}: delta={details['ce_delta']:.6f}, worse={details['fraction_positions_worse']:.1%}",flush=True)
    result={"phase":47,"population_fixed":True,"deterministic_repeat":deterministic,"phase46_metric_match":all(matches),
            "rank_sha256":fingerprint(ranks),"train_sha256":file_sha256(ROOT/'data/foundation_v11/packed/vocab-4096/train.bin'),
            "validation_sha256":file_sha256(ROOT/'data/foundation_v11/packed/vocab-4096/validation.bin'),
            "bucket_definition":"Train token-frequency ranks; disjoint 0-1%, 1-5%, 5-20%, 20-80%, 80-100% vocabulary ranks, ceil boundaries; 8192 fixed validation targets; occurrence-weighted arithmetic mean of -log P(correct). No checkpoint-dependent membership.",
            "tie_policy":"Existing NumPy argsort(counts)[::-1], fixed population/rank hash; no ranking change in PHASE47.",
            "classification":frequency_classification(list(pairs.values()),all(matches) and deterministic),"per_seed":pairs,"trajectory":trajectory,
            "limitations":["Rare probe is a small fixed validation subset, not all rare-language ability.","Taxonomy describes decoded fragments; proper names/foreign words cannot reliably be inferred from isolated BPE bytes.","Classification thresholds are descriptive: >0.05 CE in all seeds for broad regression; otherwise seed-local; top10 >=50% positive contributions for concentration."]}
    write_json(ROOT/'evaluation/foundation-v36-rare-analysis.json',result)


def evaluate_arm(arm, seed):
    path=checkpoint(arm,seed); before=file_sha256(path); payload, model=load_model(path)
    tok=FoundationTokenizer.load(ROOT/'tokenizer/foundation-v11-base-4096.json')
    train=np.memmap(ROOT/'data/foundation_v11/packed/vocab-4096/train.bin',dtype=np.uint16,mode='r')
    val=np.memmap(ROOT/'data/foundation_v11/packed/vocab-4096/validation.bin',dtype=np.uint16,mode='r')
    ranks=frequency_ranks(train,tok.vocab_size)
    positions=v35.context_positions(val,tok)
    prefixes=v35.build_prefixes(val,v35.document_ranges(val,tok.bos_id,tok.eos_id),tok)
    terminal=np.resize(np.asarray([int(p) for p in np.flatnonzero(val==tok.eos_id) if p>=128]),500)
    nonterminal=np.linspace(128,len(val)-2,500,dtype=int)
    result={"phase":47,"arm":arm,"seed":seed,"tokens_processed":payload['tokens_processed'],"checkpoint_sha256":before,
            "context":v35.context_profile(model,val,positions),"sanity":v35.sanity_checks(model,payload,positions,val,tok),
            "validation":v35.language_metrics_detailed(model,val,ranks),"frequency_probe":frequency_probe(model,val,ranks),
            "terminal_eos":v35.target_metrics(model,val,terminal,tok.eos_id),"nonterminal_eos":v35.target_metrics(model,val,nonterminal,tok.eos_id),
            "generation":v35.generation_metrics(model,tok,prefixes),"teacher_forced_horizons":v35.teacher_forced_horizons(model,val,prefixes),
            "evaluation_execution":{"device":"cpu","parallel_cpu_evaluation":"DISABLED","torch_threads":torch.get_num_threads()},
            "sampling_note":"Naturalness/semantic/completion/topic retention are fixed automatic proxies, not human ratings."}
    result['checkpoint_unchanged']=file_sha256(path)==before
    assert result['checkpoint_unchanged']
    write_json(EVAL/f'arm-{arm}/seed-{seed}-evaluation.json',result)
    print(f"Arm {arm} seed {seed}: loss={result['validation']['loss']:.6f}, rare_ce={result['frequency_probe']['buckets'][BUCKETS[-1]]['ce']:.6f}",flush=True)


def taxonomy_from_saved_probes():
    """Compare class contribution with class population, without model reruns."""
    tok=FoundationTokenizer.load(ROOT/'tokenizer/foundation-v11-base-4096.json')
    path=ROOT/'evaluation/foundation-v36-rare-analysis.json'
    audit=read_json(path)
    for seed in SEEDS:
        before=read_json(EVAL/f'audit/baseline-seed-{seed}.json')['buckets'][BUCKETS[-1]]
        after=read_json(EVAL/f'audit/gate1-seed-{seed}.json')['buckets'][BUCKETS[-1]]
        classes={}
        for token,old,new in zip(before['token_ids'],before['per_position_ce'],after['per_position_ce']):
            name=token_class(tok,token)
            row=classes.setdefault(name,{'occurrences':0,'ce_delta_sum':0.,'worse_occurrences':0})
            row['occurrences']+=1; row['ce_delta_sum']+=new-old; row['worse_occurrences']+=int(new>old)
        for row in classes.values():
            row['mean_ce_delta']=row['ce_delta_sum']/row['occurrences']
            row['contribution_to_rare_ce']=row['ce_delta_sum']/len(before['token_ids'])
        audit['per_seed'][str(seed)]['population_class_comparison']=classes
    audit['taxonomy_caveat']='Class contributions must be compared with occurrence counts; Japanese fragments dominate this Japanese probe, so counts alone do not establish a class-specific defect.'
    write_json(path,audit)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--audit',action='store_true'); parser.add_argument('--taxonomy',action='store_true'); parser.add_argument('--arm',choices=('A','B','C')); parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args(); torch.set_num_threads(4)
    if args.audit: audit()
    elif args.taxonomy: taxonomy_from_saved_probes()
    else: evaluate_arm(args.arm,args.seed)
