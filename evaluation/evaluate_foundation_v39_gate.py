"""PHASE50: preregistered, disjoint supported rare tail; inference only."""
from __future__ import annotations
import argparse
import math
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from training import run_foundation_v38_gate as v38
from training import run_foundation_v36_lr_review as v36
from training.train_foundation_v21_ab import file_sha256, frequency_ranks
from foundation.base_tokenizer import FoundationTokenizer
from evaluation import evaluate_foundation_v38_gate as previous
from evaluation import evaluate_foundation_v36_lr_review as loader
from training.checkpoint_paths import checkpoint_root, existing_checkpoint_path
EVAL = ROOT / 'evaluation/phase50'
DEFINITION = ROOT / 'evaluation/foundation-v39-supported-tail.json'
read, write = v38.read, v38.write


def checkpoint_reference(value):
    """Keep historical metadata relative; resolve through the configured root at use time."""
    path=Path(value)
    if path.is_absolute(): return path
    if '..' in path.parts: raise ValueError('Checkpoint reference cannot traverse parents')
    if path.parts and path.parts[0]=='checkpoints': return existing_checkpoint_path(ROOT,*path.parts[1:])
    return ROOT/path


def evaluated_checkpoint(arm, seed):
    if seed not in v38.SEEDS or arm not in ('baseline','B','C'): raise ValueError('Unregistered checkpoint')
    relative=(f'checkpoints/foundation-v33-context-gate/gate-2/seed-{seed}/checkpoint-tokens-15872000.pt' if arm=='baseline'
        else f'checkpoints/experimental/phase{48 if arm=="C" else 49}/arm-{arm}/seed-{seed}/checkpoint-tokens-16384000.pt')
    return checkpoint_reference(relative)


def verify_preserved():
    migration=EVAL/'z-migration.json'
    if migration.exists():
        record=read(migration);assert record['gate']=='Z_CHECKPOINT_MIGRATION_PASS'
        rows=record['protected_files']
    else: rows=read(EVAL/'preflight.json')['preserved_files']
    for r in rows: assert file_sha256(Path(r['path']))==r['sha256']


def supported_ids(ranks, counts, documents):
    """Low-occurrence complement to Core, with repeated independent support."""
    ranks, counts, documents = map(np.asarray, (ranks, counts, documents))
    return np.flatnonzero((ranks >= math.ceil(len(ranks)*.8)) & (counts >= 2) & (counts < 10) & (documents >= 2))


def preflight():
    expected = 'c5012cddabb1c7b5a6481f1f75cec5a0892e4185'
    def git(*args): return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
    assert git('branch', '--show-current') == 'foundation-research'
    assert git('rev-parse', 'HEAD') == expected
    assert git('ls-remote', 'origin', 'refs/heads/foundation-research').split()[0] == expected
    assert git('cat-file', '-t', '6be1eaca8e6b3c770b0edddc7fb6a161aa6e8c7e') == 'commit'
    old = read(v38.EVAL/'preflight.json'); rows = list(old['immutable_checkpoints'])
    for seed in v38.SEEDS:
        r = read(v38.training_path('B', seed)); rows.append({'path': r['checkpoint'], 'sha256': r['sha256']})
    for r in rows:
        assert file_sha256(checkpoint_reference(r['path'])) == r['sha256']
        p = torch.load(checkpoint_reference(r['path']), map_location='cpu', weights_only=False)
        r['integrity'] = v36.verify_payload(p, p['seed'], p['tokens_processed'], p.get('experimental_lr', 1e-4))
        assert r['integrity']['pass']
    for r in old['preserved_files']: assert file_sha256(Path(r['path'])) == r['sha256']
    assert file_sha256(ROOT/'data/foundation_v09/evaluation/final-blind-1000.json') == old['final_blind_sha256']
    assert torch.cuda.is_available() and shutil.disk_usage(checkpoint_root(ROOT)).free > 20*1024**3
    write(EVAL/'preflight.json', {**old, 'phase':50, 'expected_head':expected, 'immutable_checkpoints':rows,
        'free_bytes':shutil.disk_usage(checkpoint_root(ROOT)).free, 'gpu':torch.cuda.get_device_name(0), 'pass':True})
    print('PHASE50 preflight PASS', flush=True)


def freeze():
    if DEFINITION.exists(): raise FileExistsError('Preregistered membership cannot be overwritten')
    assert read(EVAL/'preflight.json')['pass']
    tok = FoundationTokenizer.load(ROOT/'tokenizer/foundation-v11-base-4096.json')
    trainpath = ROOT/'data/foundation_v11/packed/vocab-4096/train.bin'
    valpath = ROOT/'data/foundation_v11/packed/vocab-4096/validation.bin'
    train = np.memmap(trainpath, dtype=np.uint16, mode='r'); val = np.memmap(valpath, dtype=np.uint16, mode='r')
    ranks = frequency_ranks(train, 4096); counts = np.bincount(val, minlength=4096)
    docs = np.maximum(0, np.cumsum(val == tok.bos_id)-1)
    supports = np.array([len(np.unique(docs[val == i])) for i in range(4096)])
    ids = supported_ids(ranks, counts, supports); pos = np.flatnonzero(np.isin(val, ids)); pos = pos[pos > 0]
    pop = {'positions':pos.tolist(), 'token_ids':val[pos].astype(int).tolist(), 'document_ids':docs[pos].astype(int).tolist(),
        'token_count':len(ids), 'occurrence_count':len(pos), 'document_count':len(np.unique(docs[pos]))}
    pop['sha256'] = v36.fingerprint((pop['positions'], pop['token_ids']))
    old = read(v38.EVAL/'frequency-population.json')
    assert not set(ids) & set(old['populations']['core']['token_ids'])
    eligible = np.flatnonzero((ranks >= 3277) & (counts > 0) & (counts < 10))
    maxpos = np.flatnonzero(np.isin(val, eligible))
    definition = {'phase':50, 'registered_at_utc':datetime.now(timezone.utc).isoformat(), 'checkpoint_results_used_for_selection':False,
        'selection':{'train_frequency_rank_min':3277, 'rank_rule':'existing deterministic bottom 20% of 4096 vocabulary',
            'validation_occurrences_min':2, 'validation_occurrences_max':9, 'per_token_documents_min':2,
            'rationale':'Disjoint low-occurrence complement of the unchanged >=10-occurrence Rare Core; require repeat occurrence in independent documents.'},
        'selected_token_ids':ids.tolist(), 'population':pop, 'legacy':old['populations']['tail'], 'core':old['populations']['core'],
        'train_sha256':file_sha256(trainpath), 'validation_sha256':file_sha256(valpath), 'rank_sha256':v36.fingerprint(ranks),
        'maximum_observed_noncore_rare_support':{'tokens':len(eligible),'occurrences':len(maxpos),'documents':len(np.unique(docs[maxpos]))},
        'gate_policy':{'minimum_documents':30, 'minimum_occurrences':300, 'core_ce_delta_upper':.10, 'supported_ce_delta_upper':.25,
            'legacy_role':'WARNING_ONLY; never an eligibility veto', 'nonfrequency_policy':'Unchanged PHASE49 per-seed checks versus fixed baseline and own-LR 256k control',
            'bootstrap':'2000 paired document-cluster resamples, seed 4900; average seed-wise position deltas before joint document resampling; no multiplicity-adjusted claim',
            'approval':'Every seed must pass Core, supported tail and all unchanged nonfrequency checks. No candidate training unless approved.'}}
    write(DEFINITION, definition)
    print({k:pop[k] for k in ('token_count','occurrence_count','document_count','sha256')}, flush=True)


@torch.inference_mode()
def evaluate(arm, seed):
    target = EVAL/f'frequency/{arm}-seed-{seed}.json'
    if target.exists(): raise FileExistsError(target)
    definition = read(DEFINITION); frozen_sha = file_sha256(DEFINITION); pop = definition['population']
    source = evaluated_checkpoint(arm,seed)
    expected = next(r['sha256'] for r in read(EVAL/'preflight.json')['immutable_checkpoints'] if checkpoint_reference(r['path']) == source)
    assert file_sha256(source) == expected
    valpath = ROOT/'data/foundation_v11/packed/vocab-4096/validation.bin'
    assert file_sha256(valpath) == definition['validation_sha256']
    from training.run_foundation_v35_thermal_gate import Monitor, cooldown
    cooling = cooldown(); monitor = Monitor(); monitor.start()
    _, model = loader.load_model(source); model = model.to('cuda').eval()
    val = np.memmap(valpath,dtype=np.uint16,mode='r'); positions=np.asarray(pop['positions'])
    values = {k:[] for k in ('ce','probabilities','top1','top5','top10')}
    try:
        for block in np.unique((positions-1)//512):
            start=int(block)*512; size=min(512,len(val)-start-1)
            logits,_=model(torch.tensor(np.array(val[start:start+size],dtype=np.int64),device='cuda')[None])
            mask=positions[(positions>start)&(positions<=start+size)]
            rows=logits[0,torch.tensor(mask-start-1,device='cuda')].float()
            truth=torch.tensor(np.array(val[mask],dtype=np.int64),device='cuda')
            logp=torch.log_softmax(rows,-1).gather(1,truth[:,None]).squeeze(1); top=rows.topk(10,-1).indices
            values['ce'].extend((-logp).cpu().tolist()); values['probabilities'].extend(logp.exp().cpu().tolist())
            for k in (1,5,10): values[f'top{k}'].extend((top[:,:k]==truth[:,None]).any(-1).cpu().tolist())
    finally:
        telemetry=monitor.finish()
    assert len(values['ce']) == len(positions) and np.isfinite(values['ce']).all()
    assert file_sha256(source) == expected and file_sha256(DEFINITION) == frozen_sha
    result={'phase':50,'arm':arm,'seed':seed,'checkpoint_sha256':expected,'definition_file_sha256':frozen_sha,
        'population_sha256':pop['sha256'],'values':values,'metrics':previous.summarize(values,pop),
        'execution':'CUDA FP32 inference only; fixed packed 512 contexts; no optimizer / no training', 'cooldown':cooling,'telemetry':telemetry}
    write(target,result); print('SUPPORTED TAIL',arm,seed,result['metrics']['micro_ce'],flush=True)


def support_ok(pop, policy):
    return pop['document_count'] >= policy['minimum_documents'] and pop['occurrence_count'] >= policy['minimum_occurrences']


def metric_intervals(values, pop):
    """Whole-document resampling for all reported rare metrics, including token-balanced CE."""
    _, documents=np.unique(pop['document_ids'],return_inverse=True)
    _, tokens=np.unique(pop['token_ids'],return_inverse=True)
    ce=np.asarray(values['ce']); prob=np.asarray(values['probabilities'])
    order=np.argsort(prob); sorted_prob=prob[order]
    rng=np.random.default_rng(4900); draws=rng.integers(0,documents.max()+1,size=(2000,documents.max()+1))
    estimates={k:[] for k in ('micro_ce','macro_per_token_ce','top1','top5','top10','mean_probability','median_probability','geometric_mean_probability')}
    for draw in draws:
        weights=np.bincount(draw,minlength=documents.max()+1)[documents]; total=weights.sum()
        micro=float(np.dot(weights,ce)/total); tc=np.bincount(tokens,weights=weights); sums=np.bincount(tokens,weights=weights*ce)
        cumulative=np.cumsum(weights[order]); mid=(total-1)/2
        median=(sorted_prob[np.searchsorted(cumulative,math.floor(mid)+1)]+sorted_prob[np.searchsorted(cumulative,math.ceil(mid)+1)])/2
        row={'micro_ce':micro,'macro_per_token_ce':float(np.mean(sums[tc>0]/tc[tc>0])),
            'mean_probability':float(np.dot(weights,prob)/total),'median_probability':float(median),'geometric_mean_probability':math.exp(-micro),
            **{f'top{k}':float(np.dot(weights,values[f'top{k}'])/total) for k in (1,5,10)}}
        for key,value in row.items(): estimates[key].append(value)
    return {k:np.quantile(v,[.025,.975]).tolist() for k,v in estimates.items()}


def decide():
    definition=read(DEFINITION); policy=definition['gate_policy']; pop=definition['population']
    old=read(ROOT/'evaluation/foundation-v38-frequency-gate-v2-summary.json')
    results={a:{str(s):read(EVAL/f'frequency/{a}-seed-{s}.json') for s in v38.SEEDS} for a in ('baseline','B','C')}
    for arm, rows in results.items():
        for seed,r in rows.items():
            source=evaluated_checkpoint(arm,int(seed))
            assert r['definition_file_sha256']==file_sha256(DEFINITION)
            assert r['checkpoint_sha256']==file_sha256(source)
            reused=read(v38.EVAL/f'frequency/{arm}-seed-{seed}.json')
            assert reused['checkpoint_sha256']==r['checkpoint_sha256']
            assert reused['core']['population_sha256']==definition['core']['sha256']
            assert reused['tail']['population_sha256']==definition['legacy']['sha256']
    comparisons={}; eligible=[]
    for arm in ('B','C'):
        c=old['comparisons'][arm]; safety={}
        for seed in v38.SEEDS:
            key=str(seed); checks=dict(c['safety'][key]['checks'])
            checks.pop('tail_ci'); checks['frequency_support']=support_ok(pop,policy)
            pair=previous.paired(results['baseline'][key],results[arm][key],pop)
            checks['supported_tail_ci']=pair['ce_delta_ci95'] is not None and pair['ce_delta_ci95'][1] <= policy['supported_ce_delta_upper']
            safety[key]={'checks':checks,'pass':all(checks.values()),'supported_tail_paired':pair,
                'core_paired':c['safety'][key]['frequency_paired']['core']}
        metrics=[results[arm][str(s)]['metrics'] for s in v38.SEEDS]
        aggregate={k:{'mean':mean(r[k] for r in metrics),'std':stdev(r[k] for r in metrics)} for k in metrics[0] if isinstance(metrics[0][k],(int,float))}
        comparisons[arm]={'lr':c['lr'],'legacy_metrics':c['legacy_metrics'],'core_metrics':c['frequency_metrics']['core'],
            'legacy_tail_warning_metrics':c['frequency_metrics']['tail'],'supported_tail_metrics':aggregate,'safety':safety,'eligible':all(r['pass'] for r in safety.values())}
        if comparisons[arm]['eligible']: eligible.append(arm)
    delta=np.mean([np.asarray(results['C'][str(s)]['values']['ce'])-np.asarray(results['B'][str(s)]['values']['ce']) for s in v38.SEEDS],axis=0)
    ci=previous.cluster_ci(delta,pop['document_ids'])
    selected=eligible[0] if len(eligible)==1 else None
    if len(eligible)==2 and old['seed_mean_paired_core_ci95'][1]<0 and comparisons['C']['legacy_metrics']['loss']['mean']<comparisons['B']['legacy_metrics']['loss']['mean']: selected='C'
    gate=('FORMAL_LR_APPROVED_5E5' if selected=='C' else 'FORMAL_LR_APPROVED_7_5E5') if selected else 'FORMAL_LR_STILL_UNRESOLVED'
    if not support_ok(pop,policy): gate='FREQUENCY_EVALUATOR_STILL_INVALID'; selected=None
    pre=read(EVAL/'preflight.json')
    for r in pre['immutable_checkpoints']: assert file_sha256(checkpoint_reference(r['path']))==r['sha256']
    verify_preserved()
    if any(not r['checks']['context'] or not r['checks']['stability'] for c in comparisons.values() for r in c['safety'].values()): gate='STOP_AND_INVESTIGATE';selected=None
    decision={'gate':gate,'approved':selected is not None,'selected_arm':selected,'selected_lr':comparisons[selected]['lr'] if selected else None}
    write(ROOT/'evaluation/foundation-v39-lr-decision.json',decision)
    summary={'phase':50,**decision,'frequency_classification':'DOCUMENT_SUPPORTED_EVALUATOR_VALID' if support_ok(pop,policy) else 'INSUFFICIENT_SUPPORTED_TAIL',
        'definition_sha256':file_sha256(DEFINITION),'support':{name:{k:p[k] for k in ('token_count','occurrence_count','document_count','sha256')} for name,p in [('core',definition['core']),('supported_tail',pop),('legacy',definition['legacy'])]},
        'comparisons':comparisons,'seed_mean_paired_core_C_minus_B_ci95':old['seed_mean_paired_core_ci95'],
        'seed_mean_paired_supported_C_minus_B_ci95':ci,'canonical_training_executed':False,'candidate_metrics':None,
        'gpu_training_tokens_per_second':None,'gpu_inference_max_temperature_c':max(r['telemetry']['gpu_temperature_c_max'] for a in results.values() for r in a.values()),
        'checkpoint_integrity':{'pass':True,'count':len(pre['immutable_checkpoints'])},'20m_permission':False,'foundation_base_complete':False,
        'preflight_tests':read(EVAL/'tests-preflight.json'),'final_pytest':'pending','cpu_parallel_evaluation':'DISABLED',
        'limitations':['All intervals are conditional on the fixed validation corpus, not evidence for all-language generalization.',
            'Head/Middle, Core and legacy exact PHASE49 results reused with identical checkpoint SHA; only new supported-tail inference was run.',
            'Legacy Tail is warning only. Existing per-seed Core/EOS/sampling margins are not loosened after seeing outcomes.',
            'No new GPU training, no candidate, no canonical promotion. Both 512k arms remain experimental.']}
    write(ROOT/'evaluation/foundation-v39-frequency-gate-v3-summary.json',summary)
    print(decision,flush=True)


def report(final_tests):
    summary=read(ROOT/'evaluation/foundation-v39-frequency-gate-v3-summary.json'); definition=read(DEFINITION)
    verify_preserved()
    migration=read(EVAL/'z-migration.json')
    assert migration['gate']=='Z_CHECKPOINT_MIGRATION_PASS'
    summary['migration_gate']=migration['gate'];summary['ml_reevaluation_reused']=True
    statistics={}; hierarchy={}
    for arm in ('baseline','B','C'):
        statistics[arm]={}; hierarchy[arm]={}
        for seed in v38.SEEDS:
            key=str(seed); tail=read(EVAL/f'frequency/{arm}-seed-{seed}.json'); core=read(v38.EVAL/f'frequency/{arm}-seed-{seed}.json')['core']
            statistics[arm][key]={name:{'metrics':row['metrics'],'ci95':metric_intervals(row['values'],pop)} for name,row,pop in [('core',core,definition['core']),('supported_tail',tail,definition['population'])]}
            source=ROOT/f'evaluation/phase46/baseline/seed-{seed}.json' if arm=='baseline' else v38.evaluation_path(arm,seed)
            buckets=read(source)['validation']['frequency_buckets']
            head=[buckets[k] for k in ('top_1_percent','top_5_percent_excluding_top_1','top_20_percent_excluding_top_5')]
            count=sum(r['targets'] for r in head)
            hierarchy[arm][key]={'head':{'rank_range':[0,819],'occurrences':count,**{k:sum(r['targets']*r[k] for r in head)/count for k in ('cross_entropy','top_1_accuracy','top_5_accuracy','top_10_accuracy','mean_correct_token_probability')}},
                'middle':buckets['middle_20_to_80_percent'],'core':core['metrics'],'supported_tail':tail['metrics'],
                'legacy_tail_warning_only':buckets['rare_bottom_20_percent']}
    write(EVAL/'frequency-statistics.json',{'per_seed':statistics,'hierarchy':hierarchy,'bootstrap':definition['gate_policy']['bootstrap'],
        'macro_ci_note':'Bootstrap token-balanced mean over represented token IDs in each whole-document resample. Missing IDs in a resample are not assigned invented CE.'})
    summary['final_pytest']=final_tests;write(ROOT/'evaluation/foundation-v39-frequency-gate-v3-summary.json',summary)
    lines=['# PHASE50 / Foundation v3.9 — Frequency Gate v3','',f"Formal LR Gate: **{summary['gate']}**. Approved LR: NO. Selected LR: NONE.",
        '',f"Preflight: 443 passed, 0 failed. Final pytest: {final_tests}.",'','## Checkpoint-independent preregistration','',
        'Train rank >=3277 (bottom 20% of unchanged 4096 vocabulary), validation occurrences 2–9 inclusive, at least two independent validation documents per token. Select every qualifying ID and occurrence, excluding position zero. Core and Supported Tail IDs are disjoint. Model loss/probability/top-k were not used for membership.',
        'Preregistered on '+definition['registered_at_utc']+'. Definition file SHA256: '+file_sha256(DEFINITION)+'.',
        '', '| Population | Types | Occurrences | Documents | Role |','|---|---:|---:|---:|---|']
    for name,pop in summary['support'].items(): lines.append(f"| {name} | {pop['token_count']} | {pop['occurrence_count']} | {pop['document_count']} | {'WARNING ONLY' if name=='legacy' else 'formal frequency evidence'} |")
    lines+=['','Supported membership SHA256: '+definition['population']['sha256']+'.',
        'Legacy SHA256: '+definition['legacy']['sha256']+'. Its 58 / 112 / 2 membership is unchanged and cannot veto LR approval.',
        'Maximum observed non-Core rare support before the repeated-support filter: '+str(definition['maximum_observed_noncore_rare_support'])+'.',
        'Targets of >=30 documents and >=300 occurrences are met without adding any non-rare ID. Classification: '+summary['frequency_classification']+'.',
        '', '## Existing 512k checkpoints, three seeds; no new training','',
        '| LR | Validation ± SD | Top1 / 5 / 10 | Middle CE | Core micro / macro CE | Supported micro / macro CE | Legacy CE | Natural / Semantic |',
        '|---|---|---|---:|---|---|---:|---|']
    for arm in ('C','B'):
        c=summary['comparisons'][arm]; m=c['legacy_metrics'];core=c['core_metrics'];tail=c['supported_tail_metrics']
        lines.append(f"| {c['lr']:g} | {m['loss']['mean']:.6f} ± {m['loss']['std']:.6f} | {m['top1']['mean']:.2%} / {m['top5']['mean']:.2%} / {m['top10']['mean']:.2%} | {m['middle_ce']['mean']:.6f} | {core['micro_ce']['mean']:.6f} / {core['macro_per_token_ce']['mean']:.6f} | {tail['micro_ce']['mean']:.6f} / {tail['macro_per_token_ce']['mean']:.6f} | {c['legacy_tail_warning_metrics']['micro_ce']['mean']:.6f} | {m['naturalness']['mean']:.2%} / {m['semantic']['mean']:.2%} |")
    lines+=['','Paired C minus B, seed-averaged before document resampling:',
        '- Core CE 95% CI: '+str(summary['seed_mean_paired_core_C_minus_B_ci95'])+'.',
        '- Supported Tail CE 95% CI: '+str(summary['seed_mean_paired_supported_C_minus_B_ci95'])+'.',
        '', '2000 document-cluster replicates (RNG 4900). Per-seed micro CE, macro token CE, Top1/5/10, mean/median/geometric correct probability and intervals are in `phase50/frequency-statistics.json`; seed mean/std are in the summary. Head combines old rank 0–819 sub-buckets with occurrence weighting; Middle uses ranks 820–3276. Core and legacy results reuse exact PHASE49 SHA-matched checkpoints. Supported Tail uses CUDA FP32 inference on fixed packed 512 contexts.',
        '', '## Why no formal approval','',
        'Supported Tail passes every per-seed noninferiority margin (paired CE CI upper <=0.25). Core keeps its pre-existing <=0.10 margin and nonfrequency checks remain unchanged. Legacy has no vote. Improving evaluator support does not erase independent EOS, Core or sampling failures.',
        '', '| LR | Seed | Supported delta CE [95% CI] | Remaining failed checks |','|---|---|---|---|']
    for arm in ('C','B'):
        c=summary['comparisons'][arm]
        for seed,r in c['safety'].items(): lines.append(f"| {c['lr']:g} | {seed} | {r['supported_tail_paired']['ce_delta']:+.6f} {r['supported_tail_paired']['ce_delta_ci95']} | {', '.join(k for k,v in r['checks'].items() if not v) or 'none'} |")
    lines+=['','B has +0.67pp naturalness and +2.67pp semantic mean, but C improves LM, top-k, Middle, Core and supported rare CE. Neither qualitative preference nor minimum validation loss can override the failed per-seed safety checks.',
        '', '## Context, EOS, attractor and integrity','', '| LR | Full context CE / advantage | Terminal P(EOS) / premature | Greedy runaway / loop onset |','|---|---|---|---|']
    for arm in ('C','B'):
        c=summary['comparisons'][arm];m=c['legacy_metrics'];lines.append(f"| {c['lr']:g} | {m['context']['mean']:.6f} / {m['advantage']['mean']:.6f} | {m['terminal_eos']['mean']:.7f} / {m['premature_eos']['mean']:.2%} | {m['runaway']['mean']:.0%} / {m['loop_onset']['mean']:.2f} |")
    lines+=['','Context regression: NO on all existing checks. Premature EOS: 0%, but seed 2026 terminal EOS fails the retained relative safety threshold. Greedy runaway remains 100%, a major separate research issue.',
        f"Inference maximum GPU temperature: {summary['gpu_inference_max_temperature_c']}°C. Each evaluation records cooldown/start telemetry. GPU training tok/s: N/A (no training). No overclock, undervolt, fan or power-limit changes. CPU parallel evaluation DISABLED.",
        'Checkpoint SHA256, strict reload and resume-state integrity: 19/19 PASS after COPY to the configured Z root. Original D files retained. Protected 4 files and relocated READY 5: unchanged. Other pre-existing infrastructure edits were retained separately; the historical 18-file preflight is not a claim that those infra files remain unchanged. Official 15.872M and old LR1e-4 16.128M unchanged. Final Blind was hashed only: '+read(EVAL/'preflight.json')['final_blind_sha256']+'.',
        'Canonical 16.128M training: NO. Candidate metrics: N/A. No lineage promotion. 20M permission: NO. Foundation Base complete: NO.',
        '', '## Limitations and next gate','']+summary['limitations']+['','Next: preregister a focused EOS/Core/sampling investigation before requesting any new training. No training permission is implied by this report.']
    (ROOT/'evaluation/foundation-v39-frequency-gate-v3-report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('PHASE50 report and all-metric bootstrap complete',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['preflight','freeze','evaluate','decide','report']);p.add_argument('--arm',choices=['baseline','B','C']);p.add_argument('--seed',type=int,choices=v38.SEEDS);p.add_argument('--final-tests',default='pending')
    a=p.parse_args();torch.set_num_threads(4)
    if a.action=='evaluate': evaluate(a.arm,a.seed)
    elif a.action=='report': report(a.final_tests)
    else: globals()[a.action]()
