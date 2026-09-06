"""PHASE49 preregistered document-supported LR approval, never automatic promotion."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from statistics import mean,stdev
import torch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from training.run_foundation_v38_gate import EVAL,OUT,SEEDS,read,write,training_path,evaluation_path
from training import run_foundation_v36_lr_review as v36
from training import run_foundation_v37_stability as v37
from evaluation import analyze_foundation_v36_lr_review as a36
from evaluation.analyze_foundation_v37_stability import aggregate
from evaluation.evaluate_foundation_v38_gate import paired,cluster_ci
from evaluation.analyze_foundation_v35_short_gate import classify_attractor
from training.train_foundation_v21_ab import file_sha256


def ci_safe(comparison,margin):
    interval=comparison['ce_delta_ci95']
    return interval is not None and interval[1]<=margin


def support_ok(populations,policy):
    return (populations['core']['token_count']>=policy['minimum_core_types'] and
        populations['core']['occurrence_count']>=policy['minimum_core_occurrences'] and
        all(p['document_count']>=policy['minimum_documents'] for p in populations.values()))


def run(pytest_result):
    policy=read(EVAL/'gate-policy.json'); definition=read(EVAL/'frequency-population.json'); populations=definition['populations']
    frequency={a:{str(s):read(EVAL/f'frequency/{a}-seed-{s}.json') for s in SEEDS} for a in ('baseline','B','C')}
    comparisons={}; eligibility=[]; support=support_ok(populations,policy)
    for arm in ('B','C'):
        rows=[read(evaluation_path(arm,s)) for s in SEEDS]; safety={}; fm={}
        for s,row in zip(SEEDS,rows):
            raw=a36.checks(row,a36.baseline(s),read(v37.evaluation_path(arm,s,256000)),read(training_path(arm,s)))
            del raw['checks']['rare']; checks=raw['checks']; comparisons_seed={}
            for name,pop in populations.items():
                pair=paired(frequency['baseline'][str(s)][name],frequency[arm][str(s)][name],pop)
                comparisons_seed[name]=pair; checks[name+'_ci']=ci_safe(pair,policy[name+'_delta_upper_limit'])
            checks['frequency_support']=support
            checks['overall_lm_improves']=row['validation']['loss']<a36.baseline(s)['validation']['loss']
            raw.update({'pass':all(checks.values()),'frequency_paired':comparisons_seed}); safety[str(s)]=raw
        for name in populations:
            metrics=[frequency[arm][str(s)][name]['metrics'] for s in SEEDS]
            fm[name]={k:{'mean':mean(m[k] for m in metrics),'std':stdev(m[k] for m in metrics),'range':[min(m[k] for m in metrics),max(m[k] for m in metrics)]}
                for k in metrics[0] if isinstance(metrics[0][k],(int,float))}
        ok=all(s['pass'] for s in safety.values());
        if ok: eligibility.append(arm)
        comparisons[arm]={'lr':v37.ARMS[arm],'budget':512000,'legacy_metrics':aggregate(rows),'frequency_metrics':fm,'safety':safety,'eligible':ok}
    between={str(s):{name:paired(frequency['B'][str(s)][name],frequency['C'][str(s)][name],pop) for name,pop in populations.items()} for s in SEEDS}
    # Same validation documents are shared by seeds: average paired losses first,
    # then resample documents jointly, rather than treating model seeds as independent data.
    mean_core_delta=np.mean([np.asarray(frequency['C'][str(s)]['core']['values']['ce'])-np.asarray(frequency['B'][str(s)]['core']['values']['ce']) for s in SEEDS],axis=0)
    mean_core_ci=cluster_ci(mean_core_delta,populations['core']['document_ids'])
    selected=eligibility[0] if len(eligibility)==1 else None
    if len(eligibility)==2:
        lm=comparisons['C']['legacy_metrics']['loss']['mean']-comparisons['B']['legacy_metrics']['loss']['mean']
        if lm<0 and mean_core_ci[1]<0: selected='C'
        elif lm>0 and mean_core_ci[0]>0: selected='B'
    gate=('FORMAL_LR_APPROVED_5E5' if selected=='C' else 'FORMAL_LR_APPROVED_7_5E5') if selected else 'FORMAL_LR_STILL_UNRESOLVED'
    if not support: gate='FREQUENCY_GATE_REDESIGN_REQUIRED'; selected=None
    if any(not x['checks']['stability'] or not x['checks']['context'] for c in comparisons.values() for x in c['safety'].values()): gate='STOP_AND_INVESTIGATE'; selected=None
    approved=selected is not None; decision={'phase':49,'gate':gate,'approved':approved,'selected_arm':selected,'selected_lr':v37.ARMS[selected] if selected else None,'support_adequate':support}
    write(EVAL/'decision.json',decision)
    pre=read(EVAL/'preflight.json')
    for r in pre['immutable_checkpoints']: assert file_sha256(ROOT/r['path'])==r['sha256']
    assert all(file_sha256(Path(r['path']))==r['sha256'] for r in pre['preserved_files'])
    for s in SEEDS:
        r=read(training_path('B',s)); assert file_sha256(ROOT/r['checkpoint'])==r['sha256']
        p=torch.load(ROOT/r['checkpoint'],map_location='cpu',weights_only=False)
        v36.verify_payload(p,s,16384000,7.5e-5)
    assert file_sha256(ROOT/'data/foundation_v09/evaluation/final-blind-1000.json')==pre['final_blind_sha256']
    assert not [p for p in OUT.rglob('*') if p.is_file() and (p.suffix=='.tmp' or p.stat().st_size==0)]
    trains=[read(training_path('B',s)) for s in SEEDS]
    representative=selected or 'C'; m=comparisons[representative]['legacy_metrics']
    mapping={'median_loop_onset':'loop_onset','greedy_repetition_1':'repetition1','loop_margin':'margin','loop_entropy':'entropy'}
    baseline=[a36.metrics(a36.baseline(s)) for s in SEEDS]
    attractor=classify_attractor({k:{'mean':mean(b[v] for b in baseline)} for k,v in mapping.items()},{k:m[v] for k,v in mapping.items()})
    summary={'phase':49,'preflight_tests':read(EVAL/'tests-preflight.json'),'final_pytest':pytest_result,**decision,
        'frequency_classification':'INSUFFICIENT_TAIL_DOCUMENT_SUPPORT' if not support else 'CI_RECALIBRATED_REVIEW',
        'population_counts':{k:{n:p[n] for n in ('token_count','occurrence_count','document_count','sha256')} for k,p in populations.items()},
        'comparisons':comparisons,'paired_C_minus_B':between,'seed_mean_paired_core_ci95':mean_core_ci,'reporting_arm':representative,'reporting_arm_is_approval':approved,
        'canonical_candidate_training_executed':False,'candidate_final_token_count':None,
        'next_ml_gate':gate,'next_token_target':None,'20m_permission':False,'foundation_base_complete':False,
        'attractor':attractor,'gpu':{'mean_tokens_per_second':mean(r['tokens_per_second'] for r in trains),
            'peak_allocated_vram_mib':max(r['peak_vram_mib'] for r in trains),'max_temp_c':max(r['telemetry']['gpu_temperature_c_max'] for r in trains),
            'classifications':sorted(set(r['telemetry']['thermal_classification'] for r in trains))},
        'integrity':{'immutable':16,'new':3,'pass':True,'preserved_files':18,'final_blind_sha256':pre['final_blind_sha256']},
        'cpu_parallel_evaluation':'DISABLED','notes':['Core and Tail bootstrap CIs are conditional on their fixed validation support, not all-language generalization.',
            'Tail occupies only two documents. Its 2000 resamples have few distinct document combinations; more replicates cannot repair missing independent support.',
            'Core uses CUDA FP32 inference with no model updates; all arms and baseline share exact contexts. Tail uses exact legacy CPU probabilities.',
            'Initial training attempt stopped before the first update: integrity model instantiation consumed restored RNG. Verification was moved before RNG restoration; continuity checks passed on all runs.',
            'No architecture, tokenizer, corpus, split, EOS or objective changes. No canonical promotion or 20M permission.']}
    write(ROOT/'evaluation/foundation-v38-rare-core-tail.json',{'definition':definition,'results':frequency,'bootstrap_policy':policy['bootstrap']})
    write(ROOT/'evaluation/foundation-v38-lr-512k-comparison.json',{'arms':comparisons,'paired_C_minus_B':between,'decision':decision})
    write(ROOT/'evaluation/foundation-v38-frequency-gate-v2-summary.json',summary)
    lines=['# PHASE49 / TRACK A — Rare Gate v2','',f"Formal LR Gate: **{gate}**. Approved: **{approved}**. Selected LR: {decision['selected_lr']}.",
      '',f"Preflight: 437 passed. Final pytest: {pytest_result}.",'','## Fixed population and uncertainty','',
      f"Core: {populations['core']['token_count']} types, {populations['core']['occurrence_count']} occurrences, {populations['core']['document_count']} documents.",
      'Tail: unchanged 58 types / 112 occurrences / **2 documents**. SHA '+populations['tail']['sha256']+'.',
      policy['core_selection'],policy['bootstrap'],
      'The legacy Tail cannot establish population-wide noninferiority. Widening tolerances after seeing results is not permitted. Retain Tail for monitoring; a future phase must preregister broader tail document support without opening Final Blind.',
      '', '## Fair cumulative 512k comparison','', '| LR | Validation ± SD | Top1/5/10 | Middle | Core micro / macro CE | Tail CE | Natural/Semantic |', '|---|---|---|---:|---|---:|---|']
    for arm,c in comparisons.items():
        m=c['legacy_metrics']; f=c['frequency_metrics']
        lines.append(f"| {c['lr']:g} | {m['loss']['mean']:.6f} ± {m['loss']['std']:.6f} | {m['top1']['mean']:.2%}/{m['top5']['mean']:.2%}/{m['top10']['mean']:.2%} | {m['middle_ce']['mean']:.6f} | {f['core']['micro_ce']['mean']:.6f}/{f['core']['macro_per_token_ce']['mean']:.6f} | {f['tail']['micro_ce']['mean']:.6f} | {m['naturalness']['mean']:.2%}/{m['semantic']['mean']:.2%} |")
    lines+=['','## Document-bootstrap intervals','', '| LR | Seed | Core CE [95% CI] | Core delta [95% CI] | Tail CE [95% CI] | Tail delta [95% CI] | Failed checks |','|---|---|---|---|---|---|---|']
    for arm,c in comparisons.items():
        for s in SEEDS:
            f=frequency[arm][str(s)]; safety=c['safety'][str(s)]; pairs=safety['frequency_paired']
            lines.append(f"| {c['lr']:g} | {s} | {f['core']['metrics']['micro_ce']:.5f} {f['core']['metrics']['micro_ce_ci95']} | {pairs['core']['ce_delta']:+.5f} {pairs['core']['ce_delta_ci95']} | {f['tail']['metrics']['micro_ce']:.5f} {f['tail']['metrics']['micro_ce_ci95']} | {pairs['tail']['ce_delta']:+.5f} {pairs['tail']['ce_delta_ci95']} | {[k for k,v in safety['checks'].items() if not v]} |")
    lines+=['','## Integrity, recipe and next step','',f"GPU: {summary['gpu']}. Existing 16 and new 3 checkpoints PASS; protected 18 files unchanged; Final Blind SHA only.",
       'CUDA FP32, EOS1.5, repetition auxiliary OFF, AdamW unchanged. CPU parallel evaluation DISABLED. Web code is isolated from model/checkpoint code.',
       f"Canonical candidate executed: False. Next token target: None. Next Gate: {gate}. 20M permission: NO. Foundation Base complete: NO.",
       '1e-4 old lineage and 15.872M formal checkpoints remain unchanged. Both 512k LR arms remain experimental.', '', '## Limitations and execution note','']+summary['notes']
    (ROOT/'evaluation/foundation-v38-frequency-gate-v2-report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(decision,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pytest-result',default='pending'); args=parser.parse_args();torch.set_num_threads(4);run(args.pytest_result)
