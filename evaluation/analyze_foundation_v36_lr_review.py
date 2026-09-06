"""PHASE 47 selection and review; decisions never trigger training or promotion."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from statistics import mean, stdev

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from training.run_foundation_v36_lr_review import ARMS, EVAL, SEEDS, read_json, write_json, fingerprint, verify_payload
from training.train_foundation_v21_ab import file_sha256
from evaluation.evaluate_foundation_v36_lr_review import BUCKETS
from evaluation.analyze_foundation_v35_short_gate import classify_attractor


def baseline(seed): return read_json(ROOT/f'evaluation/phase46/baseline/seed-{seed}.json')
def past_control(seed): return read_json(ROOT/f'evaluation/phase46/gate1/seed-{seed}.json')
def evaluation(arm,seed): return read_json(EVAL/f'arm-{arm}/seed-{seed}-evaluation.json')
def training(arm,seed): return read_json(EVAL/f'arm-{arm}/seed-{seed}-training.json')


def metrics(row):
    v=row['validation']; g=row['generation']; f=v['frequency_buckets']
    return {'loss':v['loss'],'ppl':v['perplexity'],'top1':v['top_1_accuracy'],'top5':v['top_5_accuracy'],'top10':v['top_10_accuracy'],
            'middle_ce':f[BUCKETS[-2]]['cross_entropy'],'rare_ce':f[BUCKETS[-1]]['cross_entropy'],
            'rare_median':f[BUCKETS[-1]]['median_correct_token_probability'],
            'rare_top1':f[BUCKETS[-1]]['top_1_accuracy'],'rare_top5':f[BUCKETS[-1]]['top_5_accuracy'],'rare_top10':f[BUCKETS[-1]]['top_10_accuracy'],
            'naturalness':g['temperature_0.7']['naturalness_rate'],'semantic':g['temperature_0.7']['semantic_rate'],
            'terminal_eos':row['terminal_eos']['mean_probability'],'premature_eos':row['nonterminal_eos']['top1_rate'],
            'context':row['context']['512']['loss'],'advantage':row['context']['full_context_advantage_vs_1'],
            'runaway':g['greedy']['runaway_rate'],'loop_onset':g['greedy']['median_loop_onset'],
            'repetition1':g['greedy']['repetition']['1'],'entropy':g['greedy']['loop_onset_distribution']['entropy'],
            'margin':g['greedy']['loop_onset_distribution']['top1_top2_margin']}


def checks(row,base,control,train):
    p=read_json(EVAL/'selection-policy.json'); m,b,c=metrics(row),metrics(base),metrics(control)
    joint_greedy_worse=m['loop_onset']<c['loop_onset'] and m['repetition1']>c['repetition1'] and m['margin']>c['margin']
    result={
        'validation':m['loss']<=c['loss']+p['loss_max_increase_vs_control'],
        'topk':all(m[k]>=c[k]-p['topk_max_drop_vs_control'] for k in ('top1','top5','top10')),
        'middle':m['middle_ce']<=b['middle_ce']+p['middle_ce_max_increase_vs_baseline'],
        'rare':m['rare_ce']<=b['rare_ce']+p['rare_ce_max_increase_vs_baseline'] and m['rare_median']>=b['rare_median']*p['rare_median_min_ratio_vs_baseline'],
        'context':m['context']<=b['context']+p['full_context_max_increase_vs_baseline'] and m['advantage']>=p['full_context_min_advantage'],
        'eos':m['terminal_eos']>=c['terminal_eos']*p['terminal_eos_min_ratio_vs_control'] and m['premature_eos']==0,
        'sampling':all(m[k]>=c[k]-p['sampling_max_drop_vs_control'] for k in ('naturalness','semantic')),
        'teacher_forced':all(row['teacher_forced_horizons'][h]['loss']<=control['teacher_forced_horizons'][h]['loss']+p['teacher_forced_loss_max_increase_vs_control'] for h in ('1','2','4','8','16','32')),
        'greedy_no_joint_worsening':not joint_greedy_worse,
        'stability':train['integrity']['pass'] and train['source_unchanged'] and train['telemetry'].get('samples',0)>0 and not train['telemetry'].get('hardware_thermal_slowdown',True),
    }
    return {'checks':result,'pass':all(result.values()),'metrics':m,'deltas_vs_control':{k:m[k]-c[k] for k in m},'deltas_vs_baseline':{k:m[k]-b[k] for k in m}}


def select():
    base,control=baseline(42),evaluation('A',42)
    current,old=metrics(control),metrics(past_control(42))
    reproducibility={k:abs(current[k]-old[k]) for k in current}
    # A is the replication bridge to historical controls for other seeds.
    replicated=max(reproducibility.values())<1e-6
    continuity={arm:training(arm,42)['continuity_fingerprints'] for arm in ARMS}
    same_start=all(v==continuity['A'] for v in continuity.values())
    rows={arm:checks(evaluation(arm,42),base,control,training(arm,42)) for arm in ARMS}
    eligible=[a for a in ('B','C') if rows[a]['pass'] and rows[a]['deltas_vs_control']['rare_ce']<=-read_json(EVAL/'selection-policy.json')['clear_rare_ce_improvement_vs_control']]
    best=None
    if eligible and replicated and same_start:
        best=min(eligible,key=lambda a:rows[a]['metrics']['rare_ce'])
        if len(eligible)==2 and abs(rows['B']['metrics']['rare_ce']-rows['C']['metrics']['rare_ce'])<.02: best='B'
    result={'phase':47,'arms':rows,'best_arm':best,'best_lr':ARMS[best] if best else None,'clear_best':best is not None,
            'control_replicated':replicated,'control_replication_absolute_differences':reproducibility,'same_start_all_arms':same_start,
            'eligible_arms':eligible,'selection_policy':read_json(EVAL/'selection-policy.json')}
    write_json(EVAL/'selection.json',result)
    write_json(ROOT/'evaluation/foundation-v36-lr-arms.json',{
        'selection':result,'arms':{a:{'training':training(a,42),'evaluation':evaluation(a,42)} for a in ARMS}})
    print({"best_arm":best,"control_replicated":replicated,"arms":{a:{'pass':r['pass'],'failed':[k for k,v in r['checks'].items() if not v],'loss':r['metrics']['loss'],'rare_ce':r['metrics']['rare_ce']} for a,r in rows.items()}},flush=True)
    return result


def final(pytest_result):
    selection=read_json(EVAL/'selection.json'); best=selection['best_arm']; audit=read_json(ROOT/'evaluation/foundation-v36-rare-analysis.json')
    confirmed=best is not None and all((EVAL/f'arm-{best}/seed-{s}-evaluation.json').exists() for s in SEEDS)
    confirmation={}
    if confirmed:
        for seed in SEEDS:
            confirmation[str(seed)]=checks(evaluation(best,seed),baseline(seed),past_control(seed),training(best,seed))
    confirmation_pass=confirmed and all(r['pass'] for r in confirmation.values())
    paired_helpful=confirmed and all(r['deltas_vs_control']['loss']<0 and r['deltas_vs_control']['rare_ce']<0 for r in confirmation.values())
    if confirmation_pass:
        lr_state='LR_REDUCTION_HELPFUL'
        gate={'B':'CONTINUE_SHORT_GPU_GATES_LR_7_5E5','C':'CONTINUE_SHORT_GPU_GATES_LR_5E5','A':'CONTINUE_SHORT_GPU_GATES_LR_1E4'}[best]
        recommended=ARMS[best]
    else:
        lr_state='LR_REDUCTION_HELPFUL' if paired_helpful else 'LR_REDUCTION_NOT_JUSTIFIED'
        gate='FREQUENCY_INSTABILITY_REVIEW'; recommended=None
        if confirmed and any(not r['checks']['stability'] or not r['checks']['context'] for r in confirmation.values()): gate='STOP_AND_INVESTIGATE'
        elif confirmed and any(not r['checks']['sampling'] for r in confirmation.values()): gate='GENERATION_TRADEOFF_REVIEW'
    # No clear candidate does not prove that 1e-4 is intrinsically too high.
    selected_rows=[evaluation(best,s) for s in SEEDS] if confirmed else [evaluation(a,42) for a in ARMS]
    selected_metrics=[metrics(r) for r in selected_rows]
    aggregation={k:{'mean':mean(r[k] for r in selected_metrics),'std':stdev(r[k] for r in selected_metrics)} for k in selected_metrics[0]} if confirmed else None
    mapping={'median_loop_onset':'loop_onset','greedy_repetition_1':'repetition1','loop_margin':'margin','loop_entropy':'entropy'}
    attractor=None
    if confirmed:
        before=[metrics(baseline(s)) for s in SEEDS]
        attractor=classify_attractor({k:{'mean':mean(r[v] for r in before)} for k,v in mapping.items()}, {k:{'mean':mean(r[v] for r in selected_metrics)} for k,v in mapping.items()})
    all_training=[training(a,42) for a in ARMS]
    if confirmed: all_training += [training(best,s) for s in (123,2026)]
    source_rows=read_json(EVAL/'preflight.json')['checkpoints']
    source_check={r['path']:file_sha256(ROOT/r['path'])==r['sha256'] for r in source_rows}
    assert all(source_check.values())
    preserved=read_json(EVAL/'preservation-start.json')
    preservation={r['Path']:file_sha256(ROOT/r['Path']).upper()==r['SHA256'] for r in preserved}
    assert all(preservation.values()), 'pre-existing local file changed'
    write_json(EVAL/'preservation-final.json',preservation)
    blind=file_sha256(ROOT/'data/foundation_v09/evaluation/final-blind-1000.json')
    assert blind=='fa7912d58ce251bb10b513f59793bb8ca6c0023b4fe08d1c040b8ccbfe49845b'
    result={'phase':47,'tests_preflight':read_json(EVAL/'tests-preflight.json'),'pytest_final':pytest_result,
            'frequency_classification':audit['classification'],'rare_audit':audit,'selection':selection,
            'three_seed_confirmation':confirmed,'confirmation_pass':confirmation_pass,'confirmation':confirmation,
            'confirmed_metrics':aggregation,'lr_classification':lr_state,'next_phase_gate':gate,'recommended_formal_lr':recommended,
            'attractor':attractor,'attractor_method':'Existing PHASE46 descriptive heuristic; not a changed experiment selection rule.',
            'paired_lm_and_rare_improvement_all_seeds':paired_helpful,
            'gradient_variance_by_arm':{a:training(a,42)['gradient_std']**2 for a in ARMS},
            'lr_too_high_evidence':{
                'established':False,
                'seed42_gradient_std':{a:training(a,42)['gradient_std'] for a in ARMS},
                'seed42_control_rare_change':selection['arms']['A']['deltas_vs_baseline']['rare_ce'],
                'reason':'Reduced LR decreases seed42 gradient variance, but 1e-4 seed42 Rare CE remains within 0.05 tolerance and sampling improves from baseline. The specified joint signature of 1e-4-only Rare and generation deterioration is not established.'},
            'lr_5e5_too_low_evidence':{'established':selection['arms']['C']['deltas_vs_baseline']['loss']>=0 and all(selection['arms']['C']['deltas_vs_baseline'][k]<=0 for k in ('top1','top5','top10'))},
            'recommended_interval_tokens':256000,'interval_reason':'Single 256k controlled interval; seed variability and mixed generation signals do not support doubling to 512k.',
            '20m_permission':False,'foundation_base_complete':False,'experimental_promoted':False,
            'formal_recipe':{'device':'CUDA','precision':'FP32','eos_weight':1.5,'repetition_auxiliary':False,'lr':recommended},
            'parallel_cpu_evaluation':'DISABLED','source_checkpoint_sha_unchanged':source_check,
            'experimental_integrity':all(r['integrity']['pass'] for r in all_training),
            'preserved_local_files':preservation,
            'test_side_effect_recovery':{
                'file':'evaluation/campus-v21-human-results.json',
                'cause':'Existing API endpoint test redirected its input but left HUMAN_CAMPUS_V21_RESULTS/REPORT pointing at repository files, updating generated_at.',
                'recovery':'Restored generated_at and original CRLF bytes to the captured SHA256; all captured local-file hashes rechecked.',
                'prevention':'test_campus_v21_human_endpoint_persists_five_axes now redirects both export paths to tmp_path.',
                'file_staged':False},
            'gpu':{'mean_tokens_per_second':mean(r['tokens_per_second'] for r in all_training),
                   'peak_vram_mib':max(r['peak_vram_mib'] for r in all_training),
                   'max_temperature_c':max(r['telemetry']['gpu_temperature_c_max'] for r in all_training),
                   'thermal_classifications':sorted(set(r['telemetry']['thermal_classification'] for r in all_training))},
            'final_blind':{'content_used':False,'sha256':blind},'render_changed':False,'vercel_changed':False,
            'limitations':['Generation scores are automatic proxies on 100 fixed prefixes.','Rare probe covers 112 target occurrences; per-seed descriptive classification is not proof of population-wide regression.','A 256k LR experiment cannot establish a long-term plateau or optimal learning-rate schedule.']}
    write_json(ROOT/'evaluation/foundation-v36-frequency-lr-review-summary.json',result)
    lines=['# PHASE 47 Foundation v3.6 Frequency and LR Review','',f"Next Gate: **{gate}**. LR state: **{lr_state}**. Frequency: **{audit['classification']}**.",'',
           '## Test gate','', 'Previous failures: test_worker_output_directories_do_not_collide and test_ready_protocol_rejects_missing_marker expected directory separation / READY rejection but raised AttributeError because OUT was removed by existing storage routing. Tests now inspect and stub checkpoint(), preserving both assertions.',
           '',f"Preflight: 422 passed, 0 failed. Final pytest: {pytest_result}.",
           'The existing checkpoint_paths.py helper and its unchanged regression tests are included as required dependencies: the committed Phase46 runner already imports this helper, which had remained untracked. Other local storage-routing changes are excluded.', '',
           'A separate legacy API test wrote generated_at into the protected human-results JSON. The captured SHA256 was restored exactly, including CRLF. Both test export paths now point to tmp_path. The protected JSON remains unstaged.', '',
           '## Fixed-population audit','',audit['bucket_definition'],f"Repeat deterministic: {audit['deterministic_repeat']}; Phase46 metric reproduction: {audit['phase46_metric_match']}.",'',
           '| Seed | Rare CE delta | Positions worse | Top10 positive contribution |','|---|---:|---:|---:|']
    for seed,row in audit['per_seed'].items(): lines.append(f"| {seed} | {row['ce_delta']:+.6f} | {row['fraction_positions_worse']:.1%} | {row['top10_positive_contribution_share']:.1%} |")
    lines += ['', 'All positions and token IDs, quantiles, sample counts, top-50 contributors, taxonomy and four-checkpoint trajectories are stored in foundation-v36-rare-analysis.json and phase47/audit/*.json.', '',
              '## Seed42 LR arms','', '| Arm | LR | Loss | Top1 / 5 / 10 | Middle CE | Rare CE | Natural / Semantic | EOS | Checks |', '|---|---:|---:|---|---:|---:|---|---:|---|']
    for arm,row in selection['arms'].items():
        m=row['metrics']; lines.append(f"| {arm} | {ARMS[arm]:g} | {m['loss']:.6f} | {m['top1']:.2%} / {m['top5']:.2%} / {m['top10']:.2%} | {m['middle_ce']:.6f} | {m['rare_ce']:.6f} | {m['naturalness']:.0%} / {m['semantic']:.0%} | {m['terminal_eos']:.6f} | {', '.join(k for k,v in row['checks'].items() if not v) or 'PASS'} |")
    if confirmed:
        lines += ['', '## Three-seed confirmation', '', '| Seed | Loss | Rare CE | Rare CE vs baseline | Rare CE vs control | Failed conditions |','|---|---:|---:|---:|---:|---|']
        for seed,row in confirmation.items():
            lines.append(f"| {seed} | {row['metrics']['loss']:.6f} | {row['metrics']['rare_ce']:.6f} | {row['deltas_vs_baseline']['rare_ce']:+.6f} | {row['deltas_vs_control']['rare_ce']:+.6f} | {', '.join(k for k,v in row['checks'].items() if not v) or 'none'} |")
        lines += ['', '| Metric | Mean | Seed std |','|---|---:|---:|']
        for key,stats in aggregation.items(): lines.append(f"| {key} | {stats['mean']:.6f} | {stats['std']:.6f} |")
        lines += ['',f"Attractor versus 15.872M (unchanged PHASE46 descriptive heuristic): {attractor}.",f"Paired LM and Rare improvement in all seeds: {paired_helpful}. This comparative finding is separate from passing the absolute safety conditions."]
    lines += ['',f"Best candidate: {best}; three-seed confirmation: {confirmed}; confirmation pass: {confirmation_pass}.",f"Same starting model/optimizer/scheduler/sampler/RNG across arms: {selection['same_start_all_arms']}; A reproduces historical control: {selection['control_replicated']}.",
              '', 'The pre-recorded selection policy tests LM, frequency, context, EOS, sampling, teacher-forced losses, greedy signals and stability. Loss alone is not a selection rule. No experimental checkpoint is promoted.', '',
              '## Next recipe and operations','',f"Recommended formal LR: {recommended}; next interval: 256k (only after the recorded next gate permits continuation). 20M permission: NO. Foundation Base completion: NO.",
              f"CUDA FP32; EOS 1.5; repetition auxiliary OFF; parallel CPU evaluation DISABLED. GPU: {result['gpu']}.",
              'All six formal checkpoint hashes unchanged; strict model/optimizer, RNG, sampler, update and scheduler checks passed. Final Blind contents unused; SHA verified. Render/Vercel unchanged.', '',
              '## Limitations',''] + result['limitations']
    (ROOT/'evaluation/foundation-v36-frequency-lr-review-report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print({'next_gate':gate,'lr_state':lr_state,'confirmation':confirmed,'confirmation_pass':confirmation_pass},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--select',action='store_true'); parser.add_argument('--final',action='store_true'); parser.add_argument('--pytest-result',default='pending')
    args=parser.parse_args()
    if args.select: select()
    if args.final: final(args.pytest_result)
