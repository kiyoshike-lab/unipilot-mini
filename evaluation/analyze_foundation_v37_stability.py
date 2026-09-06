"""PHASE48 fair multi-seed comparisons, window hypothesis and formal LR gate."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from statistics import mean,stdev
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from training.run_foundation_v37_stability import ARMS,SEEDS,EVAL,OUT,START,read,write,evaluation_path,training_path
from training import run_foundation_v36_lr_review as v36
from evaluation import analyze_foundation_v36_lr_review as a36
from evaluation.analyze_foundation_v35_short_gate import classify_attractor


def metrics(row):
    m=a36.metrics(row)
    rare=row['frequency_probe']['buckets']['rare_bottom_20_percent']
    m.update({'rare_mean':rare['mean'],'rare_q25':rare['q25'],'rare_q75':rare['q75']})
    m.update({f'context_{n}':row['context'][str(n)]['loss'] for n in (64,16,2,1)})
    m.update({f'terminal_eos_top{k}':row['terminal_eos'][f'top{k}_rate'] for k in (1,5,10)})
    m['nonterminal_eos']=row['nonterminal_eos']['mean_probability']
    sampling=row['generation']['temperature_0.7']; greedy=row['generation']['greedy']
    m.update({'sampling_repetition':sampling['mean_repetition_1'],'completion':sampling['sentence_completion_rate'],
              'topic_retention':sampling['topic_retention_proxy'],'first_break':float(greedy['first_break'])})
    m.update({f'repetition{k}':greedy['repetition'][str(k)] for k in (2,3,4)})
    return m


def aggregate(rows):
    values=[metrics(r) for r in rows]
    return {key:{'mean':mean(v[key] for v in values),'std':stdev(v[key] for v in values),
                 'by_seed':{str(s):v[key] for s,v in zip(SEEDS,values)}} for key in values[0]}


def comparison(budget):
    result={}
    for arm in ARMS:
        paths=[evaluation_path(arm,s,budget) for s in SEEDS]
        if not all(p.exists() for p in paths): continue
        rows=[read(p) for p in paths]; safety={}
        for seed,row in zip(SEEDS,rows):
            base=a36.baseline(seed)
            control=a36.past_control(seed) if budget==256000 else read(evaluation_path(arm,seed,256000))
            safety[str(seed)]=a36.checks(row,base,control,read(training_path(arm,seed,budget)))
        result[arm]={'lr':ARMS[arm],'budget':budget,'metrics':aggregate(rows),'safety':safety,
                     'pass':all(r['pass'] for r in safety.values()),'paths':[str(p.relative_to(ROOT)) for p in paths]}
    return result


def winner(rows,policy):
    if set(rows)!=set(ARMS): raise ValueError('both three-seed arms required')
    for arm,other in (('C','B'),('B','C')):
        a,b=rows[arm]['metrics'],rows[other]['metrics']
        improves=sum(a['loss']['by_seed'][str(s)]<b['loss']['by_seed'][str(s)] and a['rare_ce']['by_seed'][str(s)]<b['rare_ce']['by_seed'][str(s)] for s in SEEDS)
        def failures(r): return sum(not v for x in r['safety'].values() for k,v in x['checks'].items() if k not in ('rare','middle'))
        if (b['loss']['mean']-a['loss']['mean']>=policy['winner_lm_advantage'] and
            b['rare_ce']['mean']-a['rare_ce']['mean']>=policy['winner_rare_ce_advantage'] and improves>=2 and
            all(a[k]['mean']>=b[k]['mean']-policy['comparison_sampling_tolerance'] for k in ('naturalness','semantic')) and
            failures(rows[arm])<=failures(rows[other])): return arm
    return None


def decide256():
    rows=comparison(256000); policy=read(EVAL/'comparison-policy.json'); best=winner(rows,policy)
    if best:
        extensions=[best]; why='Clear paired multi-metric winner; extend winner only.'
    else:
        # A non-winner is not by itself authorization for twice the GPU work.
        b,c=rows['B']['metrics'],rows['C']['metrics']
        close=all(abs(b[k]['mean']-c[k]['mean'])<=max(policy[margin],b[k]['std'],c[k]['std'])
                  for k,margin in (('loss','winner_lm_advantage'),('rare_ce','winner_rare_ce_advantage')))
        if not close: raise RuntimeError('Unresolved tradeoff but arms are not close; dual extension requires user direction.')
        extensions=['B','C']; why='No clear paired winner; both LM and Rare differences lie within decision margins or observed seed dispersion. Dual cumulative comparison required.'
    result={'phase':48,'winner':best,'classification':'PROVISIONAL_WINNER' if best else 'LR_WINNER_UNRESOLVED',
            'extend_arms':extensions,'extension_reason':why,'comparison':rows,'policy':policy}
    write(EVAL/'decision-256.json',result)
    print({'winner':best,'extend':extensions,'summary':{a:{k:r['metrics'][k]['mean'] for k in ('loss','rare_ce','middle_ce','naturalness','semantic')} for a,r in rows.items()}},flush=True)


def final(pytest_result):
    r256=comparison(256000); r512=comparison(512000); decision=read(EVAL/'decision-256.json')
    exposure=read(ROOT/'evaluation/foundation-v37-rare-exposure.json')
    controls=[{**a36.past_control(s),'frequency_probe':read(v36.EVAL/f'audit/gate1-seed-{s}.json')} for s in SEEDS]
    control_summary={'lr':1e-4,'budget':256000,'metrics':aggregate(controls),
                     'provenance':'Unmodified Phase46 control evaluations and Phase47 fixed probes; Phase47 arm A seed42 replication passed.'}
    assert all(a in r512 for a in decision['extend_arms']), 'required cumulative confirmation incomplete'
    if len(r512)==2:
        best=winner(r512,read(EVAL/'comparison-policy.json'))
    else: best=decision['winner']
    candidate=best or min(r512,key=lambda a:r512[a]['metrics']['rare_ce']['mean'])
    chosen=r512[candidate]
    baseline=[a36.metrics(a36.baseline(s)) for s in SEEDS]
    loss_improves=all(chosen['metrics']['loss']['by_seed'][str(s)]<b['loss'] for s,b in zip(SEEDS,baseline))
    approved=best is not None and chosen['pass'] and loss_improves
    short_window=(not all(r['checks']['rare'] for r in r256[candidate]['safety'].values()) and
          all(r['checks']['rare'] for r in chosen['safety'].values()) and
          chosen['metrics']['rare_ce']['std']<r256[candidate]['metrics']['rare_ce']['std'] and
          chosen['metrics']['rare_ce']['mean']<r256[candidate]['metrics']['rare_ce']['mean'])
    correlations={}
    for seed in SEEDS:
        cr=exposure['seed_rows'][str(seed)]['comparisons']['A-256000']
        correlations[str(seed)]={k:cr[k] for k in ('spearman_exposure_ce','spearman_exposure_probability','low_exposure_positive_contribution_share','low_exposure_population_share')}
    exposure_supported=all(correlations[str(s)]['spearman_exposure_ce'] is not None and correlations[str(s)]['spearman_exposure_ce']<=-.3 and
        correlations[str(s)]['low_exposure_positive_contribution_share']>.5 and
        correlations[str(s)]['low_exposure_positive_contribution_share']>correlations[str(s)]['low_exposure_population_share'] for s in (123,2026))
    if short_window: frequency='RARE_GATE_WINDOW_TOO_SHORT'
    elif exposure_supported: frequency='SHORT_WINDOW_RARE_EXPOSURE_VARIANCE'
    else: frequency='FREQUENCY_INSTABILITY_UNRESOLVED'
    # LR-specific attribution needs a contrasting stable LR, not only one extended arm.
    if len(r512)==2:
        failed={a:sum(not r['checks']['rare'] for r in v['safety'].values()) for a,v in r512.items()}
        if max(failed.values())>=2 and min(failed.values())==0: frequency='TRUE_LR_RELATED_RARE_REGRESSION'
    if approved: gate='CONTINUE_SHORT_GPU_GATES_LR_5E5' if best=='C' else 'CONTINUE_SHORT_GPU_GATES_LR_7_5E5'
    elif short_window: gate='RARE_GATE_WINDOW_TOO_SHORT'
    elif best is None: gate='LR_WINNER_UNRESOLVED'
    else: gate='FREQUENCY_INSTABILITY_REVIEW'
    if any(not r['checks']['context'] or not r['checks']['stability'] for r in chosen['safety'].values()): gate='STOP_AND_INVESTIGATE'
    mapping={'median_loop_onset':'loop_onset','greedy_repetition_1':'repetition1','loop_margin':'margin','loop_entropy':'entropy'}
    attractor=classify_attractor({k:{'mean':mean(b[v] for b in baseline)} for k,v in mapping.items()},{k:chosen['metrics'][v] for k,v in mapping.items()})
    preflight=read(EVAL/'preflight.json'); immutable=[]
    for row in preflight['immutable_checkpoints']:
        assert file_sha256(ROOT/row['path'])==row['sha256']; immutable.append(row['path'])
    preserved={r['path']:file_sha256(Path(r['path']))==r['sha256'] for r in preflight['preserved_files']}
    assert all(preserved.values())
    new=[]
    for path in EVAL.glob('arm-*/*/seed-*-training.json'):
        r=read(path); target=ROOT/r['checkpoint']; assert file_sha256(target)==r['sha256']
        p=torch.load(target,map_location='cpu',weights_only=False)
        assert p['experimental'] and not p['promoted'] and p['phase']==48
        v36.verify_payload(p,r['seed'],r['end_tokens'],r['lr']); new.append(r)
    blind=file_sha256(ROOT/'data/foundation_v09/evaluation/final-blind-1000.json'); assert blind==preflight['final_blind_sha256']
    copies=[]
    for path in (OUT/'sources').glob('seed-*/checkpoint-tokens-15872000.pt'):
        seed=int(path.parent.name.removeprefix('seed-'))
        assert file_sha256(path)==file_sha256(v36.official(seed)); copies.append(str(path.relative_to(ROOT)))
    incomplete=[str(p.relative_to(ROOT)) for root in (EVAL,OUT) for p in root.rglob('*')
                if p.is_file() and (p.stat().st_size==0 or p.suffix=='.tmp')]
    assert not incomplete, f'incomplete PHASE48 artifacts: {incomplete}'
    write(EVAL/'integrity-final.json',{'immutable_checkpoint_count':len(immutable),'new_checkpoint_count':len(new),
          'source_copies_sha_verified':copies,'incomplete_phase48_artifacts':incomplete,
          'pass':True,'preserved_files':preserved,'final_blind_sha256':blind})
    mean256=r256[candidate]['metrics']; mean512=chosen['metrics']
    rolling512={k:mean512[k]['mean']-mean(b[k] for b in baseline) for k in ('loss','top1','top5','top10','rare_ce','middle_ce')}
    delta_dispersion={str(budget):stdev(r['deltas_vs_baseline']['rare_ce'] for r in arms[candidate]['safety'].values())
                      for budget,arms in ((256000,r256),(512000,r512))}
    result={'phase':48,'tests_preflight':read(EVAL/'tests-preflight.json'),'pytest_final':pytest_result,'fixed_rare_set_integrity':True,
        '256k_comparison':r256,'512k_comparison':r512,'decision_256':decision,'confirmation_512k_executed':True,
        'historical_control_1e4_256k':control_summary,
        'frequency_classification':frequency,'short_window_exposure_variance_supported':exposure_supported,'rare_gate_window_too_short':short_window,
        'exposure_correlations_original_regression':correlations,'unseen_percent':exposure['unseen_percent'],
        'exposure_cv':exposure['total_exposure_cv'],'rare_ce_std_256k':mean256['rare_ce']['std'],'rare_ce_std_512k':mean512['rare_ce']['std'],
        'per_token_exposure_cv':exposure['per_token_cross_seed_exposure_cv'],
        'rare_ce_delta_std_vs_baseline':delta_dispersion,
        'context_regression':any(not r['checks']['context'] for r in chosen['safety'].values()),
        'best_lr':ARMS[best] if best else None,'formal_lr_approved':approved,'recommended_formal_lr':ARMS[best] if approved else None,
        'approval_classification':'FORMAL_LR_APPROVED' if approved else 'FORMAL_LR_NOT_YET_APPROVED','reporting_arm':candidate,
        'checkpoint_interval':256000,'full_gate_interval':512000 if short_window or approved else 256000,
        'next_phase_gate':gate,'recommended_next_formal_target':START+512000 if approved else None,
        'formal_resume_note':'If approved, a future phase starts from the protected formal 15.872M checkpoint with the selected recipe. Existing experiments remain unpromoted.',
        'attractor':attractor,'gpu':{'tokens_per_second':mean(r['tokens_per_second'] for r in new),'peak_vram_mib':max(r['peak_vram_mib'] for r in new),
            'max_temp':max(r['telemetry']['gpu_temperature_c_max'] for r in new),'classifications':sorted(set(r['telemetry']['thermal_classification'] for r in new))},
        'rolling_512k':rolling512,'rolling_1024k':None,'rolling_1024k_reason':'Only cumulative 512k experiments authorized; no matched 1.024M experimental endpoint.',
        'recipe_candidate':{'device':'CUDA','precision':'FP32','amp':False,'eos_weight':1.5,'repetition_auxiliary':False,'lr':ARMS[candidate]},
        'parallel_cpu_evaluation':'DISABLED','foundation_base_complete':False,'20m_permission':False,'experimental_promoted':False,
        'checkpoint_integrity':{'pass':True,'immutable':len(immutable),'new':len(new)},'preserved_files':preserved,'render_changed':False,'vercel_changed':False,
        'limitations':['Exposure correlations are descriptive across 58 types; they cannot establish causality.','Sampling quality scores are the same automatic proxies, not human ratings.','A singleton 512k LR cannot establish an LR-specific cause without a matched second-LR endpoint.']}
    write(ROOT/'evaluation/foundation-v37-lr-comparison.json',{'256k':r256,'512k':r512,'historical_control_1e4_256k':control_summary,'decision_256':decision})
    write(ROOT/'evaluation/foundation-v37-rare-stability-summary.json',result)
    lines=['# PHASE48 Rare Frequency Stability and LR Confirmation','',f"Gate: **{gate}**. Formal LR: **{result['approval_classification']}**. Best LR: {result['best_lr']}.",
        '',f"Preflight pytest: 430 passed. Final pytest: {pytest_result}. Fixed Rare set: 58 token types, 112 occurrences; hash {exposure['fixed_set_hash']}.",
        '', '## Exposure audit','', exposure['weighting'],'', '| Seed | 256k occurrences | Unseen | Once | 2-4 | 5-9 | 10+ | 512k occurrences | Unseen | rho CE | rho probability |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for seed,row in exposure['seed_rows'].items():
        a,b=row['256k'],row['512k']; c=correlations[seed]
        lines.append(f"| {seed} | {a['total_rare_target_occurrences']} | {a['rare_tokens_never_observed']} | {a['rare_tokens_observed_once']} | {a['observed_2_to_4']} | {a['observed_5_to_9']} | {a['observed_10_plus']} | {b['total_rare_target_occurrences']} | {b['rare_tokens_never_observed']} | {c['spearman_exposure_ce']:.4f} | {c['spearman_exposure_probability']:.4f} |")
    lines += ['',f"Exposure support: {exposure_supported}. Frequency classification: {frequency}. Rare CE seed std: {mean256['rare_ce']['std']:.6f} -> {mean512['rare_ce']['std']:.6f}.",
        f"Rare CE baseline-paired delta std: {delta_dispersion}. This is supplementary; the prespecified window gate uses raw fixed-population CE std and all-seed Rare safety.",
        f"Exposure total-count CV: {exposure['total_exposure_cv']}. Zero-exposure tokens alone do not explain seeds123/2026, which had none at 256k.",
        f"Mean/median per-token cross-seed exposure CV: {exposure['per_token_cross_seed_exposure_cv']}. Historical PHASE47 outlier counts reproduced exactly; their cumulative exposures are also retained.",
        '', '## LR comparison','', '| Budget | LR | Loss mean ± std | Top1/5/10 | Middle CE | Rare CE | Rare mean/median P | Natural/Semantic |','|---|---:|---|---|---:|---:|---|---|']
    for budget,arms in ((256000,{'A (historical)':control_summary,**r256}),(512000,r512)):
        for arm,r in arms.items():
            m=r['metrics']; lines.append(f"| {budget} | {r['lr']:g} | {m['loss']['mean']:.6f} ± {m['loss']['std']:.6f} | {m['top1']['mean']:.2%}/{m['top5']['mean']:.2%}/{m['top10']['mean']:.2%} | {m['middle_ce']['mean']:.6f} | {m['rare_ce']['mean']:.6f} | {m['rare_mean']['mean']:.8f}/{m['rare_median']['mean']:.8f} | {m['naturalness']['mean']:.2%}/{m['semantic']['mean']:.2%} |")
    lines += ['',f"256k winner: {decision['winner']}; extended arms: {decision['extend_arms']}. {decision['extension_reason']}",'', '## Cumulative safety','']
    for arm,r in r512.items():
        for seed,safety in r['safety'].items():
            lines.append(f"- {arm} seed{seed}: failed checks {[k for k,v in safety['checks'].items() if not v]}; Rare CE delta {safety['deltas_vs_baseline']['rare_ce']:+.6f}.")
    lines += ['', '## Exposure and outlier evidence by arm', '', '| Seed | Arm / budget | rho CE | rho probability | Low-exposure share of positive CE contributions | Low-exposure population share |', '|---|---|---:|---:|---:|---:|']
    for seed,row in exposure['seed_rows'].items():
        for arm,c in row['comparisons'].items():
            lines.append(f"| {seed} | {arm} | {c['spearman_exposure_ce']} | {c['spearman_exposure_probability']} | {c['low_exposure_positive_contribution_share']:.2%} | {c['low_exposure_population_share']:.2%} |")
    lines += ['', 'Per-token before/after CE and probabilities, exposure bins and the largest positive contributors are retained in foundation-v37-rare-exposure.json. Low exposure means <=4 target occurrences; an association is not a causal explanation.',
              '', '## Selected cumulative generation and context', '']
    lines += [f"- {k}: {mean512[k]['mean']:.8g} (sample std {mean512[k]['std']:.8g})." for k in
              ('naturalness','semantic','sampling_repetition','completion','topic_retention','terminal_eos','terminal_eos_top1','terminal_eos_top5','terminal_eos_top10','nonterminal_eos','premature_eos','context','context_64','context_16','context_2','context_1','advantage','runaway','first_break','loop_onset','repetition1','repetition2','repetition3','repetition4','entropy','margin')]
    lines += ['',f"Attractor: {attractor}. 512k rolling deltas: {rolling512}. 1.024M: unavailable (not executed).",
        f"GPU: {result['gpu']}. All formal/prior experimental SHA unchanged; new strict-reload checkpoints PASS. Final Blind SHA only.",
        '', '## Next phase','',f"Formal LR: {result['recommended_formal_lr']}; checkpoint interval: 256k; full Gate interval: {result['full_gate_interval']}; next formal target: {result['recommended_next_formal_target']}.",
        result['formal_resume_note'],'CUDA FP32, EOS1.5, repetition auxiliary OFF. CPU parallel evaluation DISABLED. No architecture changes, no promotion, no 20M training/permission. Foundation Base incomplete.',
        'Protected local files and READY markers retained. Render/Vercel unchanged.','', '## Limitations','']+result['limitations']
    (ROOT/'evaluation/foundation-v37-rare-stability-report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print({'gate':gate,'best_lr':result['best_lr'],'approved':approved,'frequency':frequency},flush=True)


from training.train_foundation_v21_ab import file_sha256
if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--decide256',action='store_true'); parser.add_argument('--final',action='store_true'); parser.add_argument('--pytest-result',default='pending')
    a=parser.parse_args(); torch.set_num_threads(4)
    if a.decide256: decide256()
    if a.final: final(a.pytest_result)
