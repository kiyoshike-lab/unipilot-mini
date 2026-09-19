"""Synthesize PHASE60 observations and freeze one future stabilization test.

Registration never grants execution authorization and never loads a model.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation import audit_foundation_v49 as a


def phase61_decision(results):
    if type(results) is not dict or set(results)!={'42','123','2026'}:
        return 'EXPERIMENT_INVALID'
    required={'valid','control_all34_vs_parent','half_lr_all34_vs_parent','half_lr_all34_vs_control'}
    for row in results.values():
        if type(row) is not dict or set(row)!=required or any(type(v) is not bool for v in row.values()) or not row['valid']:
            return 'EXPERIMENT_INVALID'
    intervention=all(r['half_lr_all34_vs_parent'] and r['half_lr_all34_vs_control'] for r in results.values())
    if not intervention:return 'CONTINUATION_STABILIZATION_NOT_ESTABLISHED'
    if all(r['control_all34_vs_parent'] for r in results.values()):return 'BOTH_STABLE_NO_DIFFERENTIAL_EVIDENCE'
    return 'CONTINUATION_STABILIZATION_SUPPORTED'


def synthesize():
    a.verify_frozen()
    matrix=a.read(a.OUT/'three-seed-failure-matrix.json');gradient=a.read(a.OUT/'gradient-clipping-audit.json')
    optimizer=a.read(a.OUT/'optimizer-state-audit.json');data=a.read(a.OUT/'data-order-audit.json')
    frequency=a.read(a.OUT/'frequency-forgetting-audit.json');normal=a.read(a.OUT/'normal-control-audit.json')
    saturation=all(r['clip_rate']==1 for r in gradient['seeds'].values())
    specialization=all(r['scalars']['validation.ce']['delta']<0 and r['scalars']['core.macro_ce']['delta']>0
                       and r['scalars']['normal.mean_ce']['delta']>0 for r in matrix['seeds'].values())
    assert saturation and specialization and frequency['signature']=='YES'
    gate='MIXED_CONTINUATION_INSTABILITY'
    summary={'phase':60,'gate':gate,'actionable':True,'new_training':False,'new_inference':False,'optimizer_steps':0,
       'signatures':['CLIPPING_SATURATION_SIGNATURE','FREQUENCY_FORGETTING_SIGNATURE','SHORT_CONTINUATION_SPECIALIZATION_SIGNATURE'],
       'dominant_claim':False,'causal_proof':False,
       'reason':'All three seeds are clipped at every step and show validation improvement with Core/normal CE worsening. Exposure-associated forgetting replicates within seeds; seed123 normal failures persist under every within-family leave-one-out. These streams support a mixed short-continuation instability signature, but do not isolate data order, parent weights, moments, or clipping causally.',
       'frequency_evidence_independence':'Spearman, zero/nonzero, and quartiles reuse the same exposure/CE pairs; they count as one stream, not three.',
       'optimizer_classification':optimizer['classification'],
       'optimizer_interpretation':'Parent global relative adaptive-direction max/min=1.0315; candidate=1.0091. This supplies no positive evidence of a gross seed-specific scale divergence. No causal optimizer gate or arbitrary outlier cutoff is introduced.',
       'seed_findings':{},'approved_lr_status':'FORMAL_LR_APPROVED_5E5','approved_lr':5e-5,'generation_policy':'UNSAFE',
       'phase57':'EXPERIMENT_INVALID','phase59':'CONTROL_STABILITY_MIXED','arm_a_context':['INTERVENTION_CONFOUNDED_BY_CONTROL_DRIFT','INSUFFICIENT_DOSE_EVIDENCE'],
       'anti_loop_training':False,'canonical':False,'20m':False,'foundation_base':False,'production':False,
       'reserve2':'SEALED_UNSCORED','final_blind':'SHA_ONLY','phase53_reserve':'RETIRED_UNSCORABLE',
       'phase61_preregistration':'CREATED','phase61_training_authorized':False,
       'limitations':['Three observational seeds; parent weights, permutation and optimizer state co-vary. No randomized mechanism intervention.',
                      'Validation improvement is small; this study does not retest PHASE55 LR model selection.',
                      'Normal controls have only four examples per family. Literal train-text counts are weak exposure proxies.',
                      'Endpoint parameter deltas are not the sum of individual step lengths. Post-clip norms are calculated, not measured.',
                      'Generation quality metrics are automatic proxies. Near-100% runaway is a pre-existing failure, not an anti-loop efficacy finding.',
                      'Historical scheduler LR fields are stale; actual optimizer group LR is 5e-5 in all six payloads. No executed scheduler was found in the frozen continuation loops.'],
       'artifact_sha256':{name:a.sha(a.OUT/name) for name in ('three-seed-failure-matrix.json','gradient-clipping-audit.json','optimizer-state-audit.json','data-order-audit.json','frequency-forgetting-audit.json','normal-control-audit.json','scheduler-metadata-audit.json')}}
    for seed in ('42','123','2026'):
        summary['seed_findings'][seed]={'scalars':matrix['seeds'][seed]['scalars'],'failed_safeguards':matrix['seeds'][seed]['failed_checks'],
            'normal_classification':normal['seeds'][seed]['classification'],'gradient':gradient['seeds'][seed],
            'net_update':optimizer['net_parameter_updates'][seed]['total'],'frequency':frequency['seeds'][seed],
            'generation':matrix['seeds'][seed]['generation']}
    a.emit(ROOT/'evaluation/foundation-v49-continuation-instability-summary.json',summary)
    # The only proposed intervention is chosen before any outcome under that LR exists.
    s=a.spec(); pre=a.read(a.OUT/'preflight.json')
    registration={'schema_version':'phase61-continuation-stability-preregistration-v1','phase':61,'registered_in_phase':60,
       'status':'REGISTERED_REQUIRES_NEW_USER_TRAINING_AUTHORIZATION','training_authorized':False,'training_executed':False,
       'purpose':'Test short continuation safety at half the approved LR; not anti-loop efficacy or PHASE55 LR reselection.',
       'audit_gate':gate,'audit_summary_sha256':a.sha(ROOT/'evaluation/foundation-v49-continuation-instability-summary.json'),
       'intervention':'HALF_CONTINUATION_LR','mechanistic_rule':'Exactly approved5e-5 / 2 = 2.5e-5; a single predefined proportional displacement reduction. No LR sweep or outcome-dependent tuning.',
       'rationale':'Similar endpoint update norms and pervasive clipping coexist with broad low-exposure forgetting and normal-family loss. Halving optimizer LR directly scales the adaptive and decoupled-decay step at a fixed moment/gradient state. This is a cautious testable displacement hypothesis, not proof that LR caused forgetting or that a nonlinear 122-step trajectory will halve.',
       'alternatives_not_selected':{'clipping':'100% clipping is descriptive; it does not establish a suitable new clip threshold with AdamW moments. Keep clip1.0.',
          'moment_reset':'No strong gross moment-scale divergence and no evidence supporting reset. Preserve all moments.',
          'data_balancing':'Exposure association is consistent but parent/permutation are confounded; no evaluation-token oversampling or outcome-selected sampling.'},
       'seeds':[42,123,2026],'seed_budget_tradeoff':'All three are needed to cover seed42/2026 Core failures and seed123 normal-family failures. Six bounded fresh runs; no extra seeds.',
       'arms':[{'id':'control','lr':5e-5},{'id':'half-lr','lr':2.5e-5}],
       'run_order':[{'seed':seed,'arm':arm,'cooldown_before':True} for seed in (42,123,2026) for arm in ('control','half-lr')],
       'parents':[r for r in pre['checkpoints'] if r['role']=='parent'],
       'parent_mutation_operations':dict.fromkeys(('copy','move','delete','overwrite','rename'),0),
       'objective':{'standard_LM':True,'EOS_weight':1.5,'anti_loop':False,'unlikelihood':False,'phase42_auxiliary':False},
       'optimizer':{'kind':'AdamW','strict_reload':True,'reset_moments':False,'clip_norm':1.0,
                    'allowed_group_change':'Only lr=arm.lr after strict reload; all moments, step counters and other group fields byte-equivalent before first update.',
                    'before_first_step_assertions':['model equality','moments equality','non-LR groups equality','parent RNG restored exactly','full permutation unchanged','exact next122 hash']},
       'scheduler':{'start_global_step':32000,'end_global_step':32122,'warmup_restart':False,
          'legacy_metadata':'Preserve parent scheduler fields except global_step. Its learning_rate/peak_learning_rate=1e-4 are historical metadata, not runtime authority.',
          'runtime_lr_authority':'optimizer.param_groups[*].lr = registered arm.lr; assert every update; no scheduler.step.',
          'output_metadata':'Include a separate phase61_runtime_lr_contract containing arm.lr, constant schedule, and preserved historical metadata warning.'},
       'sampler':{'rule':'Each fresh arm consumes its own parent.permutation[32000:32122] in order; macro_batch512 one row. No redraw/shuffle.',
                  'expected_next122_sha256':{seed:r['permutation_sha256'] for seed,r in data['seeds'].items()}},
       'budget':{'updates_per_run':122,'LM_gradient_tokens_per_run':62464,'matching_slots_per_run':15,'matching_charged_positions_per_run':1440,
                 'conservative_positions_per_run':63904,'end_processed_tokens':16446464,'total_runs':6,'maximum_optimizer_updates':732,
                 'total_LM_gradient_tokens':374784,'total_conservative_positions':383424,'extension':False,'retry_after_started_training':False},
       'data':s['data'],'matching_no_grad_forward':{**s['matching_no_grad_forward'],'state_assertions':['Python/NumPy/Torch CPU/all CUDA RNG','all model buffers and weights','optimizer','scheduler'],
               'exact_indices':list(range(32,47)),'optimizer_gradients':'None from matching forward'},
       'hardware':s['hardware'],'stop_conditions':['Any hash, state, RNG, geometry, nonfinite, telemetry or storage failure: abort; no retry or added budget.',
            'Raw gradient norm >10 for three consecutive updates or >100 once: abort; clip1.0 unchanged.',
            'Temperature >=80C pause, resume <=65C, pre-run target <=60C; >=85C or hardware thermal slowdown abort.',
            'No CPU training fallback or concurrent GPU job/full pytest/build/browser QA.',
            'No optimizer step until this registration is SHA-frozen and separately explicitly authorized; all 6 live CUDA contract dry runs must pass before training.',
            'No existing/partial output collisions; atomic exclusive checkpoint save, SHA and full strict model/optimizer/scheduler/sampler/RNG reload.',
            'Require 2x next checkpoint size plus2GiB reserve (and existing storage policy minimum), checked every update.'],
       'evaluation':{'schema':a.contract.VERSION,'schema_sha256':a.SCHEMA_SHA,'quality_schema':a.contract.SAFETY_VERSION,
           'sets':s['evaluation_sets'],'context_reference':s['context_reference'],'generation':s['generation'],
           'bootstrap':'10000 paired document draws, seed5701, fixed sorted512-block population geometry, per seed comparator.',
           'all_thresholds':s['all_safeguards'],'all_34_mappings':s['safeguard_mapping'],
           'comparators':'Control vs own parent; half-lr vs own parent AND same-seed fresh control. Never average failed seeds away.',
           'historical_controls':'PHASE57/59 control results are background only; fresh matched controls are required. PHASE57 remains INVALID.',
           'primary':'All34 checks for half-lr against both comparators must pass in every seed, including Core micro/macro<=+0.1 and all normal safeguards.'},
       'decision_rules':{'EXPERIMENT_INVALID':'Any of six runs/evaluations incomplete, contract/hash/state/RNG/geometry/thermal failure, undefined ratio or nonfinite.',
           'CONTINUATION_STABILIZATION_NOT_ESTABLISHED':'Valid complete study but any half-lr seed fails any registered safeguard against own parent or matched control.',
           'BOTH_STABLE_NO_DIFFERENTIAL_EVIDENCE':'Half-lr passes all checks on all seeds and all fresh controls also pass all34 vs own parents.',
           'CONTINUATION_STABILIZATION_SUPPORTED':'Half-lr passes all checks on all seeds and at least one fresh control fails safety vs its own parent.'},
       'checkpoint_policy':{'root_env':'UNIPILOT_CHECKPOINT_ROOT','root':str(a.ZROOT),'new_relative_root':'experimental/phase61/continuation-stability/{arm}/seed-{seed}',
                'markers':['EXPERIMENTAL','NOT_CANONICAL','NOT_PRODUCTION'],'atomic_exclusive':True,'canonical_promotion':False},
       'sealed_sets':s['sealed_sets'],'approved_lr_status':'FORMAL_LR_APPROVED_5E5','generation_policy':'UNSAFE',
       'phase57':'EXPERIMENT_INVALID','phase59':'CONTROL_STABILITY_MIXED','production':False,'20m':False,'foundation_base':False,
       'source_sha256':{**s['source_sha256'], 'evaluation/audit_foundation_v49.py':a.sha(ROOT/'evaluation/audit_foundation_v49.py'),
                       'evaluation/register_foundation_v49.py':a.sha(Path(__file__))}}
    a.artifact('phase61-continuation-stability-preregistration.json',registration)
    a.artifact('phase61-registration-receipt.json',{'gate':'PHASE61_DESIGN_REGISTERED','training_authorized':False,
        'sha256':a.sha(a.OUT/'phase61-continuation-stability-preregistration.json')})
    print(gate,'; PHASE61 design registered, training NOT authorized',flush=True)


def finish():
    pre=a.verify_frozen(); qa={scope:a.read(a.OUT/f'pytest-{scope}.json') for scope in ('targeted','full')}
    assert all(v['exit_code']==0 and v['failed']==0 and v['errors']==0 and v['repo_fixtures_unchanged'] for v in qa.values())
    summary=a.read(ROOT/'evaluation/foundation-v49-continuation-instability-summary.json');matrix=a.read(a.OUT/'three-seed-failure-matrix.json')['seeds']
    grad=a.read(a.OUT/'gradient-clipping-audit.json')['seeds'];opt=a.read(a.OUT/'optimizer-state-audit.json');data=a.read(a.OUT/'data-order-audit.json')['seeds'];freq=a.read(a.OUT/'frequency-forgetting-audit.json')['seeds'];normal=a.read(a.OUT/'normal-control-audit.json')['seeds']
    a.artifact('test-side-effect-audit.json',{'mutating_test':'tests/test_campus_ai_quality.py::test_quality_evaluation_artifacts_and_review_filter',
        'writer':'evaluation.evaluate_campus_ai_quality.evaluate writes OUTPUT_20/OUTPUT_100/CLOSE_ANALYSIS/CRITICAL_FAILURE/REVIEW_QUEUE/REPORT',
        'isolation':'Autouse pytest monkeypatch redirects all six outputs to tmp_path; production module is unchanged.',
        'repo_sha_guard':'Session fixture asserts all six repository artifacts unchanged; external QA wrapper checks all pre-existing dirty/frozen/protected files before and after each pytest invocation.',
        'new_test_behaviors':['Full real evaluator writes temporary artifacts only','Interrupted evaluator after first write leaves repo untouched'],
        'dirty5_initial':pre['dirty5'],'dirty5_preserved':True,'restored':False,'stage':False,'isolation_gate':'PASS','pytest':qa})
    a.artifact('final-qa.json',{'pytest':qa,'protected4_preserved':True,'READY5_preserved':True,'existing_dirty_preserved':True,
        'dirty_count_untracked_expanded':len(pre['dirty']),'dirty5_preserved':True,'old_phases_immutable':True,'all6_checkpoint_SHA_unchanged':True,
        'new_training':False,'new_inference':False,'optimizer_steps':0,'main_unchanged':True,'production_operations':0,
        'raw_inventory':[{'path':str(p),'sha256':a.sha(p)} for p in sorted(a.RAW.iterdir()) if p.is_file()],
        'npm_build':'NOT_REQUIRED_NO_WEB_CODE_CHANGE','browser_QA':'NOT_REQUIRED_NO_WEB_CODE_CHANGE'})
    lines=['# PHASE60 / Foundation v4.9 — Continuation instability audit','',
        'Gate: **'+summary['gate']+'**. Zero new training, zero inference, zero optimizer steps. Six existing checkpoint SHA and strict reload checks passed.',
        '',summary['reason'],'',
        'This is an observational comparison of three seeds. Clipping, data exposure, parent weights and moments were not independently randomized. No dominant causal mechanism is established.',
        '', '## Failure matrix','', '| Metric delta (candidate − own parent) | seed42 | seed123 | seed2026 |','|---|---:|---:|---:|']
    keys=list(matrix['42']['scalars'])
    for key in keys:lines.append('| '+key+' | '+' | '.join(f"{matrix[s]['scalars'][key]['delta']:+.8f}" for s in ('42','123','2026'))+' |')
    lines+=['','All 34 existing comparisons, parent/candidate values, and 10,000 paired-document bootstrap intervals are in `phase60/three-seed-failure-matrix.json`. BORDERLINE means exact equality to the original inclusive margin; no additional tolerance exists.',
        'Seed42 calculations are descriptive diagnostics only. PHASE57 stays EXPERIMENT_INVALID and Arm A was not rescored. PHASE59 stays CONTROL_STABILITY_MIXED.']
    for s in ('42','123','2026'):
        lines += ['',f"- seed{s} failed checks: {', '.join(matrix[s]['failed_checks'])}.",f"  Core bootstrap: {matrix[s]['bootstrap']['core']}."]
    lines+=['','## Gradients and endpoint update vectors','', '| Seed | Clip rate | Mean | Median | P90 | P95 | Max | Net update norm | Relative norm |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in ('42','123','2026'):
        g=grad[s]['raw_norm'];u=opt['net_parameter_updates'][s]['total']
        lines.append(f"| {s} | {grad[s]['clip_rate']:.0%} | "+' | '.join(f'{g[k]:.6f}' for k in ('mean','median','p90','p95','max'))+f" | {u['net_update_norm']:.6f} | {u['parameter_relative_net_update_norm']:.6f} |")
    lines+=['','Post-clip norms and scales are derived from historical raw norms using min(1,1/(norm+1e-6)); the old runs did not log measured post-clip norms. Endpoint displacement/122 is not mean per-step path length. Tied head is counted once.',
        '', '| Component net update norm | seed42 | seed123 | seed2026 |','|---|---:|---:|---:|']
    for k in ('embedding_tied_head','attention','FFN','LayerNorm','position_embedding'):
        lines.append('| '+k+' | '+' | '.join(f"{opt['net_parameter_updates'][s]['component:'+k]['net_update_norm']:.6f}" for s in ('42','123','2026'))+' |')
    lines+=['','| Transformer layer net update norm | seed42 | seed123 | seed2026 |','|---|---:|---:|---:|']
    for layer in range(10):lines.append('| '+str(layer)+' | '+' | '.join(f"{opt['net_parameter_updates'][s]['layer:'+str(layer)]['net_update_norm']:.6f}" for s in ('42','123','2026'))+' |')
    for pair,groups in opt['cross_seed_update_vectors'].items():lines.append(f"\n- Update cosine {pair}: {groups['total']['cosine']:.8f}; norm ratio {groups['total']['norm_ratio_first_over_second']:.8f}.")
    lines+=['','Full per-layer cosines, first/second moment norms before/after, and parameter-relative adaptive directions are in `phase60/optimizer-state-audit.json`. Coordinates belong to different parent weights; these cosines are local direction similarities only.',
        '', 'Optimizer classification: OPTIMIZER_EVIDENCE_INSUFFICIENT. Parent/candidate global adaptive relative-direction max/min ratios are 1.0315/1.0091. Position embeddings have the largest component-relative displacement (~3.4%), consistently across all seeds; a seed-specific optimizer explosion is not demonstrated.',
        '', 'Historical scheduler learning_rate and peak_learning_rate metadata remain 1e-4, while all six actual optimizer groups hold 5e-5. The frozen loops never call a scheduler. This shared stale metadata does not explain between-seed failures; future execution must assert the actual group LR at every update.',
        '', '## Exposure and forgetting','', '| Seed | Core occurrences | Tail occurrences | EOS density | Rare share | Core zero exposure | Worsened tokens | Spearman | Zero/nonzero mean CE delta |','|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for s in ('42','123','2026'):
        d,f=data[s],freq[s]
        lines.append(f"| {s} | {d['core_target_occurrences']} | {d['tail_target_occurrences']} | {d['eos_density']:.6f} | {d['rare_share_rank_ge3277']:.6f} | {f['zero_exposure_fraction']:.2%} | {f['worsened_fraction']:.2%} | {f['spearman_exposure_vs_ce_delta']:.6f} | {f['zero_exposure_delta']['mean']:+.6f} / {f['nonzero_exposure_delta']['mean']:+.6f} |")
    lines+=['','FREQUENCY_FORGETTING_SIGNATURE: YES (observational). Spearman, zero/nonzero split and exposure quartiles reuse the same token-level evidence and do not count as independent dominance proof. All three token CE distributions, full4096-ID target/input histograms, per-token Core exposure, EOS/special/repetition density and BOS document representation are retained in the audit artifacts; token raw is on Z:.',
        '', 'Consumed122-block sets share zero blocks across every pair of seeds. BOS-derived document counts are 142/146/129; category labels are unavailable. Family exposure proxies count literal substrings only and are not authoritative categories.',
        '', f"Seed2026 worsened {freq['2026']['worsened_fraction']:.2%} of 236 Core types; CE delta median/P75/P90 = "+' / '.join(f"{freq['2026']['ce_delta_distribution'][k]:.6f}" for k in ('median','p75','p90'))+'.',
        '', '## Normal controls and EOS/generation','', '| Family delta | seed42 | seed123 | seed2026 |','|---|---:|---:|---:|']
    for f in a.contract.FAMILIES:lines.append('| '+f+' | '+' | '.join(f"{normal[s]['families'][f]['delta']:+.6f}" for s in ('42','123','2026'))+' |')
    lines+=['','seed42 has no normal-control safeguard failure. seed123 fails mean/code/lists/terminology; each failed family remains failed after removal of every individual example (NORMAL_FAILURE_BROAD). seed2026 math remains failed in every leave-one-out (NORMAL_FAILURE_FAMILY_SPECIFIC). The full 20 paired examples, unchanged margins, individual positive mass shares and leave-one-out deltas are in `phase60/normal-control-audit.json`.',
        '', '| Candidate generation | seed42 | seed123 | seed2026 |','|---|---:|---:|---:|']
    for kind in ('runaway_rate','eos_completion_rate','naturalness_rate','semantic_rate','topic_retention_rate','japanese_validity_rate'):
        lines.append('| Sampling '+kind+' | '+' | '.join(f"{matrix[s]['generation']['candidate']['sampling_mean'][kind]:.6f}" for s in ('42','123','2026'))+' |')
    lines+=['','Greedy runaway remains 100% for all three candidates. Terminal/nonterminal EOS deltas are in the failure matrix above. Generation did not become safe and this audit does not measure anti-loop efficacy.',
        '', '## PHASE61 preregistration','',
        'CREATED: fresh Control 5e-5 plus exactly one half-LR continuation arm at 2.5e-5. Three seeds42/123/2026; each arm starts from its unchanged16.384M parent,122 updates,62,464 LM gradient tokens+15 matched no-grad slots=63,904 conservative positions. Six runs total:374,784 LM tokens;383,424 conservative positions. No extension.',
        'The mechanistic rule is one fixed half-scale LR intervention before any candidate outcome. Preserve moments/permutation/RNG, EOS1.5, clip1.0, CUDA FP32, no anti-loop; no sweep. Success requires every half-LR seed to pass all34 unchanged safeguards against both own parent and fresh matched control. All-six execution validity is required; failed seeds cannot be averaged away.',
        'Schema SHA256: '+a.SCHEMA_SHA+'. Full thresholds, identities, run order, runtime LR authority and decision rules are frozen in `phase60/phase61-continuation-stability-preregistration.json`. **training_authorized=false**. No PHASE61 training was run.',
        '', '## Test isolation and final checks','',
        'The mutating Campus test now reads/writes temporary outputs via a pytest-only monkeypatch. Session checksums guard all six repository output files, including the report. The five current dirty JSONs were not restored, edited or staged. Production evaluator paths/model logic are unchanged.']
    for scope,r in qa.items():lines.append(f"\n- {scope} pytest: exit{r['exit_code']}, passed{r['passed']}, failed{r['failed']}, errors{r['errors']}, skipped{r['skipped']}, warnings{r['warnings']}. Log and JUnit SHA are in `phase60/pytest-{scope}.json`.")
    lines+=['','All current dirty files (179 expanded paths;151 collapsed status entries), protected4 and READY5, old PHASE57/58/59 outputs, and six checkpoints match their starting SHA. Raw/checkpoint binaries are excluded from Git. No main, Render or Vercel production operation was performed.',
        '', 'Approved LR remains FORMAL_LR_APPROVED_5E5; Generation Policy UNSAFE; PHASE57 EXPERIMENT_INVALID; PHASE59 CONTROL_STABILITY_MIXED. Canonical promotion/20M/Foundation Base: NO. Reserve2 SEALED/UNSCORED; Final Blind SHA ONLY.']
    path=ROOT/'evaluation/foundation-v49-continuation-instability-report.md'
    with path.open('x',encoding='utf8') as f:f.write('\n'.join(lines)+'\n')
    print('PHASE60 final report/QA created',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('register','finish'));args=parser.parse_args()
    synthesize() if args.action=='register' else finish()
