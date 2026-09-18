"""Freeze PHASE59 control-only design after PHASE58 gates; never train."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation.audit_foundation_v47 import OUT,RAW,OLDRAW,check_frozen,read,sha,write,spec,VERSION
from evaluation import generation_contract_v47 as contract


def decision(seed_results):
    """Future decision only: valid paired control results for both new seeds."""
    if type(seed_results) is not dict or set(seed_results)!={'123','2026'}:
        return 'EXPERIMENT_INVALID'
    gates=[]
    for row in seed_results.values():
        if type(row) is not dict or set(row)!={'valid','all_safety_pass','core_micro_pass','core_macro_pass'} or any(type(v) is not bool for v in row.values()) or not row['valid']:
            return 'EXPERIMENT_INVALID'
        if row['all_safety_pass'] and not (row['core_micro_pass'] and row['core_macro_pass']):
            return 'EXPERIMENT_INVALID'
        gates.append(row)
    if all(r['all_safety_pass'] for r in gates):return 'CONTROL_STABLE_MULTISEED'
    if all(not (r['core_micro_pass'] and r['core_macro_pass']) for r in gates):return 'CONTROL_DRIFT_REPLICATED'
    return 'CONTROL_STABILITY_MIXED'


def register():
    pre=check_frozen();audit=read(OUT/'evaluator-contract-audit.json');forensic=read(OUT/'phase57-forensic-summary.json');drift=read(OUT/'control-drift-audit.json');dose=read(OUT/'intervention-dose-audit.json')
    if audit['gate']!='EVALUATOR_CONTRACT_READY' or not all(r['complete'] for r in (forensic,drift,dose)):
        raise RuntimeError('PHASE59_REGISTRATION_GATE_NOT_MET')
    assert audit['contract_sha256']==sha(OUT/'evaluator-contract-v2.json')
    assert forensic['phase57_status']=='EXPERIMENT_INVALID' and not forensic['old_outputs_rescored_for_efficacy']
    parents=[r for r in pre['checkpoints'] if r['label'].startswith('candidate-parent-')]
    assert len(parents)==2 and all(r['integrity']['pass'] and r['rng_roundtrip'] for r in parents)
    assert all(drift['continuity'].values()) and audit['no_grad_matching_forward']['rng_unchanged'] and audit['no_grad_matching_forward']['state_unchanged']
    old=spec();thresholds=old['quality_safeguards_vs_each_parent_and_control']
    sources=['evaluation/generation_contract_v47.py','evaluation/audit_foundation_v47.py','evaluation/register_foundation_v47_control.py','tests/test_foundation_v47_contract.py','tests/test_foundation_v47_audit.py',
             'evaluation/diagnose_foundation_v29_generation.py','evaluation/diagnose_foundation_v40.py','evaluation/evaluate_foundation_v39_gate.py','evaluation/confirm_foundation_v44.py','evaluation/run_foundation_v45_observatory.py',
             'training/foundation_v31_objective.py','training/run_foundation_v30_eos_experiment.py','training/train_foundation_v15_controlled.py','training/train_foundation_v21_ab.py','training/run_foundation_v35_thermal_gate.py','training/checkpoint_paths.py',
             'foundation/diagnostic_transformer_v17.py','tokenizer/foundation-v11-base-4096.json']
    artifact_sources=['evaluation/phase56/phase57-training-preregistration.json','evaluation/phase56/observability-preregistration.json','evaluation/phase58/evaluator-contract-v2.json','evaluation/phase58/evaluator-contract-audit.json','evaluation/phase58/control-drift-audit.json','evaluation/phase58/intervention-dose-audit.json']
    value={
        'schema_version':'phase59-control-stability-preregistration-v1','phase':59,'registered_in_phase':58,
        'status':'REGISTERED_REQUIRES_NEW_USER_TRAINING_AUTHORIZATION','training_authorized':False,'training_executed':False,
        'purpose':'Continuation local stability across unused seed123/2026 parents. No intervention efficacy; no reversal of PHASE55 LR model selection.',
        'design':'CONTROL_STABILITY_REPLICATION','seeds':[123,2026],'run_order':[123,'cooldown',2026],
        'parents':parents,'approved_lr':5e-5,'EOS_weight':1.5,'anti_loop_intervention':False,'coefficient':0.0,'old_repetition_auxiliary':False,
        'optimizer_continuity':'Strict-load each exact parent optimizer; preserve every group hyperparameter and moments. No reset; approved constant LR5e-5; clip_grad_norm1.0.',
        'scheduler':'Preserve parent state; global_step advances32000->32122, constant LR. No warmup/restart.',
        'sampler':'Preserve full permutation; consume parent.permutation[32000:32122] in order; hashes recorded separately per parent. Same controlled macro_batch context512 (one row512). No shuffle/redraw.',
        'rng':'Restore and compare Python, NumPy, PyTorch CPU, all CUDA RNG states before first update. No seed42 retraining.',
        'budget':{'updates_each':122,'start_update':32000,'end_update':32122,'LM_tokens_each':62464,'matching_no_grad_slots_each':15,'matching_forward_positions_charged_each':1440,'conservative_accounting_each':63904,'gradient_bearing_positions_each':62464,'end_processed_tokens_each':16446464,'extension':False,'maximum_updates_each':122},
        'matching_no_grad_forward':{'required':True,'purpose':'Preserve PHASE57 Control exposure/order despite no observed RNG/state change; additional runtime only, no aux gradients',
            'cache_path':str(OLDRAW/'cache-raw.json'),'cache_sha256':sha(OLDRAW/'cache-raw.json'),'cadence':'absolute_update %8 ==0','episode_cursor':'(absolute_update//8-1)%128; exact indices32..46',
            'inputs':'episode.prefix + episode.generated[:31]; actual length <=95, charge96 positions',
            'mode':'eval(), torch.no_grad(), restore train(); assert no RNG or model-buffer change around matching forward',
            'no_regeneration_no_new_negatives':True},
        'evaluation_schema_version':contract.VERSION,'evaluation_schema_path':'evaluation/phase58/evaluator-contract-v2.json','evaluation_schema_sha256':sha(OUT/'evaluator-contract-v2.json'),
        'quality_schema_version':contract.SAFETY_VERSION,'all_safeguards':thresholds,'safeguard_mapping':contract.mapping(thresholds),
        'comparison':'Each seed candidate versus its own unchanged parent; no cross-seed averaging of failed safety margins',
        'evaluation_sets':old['evaluation_sets'],
        'context_reference':{'path':'evaluation/phase56/observability-preregistration.json','sha256':sha(ROOT/'evaluation/phase56/observability-preregistration.json'),'selection':'identical24 consumed Diagnostic IDs, never gradient data'},
        'data':old['training_data']['base'],
        'core_primary_safety':{'micro_ce_increase_max':.1,'macro_ce_increase_max':.1,'population_unchanged':True},
        'bootstrap':'Paired document10000, seed5701, micro CE upper separately for Core and Supported Tail; exact fixed membership and sorted512-block geometry',
        'generation':'100 PHASE51 prompts. Greedy128. Sampling temp0.7/max64, RNG44000/51000/52000+index. Explicit legacy-v1 conversion. Safeguards read equal-size sampling means only, never greedy fallback.',
        'completion_metrics':'automatic proxy and EOS completion separate. Shape normal-control output is descriptive only; teacher-forced CE/EOS are safety metrics.',
        'decision_labels':['CONTROL_STABLE_MULTISEED','CONTROL_DRIFT_REPLICATED','CONTROL_STABILITY_MIXED','EXPERIMENT_INVALID'],
        'decision_rules':{'CONTROL_STABLE_MULTISEED':'Both seeds pass ALL34 concrete comparisons.',
            'CONTROL_DRIFT_REPLICATED':'Both seeds fail at least one Core primary mean CE margin (>0.1 micro OR macro); no averaging, bootstrap failures alone do not establish primary-mean drift.',
            'CONTROL_STABILITY_MIXED':'All valid remaining combinations, including one seed fails or non-Core safety fails.',
            'EXPERIMENT_INVALID':'Either seed incomplete, schema/hash/continuity/geometry failure, missing metrics, undefined ratios, nonfinite or hardware abort.'},
        'no_partial_decision':True,'next_steps':{'CONTROL_STABLE_MULTISEED':'Consider separately authorized fresh anti-loop registration; no automatic training.',
            'CONTROL_DRIFT_REPLICATED':'Investigate continuation training scheme before any anti-loop trial.',
            'CONTROL_STABILITY_MIXED':'Investigate seed/permutation sensitivity; no automatic retry.'},
        'hardware':{'device':'CUDA','gpu':'NVIDIA GeForce RTX 2070 SUPER','dtype':'float32','AMP':False,'TF32':False,'CPU_training_fallback':False,'parallel_CPU_evaluation':False,'parallel_build_browser_pytest':False,'cooldown_target_c':60,'pause_c':80,'abort_c':85,'hardware_thermal_slowdown':'ABORT','telemetry_failure':'ABORT','settings_modification':False},
        'stop_conditions':old['stop_conditions']+['No optimizer step until a new explicit PHASE59 user authorization and live contract dry-run per parent pass.','Abort incomplete pair; no efficacy conclusion, no rerun, no more than122 updates each.','Check finite weights/optimizer each update; telemetry available at every update; verify next-step checkpoint disk reserve.'],
        'implementation_preflight_required':['Full measurement producer must emit canonical quality-safety-v2; coverage34/34 and all fixtures must pass before any optimizer step.','Use a new PHASE59 runner; do not execute or patch PHASE57 training/gate commands.','Fresh live CUDA generation dry-run against both exact parents; negative rename/missing/type/NaN tests pass.','Validate tokenizer/train/population/schema/source SHA, strict reload and RNG roundtrip; verify current working-tree dependency hashes, including listed pre-existing dirty modules.'],
        'checkpoint_policy':{'root_env':'UNIPILOT_CHECKPOINT_ROOT','expected_root':str(checkpoint_root_value(pre)),'new_relative_root':'experimental/phase59/control-stability','atomic_exclusive_save':True,'SHA_strict_reload_resume_integrity':True,'required_markers':['EXPERIMENTAL','NOT_CANONICAL','NOT_PRODUCTION'],'existing_copy_move_delete_overwrite':[0,0,0,0]},
        'sealed_sets':pre['sealed'],'phase57_status':'EXPERIMENT_INVALID','old_outputs_rescored_for_efficacy':False,
        'source_sha256':{p:sha(ROOT/p) for p in sources},'artifact_sha256':{p:sha(ROOT/p) for p in artifact_sources},
        'canonical':False,'20m':False,'foundation_base':False,'production':False,
    }
    write(OUT/'phase59-control-stability-preregistration.json',value)
    write(OUT/'phase59-registration-receipt.json',{'schema_version':VERSION,'gate':'PHASE59_DESIGN_REGISTERED','preregistration_sha256':sha(OUT/'phase59-control-stability-preregistration.json'),'conditions':{'contract_ready':True,'forensic_complete':True,'control_drift_audit_complete':True,'dose_audit_complete':True,'parents_valid':True,'evaluation_artifacts_available':True,'sealed_set_safety':True},'training_authorized':False,'new_training':False})
    print('PHASE59 preregistration CREATED; training requires separate user authorization',flush=True)


def checkpoint_root_value(pre):return pre['root']
if __name__=='__main__':register()
