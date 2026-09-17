"""Post-root-cause registration only. Never opens a model or starts training."""
from __future__ import annotations
import sys
import xml.etree.ElementTree as ET
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.diagnose_foundation_v40 import read,new_json,file_sha256


def main():
    out=ROOT/'evaluation/phase56';summary=read(out/'generation-observatory-summary.json');obs=read(out/'observability-preregistration.json')
    assert summary['root_cause_gate']=='MIXED_CAUSE_WITH_ACTIONABLE_TARGET' and all(summary['signals'].values())
    suites=ET.parse(out/'objective-unit-tests.xml').getroot().findall('testsuite')
    assert sum(int(s.attrib['tests']) for s in suites)==11
    assert all(int(s.attrib['errors'])==int(s.attrib['failures'])==int(s.attrib['skipped'])==0 for s in suites)
    parent=next(r for r in obs['checkpoints'] if r['arm']=='C' and r['seed']==42)
    def ref(path):return {'path':path,'sha256':file_sha256(ROOT/path)}
    spec={
      'phase':57,'registered_in_phase':56,'registered_at_utc':datetime.now(timezone.utc).isoformat(),
      'status':'PHASE57_TRAINING_READY','scope':'Design and synthetic objective contract ready; no execution in PHASE56. PHASE57 must implement and pass runtime preflight before any optimizer step.',
      'root_cause_gate':summary['root_cause_gate'],'root_evidence':ref('evaluation/phase56/generation-observatory-summary.json'),
      'parent_checkpoint':{'root_environment':'UNIPILOT_CHECKPOINT_ROOT','relative_path':'experimental/phase48/arm-C/seed-42/checkpoint-tokens-16384000.pt','sha256':parent['sha256'],'processed_tokens':16384000,'seed':42},
      'seed':42,'approved_lr':5e-5,'eos_weight':1.5,'device':'CUDA','precision':'FP32','tf32':False,'amp':False,'architecture_change':False,
      'continuity':'Strict reload parent model/optimizer/scheduler/permutation and Python/NumPy/PyTorch/CUDA RNG. Same next permutation entries, macro_batch512, optimizer hyperparameters and gradient clipping1.0. Constant approved LR; scheduler global_step follows updates. Parent tensors are read-only.',
      'arms':[
        {'id':'control','objective':'L_LM = mean(w(y)*CE(z,y)), w(EOS)=1.5 else1.0','coefficient':0.0,'old_repetition_auxiliary':False},
        {'id':'A','name':'generated-prefix contiguous-cycle unlikelihood with reference veto','coefficient':0.05,
         'objective':'L = L_LM + 0.05*L_GP on every8th update, otherwise L_LM. L_GP = mean_{(t,v) in M}[-log(1-min(p_theta(v|prefix,g_<t),1-1e-6))]; empty mask gives differentiable zero.',
         'mask':'Detached parent greedy continuation only; t>=4w with smallest widthw in1..8, previous4 contiguous cycles identical and nextgenerated token extends cycle. Exclude special tokens including EOS/BOS/PAD/UNK, actual aligned TRAIN target, and any repeated4w+1-token snippet occurring anywhere in the original32-token TRAIN continuation. Exact prototype determines mask.',
         'objective_prototype':ref('evaluation/generation_prefix_objective_v45.py'),
         'normal_protection':'No negative loss on teacher-forced LM stream; aligned reference target and reference-supported repetition veto. Frozen normal controls evaluate CE non-degradation, not a zero-accuracy improvement claim.',
         'phase42_difference':'PHASE42 searched3/4gram candidates in ground-truth teacher-forced histories on every update. A instead conditions on offline free-running parent prefixes, requires4 contiguous cycles and reference veto, and applies once per8 updates. Same mathematical UL family, materially different histories, mask and dose; never relabel the old auxiliary.'}
      ],'arm_B':None,
      'training_data':{
        'base':ref('data/foundation_v11/packed/vocab-4096/train.bin'),
        'rollout_selection':'First128 TRAIN documents ordered by SHA256(phase57-generated-prefix-v1|zero-based document index), with >=95 non-special text tokens and no special in first95. Prefix BOS+first63 text tokens (64 total); detached parent generates32 greedy tokens; reference is text[63:95]. No Diagnostic, validation, Confirmatory or Reserve tokens in gradients/cache.',
        'cache':'Generate once before optimizer steps from unchanged parent, eval(), same PHASE51 special mask and no decoder repetition controls. No EOS-reaching or <32-token episode eligible. Store local Z raw+ID/hash manifest; freeze cache SHA before either arm; identical cache order for both arms. No new cache selection by outcome.',
        'aux_batch':'Every8th update use next frozen cache episode, cycling deterministically. Forward prefix+generated first31 tokens predicts32 continuation choices; charge96 input positions conservatively even though actual length95. Auxiliary eval-mode/dropout OFF with gradients only for A; restore train mode. Control executes matching forward without auxiliary gradient. Main LM dropout/RNG stream must remain identical across arms before diverging weights.',
        'cache_insufficient':'If <20 eligible negative events across128 frozen episodes, STOP_NO_SAFE_OBJECTIVE before training. Never expand/redraw cache or change mask after this gate.'},
      'token_budget':{
        'first_gate_ceiling':64000,'maximum_ceiling':128000,
        'first_gate_updates':122,'first_gate_lm_tokens':62464,'first_gate_aux_positions':1440,'first_gate_total_gradient_input_ceiling':63904,
        'maximum_updates':244,'maximum_lm_tokens':124928,'maximum_aux_positions':2880,'maximum_total_gradient_input_ceiling':127808,
        'accounting':'512 LM tokens/update plus96 auxiliary positions at updates8,16,...; both arms charged identical slots. Below64k/128k ceilings, not an extra unreported auxiliary training budget. tokens_processed advances only LM tokens; auxiliary_positions and total_gradient_inputs stored separately. Frozen-cache inference is not training and is logged separately (<=128*96 positions). No256k+ run.',
        'extension':'At first gate, stop SUCCESS_SCREEN if all success+safety thresholds pass. Only if all safety pass and both runaway rates are<=.75 and at least one improves>=.10 absolute versus parent, permit continuation of the same control/A pair to maximum. Otherwise STOP_NO_EFFECT. Maximum gate must meet full success thresholds; no further budget.'},
      'evaluation_sets':{
        'generation':ref('evaluation/phase51/preregistration.json'),
        'generation_contract':'All100 frozen legacy prompts; greedy128 tokens; sampling temperature0.7 max64, no topk/topp/rep controls, RNG bases44000/51000/52000 plus prompt index. Same loop_details and budget-exhaustion runaway definition. Parent+control+A paired. No repeat PHASE55 Confirmatory.',
        'validation':ref('data/foundation_v11/packed/vocab-4096/validation.bin'),
        'lm_contract':'All validation next-token positions1..N-1, nonoverlapping512-input blocks (short final block included), no dropped targets; token-weighted CE and Top1/5/10. Same parent/control/A geometry.',
        'frequency_population':ref('evaluation/foundation-v39-supported-tail.json'),
        'frequency_contract':'Unchanged Core and disjoint Supported Tail membership/positions, same packed512 block geometry as evaluate_foundation_v39_gate; micro CE and macro-per-token CE, Top1/5/10 and paired document bootstrap10000 seed5701. Do not redefine old gates.',
        'eos_contract':'PHASE51 frozen terminal and nonterminal positions, context128; P(EOS), rank, Top1/5/10, nonterminal argmax EOS rate. No terminal probability alone establishes useful generation.',
        'context_contract':'Same PHASE56 selected24 Diagnostic IDs, consumed diagnostic only, never gradients. Entire each document, contexts128 and512 using existing document_metrics in confirm_foundation_v44.py; equal-document mean CE; relative long-context benefit compared with each comparator.',
        'normal_controls':ref('evaluation/phase55/normal-controls-preregistration.json'),
        'normal_contract':'All20 PHASE54-family/PHASE55-concretized controls; exact normal_truth from frozen PHASE56 runner, EOS appended, teacher-forced CE with identical prefix. Compare mean and each family CE plus terminal EOS. Also report output shape accuracy descriptively; PHASE55 floor0/60 forbids declaring non-degradation solely from accuracy.'},
      'success_thresholds':{'greedy_runaway_max':.50,'sampling_runaway_max':.50,'preferred_each_max':.25,'sampling_rule':'Pooled across3 fixed RNGs and each RNG individually <=.50; preferred each<=.25.',
         'attribution':'Both runaway reductions versus matched control >=.10 absolute, plus all safety thresholds versus BOTH unchanged parent and matched control. If control improves too without intervention separation, report CONTROL_IMPROVEMENT_NOT_INTERVENTION_EVIDENCE.',
         'interpretation':'Mechanistic seed42 screen only, repeatedly used diagnostic/validation sets, not fresh confirmatory evidence or model promotion. Multiseed123/2026 requires separate approval/registration if a candidate survives.'},
      'quality_safeguards_vs_each_parent_and_control':{
        'validation_ce_increase_max':.05,'validation_top1_drop_absolute_max':.01,'validation_top5_top10_drop_absolute_max':.02,
        'context_ce_increase_each_128_512_max':.05,'context_long_benefit_loss_max':.02,
        'terminal_mean_eos_probability_ratio_min':.90,'terminal_eos_top1_drop_absolute_max':.02,'nonterminal_eos_probability_increase_max':.001,'premature_argmax_eos_increase_max':.01,
        'core_micro_macro_ce_increase_max':.10,'supported_tail_micro_macro_ce_increase_max':.25,'frequency_ce_paired_ci95_upper_max':{'core':.10,'supported_tail':.25},
        'frequency_top1_drop_absolute_max':.005,'frequency_top5_top10_drop_absolute_max':.01,
        'normal_mean_ce_increase_max':.05,'normal_each_family_ce_increase_max':.10,'normal_terminal_eos_probability_ratio_min':.90,
        'sampling_naturalness_semantic_proxy_drop_max':.05,'sampling_topic_retention_drop_max':.05,'japanese_validity_drop_max':.01,
        'missing_metric':'FAIL_CLOSED; no averaging away a failed safeguard; zero denominator or undefined ratio requires review, not PASS'},
      'stop_conditions':[
        'Any parent/resolver/hash/state/permutation/RNG mismatch, sealed-set access, nonfinite loss/gradient/weights/optimizer or unknown partial artifact: stop before next update.',
        'GPU>=80C pause/cool<=65C; >=85C or hardware thermal slowdown abort. No parallel CPU evaluation, npm build or browser QA while GPU training/inference runs.',
        'Any quality safeguard failure at first gate: stop A; no coefficient/threshold/population edits and no extension. Control remains necessary matched evidence, never promotion.',
        'Raw main-gradient norm>10 for3 consecutive updates or >100 once: stop; existing clip1.0 remains. Auxiliary contribution>25% of LM loss for3 auxiliary updates: stop.',
        'Insufficient disk for2x estimated next checkpoint plus2GiB, output already exists, or budget ceiling reached: stop; never overwrite/copy/move/delete parent checkpoints.',
        'No safe normal-control outcome, incomplete evaluator or fewer than20 negative cache events: stop before training; do not declare readiness execution PASS from synthetic tests alone.'
      ],
      'unit_test_evidence':ref('evaluation/phase56/objective-unit-tests.xml'),
      'normal_safety_limit':'Synthetic tests establish fixture behavior and gradient sign, NOT universal semantic safety. Generated text may have legitimate repetition absent from the reference; bounded dose and strict teacher-forced safeguards are essential. No deployment claim.',
      'new_training_in_phase56':False,'canonical':False,'20m':False,'foundation_base':False
    }
    spec['source_hashes']={p:file_sha256(ROOT/p) for p in ['evaluation/register_foundation_v45_intervention.py','evaluation/generation_prefix_objective_v45.py','tests/test_foundation_v45_objective.py','training/foundation_v31_objective.py','evaluation/run_foundation_v45_observatory.py','evaluation/confirm_foundation_v44.py','evaluation/evaluate_foundation_v39_gate.py','evaluation/diagnose_foundation_v29_generation.py']}
    new_json(out/'phase57-training-preregistration.json',spec)
    new_json(out/'training-design-review.json',{'root_gate_precedes_design':True,'readiness':'PHASE57_TRAINING_READY','preregistration_sha256':file_sha256(out/'phase57-training-preregistration.json'),'interventions':1,'unit_tests':11,'new_training':False,'not_universal_safety_proof':True,'execution_requires_future_runtime_preflight':True})
    print('PHASE57_TRAINING_READY: control +1 intervention; no training',flush=True)


if __name__=='__main__':main()
