from pathlib import Path
from evaluation.diagnose_foundation_v40 import read

def test_track_c_isolated_and_frozen_grid_has_no_safe_candidate():
    r=read(Path('evaluation/phase53/generation-policy-results.json'))
    assert r['fresh_data_used'] is False and r['future_reserve_opened_or_scored'] is False
    assert r['prompt_set']['count']==24 and len(r['raw_artifacts'])==20
    assert r['best_safe_decoder_candidate']=='NONE'
    assert r['attractor_gate']=='GENERATION_POLICY_UNSAFE'
    assert all(not v['passes_all'] for v in r['quality_screen'].values())

def test_greedy_reproduces_and_stop_replay_is_not_eos_safe():
    r=read(Path('evaluation/phase53/generation-policy-results.json'))
    assert r['results']['C:greedy']['metrics']['runaway']==1.0
    assert r['results']['B:greedy']['metrics']['runaway']==1.0
    for stop in r['loop_stop_replay'].values():
        assert stop['triggered']>0 and stop['triggered_eos_completion']==0
        assert stop['false_positive_evidence'].startswith('No labelled normal-repeat')

def test_track_c_does_not_change_lr_or_training_permissions():
    s=read(Path('evaluation/foundation-v42-generation-policy-summary.json'))
    assert s['formal_lr']=='FORMAL_LR_UNRESOLVED' and not s['model_fixed']
    assert not s['new_training'] and not s['canonical_training'] and not s['20m_permission']
