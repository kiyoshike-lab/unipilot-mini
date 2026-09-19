from copy import deepcopy
from evaluation.register_foundation_v49 import phase61_decision


def stable():
    return {str(s):{'valid':True,'control_all34_vs_parent':True,
                   'half_lr_all34_vs_parent':True,'half_lr_all34_vs_control':True}
            for s in (42,123,2026)}


def test_both_stable_does_not_claim_differential_stabilization():
    assert phase61_decision(stable())=='BOTH_STABLE_NO_DIFFERENTIAL_EVIDENCE'


def test_intervention_must_pass_every_seed_and_comparator():
    for seed in ('42','123','2026'):
        for field in ('half_lr_all34_vs_parent','half_lr_all34_vs_control'):
            results=stable();results[seed][field]=False
            assert phase61_decision(results)=='CONTINUATION_STABILIZATION_NOT_ESTABLISHED'


def test_control_safety_failure_is_valid_but_incomplete_run_is_not():
    results=stable();results['42']['control_all34_vs_parent']=False
    assert phase61_decision(results)=='CONTINUATION_STABILIZATION_SUPPORTED'
    results['42']['valid']=False
    assert phase61_decision(results)=='EXPERIMENT_INVALID'
    del results['42']
    assert phase61_decision(results)=='EXPERIMENT_INVALID'


def test_wrong_types_do_not_silently_pass():
    results=stable();results['123']['valid']=1
    assert phase61_decision(results)=='EXPERIMENT_INVALID'
