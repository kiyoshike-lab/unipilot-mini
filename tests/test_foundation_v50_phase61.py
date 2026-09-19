"""Static PHASE61 bounds; CUDA execution is covered by the live receipt."""
from evaluation import run_foundation_v50_phase61 as p
from evaluation import generation_contract_v47 as c

def test_only_registered_arms_seeds_and_budget_are_reachable():
    assert p.SEEDS == (42,123,2026)
    assert p.ARMS == (('control',5e-5),('half-lr',2.5e-5))
    assert "choices=('preflight','dry-run','train','evaluate')" in p.Path(p.__file__).read_text(encoding='utf8')

def test_registered_contract_remains_34_checks():
    spec=p.read(p.SPEC)
    assert p.sha(p.SPEC) == p.SPEC_SHA
    assert len(c.mapping(spec['evaluation']['all_thresholds'])) == 34

def test_phase61_decision_is_fail_closed_for_incomplete_runs():
    values={str(seed):{'valid':True,'control_all34_vs_parent':True,'half_lr_all34_vs_parent':True,'half_lr_all34_vs_control':True} for seed in p.SEEDS}
    assert p.phase61_decision(values)=='BOTH_STABLE_NO_DIFFERENTIAL_EVIDENCE'
    values['42']['valid']=False
    assert p.phase61_decision(values)=='EXPERIMENT_INVALID'
