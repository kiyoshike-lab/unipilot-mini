from copy import deepcopy
import numpy as np
import pytest
from evaluation.register_foundation_v47_control import decision
from evaluation.audit_foundation_v47 import base_candidates
from evaluation.generation_prefix_objective_v45 import cycle_negatives

def row(stable=True,micro=True,macro=True):
    return {'valid':True,'all_safety_pass':stable,'core_micro_pass':micro,'core_macro_pass':macro}

@pytest.mark.parametrize('a,b,expected',[
    (row(),row(),'CONTROL_STABLE_MULTISEED'),
    (row(False,False),row(False,True,False),'CONTROL_DRIFT_REPLICATED'),
    (row(),row(False,False),'CONTROL_STABILITY_MIXED'),
    (row(False),row(False),'CONTROL_STABILITY_MIXED'),
    (row(True,False),row(),'EXPERIMENT_INVALID'),
])
def test_registration_decision_exhausts_valid_cases(a,b,expected):
    assert decision({'123':a,'2026':b})==expected

def test_incomplete_or_invalid_seed_is_never_averaged_away():
    assert decision({'123':row()})=='EXPERIMENT_INVALID'
    bad=row();bad['valid']=False
    assert decision({'123':row(),'2026':bad})=='EXPERIMENT_INVALID'
    bad=row();bad['all_safety_pass']=1
    assert decision({'123':row(),'2026':bad})=='EXPERIMENT_INVALID'

def test_veto_denominator_is_not_an_alternative_training_mask():
    generated=[4]*32;reference=[4]*32
    before=base_candidates(generated,reference,{0,1,2})
    assert len(before)==28 and cycle_negatives(generated,reference,{0,1,2})==[]
    assert len(cycle_negatives(generated,[5]*32,{0,1,2}))==len(before)
    assert not base_candidates([2]*32,reference,{0,1,2})

def test_consumed_cache_cursor_and_budget_match_historical_code():
    updates=[u for u in range(32001,32123) if u%8==0]
    assert len(updates)==15
    assert [(u//8-1)%128 for u in updates]==list(range(32,47))
    assert 122*512+len(updates)*96==63904
