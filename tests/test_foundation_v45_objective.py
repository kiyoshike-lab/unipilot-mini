import pytest
import torch
from evaluation.generation_prefix_objective_v45 import cycle_negatives,generated_prefix_unlikelihood
from evaluation.run_foundation_v45_observatory import normal_truth

@pytest.mark.parametrize('family',['math','code','lists','definitions','terminology'])
def test_normal_reference_repetition_has_zero_penalty(family):
    for n in (2,3,4,5):
        ids=list(map(ord,normal_truth(family,n)))
        assert cycle_negatives(ids,ids,set())==[]
        assert generated_prefix_unlikelihood(torch.zeros(2,3),[]).item()==0

def test_pathological_generated_cycle_penalized_and_gradient_direction():
    g=[1,2]*8;ref=list(range(20,36));neg=cycle_negatives(g,ref,{0,3})
    assert neg and all(t>=8 for t,_ in neg)
    logits=torch.zeros(16,40,requires_grad=True);loss=generated_prefix_unlikelihood(logits,neg)
    assert loss.item()>0;loss.backward()
    assert all(logits.grad[t,v]>0 for t,v in neg) # descent lowers cycle-token logits
    confident=logits.detach().clone()
    for t,v in neg:confident[t,v]=5
    assert generated_prefix_unlikelihood(confident,neg)>loss

def test_normal_long_math_or_code_repetition_vetoed_even_if_offset():
    # Valid requested repetition elsewhere in the reference must not be punished.
    for cycle in ([1],[1,2],[1,2,3,4]):
        g=cycle*6;ref=[9,8,7]+g+[6,5,4]
        assert cycle_negatives(g,ref,set())==[]

def test_no_penalty_for_specials_short_cycles_or_unscored_reference():
    assert not cycle_negatives([1,2]*4,list(range(20,28)),set())
    assert not cycle_negatives([0]*20,[1]*20,{0})
    assert not cycle_negatives([1]*20,[],set())
    assert not cycle_negatives([1,2,3,4,5,6],list(range(20,26)),set())
