import numpy as np
import pytest
from evaluation.evaluate_foundation_v38_gate import core_ids,cluster_ci,paired
from evaluation.analyze_foundation_v38_gate import ci_safe,support_ok
from training.run_foundation_v38_gate import checkpoint

def test_core_membership_only_uses_frequency_and_support():
    assert core_ids(np.arange(10),[100]*8+[9,10]).tolist()==[9]

def test_document_bootstrap_keeps_clusters_together():
    assert cluster_ci([0,0,2,2],[0,0,1,1])==pytest.approx([0,2])
    assert cluster_ci([1,1],[0,0]) is None

def test_ci_is_paired_not_difference_of_independent_intervals():
    pop={'sha256':'x','document_ids':[0,0,1,1]}
    before={'population_sha256':'x','values':{'ce':[1,2,9,10],'probabilities':[.2]*4}}
    after={'population_sha256':'x','values':{'ce':[1.1,2.1,9.1,10.1],'probabilities':[.3]*4}}
    assert paired(before,after,pop)['ce_delta_ci95']==pytest.approx([.1,.1])
    with pytest.raises(AssertionError): paired(before,{**after,'population_sha256':'y'},pop)

def test_wide_ci_containing_zero_cannot_approve():
    assert not ci_safe({'ce_delta_ci95':[-1,1]},.25)
    assert ci_safe({'ce_delta_ci95':[-.1,.05]},.1)

def test_two_tail_documents_are_insufficient_for_approval():
    policy={'minimum_core_types':100,'minimum_core_occurrences':1000,'minimum_documents':10}
    assert not support_ok({'core':{'token_count':236,'occurrence_count':4436,'document_count':146},'tail':{'document_count':2}},policy)

def test_candidate_lineage_is_isolated_and_token_capped():
    assert checkpoint('B',42)!=checkpoint('B',42,True)
    assert checkpoint('B',42,True).name=='checkpoint-tokens-16128000.pt'
    with pytest.raises(ValueError): checkpoint('A',42)
