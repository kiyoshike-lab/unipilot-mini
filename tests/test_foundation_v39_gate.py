import numpy as np
import pytest
from evaluation.evaluate_foundation_v39_gate import supported_ids, support_ok
from evaluation.evaluate_foundation_v38_gate import cluster_ci


def test_supported_tail_is_disjoint_low_occurrence_and_independent():
    ranks=np.arange(4096); counts=np.full(4096,5); docs=np.full(4096,2)
    counts[3277]=10; counts[3278]=1; docs[3279]=1
    ids=supported_ids(ranks,counts,docs)
    assert np.array_equal(ids,np.arange(3280,4096))


def test_supported_selection_has_no_checkpoint_input():
    import inspect
    assert list(inspect.signature(supported_ids).parameters)==['ranks','counts','documents']


@pytest.mark.parametrize('docs,occ,expected',[(30,300,True),(29,300,False),(30,299,False),(141,1528,True)])
def test_formal_support_does_not_use_legacy(docs,occ,expected):
    assert support_ok({'document_count':docs,'occurrence_count':occ},{'minimum_documents':30,'minimum_occurrences':300}) is expected


def test_cluster_ci_uses_documents_not_positions_as_independent_units():
    assert cluster_ci([1,1,1],[0,0,0]) is None
    assert cluster_ci([1,1,1,3],[0,0,0,1])==[1.0,3.0]


def test_all_metric_intervals_have_exact_constant_population_bounds():
    from evaluation.evaluate_foundation_v39_gate import metric_intervals
    values={'ce':[2.,2.,2.],'probabilities':[np.exp(-2)]*3,'top1':[False]*3,'top5':[True]*3,'top10':[True]*3}
    ci=metric_intervals(values,{'token_ids':[1,1,2],'document_ids':[0,0,1]})
    assert ci['micro_ce']==ci['macro_per_token_ce']==[2.,2.]
    assert ci['top1']==[0.,0.] and ci['top5']==[1.,1.]


def test_phase50_reference_uses_env_and_preserves_noncheckpoint_paths(monkeypatch,tmp_path):
    from evaluation.evaluate_foundation_v39_gate import checkpoint_reference, evaluated_checkpoint, ROOT
    target=tmp_path/'experimental/phase49/arm-B/seed-123/checkpoint-tokens-16384000.pt'
    target.parent.mkdir(parents=True);target.write_bytes(b'fixture')
    monkeypatch.setenv('UNIPILOT_CHECKPOINT_ROOT',str(tmp_path))
    assert evaluated_checkpoint('B',123)==target
    assert checkpoint_reference('evaluation/result.json')==ROOT/'evaluation/result.json'
    with pytest.raises(ValueError): checkpoint_reference('checkpoints/../outside.pt')
