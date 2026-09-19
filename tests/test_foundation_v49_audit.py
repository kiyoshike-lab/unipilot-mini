"""Numerical edge cases for observational audit calculations."""
import numpy as np
import pytest
from evaluation import audit_foundation_v49 as a


def test_tied_ranks_and_undefined_correlation():
    assert a.average_ranks([5,2,2,9]).tolist() == [2, .5, .5, 3]
    assert a.spearman([1,1,1],[2,3,4]) is None
    assert a.spearman([1,1,2,3],[3,3,2,1]) == pytest.approx(-1)


def test_empty_group_not_imputed_and_nonfinite_rejected():
    assert a.dist([])['mean'] is None
    with pytest.raises(ValueError): a.dist([1,float('nan')])


def test_paired_document_bootstrap_respects_cluster_counts():
    value = a.bootstrap([.2,.2,.2,.2],[1,1,2,3])
    assert value['mean'] == pytest.approx(.2)
    assert value['lower'] == pytest.approx(.2)
    assert value['upper'] == pytest.approx(.2)


def test_borderline_uses_exact_registered_threshold_only():
    assert a.comparison_status(.1,.1,True) == 'BORDERLINE'
    assert a.comparison_status(.10000001,.1,False) == 'FAIL'
    assert a.comparison_status(.09999999,.1,True) == 'PASS'


@pytest.mark.parametrize('name,expected',[
    ('embeddings.token.weight','embedding_tied_head'),
    ('embeddings.position.weight','position_embedding'),
    ('blocks.0.attention.query.weight','attention'),
    ('blocks.2.feed_forward.network.0.weight','FFN'),
    ('blocks.2.norm1.weight','LayerNorm'),
])
def test_parameter_groups(name,expected):
    assert a.category(name) == expected
