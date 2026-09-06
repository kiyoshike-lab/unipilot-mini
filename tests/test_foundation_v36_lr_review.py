import copy
import numpy as np
import pytest
import torch

from evaluation.evaluate_foundation_v36_lr_review import bucket_masks, probability_metrics, concentration, frequency_classification
from training.run_foundation_v36_lr_review import ARMS, checkpoint, fingerprint, official, set_lr_only


def test_rare_bucket_is_fixed_by_corpus_rank_not_predictions():
    ranks = np.arange(100)
    targets = np.array([0, 4, 20, 79, 80, 99, 80])
    mask = bucket_masks(targets, ranks)
    assert mask['rare_bottom_20_percent'].tolist() == [False,False,False,False,True,True,True]
    assert np.stack(list(mask.values())).sum(0).tolist() == [1]*7
    assert probability_metrics(np.array([.2,.5,.8]))['sample_count'] == int(mask['rare_bottom_20_percent'].sum())


def test_rare_percentile_and_occurrence_weighted_ce():
    result = probability_metrics([.1,.2,.3,.4])
    assert result['q25'] == pytest.approx(.175)
    assert result['q75'] == pytest.approx(.325)
    assert result['median'] == pytest.approx(.25)
    assert result['ce'] == pytest.approx(-np.log([.1,.2,.3,.4]).mean())
    assert probability_metrics([])['ce'] is None


def test_lr_arm_paths_cannot_collide_or_overwrite_formal():
    paths = {checkpoint(a,s) for a in ARMS for s in (42,123,2026)}
    assert len(paths) == 9
    assert all('experimental' in p.parts and 'phase47' in p.parts for p in paths)
    assert all(official(s) not in paths for s in (42,123,2026))
    with pytest.raises(ValueError): checkpoint('D',42)


def test_lr_change_preserves_adam_moments_steps_and_other_groups():
    parameter = torch.nn.Parameter(torch.tensor([1.,2.]))
    opt = torch.optim.AdamW([parameter],lr=1e-4,betas=(.9,.95))
    parameter.square().sum().backward(); opt.step()
    before = copy.deepcopy(opt.state_dict())
    assert set_lr_only(opt,7.5e-5) == fingerprint(before['state'])
    after = opt.state_dict()
    assert fingerprint(after['state']) == fingerprint(before['state'])
    for old,new in zip(before['param_groups'],after['param_groups']):
        assert new['lr'] == 7.5e-5
        assert {k:v for k,v in old.items() if k!='lr'} == {k:v for k,v in new.items() if k!='lr'}


def test_frequency_classification_separates_seed_local_and_broad():
    row = {'ce_delta':.2,'top10_positive_contribution_share':.2}
    assert frequency_classification([row]*3) == 'TRUE_RARE_REGRESSION'
    assert frequency_classification([row,row,{**row,'ce_delta':.02}]) == 'SEED_LOCAL_VARIANCE'
    assert frequency_classification([{**row,'top10_positive_contribution_share':.8}]*3) == 'OUTLIER_DRIVEN'
    assert frequency_classification([row],False) == 'EVALUATOR_ARTIFACT'


def test_concentration_rejects_changed_rare_population():
    a={'positions':[1,2], 'token_ids':[80,90], 'per_position_ce':[2.,4.]}
    with pytest.raises(AssertionError): concentration(a,{**a,'token_ids':[80,91]})
    result=concentration(a,{**a,'per_position_ce':[3.,5.]})
    assert result['ce_delta']==1.
    assert result['fraction_positions_worse']==1.


def test_candidate_lower_loss_cannot_override_rare_regression():
    from evaluation.analyze_foundation_v36_lr_review import checks, baseline
    base = baseline(42)
    candidate = copy.deepcopy(base)
    candidate['validation']['loss'] -= .1
    candidate['validation']['frequency_buckets']['rare_bottom_20_percent']['cross_entropy'] += .3
    train = {'integrity':{'pass':True},'source_unchanged':True,'telemetry':{'samples':1,'hardware_thermal_slowdown':False}}
    decision = checks(candidate,base,base,train)
    assert decision['checks']['validation']
    assert not decision['checks']['rare']
    assert not decision['pass']


def test_improved_rare_ce_does_not_hide_median_collapse():
    from evaluation.analyze_foundation_v36_lr_review import checks, baseline
    base = baseline(42)
    candidate = copy.deepcopy(base)
    rare = candidate['validation']['frequency_buckets']['rare_bottom_20_percent']
    rare['cross_entropy'] -= .1
    rare['median_correct_token_probability'] *= .77
    train = {'integrity':{'pass':True},'source_unchanged':True,'telemetry':{'samples':1,'hardware_thermal_slowdown':False}}
    assert not checks(candidate,base,base,train)['checks']['rare']
