import numpy as np
import pytest
from evaluation.evaluate_foundation_v37_stability import exposure_counts, exposure_summary, spearman, paired_exposure
from training.run_foundation_v37_stability import checkpoint, checkpoint256, reused
from training.run_foundation_v36_lr_review import checkpoint as old_checkpoint


def test_counts_are_targets_not_inputs_and_cumulative_windows_add():
    data=np.zeros(1537,dtype=np.uint16); data[512]=1; data[1024]=2; data[1536]=3
    order=np.array([2,0,1])
    a=exposure_counts(data,order,0,1,4); b=exposure_counts(data,order,1,2,4)
    both=exposure_counts(data,order,0,3,4)
    assert a[3]==1 and a[2]==0 and a.sum()==512
    assert np.array_equal(a+b,both) and both.sum()==1536


def test_exposure_buckets_partition_fixed_token_population():
    result=exposure_summary(np.array([0,1,2,4,5,9,10]),np.arange(7))
    assert result['rare_tokens_never_observed']==1 and result['rare_tokens_observed_once']==1
    assert result['observed_2_to_4']==2 and result['observed_5_to_9']==2 and result['observed_10_plus']==1
    assert result['unseen_percent']==pytest.approx(100/7)


def test_spearman_ties_monotonicity_and_constant_input():
    assert spearman([1,1,2,3],[5,5,4,1])==pytest.approx(-1)
    assert spearman([1,1,1],[1,2,3]) is None


def test_existing_four_experiments_are_reused_and_new_outputs_isolated():
    assert all(checkpoint256('C',s)==old_checkpoint('C',s) for s in (42,123,2026))
    assert checkpoint256('B',42)==old_checkpoint('B',42)
    assert not reused('B',123)
    assert checkpoint('C',42,512000)!=checkpoint256('C',42)
    assert 'phase48' in checkpoint('B',123,256000).parts
    with pytest.raises(ValueError): checkpoint('A',42,256000)


def test_paired_exposure_refuses_population_change():
    old={'token_ids':[1,2],'positions':[1,2],'per_position_ce':[2.,3.],'probabilities':[.1,.2]}
    with pytest.raises(AssertionError): paired_exposure(old,{**old,'token_ids':[1,3]},np.zeros(4),None)


def test_lr_winner_requires_paired_rare_and_lm_evidence():
    from evaluation.analyze_foundation_v37_stability import winner
    def row(loss, rare):
        return {'metrics':{k:{'mean':v,'by_seed':{str(s):v for s in (42,123,2026)}}
                           for k,v in {'loss':loss,'rare_ce':rare,'naturalness':.75,'semantic':.65}.items()},
                'safety':{str(s):{'checks':{'context':True,'rare':False}} for s in (42,123,2026)}}
    policy={'winner_lm_advantage':.01,'winner_rare_ce_advantage':.02,'comparison_sampling_tolerance':.08}
    assert winner({'B':row(4.4,9.8),'C':row(4.37,9.7)},policy)=='C'
    assert winner({'B':row(4.4,9.8),'C':row(4.37,9.81)},policy) is None
    assert winner({'B':row(4.4,9.8),'C':row(4.399,9.799)},policy) is None


def test_cumulative_comparison_uses_own_256k_control(monkeypatch):
    from evaluation import analyze_foundation_v37_stability as analyzer
    from pathlib import Path
    seen=[]
    monkeypatch.setattr(analyzer,'evaluation_path',lambda arm,seed,budget: Path(f'{arm}-{seed}-{budget}'))
    monkeypatch.setattr(Path,'exists',lambda self: True)
    monkeypatch.setattr(Path,'relative_to',lambda self,root: self)
    monkeypatch.setattr(analyzer,'read',lambda path: str(path))
    monkeypatch.setattr(analyzer,'aggregate',lambda rows: {})
    monkeypatch.setattr(analyzer.a36,'baseline',lambda seed: f'baseline-{seed}')
    monkeypatch.setattr(analyzer.a36,'checks',lambda row,base,control,train: seen.append((row,base,control)) or {'pass':True})
    analyzer.comparison(512000)
    assert len(seen)==6
    assert all(control==row.replace('512000','256000') and base.startswith('baseline-') for row,base,control in seen)
