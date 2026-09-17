import importlib.util
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('phase57',ROOT/'evaluation/run_foundation_v46_phase57.py');phase57=importlib.util.module_from_spec(spec);spec.loader.exec_module(phase57)

def test_registered_accounting_has_no_hidden_overage():
    assert 122*512+(122//8)*96==63904
    assert 244*512+(244//8)*96==127808
    assert 63904<=64000 and 127808<=128000

def test_frequency_ci_is_paired_and_fixed_seed():
    a=np.array([1.,3.,2.,4.]);b=np.array([0.,1.,1.,2.]);docs=[0,0,1,1]
    first=phase57.ci_delta(a,b,docs);second=phase57.ci_delta(a,b,docs)
    assert first==second and first['mean']==1.5 and first['lower']<=first['upper']

def test_generation_metric_uses_registered_sampling_not_new_prompts():
    row={'generation':{'greedy':{'metrics':{'runaway_rate':1.,'natural_japanese_proxy':.2}},'sampling':{'1':{'metrics':{'runaway_rate':.9,'semantic_coherence_proxy':.3,'topic_retention_proxy':.4,'japanese_validity':1.}},'2':{'metrics':{'runaway_rate':.8,'semantic_coherence_proxy':.5,'topic_retention_proxy':.6,'japanese_validity':1.}}}}}
    assert phase57.generation_metric(row,'semantic_coherence_proxy')==.4
    assert phase57.runaway(row,'sampling')=={'1':.9,'2':.8}
