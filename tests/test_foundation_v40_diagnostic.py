import ast
from pathlib import Path
import numpy as np
from evaluation.diagnose_foundation_v40 import cluster_ci, new_json, read, PREREG


def test_cluster_bootstrap_pairs_documents_not_occurrences():
    delta=np.array([1.,1.,-1.]);groups=np.array([0,0,1])
    assert cluster_ci(delta,groups,51,1000)==cluster_ci(delta,groups,51,1000)
    assert cluster_ci(delta,groups,51,1000)['mean']==1/3
    zero=cluster_ci(np.zeros(3),groups,51,1000)
    assert zero=={'mean':0.,'lower':0.,'upper':0.}


def test_artifact_writer_never_overwrites(tmp_path):
    import pytest
    path=tmp_path/'record.json';new_json(path,{'complete':True})
    with pytest.raises(FileExistsError):new_json(path,{'complete':False})
    assert read(path)['complete']


def test_preregistered_population_and_training_prohibition():
    spec=read(PREREG)
    assert spec['new_training'] is False and spec['cpu_heavy_parallel_evaluation'] is False
    assert spec['device']=='cuda' and spec['dtype']=='float32'
    assert len(spec['prompts'])==100 and len(spec['terminal_positions'])==len(set(spec['terminal_positions']))==146
    assert len(set(spec['nonterminal_positions']))==500
    assert not set(spec['terminal_positions'])&set(spec['nonterminal_positions'])
    assert spec['sampling_decoder']['temperature']==.7 and spec['sampling_decoder']['top_k'] is None and spec['sampling_decoder']['top_p'] is None
    assert spec['sampling_max_new_tokens']==64 and spec['sampling_rng_bases']==[44000,51000,52000]
    tree=ast.parse(Path('evaluation/diagnose_foundation_v40.py').read_text(encoding='utf-8'))
    calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    assert not {'backward','step','save','unlink','rename','replace','copy','move'} & set(calls)
