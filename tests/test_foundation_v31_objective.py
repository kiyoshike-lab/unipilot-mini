import torch
from training.foundation_v31_objective import repetition_negative_candidates, unlikelihood_loss, weighted_lm_loss
from training import run_foundation_v31_pipeline as pipeline

def test_negative_closes_repeated_trigram_and_excludes_truth():
    x=torch.tensor([[1,2,3,1,2]]); y=torch.tensor([[2,3,1,2,9]])
    rows=repetition_negative_candidates(x,y,(3,))
    assert (0,4,3) in rows
    assert all(candidate != int(y[b,p]) for b,p,candidate in rows)

def test_unlikelihood_is_finite_and_zero_without_negatives():
    z=torch.randn(1,5,10)
    assert torch.isfinite(unlikelihood_loss(z,[(0,4,3)]))
    assert unlikelihood_loss(z,[]).item()==0

def test_eos_weighting_leaves_non_eos_contribution_unchanged():
    z=torch.randn(1,4,8); y=torch.tensor([[1,2,3,4]])
    _,_,a=weighted_lm_loss(z,y,2,1.0); _,_,b=weighted_lm_loss(z,y,2,1.5)
    assert torch.equal(a,b)

def test_repeated_fourgram_is_detected():
    x=torch.tensor([[1,2,3,4,1,2,3]]); y=torch.tensor([[2,3,4,1,2,3,9]])
    assert (0,6,4) in repetition_negative_candidates(x,y,(4,))

def test_true_japanese_function_word_target_is_never_negative():
    x=torch.tensor([[7,8,9,7,8]]); y=torch.tensor([[8,9,7,8,9]])
    assert all(candidate != int(y[b,p]) for b,p,candidate in repetition_negative_candidates(x,y))

def test_eos_and_repetition_losses_coexist_finitely():
    z=torch.randn(1,6,16,requires_grad=True); y=torch.tensor([[1,2,3,1,2,4]])
    ce,_,_=weighted_lm_loss(z,y,4,1.5); auxiliary=unlikelihood_loss(z,[(0,4,3)])
    total=ce+.03*auxiliary; total.backward()
    assert torch.isfinite(total) and torch.isfinite(z.grad).all()

def test_worker_output_directories_do_not_collide():
    assert pipeline.OUT != pipeline.EVAL != pipeline.LOG
    assert 'experimental' in pipeline.OUT.parts and 'cpu-worker' in pipeline.EVAL.parts

def test_ready_protocol_rejects_missing_marker(tmp_path,monkeypatch):
    monkeypatch.setattr(pipeline,'OUT',tmp_path)
    try: pipeline.cpu_worker('A')
    except RuntimeError as error: assert 'READY' in str(error)
    else: raise AssertionError('missing READY marker accepted')

def test_arm_definitions_are_conservative_and_separable():
    assert pipeline.ARMS == {'A':(1.0,0.0),'B':(1.5,0.0),'C':(1.5,0.01),'D':(1.5,0.03),'E':(1.5,0.05)}
    assert max(value[1] for value in pipeline.ARMS.values()) < .1
