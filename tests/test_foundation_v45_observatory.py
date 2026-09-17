import torch
from foundation.diagnostic_transformer_v17 import DiagnosticConfigV17,DiagnosticTransformerV17
from evaluation.generation_observatory_v45 import Observer,cosine,repeat_onsets
from evaluation.run_foundation_v45_observatory import normal_truth

def test_observer_does_not_change_logits_or_state():
    torch.manual_seed(56);m=DiagnosticTransformerV17(DiagnosticConfigV17('tiny',32,16,8,2,2,16,dropout=0)).eval()
    x=torch.tensor([[1,3,4]]);before={k:v.clone() for k,v in m.state_dict().items()};plain=m(x)[0]
    tap=Observer(m);out=m(x)[0];assert torch.equal(plain,out);assert len(tap.hidden)==len(tap.attention)==2
    assert all(abs(float(a.sum())-1)<1e-6 for a in tap.attention.values());tap.close()
    assert all(torch.equal(before[k],v) for k,v in m.state_dict().items())

def test_cosine_and_repeat_onsets():
    assert abs(cosine([1,0],[1,0])-1)<1e-8
    assert repeat_onsets([1,2,3,1,2,3])['3']==6
    assert repeat_onsets([1,2,3])['3'] is None

def test_normal_references_preserve_intended_repetitions():
    assert normal_truth('math',3)=='1+1+1=3。'
    assert normal_truth('code',4).count('print')==4
    assert normal_truth('lists',5).count('確認済み')==5
    assert normal_truth('definitions',3).count('定義:')==3
    assert normal_truth('terminology',2).count('標本平均')==2
