"""Read-only hooks. No architecture edits, gradients, parameter writes or training."""
from __future__ import annotations
import numpy as np
import torch
from evaluation.diagnose_foundation_v29_generation import loop_details,ngram_repetition,SPECIAL

def cosine(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return float(np.dot(a,b)/max(1e-12,float(np.linalg.norm(a)*np.linalg.norm(b))))

class Observer:
    def __init__(self,model):
        self.hidden={};self.attention={};self.handles=[]
        for i,block in enumerate(model.blocks):
            def hidden_hook(module,args,output,index=i):self.hidden[index]=output[0][0,-1].detach().float().cpu().numpy().copy()
            self.handles.append(block.register_forward_hook(hidden_hook))
            if hasattr(block.attention,'attention_dropout'):
                def attention_hook(module,args,output,index=i):self.attention[index]=output[0,:,-1,:].detach().float().mean(0).cpu().numpy().copy()
                self.handles.append(block.attention.attention_dropout.register_forward_hook(attention_hook))
    def clear(self):self.hidden.clear();self.attention.clear()
    def close(self):
        for handle in self.handles:handle.remove()
        self.handles=[]

def repeat_onsets(ids):
    result={}
    for n in (3,4,5,6,8):
        seen=set();first=None
        for i in range(len(ids)-n+1):
            g=tuple(ids[i:i+n])
            if g in seen:first=i+n;break
            seen.add(g)
        result[str(n)]=first
    return result

@torch.inference_mode()
def trajectory(model,tok,prefix,maximum=128,truth=None,replay=None,replace_step=None,replacement=None,reference_loop=None):
    """Predict then feed selected token; 1-based prediction step, raw-state index is its input position.

    At a perturbation step, replay original earlier choices, replace one choice,
    then continue greedy. No unobservable counterfactual padding/position edits.
    """
    device=next(model.parameters()).device;observer=Observer(model);past=None
    current=torch.tensor([prefix],device=device);ids=[];rows=[];hidden=[];logits_all=[];attention=[]
    forbidden=[tok.special_to_id[name] for name in SPECIAL if name!='<EOS>']
    try:
        for index in range(maximum if truth is None else min(maximum,len(truth))):
            observer.clear();out,_,past=model(current,past_key_values=past,use_cache=True);scores=out[0,-1].float()
            probabilities=scores.softmax(-1);top=probabilities.topk(2);allowed=scores.clone();allowed[forbidden]=-torch.inf
            greedy=int(allowed.argmax());selected=greedy
            if truth is not None:selected=int(truth[index])
            elif replace_step is not None and index+1<replace_step:selected=int(replay[index])
            elif replace_step==index+1:
                selected=int(allowed.topk(2).indices[1]) if replacement=='top2' else int(replacement)
            h=np.stack([observer.hidden[i] for i in range(len(model.blocks))]);hidden.append(h)
            att=[observer.attention.get(i,np.array([],dtype=np.float32)) for i in range(len(model.blocks))];attention.append(att)
            raw=scores.cpu().numpy().copy();logits_all.append(raw)
            rows.append({'step':index+1,'position':len(prefix)+index-1,'token_id':selected,'token_text':tok.decode([selected],skip_special=False),
                'eos_probability':float(probabilities[tok.eos_id]),'eos_rank':int((scores>scores[tok.eos_id]).sum())+1,
                'eos_logit':float(scores[tok.eos_id]),'top1_minus_eos_logit':float(scores.max()-scores[tok.eos_id]),
                'top1_probability':float(top.values[0]),'top2_probability':float(top.values[1]),'margin':float(top.values[0]-top.values[1]),
                'entropy':float(-(probabilities*probabilities.clamp_min(1e-30).log()).sum()),'selected_probability':float(probabilities[selected]),
                'selected_nll':float(-probabilities[selected].clamp_min(1e-30).log()),'logit_norm':float(scores.norm()),'hidden_norm':float(np.linalg.norm(h[-1])),
                'forced':selected!=greedy})
            ids.append(selected);current=torch.tensor([[selected]],device=device)
            if selected==tok.eos_id:break
    finally:observer.close()
    loop=loop_details(ids);alignment=reference_loop or loop;cycle=alignment['loop_length'];onset=alignment['loop_onset'] or 65
    for j,row in enumerate(rows):
        layers=[]
        for k,h in enumerate(hidden[j]):
            att=attention[j][k];width=cycle or 1
            layers.append({'layer':k,'l2_norm':float(np.linalg.norm(h)),
                'previous_cosine':cosine(h,hidden[j-1][k]) if j else None,
                'cycle_cosine':cosine(h,hidden[j-cycle][k]) if cycle and j>=cycle else None,
                'attention_recent_1_4':{str(n):float(att[-n:].sum()) for n in (1,2,3,4)} if len(att) else None,
                'attention_previous_cycle':float(att[-2*width:-width].sum()) if len(att)>=2*width else None,
                'attention_earlier_context':float(att[:-max(4,2*width)].sum()) if len(att)>max(4,2*width) else 0.,
                'attention_bos':float(att[0]) if len(att) else None,
                'attention_expected_position':float(np.dot(att,np.arange(len(att)))) if len(att) else None})
        row['layers']=layers
        row['logit_previous_cosine']=cosine(logits_all[j],logits_all[j-1]) if j else None
        row['logit_cycle_centered_cosine']=cosine(logits_all[j]-logits_all[j].mean(),logits_all[j-cycle]-logits_all[j-cycle].mean()) if cycle and j>=cycle else None
    chosen=sorted({p for p in [0,31,63,95,127,*range(onset-5,onset+4)] if 0<=p<len(rows)})
    vectors={'steps':np.array([i+1 for i in chosen]),'hidden':np.stack([hidden[i] for i in chosen]),'logits':np.stack([logits_all[i] for i in chosen])}
    result={'ids':ids,'rows':rows,'loop':loop,'alignment_loop':alignment,'ngram_repeat_onsets':repeat_onsets(ids),
        'repetition':{str(n):ngram_repetition(ids,n) for n in (3,4,5,6,8)},'eos_reached':ids[-1:]==[tok.eos_id],
        'runaway':truth is None and len(ids)==maximum and ids[-1:]!=[tok.eos_id],
        'teacher_forced':truth is not None,'replace_step':replace_step,'replacement':replacement,'ce':float(np.mean([r['selected_nll'] for r in rows])),
        'hook_layers':len(model.blocks),'attention_collection':'PASS' if all(len(a) for step in attention for a in step) else 'ATTENTION_NOT_OBSERVABLE',
        'prediction_state_convention':'layer hidden after block residual at last INPUT token predicts selected NEXT token; never mislabeled as selected-token embedding'}
    return result,vectors
