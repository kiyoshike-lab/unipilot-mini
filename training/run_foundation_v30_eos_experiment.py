"""PHASE 41 experimental-only EOS loss weighting continuation."""
from __future__ import annotations
import argparse, json, math, random
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from foundation.base_tokenizer import FoundationTokenizer
from foundation.diagnostic_transformer_v17 import DiagnosticConfigV17, DiagnosticTransformerV17
from training.optimizer import create_optimizer
from training.train_foundation_v15_controlled import macro_batch
from training.train_foundation_v21_ab import file_sha256, move_optimizer_state_to_device, random_state, restore_random_state
from evaluation.diagnose_foundation_v29_generation import (build_prefixes, document_ranges, generate_batch, target_metrics, summarize_generation, loop_details)

TOKENS_PER_UPDATE=512
BASE=ROOT/'checkpoints/foundation-v28-current/current'
OUT=ROOT/'checkpoints/foundation-v30-eos-experimental'

def load(path, device):
    p=torch.load(path,map_location='cpu',weights_only=False)
    m=DiagnosticTransformerV17(DiagnosticConfigV17(**p['config'])).to(device); m.load_state_dict(p['model_state']); m.eval()
    o=create_optimizer(m,1e-4,0.1); o.load_state_dict(p['optimizer_state']); move_optimizer_state_to_device(o,device)
    restore_random_state(p['random_state'],device,cuda_seed=int(p['seed']))
    return p,m,o

@torch.inference_mode()
def lm(model, val, device):
    n=8192; loss=0.; correct=[0,0,0]
    for s in range(0,n,512):
        a=np.asarray(val[s:s+513],dtype=np.int64); x=torch.tensor(a[:-1],device=device)[None]; y=torch.tensor(a[1:],device=device)[None]
        z,_=model(x,y); loss+=float(F.cross_entropy(z.flatten(0,1),y.flatten()))*512; top=z.topk(10,-1).indices
        correct=[correct[i]+int((top[...,:k]==y[...,None]).any(-1).sum()) for i,k in enumerate((1,5,10))]
    return {'loss':loss/n,'ppl':math.exp(loss/n),'top1':correct[0]/n,'top5':correct[1]/n,'top10':correct[2]/n}

def evaluate(model,tok,val,train,device):
    unique=np.asarray([int(p) for p in np.flatnonzero(val==tok.eos_id) if p>=128]); terminal=np.resize(unique,500)
    non=np.asarray([p for p in range(128,len(val)-1, max(1,(len(val)-129)//500)) if val[p] not in (tok.eos_id,tok.bos_id)])[:500]
    ranges=document_ranges(val,tok.bos_id,tok.eos_id); prefixes=build_prefixes(val,ranges,tok)
    greedy=generate_batch(model,tok,[p['prefix_ids'] for p in prefixes],{'name':'greedy','kind':'greedy','temperature':1.,'top_k':None,'top_p':None,'repetition_penalty':1.,'no_repeat_ngram':None,'eos_threshold':None},list(range(100)),128,False)
    sampling=generate_batch(model,tok,[p['prefix_ids'] for p in prefixes],{'name':'t07','kind':'sampling','temperature':.7,'top_k':None,'top_p':None,'repetition_penalty':1.,'no_repeat_ngram':None,'eos_threshold':None},list(range(100)),64,False)
    eos_terminal=target_metrics(model,val,terminal,tok.eos_id); eos_nonterminal=target_metrics(model,val,non,tok.eos_id)
    on=[r['loop']['loop_onset'] for r in greedy if r['loop']['loop_onset']]
    return {'lm':lm(model,val,device),'terminal_eos':eos_terminal,'nonterminal_eos':eos_nonterminal,'greedy':{**summarize_generation(greedy),'median_loop_onset':float(np.median(on)) if on else None},'sampling_t07':summarize_generation(sampling)}

def run(weight,seed,budget,device):
    source=BASE/f'seed-{seed}/checkpoint-tokens-15360000.pt'; before=file_sha256(source)
    p,m,opt=load(source,device); start=int(p['update']); train=np.memmap(ROOT/'data/foundation_v11/packed/vocab-4096/train.bin',dtype=np.uint16,mode='r'); val=np.memmap(ROOT/'data/foundation_v11/packed/vocab-4096/validation.bin',dtype=np.uint16,mode='r'); tok=FoundationTokenizer.load(ROOT/'tokenizer/foundation-v11-base-4096.json')
    norms=[]; eos_losses=[]; non_losses=[]; m.train()
    for update in range(start+1,start+budget//512+1):
        x,y=macro_batch(train,int(p['permutation'][update-1]),512); x=x.to(device); y=y.to(device); opt.zero_grad(True); z,_=m(x); per=F.cross_entropy(z.flatten(0,1),y.flatten(),reduction='none').view_as(y); mask=y==tok.eos_id; loss=(per*torch.where(mask,torch.tensor(weight,device=device),torch.tensor(1.,device=device))).mean(); loss.backward(); norms.append(float(torch.nn.utils.clip_grad_norm_(m.parameters(),1.0))); eos_losses.append(float(per[mask].mean()) if mask.any() else 0.); non_losses.append(float(per[~mask].mean())); opt.step()
    m.eval(); result={'arm':'control' if weight==1 else f'eos_weight_{weight:g}','weight':weight,'seed':seed,'budget_tokens':budget,'start_tokens':15360000,'end_tokens':15360000+budget,'gradient':{'mean_norm':float(np.mean(norms)),'max_norm':float(np.max(norms)),'eos_loss':float(np.mean(eos_losses)),'non_eos_loss':float(np.mean(non_losses))},'evaluation':evaluate(m,tok,val,train,device)}
    target=OUT/result['arm']/f'seed-{seed}'/f'checkpoint-tokens-{result["end_tokens"]}.pt'; target.parent.mkdir(parents=True,exist_ok=True)
    payload={**p,'model_state':m.state_dict(),'optimizer_state':opt.state_dict(),'update':start+budget//512,'tokens_processed':result['end_tokens'],'random_state':random_state(device),'experimental':True,'phase':41,'eos_loss_weight':weight,'source_checkpoint':str(source),'source_sha256':before}
    torch.save(payload,target); loaded=torch.load(target,map_location='cpu',weights_only=False); verify=DiagnosticTransformerV17(DiagnosticConfigV17(**loaded['config'])); verify.load_state_dict(loaded['model_state'],strict=True); result['checkpoint']={'path':str(target.relative_to(ROOT)),'sha256':file_sha256(target),'strict_reload':True,'source_unchanged':file_sha256(source)==before}; return result

def main():
 a=argparse.ArgumentParser(); a.add_argument('--budget',type=int,default=256000); a.add_argument('--seed',type=int,default=42); a.add_argument('--weights',nargs='+',type=float,default=[1,1.25,1.5,2]); args=a.parse_args(); d=torch.device('cuda'); rows=[run(w,args.seed,args.budget,d) for w in args.weights]; (ROOT/'evaluation/foundation-v30-eos-arms.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8'); print(json.dumps(rows,ensure_ascii=False))
if __name__=='__main__': main()
