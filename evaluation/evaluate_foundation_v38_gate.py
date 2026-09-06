"""Checkpoint-independent Core/Tail populations and document-cluster uncertainty."""
from __future__ import annotations
import argparse
import math
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from training.run_foundation_v38_gate import EVAL,SEEDS,read,write,checkpoint,evaluation_path
from training import run_foundation_v36_lr_review as v36
from training import run_foundation_v37_stability as v37
from evaluation import evaluate_foundation_v35_short_gate as ev35
from evaluation import evaluate_foundation_v36_lr_review as ev36
from foundation.base_tokenizer import FoundationTokenizer
from training.train_foundation_v21_ab import file_sha256,frequency_ranks

RARE='rare_bottom_20_percent'


def core_ids(ranks,validation_counts):
    return np.flatnonzero((np.asarray(ranks)>=math.ceil(len(ranks)*.8))&(np.asarray(validation_counts)>=10))


def freeze():
    path=EVAL/'frequency-population.json'
    if path.exists(): raise FileExistsError('population is immutable after definition')
    tok=FoundationTokenizer.load(ROOT/'tokenizer/foundation-v11-base-4096.json')
    train=np.memmap(ROOT/'data/foundation_v11/packed/vocab-4096/train.bin',dtype=np.uint16,mode='r')
    val=np.memmap(ROOT/'data/foundation_v11/packed/vocab-4096/validation.bin',dtype=np.uint16,mode='r')
    ranks=frequency_ranks(train,4096); ids=core_ids(ranks,np.bincount(val,minlength=4096))
    positions=np.flatnonzero(np.isin(val,ids)); positions=positions[positions>0]
    legacy=read(v36.EVAL/'audit/baseline-seed-42.json'); tail=legacy['buckets'][RARE]
    assert v36.fingerprint(ranks)==legacy['rank_sha256']
    docs=np.maximum(0,np.cumsum(val==tok.bos_id)-1)
    populations={}
    for name,pos in (('core',positions),('tail',np.asarray(tail['positions']))):
        values={'positions':pos.tolist(),'token_ids':val[pos].astype(int).tolist(),'document_ids':docs[pos].astype(int).tolist()}
        values.update({'token_count':len(set(values['token_ids'])),'occurrence_count':len(pos),'document_count':len(set(values['document_ids']))})
        values['sha256']=v36.fingerprint((values['positions'],values['token_ids']))
        populations[name]=values
    assert populations['tail']['token_ids']==tail['token_ids']
    assert populations['tail']['sha256']==read(ROOT/'evaluation/foundation-v37-rare-exposure.json')['fixed_set_hash']
    write(path,{'phase':49,'core_rule':read(EVAL/'gate-policy.json')['core_selection'],'populations':populations,
        'train_sha256':file_sha256(ROOT/'data/foundation_v11/packed/vocab-4096/train.bin'),
        'validation_sha256':file_sha256(ROOT/'data/foundation_v11/packed/vocab-4096/validation.bin'),
        'rank_sha256':v36.fingerprint(ranks),'checkpoint_results_used_for_selection':False,
        'core_tail_id_overlap':len(set(populations['core']['token_ids'])&set(populations['tail']['token_ids']))})
    print({k:{n:r[n] for n in ('token_count','occurrence_count','document_count')} for k,r in populations.items()},flush=True)


def cluster_ci(values,documents,replicates=2000):
    values=np.asarray(values,dtype=float); _,inv=np.unique(documents,return_inverse=True)
    totals=np.bincount(inv,weights=values); counts=np.bincount(inv); rng=np.random.default_rng(4900)
    if len(counts)<2: return None
    draws=rng.integers(0,len(counts),size=(replicates,len(counts)))
    estimates=totals[draws].sum(1)/counts[draws].sum(1)
    return np.quantile(estimates,[.025,.975]).tolist()


def summarize(values,population):
    ce=np.asarray(values['ce']); p=np.asarray(values['probabilities']); ids=np.asarray(population['token_ids']); docs=population['document_ids']
    return {'micro_ce':float(ce.mean()),'macro_per_token_ce':float(np.mean([ce[ids==t].mean() for t in np.unique(ids)])),
        'top1':float(np.mean(values['top1'])),'top5':float(np.mean(values['top5'])),'top10':float(np.mean(values['top10'])),
        'mean_probability':float(p.mean()),'median_probability':float(np.median(p)),
        'geometric_mean_probability':float(np.exp(-ce.mean())),'q25':float(np.quantile(p,.25)),'q75':float(np.quantile(p,.75)),
        'micro_ce_ci95':cluster_ci(ce,docs),'mean_probability_ci95':cluster_ci(p,docs),
        **{k:population[k] for k in ('token_count','occurrence_count','document_count')}}


def paired(before,after,population):
    assert before['population_sha256']==after['population_sha256']==population['sha256']
    delta=np.asarray(after['values']['ce'])-np.asarray(before['values']['ce'])
    return {'ce_delta':float(delta.mean()),'ce_delta_ci95':cluster_ci(delta,population['document_ids']),
        'probability_delta_ci95':cluster_ci(np.asarray(after['values']['probabilities'])-np.asarray(before['values']['probabilities']),population['document_ids'])}


def legacy(seed):
    path=evaluation_path('B',seed)
    if path.exists(): raise FileExistsError(path)
    source=checkpoint('B',seed); digest=file_sha256(source); p,model=ev36.load_model(source)
    tok=FoundationTokenizer.load(ROOT/'tokenizer/foundation-v11-base-4096.json')
    train=np.memmap(ROOT/'data/foundation_v11/packed/vocab-4096/train.bin',dtype=np.uint16,mode='r')
    val=np.memmap(ROOT/'data/foundation_v11/packed/vocab-4096/validation.bin',dtype=np.uint16,mode='r')
    ranks=frequency_ranks(train,4096); positions=ev35.context_positions(val,tok)
    prefixes=ev35.build_prefixes(val,ev35.document_ranges(val,tok.bos_id,tok.eos_id),tok)
    probe=ev36.frequency_probe(model,val,ranks); base=read(v36.EVAL/f'audit/baseline-seed-{seed}.json')
    assert probe['population_sha256']==base['population_sha256'] and probe['rank_sha256']==base['rank_sha256']
    probe['buckets']={RARE:probe['buckets'][RARE]}
    terminal=np.resize(np.array([i for i in np.flatnonzero(val==tok.eos_id) if i>=128]),500)
    result={'phase':49,'arm':'B','seed':seed,'tokens_processed':p['tokens_processed'],'frequency_probe':probe,
        'validation':ev35.language_metrics_detailed(model,val,ranks),'context':ev35.context_profile(model,val,positions),
        'sanity':ev35.sanity_checks(model,p,positions,val,tok),'terminal_eos':ev35.target_metrics(model,val,terminal,tok.eos_id),
        'nonterminal_eos':ev35.target_metrics(model,val,np.linspace(128,len(val)-2,500,dtype=int),tok.eos_id),
        'generation':ev35.generation_metrics(model,tok,prefixes),'teacher_forced_horizons':ev35.teacher_forced_horizons(model,val,prefixes),
        'checkpoint_sha256':digest,'evaluation_execution':{'device':'cpu','threads':4,'parallel_cpu_evaluation':'DISABLED'}}
    assert file_sha256(source)==digest; result['checkpoint_unchanged']=True; write(path,result)
    print('legacy CPU evaluation complete',seed,result['validation']['loss'],flush=True)


@torch.inference_mode()
def frequency(arm,seed):
    target=EVAL/f'frequency/{arm}-seed-{seed}.json'
    if target.exists(): raise FileExistsError(target)
    definition=read(EVAL/'frequency-population.json'); pop=definition['populations']['core']
    valpath=ROOT/'data/foundation_v11/packed/vocab-4096/validation.bin'
    assert file_sha256(valpath)==definition['validation_sha256']
    path=v36.official(seed) if arm=='baseline' else checkpoint(arm,seed)
    digest=file_sha256(path); _,model=ev36.load_model(path); model=model.to('cuda').eval()
    val=np.memmap(valpath,dtype=np.uint16,mode='r'); positions=np.array(pop['positions']); values={k:[] for k in ('ce','probabilities','top1','top5','top10')}
    for block in np.unique((positions-1)//512):
        start=int(block)*512; size=min(512,len(val)-start-1); inputs=torch.tensor(np.array(val[start:start+size],dtype=np.int64),device='cuda')[None]
        logits,_=model(inputs); mask=positions[(positions>start)&(positions<=start+size)]
        rows=logits[0,torch.tensor(mask-start-1,device='cuda')].float(); truth=torch.tensor(np.array(val[mask],dtype=np.int64),device='cuda')
        logp=torch.log_softmax(rows,-1).gather(1,truth[:,None]).squeeze(1); top=rows.topk(10,-1).indices
        values['ce'].extend((-logp).cpu().tolist()); values['probabilities'].extend(logp.exp().cpu().tolist())
        for k in (1,5,10): values[f'top{k}'].extend((top[:,:k]==truth[:,None]).any(-1).cpu().tolist())
    assert len(values['ce'])==len(positions) and np.isfinite(values['ce']).all()
    del model; torch.cuda.empty_cache()
    # Exact legacy CPU probabilities are retained for the 112-position Tail.
    if arm=='baseline': tail=read(v36.EVAL/f'audit/baseline-seed-{seed}.json')['buckets'][RARE]; detail=read(ROOT/f'evaluation/phase46/baseline/seed-{seed}.json')
    else: detail=read(evaluation_path(arm,seed)); tail=detail['frequency_probe']['buckets'][RARE]
    tp=definition['populations']['tail']; assert tail['positions']==tp['positions'] and tail['token_ids']==tp['token_ids']
    core={'population_sha256':pop['sha256'],'values':values,'metrics':summarize(values,pop),'device':'CUDA FP32, no autocast; fixed packed 512-token contexts'}
    tv={'ce':tail['per_position_ce'],'probabilities':tail['probabilities']}
    # Top-k scalar values are exact legacy metrics; do not invent per-position ranks.
    raw=detail['validation']['frequency_buckets'][RARE]
    tm=summarize({**tv,**{f'top{k}':[raw[f'top_{k}_accuracy']] for k in (1,5,10)}},tp)
    result={'phase':49,'arm':arm,'seed':seed,'checkpoint_sha256':digest,'core':core,
        'tail':{'population_sha256':tp['sha256'],'values':tv,'metrics':tm,'device':'Exact reused CPU probe'},'final_blind_used':False}
    assert file_sha256(path)==digest; write(target,result)
    print('frequency complete',arm,seed,core['metrics']['micro_ce'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--freeze',action='store_true'); parser.add_argument('--legacy',action='store_true')
    parser.add_argument('--arm',choices=('baseline','B','C')); parser.add_argument('--seed',type=int,choices=SEEDS)
    a=parser.parse_args(); torch.set_num_threads(4)
    if a.freeze: freeze()
    elif a.legacy: legacy(a.seed)
    else: frequency(a.arm,a.seed)
