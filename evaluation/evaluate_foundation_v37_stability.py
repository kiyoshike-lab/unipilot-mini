"""PHASE48 CPU metrics and fixed-token training exposure audit."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from training.run_foundation_v37_stability import ARMS, SEEDS, EVAL, START, read, write, checkpoint, checkpoint256, evaluation_path
from training import run_foundation_v36_lr_review as v36
from evaluation import evaluate_foundation_v36_lr_review as ev36
from evaluation import evaluate_foundation_v35_short_gate as ev35
from training.train_foundation_v21_ab import file_sha256, frequency_ranks
from foundation.base_tokenizer import FoundationTokenizer

RARE='rare_bottom_20_percent'


def evaluate(arm,seed,budget):
    target=evaluation_path(arm,seed,budget)
    if target.exists(): raise FileExistsError('reuse existing evaluation: '+str(target))
    path=checkpoint(arm,seed,budget); digest=file_sha256(path); p,model=ev36.load_model(path)
    tok=FoundationTokenizer.load(ROOT/'tokenizer/foundation-v11-base-4096.json')
    train=np.memmap(ROOT/'data/foundation_v11/packed/vocab-4096/train.bin',dtype=np.uint16,mode='r')
    val=np.memmap(ROOT/'data/foundation_v11/packed/vocab-4096/validation.bin',dtype=np.uint16,mode='r')
    ranks=frequency_ranks(train,tok.vocab_size); positions=ev35.context_positions(val,tok)
    prefixes=ev35.build_prefixes(val,ev35.document_ranges(val,tok.bos_id,tok.eos_id),tok)
    terminal=np.resize(np.asarray([int(i) for i in np.flatnonzero(val==tok.eos_id) if i>=128]),500)
    nonterminal=np.linspace(128,len(val)-2,500,dtype=int)
    probe=ev36.frequency_probe(model,val,ranks)
    base=read(v36.EVAL/f'audit/baseline-seed-{seed}.json')
    assert probe['population_sha256']==base['population_sha256'] and probe['rank_sha256']==base['rank_sha256']
    for key in ('positions','token_ids'): assert probe['buckets'][RARE][key]==base['buckets'][RARE][key]
    probe['buckets']={RARE:probe['buckets'][RARE]}
    result={'phase':48,'arm':arm,'seed':seed,'cumulative_budget':budget,'tokens_processed':p['tokens_processed'],
        'checkpoint_sha256':digest,'fixed_rare_set_integrity':True,'frequency_probe':probe,
        'validation':ev35.language_metrics_detailed(model,val,ranks),'context':ev35.context_profile(model,val,positions),
        'sanity':ev35.sanity_checks(model,p,positions,val,tok),
        'terminal_eos':ev35.target_metrics(model,val,terminal,tok.eos_id),
        'nonterminal_eos':ev35.target_metrics(model,val,nonterminal,tok.eos_id),
        'generation':ev35.generation_metrics(model,tok,prefixes),'teacher_forced_horizons':ev35.teacher_forced_horizons(model,val,prefixes),
        'evaluation_execution':{'device':'cpu','threads':4,'parallel_cpu_evaluation':'DISABLED'},
        'sampling_note':'Identical Phase46/47 automatic proxies and fixed seeds; not human judgments.'}
    assert file_sha256(path)==digest
    result['checkpoint_unchanged']=True; write(target,result)
    print({'arm':arm,'seed':seed,'budget':budget,'loss':result['validation']['loss'],'rare_ce':probe['buckets'][RARE]['ce']},flush=True)


def exposure_counts(train,permutation,update,steps,vocab):
    counts=np.zeros(vocab,dtype=np.int64)
    for index in permutation[update:update+steps]:
        start=int(index)*512; block=train[start+1:start+513]
        if len(block)!=512: raise ValueError('incomplete target block')
        counts+=np.bincount(block,minlength=vocab)
    return counts


def average_ranks(values):
    _,inverse,counts=np.unique(values,return_inverse=True,return_counts=True)
    ranks=np.cumsum(counts)-counts+(counts-1)/2
    return ranks[inverse]


def spearman(a,b):
    a,b=average_ranks(a),average_ranks(b)
    if len(a)<3 or np.std(a)==0 or np.std(b)==0: return None
    return float(np.corrcoef(a,b)[0,1])


def exposure_summary(counts,ids):
    a=np.asarray(counts)[ids]
    return {'total_rare_target_occurrences':int(a.sum()),'unique_rare_tokens_observed':int(np.sum(a>0)),
        'rare_tokens_never_observed':int(np.sum(a==0)),'rare_tokens_observed_once':int(np.sum(a==1)),
        'observed_2_to_4':int(np.sum((a>=2)&(a<=4))),'observed_5_to_9':int(np.sum((a>=5)&(a<=9))),
        'observed_10_plus':int(np.sum(a>=10)),'unseen_percent':100*float(np.mean(a==0))}


def paired_exposure(before,after,counts,tok):
    assert before['token_ids']==after['token_ids'] and before['positions']==after['positions']
    target_ids=np.asarray(before['token_ids']); rows=[]
    for tid in np.unique(target_ids):
        mask=target_ids==tid
        old_ce=float(np.mean(np.asarray(before['per_position_ce'])[mask])); new_ce=float(np.mean(np.asarray(after['per_position_ce'])[mask]))
        old_p=float(np.mean(np.asarray(before['probabilities'])[mask])); new_p=float(np.mean(np.asarray(after['probabilities'])[mask]))
        rows.append({'token_id':int(tid),'text':tok.decode([int(tid)]),'train_exposure':int(counts[tid]),'eval_occurrences':int(mask.sum()),
            'ce_before':old_ce,'ce_after':new_ce,'ce_delta':new_ce-old_ce,'probability_before':old_p,'probability_after':new_p,
            'probability_delta':new_p-old_p,'ce_contribution':(new_ce-old_ce)*int(mask.sum())/len(mask)})
    x=[r['train_exposure'] for r in rows]; ce=[r['ce_delta'] for r in rows]; prob=[r['probability_delta'] for r in rows]
    buckets={}
    for name,low,high in (('0',0,1),('1',1,2),('2-4',2,5),('5+',5,float('inf'))):
        selected=[r for r in rows if low<=r['train_exposure']<high]
        buckets[name]={'token_types':len(selected)}
        for key in ('ce_delta','probability_delta'):
            values=[r[key] for r in selected]
            buckets[name][key]={'mean':float(np.mean(values)) if values else None,'median':float(np.median(values)) if values else None}
    positive=sum(max(0,r['ce_contribution']) for r in rows)
    return {'spearman_exposure_ce':spearman(x,ce),'spearman_exposure_probability':spearman(x,prob),'per_token':rows,'exposure_buckets':buckets,
        'low_exposure_positive_contribution_share':sum(max(0,r['ce_contribution']) for r in rows if r['train_exposure']<=4)/positive if positive else 0.,
        'low_exposure_population_share':sum(r['eval_occurrences'] for r in rows if r['train_exposure']<=4)/len(target_ids),
        'top50_positive_contributors':sorted([r for r in rows if r['ce_contribution']>0],key=lambda r:r['ce_contribution'],reverse=True)[:50]}


def exposure_audit():
    tok=FoundationTokenizer.load(ROOT/'tokenizer/foundation-v11-base-4096.json')
    data=np.memmap(ROOT/'data/foundation_v11/packed/vocab-4096/train.bin',dtype=np.uint16,mode='r')
    ranks=frequency_ranks(data,tok.vocab_size); seed_rows={}; populations=[]
    prior=read(ROOT/'evaluation/foundation-v36-rare-analysis.json')
    for seed in SEEDS:
        p=torch.load(v36.official(seed),map_location='cpu',weights_only=False)
        before=read(v36.EVAL/f'audit/baseline-seed-{seed}.json'); rare=before['buckets'][RARE]
        ids=np.unique(rare['token_ids']); populations.append(v36.fingerprint((rare['positions'],rare['token_ids'])))
        assert v36.fingerprint(ranks)==before['rank_sha256']
        c256=exposure_counts(data,p['permutation'],p['update'],500,tok.vocab_size)
        csecond=exposure_counts(data,p['permutation'],p['update']+500,500,tok.vocab_size)
        c512=exposure_counts(data,p['permutation'],p['update'],1000,tok.vocab_size)
        assert np.array_equal(c512,c256+csecond) and c256.sum()==256000 and c512.sum()==512000
        row={'256k':exposure_summary(c256,ids),'512k':exposure_summary(c512,ids),'per_token_counts':{str(t):{'256k':int(c256[t]),'second_256k':int(csecond[t]),'512k':int(c512[t])} for t in ids},'comparisons':{}}
        row['phase47_top_outliers']=[]
        for old in prior['per_seed'][str(seed)]['top50']:
            tid=old['token_id']; assert int(c256[tid])==old['training_interval_occurrences']
            row['phase47_top_outliers'].append({**old,'cumulative_512k_exposure':int(c512[tid])})
        for arm in ('A','B','C'):
            for budget,counts in ((256000,c256),(512000,c512)):
                if arm=='A':
                    if budget!=256000: continue
                    after=read(v36.EVAL/f'audit/gate1-seed-{seed}.json')['buckets'][RARE]
                else:
                    path=evaluation_path(arm,seed,budget)
                    if not path.exists(): continue
                    after=read(path)['frequency_probe']['buckets'][RARE]
                row['comparisons'][f'{arm}-{budget}']=paired_exposure(rare,after,counts,tok)
        seed_rows[str(seed)]=row
    assert len(set(populations))==1
    token_cv={}
    for stage in ('256k','512k'):
        counts=np.array([[r['per_token_counts'][str(t)][stage] for t in ids] for r in seed_rows.values()])
        averages=counts.mean(axis=0); observed=averages>0
        cv=counts[:,observed].std(axis=0,ddof=1)/averages[observed]
        token_cv[stage]={'mean':float(cv.mean()),'median':float(np.median(cv)),
                         'all_seed_unseen_types':int((~observed).sum())}
    result={'phase':48,'fixed_rare_set_integrity':True,'fixed_set_hash':populations[0],'token_ids':ids.tolist(),
        'unique_eval_token_count':len(ids),'evaluation_occurrence_count':len(rare['token_ids']),
        'rank_hash':v36.fingerprint(ranks),'definition':prior['bucket_definition'],
        'phase47_outlier_exposure_reproduced':True,'per_token_cross_seed_exposure_cv':token_cv,
        'weighting':'Counts use exact target positions from resumed sampler. Correlations use 58 token-type means with average ranks for ties; CE contributions retain occurrence weighting.',
        'seed_rows':seed_rows,'cumulative_counts_verified':True,
        'unseen_percent':{stage:{s:r[stage]['unseen_percent'] for s,r in seed_rows.items()} for stage in ('256k','512k')},
        'total_exposure_cv':{stage:float(np.std([r[stage]['total_rare_target_occurrences'] for r in seed_rows.values()],ddof=1)/np.mean([r[stage]['total_rare_target_occurrences'] for r in seed_rows.values()])) for stage in ('256k','512k')}}
    write(ROOT/'evaluation/foundation-v37-rare-exposure.json',result)
    print({'fixed_set':'PASS','unseen_percent':result['unseen_percent']},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--exposure',action='store_true'); parser.add_argument('--arm',choices=ARMS)
    parser.add_argument('--seed',type=int,choices=SEEDS); parser.add_argument('--budget',type=int,choices=(256000,512000),default=256000)
    a=parser.parse_args(); torch.set_num_threads(4)
    if a.exposure: exposure_audit()
    else: evaluate(a.arm,a.seed,a.budget)
