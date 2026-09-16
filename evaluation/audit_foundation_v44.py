"""Historical observational audit + fixed normal-repeat controls; never train."""
from __future__ import annotations
import gc,gzip,json,re,sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.diagnose_foundation_v40 import read,new_json,file_sha256,evaluated_checkpoint,load_model,generate_batch,GREEDY,cooldown,Monitor
from evaluation.confirm_foundation_v44 import gate,thermal_guard
from foundation.base_tokenizer import FoundationTokenizer
OUT=ROOT/'evaluation/phase55';RAW=Path(r'Z:\AI\unipilot-mini\evaluation\phase55')

def controls():
    gate();old=read(ROOT/'evaluation/foundation-v43-training-fix-preregistration.json')['normal_controls']
    prompts=[]
    for family in old['families']:
        for n in old['values']:prompts.append({'id':f"{family['id']}-{n}",'family':family['id'],'n':n,'prompt':family['template']+f'\nn={n}。','allowed':family['allowed'],'pathological':family['pathological']})
    spec={'source_sha256':file_sha256(ROOT/'evaluation/foundation-v43-training-fix-preregistration.json'),'prompts':prompts,
        'models':['baseline-42','C-42','B-42'],'decoder':GREEDY,'max_new_tokens':128,'rng':'0..19 (greedy unused)',
        'purpose':'normal repetition descriptive risk check; not LR selection; no human correctness certification','new_training':False}
    if not (OUT/'normal-controls-preregistration.json').exists():new_json(OUT/'normal-controls-preregistration.json',spec)
    else:assert read(OUT/'normal-controls-preregistration.json')==spec
    tok=FoundationTokenizer.load(ROOT/'tokenizer/foundation-v11-base-4096.json');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    for arm in ('baseline','C','B'):
        p=RAW/f'normal-controls-{arm}-42.json'
        if p.exists():assert read(p)['complete'];continue
        checkpoint=evaluated_checkpoint(arm,42);migration=read(ROOT/'evaluation/phase50/z-migration.json')
        known=next(r for r in migration['rows'] if Path(r['destination'])==checkpoint);assert file_sha256(checkpoint)==known['sha256']
        assert cooldown()['target_reached'];model=load_model(checkpoint,torch.device('cuda'));m=Monitor();m.start();rows=[]
        try:
            with torch.inference_mode():
                for i,prompt in enumerate(prompts):
                    thermal_guard(m)
                    generated=generate_batch(model,tok,[[tok.bos_id]+tok.encode(prompt['prompt'])],GREEDY,[i],128)[0]
                    text=generated['text'];n=prompt['n'];family=prompt['family']
                    if family=='math':shape=bool(re.search(re.escape('+'.join(['1']*n))+r'\s*[=＝]\s*'+str(n)+r'(?!\d)',re.sub(r'\s+','',text)))
                    elif family=='code':shape=[int(x) for x in re.findall(r'print\((\d+)\)',text)]==list(range(1,n+1))
                    elif family=='lists':shape=len(re.findall('確認済み',text))==n and all(re.search(rf'(?m)^\s*{j}[.、)．]',text) for j in range(1,n+1))
                    elif family=='definitions':shape=len(re.findall(r'定義[:：]',text))==n and all(t in text for t in ['変数','関数','集合','写像','命題'][:n])
                    else:shape=text.count('標本平均')==n and len(set(re.split('[。！？]',text))- {''})>=n
                    rows.append({'id':prompt['id'],'family':family,'shape_proxy':bool(shape),'shape_not_semantic_correctness':True,**generated})
        finally:telemetry=m.finish();del model;gc.collect();torch.cuda.empty_cache()
        new_json(p,{'complete':True,'checkpoint_sha256':known['sha256'],'rows':rows,'thermal':telemetry,'human_boundary_review':'PENDING'})
        print('Normal repetition controls completed',arm,flush=True)

def audit():
    gate();old=read(ROOT/'evaluation/foundation-v41-attractor-summary.json');p=ROOT/old['raw_timeseries_path']
    assert file_sha256(p)==old['raw_timeseries_sha256'];traces=read(p)
    plan={'source_sha256':file_sha256(p),'windows':'up to32 steps strictly before onset and next32 strictly after onset; boundaries reported, no fabricated missing steps',
        'copy_rule':'Top20 most frequent generated loop-region ngrams separately for n=3,4,5,6, tie lexicographic token IDs. Count complete packed training corpus, no cross-document hits. Document/category support uses verified tokenization and JSONL ordering.',
        'classification_rule':'Observational association cannot establish dominant causal mechanism. Without intervention/calibration/position control sufficient to distinguish competing mechanisms: INSUFFICIENT_EVIDENCE; MORE_ROOT_CAUSE_WORK_REQUIRED. Descriptive self-copy signature reported separately.',
        'no_text_quotations':True,'new_training':False}
    new_json(OUT/'root-cause-plan.json',plan)
    windows=[];counts={n:Counter() for n in range(3,7)};groups=defaultdict(list)
    fields=('entropy','top1_probability','top2_probability','margin','eos_probability','unique_token_ratio')
    for row in traces:
        series=row['full_time_series'];onset=row['loop_onset'];before=[t for t in series if onset-32<=t['step']<onset];after=[t for t in series if onset<t['step']<=onset+32]
        compact={'arm':row['arm'],'seed':row['seed'],'prompt_id':row['prompt_id'],'family':row['family'],'prefix_length':row['prefix_length'],'onset':onset,'cycle_length':row['cycle_length'],
            'before_steps':len(before),'after_steps':len(after),'before':{k:float(np.mean([t[k] for t in before])) if before else None for k in fields},'after':{k:float(np.mean([t[k] for t in after])) if after else None for k in fields}}
        windows.append(compact);groups[f"{row['arm']}-{row['seed']}"].append(compact)
        ids=[t['token_id'] for t in series[onset-1:]]
        for n in counts:counts[n].update(tuple(ids[i:i+n]) for i in range(len(ids)-n+1))
    selected={n:sorted(c.items(),key=lambda kv:(-kv[1],kv[0]))[:20] for n,c in counts.items()}
    trainpath=ROOT/'data/foundation_v11/packed/vocab-4096/train.bin';manifest=read(ROOT/'data/foundation_v11/packed/vocab-4096/manifest.json')
    assert file_sha256(trainpath)==manifest['splits']['train']['sha256']
    train=np.memmap(trainpath,dtype=np.uint16,mode='r');tok=FoundationTokenizer.load(ROOT/'tokenizer/foundation-v11-base-4096.json')
    starts=np.flatnonzero(train==tok.bos_id);ends=np.flatnonzero(train==tok.eos_id);assert len(starts)==len(ends)==10012
    categories=[]
    with gzip.open(ROOT/'data/foundation_v11/documents/train.jsonl.gz','rt',encoding='utf-8') as handle:
        for i,line in enumerate(handle):
            doc=json.loads(line);encoded=tok.encode(doc['text']);assert np.array_equal(train[starts[i]+1:ends[i]],encoded),'Packed/JSONL identity mismatch'
            categories.append(doc.get('category','UNKNOWN'))
            if i%2000==0:print('Verified training document mapping',i,flush=True)
    assert len(categories)==len(starts)
    first_cache={};support=[]
    for n,items in selected.items():
        for gram,count in items:
            if gram[0] not in first_cache:first_cache[gram[0]]=np.flatnonzero(train==gram[0])
            positions=first_cache[gram[0]];positions=positions[positions+n<=len(train)]
            for j,token in enumerate(gram[1:],1):positions=positions[train[positions+j]==token]
            documents=np.searchsorted(starts,positions,side='right')-1
            mask=positions+n<=ends[documents];positions=positions[mask];documents=documents[mask]
            doc_ids=set(map(int,documents));catcounts=Counter(categories[d] for d in doc_ids)
            support.append({'n':n,'token_ids':list(gram),'generated_occurrences':count,'train_frequency':len(positions),'document_support':len(doc_ids),'category_support':dict(catcounts)})
    normal={}
    for arm in ('baseline','C','B'):
        raw=read(RAW/f'normal-controls-{arm}-42.json');rs=raw['rows']
        normal[arm]={'examples':len(rs),'shape_proxy_passes':sum(r['shape_proxy'] for r in rs),'runaway_rate':float(np.mean([r['runaway'] for r in rs])),'eos_rate':float(np.mean([r['eos_reached'] for r in rs])),
            'by_family':{f:{'examples':sum(r['family']==f for r in rs),'shape_passes':sum(r['family']==f and r['shape_proxy'] for r in rs)} for f in ('math','lists','definitions','code','terminology')},'human_correctness':'NOT_MEASURED'}
    new_json(OUT/'attractor-root-cause.json',{'classification':'INSUFFICIENT_EVIDENCE','descriptive_signature':'SELF_COPY_ATTRACTOR with low EOS and probability concentration; not proof of dominance',
        'training_fix_gate':'MORE_ROOT_CAUSE_WORK_REQUIRED','formal_lr':read(ROOT/'evaluation/foundation-v44-confirmatory-summary.json')['formal_lr_gate'],'proposed_arms':[],
        'source_sha256':file_sha256(p),'trace_count':len(traces),'windows':windows,
        'by_checkpoint':{k:{'before_steps_mean':float(np.mean([r['before_steps'] for r in rs])),'after_steps_mean':float(np.mean([r['after_steps'] for r in rs])),
            'before':{f:float(np.mean([r['before'][f] for r in rs if r['before'][f] is not None])) for f in fields},'after':{f:float(np.mean([r['after'][f] for r in rs if r['after'][f] is not None])) for f in fields}} for k,rs in groups.items()},
        'copy_support':support,'train_sha256':file_sha256(trainpath),'document_category_mapping':'all10012 retokenized and exact packed match',
        'hidden_logit_cosine':'NOT_MEASURED: historical traces contain top5 probabilities, not full vectors/hidden states; cannot reconstruct cosine. No new architecture hooks introduced.',
        'normal_controls':normal,'normal_control_source_sha256':file_sha256(OUT/'normal-controls-preregistration.json'),
        'phase53_policy':'GENERATION_POLICY_UNSAFE; historical sampling/generation findings retained',
        'limitations':['32-step windows are descriptive; early/late boundaries have fewer steps.','Tokenized ngram support does not distinguish learned grammar, memorization, exposure bias or data-induced dynamics.','Entropy/margins are not calibrated confidence or causal interventions.','Normal control shape is not semantic correctness; independent human boundary review pending.'],
        'new_training':False,'canonical':False,'20m':False})
    print('Root audit complete; INSUFFICIENT_EVIDENCE / MORE_ROOT_CAUSE_WORK_REQUIRED',flush=True)

if __name__=='__main__':{'controls':controls,'audit':audit}[sys.argv[1]]()
