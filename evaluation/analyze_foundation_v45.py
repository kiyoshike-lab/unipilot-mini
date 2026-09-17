"""Analyze preregistered observations; no inference, LR selection or training."""
from __future__ import annotations
import gzip,json,sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.diagnose_foundation_v40 import read,new_json,file_sha256
from evaluation.run_foundation_v45_observatory import gate,OUT,RAW,SPEC
from evaluation.generation_observatory_v45 import loop_details
from foundation.base_tokenizer import FoundationTokenizer

FIELDS=('entropy','top1_probability','margin','eos_probability','eos_rank','eos_logit','top1_minus_eos_logit','hidden_norm','logit_norm','logit_cycle_centered_cosine')
def average(xs):
    xs=[x for x in xs if x is not None];return float(np.mean(xs)) if xs else None
def window(run,onset,side):
    onset=onset or 65
    rows=[r for r in run['rows'] if (onset-32<=r['step']<onset if side=='before' else onset<r['step']<=onset+32)]
    return {'steps':len(rows),'metrics':{k:average([r[k] for r in rows]) for k in FIELDS},'layers':[{k:average([r['layers'][i][k] for r in rows]) for k in ('l2_norm','previous_cosine','cycle_cosine','attention_previous_cycle','attention_earlier_context','attention_bos','attention_expected_position')} for i in range(run['hook_layers'])],
        'attention_recent1_4':{str(n):[average([r['layers'][i]['attention_recent_1_4'][str(n)] for r in rows]) for i in range(run['hook_layers'])] for n in (1,2,3,4)}}

def main():
    gate();spec=read(SPEC);index=read(OUT/'run-index.json');assert index['preregistration_sha256']==file_sha256(SPEC)
    records=index['runs'];by={};patterns={n:Counter() for n in (3,4,5,6,8)};rows=[];perturb=[];normal=[];legacy=[]
    for record in records:
        p=Path(record['path']);assert file_sha256(p)==record['sha256'];r=read(p)
        assert file_sha256(Path(r['vectors_path']))==record['vectors_sha256'];by[record['arm'],record['seed'],record['name']]=r
    for arm,seed in [('C',42),('C',123),('C',2026),('baseline',42)]:
        for i in range(8 if arm=='baseline' else 24):
            name=f'doc{i:02}';free=by[arm,seed,name+'-free'];teacher=by[arm,seed,name+'-teacher'];onset=free['loop']['loop_onset']
            pair={'arm':arm,'seed':seed,'document_id':spec['selected_document_ids'][i],'loop':free['loop'],'free_runaway':free['runaway'],
                'free_repetition':free['repetition'],'teacher_repetition':teacher['repetition'],
                'free_before':window(free,onset,'before'),'free_after':window(free,onset,'after'),'teacher_before':window(teacher,onset,'before'),'teacher_after':window(teacher,onset,'after'),
                'ngram_repeat_onsets':free['ngram_repeat_onsets'],'teacher_repeat_onsets':teacher['ngram_repeat_onsets']}
            if arm=='C':
                short=by[arm,seed,name+'-short'];pair['position_context']={'long_onset':onset,'short_onset':short['loop']['loop_onset'],'long_rep4':free['repetition']['4'],'short_rep4':short['repetition']['4'],'short_runaway':short['runaway'],'pure_position_causal_claim':False}
                ids=free['ids'][(onset or 1)-1:]
                for n in patterns:patterns[n].update(tuple(ids[j:j+n]) for j in range(len(ids)-n+1))
                for offset in (4,8,16):
                    for kind in ('top2','truth'):
                        key=(arm,seed,name+f'-{kind}-{offset}')
                        if key not in by:continue
                        r=by[key];step=r['replace_step'];changed=r['ids'][step-1]!=free['ids'][step-1];suffix=r['ids'][step:];loop=loop_details(suffix)
                        perturb.append({'seed':seed,'document_id':pair['document_id'],'type':kind,'offset':offset,'step':step,'changed':changed,'remaining_steps':len(suffix),'eligible':changed and len(suffix)>=16,
                            'suffix_loop':loop['loop_onset'] is not None,'suffix_onset':loop['loop_onset'],'runaway':r['runaway'],'eos':r['eos_reached'],'free_onset':onset,'new_full_onset':r['loop']['loop_onset']})
            rows.append(pair)
        if arm=='C':
            for k,r in by.items():
                if k[:2]!=(arm,seed):continue
                if k[2].startswith('normal-'):
                    normal.append({'seed':seed,'id':k[2],'ce':r['ce'],'target_tokens':len(r['ids']),'terminal_eos_probability':r['rows'][-1]['eos_probability'],'nonterminal_eos_probability':average([x['eos_probability'] for x in r['rows'][:-1]]),'instruction_accuracy_claim':False})
                elif k[2].startswith('legacy'):legacy.append({'seed':seed,'id':k[2],'runaway':r['runaway'],'loop':r['loop'],'repetition4':r['repetition']['4']})
    # Entire training corpus; hash-bound category/order evidence reused from PHASE55.
    tok=FoundationTokenizer.load(ROOT/'tokenizer/foundation-v11-base-4096.json');trainpath=ROOT/'data/foundation_v11/packed/vocab-4096/train.bin'
    previous=read(ROOT/'evaluation/phase55/attractor-root-cause.json');assert file_sha256(trainpath)==previous['train_sha256']
    gzip_path=ROOT/'data/foundation_v11/documents/train.jsonl.gz';known=next(r for r in read(ROOT/'evaluation/phase54/leakage-report.json')['input_files'] if Path(r['path'])==gzip_path.relative_to(ROOT));assert file_sha256(gzip_path)==known['sha256']
    train=np.memmap(trainpath,dtype=np.uint16,mode='r');starts=np.flatnonzero(train==tok.bos_id);ends=np.flatnonzero(train==tok.eos_id)
    with gzip.open(gzip_path,'rt',encoding='utf-8') as f:categories=[json.loads(line).get('category','UNKNOWN') for line in f]
    assert len(categories)==len(starts)==len(ends)==10012
    first_cache={};support=[];novel={}
    for n,counter in patterns.items():
        part=[]
        for gram,count in sorted(counter.items()):
            if gram[0] not in first_cache:first_cache[gram[0]]=np.flatnonzero(train==gram[0])
            pos=first_cache[gram[0]];pos=pos[pos+n<=len(train)]
            for j,token in enumerate(gram[1:],1):pos=pos[train[pos+j]==token]
            docs=np.searchsorted(starts,pos,side='right')-1;mask=pos+n<=ends[docs];pos,docs=pos[mask],docs[mask]
            unique=set(map(int,docs));part.append({'n':n,'token_ids':list(gram),'generated_occurrences':count,'train_frequency':len(pos),'document_support':len(unique),'category_support':dict(Counter(categories[d] for d in unique))})
        total=sum(r['generated_occurrences'] for r in part);novel[str(n)]={'unique_patterns':len(part),'novel_types':sum(r['train_frequency']==0 for r in part),
            'novel_type_rate':sum(r['train_frequency']==0 for r in part)/max(1,len(part)),
            'occurrence_weighted_novel_rate':sum(r['generated_occurrences'] for r in part if r['train_frequency']==0)/max(1,total)}
        support.extend(part);print('Training support n',n,len(part),flush=True)
    new_json(RAW/'all-ngram-support.json',support)
    checkpoints={};feedback=[]
    for seed in (42,123,2026):
        rs=[r for r in rows if r['arm']=='C' and r['seed']==seed];layers=[]
        for i in range(10):
            f=average([r['free_after']['layers'][i]['cycle_cosine'] for r in rs]);t=average([r['teacher_after']['layers'][i]['cycle_cosine'] for r in rs]);layers.append({'layer':i,'free_cycle_cosine':f,'teacher_cycle_cosine':t,'delta':None if f is None or t is None else f-t})
        rep=average([r['free_repetition']['4']-r['teacher_repetition']['4'] for r in rs]);passing=sum(l['delta'] is not None and l['delta']>=.05 for l in layers)
        feedback.append(rep>=.2 and passing>=5)
        locks=[l['layer'] for l in layers if l['delta'] is not None and l['free_cycle_cosine']>=.95 and l['delta']>=.05]
        checkpoints[str(seed)]={'documents':len(rs),'free_runaway':average([r['free_runaway'] for r in rs]),'free_minus_teacher_repetition4':rep,'cycle_layers':layers,'feedback_signal':feedback[-1],
            'earliest_cycle_lock_layer':min(locks) if locks else 'NOT_ESTABLISHED',
            'free_before':{k:average([r['free_before']['metrics'][k] for r in rs]) for k in FIELDS},'free_after':{k:average([r['free_after']['metrics'][k] for r in rs]) for k in FIELDS},
            'teacher_after':{k:average([r['teacher_after']['metrics'][k] for r in rs]) for k in FIELDS},
            'attention_free_after':{key:[average([r['free_after']['layers'][i][key] for r in rs]) for i in range(10)] for key in ('attention_previous_cycle','attention_earlier_context','attention_bos')},
            'context_short_minus_long_rep4':average([r['position_context']['short_rep4']-r['position_context']['long_rep4'] for r in rs])}
    eligible=[r for r in perturb if r['eligible']];reloop=average([r['suffix_loop'] for r in eligible]);signals={'feedback':sum(feedback)>=2,'persistent_basin':len(eligible)>=24 and reloop is not None and reloop>=.8,'novel6':novel['6']['novel_type_rate']>=.5}
    root='MIXED_CAUSE_WITH_ACTIONABLE_TARGET' if all(signals.values()) else 'INSUFFICIENT_EVIDENCE'
    summary={'phase':56,'approved_lr':5e-5,'confirmatory_rerun':False,'diagnostic_documents_source':299,'diagnostic_documents_selected':24,'consumed_for_diagnostics':True,
        'generation_and_teacher_runs':len(records),'seeds':[42,123,2026],'baseline_reference_seed':42,'hidden_state_collection':'PASS','attention_collection':'PASS',
        'hook_transparency':'exact logits equal; all model tensors unchanged in all4 checkpoint runs','checkpoints':checkpoints,'signals':signals,'root_cause_gate':root,
        'perturbation':{'runs':len(perturb),'changed_runs':sum(r['changed'] for r in perturb),'eligible_changed_with16_remaining':len(eligible),'eligible_reloop_rate':reloop,'eos_rate_all':average([r['eos'] for r in perturb]),'rows':perturb},
        'novel_loop_ngrams':novel,'training_support_raw':{'path':str(RAW/'all-ngram-support.json'),'sha256':file_sha256(RAW/'all-ngram-support.json')},
        'normal_teacher_forced':normal,'legacy':legacy,'pair_details':rows,'preregistration_sha256':file_sha256(SPEC),
        'limitations':['Small24-document mechanistic sample, correlated perturbations not independent observations.','Raw-state cosine may share embedding/position components; earliest-lock layer is not causal origin.','Attention summaries describe weights, not causal attribution.','Teacher-forcing removes feedback but also changes content; identifies regime sensitivity, not a unique causal mechanism.','Natural prefix shortening changes both content and position; pure position effect remains UNKNOWN.','No undertraining-vs-architecture controlled experiment; no dominant label justified.'],
        'new_training':False,'canonical':False,'20m':False,'foundation_base':False}
    new_json(OUT/'generation-observatory-summary.json',summary)
    new_json(ROOT/'evaluation/foundation-v45-root-cause-summary.json',{k:v for k,v in summary.items() if k not in ('pair_details','normal_teacher_forced','legacy')}|{'detailed_artifact':'evaluation/phase56/generation-observatory-summary.json','training_readiness_gate':'PENDING_POST_ROOT_OBJECTIVE_REVIEW' if root!='INSUFFICIENT_EVIDENCE' else 'MORE_OBSERVABILITY_REQUIRED'})
    print('ROOT_CAUSE_GATE',root,signals,'eligible reloop',reloop,flush=True)

if __name__=='__main__':main()
