"""Frozen one-shot, equal-document paired inference. No optimizer step or training.

freeze -> evaluate (resumes complete checkpoint outputs only) -> summarize.
Reserve paths are never read. Artifacts are exclusive, not overwritten.
"""
from __future__ import annotations
import gc, math, os, sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.diagnose_foundation_v40 import (read,new_json,file_sha256,resolver_gate,evaluated_checkpoint,
    verify_payload,load_model,generate_batch,cooldown,Monitor,SAMPLE_T07,metrics)
from evaluation.build_foundation_v42_holdout import now
from foundation.base_tokenizer import FoundationTokenizer
OUT=ROOT/'evaluation/phase55'
RAW=Path(r'Z:\AI\unipilot-mini\evaluation\phase55')
SPEC=OUT/'lr-confirmatory-preregistration.json'
SEEDS=(42,123,2026)

def gate():
    assert os.environ.get('UNIPILOT_CHECKPOINT_ROOT')==r'Z:\AI\unipilot-mini\checkpoints'
    return resolver_gate()

def freeze():
    migration=gate();manifest=read(OUT/'fresh-holdout-v2-manifest.json');assert manifest['gate']=='FRESH_HOLDOUT_V2_READY'
    split=manifest['splits']['confirmatory'];assert file_sha256(Path(split['path']))==split['sha256']
    docs=read(split['path']);tok=FoundationTokenizer.load(ROOT/'tokenizer/foundation-v11-base-4096.json')
    prompts=[{'id':f"jawiki:{r['page_id']}:{r['revision_id']}",'prefix_ids':[tok.bos_id]+tok.encode(r['text'])[:63]} for r in docs[:64]]
    rows=[];torch.set_num_threads(2)
    for arm in ('C','B'):
        for seed in SEEDS:
            p=evaluated_checkpoint(arm,seed);sha=file_sha256(p)
            known=next(r for r in migration['rows'] if Path(r['destination'])==p);assert sha==known['sha256']
            payload=torch.load(p,map_location='cpu',weights_only=False)
            result=verify_payload(payload,seed,16384000,5e-5 if arm=='C' else 7.5e-5)
            rows.append({'arm':arm,'seed':seed,'path':str(p),'sha256':sha,'bytes':p.stat().st_size,'integrity':result})
            del payload;gc.collect();print('Strict/resume integrity PASS',arm,seed,flush=True)
    new_json(OUT/'integrity.json',{'rows':rows,'resume_detail_reference':'evaluation/phase50/z-migration.json; identical checkpoint SHA, Python/NumPy/PyTorch/CUDA RNG and parent metadata preserved',
        'protected_files':migration['protected_files'],'new_training':False})
    new_json(SPEC,{'phase':55,'frozen_at':now(),'results_seen':False,'confirmatory':split,'checkpoints':rows,
        'primary':'equal-document mean CE C-B at max context512; equal mean of three paired seeds per document; seed-specific comparisons descriptive',
        'lm':'All document tokens including terminal EOS, excluding BOS target. Prediction chunks128. Full input includes up to384 preceding tokens (max512); short input only current128 block. Identical targets, no cross-document context, no padding in loss. Equal weight per document, never token-weighted across documents.',
        'secondary':['Top1','Top5','Top10','exp(equal-document CE)','context512','context128'],
        'bootstrap':{'replicates':10000,'seed':550055,'unit':'paired document after equal seed mean','interval':'percentile 95%; seeds fixed, no population-of-training-seeds inference','weight':'equal documents; no exclusions'},
        'decision':'Primary upper<0 C; lower>0 B; otherwise FORMAL_LR_UNRESOLVED. Winner approved only if no preregistered relative safety regression; no new training authorized.',
        'safety':'Selected minus other arm, paired document/prompt bootstrap averaged3 seeds. Clear material worsening requires entire95% CI beyond boundary, not just point estimate. Shared absolute failures reported separately, never presented as production safe. Multiple safety flags are conservative OR; limited power is reported.',
        'safety_thresholds':{'context128_ce_increase':.03,'context_benefit_loss':.03,'terminal_eos_probability_relative_drop':.20,'premature_eos_absolute_increase':.01,
            'sampling_runaway_increase':.08,'sampling_naturalness_drop':.08,'sampling_semantic_drop':.08,'sampling_japanese_validity_drop':.08},
        'eos':'Last up-to128 document tokens -> terminal EOS probability/rank/top1/5/10. Nonterminal EOS probability and premature argmax rate over all nonterminal LM targets at max512, equal-document means.',
        'prompts':prompts,'prompt_rule':'first min(64,N) confirmatory IDs in frozen numeric-page order, BOS plus first63 text tokens; checkpoint independent',
        'sampling_rng_bases':[550000,551000],'sampling_seed_rule':'base + prompt index; shared across model seeds and LR arms, independent CUDA generators',
        'decoder':SAMPLE_T07,'max_new_tokens':64,'prompt_batch_size':8,
        'sampling_metrics':'Existing PHASE51 metrics: completion, topic-token-overlap, Japanese validity, repetition1..4, runaway (budget exhaustion without EOS), EOS completion, naturalness/semantic automatic proxies; not human semantics or production validation.',
        'device':'cuda','dtype':'FP32','cpu_parallel_evaluation':False,'thermal':'cooldown<=65 before each model; pause to65 if monitored>=80; abort>=85 or hardware slowdown; no heavy CPU QA parallel',
        'new_training':False,'source_sha256':{str(p.relative_to(ROOT)):file_sha256(p) for p in [Path(__file__),ROOT/'evaluation/diagnose_foundation_v29_generation.py',ROOT/'evaluation/investigate_foundation_v14.py',ROOT/'evaluation/diagnose_foundation_v40.py',ROOT/'tokenizer/foundation-v11-base-4096.json']}})
    new_json(OUT/'confirmatory-freeze.json',{'sha256':file_sha256(SPEC),'frozen_at':now(),'scoring_started':False})
    print('CONFIRMATORY frozen',file_sha256(SPEC),flush=True)

def check_spec():
    gate();spec=read(SPEC);sha=file_sha256(SPEC);assert sha==read(OUT/'confirmatory-freeze.json')['sha256']
    for p,h in spec['source_sha256'].items():assert file_sha256(ROOT/p)==h,p
    assert file_sha256(Path(spec['confirmatory']['path']))==spec['confirmatory']['sha256']
    return spec,sha

def thermal_guard(monitor):
    if not monitor.samples:return
    s=monitor.samples[-1]
    assert s['gpu_temperature_c']<85 and not s['hardware_thermal_slowdown'],'THERMAL_STOP'
    if s['gpu_temperature_c']>=80:
        assert cooldown()['target_reached'],'COOLDOWN_FAILED'

@torch.inference_mode()
def document_metrics(model,ids,context,eos):
    sums={'ce':0.,'top1':0.,'top5':0.,'top10':0.,'nonterminal_eos_probability':0.,'premature_eos':0.};count=0;non=0
    for offset in range(0,len(ids)-1,128):
        end=min(offset+128,len(ids)-1);start=max(0,offset-(context-128))
        logits,_=model(torch.tensor([ids[start:end]],device='cuda'))
        scores=logits[0,offset-start:].float();y=torch.tensor(ids[offset+1:end+1],device='cuda');n=len(y)
        sums['ce']+=float(F.cross_entropy(scores,y,reduction='sum'))
        top=scores.topk(10,-1).indices
        for k in (1,5,10):sums[f'top{k}']+=float((top[:,:k]==y[:,None]).any(-1).sum())
        mask=y!=eos;non+=int(mask.sum());sums['premature_eos']+=float((top[:,0][mask]==eos).sum())
        sums['nonterminal_eos_probability']+=float(scores.softmax(-1)[mask,eos].sum());count+=n
    return {**{k:v/(non if k in ('premature_eos','nonterminal_eos_probability') else count) for k,v in sums.items()},'tokens':count}

@torch.inference_mode()
def terminal(model,ids,eos):
    logits,_=model(torch.tensor([ids[max(0,len(ids)-129):-1]],device='cuda'));s=logits[0,-1].float()
    rank=int((s>s[eos]).sum())+1
    return {'probability':float(s.softmax(-1)[eos]),'rank':rank,**{f'top{k}':rank<=k for k in (1,5,10)}}

def evaluate():
    spec,sha=check_spec();torch.set_num_threads(2);assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tok=FoundationTokenizer.load(ROOT/'tokenizer/foundation-v11-base-4096.json')
    docs=read(spec['confirmatory']['path']);encoded=[[tok.bos_id]+tok.encode(r['text'])+[tok.eos_id] for r in docs]
    if not (OUT/'confirmatory-consumption.json').exists():
        new_json(OUT/'confirmatory-consumption.json',{'consumed_for_model_selection':True,'state':'SCORING_STARTED','started_at':now(),'preregistration_sha256':sha,'never_blind_again':True})
    for identity in spec['checkpoints']:
        arm,seed=identity['arm'],identity['seed'];target=RAW/f'{arm}-{seed}.json'
        if target.exists():
            old=read(target);assert old['complete'] and old['preregistration_sha256']==sha
            print('Reuse completed inference',arm,seed,flush=True);continue
        assert file_sha256(Path(identity['path']))==identity['sha256']
        assert cooldown()['target_reached'];model=load_model(Path(identity['path']),torch.device('cuda'))
        monitor=Monitor();monitor.start();rows=[];sampling={}
        try:
            with torch.inference_mode():
                for i,ids in enumerate(encoded):
                    thermal_guard(monitor)
                    rows.append({'id':spec['confirmatory']['document_ids'][i],'full':document_metrics(model,ids,512,tok.eos_id),
                        'short':document_metrics(model,ids,128,tok.eos_id),'terminal':terminal(model,ids,tok.eos_id)})
                    if i%25==0:print(arm,seed,'documents',i,'/',len(encoded),flush=True)
                for base in spec['sampling_rng_bases']:
                    generated=[]
                    for start in range(0,len(spec['prompts']),8):
                        thermal_guard(monitor);ps=spec['prompts'][start:start+8]
                        generated.extend(generate_batch(model,tok,[p['prefix_ids'] for p in ps],spec['decoder'],list(range(base+start,base+start+len(ps))),spec['max_new_tokens']))
                    sampling[str(base)]={'metrics':metrics(generated,spec['prompts']),'rows':generated}
        finally:
            telemetry=monitor.finish();del model;gc.collect();torch.cuda.empty_cache()
        assert telemetry['samples']>0 and telemetry['gpu_temperature_c_max']<85 and not telemetry['hardware_thermal_slowdown']
        new_json(target,{'arm':arm,'seed':seed,'complete':True,'preregistration_sha256':sha,'checkpoint_sha256':identity['sha256'],
            'documents':rows,'sampling':sampling,'thermal':telemetry,'new_training':False})
        print('COMPLETE',arm,seed,'thermal max',telemetry['gpu_temperature_c_max'],flush=True)

def ci(values):
    v=np.asarray(values,float);rng=np.random.default_rng(550055)
    boot=np.mean(v[rng.integers(0,len(v),(10000,len(v)))],1)
    return {'mean':float(v.mean()),'lower':float(np.quantile(boot,.025)),'upper':float(np.quantile(boot,.975)),'documents_or_prompts':len(v)}

def summarize():
    spec,sha=check_spec();raw={(a,s):read(RAW/f'{a}-{s}.json') for a in ('C','B') for s in SEEDS}
    assert all(r['complete'] and r['preregistration_sha256']==sha for r in raw.values())
    def values(arm,group,key):return np.array([[r[group][key] for r in raw[arm,s]['documents']] for s in SEEDS])
    primary=ci((values('C','full','ce')-values('B','full','ce')).mean(0))
    selected='C' if primary['upper']<0 else 'B' if primary['lower']>0 else None
    metrics_out={};safety={};thresholds=spec['safety_thresholds']
    for arm in ('C','B'):
        byseed=[]
        for seed in SEEDS:
            r=raw[arm,seed];aggregated={g:{k:float(np.mean([d[g][k] for d in r['documents']])) for k in r['documents'][0][g] if k!='tokens'} for g in ('full','short','terminal')}
            aggregated['perplexity']=math.exp(aggregated['full']['ce'])
            aggregated['sampling']={k:float(np.mean([x['metrics'][k] for x in r['sampling'].values()])) for k in next(iter(r['sampling'].values()))['metrics']}
            byseed.append({'seed':seed,**aggregated})
        metrics_out[arm]={'by_seed':byseed,'full':{k:float(values(arm,'full',k).mean()) for k in ('ce','top1','top5','top10','nonterminal_eos_probability','premature_eos')},'short_ce':float(values(arm,'short','ce').mean()),'terminal_probability':float(values(arm,'terminal','probability').mean()),
            'sampling':{k:float(np.mean([r['sampling'][k] for r in byseed])) for k in byseed[0]['sampling']}}
    if selected:
        other='B' if selected=='C' else 'C'
        def test(name,delta,threshold):
            result=ci(delta);safety[name]={**result,'material_worsening_boundary':threshold,'clear_material_regression':result['lower']>threshold}
        test('context128_ce_increase',(values(selected,'short','ce')-values(other,'short','ce')).mean(0),thresholds['context128_ce_increase'])
        test('context_benefit_loss',((values(selected,'full','ce')-values(selected,'short','ce'))-(values(other,'full','ce')-values(other,'short','ce'))).mean(0),thresholds['context_benefit_loss'])
        # Relative EOS drop null: selected probability < .8*other; reversed as positive worsening.
        test('terminal_eos_relative_drop',((1-thresholds['terminal_eos_probability_relative_drop'])*values(other,'terminal','probability')-values(selected,'terminal','probability')).mean(0),0)
        test('premature_eos_absolute_increase',(values(selected,'full','premature_eos')-values(other,'full','premature_eos')).mean(0),thresholds['premature_eos_absolute_increase'])
        for field,label,sign in [('runaway','sampling_runaway_increase',1),('natural_japanese_proxy','sampling_naturalness_drop',-1),('semantic_coherence_proxy','sampling_semantic_drop',-1),('character_valid','sampling_japanese_validity_drop',-1)]:
            def sample(arm):return np.array([[[float(r[field]) for r in v['rows']] for v in raw[arm,s]['sampling'].values()] for s in SEEDS]).mean((0,1))
            test(label,sign*(sample(selected)-sample(other)),thresholds[label])
    state='FORMAL_LR_UNRESOLVED' if not selected else 'SAFETY_BLOCK' if any(r['clear_material_regression'] for r in safety.values()) else 'FORMAL_LR_APPROVED_5E5' if selected=='C' else 'FORMAL_LR_APPROVED_7_5E5'
    result={'phase':55,'confirmatory_gate':'VALID_COMPLETED','formal_lr_gate':state,'approved_lr':(5e-5 if selected=='C' else 7.5e-5) if state.startswith('FORMAL_LR_APPROVED') else None,
        'preregistration_sha256':sha,'consumed_for_model_selection':True,'primary_paired_ci':primary,
        'per_seed_ci':{str(s):ci(values('C','full','ce')[i]-values('B','full','ce')[i]) for i,s in enumerate(SEEDS)},'metrics':metrics_out,'safety':safety,
        'shared_generation_blocker':'GENERATION_POLICY_UNSAFE; relative research LR approval does not validate generation safety',
        'uncertainty':'One fresh lexical holdout, unequal category coverage, automatic proxies, fixed3 training seeds and limited64prompt power. No guarantee of no regression; confidence intervals reported.',
        'raw_artifacts':[{'path':str(RAW/f'{a}-{s}.json'),'sha256':file_sha256(RAW/f'{a}-{s}.json')} for a in ('C','B') for s in SEEDS],
        'new_training':False,'canonical':False,'20m':False,'foundation_base':False}
    new_json(ROOT/'evaluation/foundation-v44-confirmatory-summary.json',result)
    print(state,'primary',primary,flush=True)

if __name__=='__main__':{'freeze':freeze,'evaluate':evaluate,'summarize':summarize}[sys.argv[1]]()
