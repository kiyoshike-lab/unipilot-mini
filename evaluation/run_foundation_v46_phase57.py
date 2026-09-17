"""PHASE57 registered CUDA-only Control vs Arm A experiment.

This file consumes the byte-frozen PHASE56 registration.  It never edits that
registration, never reads sealed/Blind content, and only writes fresh Phase57
experimental outputs.  `run64` is the sole first-gate entry point.
"""
from __future__ import annotations
import argparse, gc, hashlib, json, math, os, re, shutil, subprocess, sys, time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import psutil
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.diagnose_foundation_v40 import read,new_json,file_sha256,resolver_gate,verify_payload
from evaluation.diagnose_foundation_v29_generation import document_ranges,generate_batch,load_model,loop_details,ngram_repetition
from evaluation.diagnose_foundation_v40 import eos_rows,metrics
from evaluation.evaluate_foundation_v33_context_gate import GREEDY,SAMPLE_T07
from evaluation.confirm_foundation_v44 import document_metrics,terminal
from evaluation.run_foundation_v45_observatory import normal_truth
from evaluation.generation_prefix_objective_v45 import cycle_negatives,generated_prefix_unlikelihood
from foundation.base_tokenizer import FoundationTokenizer
from foundation.diagnostic_transformer_v17 import DiagnosticConfigV17,DiagnosticTransformerV17
from training.checkpoint_paths import checkpoint_path,ensure_checkpoint_storage,existing_checkpoint_path
from training.foundation_v31_objective import weighted_lm_loss
from training.optimizer import create_optimizer
from training.run_foundation_v30_eos_experiment import load
from training.run_foundation_v35_thermal_gate import Monitor,cooldown,query_gpu
from training.train_foundation_v15_controlled import macro_batch
from training.train_foundation_v21_ab import file_sha256 as train_sha256,random_state

OUT=ROOT/'evaluation/phase57'; RAW=Path(r'Z:\AI\unipilot-mini\evaluation\phase57')
SPEC=ROOT/'evaluation/phase56/phase57-training-preregistration.json'
START_HEAD='e91bb8f8f8ad6966a9305c748e9684194db11783'
ROOT_EXPECTED=r'Z:\AI\unipilot-mini\checkpoints'
SPECIAL_NAMES=('<PAD>','<BOS>','<EOS>','<UNK>','<USER>','<ASSISTANT>','<SYSTEM>')

def now(): return datetime.now(timezone.utc).isoformat()
def git(*args): return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
def digest(value: str): return hashlib.sha256(value.encode('utf-8')).hexdigest()
def fingerprint(value):
    from training.run_foundation_v36_lr_review import fingerprint as inner
    return inner(value)
def require_new(path: Path):
    if path.exists() or path.with_suffix(path.suffix+'.tmp').exists(): raise FileExistsError(f'existing/partial artifact: {path}')
def atomic_json(path: Path,value):
    require_new(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('x',encoding='utf-8') as f: json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(path)
def raw_json(path:Path,value): atomic_json(path,value)
def source_spec():
    if os.environ.get('UNIPILOT_CHECKPOINT_ROOT')!=ROOT_EXPECTED: raise RuntimeError('PROCESS_ENV_RESOLVER_MISMATCH')
    resolver_gate();spec=read(SPEC);review=read(ROOT/'evaluation/phase56/training-design-review.json')
    if file_sha256(SPEC)!=review['preregistration_sha256']: raise RuntimeError('PREREGISTRATION_RECEIPT_MISMATCH')
    if spec['status']!='PHASE57_TRAINING_READY' or spec['seed']!=42 or spec['approved_lr']!=5e-5: raise RuntimeError('PREREGISTRATION_SCOPE_MISMATCH')
    for rel,sha in spec['source_hashes'].items():
        if file_sha256(ROOT/rel)!=sha: raise RuntimeError(f'PREREGISTRATION_SOURCE_CHANGED:{rel}')
    for item in [spec['root_evidence'],spec['training_data']['base'],spec['evaluation_sets']['generation'],spec['evaluation_sets']['validation'],spec['evaluation_sets']['frequency_population'],spec['evaluation_sets']['normal_controls'],spec['unit_test_evidence']]:
        if file_sha256(ROOT/item['path'])!=item['sha256']: raise RuntimeError(f'REGISTERED_INPUT_CHANGED:{item["path"]}')
    blind=ROOT/'data/foundation_v09/evaluation/final-blind-1000.json'
    if file_sha256(blind)!='fa7912d58ce251bb10b513f59793bb8ca6c0023b4fe08d1c040b8ccbfe49845b': raise RuntimeError('FINAL_BLIND_SHA_MISMATCH')
    return spec
def parent_path(spec):
    path=existing_checkpoint_path(ROOT,*Path(spec['parent_checkpoint']['relative_path']).parts)
    if not path.is_relative_to(Path(ROOT_EXPECTED)) or file_sha256(path)!=spec['parent_checkpoint']['sha256']: raise RuntimeError('PARENT_SHA_OR_PATH_MISMATCH')
    return path
def strict_payload(path,spec,tokens):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    check=verify_payload(payload,42,tokens,5e-5)
    if not check['pass'] or payload['tokens_processed']!=tokens or payload['seed']!=42: raise RuntimeError('STRICT_RESUME_INTEGRITY_FAIL')
    if payload['config']['vocab_size']!=4096 or payload['precision_mode']!='fp32': raise RuntimeError('ARCHITECTURE_OR_PRECISION_MISMATCH')
    return payload,check
def output_checkpoint(arm,total_updates):
    tokens=16_384_000+total_updates*512
    return checkpoint_path(ROOT,'experimental','phase57',f'arm-{arm}','seed-42',f'checkpoint-tokens-{tokens}.pt')
def raw_path(name): return RAW/name
def runtime_guard(monitor):
    if not monitor.samples:return
    sample=monitor.samples[-1]
    if sample['gpu_temperature_c']>=85 or sample['hardware_thermal_slowdown']: raise RuntimeError('THERMAL_STOP')
    if sample['gpu_temperature_c']>=80:
        cool=cooldown()
        if not cool['target_reached']: raise RuntimeError('THERMAL_COOLDOWN_FAILED')
def source_unchanged(path,sha):
    if file_sha256(path)!=sha: raise RuntimeError('PARENT_MUTATED')
def special_ids(tok): return {tok.special_to_id[x] for x in SPECIAL_NAMES if x in tok.special_to_id}

def preflight():
    spec=source_spec();require_new(OUT/'preflight.json')
    if OUT.exists(): raise RuntimeError('PHASE57_REPOSITORY_OUTPUT_ALREADY_EXISTS')
    if git('branch','--show-current')!='foundation-research': raise RuntimeError('PHASE57_PREFLIGHT_BLOCKED_BRANCH')
    # e91 was checked before this implementation commit; this exact fact is recorded rather than reinterpreting code commits as a bad experiment start.
    if subprocess.run(['git','merge-base','--is-ancestor',START_HEAD,'HEAD'],cwd=ROOT).returncode!=0: raise RuntimeError('AUTHORIZED_START_HEAD_NOT_ANCESTOR')
    parent=parent_path(spec);payload,integrity=strict_payload(parent,spec,16_384_000);tok=FoundationTokenizer.load(ROOT/'tokenizer/foundation-v11-base-4096.json')
    tok_path=ROOT/'tokenizer/foundation-v11-base-4096.json';data=ROOT/spec['training_data']['base']['path']
    if not torch.cuda.is_available() or torch.version.cuda is None: raise RuntimeError('CUDA_REQUIRED')
    gpu=torch.cuda.get_device_name(0)
    if gpu!='NVIDIA GeForce RTX 2070 SUPER': raise RuntimeError(f'GPU_MISMATCH:{gpu}')
    if not torch.cuda.is_available():raise RuntimeError('CUDA_REQUIRED')
    size=parent.stat().st_size;z=shutil.disk_usage(ROOT_EXPECTED);c=shutil.disk_usage('C:\\')
    if z.free<2*size+2*1024**3: raise RuntimeError('INSUFFICIENT_Z_DISK_FOR_ATOMIC_CHECKPOINTS')
    for path in [output_checkpoint('control',122),output_checkpoint('A',122),output_checkpoint('control',244),output_checkpoint('A',244),RAW/'cache-raw.json',RAW/'cache-manifest.json']:
        if path.exists() or path.with_suffix(path.suffix+'.tmp').exists(): raise RuntimeError(f'PHASE57_ARTIFACT_ALREADY_EXISTS:{path}')
    probe=query_gpu();cuda_props=torch.cuda.get_device_properties(0)
    receipt={'phase':57,'authorized_training':True,'at':now(),'branch':'foundation-research','authorized_start_head':START_HEAD,'implementation_head':git('rev-parse','HEAD'),'origin_at_initial_gate':START_HEAD,
      'resolver_gate':'PASS','checkpoint_root':ROOT_EXPECTED,'preregistration_path':str(SPEC.relative_to(ROOT)),'preregistration_sha256':file_sha256(SPEC),
      'parent_checkpoint':{'path':str(parent),'sha256':file_sha256(parent),'bytes':size,'integrity':integrity,'tokens_processed':payload['tokens_processed'],'metadata':{'seed':payload['seed'],'phase':payload.get('phase'),'experimental_lr':payload.get('experimental_lr'),'eos_weight':payload['eos_loss_weight'],'repetition_auxiliary':payload['repetition_auxiliary'],'parent':payload.get('source_sha256')}},
      'tokenizer':{'path':str(tok_path.relative_to(ROOT)),'sha256':file_sha256(tok_path),'vocab_size':tok.vocab_size},'training_data':{'path':str(data.relative_to(ROOT)),'sha256':file_sha256(data),'tokens':data.stat().st_size//2},
      'cuda':{'available':True,'torch':torch.__version__,'build':torch.version.cuda,'gpu':gpu,'vram_mib':cuda_props.total_memory/1048576,'start':probe},'ram':dict(psutil.virtual_memory()._asdict()),
      'free_bytes':{'C':c.free,'Z':z.free},'atomic_storage_requirement_bytes':2*size+2*1024**3,'objective_contract':'PENDING_TARGETED_TESTS','negative_sample_sufficiency':'PENDING_CACHE','copy_move_delete_overwrite_existing_checkpoint':[0,0,0,0],
      'final_blind':'SHA_ONLY_PASS','reserve2':'SEALED_UNSCORED_HASH_ONLY','confirmatory_rerun':False,'new_arms':0,'run_order':['control-64k','cooldown','A-64k']}
    del payload;gc.collect();atomic_json(OUT/'preflight.json',receipt);print('PHASE57 PREFLIGHT PASS',flush=True)

@torch.inference_mode()
def build_cache():
    spec=source_spec();pre=read(OUT/'preflight.json');require_new(RAW/'cache-raw.json');require_new(RAW/'cache-manifest.json')
    parent=parent_path(spec);source_unchanged(parent,spec['parent_checkpoint']['sha256']);tok=FoundationTokenizer.load(ROOT/'tokenizer/foundation-v11-base-4096.json')
    train=np.memmap(ROOT/spec['training_data']['base']['path'],dtype=np.uint16,mode='r');ranges=document_ranges(train,tok.bos_id,tok.eos_id);sp=special_ids(tok)
    eligible=[]
    for idx,(start,end) in enumerate(ranges):
        text=np.asarray(train[start:end],dtype=np.int64).tolist()
        if len(text)>=95 and not any(x in sp for x in text[:95]): eligible.append((digest(f'phase57-generated-prefix-v1|{idx}'),idx,text[:95]))
    eligible.sort();selected=eligible[:128]
    if len(selected)!=128:raise RuntimeError('INSUFFICIENT_TRAINING_DOCUMENTS')
    if not cooldown()['target_reached']:raise RuntimeError('THERMAL_COOLDOWN_FAILED')
    model=load_model(parent,torch.device('cuda'));monitor=Monitor();monitor.start();episodes=[]
    try:
        for rank,(_,idx,text) in enumerate(selected):
            runtime_guard(monitor);prefix=[tok.bos_id]+text[:63]
            # Greedy has no RNG effect; deterministic index is retained for an auditable call contract.
            row=generate_batch(model,tok,[prefix],GREEDY,[570000+idx],32,trace=False)[0]
            generated=[int(x) for x in row['ids']];reference=[int(x) for x in text[63:95]]
            negatives=[] if row['eos_reached'] or len(generated)!=32 else cycle_negatives(generated,reference,sp)
            episodes.append({'document_index':idx,'selection_sha256':digest(f'phase57-generated-prefix-v1|{idx}'),'prefix':prefix,'generated':generated,'reference':reference,'eos_reached':bool(row['eos_reached']),'negative_pairs':negatives})
            if rank%16==0: print('cache episode',rank,flush=True)
    finally:
        telemetry=monitor.finish();del model;gc.collect();torch.cuda.empty_cache()
    source_unchanged(parent,spec['parent_checkpoint']['sha256'])
    if not telemetry['samples'] or telemetry['gpu_temperature_c_max']>=85 or telemetry['hardware_thermal_slowdown']:raise RuntimeError('CACHE_THERMAL_FAIL')
    neg=sum(len(e['negative_pairs']) for e in episodes)
    raw={'phase':57,'kind':'training_only_generated_prefix_cache','parent_sha256':spec['parent_checkpoint']['sha256'],'tokenizer_sha256':file_sha256(ROOT/'tokenizer/foundation-v11-base-4096.json'),'episodes':episodes,'new_training':False}
    raw_json(RAW/'cache-raw.json',raw);manifest={'phase':57,'cache_raw_path':str(RAW/'cache-raw.json'),'cache_raw_sha256':file_sha256(RAW/'cache-raw.json'),'episodes':len(episodes),'eligible_episodes':sum(bool(e['negative_pairs']) for e in episodes),'negative_events':neg,'selection_rule':spec['training_data']['rollout_selection'],'eos_or_short_excluded':sum(not e['negative_pairs'] for e in episodes),'thermal':telemetry,'sufficiency':'PASS' if neg>=20 else 'INSUFFICIENT_TRAINING_NEGATIVES','new_training':False}
    raw_json(RAW/'cache-manifest.json',manifest)
    if neg<20: raise RuntimeError('INSUFFICIENT_TRAINING_NEGATIVES')
    pre['objective_contract']='PASS';pre['negative_sample_sufficiency']='PASS';pre['cache_manifest_sha256']=file_sha256(RAW/'cache-manifest.json');pre['cache_raw_sha256']=manifest['cache_raw_sha256']
    # preflight receipt uses a one-time append-only companion rather than overwriting its original record.
    atomic_json(OUT/'cache-preflight.json',pre);print('PHASE57 CACHE PASS',neg,flush=True)

def cache():
    manifest=read(RAW/'cache-manifest.json');raw=read(RAW/'cache-raw.json')
    if file_sha256(RAW/'cache-raw.json')!=manifest['cache_raw_sha256'] or manifest['negative_events']<20: raise RuntimeError('CACHE_INTEGRITY_OR_SUFFICIENCY_FAIL')
    return raw['episodes'],manifest
def save_checkpoint(target,payload,model,optimizer,spec,source_sha,end_update,arm,training):
    require_new(target);ensure_checkpoint_storage(target);target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_suffix('.pt.tmp');require_new(tmp)
    saved={**payload,'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),'scheduler_state':{**payload['scheduler_state'],'global_step':end_update},'random_state':random_state('cuda'),'update':end_update,'tokens_processed':end_update*512,
      'phase':57,'arm':arm,'experimental':True,'formal_research':False,'promoted':False,'canonical':False,'not_canonical':True,'not_production':True,'precision_mode':'fp32','eos_loss_weight':1.5,'repetition_auxiliary':False,
      'generated_prefix_contiguous_cycle_unlikelihood':arm=='A','generated_prefix_coefficient':(.05 if arm=='A' else 0.0),'source_checkpoint_sha256':source_sha,'parent_checkpoint_sha256':spec['parent_checkpoint']['sha256'],'phase57_training':training}
    torch.save(saved,tmp);loaded=torch.load(tmp,map_location='cpu',weights_only=False);integrity=verify_payload(loaded,42,end_update*512,5e-5)
    strict=DiagnosticTransformerV17(DiagnosticConfigV17(**loaded['config']));strict.load_state_dict(loaded['model_state'],strict=True);opt=create_optimizer(strict,5e-5,.1);opt.load_state_dict(loaded['optimizer_state'])
    extra={'markers':all(loaded[k] is v for k,v in {'experimental':True,'formal_research':False,'promoted':False,'canonical':False,'not_canonical':True,'not_production':True}.items()),'phase':loaded['phase']==57,'arm':loaded['arm']==arm,'parent_sha':loaded['parent_checkpoint_sha256']==spec['parent_checkpoint']['sha256'],'source_sha':loaded['source_checkpoint_sha256']==source_sha}
    if not integrity['pass'] or not all(extra.values()):raise RuntimeError(f'OUTPUT_STRICT_RELOAD_FAIL:{extra}')
    tmp.replace(target);return {'path':str(target),'sha256':file_sha256(target),'integrity':integrity,'strict_model_reload':True,'strict_optimizer_reload':True,'metadata':extra}

def train_arm(arm,total_updates):
    spec=source_spec();episodes,manifest=cache();first=total_updates==122;source=parent_path(spec) if first else output_checkpoint(arm,122);source_sha=file_sha256(source)
    payload,integrity=strict_payload(source,spec,16_384_000 if first else 16_384_000+122*512)
    if not first and (payload.get('phase')!=57 or payload.get('arm')!=arm or payload.get('parent_checkpoint_sha256')!=spec['parent_checkpoint']['sha256']):raise RuntimeError('RESUME_LINEAGE_MISMATCH')
    start=int(payload['update']);end=32000+total_updates
    if end<=start:raise RuntimeError('NON_FORWARD_TRAINING_BUDGET')
    target=output_checkpoint(arm,total_updates);require_new(target);cool=cooldown()
    if not cool['target_reached']:raise RuntimeError('THERMAL_COOLDOWN_FAILED')
    model_payload,model,optimizer=load(source,torch.device('cuda'));assert model_payload['update']==start
    if not all(g['lr']==5e-5 for g in optimizer.param_groups):raise RuntimeError('APPROVED_LR_MISMATCH')
    continuity={k:fingerprint(payload[k]) for k in ('model_state','optimizer_state','scheduler_state','permutation','random_state')};assert fingerprint(optimizer.state_dict())==continuity['optimizer_state']
    tok=FoundationTokenizer.load(ROOT/'tokenizer/foundation-v11-base-4096.json');data=np.memmap(ROOT/spec['training_data']['base']['path'],dtype=np.uint16,mode='r')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;model.train();torch.cuda.reset_peak_memory_stats();monitor=Monitor();monitor.start();stats=[];over10=0;aux_high=0;started=time.perf_counter()
    try:
      for update in range(start+1,end+1):
        runtime_guard(monitor);x,y=macro_batch(data,int(payload['permutation'][update-1]),512);x,y=x.cuda(),y.cuda();optimizer.zero_grad(set_to_none=True);logits,_=model(x);lm,eos,non=weighted_lm_loss(logits,y,tok.eos_id,1.5);aux=lm.detach()*0.;total=lm;negative_count=0
        if update%8==0:
          episode=episodes[(update//8-1)%len(episodes)];negative_count=len(episode['negative_pairs']);aux_x=torch.tensor([episode['prefix']+episode['generated'][:31]],device='cuda');model.eval()
          if arm=='A':
            aux_logits,_=model(aux_x);aux=generated_prefix_unlikelihood(aux_logits[0,63:95],episode['negative_pairs']);total=lm+.05*aux
          else:
            with torch.no_grad(): model(aux_x)
          model.train()
          ratio=float((.05*aux/lm.detach().clamp_min(1e-12)).detach()) if arm=='A' else 0.;aux_high=aux_high+1 if ratio>.25 else 0
          if aux_high>=3:raise RuntimeError('AUXILIARY_CONTRIBUTION_STOP')
        if not torch.isfinite(total):raise RuntimeError('NONFINITE_LOSS')
        total.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.0));
        if not math.isfinite(norm):raise RuntimeError('NONFINITE_GRADIENT')
        over10=over10+1 if norm>10 else 0
        if norm>100 or over10>=3:raise RuntimeError('GRADIENT_STOP')
        optimizer.step();stats.append({'update':update,'lm_loss':float(lm.detach()),'eos_loss':float(eos.detach()),'non_eos_loss':float(non.detach()),'aux_loss':float(aux.detach()),'total_loss':float(total.detach()),'aux_negative_count':negative_count,'gradient_norm':norm})
      torch.cuda.synchronize()
    finally:
      elapsed=time.perf_counter()-started;telemetry=monitor.finish()
    if telemetry['gpu_temperature_c_max']>=85 or telemetry['hardware_thermal_slowdown']:raise RuntimeError('THERMAL_STOP')
    source_unchanged(source,source_sha);checkpoint=save_checkpoint(target,payload,model,optimizer,spec,source_sha,end,arm,{'stage':'64k' if first else '128k','stats_count':len(stats),'cache_sha256':manifest['cache_raw_sha256']});source_unchanged(parent_path(spec),spec['parent_checkpoint']['sha256'])
    result={'phase':57,'arm':arm,'seed':42,'stage':'64k' if first else '128k','start_update':start,'end_update':end,'start_processed_tokens':start*512,'end_processed_tokens':end*512,'lm_tokens':(end-start)*512,'auxiliary_positions':((end//8)-(start//8))*96,'total_gradient_input_tokens':(end-start)*512+((end//8)-(start//8))*96,'coefficient':(.05 if arm=='A' else 0.),'objective':'generated-prefix contiguous-cycle UL with reference veto' if arm=='A' else 'weighted LM control','parent_sha256':spec['parent_checkpoint']['sha256'],'source_sha256':source_sha,'continuity':continuity,'resume_integrity':integrity,'cache_manifest_sha256':file_sha256(RAW/'cache-manifest.json'),'checkpoint':checkpoint,'cooldown':cool,'training':{'seconds':elapsed,'tokens_per_second':((end-start)*512)/elapsed,'mean_lm_loss':float(np.mean([x['lm_loss'] for x in stats])),'mean_auxiliary_loss':float(np.mean([x['aux_loss'] for x in stats])),'mean_total_loss':float(np.mean([x['total_loss'] for x in stats])),'mean_gradient_norm':float(np.mean([x['gradient_norm'] for x in stats])),'max_gradient_norm':float(np.max([x['gradient_norm'] for x in stats])),'peak_vram_mib':torch.cuda.max_memory_allocated()/1048576,'telemetry':telemetry,'all_finite':True},'stats':stats,'new_training':True,'canonical':False,'20m':False}
    raw_json(raw_path(f'{arm}-{result["stage"]}-training.json'),result);print('TRAIN COMPLETE',arm,result['stage'],result['training']['tokens_per_second'],flush=True)

def frequency_values(model,val,pop):
    positions=np.asarray(pop['positions']);result={k:[] for k in ('ce','probabilities','top1','top5','top10')}
    for block in np.unique((positions-1)//512):
        start=int(block)*512;size=min(512,len(val)-start-1);x=torch.tensor(np.asarray(val[start:start+size],dtype=np.int64),device='cuda')[None];logits,_=model(x);mask=positions[(positions>start)&(positions<=start+size)];rows=logits[0,torch.tensor(mask-start-1,device='cuda')].float();truth=torch.tensor(np.asarray(val[mask],dtype=np.int64),device='cuda');logp=rows.log_softmax(-1).gather(1,truth[:,None]).squeeze(1);top=rows.topk(10,-1).indices;result['ce'].extend((-logp).cpu().tolist());result['probabilities'].extend(logp.exp().cpu().tolist())
        for k in (1,5,10):result[f'top{k}'].extend((top[:,:k]==truth[:,None]).any(-1).cpu().tolist())
    return result
def frequency_summary(values,pop):
    ce=np.asarray(values['ce']);ids=np.asarray(pop['token_ids']);p=np.asarray(values['probabilities']);return {'micro_ce':float(ce.mean()),'macro_per_token_ce':float(np.mean([ce[ids==x].mean() for x in np.unique(ids)])),'top1':float(np.mean(values['top1'])),'top5':float(np.mean(values['top5'])),'top10':float(np.mean(values['top10'])),'mean_probability':float(p.mean()),'token_count':pop['token_count'],'occurrence_count':pop['occurrence_count'],'document_count':pop['document_count']}
def ci_delta(after,before,docs,seed=5701):
    d=np.asarray(after,float)-np.asarray(before,float);_,inv=np.unique(docs,return_inverse=True);s=np.bincount(inv,weights=d);c=np.bincount(inv);rng=np.random.default_rng(seed);pick=rng.integers(0,len(s),(10000,len(s)));boot=s[pick].sum(1)/c[pick].sum(1);return {'mean':float(d.mean()),'lower':float(np.quantile(boot,.025)),'upper':float(np.quantile(boot,.975))}
@torch.inference_mode()
def full_validation(model,val):
    count=loss=0.;correct={1:0,5:0,10:0}
    for start in range(0,len(val)-1,512):
        size=min(512,len(val)-start-1);x=torch.tensor(np.asarray(val[start:start+size],dtype=np.int64),device='cuda')[None];y=torch.tensor(np.asarray(val[start+1:start+size+1],dtype=np.int64),device='cuda')[None];z,_=model(x);loss+=float(F.cross_entropy(z.flatten(0,1),y.flatten(),reduction='sum'));top=z.topk(10,-1).indices;count+=size
        for k in correct:correct[k]+=int((top[...,:k]==y[...,None]).any(-1).sum())
    return {'tokens':int(count),'ce':loss/count,'top1':correct[1]/count,'top5':correct[5]/count,'top10':correct[10]/count}
def normal_metrics(model,tok,controls):
    rows=[]
    for p in controls:
        x=[tok.bos_id]+tok.encode(p['prompt']);y=tok.encode(normal_truth(p['family'],p['n']))+[tok.eos_id];ids=x+y;z,_=model(torch.tensor([ids[:-1]],device='cuda'));scores=z[0,len(x)-1:len(x)-1+len(y)].float();truth=torch.tensor(y,device='cuda');ce=float(F.cross_entropy(scores,truth));eos=float(scores[-1].softmax(-1)[tok.eos_id]);rows.append({'id':p['id'],'family':p['family'],'ce':ce,'terminal_eos_probability':eos})
    generated=generate_batch(model,tok,[[tok.bos_id]+tok.encode(p['prompt']) for p in controls],GREEDY,list(range(len(controls))),64,trace=False)
    def shape(row,p):
        text=row['text'];family=p['family'];n=p['n']
        if family=='math': return bool(re.search(re.escape('+'.join(['1']*n))+r'\\s*[=＝]\\s*'+str(n)+r'(?!\\d)',re.sub(r'\\s+','',text)))
        if family=='code': return [int(x) for x in re.findall(r'print\\((\\d+)\\)',text)]==list(range(1,n+1))
        if family=='lists': return text.count('確認済み')==n and all(re.search(rf'(?m)^\\s*{j}[.、)．]',text) for j in range(1,n+1))
        if family=='definitions': return len(re.findall('定義[:：]',text))==n and all(t in text for t in ['変数','関数','集合','写像','命題'][:n])
        return text.count('標本平均')==n and len(set(re.split('[。！？]',text))- {''})>=n
    for row,p in zip(rows,controls): row['shape_proxy']=shape(generated[controls.index(p)],p);row['shape_not_semantic_correctness']=True
    families={f:float(np.mean([r['ce'] for r in rows if r['family']==f])) for f in sorted({r['family'] for r in rows})};return {'mean_ce':float(np.mean([r['ce'] for r in rows])),'family_ce':families,'terminal_eos_probability':float(np.mean([r['terminal_eos_probability'] for r in rows])),'shape_proxy_passes':sum(r['shape_proxy'] for r in rows),'shape_proxy_examples':len(rows),'shape_not_semantic_correctness':True,'rows':rows}
def context_metrics(model,tok,ids):
    return {str(c):{'mean_ce':float(np.mean([document_metrics(model,x,c,tok.eos_id)['ce'] for x in ids])),'documents':len(ids)} for c in (128,512)}
def run_generation(model,tok,prompts,spec):
    prefix=[p['prefix_ids'] for p in prompts];sampling={}
    for base in spec['sampling_rng_bases']:
        rows=generate_batch(model,tok,prefix,SAMPLE_T07,[base+i for i in range(len(prefix))],64,trace=False);sampling[str(base)]={'metrics':metrics(rows,prompts),'rows':rows}
    greedy=generate_batch(model,tok,prefix,GREEDY,list(range(len(prefix))),128,trace=True)
    return {'sampling':sampling,'greedy':{'metrics':metrics(greedy,prompts),'rows':greedy}}
def evaluate_model(label,path):
    spec=source_spec();target=raw_path(f'{label}-evaluation.json');require_new(target);sha=file_sha256(path);tok=FoundationTokenizer.load(ROOT/'tokenizer/foundation-v11-base-4096.json');val=np.memmap(ROOT/spec['evaluation_sets']['validation']['path'],dtype=np.uint16,mode='r');freq=read(ROOT/spec['evaluation_sets']['frequency_population']['path']);phase51=read(ROOT/spec['evaluation_sets']['generation']['path']);obs=read(ROOT/'evaluation/phase56/observability-preregistration.json');docs=read(obs['diagnostic_source']['path']);lookup={f"jawiki:{x['page_id']}:{x['revision_id']}":x for x in docs};context_ids=[[tok.bos_id]+tok.encode(lookup[i]['text'])+[tok.eos_id] for i in obs['selected_document_ids']];controls=read(ROOT/spec['evaluation_sets']['normal_controls']['path'])['prompts']
    if not cooldown()['target_reached']:raise RuntimeError('THERMAL_COOLDOWN_FAILED')
    model=load_model(path,torch.device('cuda'));monitor=Monitor();monitor.start()
    try:
        validation=full_validation(model,val);runtime_guard(monitor);core=frequency_values(model,val,freq['core']);runtime_guard(monitor);tail=frequency_values(model,val,freq['population']);runtime_guard(monitor);term=eos_rows(model,tok,val,phase51['terminal_positions']);non=eos_rows(model,tok,val,phase51['nonterminal_positions']);runtime_guard(monitor);context=context_metrics(model,tok,context_ids);runtime_guard(monitor);normal=normal_metrics(model,tok,controls);runtime_guard(monitor);generation=run_generation(model,tok,phase51['prompts'],phase51)
    finally:
        telemetry=monitor.finish();del model;gc.collect();torch.cuda.empty_cache()
    if telemetry['gpu_temperature_c_max']>=85 or telemetry['hardware_thermal_slowdown']:raise RuntimeError('THERMAL_STOP')
    source_unchanged(path,sha);result={'phase':57,'label':label,'checkpoint_sha256':sha,'validation':validation,'core':{'population_sha256':freq['core']['sha256'],'metrics':frequency_summary(core,freq['core']),'values':core},'supported_tail':{'population_sha256':freq['population']['sha256'],'metrics':frequency_summary(tail,freq['population']),'values':tail},'eos':{'terminal_mean_probability':float(np.mean([x['probability'] for x in term])),'terminal_top1':float(np.mean([x['top1'] for x in term])),'nonterminal_mean_probability':float(np.mean([x['probability'] for x in non])),'premature_argmax_eos':float(np.mean([x['top1'] for x in non])),'terminal':term,'nonterminal':non},'context':context,'normal_controls':normal,'generation':generation,'thermal':telemetry,'new_training':False}
    raw_json(target,result);return result
def safe_div(num,den):
    if not math.isfinite(num) or not math.isfinite(den) or den<=0: return None
    return num/den
def safety(a,b,parent):
    spec=source_spec()['quality_safeguards_vs_each_parent_and_control'];out={}
    for name,other in [('parent',parent),('control',b)]:
        fails=[]
        def le(label,value,limit):
            passed=math.isfinite(value) and value<=limit;out[f'{name}_{label}']={'delta':value,'limit':limit,'pass':passed};fails.append(not passed)
        def ge(label,value,limit):
            passed=value is not None and math.isfinite(value) and value>=limit;out[f'{name}_{label}']={'value':value,'limit':limit,'pass':passed};fails.append(not passed)
        le('validation_ce',a['validation']['ce']-other['validation']['ce'],spec['validation_ce_increase_max']);le('validation_top1_drop',other['validation']['top1']-a['validation']['top1'],spec['validation_top1_drop_absolute_max']);le('validation_top5_drop',other['validation']['top5']-a['validation']['top5'],spec['validation_top5_top10_drop_absolute_max']);le('validation_top10_drop',other['validation']['top10']-a['validation']['top10'],spec['validation_top5_top10_drop_absolute_max'])
        for c in ('128','512'):le(f'context_{c}_ce',a['context'][c]['mean_ce']-other['context'][c]['mean_ce'],spec['context_ce_increase_each_128_512_max'])
        le('context_long_benefit_loss',(a['context']['512']['mean_ce']-a['context']['128']['mean_ce'])-(other['context']['512']['mean_ce']-other['context']['128']['mean_ce']),spec['context_long_benefit_loss_max']);ge('terminal_eos_ratio',safe_div(a['eos']['terminal_mean_probability'],other['eos']['terminal_mean_probability']),spec['terminal_mean_eos_probability_ratio_min']);le('terminal_eos_top1_drop',other['eos']['terminal_top1']-a['eos']['terminal_top1'],spec['terminal_eos_top1_drop_absolute_max']);le('nonterminal_eos_increase',a['eos']['nonterminal_mean_probability']-other['eos']['nonterminal_mean_probability'],spec['nonterminal_eos_probability_increase_max']);le('premature_eos_increase',a['eos']['premature_argmax_eos']-other['eos']['premature_argmax_eos'],spec['premature_argmax_eos_increase_max'])
        for group,limit in [('core',spec['core_micro_macro_ce_increase_max']),('supported_tail',spec['supported_tail_micro_macro_ce_increase_max'])]:
            for field in ('micro_ce','macro_per_token_ce'):le(f'{group}_{field}',a[group]['metrics'][field]-other[group]['metrics'][field],limit)
            for field,limit in [('top1',spec['frequency_top1_drop_absolute_max']),('top5',spec['frequency_top5_top10_drop_absolute_max']),('top10',spec['frequency_top5_top10_drop_absolute_max'])]:le(f'{group}_{field}_drop',other[group]['metrics'][field]-a[group]['metrics'][field],limit)
            ci=ci_delta(a[group]['values']['ce'],other[group]['values']['ce'],freq_docs(group),5701);out[f'{name}_{group}_ce_ci95']=ci;fails.append(ci['upper']>spec['frequency_ce_paired_ci95_upper_max'][('core' if group=='core' else 'supported_tail')])
        le('normal_mean_ce',a['normal_controls']['mean_ce']-other['normal_controls']['mean_ce'],spec['normal_mean_ce_increase_max']);ge('normal_terminal_eos_ratio',safe_div(a['normal_controls']['terminal_eos_probability'],other['normal_controls']['terminal_eos_probability']),spec['normal_terminal_eos_probability_ratio_min'])
        for f in a['normal_controls']['family_ce']:le(f'normal_{f}_ce',a['normal_controls']['family_ce'][f]-other['normal_controls']['family_ce'][f],spec['normal_each_family_ce_increase_max'])
        # automatic proxy/validity across all frozen sampling rows
        for metric,limit in [('natural_japanese_proxy',spec['sampling_naturalness_semantic_proxy_drop_max']),('semantic_coherence_proxy',spec['sampling_naturalness_semantic_proxy_drop_max']),('topic_retention_proxy',spec['sampling_topic_retention_drop_max']),('japanese_validity',spec['japanese_validity_drop_max'])]:le(f'sampling_{metric}_drop',generation_metric(other,metric)-generation_metric(a,metric),limit)
        out[f'{name}_all_pass']=not any(fails)
    return out
def freq_docs(group):return read(ROOT/'evaluation/foundation-v39-supported-tail.json')['core' if group=='core' else 'population']['document_ids']
def generation_metric(result,key):
    if key in result['generation']['greedy']['metrics']:return result['generation']['greedy']['metrics'][key]
    return float(np.mean([v['metrics'].get(key,0.) for v in result['generation']['sampling'].values()]))
def runaway(result,kind):
    if kind=='greedy':return result['generation']['greedy']['metrics']['runaway_rate']
    return {b:x['metrics']['runaway_rate'] for b,x in result['generation']['sampling'].items()}
def compact_evaluation(row):
    return {'phase':row['phase'],'label':row['label'],'checkpoint_sha256':row['checkpoint_sha256'],'validation':row['validation'],
      'core':{'population_sha256':row['core']['population_sha256'],'metrics':row['core']['metrics']},'supported_tail':{'population_sha256':row['supported_tail']['population_sha256'],'metrics':row['supported_tail']['metrics']},
      'eos':{k:v for k,v in row['eos'].items() if k not in ('terminal','nonterminal')},'context':row['context'],
      'normal_controls':{k:v for k,v in row['normal_controls'].items() if k!='rows'},
      'generation':{'greedy':row['generation']['greedy']['metrics'],'sampling':{k:v['metrics'] for k,v in row['generation']['sampling'].items()}},'thermal':row['thermal'],'raw_only':True}
def compact_training(row):
    return {k:v for k,v in row.items() if k!='stats'}
def existing_evaluation(label,path):
    target=raw_path(f'{label}-evaluation.json')
    if target.exists():
        value=read(target)
        if value['checkpoint_sha256']!=file_sha256(path): raise RuntimeError('EVALUATION_CHECKPOINT_HASH_MISMATCH')
        return value
    return evaluate_model(label,path)
def gate_stage(total_updates):
    spec=source_spec();stage='64k' if total_updates==122 else '128k';gate_path=OUT/f'gate-{stage}.json';require_new(gate_path)
    parent=existing_evaluation('parent',parent_path(spec));control=existing_evaluation(f'control-{stage}',output_checkpoint('control',total_updates));arm=existing_evaluation(f'arm-a-{stage}',output_checkpoint('A',total_updates))
    s=safety(arm,control,parent);control_safety=safety(control,parent,parent);g=runaway(arm,'greedy');samples=runaway(arm,'sampling');cg=runaway(control,'greedy');cs=runaway(control,'sampling');pg=runaway(parent,'greedy');ps=runaway(parent,'sampling');all_safety=s['parent_all_pass'] and s['control_all_pass'];control_drift=not control_safety['parent_all_pass'];sampling_mean=float(np.mean(list(samples.values())));control_sampling_mean=float(np.mean(list(cs.values())));parent_sampling_mean=float(np.mean(list(ps.values())))
    full_success=g<=.5 and all(x<=.5 for x in samples.values()) and sampling_mean<=.5 and (cg-g)>=.1 and (control_sampling_mean-sampling_mean)>=.1
    extension=all_safety and g<=.75 and sampling_mean<=.75 and max(pg-g,parent_sampling_mean-sampling_mean)>=.1
    if control_drift: gate='CONTROL_DRIFT_REVIEW_REQUIRED'
    elif not all_safety: gate=f'ARM_A_{stage.upper()}_REJECT'
    elif full_success: gate=f'ARM_A_{stage.upper()}_PASS'
    elif total_updates==122 and extension: gate='ARM_A_64K_CONTINUE_TO_128K'
    else: gate=f'ARM_A_{stage.upper()}_INSUFFICIENT'
    result={'phase':57,'stage':stage,'gate':gate,'arm_a_success':full_success,'control_drift':control_drift,'safety':s,'control_vs_parent_safety':control_safety,'generation':{'parent':{'greedy':pg,'sampling':ps},'control':{'greedy':cg,'sampling':cs},'arm_a':{'greedy':g,'sampling':samples},'attribution_reduction_control_minus_a':{'greedy':cg-g,'sampling_mean':control_sampling_mean-sampling_mean},'parent_reduction_arm_a':{'greedy':pg-g,'sampling_mean':parent_sampling_mean-sampling_mean}},'extension_authorized':total_updates==122 and extension and not full_success,'reason':'Registered thresholds only; no post-hoc extension.','new_training':True,'canonical':False,'20m':False}
    atomic_json(gate_path,result)
    for name,row in [(f'control-{stage}-summary.json',control),(f'arm-a-{stage}-summary.json',arm)]:
        arm_id='control' if name.startswith('control') else 'A';atomic_json(OUT/name,{'phase':57,'evaluation':compact_evaluation(row),'training':compact_training(read(raw_path(f'{arm_id}-{stage}-training.json'))),'gate':gate})
    print('PHASE57',stage.upper(),'GATE',gate,flush=True);return result
def gate64(): return gate_stage(122)
def gate128(): return gate_stage(244)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('preflight','cache','train64','gate64','train128','gate128'));args=parser.parse_args();torch.set_num_threads(2)
    if args.action=='preflight':preflight()
    elif args.action=='cache':build_cache()
    elif args.action=='train64':
        train_arm('control',122);cool=cooldown();
        if not cool['target_reached']:raise RuntimeError('INTER_ARM_COOLDOWN_FAILED')
        train_arm('A',122)
    elif args.action=='gate64':gate64()
    elif args.action=='train128':
        gate=read(OUT/'gate-64k.json')
        if gate['gate']!='ARM_A_64K_CONTINUE_TO_128K': raise RuntimeError('128K_NOT_AUTHORIZED_BY_64K_GATE')
        train_arm('control',244);cool=cooldown()
        if not cool['target_reached']:raise RuntimeError('INTER_ARM_COOLDOWN_FAILED')
        train_arm('A',244)
    elif args.action=='gate128':gate128()
if __name__=='__main__':main()
