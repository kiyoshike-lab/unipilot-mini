"""PHASE61: frozen two-arm CUDA continuation stability study.

The only intentional difference between a same-seed control and half-LR arm is
the runtime optimizer LR.  All artifacts are exclusive-create; this runner has
no retry, extension, promotion, or alternate-LR command.
"""
from __future__ import annotations

import argparse, copy, gc, hashlib, json, math, os, shutil, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import generation_contract_v47 as contract
from evaluation import run_foundation_v46_phase57 as measure
from evaluation.diagnose_foundation_v29_generation import generate_batch, load_model
from evaluation.diagnose_foundation_v40 import eos_rows, metrics
from evaluation.evaluate_foundation_v33_context_gate import GREEDY, SAMPLE_T07
from evaluation.register_foundation_v49 import phase61_decision
from foundation.base_tokenizer import FoundationTokenizer
from foundation.diagnostic_transformer_v17 import DiagnosticConfigV17, DiagnosticTransformerV17
from training.checkpoint_paths import checkpoint_path, checkpoint_root, ensure_checkpoint_storage, existing_checkpoint_path
from training.foundation_v31_objective import weighted_lm_loss
from training.optimizer import create_optimizer
from training.run_foundation_v30_eos_experiment import load
from training.run_foundation_v35_thermal_gate import Monitor, cooldown, query_gpu
from training.run_foundation_v36_lr_review import fingerprint, verify_payload
from training.train_foundation_v15_controlled import macro_batch
from training.train_foundation_v21_ab import random_state, restore_random_state

OUT = ROOT / 'evaluation/phase61'
RAW = Path(r'Z:\AI\unipilot-mini\evaluation\phase61')
SPEC = ROOT / 'evaluation/phase60/phase61-continuation-stability-preregistration.json'
ZROOT = Path(r'Z:\AI\unipilot-mini\checkpoints')
START = '4cf502e45a9f09166b4f0b39f56b557e10b317e5'
SPEC_SHA = '1f21f56d345e34a2aa58df7180ed40f5b3071feacab75cd31385b7fd2ffa89fc'
CACHE = Path(r'Z:\AI\unipilot-mini\evaluation\phase57\cache-raw.json')
CACHE_SHA = '4978d51cb66326bccee71a966d6337555387ce74f5ac2d4415506076a69416dc'
SEEDS = (42, 123, 2026)
ARMS = (('control', 5e-5), ('half-lr', 2.5e-5))
PARENT_SHA = {42:'a55369c0e98779839750d727517ce6621bc40b5746f975a96bd9cbc0574db2e8',123:'78df96b70016dda180f4529833d904e2d448e11469a87278d4881ed09c93dc20',2026:'7dcf5fa2da58777040cf9c03f1f513d707e6866f2eaf237bf203bd1b0135f069'}
PERM_SHA = {42:'58551fa0110f3221caa72cd1382ce6e5ca05a0757bd88776071b5efeddb426f8',123:'685d76f443d25bd4d271e1550a174e0b6380ad5b8d47020607e12947eb3440c9',2026:'8e6cd015ea3e49605306d2be18a67e6c95b1126bcfdf82eb4e25bdda4efc136c'}

def now(): return datetime.now(timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''): h.update(chunk)
    return h.hexdigest()
def git(*args): return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
def new(path):
    p=Path(path)
    if p.exists() or p.with_suffix(p.suffix+'.tmp').exists(): raise FileExistsError('existing or partial artifact: '+str(p))
def emit(path, value):
    p=Path(path); new(p); p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('x',encoding='utf8') as f: json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(p)
def finite(value):
    if torch.is_tensor(value): return bool(torch.isfinite(value).all())
    if isinstance(value,dict): return all(finite(x) for x in value.values())
    if isinstance(value,(list,tuple)): return all(finite(x) for x in value)
    return True
def parent(seed): return existing_checkpoint_path(ROOT,'experimental','phase48','arm-C',f'seed-{seed}','checkpoint-tokens-16384000.pt')
def target(seed,arm): return checkpoint_path(ROOT,'experimental','phase61','continuation-stability',arm,f'seed-{seed}','checkpoint-tokens-16446464.pt')
def run_artifact(seed,arm): return OUT/f'run-seed{seed}-{arm}.json'
def root_gate():
    if os.environ.get('UNIPILOT_CHECKPOINT_ROOT') != str(ZROOT) or checkpoint_root(ROOT) != ZROOT: raise RuntimeError('PROCESS_ENV_RESOLVER_MISMATCH')
def source_spec():
    root_gate(); spec=read(SPEC)
    if sha(SPEC)!=SPEC_SHA or spec['status']!='REGISTERED_REQUIRES_NEW_USER_TRAINING_AUTHORIZATION' or spec['training_authorized'] is not False or spec['seeds']!=list(SEEDS): raise RuntimeError('PREREGISTRATION_INTEGRITY_FAIL')
    if [(x['id'],x['lr']) for x in spec['arms']] != list(ARMS): raise RuntimeError('PREREGISTRATION_SCOPE_MISMATCH')
    if spec['evaluation']['schema_sha256']!='6576428940779e2908a8aacc67c4fb6bd2ca8bdc945e6202b88d5116ec701c6b' or len(spec['evaluation']['all_34_mappings'])!=34: raise RuntimeError('EVALUATOR_CONTRACT_FAIL')
    if spec['evaluation']['all_34_mappings'] != contract.mapping(spec['evaluation']['all_thresholds']): raise RuntimeError('EVALUATOR_SCHEMA_CODE_MISMATCH')
    for rel,digest in spec['source_sha256'].items():
        if sha(ROOT/rel)!=digest: raise RuntimeError('REGISTERED_SOURCE_CHANGED:'+rel)
    return spec
def strict_parent(seed):
    p=parent(seed)
    if not p.is_relative_to(ZROOT) or sha(p)!=PARENT_SHA[seed]: raise RuntimeError('PARENT_SHA_OR_PATH_MISMATCH')
    payload=torch.load(p,map_location='cpu',weights_only=False)
    integrity=verify_payload(payload,seed,16_384_000,5e-5)
    if fingerprint(payload['permutation'][32000:32122])!=PERM_SHA[seed]: raise RuntimeError('PERMUTATION_HASH_MISMATCH')
    restore_random_state(payload['random_state'],'cuda',cuda_seed=seed)
    if fingerprint(random_state('cuda')) != fingerprint(payload['random_state']): raise RuntimeError('RNG_ROUNDTRIP_MISMATCH')
    return payload,{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size,'integrity':integrity,'next_122_permutation_sha256':PERM_SHA[seed],'rng_roundtrip':True}
def source_unchanged(path,digest):
    if sha(path)!=digest: raise RuntimeError('PARENT_MUTATED')
def protected():
    rows=read(ROOT/'evaluation/phase56/safety-preflight.json')['protected_files']
    for row in rows:
        if sha(ROOT/row['path'])!=row['sha256']: raise RuntimeError('PROTECTED_SHA_CHANGED:'+row['path'])
    return rows
def dirty():
    rows=[]
    for line in subprocess.check_output(['git','status','--porcelain','--untracked-files=all'],cwd=ROOT,text=True).splitlines():
        rel=line[3:];path=ROOT/rel;rows.append({'status':line[:2],'path':rel,'sha256':sha(path) if path.is_file() else None})
    return rows
def artifact_clean():
    if OUT.exists() or RAW.exists(): raise RuntimeError('PHASE61_ARTIFACT_OR_PARTIAL_EXISTS')
    for seed in SEEDS:
        for arm,_ in ARMS:
            new(target(seed,arm))
def runtime_guard(monitor, estimated):
    free=shutil.disk_usage(ZROOT).free
    if free < 2*estimated+2*1024**3: raise RuntimeError('DISK_RESERVE_STOP')
    sample=query_gpu()
    if sample['gpu_temperature_c']>=85 or sample['hardware_thermal_slowdown']: raise RuntimeError('THERMAL_STOP')
    if sample['gpu_temperature_c']>=80:
        cool=cooldown()
        if not cool['target_reached'] or cool['end']['gpu_temperature_c']>65: raise RuntimeError('THERMAL_COOLDOWN_FAILED')
    if monitor.samples and monitor.samples[-1]['hardware_thermal_slowdown']: raise RuntimeError('THERMAL_STOP')

def preflight():
    spec=source_spec();artifact_clean()
    if git('branch','--show-current')!='foundation-research' or git('rev-parse','HEAD')!=START or git('ls-remote','origin','refs/heads/foundation-research').split()[0]!=START: raise RuntimeError('PHASE61_PREFLIGHT_BLOCKED')
    if not torch.cuda.is_available() or torch.version.cuda is None or torch.cuda.get_device_name(0)!='NVIDIA GeForce RTX 2070 SUPER': raise RuntimeError('CUDA_GPU_REQUIRED')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if sha(CACHE)!=CACHE_SHA or len(read(CACHE)['episodes'])!=128: raise RuntimeError('MATCHING_CACHE_INTEGRITY_FAIL')
    parents=[]
    for seed in SEEDS:
        _,row=strict_parent(seed);parents.append({'seed':seed,**row})
    data=ROOT/spec['data']['path']
    if sha(data)!=spec['data']['sha256']: raise RuntimeError('TRAINING_DATA_SHA_MISMATCH')
    if sha(ROOT/'data/foundation_v09/evaluation/final-blind-1000.json') != spec['sealed_sets']['final_blind_sha256']: raise RuntimeError('FINAL_BLIND_SHA_MISMATCH')
    largest=max(x['bytes'] for x in parents);z=shutil.disk_usage(ZROOT);c=shutil.disk_usage('C:\\'); required=2*largest+2*1024**3
    if z.free<required: raise RuntimeError('INSUFFICIENT_Z_DISK_FOR_ATOMIC_CHECKPOINTS')
    receipt={'phase':61,'explicit_phase61_authorization':True,'training_authorized_by_user_instruction':True,'at':now(),'start_head':START,'origin_sha':START,'branch':'foundation-research','checkpoint_root':str(ZROOT),'preregistration_path':str(SPEC.relative_to(ROOT)),'preregistration_sha256':sha(SPEC),'evaluator_schema_sha256':spec['evaluation']['schema_sha256'],'safeguard_mapping':'34/34 PASS','parents':parents,'matching_cache_sha256':sha(CACHE),'data_sha256':sha(data),'cuda':{'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda_build':torch.version.cuda,'fp32':True,'amp':False,'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'tf32_cudnn':torch.backends.cudnn.allow_tf32,'start':query_gpu()},'free_bytes':{'C':c.free,'Z':z.free},'atomic_reserve_required_bytes':required,'protected_files':protected(),'dirty_files':dirty(),'parent_operations':dict.fromkeys(('copy','move','delete','overwrite','rename'),0),'final_blind':'SHA_ONLY_PASS','reserve2':'SEALED_UNSCORED_HASH_ONLY','phase53_reserve':'RETIRED_UNSCORABLE_NOT_OPENED'}
    emit(OUT/'preflight.json',receipt);print('PHASE61_PREFLIGHT_PASS',flush=True)

def matching_forward(model,opt,payload,episode):
    before={'rng':fingerprint(random_state('cuda')),'model':fingerprint(model.state_dict()),'optimizer':fingerprint(opt.state_dict()),'scheduler':fingerprint(payload['scheduler_state'])}
    mode=model.training;model.eval()
    with torch.no_grad(): model(torch.tensor([episode['prefix']+episode['generated'][:31]],device='cuda'))
    model.train(mode)
    after={'rng':fingerprint(random_state('cuda')),'model':fingerprint(model.state_dict()),'optimizer':fingerprint(opt.state_dict()),'scheduler':fingerprint(payload['scheduler_state'])}
    if before!=after: raise RuntimeError('MATCHING_FORWARD_STATE_MUTATION')
    return {'state_unchanged':True,'episode_document_index':episode['document_index']}

@torch.inference_mode()
def dry_one(seed,arm,lr):
    spec=source_spec();src=parent(seed);digest=sha(src);cool=cooldown()
    if not cool['target_reached']: raise RuntimeError('THERMAL_COOLDOWN_FAILED')
    tok=FoundationTokenizer.load(ROOT/'tokenizer/foundation-v11-base-4096.json');model=load_model(src,torch.device('cuda'))
    prompts=[{'prefix_ids':[tok.bos_id]+tok.encode(x)} for x in ('日本の大学では','数学の基礎は')];prefix=[p['prefix_ids'] for p in prompts]
    candidate=contract.fixture(spec['evaluation']['all_thresholds']);baseline=copy.deepcopy(candidate);sampling={};mon=Monitor();mon.start()
    try:
        for base in contract.RNG_BASES:
            sampling[base]=contract.convert_legacy(metrics(generate_batch(model,tok,prefix,SAMPLE_T07,[int(base)+i for i in range(2)],8,trace=False),prompts),source_version=contract.LEGACY_VERSION)
        greedy=contract.convert_legacy(metrics(generate_batch(model,tok,prefix,GREEDY,[0,1],8,trace=True),prompts),source_version=contract.LEGACY_VERSION)
        candidate['sampling']=sampling;baseline['sampling']=copy.deepcopy(sampling)
        fields={path:candidate['scalars'][key] for key,path in contract.SCALAR_PRODUCERS.items()};candidate=contract.from_measurement_fields(fields,sampling,spec['evaluation']['all_thresholds']);gate=contract.safety_gate(candidate,baseline,spec['evaluation']['all_thresholds'])
    finally:
        thermal=mon.finish();del model;gc.collect();torch.cuda.empty_cache()
    source_unchanged(src,digest)
    if not thermal.get('samples') or thermal['gpu_temperature_c_max']>=85 or thermal['hardware_thermal_slowdown'] or gate['gate']!='CONTROL_SAFETY_PASS': raise RuntimeError('LIVE_CONTRACT_DRY_RUN_FAIL')
    return {'seed':seed,'arm':arm,'lr':lr,'pass':True,'cooldown':cool,'thermal':thermal,'greedy':greedy,'contract_gate':gate}
def dry_run():
    if not (OUT/'preflight.json').exists():raise RuntimeError('PREFLIGHT_REQUIRED')
    rows=[dry_one(seed,arm,lr) for seed in SEEDS for arm,lr in ARMS]
    emit(OUT/'live-contract-dry-run.json',{'phase':61,'kind':'six_cuda_live_contract_dry_runs','results':rows,'all_pass':all(x['pass'] for x in rows),'new_training':False});print('PHASE61_LIVE_CONTRACT_DRY_RUN_PASS',flush=True)

def save_checkpoint(seed,arm,lr,payload,model,opt,source_sha,stats,telemetry):
    dst=target(seed,arm);new(dst);ensure_checkpoint_storage(dst);dst.parent.mkdir(parents=True,exist_ok=True);tmp=dst.with_suffix('.pt.tmp');new(tmp)
    end=32122; runtime={'arm':arm,'optimizer_group_lr':lr,'constant_schedule':True,'scheduler_step_called':False,'historical_scheduler_metadata_warning':'learning_rate and peak_learning_rate are preserved historical metadata, not runtime authority'}
    saved={**payload,'model_state':model.state_dict(),'optimizer_state':opt.state_dict(),'scheduler_state':{**payload['scheduler_state'],'global_step':end},'random_state':random_state('cuda'),'update':end,'tokens_processed':end*512,'phase':61,'arm':arm,'experimental':True,'EXPERIMENTAL':True,'canonical':False,'NOT_CANONICAL':True,'not_canonical':True,'NOT_PRODUCTION':True,'not_production':True,'precision_mode':'fp32','eos_loss_weight':1.5,'repetition_auxiliary':False,'parent_checkpoint_sha256':source_sha,'phase61_runtime_lr_contract':runtime,'phase61_training':{'updates':122,'lm_gradient_tokens':62464,'matching_slots':15,'conservative_positions':63904,'stats_count':len(stats)}}
    torch.save(saved,tmp);loaded=torch.load(tmp,map_location='cpu',weights_only=False);integrity=verify_payload(loaded,seed,16_446_464,lr)
    strict=DiagnosticTransformerV17(DiagnosticConfigV17(**loaded['config']));strict.load_state_dict(loaded['model_state'],strict=True);verify_opt=create_optimizer(strict,lr,.1);verify_opt.load_state_dict(loaded['optimizer_state'])
    checks={'strict_model_reload':True,'strict_optimizer_reload':True,'scheduler':loaded['scheduler_state']['global_step']==end,'sampler':fingerprint(loaded['permutation'])==fingerprint(payload['permutation']),'rng':fingerprint(loaded['random_state'])==fingerprint(saved['random_state']),'runtime_lr':all(g['lr']==lr for g in verify_opt.param_groups) and loaded['phase61_runtime_lr_contract']==runtime,'markers':all(loaded[k] is True for k in ('EXPERIMENTAL','NOT_CANONICAL','NOT_PRODUCTION')),'finite_optimizer':finite(verify_opt.state_dict())}
    if not integrity['pass'] or not all(checks.values()): raise RuntimeError('OUTPUT_STRICT_RELOAD_FAIL:'+repr(checks))
    tmp.replace(dst);return {'path':str(dst),'sha256':sha(dst),'bytes':dst.stat().st_size,'integrity':integrity,'resume_integrity':checks}

def train_one(seed,arm,lr):
    spec=source_spec();dry=read(OUT/'live-contract-dry-run.json')
    if not dry['all_pass']:raise RuntimeError('LIVE_DRY_RUN_REQUIRED')
    source=parent(seed);source_sha=sha(source);payload,parent_integrity=strict_parent(seed);cool=cooldown()
    if not cool['target_reached']:raise RuntimeError('THERMAL_COOLDOWN_FAILED')
    loaded,model,opt=load(source,torch.device('cuda'))
    if loaded['update']!=32000:raise RuntimeError('PARENT_STEP_MISMATCH')
    continuity={k:fingerprint(payload[k]) for k in ('model_state','optimizer_state','scheduler_state','permutation','random_state')}
    if fingerprint(opt.state_dict())!=continuity['optimizer_state']:raise RuntimeError('OPTIMIZER_STATE_MISMATCH')
    state_before=fingerprint(opt.state_dict()['state']);groups_before=[{k:v for k,v in g.items() if k!='lr'} for g in opt.param_groups]
    for group in opt.param_groups:group['lr']=lr
    if fingerprint(opt.state_dict()['state'])!=state_before or groups_before != [{k:v for k,v in g.items() if k!='lr'} for g in opt.param_groups] or not all(g['lr']==lr for g in opt.param_groups):raise RuntimeError('LR_ONLY_ASSIGNMENT_FAIL')
    tok=FoundationTokenizer.load(ROOT/'tokenizer/foundation-v11-base-4096.json');data=np.memmap(ROOT/spec['data']['path'],dtype=np.uint16,mode='r');episodes=read(CACHE)['episodes'];estimated=parent_integrity['bytes']
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;model.train();torch.cuda.reset_peak_memory_stats();mon=Monitor();mon.start();stats=[];over10=0;started=time.perf_counter()
    try:
        for update in range(32001,32123):
            runtime_guard(mon,estimated)
            if not all(g['lr']==lr for g in opt.param_groups):raise RuntimeError('RUNTIME_LR_CONTRACT_FAIL')
            x,y=macro_batch(data,int(payload['permutation'][update-1]),512);x,y=x.cuda(),y.cuda();opt.zero_grad(set_to_none=True);logits,_=model(x);lm,eos,non=weighted_lm_loss(logits,y,tok.eos_id,1.5)
            matching=None
            if update%8==0:matching=matching_forward(model,opt,payload,episodes[(update//8-1)%128])
            if not torch.isfinite(lm):raise RuntimeError('NONFINITE_LOSS')
            lm.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.0));over10=over10+1 if norm>10 else 0
            if not math.isfinite(norm) or norm>100 or over10>=3:raise RuntimeError('GRADIENT_STOP')
            opt.step()
            if not finite(model.state_dict()) or not finite(opt.state_dict()):raise RuntimeError('NONFINITE_WEIGHT_OR_OPTIMIZER')
            stats.append({'update':update,'lm_loss':float(lm.detach()),'eos_loss':float(eos.detach()),'non_eos_loss':float(non.detach()),'gradient_norm_raw':norm,'clipped':norm>1,'matching_forward':matching,'runtime_lr':lr})
        torch.cuda.synchronize()
    finally:
        elapsed=time.perf_counter()-started;thermal=mon.finish()
    if len(stats)!=122 or sum(x['matching_forward'] is not None for x in stats)!=15:raise RuntimeError('TRAINING_BUDGET_OR_MATCHING_GEOMETRY_FAIL')
    if not thermal.get('samples') or thermal['gpu_temperature_c_max']>=85 or thermal['hardware_thermal_slowdown']:raise RuntimeError('THERMAL_STOP')
    source_unchanged(source,source_sha);checkpoint=save_checkpoint(seed,arm,lr,payload,model,opt,source_sha,stats,thermal);source_unchanged(source,source_sha)
    norms=np.asarray([x['gradient_norm_raw'] for x in stats]);raw={'phase':61,'seed':seed,'arm':arm,'lr':lr,'parent_sha256':source_sha,'start_update':32000,'end_update':32122,'lm_gradient_tokens':62464,'matching_slots':15,'conservative_positions':63904,'checkpoint':checkpoint,'continuity':continuity,'parent_integrity':parent_integrity,'cooldown':cool,'training':{'seconds':elapsed,'tokens_per_second':62464/elapsed,'mean_lm_loss':float(np.mean([x['lm_loss'] for x in stats])),'gradient_norm':{k:float(v) for k,v in {'mean':norms.mean(),'median':np.median(norms),'p90':np.quantile(norms,.9),'p95':np.quantile(norms,.95),'max':norms.max()}.items()},'clip_rate':float(np.mean(norms>1)),'peak_vram_mib':torch.cuda.max_memory_allocated()/1048576,'telemetry':thermal},'new_training':True,'experimental':True,'canonical':False,'runtime_lr_contract':'PASS','stats':stats}
    emit(RAW/f'run-seed{seed}-{arm}-training-raw.json',raw);emit(run_artifact(seed,arm),{k:v for k,v in raw.items() if k!='stats'});print('PHASE61_TRAIN_PASS',seed,arm,flush=True)
def train():
    for seed in SEEDS:
        for arm,lr in ARMS: train_one(seed,arm,lr)

def eval_model(label,path):
    spec=source_spec();dst=RAW/f'{label}-evaluation-raw.json';new(dst);digest=sha(path);tok=FoundationTokenizer.load(ROOT/'tokenizer/foundation-v11-base-4096.json');val=np.memmap(ROOT/spec['evaluation']['sets']['validation']['path'],dtype=np.uint16,mode='r');freq=read(ROOT/spec['evaluation']['sets']['frequency_population']['path']);phase51=read(ROOT/spec['evaluation']['sets']['generation']['path']);obs=read(ROOT/spec['evaluation']['context_reference']['path']);docs=read(ROOT/obs['diagnostic_source']['path']);lookup={f"jawiki:{x['page_id']}:{x['revision_id']}":x for x in docs};ids=[[tok.bos_id]+tok.encode(lookup[i]['text'])+[tok.eos_id] for i in obs['selected_document_ids']];controls=read(ROOT/spec['evaluation']['sets']['normal_controls']['path'])['prompts'];cool=cooldown()
    if not cool['target_reached']:raise RuntimeError('THERMAL_COOLDOWN_FAILED')
    model=load_model(path,torch.device('cuda'));mon=Monitor();mon.start()
    try:
        validation=measure.full_validation(model,val);core=measure.frequency_values(model,val,freq['core']);tail=measure.frequency_values(model,val,freq['population']);term=eos_rows(model,tok,val,phase51['terminal_positions']);non=eos_rows(model,tok,val,phase51['nonterminal_positions']);context=measure.context_metrics(model,tok,ids);normal=measure.normal_metrics(model,tok,controls);generation=measure.run_generation(model,tok,phase51['prompts'],phase51)
    finally:
        thermal=mon.finish();del model;gc.collect();torch.cuda.empty_cache()
    if not thermal.get('samples') or thermal['gpu_temperature_c_max']>=85 or thermal['hardware_thermal_slowdown']:raise RuntimeError('THERMAL_STOP')
    source_unchanged(path,digest);value={'phase':61,'label':label,'checkpoint_sha256':digest,'validation':validation,'core':{'metrics':measure.frequency_summary(core,freq['core']),'values':core,'document_ids':freq['core']['document_ids']},'supported_tail':{'metrics':measure.frequency_summary(tail,freq['population']),'values':tail,'document_ids':freq['population']['document_ids']},'eos':{'terminal_mean_probability':float(np.mean([x['probability'] for x in term])),'terminal_top1':float(np.mean([x['top1'] for x in term])),'nonterminal_mean_probability':float(np.mean([x['probability'] for x in non])),'premature_argmax_eos':float(np.mean([x['top1'] for x in non]))},'context':context,'normal_controls':normal,'generation':generation,'thermal':thermal}
    emit(dst,value);return value
def safety_value(row,ci,spec):
    fields={'validation.ce':row['validation']['ce'],'validation.top1':row['validation']['top1'],'validation.top5':row['validation']['top5'],'validation.top10':row['validation']['top10'],'context.128.ce':row['context']['128']['mean_ce'],'context.512.ce':row['context']['512']['mean_ce'],'eos.terminal_probability':row['eos']['terminal_mean_probability'],'eos.terminal_top1':row['eos']['terminal_top1'],'eos.nonterminal_probability':row['eos']['nonterminal_mean_probability'],'eos.premature_argmax':row['eos']['premature_argmax_eos'],'normal.mean_ce':row['normal_controls']['mean_ce'],'normal.terminal_probability':row['normal_controls']['terminal_eos_probability']}
    for group in ('core','supported_tail'):
        for key in ('micro_ce','macro_ce','top1','top5','top10'):fields[f'{group}.{key}']=row[group]['metrics']['macro_per_token_ce' if key=='macro_ce' else key]
        fields[f'{group}.paired_ce_ci95_upper']=ci[group]['upper']
    for family in contract.FAMILIES:fields[f'normal.{family}.ce']=row['normal_controls']['family_ce'][family]
    projection={path:fields[key] for key,path in contract.SCALAR_PRODUCERS.items()};sampling={base:contract.convert_legacy(row['generation']['sampling'][base]['metrics'],source_version=contract.LEGACY_VERSION) for base in contract.RNG_BASES};return contract.from_measurement_fields(projection,sampling,spec['evaluation']['all_thresholds'])
def compact(row): return {'checkpoint_sha256':row['checkpoint_sha256'],'validation':row['validation'],'core':row['core']['metrics'],'supported_tail':row['supported_tail']['metrics'],'eos':row['eos'],'context':row['context'],'normal_controls':{k:v for k,v in row['normal_controls'].items() if k!='rows'},'generation':{'greedy':row['generation']['greedy']['metrics'],'sampling':{k:v['metrics'] for k,v in row['generation']['sampling'].items()}},'thermal':row['thermal'],'raw_on_Z':True}
def evaluate_all():
    spec=source_spec()
    if not all(target(seed,arm).exists() for seed in SEEDS for arm,_ in ARMS):raise RuntimeError('ALL_SIX_RUNS_REQUIRED')
    results={}
    for seed in SEEDS:
        p=eval_model(f'seed{seed}-parent',parent(seed));c=eval_model(f'seed{seed}-control',target(seed,'control'));h=eval_model(f'seed{seed}-half-lr',target(seed,'half-lr'))
        ci=lambda after,before:{g:measure.ci_delta(after[g]['values']['ce'],before[g]['values']['ce'],after[g]['document_ids'],5701) for g in ('core','supported_tail')}
        zero={g:{'mean':0.,'lower':0.,'upper':0.} for g in ('core','supported_tail')};c_parent=contract.safety_gate(safety_value(c,ci(c,p),spec),safety_value(p,zero,spec),spec['evaluation']['all_thresholds']);h_parent=contract.safety_gate(safety_value(h,ci(h,p),spec),safety_value(p,zero,spec),spec['evaluation']['all_thresholds']);h_control=contract.safety_gate(safety_value(h,ci(h,c),spec),safety_value(c,zero,spec),spec['evaluation']['all_thresholds'])
        gate={'phase':61,'seed':seed,'control_vs_parent':c_parent,'half_lr_vs_parent':h_parent,'half_lr_vs_control':h_control,'all34':{'control_vs_parent':c_parent['gate']=='CONTROL_SAFETY_PASS','half_lr_vs_parent':h_parent['gate']=='CONTROL_SAFETY_PASS','half_lr_vs_control':h_control['gate']=='CONTROL_SAFETY_PASS'},'comparators':{'parent':compact(p),'control':compact(c),'half_lr':compact(h)}}
        emit(OUT/f'seed{seed}-gate.json',gate);results[str(seed)]={'valid':True,'control_all34_vs_parent':gate['all34']['control_vs_parent'],'half_lr_all34_vs_parent':gate['all34']['half_lr_vs_parent'],'half_lr_all34_vs_control':gate['all34']['half_lr_vs_control']}
    final=phase61_decision(results);emit(OUT/'final-gate.json',{'phase':61,'final_gate':final,'seeds':results,'approved_research_lr':5e-5,'stabilization_candidate_lr':2.5e-5,'formal_lr_changed':False,'generation_policy':'UNSAFE','phase57':'EXPERIMENT_INVALID','phase59':'CONTROL_STABILITY_MIXED','canonical':False,'20m':False,'foundation_base':False,'production':False});print('PHASE61_GATE',final,flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('preflight','dry-run','train','evaluate'));args=parser.parse_args();torch.set_num_threads(2)
    {'preflight':preflight,'dry-run':dry_run,'train':train,'evaluate':evaluate_all}[args.action]()
if __name__=='__main__': main()
