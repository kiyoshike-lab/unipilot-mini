"""PHASE59 registered control-only multi-seed stability replication.

This runner deliberately has no intervention arm and no extension command.  It
uses PHASE57 measurement primitives but never its obsolete decision gate.
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
from evaluation.register_foundation_v47_control import decision
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

OUT = ROOT / 'evaluation/phase59'
RAW = Path(r'Z:\AI\unipilot-mini\evaluation\phase59')
SPEC = ROOT / 'evaluation/phase58/phase59-control-stability-preregistration.json'
ZROOT = Path(r'Z:\AI\unipilot-mini\checkpoints')
START = 'fcdf87966e37aaa3be8cec94ab30f8552cd232ec'
CACHE = Path(r'Z:\AI\unipilot-mini\evaluation\phase57\cache-raw.json')
CACHE_SHA = '4978d51cb66326bccee71a966d6337555387ce74f5ac2d4415506076a69416dc'
SEEDS = (123, 2026)
EXPECTED_PARENT = {123: '78df96b70016dda180f4529833d904e2d448e11469a87278d4881ed09c93dc20', 2026: '7dcf5fa2da58777040cf9c03f1f513d707e6866f2eaf237bf203bd1b0135f069'}
EXPECTED_PERM = {123: '685d76f443d25bd4d271e1550a174e0b6380ad5b8d47020607e12947eb3440c9', 2026: '8e6cd015ea3e49605306d2be18a67e6c95b1126bcfdf82eb4e25bdda4efc136c'}

def now(): return datetime.now(timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2), b''): h.update(b)
    return h.hexdigest()
def git(*args): return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
def require_new(path):
    p=Path(path)
    if p.exists() or p.with_suffix(p.suffix+'.tmp').exists(): raise FileExistsError('existing/partial artifact: '+str(p))
def write_new(path, value):
    p=Path(path); require_new(p); p.parent.mkdir(parents=True, exist_ok=True); tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    tmp.replace(p)
def finite(value):
    if torch.is_tensor(value): return bool(torch.isfinite(value).all())
    if isinstance(value, dict): return all(finite(v) for v in value.values())
    if isinstance(value, (tuple,list)): return all(finite(v) for v in value)
    return True
def root_gate():
    if os.environ.get('UNIPILOT_CHECKPOINT_ROOT') != str(ZROOT) or checkpoint_root(ROOT) != ZROOT:
        raise RuntimeError('PROCESS_ENV_RESOLVER_MISMATCH')
def source_spec():
    root_gate(); s=read(SPEC)
    if s['status'] != 'REGISTERED_REQUIRES_NEW_USER_TRAINING_AUTHORIZATION' or s['approved_lr'] != 5e-5 or s['seeds'] != [123,2026]: raise RuntimeError('PREREGISTRATION_SCOPE_MISMATCH')
    if sha(SPEC) != '0a16e3e6e9589f984ad7d88039b0a28c91c430837170ec499106ab7cc9967261': raise RuntimeError('PREREGISTRATION_SHA_MISMATCH')
    if sha(ROOT/s['evaluation_schema_path']) != s['evaluation_schema_sha256'] or s['evaluation_schema_sha256'] != '6576428940779e2908a8aacc67c4fb6bd2ca8bdc945e6202b88d5116ec701c6b': raise RuntimeError('EVALUATOR_SCHEMA_MISMATCH')
    if read(ROOT/s['evaluation_schema_path']) != contract.schema_document(s['all_safeguards']): raise RuntimeError('EVALUATOR_SCHEMA_CODE_MISMATCH')
    if len(contract.mapping(s['all_safeguards'])) != 34: raise RuntimeError('SAFEGUARD_COVERAGE_FAIL')
    for rel, expected in {**s['source_sha256'], **s['artifact_sha256']}.items():
        if sha(ROOT/rel) != expected: raise RuntimeError('REGISTERED_INPUT_CHANGED:'+rel)
    return s
def parent(seed): return existing_checkpoint_path(ROOT,'experimental','phase48','arm-C',f'seed-{seed}','checkpoint-tokens-16384000.pt')
def candidate(seed): return checkpoint_path(ROOT,'experimental','phase59','control-stability',f'seed-{seed}','checkpoint-tokens-16446464.pt')
def strict_parent(seed, path=None):
    p=Path(path or parent(seed));
    if not p.is_relative_to(ZROOT) or sha(p)!=EXPECTED_PARENT[seed]: raise RuntimeError('PARENT_SHA_OR_PATH_MISMATCH')
    payload=torch.load(p,map_location='cpu',weights_only=False); integrity=verify_payload(payload,seed,16_384_000,5e-5)
    if fingerprint(payload['permutation'][32000:32122]) != EXPECTED_PERM[seed]: raise RuntimeError('PERMUTATION_HASH_MISMATCH')
    restore_random_state(payload['random_state'],'cuda',cuda_seed=seed)
    if fingerprint(random_state('cuda')) != fingerprint(payload['random_state']): raise RuntimeError('RNG_ROUNDTRIP_MISMATCH')
    return payload, {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size,'integrity':integrity,'rng_roundtrip':True,'next_122_permutation_sha256':EXPECTED_PERM[seed]}
def runtime_guard(monitor):
    sample=query_gpu()
    if sample['gpu_temperature_c'] >= 85 or sample['hardware_thermal_slowdown']: raise RuntimeError('THERMAL_STOP')
    if sample['gpu_temperature_c'] >= 80:
        cool=cooldown()
        if not cool['target_reached'] or cool['end']['gpu_temperature_c']>65: raise RuntimeError('THERMAL_COOLDOWN_FAILED')
    if monitor.samples and monitor.samples[-1]['hardware_thermal_slowdown']: raise RuntimeError('THERMAL_STOP')
def source_unchanged(path, digest):
    if sha(path)!=digest: raise RuntimeError('PARENT_MUTATED')
def artifact_clean():
    for p in (OUT,RAW):
        if p.exists(): raise RuntimeError('PHASE59_ARTIFACT_OR_PARTIAL_EXISTS:'+str(p))
    for p in ZROOT.glob('experimental/phase59/control-stability/**/*.tmp'):
        raise RuntimeError('PHASE59_PARTIAL_CHECKPOINT:'+str(p))
def protected():
    rows=read(ROOT/'evaluation/phase56/safety-preflight.json')['protected_files']
    for r in rows:
        if sha(ROOT/r['path']) != r['sha256']: raise RuntimeError('PROTECTED_SHA_CHANGED:'+r['path'])
    return rows
def dirty_manifest():
    rows=[]
    for line in subprocess.check_output(['git','status','--porcelain','--untracked-files=all'],cwd=ROOT,text=True).splitlines():
        rel=line[3:]; p=ROOT/rel
        rows.append({'status':line[:2],'path':rel,'sha256':sha(p) if p.is_file() else None})
    return rows
def preflight():
    spec=source_spec(); artifact_clean()
    if git('branch','--show-current')!='foundation-research': raise RuntimeError('PHASE59_PREFLIGHT_BLOCKED_BRANCH')
    if subprocess.run(['git','merge-base','--is-ancestor',START,'HEAD'],cwd=ROOT).returncode: raise RuntimeError('AUTHORIZED_START_NOT_ANCESTOR')
    if torch.cuda.is_available() is False or torch.version.cuda is None or torch.cuda.get_device_name(0)!='NVIDIA GeForce RTX 2070 SUPER': raise RuntimeError('CUDA_GPU_REQUIRED')
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    if sha(CACHE)!=CACHE_SHA: raise RuntimeError('MATCHING_CACHE_SHA_MISMATCH')
    episodes=read(CACHE)['episodes']
    if len(episodes)!=128: raise RuntimeError('MATCHING_CACHE_GEOMETRY_MISMATCH')
    parents=[]; payloads=[]
    for seed in SEEDS:
        payload, row=strict_parent(seed); parents.append({'seed':seed,**row}); payloads.append(payload)
    data=ROOT/spec['data']['path']; tok=ROOT/'tokenizer/foundation-v11-base-4096.json'
    if sha(data)!=spec['data']['sha256'] or sha(tok)!=spec['source_sha256']['tokenizer/foundation-v11-base-4096.json']: raise RuntimeError('TRAINING_INPUT_SHA_MISMATCH')
    size=max(r['bytes'] for r in parents); z=shutil.disk_usage(ZROOT); c=shutil.disk_usage('C:\\')
    required=2*size+2*1024**3
    if z.free<required: raise RuntimeError('INSUFFICIENT_Z_DISK_FOR_ATOMIC_CHECKPOINTS')
    final=ROOT/'data/foundation_v09/evaluation/final-blind-1000.json'
    if sha(final)!='fa7912d58ce251bb10b513f59793bb8ca6c0023b4fe08d1c040b8ccbfe49845b': raise RuntimeError('FINAL_BLIND_SHA_MISMATCH')
    receipt={'phase':59,'authorized_training':True,'explicit_user_authorization':True,'at':now(),'authorized_start_head':START,'implementation_head':git('rev-parse','HEAD'),'origin_at_initial_gate':START,'branch':git('branch','--show-current'),'checkpoint_root':str(ZROOT),'preregistration_sha256':sha(SPEC),'evaluation_schema_version':contract.VERSION,'evaluation_schema_sha256':spec['evaluation_schema_sha256'],'safeguard_coverage':'34/34','parents':parents,'cache_sha256':sha(CACHE),'cache_episode_indices':list(range(32,47)),'data_sha256':sha(data),'tokenizer_sha256':sha(tok),'cuda':{'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda,'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'tf32_cudnn':torch.backends.cudnn.allow_tf32,'fp32':True,'start':query_gpu()},'ram':dict(psutil.virtual_memory()._asdict()),'free_bytes':{'C':c.free,'Z':z.free},'atomic_storage_requirement_bytes':required,'protected_files':protected(),'dirty_files':dirty_manifest(),'checkpoint_copy_move_delete_overwrite':[0,0,0,0],'phase57_status':'EXPERIMENT_INVALID','generation_policy':'UNSAFE','final_blind':'SHA_ONLY_PASS','reserve2':'SEALED_UNSCORED_HASH_ONLY'}
    del payloads; gc.collect(); write_new(OUT/'preflight.json',receipt); print('PHASE59_PREFLIGHT_PASS',flush=True)
def matching_forward(model, optimizer, payload, episode):
    before={'rng':fingerprint(random_state('cuda')),'model':fingerprint(model.state_dict()),'optimizer':fingerprint(optimizer.state_dict()),'scheduler':fingerprint(payload['scheduler_state'])}
    mode=model.training; model.eval()
    with torch.no_grad(): model(torch.tensor([episode['prefix']+episode['generated'][:31]],device='cuda'))
    model.train(mode)
    after={'rng':fingerprint(random_state('cuda')),'model':fingerprint(model.state_dict()),'optimizer':fingerprint(optimizer.state_dict()),'scheduler':fingerprint(payload['scheduler_state'])}
    if before!=after: raise RuntimeError('MATCHING_FORWARD_STATE_MUTATION')
    return {'state_unchanged':True,'episode_document_index':episode['document_index']}
@torch.inference_mode()
def dry_one(seed):
    spec=source_spec(); p=parent(seed); before=sha(p); cool=cooldown()
    if not cool['target_reached']: raise RuntimeError('THERMAL_COOLDOWN_FAILED')
    tok=FoundationTokenizer.load(ROOT/'tokenizer/foundation-v11-base-4096.json'); model=load_model(p,torch.device('cuda')); prompts=[{'prefix_ids':[tok.bos_id]+tok.encode(x)} for x in ('日本の大学では','数学の基礎は')]; prefix=[x['prefix_ids'] for x in prompts]; candidate_value=contract.fixture(spec['all_safeguards']); parent_value=copy.deepcopy(candidate_value); sampling={}; mon=Monitor(); mon.start()
    try:
        for base in contract.RNG_BASES:
            rows=generate_batch(model,tok,prefix,SAMPLE_T07,[int(base)+i for i in range(2)],8,trace=False)
            sampling[base]=contract.convert_legacy(metrics(rows,prompts),source_version=contract.LEGACY_VERSION)
        greedy=contract.convert_legacy(metrics(generate_batch(model,tok,prefix,GREEDY,[0,1],8,trace=True),prompts),source_version=contract.LEGACY_VERSION)
        candidate_value['sampling']=sampling; parent_value['sampling']=copy.deepcopy(sampling)
        projected={path:candidate_value['scalars'][key] for key,path in contract.SCALAR_PRODUCERS.items()}; candidate_value=contract.from_measurement_fields(projected,sampling,spec['all_safeguards']); gate=contract.safety_gate(candidate_value,parent_value,spec['all_safeguards'])
    finally:
        thermal=mon.finish(); del model; gc.collect(); torch.cuda.empty_cache()
    source_unchanged(p,before)
    if not thermal.get('samples') or thermal['gpu_temperature_c_max']>=85 or thermal['hardware_thermal_slowdown'] or gate['gate']!='CONTROL_SAFETY_PASS': raise RuntimeError('LIVE_CONTRACT_DRY_RUN_FAIL')
    return {'seed':seed,'pass':True,'generation':sampling,'greedy':greedy,'cooldown':cool,'thermal':thermal,'contract_gate':gate}
def dry_run():
    if not (OUT/'preflight.json').exists(): raise RuntimeError('PREFLIGHT_REQUIRED')
    value={'phase':59,'kind':'live_cuda_contract_dry_run','results':[dry_one(seed) for seed in SEEDS],'quality_claim':False,'new_training':False}
    write_new(OUT/'live-contract-dry-run.json',value); print('PHASE59_LIVE_CONTRACT_DRY_RUN_PASS',flush=True)
def save_checkpoint(seed,payload,model,optimizer,source_sha,stats,telemetry):
    target=candidate(seed); require_new(target); ensure_checkpoint_storage(target); target.parent.mkdir(parents=True,exist_ok=True); tmp=target.with_suffix('.pt.tmp'); require_new(tmp)
    end=32122; saved={**payload,'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),'scheduler_state':{**payload['scheduler_state'],'global_step':end},'random_state':random_state('cuda'),'update':end,'tokens_processed':end*512,'phase':59,'arm':'control','experimental':True,'EXPERIMENTAL':True,'formal_research':False,'promoted':False,'canonical':False,'not_canonical':True,'NOT_CANONICAL':True,'not_production':True,'NOT_PRODUCTION':True,'precision_mode':'fp32','eos_loss_weight':1.5,'repetition_auxiliary':False,'parent_checkpoint_sha256':source_sha,'phase59_training':{'objective':'weighted LM control','updates':122,'matching_no_grad_slots':15,'stats_count':len(stats)}}
    torch.save(saved,tmp); loaded=torch.load(tmp,map_location='cpu',weights_only=False); integrity=verify_payload(loaded,seed,16_446_464,5e-5)
    strict=DiagnosticTransformerV17(DiagnosticConfigV17(**loaded['config'])); strict.load_state_dict(loaded['model_state'],strict=True); opt=create_optimizer(strict,5e-5,.1); opt.load_state_dict(loaded['optimizer_state'])
    checks={'strict_model_reload':True,'strict_optimizer_reload':True,'scheduler':loaded['scheduler_state']['global_step']==32122,'sampler':fingerprint(loaded['permutation'])==fingerprint(payload['permutation']),'rng':fingerprint(loaded['random_state'])==fingerprint(saved['random_state']),'markers':all(loaded[k] is v for k,v in {'EXPERIMENTAL':True,'NOT_CANONICAL':True,'NOT_PRODUCTION':True}.items()),'finite_optimizer':finite(opt.state_dict())}
    if not integrity['pass'] or not all(checks.values()): raise RuntimeError('OUTPUT_STRICT_RELOAD_FAIL:'+repr(checks))
    tmp.replace(target); return {'path':str(target),'sha256':sha(target),'bytes':target.stat().st_size,'integrity':integrity,'resume_integrity':checks}
def train_one(seed):
    spec=source_spec(); pre=read(OUT/'preflight.json'); dry=read(OUT/'live-contract-dry-run.json')
    if not all(r['pass'] for r in dry['results']): raise RuntimeError('LIVE_DRY_RUN_REQUIRED')
    source=parent(seed); source_sha=sha(source); payload, integrity=strict_parent(seed); cool=cooldown()
    if not cool['target_reached']: raise RuntimeError('THERMAL_COOLDOWN_FAILED')
    model_payload,model,opt=load(source,torch.device('cuda'))
    if model_payload['update']!=32000 or any(g['lr']!=5e-5 for g in opt.param_groups): raise RuntimeError('OPTIMIZER_CONTINUITY_MISMATCH')
    continuity={k:fingerprint(payload[k]) for k in ('model_state','optimizer_state','scheduler_state','permutation','random_state')}
    if fingerprint(opt.state_dict())!=continuity['optimizer_state']: raise RuntimeError('OPTIMIZER_STATE_MISMATCH')
    tok=FoundationTokenizer.load(ROOT/'tokenizer/foundation-v11-base-4096.json'); data=np.memmap(ROOT/spec['data']['path'],dtype=np.uint16,mode='r'); episodes=read(CACHE)['episodes']; torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;model.train();torch.cuda.reset_peak_memory_stats();mon=Monitor();mon.start();stats=[];over10=0;started=time.perf_counter()
    try:
        for update in range(32001,32123):
            runtime_guard(mon); ensure_checkpoint_storage(candidate(seed)); x,y=macro_batch(data,int(payload['permutation'][update-1]),512);x,y=x.cuda(),y.cuda();opt.zero_grad(set_to_none=True); logits,_=model(x);lm,eos,non=weighted_lm_loss(logits,y,tok.eos_id,1.5)
            match=None
            if update%8==0: match=matching_forward(model,opt,payload,episodes[(update//8-1)%len(episodes)])
            if not torch.isfinite(lm): raise RuntimeError('NONFINITE_LOSS')
            lm.backward(); norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)); over10=over10+1 if norm>10 else 0
            if not math.isfinite(norm) or norm>100 or over10>=3: raise RuntimeError('GRADIENT_STOP')
            opt.step()
            if not finite(model.state_dict()) or not finite(opt.state_dict()): raise RuntimeError('NONFINITE_WEIGHT_OR_OPTIMIZER')
            stats.append({'update':update,'lm_loss':float(lm.detach()),'eos_loss':float(eos.detach()),'non_eos_loss':float(non.detach()),'gradient_norm_raw':norm,'clipped':norm>1,'matching_forward':match})
        torch.cuda.synchronize()
    finally:
        elapsed=time.perf_counter()-started; thermal=mon.finish()
    if len(stats)!=122 or sum(x['matching_forward'] is not None for x in stats)!=15: raise RuntimeError('TRAINING_BUDGET_OR_MATCHING_GEOMETRY_FAIL')
    if not thermal.get('samples') or thermal['gpu_temperature_c_max']>=85 or thermal['hardware_thermal_slowdown']: raise RuntimeError('THERMAL_STOP')
    source_unchanged(source,source_sha); checkpoint=save_checkpoint(seed,payload,model,opt,source_sha,stats,thermal); source_unchanged(source,source_sha)
    raw={'phase':59,'seed':seed,'parent_sha256':source_sha,'start_update':32000,'end_update':32122,'lm_tokens':62464,'matching_no_grad_slots':15,'matching_forward_positions_charged':1440,'conservative_positions':63904,'gradient_bearing_positions':62464,'checkpoint':checkpoint,'continuity':continuity,'parent_integrity':integrity,'cooldown':cool,'training':{'seconds':elapsed,'tokens_per_second':62464/elapsed,'mean_lm_loss':float(np.mean([x['lm_loss'] for x in stats])),'mean_gradient_norm':float(np.mean([x['gradient_norm_raw'] for x in stats])),'max_gradient_norm':float(np.max([x['gradient_norm_raw'] for x in stats])),'clip_rate':float(np.mean([x['clipped'] for x in stats])),'peak_vram_mib':torch.cuda.max_memory_allocated()/1048576,'telemetry':thermal},'stats':stats,'new_training':True,'experimental':True,'canonical':False}
    write_new(RAW/f'seed-{seed}-training-raw.json',raw); write_new(OUT/f'seed{seed}-training-summary.json',{k:v for k,v in raw.items() if k!='stats'}); print('PHASE59_TRAIN_PASS',seed,flush=True)
def train():
    if not (OUT/'live-contract-dry-run.json').exists(): raise RuntimeError('LIVE_DRY_RUN_REQUIRED')
    train_one(123); cool=cooldown()
    if not cool['target_reached']: raise RuntimeError('INTER_SEED_COOLDOWN_FAILED')
    train_one(2026)
def evaluate(label,path):
    spec=source_spec(); target=RAW/f'{label}-evaluation-raw.json'; require_new(target); digest=sha(path); tok=FoundationTokenizer.load(ROOT/'tokenizer/foundation-v11-base-4096.json'); val=np.memmap(ROOT/spec['evaluation_sets']['validation']['path'],dtype=np.uint16,mode='r'); freq=read(ROOT/spec['evaluation_sets']['frequency_population']['path']); phase51=read(ROOT/spec['evaluation_sets']['generation']['path']); obs=read(ROOT/'evaluation/phase56/observability-preregistration.json'); docs=read(ROOT/obs['diagnostic_source']['path']); lookup={f"jawiki:{x['page_id']}:{x['revision_id']}":x for x in docs}; ids=[[tok.bos_id]+tok.encode(lookup[i]['text'])+[tok.eos_id] for i in obs['selected_document_ids']]; controls=read(ROOT/spec['evaluation_sets']['normal_controls']['path'])['prompts']; cool=cooldown()
    if not cool['target_reached']: raise RuntimeError('THERMAL_COOLDOWN_FAILED')
    model=load_model(path,torch.device('cuda')); mon=Monitor();mon.start()
    try:
        validation=measure.full_validation(model,val); core=measure.frequency_values(model,val,freq['core']); tail=measure.frequency_values(model,val,freq['population']); term=eos_rows(model,tok,val,phase51['terminal_positions']); non=eos_rows(model,tok,val,phase51['nonterminal_positions']); context=measure.context_metrics(model,tok,ids); normal=measure.normal_metrics(model,tok,controls); generation=measure.run_generation(model,tok,phase51['prompts'],phase51)
    finally:
        thermal=mon.finish();del model;gc.collect();torch.cuda.empty_cache()
    if not thermal.get('samples') or thermal['gpu_temperature_c_max']>=85 or thermal['hardware_thermal_slowdown']: raise RuntimeError('THERMAL_STOP')
    source_unchanged(path,digest); result={'phase':59,'label':label,'checkpoint_sha256':digest,'validation':validation,'core':{'population_sha256':freq['core']['sha256'],'metrics':measure.frequency_summary(core,freq['core']),'values':core,'document_ids':freq['core']['document_ids']},'supported_tail':{'population_sha256':freq['population']['sha256'],'metrics':measure.frequency_summary(tail,freq['population']),'values':tail,'document_ids':freq['population']['document_ids']},'eos':{'terminal_mean_probability':float(np.mean([x['probability'] for x in term])),'terminal_top1':float(np.mean([x['top1'] for x in term])),'nonterminal_mean_probability':float(np.mean([x['probability'] for x in non])),'premature_argmax_eos':float(np.mean([x['top1'] for x in non])),'terminal':term,'nonterminal':non},'context':context,'normal_controls':normal,'generation':generation,'thermal':thermal,'cooldown':cool}
    write_new(target,result); return result
def compact(row):
    return {'phase':row['phase'],'label':row['label'],'checkpoint_sha256':row['checkpoint_sha256'],'validation':row['validation'],'core':{'population_sha256':row['core']['population_sha256'],'metrics':row['core']['metrics']},'supported_tail':{'population_sha256':row['supported_tail']['population_sha256'],'metrics':row['supported_tail']['metrics']},'eos':{k:v for k,v in row['eos'].items() if k not in ('terminal','nonterminal')},'context':row['context'],'normal_controls':{k:v for k,v in row['normal_controls'].items() if k!='rows'},'generation':{'greedy':row['generation']['greedy']['metrics'],'sampling':{k:v['metrics'] for k,v in row['generation']['sampling'].items()}},'thermal':row['thermal'],'raw_only':True}
def safety_value(row, ci):
    fields={'validation.ce':row['validation']['ce'],'validation.top1':row['validation']['top1'],'validation.top5':row['validation']['top5'],'validation.top10':row['validation']['top10'],'context.128.ce':row['context']['128']['mean_ce'],'context.512.ce':row['context']['512']['mean_ce'],'eos.terminal_probability':row['eos']['terminal_mean_probability'],'eos.terminal_top1':row['eos']['terminal_top1'],'eos.nonterminal_probability':row['eos']['nonterminal_mean_probability'],'eos.premature_argmax':row['eos']['premature_argmax_eos'],'normal.mean_ce':row['normal_controls']['mean_ce'],'normal.terminal_probability':row['normal_controls']['terminal_eos_probability']}
    for group in ('core','supported_tail'):
        for key in ('micro_ce','macro_ce','top1','top5','top10'): fields[f'{group}.{key}']=row[group]['metrics']['macro_per_token_ce' if key=='macro_ce' else key]
        fields[f'{group}.paired_ce_ci95_upper']=ci[group]['upper']
    for f in contract.FAMILIES: fields[f'normal.{f}.ce']=row['normal_controls']['family_ce'][f]
    projection={path:fields[key] for key,path in contract.SCALAR_PRODUCERS.items()}; sampling={base:contract.convert_legacy(row['generation']['sampling'][base]['metrics'],source_version=contract.LEGACY_VERSION) for base in contract.RNG_BASES}; return contract.from_measurement_fields(projection,sampling,source_spec()['all_safeguards'])
def evaluate_all():
    if not all(candidate(s).exists() for s in SEEDS): raise RuntimeError('BOTH_TRAINED_CHECKPOINTS_REQUIRED')
    results={}
    for seed in SEEDS:
        p=evaluate(f'seed-{seed}-parent',parent(seed)); c=evaluate(f'seed-{seed}-candidate',candidate(seed)); ci={g:measure.ci_delta(c[g]['values']['ce'],p[g]['values']['ce'],c[g]['document_ids'],5701) for g in ('core','supported_tail')}; pc={g:{'mean':0.,'lower':0.,'upper':0.} for g in ci}; gate=contract.safety_gate(safety_value(c,ci),safety_value(p,pc),source_spec()['all_safeguards']); summary={'phase':59,'seed':seed,'parent':compact(p),'candidate':compact(c),'paired_document_bootstrap_10000_seed5701':ci,'safety_gate':gate,'all_34_safeguards_pass':gate['gate']=='CONTROL_SAFETY_PASS'}; write_new(OUT/f'seed{seed}-evaluation-summary.json',summary); results[str(seed)]=summary
    rows={seed:{'valid':True,'all_safety_pass':r['all_34_safeguards_pass'],'core_micro_pass':r['safety_gate']['checks']['core.micro_ce']['pass'],'core_macro_pass':r['safety_gate']['checks']['core.macro_ce']['pass']} for seed,r in results.items()}; final=decision(rows); gate={'phase':59,'seed_results':rows,'final_gate':final,'new_training':True,'experimental':True,'canonical':False,'generation_policy':'UNSAFE','phase57_status':'EXPERIMENT_INVALID'}; write_new(OUT/'control-stability-gate.json',gate); report='# PHASE 59 — Control Stability Replication\n\nFinal gate: **'+final+'**\n\n- Approved LR: 5e-5\n- Seeds: 123 / 2026\n- New checkpoints: EXPERIMENTAL, NOT_CANONICAL, NOT_PRODUCTION\n- Generation Policy: UNSAFE\n- PHASE57: EXPERIMENT_INVALID\n'; write_new(ROOT/'evaluation/foundation-v48-control-stability-report.md',report); write_new(ROOT/'evaluation/foundation-v48-control-stability-summary.json',{'phase':59,'final_gate':final,'seeds':results,'generation_policy':'UNSAFE','canonical_promotion':False,'20m':False,'foundation_base':False}); print('PHASE59_GATE',final,flush=True)
def main():
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=('preflight','dry-run','train','evaluate')); args=parser.parse_args(); torch.set_num_threads(2)
    if args.action=='preflight': preflight()
    elif args.action=='dry-run': dry_run()
    elif args.action=='train': train()
    else: evaluate_all()
if __name__=='__main__': main()
