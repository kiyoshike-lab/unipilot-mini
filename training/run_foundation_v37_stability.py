"""PHASE48: fill missing LR arms, then resume selected experimental arms to 512k."""
from __future__ import annotations
import argparse
import sys
import shutil
import time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from training import run_foundation_v36_lr_review as v36
from training.run_foundation_v35_thermal_gate import Monitor, cooldown
from training.run_foundation_v30_eos_experiment import load
from training.train_foundation_v21_ab import file_sha256, random_state
from training.train_foundation_v15_controlled import macro_batch
from training.foundation_v31_objective import weighted_lm_loss
from foundation.base_tokenizer import FoundationTokenizer

SEEDS=(42,123,2026)
ARMS={'B':7.5e-5,'C':5e-5}
EVAL=ROOT/'evaluation/phase48'
OUT=ROOT/'checkpoints/experimental/phase48'
START=15_872_000
BUDGET=256_000
read=v36.read_json
write=v36.write_json


def checkpoint(arm,seed,budget):
    if arm not in ARMS or seed not in SEEDS or budget not in (256000,512000): raise ValueError('invalid experiment')
    return OUT/f'arm-{arm}/seed-{seed}/checkpoint-tokens-{START+budget}.pt'


def reused(arm,seed): return arm=='C' or (arm=='B' and seed==42)


def checkpoint256(arm,seed):
    return v36.checkpoint(arm,seed) if reused(arm,seed) else checkpoint(arm,seed,256000)


def evaluation_path(arm,seed,budget):
    if budget==256000 and reused(arm,seed): return v36.EVAL/f'arm-{arm}/seed-{seed}-evaluation.json'
    return EVAL/f'arm-{arm}/{budget}/seed-{seed}-evaluation.json'


def training_path(arm,seed,budget):
    if budget==256000 and reused(arm,seed): return v36.EVAL/f'arm-{arm}/seed-{seed}-training.json'
    return EVAL/f'arm-{arm}/{budget}/seed-{seed}-training.json'


def preflight():
    assert shutil.disk_usage(ROOT).free>=20*1024**3
    assert torch.cuda.is_available() and torch.version.cuda
    assert torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 2070 SUPER'
    prior=read(v36.EVAL/'integrity-final.json')['checkpoints']
    immutable=[]
    for r in prior:
        path=ROOT/r['path']
        # Compare with original recorded SHA, not only current readability.
        if 'phase47' in path.parts:
            arm=next(p.removeprefix('arm-') for p in path.parts if p.startswith('arm-'))
            seed=int(next(p.removeprefix('seed-') for p in path.parts if p.startswith('seed-')))
            expected=read(v36.EVAL/f'arm-{arm}/seed-{seed}-training.json')['sha256']
        else:
            expected=next(x['sha256'] for x in read(v36.EVAL/'preflight.json')['checkpoints'] if Path(x['path'])==Path(r['path']))
        assert file_sha256(path)==expected
        p=torch.load(path,map_location='cpu',weights_only=False)
        lr=p.get('experimental_lr',1e-4)
        integrity=v36.verify_payload(p,p['seed'],p['tokens_processed'],lr)
        immutable.append({'path':path.relative_to(ROOT).as_posix(),'sha256':expected,'integrity':integrity})
    local=[]
    for r in read(v36.EVAL/'preservation-start.json'):
        # Keep all current user-local files, including the already committed dependencies.
        path=ROOT/r['Path']; local.append({'path':str(path),'sha256':file_sha256(path)})
    blind=file_sha256(ROOT/'data/foundation_v09/evaluation/final-blind-1000.json')
    assert blind=='fa7912d58ce251bb10b513f59793bb8ca6c0023b4fe08d1c040b8ccbfe49845b'
    write(EVAL/'preflight.json',{'phase':48,'free_bytes':shutil.disk_usage(ROOT).free,'gpu':torch.cuda.get_device_name(0),
          'cuda_build':torch.version.cuda,'immutable_checkpoints':immutable,'preserved_files':local,'final_blind_sha256':blind,'pass':True})
    print({'preflight':'PASS','checkpoint_count':len(immutable)},flush=True)


def train(arm,seed,budget):
    assert read(EVAL/'tests-preflight.json')['failed']==0
    assert shutil.disk_usage(ROOT).free>=20*1024**3
    if budget==256000:
        if reused(arm,seed): raise RuntimeError('reuse existing PHASE47 experiment')
        source=v36.official(seed)
        copy_path=OUT/f'sources/seed-{seed}/checkpoint-tokens-15872000.pt'
        copy_path.parent.mkdir(parents=True,exist_ok=True)
        if not copy_path.exists(): shutil.copy2(source,copy_path)
        assert file_sha256(copy_path)==file_sha256(source)
        training_source=copy_path
    else:
        assert arm in read(EVAL/'decision-256.json')['extend_arms']
        source=checkpoint256(arm,seed); training_source=source
    target=checkpoint(arm,seed,budget)
    if target.exists() or target.with_suffix('.pt.tmp').exists(): raise FileExistsError(target)
    source_sha=file_sha256(source)
    if budget==256000:
        expected=next(r['sha256'] for r in read(EVAL/'preflight.json')['immutable_checkpoints'] if Path(r['path'])==source.relative_to(ROOT))
    else:
        expected=read(training_path(arm,seed,256000))['sha256']
    assert source_sha==expected
    expected_tokens=START+budget-BUDGET
    p=torch.load(training_source,map_location='cpu',weights_only=False)
    assert p['tokens_processed']==expected_tokens
    initial_lr=1e-4 if budget==256000 else ARMS[arm]
    v36.verify_payload(p,seed,expected_tokens,initial_lr)
    cooling=cooldown(); device=torch.device('cuda')
    p,model,opt=load(training_source,device)
    continuity={k:v36.fingerprint(p[k]) for k in ('model_state','optimizer_state','scheduler_state','random_state','permutation')}
    assert v36.fingerprint(opt.state_dict())==continuity['optimizer_state']
    assert v36.fingerprint(random_state(device))==continuity['random_state']
    if budget==256000: v36.set_lr_only(opt,ARMS[arm])
    else: assert all(g['lr']==ARMS[arm] for g in opt.param_groups)
    tok=FoundationTokenizer.load(ROOT/'tokenizer/foundation-v11-base-4096.json')
    data=np.memmap(ROOT/'data/foundation_v11/packed/vocab-4096/train.bin',dtype=np.uint16,mode='r')
    end=p['update']+500; losses=[]; norms=[]
    model.train(); torch.cuda.reset_peak_memory_stats(); monitor=Monitor(); monitor.start(); began=time.perf_counter()
    try:
        for step in range(p['update']+1,end+1):
            x,y=macro_batch(data,int(p['permutation'][step-1]),512); x,y=x.to(device),y.to(device)
            opt.zero_grad(set_to_none=True); z,_=model(x)
            loss,_,_=weighted_lm_loss(z,y,tok.eos_id,1.5); assert torch.isfinite(loss)
            loss.backward(); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); assert torch.isfinite(norm)
            opt.step(); losses.append(float(loss.detach())); norms.append(float(norm))
        torch.cuda.synchronize()
    finally:
        seconds=time.perf_counter()-began; telemetry=monitor.finish()
    saved={**p,'model_state':model.state_dict(),'optimizer_state':opt.state_dict(),
        'scheduler_state':{**p['scheduler_state'],'global_step':end},'update':end,'tokens_processed':START+budget,
        'random_state':random_state(device),'phase':48,'experimental':True,'formal_research':False,'promoted':False,
        'arm':arm,'experimental_lr':ARMS[arm],'source_sha256':source_sha,'source_checkpoint':source.relative_to(ROOT).as_posix(),
        'cumulative_experimental_tokens':budget,'maximum_allowed_tokens_per_run':START+512000}
    target.parent.mkdir(parents=True,exist_ok=True); temp=target.with_suffix('.pt.tmp'); torch.save(saved,temp)
    loaded=torch.load(temp,map_location='cpu',weights_only=False)
    integrity=v36.verify_payload(loaded,seed,START+budget,ARMS[arm])
    assert v36.fingerprint(loaded['permutation'])==continuity['permutation']
    assert file_sha256(source)==source_sha
    temp.replace(target)
    result={'phase':48,'arm':arm,'seed':seed,'lr':ARMS[arm],'start_tokens':expected_tokens,'end_tokens':START+budget,
        'budget':BUDGET,'cumulative_budget':budget,'checkpoint':target.relative_to(ROOT).as_posix(),'sha256':file_sha256(target),
        'source_sha256':source_sha,'source_unchanged':True,'continuity':continuity,'integrity':integrity,'cooldown':cooling,
        'train_loss':float(np.mean(losses)),'gradient_mean':float(np.mean(norms)),'gradient_std':float(np.std(norms,ddof=1)),
        'gradient_max':max(norms),'tokens_per_second':BUDGET/seconds,'seconds':seconds,
        'peak_vram_mib':torch.cuda.max_memory_allocated()/1048576,'telemetry':telemetry,
        'parallel_cpu_evaluation':'DISABLED','device':'CUDA','precision':'FP32','amp':False,'eos_weight':1.5,'repetition_auxiliary':False}
    write(training_path(arm,seed,budget),result)
    print({'arm':arm,'seed':seed,'cumulative':budget,'tps':result['tokens_per_second'],'max_temp':telemetry.get('gpu_temperature_c_max'),'integrity':'PASS'},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--preflight',action='store_true'); parser.add_argument('--arm',choices=ARMS)
    parser.add_argument('--seed',type=int,choices=SEEDS); parser.add_argument('--budget',type=int,choices=(256000,512000),default=256000)
    a=parser.parse_args(); torch.set_num_threads(4)
    if a.preflight: preflight()
    else: train(a.arm,a.seed,a.budget)
