"""Copy-only PHASE50 migration. No training, overwrite, move, deletion or regeneration."""
from __future__ import annotations
import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from training.checkpoint_paths import checkpoint_root
from training.run_foundation_v36_lr_review import verify_payload, fingerprint, read_json, write_json
from training.train_foundation_v21_ab import file_sha256
REPORT=ROOT/'evaluation/phase50/z-migration.json'


def relative_record(value):
    path=Path(value)
    if path.is_absolute() or not path.parts or path.parts[0]!='checkpoints' or '..' in path.parts:
        raise ValueError(f'Unsafe historical relative checkpoint record: {value}')
    return Path(*path.parts[1:])


def find_matching(relative, expected, roots):
    found=[]
    for root in roots:
        candidate=(root/relative).resolve()
        if not candidate.is_relative_to(root.resolve()): raise ValueError(candidate)
        if candidate.is_file() and file_sha256(candidate)==expected: found.append(candidate)
    if not found: raise FileNotFoundError(f'No SHA-matching source: {relative}')
    return found[0]


def copy_exclusive(source, destination):
    """Exclusive destination creation. A failed copy is retained and must block retries."""
    destination.parent.mkdir(parents=True,exist_ok=True)
    with source.open('rb') as src, destination.open('xb') as dst:
        shutil.copyfileobj(src,dst,8*1024**2)
        dst.flush();os.fsync(dst.fileno())


def metadata_expectation(relative):
    seed=int(re.search(r'seed-(\d+)',relative.as_posix()).group(1))
    tokens=int(re.search(r'checkpoint-tokens-(\d+)',relative.name).group(1))
    lr=5e-5 if 'arm-C' in relative.parts else 7.5e-5 if 'arm-B' in relative.parts else 1e-4
    return seed,tokens,lr


def validate_payload(p,relative,roots):
    seed,tokens,lr=metadata_expectation(relative)
    integrity=verify_payload(p,seed,tokens,lr)
    # Exercise deserialization of each RNG independently without taking a training step.
    rng=p['random_state']; checks=dict(integrity['checks'])
    py=random.Random();py.setstate(rng['python']);checks['python_rng']=fingerprint(py.getstate())==fingerprint(rng['python'])
    nr=np.random.RandomState();nr.set_state(rng['numpy']);checks['numpy_rng']=fingerprint(nr.get_state())==fingerprint(rng['numpy'])
    gen=torch.Generator(device='cpu');gen.set_state(rng['torch_cpu']);checks['pytorch_rng']=torch.equal(gen.get_state(),rng['torch_cpu'])
    cuda_states=rng['torch_cuda'];checks['cuda_rng']=bool(cuda_states) and len(cuda_states)==torch.cuda.device_count()
    for index,state in enumerate(cuda_states):
        cg=torch.Generator(device=f'cuda:{index}');cg.set_state(state)
        checks['cuda_rng']=checks['cuda_rng'] and torch.equal(cg.get_state().cpu(),state.cpu())
    perm=p['permutation'];checks['sampler_permutation']=perm.ndim==1 and perm.dtype==torch.int64 and len(perm)>p['update'] and bool((perm>=0).all())
    parent=p.get('parent_checkpoint') or p.get('source_checkpoint');parent_sha=p.get('parent_sha256') or p.get('source_sha256')
    if not parent or not parent_sha: raise ValueError(f'Missing parent metadata: {relative}')
    parent_record=Path(parent)
    if parent_record.is_absolute():
        parent_path=parent_record
        if not parent_path.is_file() or file_sha256(parent_path)!=parent_sha: raise ValueError(f'Parent SHA mismatch: {parent}')
    else: parent_path=find_matching(relative_record(parent),parent_sha,roots)
    checks['parent_metadata']=True
    components={k:fingerprint(p[k]) for k in ('model_state','optimizer_state','scheduler_state','permutation','random_state')}
    if not all(checks.values()): raise RuntimeError(f'Resume integrity failed: {checks}')
    return {'pass':True,'checks':checks,'components':components,'metadata':{'seed':seed,'lr':lr,'processed_tokens':tokens,
        'parent_record':parent,'parent_path':str(parent_path),'parent_sha256':parent_sha}}


def run(source_roots):
    if not os.getenv('UNIPILOT_CHECKPOINT_ROOT'): raise RuntimeError('Explicit destination environment required')
    destination_root=checkpoint_root(ROOT);roots=[Path(p).resolve() for p in source_roots]
    pre=read_json(ROOT/'evaluation/phase50/preflight.json')
    previous=read_json(REPORT) if REPORT.exists() else None
    report={'gate':'Z_CHECKPOINT_MIGRATION_BLOCKED','started_at_utc':datetime.now(timezone.utc).isoformat(),
        'destination_root':str(destination_root),'source_roots':[str(p) for p in roots],'free_bytes_before':shutil.disk_usage(destination_root).free,
        'new_gpu_training':False,'rows':[],'missing':[],'copied':previous.get('copied',[]) if previous else [],'protected_files':[]}
    try:
        assert destination_root.drive.upper()=='Z:' and destination_root.name=='checkpoints'
        assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='foundation-research'
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==pre['expected_head']
        assert torch.cuda.is_available()
        inventory=[{'path':str(p),'bytes':p.stat().st_size} for p in destination_root.rglob('*') if p.is_file()]
        report['initial_destination_inventory']=inventory
        suspicious=[r for r in inventory if r['bytes']==0 or re.search(r'\.(partial|tmp|incomplete)$',r['path'])]
        report['initial_partial_artifacts']=suspicious
        assert not suspicious, 'Existing partial or zero-byte destination artifact'
        for r in pre['preserved_files']:
            path=Path(r['path'])
            if path.name.endswith('.READY.json'):
                rel=relative_record(path.relative_to(ROOT));path=find_matching(rel,r['sha256'],roots)
            elif path.name not in ('.gitignore','campus-v21-human-results.json','campus-v21-quick-human-report.md','campus-v21-quick-human-results.json'):
                continue  # Other infra edits predate this resumed turn; never revert them.
            assert file_sha256(path)==r['sha256']
            report['protected_files'].append({'path':str(path),'sha256':r['sha256']})
        candidates=[]
        for record in pre['immutable_checkpoints']:
            rel=relative_record(record['path']);dest=(destination_root/rel).resolve()
            assert dest.is_relative_to(destination_root)
            source=find_matching(rel,record['sha256'],roots)
            if dest.exists() and (dest.stat().st_size!=source.stat().st_size or file_sha256(dest)!=record['sha256']):
                raise FileExistsError(f'Destination collision; refusing overwrite: {dest}')
            candidates.append((record,rel,source,dest))
        report['missing_before']=[str(dest) for _,_,_,dest in candidates if not dest.exists()]
        required_bytes=sum(src.stat().st_size for _,_,src,dst in candidates if not dst.exists())
        assert shutil.disk_usage(destination_root).free-required_bytes>20*1024**3
        write_json(REPORT,report)
        for record,relative,source,dest in candidates:
            payload=torch.load(source,map_location='cpu',weights_only=False)
            before=validate_payload(payload,relative,roots);del payload
            copied=False
            if not dest.exists():
                copy_exclusive(source,dest);copied=True;report['copied'].append(str(dest));write_json(REPORT,report)
            checks={'size_match':source.stat().st_size==dest.stat().st_size,
                'sha_match':file_sha256(source)==file_sha256(dest)==record['sha256']}
            payload=torch.load(dest,map_location='cpu',weights_only=False)
            after=validate_payload(payload,relative,roots);del payload
            checks.update({k+'_match':value==after['components'][k] for k,value in before['components'].items()})
            assert all(checks.values())
            report['rows'].append({'relative_path':record['path'],'source':str(source),'destination':str(dest),'bytes':dest.stat().st_size,
                'sha256':record['sha256'],'copied_this_run':copied,'copy_checks':checks,'resume_state':after,'pass':True})
            write_json(REPORT,report);print('MIGRATION PASS',len(report['rows']),record['path'],flush=True)
        for r in report['protected_files']: assert file_sha256(Path(r['path']))==r['sha256']
        assert file_sha256(ROOT/'data/foundation_v09/evaluation/final-blind-1000.json')==pre['final_blind_sha256']
        report['final_blind_sha256']=pre['final_blind_sha256']
        report['missing']=[];report['free_bytes_after']=shutil.disk_usage(destination_root).free
        report['gate']='Z_CHECKPOINT_MIGRATION_PASS';report['finished_at_utc']=datetime.now(timezone.utc).isoformat()
        write_json(REPORT,report);print(report['gate'],flush=True)
    except Exception as exc:
        report['error']=repr(exc);write_json(REPORT,report);print('Z_CHECKPOINT_MIGRATION_BLOCKED',repr(exc),flush=True);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-root',action='append',required=True);args=parser.parse_args()
    torch.set_num_threads(4);run(args.source_root)
