"""PHASE62 evidence and custody only. No training or model inference."""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.gpu_execution_lock import atomic_json, identity, inventory, sha

OUT = ROOT/'evaluation/phase62'
ZROOT = Path(r'Z:\AI\unipilot-mini\checkpoints')
RAW = ZROOT.parent/'evaluation/phase62'
START = '0a31cdbfbd1d18177ff998496b846e63b09e615e'
SESSION = Path(r'C:\Users\nlgid\.codex\sessions\2026\09\05\rollout-2026-09-05T02-05-45-01a06d62-2d48-7de2-805d-647a20707725.jsonl')
OWN = {'training/gpu_execution_lock.py','training/exclusive_cuda_runner.py','scripts/phase62_audit.py',
       'scripts/run_phase62_qa.py','tests/test_gpu_execution_lock.py','tests/test_exclusive_cuda_runner.py'}

def read(p): return json.loads(Path(p).read_text(encoding='utf8'))
def git(*args): return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
def artifact(name, value): atomic_json(OUT/name, {'phase':62,'new_training':False,'optimizer_steps':0,**value})
def baseline_status():
    rows=[]
    for line in subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=all'],cwd=ROOT,text=True).splitlines():
        rel=line[3:]
        if rel in OWN or rel.startswith('evaluation/phase62/') or rel.startswith('evaluation/foundation-v51-'): continue
        p=ROOT/rel
        rows.append({'status':line[:2],'path':rel,'sha256':sha(p) if p.is_file() else None})
    return rows

def preserve():
    pre=read(OUT/'preflight.json')
    if baseline_status()!=pre['dirty']: raise RuntimeError('EXISTING_DIRTY_CHANGED')
    for row in pre['protected']+pre['frozen']:
        if sha(Path(row['path']))!=row['sha256']: raise RuntimeError('PRESERVATION_FAILED:'+row['path'])
    return True

def preflight():
    if git('branch','--show-current')!='foundation-research' or git('rev-parse','HEAD')!=START or git('ls-remote','origin','refs/heads/foundation-research').split()[0]!=START: raise RuntimeError('PHASE62_PREFLIGHT_BLOCKED')
    if os.environ.get('UNIPILOT_CHECKPOINT_ROOT')!=str(ZROOT): raise RuntimeError('PROCESS_ENV_RESOLVER_MISMATCH')
    from training.checkpoint_paths import checkpoint_root
    if checkpoint_root(ROOT)!=ZROOT: raise RuntimeError('PROCESS_ENV_RESOLVER_MISMATCH')
    protected=read(ROOT/'evaluation/phase61/preflight.json')['protected_files']
    frozen=[]
    for base in (ROOT/'evaluation/phase61', ZROOT/'experimental/phase61'):
        for p in sorted(base.rglob('*')):
            if p.is_file():frozen.append({'path':str(p),'sha256':sha(p)})
    for p in [ROOT/'evaluation/phase60/phase61-continuation-stability-preregistration.json', ROOT/'evaluation/foundation-v50-continuation-stability-report.md',ROOT/'evaluation/foundation-v50-continuation-stability-summary.json']:
        frozen.append({'path':str(p),'sha256':sha(p)})
    rows=inventory()
    value={'at':datetime.now(timezone.utc).isoformat(),'cwd':str(ROOT),'branch':'foundation-research','head':START,'origin':START,'main':git('rev-parse','main'),
           'checkpoint_root':str(ZROOT),'free_bytes':shutil.disk_usage(ZROOT).free,
           'dirty':baseline_status(),'protected':protected,'frozen':frozen,'gpu_inventory':rows,
           'processes':[identity(r['pid']) for r in rows]}
    artifact('preflight.json',value);RAW.mkdir(parents=True,exist_ok=False)
    preserve();print('PHASE62_PREFLIGHT_PASS',len(value['dirty']),flush=True)

def forensic():
    preserve();events=[]
    # Read only this task's execution records in the incident window; no unrelated
    # conversation content or dataset bodies are copied into the report.
    with SESSION.open(encoding='utf8') as f:
        for line in f:
            record=json.loads(line)
            stamp=record.get('timestamp','')
            if not ('2026-09-19T14:10:00' <= stamp < '2026-09-19T14:16:00'): continue
            payload=record.get('payload',{})
            if record.get('type')=='event_msg' and payload.get('item',{}).get('type')=='CommandExecution':
                item=payload['item']
                events.append({'timestamp':stamp,'event_type':payload['type'],**{k:v for k,v in item.items() if k in ('id','process_id','command','cwd','status','stdout','stderr','exit_code','started_at_ms','completed_at_ms','duration_ms')}})
    atomic_json(RAW/'host-command-events.json',events)
    compact=[{k:v for k,v in x.items() if k not in ('stdout','stderr')} for x in events]
    artifact('host-timeline.json',{'events':compact,'raw_path':str(RAW/'host-command-events.json'),'raw_sha256':sha(RAW/'host-command-events.json')})
    print(json.dumps(compact,ensure_ascii=False),flush=True)

def custody():
    preserve()
    import random
    import numpy as np
    import torch
    from training.run_foundation_v36_lr_review import verify_payload,fingerprint
    torch.set_num_threads(2)
    rows=[]
    for seed,arm,lr,status in [(42,'control',5e-5,'COMPLETED_BEFORE_ABORT'),(42,'half-lr',2.5e-5,'COMPLETED_BEFORE_ABORT'),(123,'control',5e-5,'INCOMPLETE_STUDY_RUN')]:
        path=ZROOT/f'experimental/phase61/continuation-stability/{arm}/seed-{seed}/checkpoint-tokens-16446464.pt'
        digest=sha(path);p=torch.load(path,map_location='cpu',weights_only=False)
        parent=torch.load(ZROOT/f'experimental/phase48/arm-C/seed-{seed}/checkpoint-tokens-16384000.pt',map_location='cpu',weights_only=False)
        check=verify_payload(p,seed,16446464,lr)
        rng=p['random_state'];random.Random().setstate(rng['python']);np.random.RandomState().set_state(rng['numpy']);torch.Generator(device='cpu').set_state(rng['torch_cpu'])
        extra={'full_permutation':fingerprint(p['permutation'])==fingerprint(parent['permutation']),
               'scheduler_preserved_except_step':{k:v for k,v in p['scheduler_state'].items() if k!='global_step'}=={k:v for k,v in parent['scheduler_state'].items() if k!='global_step'},
               'python_numpy_cpu_rng_load':True,'cuda_rng_present':bool(rng['torch_cuda']),
               'cuda_rng_byte_tensors':all(t.dtype==torch.uint8 and t.ndim==1 and t.numel()>0 for t in rng['torch_cuda']),
               'all_adam_moments':all({'exp_avg','exp_avg_sq','step'}<=set(v) for v in p['optimizer_state']['state'].values()),
               'parent_link':p['parent_checkpoint_sha256']==sha(ZROOT/f'experimental/phase48/arm-C/seed-{seed}/checkpoint-tokens-16384000.pt'),
               'immutable':sha(path)==digest}
        if not all(extra.values()):raise RuntimeError(str(extra))
        rows.append({'seed':seed,'arm':arm,'path':str(path),'sha256':digest,'size':path.stat().st_size,'mtime_utc':datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat(),
                     'processed_tokens':p['tokens_processed'],'optimizer_steps':sorted(set(int(x['step']) for x in p['optimizer_state']['state'].values())),
                     'integrity':check,'extra':extra,'cuda_rng_validation':'STRUCTURAL_ONLY_NO_CUDA_CONTEXT_CREATED',
                     'study_status':status,'custody_labels':['EXPERIMENTAL','NOT_CANONICAL','NOT_PRODUCTION','INVALID_STUDY_CONTEXT'],
                     'binary_labels_modified':False,'efficacy_use':False})
    artifact('phase61-checkpoint-custody.json',{'checkpoints':rows,'remaining_runs':'NOT_RUN','phase61':'EXPERIMENT_INVALID','efficacy_claims':'NONE','half_lr_efficacy':'UNDETERMINED'})
    preserve();print('CUSTODY 3/3 PASS; no inference / optimizer steps',flush=True)


def forensic_detail():
    selected={};records=[];timing=[]
    with SESSION.open(encoding='utf8') as f:
        for line in f:
            r=json.loads(line);stamp=r.get('timestamp','')
            if not ('2026-09-19T14:10:50'<=stamp<'2026-09-19T14:14:00'):continue
            p=r.get('payload',{})
            if r.get('type')=='response_item' and p.get('type')=='custom_tool_call' and p.get('name')=='exec':
                text=p.get('input','')
                if ('p.train_one(42' in text or "phase61.py' train" in text) and 'exec_command' in text:
                    selected[p['call_id']]=stamp
                    records.append({'timestamp':stamp,'kind':'assistant_launch_request','call_id':p['call_id'],'code':text})
            elif p.get('type')=='custom_tool_call_output' and p.get('call_id') in selected:
                records.append({'timestamp':stamp,'kind':'model_visible_return','call_id':p['call_id'],'output':p.get('output')})
            if r.get('type')=='event_msg' and p.get('item',{}).get('type')=='CommandExecution':
                item=p['item'];command=' '.join(item.get('command',[]))
                if 'p.train_one(42' in command or "phase61.py' train" in command or 'Stop-Process -Force' in command:
                    timing.append({'timestamp':stamp,'event_type':p.get('type'),'exec_session_identifier':item.get('process_id'),
                                   'command':command,'status':item.get('status'),'exit_code':item.get('exit_code'),
                                   'started_at_ms':p.get('started_at_ms'),'completed_at_ms':p.get('completed_at_ms')})
    atomic_json(RAW/'assistant-launch-and-return-evidence.json',records)
    artifact('phase61-process-forensic.json',{
        'gate':'MIXED_PROCESS_LIFECYCLE_FAILURE',
        'primary_mechanism':'HOST_RETURN_BEFORE_CHILD_EXIT; output-only tool wrapper discarded lifecycle/session metadata; assistant launched two duplicate training calls without waiting',
        'secondary_mechanisms':['No common GPU lock or durable run-start claim','Output collision checked only after the 122-update loop','Receipt publication was inside a still-running multi-run process','PowerShell trailing Write-Output masked a child exit code 1 as shell exit code 0'],
        'host_return_means':'asynchronous yield, NOT proof of process termination; no evidence of a forced 30-second kill',
        'tree_A':{'shell_pid':13160,'venv_launcher_pid':15136,'python_child_pid':2792,'start_local':'2026-09-19T23:11:04+09:00','command':'python evaluation/run_foundation_v50_phase61.py train','end_bound_utc':'2026-09-19T14:13:55.927Z','exact_os_exit_timestamp':None},
        'tree_B':{'shell_pid':7960,'venv_launcher_pid':18700,'python_child_pid':15076,'start_local':'2026-09-19T23:12:48+09:00','command':'third invocation: python -u -c ... p.train_one(42, control, 5e-5)','association':'inferred from timestamp and logged command; CIM command line blank','exact_os_exit_timestamp':None,'host_command_completion_utc':'2026-09-19T14:13:48.469Z'},
        'additional_duplicate':{'command':'second invocation: python -u -c ... p.train_one(42, control, 5e-5)','os_pid':None,'host_command_completion_utc':'2026-09-19T14:13:05.973Z','python_exit_code':1,'failure':'FileExistsError in save_checkpoint AFTER train_one finished 122 updates'},
        'overlap_runs':'Original seed42 Half-LR / seed123 Control overlapped duplicate seed42 Control executions; exact per-update time intersection is unavailable',
        'proven_extra_optimizer_updates':244,'proven_minimum_total_updates':610,
        'count_basis':'Two duplicate tracebacks reached save_checkpoint after fixed 122-update loops, plus three saved 122-update original runs. No reliable complete ledger for any additional in-memory progress.',
        'cuda_device':'RTX 2070 SUPER / cuda:0 from frozen runner; per-PID device ownership was not recorded',
        'effective_cwd':str(ROOT),'host_cwd':'C:/Users/nlgid/Documents/Codex/2026-09-05/yaml-phase-38-c-users-nlgid',
        'stdout_stderr':'Initial partial stdout only was surfaced. Later persisted execution completion events contain two FileExistsError tracebacks and original TRAIN_PASS lines. No separate historic EOF or GPU release receipt exists.',
        'candidates':{'A_host_early_return':'OBSERVED_ASYNC_YIELD','B_detach_bug':'NOT_ESTABLISHED','C_missing_wait':'CONFIRMED_AT_ASSISTANT_ORCHESTRATION','D_shell_exit_before_child':'NOT_ESTABLISHED','E_background_creation':'NO_EXPLICIT_BACKGROUND_TRAIN_COMMAND_FOUND','F_Start_Process':'NOT_USED_FOR_THE_TRAINING_LAUNCHES','G_multiprocessing_orphan':'NO_EVIDENCE; venv launcher/child is not evidence of multiprocessing','H_runtime_timeout':'YIELD_OBSERVED; forced termination not established'},
        'timeline':timing,'launch_return_raw':str(RAW/'assistant-launch-and-return-evidence.json'),'launch_return_sha256':sha(RAW/'assistant-launch-and-return-evidence.json'),
        'checkpoint_timestamps':read(OUT/'phase61-checkpoint-custody.json')['checkpoints'],
        'receipt_timestamps':[{'path':str(p),'mtime_utc':datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat()} for p in (ROOT/'evaluation/phase61').glob('run-*.json')],
        'phase61_status':'EXPERIMENT_INVALID','efficacy':'UNDETERMINED','valid_efficacy_claims':'NONE','historical_gates_recomputed':False})
    print(json.dumps({'launches':len(selected),'timing':timing}),flush=True)


def register():
    preserve()
    from training.exclusive_cuda_runner import STATES,KINDS
    qa=[read(OUT/name) for name in ('pytest-targeted-verified.json','pytest-full.json')]
    if any(x['exit_code']!=0 or x['failed'] or x['errors'] or not x['preservation_before_after'] for x in qa):raise RuntimeError('QA_NOT_PASS')
    sources={p:sha(ROOT/p) for p in ('training/gpu_execution_lock.py','training/exclusive_cuda_runner.py',
                                    'tests/test_gpu_execution_lock.py','tests/test_exclusive_cuda_runner.py')}
    lock_contract={'schema_version':'single-gpu-lock-contract-v1','path':str(ZROOT.parent/'runtime/gpu.lock'),
        'source_sha256':sources['training/gpu_execution_lock.py'],'atomic_exclusive_create':True,
        'device_scope':'single host, all CUDA work kinds, shared runtime derived from checkpoint root parent',
        'owner_fields':['schema_version','pid','parent_pid','hostname','process_start_time','phase','run_id','cuda_device','command_hash','repo_head','checkpoint_parent_sha256','created_timestamp','nonce','job_name'],
        'held_result':'GPU_LOCK_HELD','automatic_expiry':False,'automatic_cleanup':False,'automatic_kill':False,
        'stale_cleanup':'Explicit fresh verification under the mutation guard: local owner PID absent AND start identity mismatch AND no foreign compute/GPU context AND known children absent AND Job Object clear AND persisted child-tree exit evidence (or child never started). Archive the verification before removing exactly this stale lock.',
        'crashed_tree_uncertainty':'If owner crashed after starting a child without persisting child-tree exit, helper refuses cleanup even when nvidia-smi is empty. A separate forensic verification protocol is required; never guess or force-unlock.',
        'pid_reuse':'Different creation time identifies another process; live reused PID still blocks cleanup.',
        'release':'Only exact owner identity+nonce; verified normal completion and GPU release; persistent OS mutation guard serializes acquire/release/cleanup.',
        'display_policy':'Pure G rows excluded. C rows never display-exempt. C+G excluded only with explicitly approved exact PID+creation time+hostname+exe baseline; no wildcard/basename exemptions. Unknown types/failed query/races fail closed.',
        'current_WDDM_observation':'NVIDIA reports display applications as C+G with N/A memory. This is ambiguous, not proof that every row is a training process or proof that GPU is free.',
        'display_allowlist_in_phase62':[],'scope':list(KINDS)}
    ownership={'schema_version':'process-ownership-contract-v1','source_sha256':sources['training/exclusive_cuda_runner.py'],
        'states':list(STATES),'state_journal':'exclusive append-only numbered artifacts; no state skipping; completion predicate requires RELEASED and receipt SHA plus current artifact SHA',
        'launch':'Windows Popen shell=False, CREATE_SUSPENDED; persist child identity, assign non-breakaway Job Object, then resume. Exactly one child tree per supervisor invocation.',
        'wait':'OS process exit code 0 AND Job Object membership empty AND both captured pipes reached EOF and closed AND no child GPU contexts AND trusted artifact strict verifier PASS.',
        'deadline_seconds':3600,'deadline_action':'INCOMPLETE/ABORTED ownership receipt; retain GPU lock and live Job handle; never terminate/kill; never start another run.',
        'stdout_stderr':'separate flushed logs, joined drain threads, SHA in final receipt; shell status is not substituted for Python exit code',
        'host_early_return':'Host yield is progress only. Preserve full exec result including session_id, poll write_stdin to terminal OS result, and verify RELEASED journal+receipt before next run. Never print only result.output. Direct legacy training invocation after an unobserved yield is prohibited.',
        'host_parent_dies':'ABORTED, lock retained, no automatic child termination or next run.',
        'checkpoint_receipt_atomicity':'No cross-file atomic transaction is claimed. Child saves exclusively+atomically; trusted verifier strict-reloads; supervisor observes process/tree/pipe/GPU exit before publishing receipt. RELEASED is the final commit marker. Checkpoint alone is incomplete.',
        'next_run':'previous COMPLETE with valid receipt and artifact SHA, previous RELEASED, acquire single lock, foreign CUDA guard, temperature<=60, no full pytest/build/browser QA',
        'cpu_heavy_jobs':'prelaunch command inventory refuses pytest/npm build/playwright; external noncooperating tools cannot be forcibly prevented',
        'legacy_runner_policy':'Frozen historical source files remain unchanged. PHASE61 train_one/train must never be called for clean replication; they contain no ownership protection. All future GPU study actions must use this supervisor API.',
        'worker_integration':'Future PHASE63 implementation supplies a bounded one-run worker and trusted strict artifact validator. It must be hashed, QA-verified and bound in an execution receipt before any optimizer step; this registration is not a training command or authorization.'}
    artifact('gpu-lock-contract.json',lock_contract)
    artifact('process-ownership-contract.json',ownership)
    oldpath=ROOT/'evaluation/phase60/phase61-continuation-stability-preregistration.json';old=read(oldpath)
    if sha(oldpath)!='1f21f56d345e34a2aa58df7180ed40f5b3071feacab75cd31385b7fd2ffa89fc':raise RuntimeError('PREREGISTRATION_CHANGED')
    scientific_keys=('purpose','intervention','mechanistic_rule','seeds','arms','run_order','parents','objective','optimizer','scheduler','sampler','budget','data','matching_no_grad_forward','hardware','evaluation','decision_rules','sealed_sets','approved_lr_status','generation_policy')
    spec=copy.deepcopy(old)
    spec.update({'phase':63,'registered_in_phase':62,'schema_version':'phase63-clean-continuation-stability-preregistration-v1',
                 'status':'REGISTERED_REQUIRES_NEW_USER_TRAINING_AUTHORIZATION','training_authorized':False,'training_executed':False,
                 'new_training':False,'source_phase60_preregistration':str(oldpath.relative_to(ROOT)),
                 'source_phase60_preregistration_sha256':sha(oldpath),'phase61_status':'EXPERIMENT_INVALID',
                 'phase61_efficacy':'UNDETERMINED','phase61_checkpoints_reused':False,
                 'scientific_design_changed':False,'execution_design_changed':True,'scientific_keys_unchanged':list(scientific_keys),
                 'execution_source_sha256':sources,'gpu_lock_contract_sha256':sha(OUT/'gpu-lock-contract.json'),
                 'process_ownership_contract_sha256':sha(OUT/'process-ownership-contract.json'),
                 'execution_contract':ownership,'gpu_lock_contract':lock_contract,
                 'raw_root':str(ZROOT.parent/'evaluation/phase63'),'new_receipts_root':str(ZROOT.parent/'runtime/runs/phase63-*'),
                 'future_live_preflight':['explicit new training authorization','original parent hashes and full strict state reload','all source and schema hashes','6 execution-context live evaluator dry runs under same lock before any update','34/34 concrete comparisons','GPU process inventory and explicitly verified display exceptions','fresh output+receipt+run-start paths','worker/validator SHA bound in execution receipt','OS lifecycle test with full session metadata and receipt polling'],
                 'retry_after_started_run':False,'process_violation_invalidates_entire_study':True})
    spec['checkpoint_policy']['new_relative_root']='experimental/phase63/continuation-stability/{arm}/seed-{seed}'
    # Only orchestration adds constraints; all numerical treatment and thresholds match.
    assert all(spec[k]==old[k] for k in scientific_keys)
    for source,digest in old['source_sha256'].items():
        if sha(ROOT/source)!=digest:raise RuntimeError('FROZEN_SCIENTIFIC_SOURCE_CHANGED:'+source)
    for seed in spec['seeds']:
        for arm in ('control','half-lr'):
            p=ZROOT/f'experimental/phase63/continuation-stability/{arm}/seed-{seed}'
            if p.exists():raise RuntimeError('PHASE63_OUTPUT_COLLISION')
    path=OUT/'phase63-clean-continuation-stability-preregistration.json';atomic_json(path,spec)
    artifact('phase63-registration-receipt.json',{'preregistration_sha256':sha(path),'scientific_keys_equal':list(scientific_keys),
        'scientific_design_changed':False,'execution_design_changed':True,'training_authorized':False,'training_executed':False,
        'source_phase60_sha256':sha(oldpath),'source_hashes_pass':True,'fresh_paths_absent':True})
    print('PHASE63_REGISTERED; training_authorized=false',flush=True)


def finish():
    preserve();qa={scope:read(OUT/filename) for scope,filename in [('targeted','pytest-targeted-verified.json'),('full','pytest-full.json')]}
    assert all(x['exit_code']==0 and x['failed']==0 and x['errors']==0 for x in qa.values())
    forensic=read(OUT/'phase61-process-forensic.json');custody=read(OUT/'phase61-checkpoint-custody.json')
    reg=read(OUT/'phase63-registration-receipt.json')
    summary={'phase':62,'gate':'EXECUTION_HARDENING_QA_PASS','phase61':'EXPERIMENT_INVALID',
        'forensic_gate':forensic['gate'],'root_cause':forensic['primary_mechanism'],
        'additional_duplicate_updates_proven':244,'minimum_phase61_updates_proven':610,
        'phase61_efficacy_claims':'NONE','half_lr_validated':False,'checkpoint_custody':'3/3 PASS; no binary mutation',
        'hardening':{k:'PASS' for k in ('exclusive_lock','atomic_lock','pid_creation_time_ownership','inventory_parser','child_wait','no_early_complete','crash_fail_closed','receipt_contract','state_machine')},
        'scope_limitations':['Lifecycle tests use real Windows CPU-only dummy child/grandchild processes; zero CUDA training was executed.',
            'Legacy runners remain frozen and unguarded; direct use for PHASE63 is prohibited. New bounded worker must be integrated and verified before future execution.',
            'WDDM C+G/N/A display entries require explicit identity-based review at future preflight; no blanket exemption was created.',
            'Hard crash without child-tree exit evidence intentionally leaves cleanup blocked. No live process is killed.'],
        'phase63_preregistration':'CREATED','phase63_sha256':reg['preregistration_sha256'],'scientific_design_changed':False,'execution_design_changed':True,
        'phase63_training_authorized':False,'phase63_training_executed':False,'approved_lr':5e-5,'half_lr_status':'UNVALIDATED STABILIZATION CANDIDATE',
        'generation_policy':'UNSAFE','phase57':'EXPERIMENT_INVALID','phase59':'CONTROL_STABILITY_MIXED',
        'new_training':False,'optimizer_steps':0,'canonical':False,'20m':False,'foundation_base':False,
        'reserve2':'SEALED_UNSCORED','final_blind':'SHA_ONLY_NO_BODY_ACCESS','production':False,'web_changed':False,
        'qa':qa,'preserved':{'protected4':True,'READY5':True,'dirty_JSON5':True,'existing_dirty_179':True,'phase61_artifacts':True,'checkpoints':True},
        'main_unchanged':git('rev-parse','main')==read(OUT/'preflight.json')['main'],
        'Render_Production_unchanged':True,'Vercel_Production_unchanged':True}
    atomic_json(ROOT/'evaluation/foundation-v51-execution-hardening-summary.json',summary)
    lines=['# PHASE62 / Foundation v5.1 — Single-GPU execution hardening','',
        'PHASE61 remains **EXPERIMENT_INVALID**. The 2.5e-5 efficacy question is **UNDETERMINED**. PHASE62 performed zero training, zero optimizer updates and no generation/quality evaluation.',
        '', '## Forensic findings','',
        'Gate: MIXED_PROCESS_LIFECYCLE_FAILURE. The host returned partial command output while its OS process remained alive. The assistant wrapper printed only result.output, dropping session/exit information, and the assistant launched two duplicate seed42 Control calls without waiting. This was an orchestration error, not evidence that the host killed training at 30 seconds.',
        'Later execution records show both duplicate calls completed the fixed122-update loop and then failed at the existing checkpoint path. This proves244 additional duplicate optimizer updates and at least610 total updates across the three saved original runs plus two duplicates. Exact total in-memory progress after the third saved checkpoint cannot be established. These duplicate runs were missing from the PHASE61 final report.',
        '', '| Command | Launch (UTC) | Model-visible partial return (UTC) | Actual host completion (UTC) |',
        '|---|---|---|---|',
        '| Original six-run command | 14:11:04.144 | 14:11:34.374 | 14:13:55.927; exit1 after stop |',
        '| Duplicate seed42 Control #1 | 14:12:04.425 | 14:12:34.603 | 14:13:05.973; Python exit1 |',
        '| Duplicate seed42 Control #2 | 14:12:47.855 | 14:13:18.057 | 14:13:48.469; Python exit1 masked by shell exit0 |',
        '', 'All dates are2026-09-19. The recorded process tree A was shell13160 → venv launcher15136 → Python2792; tree B was shell7960 → launcher18700 → Python15076. Tree B is associated with duplicate#2 by creation time; its CIM command line was unavailable. Duplicate#1 OS PID and exact OS process exit timestamps were not recorded. Exec session IDs are not OS PIDs.',
        'Original seed42 Half-LR and seed123 Control overlapped duplicate seed42 Control execution. Per-update wall timestamps and per-PID CUDA-device inventories were not captured, so exact optimizer-step intersections cannot be reconstructed. Stdout completion logs, command bounds and filesystem save times are retained as evidence, not invented precision.',
        '', '## Checkpoint custody','', '| Run | SHA256 | Frozen status |','|---|---|---|']
    for x in custody['checkpoints']:lines.append(f"| {x['seed']} {x['arm']} | {x['sha256']} | {x['study_status']} |")
    lines+=['','All three pass strict model/optimizer reload, finite state,16,446,464 tokens,32122 optimizer/scheduler step and unchanged full parent permutation. Python/NumPy/Torch CPU RNG loading passes. CUDA RNG byte structure was validated on CPU; no live CUDA RNG roundtrip or model inference is claimed. Seed123 Control remains incomplete because its run receipt is absent. Labels EXPERIMENTAL / NOT_CANONICAL / NOT_PRODUCTION / INVALID_STUDY_CONTEXT live in the custody manifest; binaries and PHASE61 receipts were not modified.',
        '', '## Execution contract','',
        'All four new GPU execution kinds share one lock under checkpoint_root.parent/runtime. Atomic exclusive creation, PID+creation time+hostname ownership, and an OS mutation guard prevent double starts and wrong-owner removal. A suspended child is assigned to a Windows Job Object before it runs. Job accounting includes descendants; receipt publication waits for exit0, empty child tree, stdout/stderr EOF, released CUDA contexts, and a strict verified artifact hash.',
        'The state sequence is PREPARED → LOCKED → RUNNING → CHECKPOINT_SAVED → PROCESS_EXITED → RECEIPT_WRITTEN → RELEASED. The checkpoint state records a verified durable artifact observation; actual child save precedes its exit. A receipt without RELEASED, or a checkpoint without a receipt, is incomplete. There is no cross-file transaction claim.',
        'Crash/timeout retains the ownership lock and an INCOMPLETE/ABORTED record, without killing any live process. Stale cleanup requires a new explicit verification, owner PID absence, creation-time mismatch, clear GPU inventory, clear job tree and persisted child exit evidence (or no child ever started). An unresolved orphan-tree history blocks cleanup.',
        'Live NVIDIA inspection on this WDDM host returns C+G/N/A for display applications. Pure graphics rows are excluded; unknown compute rows block. C+G exceptions require exact current process identity and executable review. This phase creates no broad allowlist and does not claim the current ambiguous list is a clean training preflight.',
        '', '## PHASE63 registration','',
        'CREATED, training_authorized=false, training_executed=false. Original PHASE48 parents; fresh six runs in order42 Control/Half-LR,123 Control/Half-LR,2026 Control/Half-LR. LR5e-5 versus2.5e-5;122 updates each. Scientific keys, evaluation sets,34 safeguards and decision logic match the immutable PHASE60 registration exactly. Only execution controls/output phase change. New roots are experimental/phase63/continuation-stability and Z:/AI/unipilot-mini/evaluation/phase63.',
        'Registration freezes the tested lock and supervisor SHA. Future execution still requires explicit authorization and a bounded one-run worker/strict validator integrated with this supervisor and SHA-bound before optimizer step. Frozen PHASE61 train/train_one entry points are not safe launch paths and must not be reused directly.',
        '', '## QA and preservation','']
    for scope,x in qa.items():lines.append(f"- {scope}: exit{x['exit_code']}, passed{x['passed']}, failed{x['failed']}, errors{x['errors']}, skipped{x['skipped']}, warnings{x['warnings']}. Logs/JUnit hashes are in the QA receipts.")
    lines+=['','Windows dummy-process tests verify concurrent rejection, PID reuse, stale/live owner checks, child and grandchild wait, no completion merely from checkpoint existence, crash/deadline handling, receipt requirement, inventory parsing and state transitions. They do not train on GPU.',
        'The intermediate targeted-final receipt records a test-only NameError; it was fixed before targeted-verified and full QA. Earlier receipts are retained rather than overwritten.',
        'Protected4, READY5, dirty JSON5, all179 existing dirty paths, PHASE61 evidence and all checkpoint bytes remain unchanged. Raw logs and checkpoint binaries remain outside Git staging. Main and Render/Vercel Production are unchanged. Web was not edited; npm build/browser QA were not required.',
        'Approved LR5e-5;2.5e-5 UNVALIDATED; Generation Policy UNSAFE; PHASE57 EXPERIMENT_INVALID; PHASE59 CONTROL_STABILITY_MIXED; PHASE61 EXPERIMENT_INVALID. Canonical/20M/Foundation Base/Production:NO.']
    path=ROOT/'evaluation/foundation-v51-execution-hardening-report.md'
    with path.open('x',encoding='utf8') as f:f.write('\n'.join(lines)+'\n')
    artifact('final-preservation.json',{'pass':preserve(),'dirty_count':179,'phase61_checkpoints_unmodified':True,
        'raw_inventory':[{'path':str(p),'sha256':sha(p)} for p in sorted(RAW.iterdir()) if p.is_file()],
        'new_training':False,'optimizer_steps':0})
    print('PHASE62_REPORT_COMPLETE',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('preflight','forensic','forensic_detail','custody','preserve','register','finish'));args=parser.parse_args()
    globals()[args.action]()
