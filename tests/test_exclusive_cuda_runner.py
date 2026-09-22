import json
import os
from pathlib import Path
import sys
import threading
import time

import pytest
from training import exclusive_cuda_runner as r
from training import checkpoint_paths
from training.gpu_execution_lock import ExecutionBlocked, sha

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def config(tmp_path,monkeypatch):
    root=tmp_path/'checkpoints';root.mkdir()
    monkeypatch.setenv('UNIPILOT_CHECKPOINT_ROOT',str(root))
    monkeypatch.setattr(checkpoint_paths,'checkpoint_root',lambda repo:root)
    monkeypatch.setattr(r,'heavy_jobs',lambda:[])
    artifact=root/'model-dummy.bin'
    def verifier():
        assert artifact.read_bytes()==b'dummy'
        return {'strict_reload':True,'updates_complete':True,'artifact_sha256':sha(artifact),'artifact_path':str(artifact),
                'test_fixture_only':True,'optimizer_steps':0}
    return dict(cwd=ROOT,root=root,runtime=tmp_path/'runtime',run_id='test',phase=62,
                kind='cuda-dry-run',device='test-no-cuda',repo_head='abc',parent_sha='def',
                verifier=verifier,temperature=lambda:{'temperature_c':40,'hardware_thermal_slowdown':False},
                query=lambda:[],maximum_seconds=15,outputs=[artifact])


def command(config,tail=''):
    p=config['root']/'model-dummy.bin'
    return [sys.executable,'-c',f"from pathlib import Path; Path({str(p)!r}).write_bytes(b'dummy'); "+tail]


def test_state_machine_no_skips_and_receipt_required(tmp_path):
    j=r.Journal(tmp_path/'journal')
    with pytest.raises(ExecutionBlocked,match='TRANSITION'):j.advance('PROCESS_EXITED')
    assert not r.is_complete(j.directory)


@pytest.mark.skipif(os.name!='nt',reason='real Windows Job Object integration')
def test_waits_for_checkpoint_child_and_both_pipe_eofs(config):
    result=[]
    cmd=command(config,"import time; print('before',flush=True); time.sleep(1.2); print('after',flush=True)")
    t=threading.Thread(target=lambda:result.append(r.execute(cmd,**config)))
    start=time.monotonic();t.start()
    artifact=config['root']/'model-dummy.bin'
    while not artifact.exists() and time.monotonic()-start<5:time.sleep(.01)
    assert artifact.exists()
    assert t.is_alive() and not r.is_complete(config['runtime']/'runs/test')
    # A host returning while this thread is alive cannot start a second job.
    with pytest.raises(ExecutionBlocked,match='GPU_LOCK_HELD'):
        r.execute(cmd,**{**config,'run_id':'second'})
    t.join(10);assert not t.is_alive() and result
    assert time.monotonic()-start>=1.2
    assert r.is_complete(config['runtime']/'runs/test')
    assert not (config['runtime']/'gpu.lock').exists()
    assert result[0]['stdout_closed'] and result[0]['stderr_closed']


@pytest.mark.skipif(os.name!='nt',reason='real Windows Job Object integration')
def test_job_object_waits_for_grandchild_after_parent_exit(config):
    cmd=command(config,"import subprocess,sys; subprocess.Popen([sys.executable,'-c','import time;time.sleep(1.2)'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)")
    start=time.monotonic();r.execute(cmd,**config)
    assert time.monotonic()-start>=1.2
    assert r.is_complete(config['runtime']/'runs/test')


@pytest.mark.skipif(os.name!='nt',reason='real Windows Job Object integration')
def test_crash_retains_lock_and_blocks_next_run(config):
    with pytest.raises(ExecutionBlocked,match='CHILD_EXIT_NONZERO'):
        r.execute(command(config,'raise SystemExit(7)'),**config)
    path=config['runtime']/'runs/test'
    assert json.loads((path/'aborted.json').read_text())['status']=='INCOMPLETE'
    assert not r.is_complete(path) and (config['runtime']/'gpu.lock').exists()
    with pytest.raises(ExecutionBlocked,match='PREVIOUS_RUN_INCOMPLETE'):
        r.execute(command(config),**{**config,'run_id':'next','previous':path})


@pytest.mark.skipif(os.name!='nt',reason='real Windows Job Object integration')
def test_missing_verified_artifact_cannot_complete(config):
    config['verifier']=lambda:{'strict_reload':False,'updates_complete':True}
    with pytest.raises(ExecutionBlocked,match='ARTIFACT_VERIFICATION_FAILED'):
        r.execute(command(config),**config)
    assert (config['runtime']/'gpu.lock').exists()


@pytest.mark.parametrize('kind',r.KINDS)
def test_every_gpu_entry_kind_obeys_same_lock(config,kind):
    from training.gpu_execution_lock import GPULock
    lock=GPULock(config['runtime']/'gpu.lock',phase=62,run_id='owner',device='test',command=['dummy'],repo_head='a',parent_sha='b')
    lock.acquire()
    with pytest.raises(ExecutionBlocked,match='GPU_LOCK_HELD'):
        r.execute(command(config),**{**config,'kind':kind})


@pytest.mark.skipif(os.name!='nt',reason='real Windows Job Object integration')
def test_deadline_keeps_live_child_lock_and_has_no_complete(config):
    from training.gpu_execution_lock import job_clear
    with pytest.raises(ExecutionBlocked,match='WAIT_DEADLINE'):
        r.execute(command(config,'import time;time.sleep(.6)'),**{**config,'maximum_seconds':.15})
    path=config['runtime']/'runs/test'
    owner=json.loads((config['runtime']/'gpu.lock').read_text())
    assert not job_clear(owner['job_name']) and not r.is_complete(path)
    time.sleep(.8)
    assert job_clear(owner['job_name']) and (config['runtime']/'gpu.lock').exists()


def test_output_collision_prevents_child_launch(config):
    config['outputs'][0].write_bytes(b'existing')
    with pytest.raises(ExecutionBlocked,match='OUTPUT_COLLISION'):
        r.execute(command(config),**config)
    assert not (config['runtime']/'runs/test/child-created.json').exists()
    assert config['outputs'][0].read_bytes()==b'existing'


@pytest.mark.skipif(os.name!='nt',reason='real Windows Job Object integration')
def test_receipt_deletion_or_artifact_change_revokes_complete(config):
    r.execute(command(config),**config)
    path=config['runtime']/'runs/test'
    assert r.is_complete(path)
    config['outputs'][0].write_bytes(b'changed')
    assert not r.is_complete(path)
    (path/'receipt.json').unlink()
    assert not r.is_complete(path)


def test_cannot_choose_a_second_runtime_lock(config):
    with pytest.raises(ExecutionBlocked,match='SINGLE_RUNTIME_ROOT'):
        r.execute(command(config),**{**config,'runtime':config['runtime']/'other'})


def test_invalid_temperature_does_not_launch(config):
    config['temperature']=lambda:{'temperature_c':float('nan'),'hardware_thermal_slowdown':False}
    with pytest.raises(ExecutionBlocked,match='TEMPERATURE_GATE'):
        r.execute(command(config),**config)
    assert not (config['runtime']/'runs/test/child-created.json').exists()
