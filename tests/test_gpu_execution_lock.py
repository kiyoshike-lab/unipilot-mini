import concurrent.futures
import copy
import json
import os
import socket
import subprocess
import sys
import time

import pytest
from training import gpu_execution_lock as g


def make(path, run='dummy'):
    return g.GPULock(path, phase=62, run_id=run, device='test-no-cuda',
                     command=['dummy'], repo_head='abc', parent_sha='parent')


def test_exclusive_double_start(tmp_path):
    path=tmp_path/'gpu.lock'
    def acquire(i):
        try:return make(path,str(i)).acquire()
        except g.ExecutionBlocked as e:return str(e)
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        results=list(pool.map(acquire,range(2)))
    assert sum(isinstance(x,dict) for x in results)==1
    assert results.count('GPU_LOCK_HELD')==1


def test_cross_process_double_start(tmp_path):
    path=tmp_path/'gpu.lock';lock=make(path);lock.acquire()
    command=[sys.executable,'-c',f"from training.gpu_execution_lock import GPULock; GPULock({str(path)!r},phase=62,run_id='other',device='test',command=['x'],repo_head='a',parent_sha='b').acquire()"]
    value=subprocess.run(command,capture_output=True,text=True,timeout=10)
    assert value.returncode!=0 and 'GPU_LOCK_HELD' in value.stderr
    assert json.loads(path.read_text())==lock.owner


def test_live_owner_cannot_cleanup_or_release_incomplete(tmp_path):
    path=tmp_path/'gpu.lock';lock=make(path);lock.acquire()
    proof=g.stale_verification(path,[])
    assert not proof['stale_verified']
    with pytest.raises(g.ExecutionBlocked,match='STALE_CLEANUP_DENIED'):
        g.cleanup_stale(path,proof,query=lambda:[],archive=tmp_path/'audit.json')
    with pytest.raises(g.ExecutionBlocked,match='INCOMPLETE'):
        lock.release(complete=False,gpu_clear=True)
    assert path.exists()
    lock.release(complete=True,gpu_clear=True)
    assert not path.exists()


def test_stale_requires_explicit_verification_and_rechecks(tmp_path,monkeypatch):
    path=tmp_path/'gpu.lock';make(path).acquire()
    monkeypatch.setattr(g,'identity',lambda pid=None:None)
    proof=g.stale_verification(path,[])
    assert proof['stale_verified']
    with pytest.raises(g.ExecutionBlocked):
        g.cleanup_stale(path,{},query=lambda:[],archive=tmp_path/'bad.json')
    g.cleanup_stale(path,proof,query=lambda:[],archive=tmp_path/'cleanup.json')
    assert not path.exists() and (tmp_path/'cleanup.json').exists()


def test_pid_reuse_does_not_identify_original_owner_or_allow_cleanup(tmp_path,monkeypatch):
    path=tmp_path/'gpu.lock';owner=make(path).acquire();reused={**owner,'process_start_time':owner['process_start_time']+1}
    assert not g.same_process(owner,reused)
    monkeypatch.setattr(g,'identity',lambda pid=None:reused)
    proof=g.stale_verification(path,[])
    assert proof['checks']['start_identity_mismatch'] and not proof['stale_verified']


def test_other_owner_cannot_unlock(tmp_path):
    path=tmp_path/'gpu.lock';lock=make(path);lock.acquire()
    with pytest.raises(g.ExecutionBlocked,match='OWNER_IDENTITY'):
        make(path,'other').release(complete=True,gpu_clear=True)
    assert path.exists()


def xml(kind='C',name='python.exe',pid=123):
    return f'<nvidia_smi_log><gpu><uuid>GPU-x</uuid><processes><process_info><pid>{pid}</pid><type>{kind}</type><process_name>{name}</process_name><used_memory>N/A</used_memory></process_info></processes></gpu></nvidia_smi_log>'


def test_inventory_type_and_memory_unknown_are_not_silently_safe(monkeypatch):
    live={'pid':123,'hostname':socket.gethostname(),'process_start_time':1,'exe':'python.exe'}
    monkeypatch.setattr(g,'identity',lambda pid=None:live)
    row=g.parse_inventory(xml())[0]
    assert row['gpu_memory']=='N/A' and row['pid']==123 and row['timestamp']>0
    assert g.foreign_processes([row])
    assert not g.foreign_processes(g.parse_inventory(xml('G')))
    assert g.foreign_processes(g.parse_inventory(xml('C+G')))
    assert not g.foreign_processes(g.parse_inventory(xml('C+G')),display=[live])
    assert g.foreign_processes(g.parse_inventory(xml('C')),display=[live])
    assert g.foreign_processes(g.parse_inventory(xml('C+G')),display=[{**live,'process_start_time':2}])
    with pytest.raises(g.ExecutionBlocked):g.parse_inventory(xml('Unknown'))
    with pytest.raises(g.ExecutionBlocked):g.parse_inventory('<nvidia_smi_log/>')


def test_atomic_publication_never_overwrites(tmp_path):
    p=tmp_path/'receipt.json';g.atomic_json(p,{'first':True})
    with pytest.raises(FileExistsError):g.atomic_json(p,{'first':False})
    assert json.loads(p.read_text())=={'first':True}


def test_stale_cleanup_requires_persisted_child_exit(tmp_path,monkeypatch):
    path=tmp_path/'gpu.lock';owner=make(path).acquire()
    g.atomic_json(tmp_path/'runs'/owner['run_id']/'child-created.json',{'pid':99999})
    monkeypatch.setattr(g,'identity',lambda pid=None:None)
    assert not g.stale_verification(path,[])['stale_verified']


def test_gpu_presence_blocks_stale_cleanup(tmp_path,monkeypatch):
    path=tmp_path/'gpu.lock';owner=make(path).acquire()
    other={**owner,'pid':123,'exe':'python.exe'}
    monkeypatch.setattr(g,'identity',lambda pid=None: other if pid==123 else None)
    proof=g.stale_verification(path,g.parse_inventory(xml()))
    assert not proof['stale_verified'] and not proof['checks']['gpu_clear']
