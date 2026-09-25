"""Fail-closed single-host GPU ownership. Never kill or auto-expire an owner."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import uuid
import xml.etree.ElementTree as ET
from contextlib import contextmanager

import psutil


class ExecutionBlocked(RuntimeError):
    pass


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    """Publish a fully flushed new artifact without replacing an existing file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('x', encoding='utf8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    # Hardlink is atomic and fails if destination exists (unlike replace).
    os.link(tmp, path)
    tmp.unlink()


def identity(pid=None):
    try:
        p = psutil.Process(os.getpid() if pid is None else pid)
        return {'pid': p.pid, 'parent_pid': p.ppid(), 'process_start_time': p.create_time(),
                'hostname': socket.gethostname(), 'exe': p.exe()}
    except psutil.NoSuchProcess:
        return None


def same_process(a, b):
    """Compare a live process instance, not just a recyclable PID.

    The lock schema predates executable hashes, so lock ownership binds the
    canonical executable path in addition to PID/start-time/host.  Approval
    matching with executable SHA256 lives in ``gpu_execution_guard``.
    """
    try:
        return bool(a and b and all(a[k] == b[k] for k in ('pid', 'hostname', 'process_start_time')) and
                    os.path.normcase(os.path.normpath(a['exe'])) == os.path.normcase(os.path.normpath(b['exe'])))
    except (KeyError, TypeError):
        return False


def job_clear(name):
    """An orphan can acquire CUDA later, so GPU absence alone is insufficient."""
    if os.name != 'nt':
        raise ExecutionBlocked('JOB_ACCOUNTING_UNAVAILABLE')
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    k.OpenJobObjectW.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.LPCWSTR]
    k.OpenJobObjectW.restype=wintypes.HANDLE
    k.QueryInformationJobObject.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD,ctypes.c_void_p]
    k.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=k.OpenJobObjectW(4,False,name)  # JOB_OBJECT_QUERY
    if not handle:
        if ctypes.get_last_error()==2:return True  # Gone only after last member exits.
        raise ExecutionBlocked('JOB_ACCOUNTING_UNAVAILABLE')
    try:
        value=ctypes.create_string_buffer(8+8*4096)
        if not k.QueryInformationJobObject(handle,3,value,len(value),None):
            raise ExecutionBlocked('JOB_ACCOUNTING_UNAVAILABLE')
        return int.from_bytes(value.raw[4:8],'little')==0
    finally:k.CloseHandle(handle)


def parse_inventory(xml_text):
    root = ET.fromstring(xml_text)
    gpus = root.findall('gpu')
    if not gpus:
        raise ExecutionBlocked('CUDA_INVENTORY_UNAVAILABLE')
    rows = []
    for gpu in gpus:
        processes = gpu.find('processes')
        if processes is None or (processes.text or '').strip() not in ('', 'None'):
            raise ExecutionBlocked('CUDA_PROCESS_TABLE_UNAVAILABLE')
        for p in processes.findall('process_info'):
            kind = p.findtext('type')
            if kind not in ('G', 'C', 'C+G'):
                raise ExecutionBlocked('CUDA_PROCESS_TYPE_UNKNOWN')
            rows.append({'pid': int(p.findtext('pid')), 'process_name': p.findtext('process_name'),
                         'gpu_memory': p.findtext('used_memory'), 'type': kind,
                         'cuda_device': gpu.findtext('uuid'), 'timestamp': time.time()})
    return rows


def inventory():
    result = subprocess.run(['nvidia-smi', '-q', '-x'], capture_output=True, text=True,
                            check=True, timeout=10)
    return parse_inventory(result.stdout)


def foreign_processes(rows, allowed=(), display=()):
    """WDDM display exclusions are exact process identities, never name wildcards.

    Pure compute processes are never display-exempt. A C+G process must match an
    explicitly recorded display baseline, creation time and executable path.
    """
    foreign = []
    for row in rows:
        if row['type'] == 'G':
            continue
        live = identity(row['pid'])
        if live is None:  # Inventory changed: caller must retry the observation.
            raise ExecutionBlocked('CUDA_INVENTORY_RACE')
        if any(same_process(live, a) for a in allowed):
            continue
        if row['type'] == 'C+G' and any(same_process(live, a) and
                live['exe'].casefold() == a['exe'].casefold() and
                row['process_name'].casefold() == a['exe'].casefold() for a in display):
            continue
        foreign.append({**row, 'identity': live})
    return foreign


@contextmanager
def mutation_guard(path):
    """Serialize acquisition/release/cleanup; this tiny OS guard is never removed."""
    guard = Path(str(path) + '.guard')
    guard.parent.mkdir(parents=True, exist_ok=True)
    with guard.open('a+b') as f:
        # Windows permits byte-range locking past EOF. Initializing the byte
        # outside the lock would race a second opener's mandatory byte lock.
        f.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ExecutionBlocked('GPU_LOCK_HELD') from exc
        try:
            yield
        finally:
            f.seek(0)
            if os.name == 'nt':
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f, fcntl.LOCK_UN)


class GPULock:
    def __init__(self, path, *, phase, run_id, device, command, repo_head, parent_sha):
        self.path = Path(path)
        self.owner = {**identity(), 'schema_version': 'gpu-owner-v1', 'phase': phase,
                      'run_id': run_id, 'cuda_device': device, 'repo_head': repo_head,
                      'checkpoint_parent_sha256': parent_sha, 'created_timestamp': time.time(),
                      'command_hash': hashlib.sha256(json.dumps(command).encode()).hexdigest(),
                      'nonce': uuid.uuid4().hex}
        self.owner['job_name']='Local\\UniPilot-GPU-'+self.owner['nonce']

    def acquire(self):
        with mutation_guard(self.path):
            try:
                # Exclusive file creation is the lock linearization point.
                with self.path.open('x', encoding='utf8') as f:
                    json.dump(self.owner, f); f.flush(); os.fsync(f.fileno())
            except FileExistsError as exc:
                raise ExecutionBlocked('GPU_LOCK_HELD') from exc
        return self.owner

    def release(self, *, complete, gpu_clear):
        if complete is not True or gpu_clear is not True:
            raise ExecutionBlocked('INCOMPLETE_OWNER_RETAINED')
        with mutation_guard(self.path):
            stored = json.loads(self.path.read_text(encoding='utf8'))
            if stored != self.owner or not same_process(stored, identity()):
                raise ExecutionBlocked('OWNER_IDENTITY_MISMATCH')
            self.path.unlink()


def stale_verification(path, rows, *, display=(), owned_children=()):
    owner = json.loads(Path(path).read_text(encoding='utf8'))
    if owner['hostname'] != socket.gethostname():
        raise ExecutionBlocked('REMOTE_OWNER_UNVERIFIABLE')
    current = identity(owner['pid'])
    run_dir=Path(path).parent/'runs'/owner['run_id']
    tree_observed_exited=(not (run_dir/'child-created.json').exists() or
                          (run_dir/'04-PROCESS_EXITED.json').exists())
    # A reused but live PID is not the original owner; cleanup is still denied.
    checks = {'pid_absent': current is None, 'start_identity_mismatch': not same_process(owner, current),
              'gpu_clear': not foreign_processes(rows, display=display),
              'job_tree_clear': job_clear(owner['job_name']),
              'persisted_child_tree_exit_or_never_started': tree_observed_exited,
              'owned_children_gone': all(identity(c['pid']) is None for c in owned_children)}
    return {'owner': owner, 'lock_sha256': sha(path), 'checks': checks,
            'stale_verified': all(checks.values()), 'observed_at': time.time()}


def cleanup_stale(path, verification, *, query=inventory, display=(), owned_children=(), archive):
    """Explicit action only; revalidate under the same guard and archive evidence."""
    with mutation_guard(path):
        fresh = stale_verification(path, query(), display=display, owned_children=owned_children)
        if verification.get('stale_verified') is not True or not fresh['stale_verified'] or \
                fresh['lock_sha256'] != verification['lock_sha256']:
            raise ExecutionBlocked('STALE_CLEANUP_DENIED')
        atomic_json(archive, {'action': 'VERIFIED_STALE_CLEANUP', 'initial': verification, 'fresh': fresh})
        Path(path).unlink()
