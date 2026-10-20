"""Fail-closed single-host GPU ownership. Never kill or auto-expire an owner."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import uuid
import xml.etree.ElementTree as ET
from contextlib import contextmanager

import psutil


class ExecutionBlocked(RuntimeError):
    pass


RECOVERY_AUTHORIZATION_SCHEMA = 'phase63-lock-recovery-authorization-v1'
RECOVERY_CLAIM_SCHEMA = 'phase63-lock-recovery-claim-v1'
RECOVERY_SCOPE = 'PHASE63_SINGLE_ABORTED_LOCK_FINALIZATION'
PRE_CUDA_FAILURE_CODE = 'BINDING_OUTPUT_MISMATCH'


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


def _read_json(path, label):
    try:
        value = json.loads(Path(path).read_text(encoding='utf8'))
    except (OSError, ValueError, TypeError) as exc:
        raise ExecutionBlocked('LOCK_RECOVERY_' + label + '_UNREADABLE') from exc
    if not isinstance(value, dict):
        raise ExecutionBlocked('LOCK_RECOVERY_' + label + '_INVALID')
    return value


def _exact_keys(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ExecutionBlocked('LOCK_RECOVERY_' + label + '_INVALID')
    return value


def _canonical_path(path):
    return os.path.normcase(os.path.normpath(os.path.abspath(str(path))))


def _canonical_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode('utf8')).hexdigest()


def recovery_claim_path(authorization_path):
    path = Path(authorization_path)
    return path.with_name(path.name + '.phase63-lock-recovery-claim.json')


def _recovery_claim_expected(authorization, authorization_sha256):
    return {'schema_version': RECOVERY_CLAIM_SCHEMA, 'authorization_sha256': authorization_sha256,
            'authorization_id': authorization['authorization_id'], 'purpose': authorization['purpose'],
            'lock_sha256': authorization['target_lock']['sha256'],
            'execution_instance': authorization['execution']['run_id']}


def claim_recovery_authorization(authorization_path, authorization, authorization_sha256):
    """Atomically consume only a validated lock-recovery authorization."""
    target = recovery_claim_path(authorization_path)
    expected = _recovery_claim_expected(authorization, authorization_sha256)
    record = {'claim': expected, 'claimed_at_utc': datetime.now(UTC).isoformat()}
    encoded = (json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                          allow_nan=False) + '\n').encode('utf8')
    try:
        with target.open('xb') as handle:
            handle.write(encoded); handle.flush(); os.fsync(handle.fileno())
    except FileExistsError:
        existing = _read_json(target, 'RECOVERY_CLAIM_STATE')
        if existing.get('claim') == expected:
            raise ExecutionBlocked('LOCK_RECOVERY_AUTHORIZATION_ALREADY_CONSUMED')
        raise ExecutionBlocked('LOCK_RECOVERY_AUTHORIZATION_CLAIM_CONFLICT')
    except OSError as exc:
        raise ExecutionBlocked('LOCK_RECOVERY_AUTHORIZATION_CLAIM_WRITE_FAILED') from exc
    return record


def _parse_time(value, label):
    if not isinstance(value, str):
        raise ExecutionBlocked('LOCK_RECOVERY_' + label + '_INVALID')
    try:
        result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as exc:
        raise ExecutionBlocked('LOCK_RECOVERY_' + label + '_INVALID') from exc
    if result.tzinfo is None:
        raise ExecutionBlocked('LOCK_RECOVERY_' + label + '_INVALID')
    return result.astimezone(UTC)


def _assert_hex(value, label):
    if not isinstance(value, str) or re.fullmatch(r'[0-9a-f]{64}', value) is None:
        raise ExecutionBlocked('LOCK_RECOVERY_' + label + '_INVALID')


def _expected_phase63_output(binding, seed, arm):
    return (Path(binding['checkpoint_root']) / 'experimental' / 'phase63' / 'continuation-stability' /
            arm / f'seed-{seed}' / 'checkpoint-tokens-16446464.pt')


def aborted_lock_recovery_proof(lock_path, *, execution_authorization_path, binding_path,
                                identity_provider=identity, job_clear_provider=job_clear):
    """Build a no-GPU proof that this exact aborted execution is terminal.

    This deliberately does not query global GPU inventory: unrelated Windows
    C+G clients cannot establish that this exact, already-ended UniPilot tree
    is live. Exact owner/child identities and the original Job are required.
    """
    lock_path = Path(lock_path)
    owner = _read_json(lock_path, 'LOCK')
    required_owner = {'pid', 'parent_pid', 'process_start_time', 'hostname', 'exe', 'schema_version', 'phase',
                      'run_id', 'cuda_device', 'repo_head', 'checkpoint_parent_sha256', 'created_timestamp',
                      'command_hash', 'nonce', 'job_name'}
    _exact_keys(owner, required_owner, 'LOCK')
    if owner['schema_version'] != 'gpu-owner-v1' or owner['phase'] != 63 or not isinstance(owner['run_id'], str):
        raise ExecutionBlocked('LOCK_RECOVERY_LOCK_INVALID')
    run_dir = lock_path.parent / 'runs' / owner['run_id']
    aborted_path = run_dir / 'aborted.json'; child_path = run_dir / 'child-created.json'
    stderr_path = run_dir / 'stderr.log'; stdout_path = run_dir / 'stdout.log'
    aborted = _read_json(aborted_path, 'ABORTED')
    _exact_keys(aborted, {'status', 'ownership', 'reason', 'owner', 'child', 'lock_retained', 'automatic_terminate'}, 'ABORTED')
    child_record = _read_json(child_path, 'CHILD')
    _exact_keys(child_record, {'child', 'suspended'}, 'CHILD')
    child = child_record['child']
    if (aborted['status'] != 'INCOMPLETE' or aborted['ownership'] != 'ABORTED' or
            aborted['reason'] != "ExecutionBlocked('CHILD_EXIT_NONZERO:1')" or
            aborted['owner'] != owner or aborted['child'] is not None or
            aborted['lock_retained'] is not True or aborted['automatic_terminate'] is not False or
            not isinstance(child, dict) or child.get('parent_pid') != owner['pid']):
        raise ExecutionBlocked('LOCK_RECOVERY_ABORT_CHAIN_INVALID')
    authorization = _read_json(execution_authorization_path, 'EXECUTION_AUTHORIZATION')
    claim = _read_json(Path(execution_authorization_path).with_name(Path(execution_authorization_path).name + '.phase63-claim.json'),
                       'EXECUTION_CLAIM')
    binding = _read_json(binding_path, 'BINDING')
    if (authorization.get('action') != 'no_train' or authorization.get('phase') != 63 or
            authorization.get('binding_sha256') != sha(binding_path) or
            claim.get('claim', {}).get('authorization_sha256') != sha(execution_authorization_path) or
            claim.get('claim', {}).get('binding_sha256') != sha(binding_path) or
            claim.get('claim', {}).get('execution_instance') != owner['run_id'] or
            claim.get('claim', {}).get('action') != 'no_train'):
        raise ExecutionBlocked('LOCK_RECOVERY_EXECUTION_AUTHORIZATION_NOT_TERMINAL')
    if (binding.get('implementation_source_commit') != owner['repo_head'] or
            not isinstance(binding.get('source_manifest_sha256'), str) or
            not isinstance(binding.get('source_manifest'), list)):
        raise ExecutionBlocked('LOCK_RECOVERY_BINDING_CHAIN_INVALID')
    allowed_runs = authorization.get('allowed_runs')
    if not isinstance(allowed_runs, list) or len(allowed_runs) != 1 or not isinstance(allowed_runs[0], dict):
        raise ExecutionBlocked('LOCK_RECOVERY_BINDING_CHAIN_INVALID')
    target = allowed_runs[0]
    row = next((row for row in binding.get('runs', []) if isinstance(row, dict) and
                row.get('seed') == target.get('seed') and row.get('arm') == target.get('arm')), None)
    if row is None or Path(row.get('output_path', '')) == _expected_phase63_output(binding, target.get('seed'), target.get('arm')):
        raise ExecutionBlocked('LOCK_RECOVERY_PRE_CUDA_CHAIN_INVALID')
    stderr = stderr_path.read_text(encoding='utf8', errors='replace')
    if 'RuntimeError: ' + PRE_CUDA_FAILURE_CODE not in stderr:
        raise ExecutionBlocked('LOCK_RECOVERY_PRE_CUDA_CHAIN_INVALID')
    owner_now = identity_provider(owner['pid']); child_now = identity_provider(child.get('pid'))
    if same_process(owner, owner_now):
        raise ExecutionBlocked('LOCK_RECOVERY_OWNER_STILL_LIVE')
    if same_process(child, child_now):
        raise ExecutionBlocked('LOCK_RECOVERY_CHILD_STILL_LIVE')
    if job_clear_provider(owner['job_name']) is not True:
        raise ExecutionBlocked('LOCK_RECOVERY_JOB_NOT_CLEAR')
    return {'lock_path': _canonical_path(lock_path), 'lock_sha256': sha(lock_path),
            'lock_owner_sha256': _canonical_sha(owner), 'run_id': owner['run_id'],
            'binding_path': _canonical_path(binding_path), 'binding_sha256': sha(binding_path),
            'implementation_head': binding['implementation_source_commit'],
            'source_manifest_sha256': binding['source_manifest_sha256'],
            'execution_authorization_path': _canonical_path(execution_authorization_path),
            'execution_authorization_sha256': sha(execution_authorization_path),
            'execution_claim_path': _canonical_path(Path(execution_authorization_path).with_name(Path(execution_authorization_path).name + '.phase63-claim.json')),
            'execution_claim_sha256': sha(Path(execution_authorization_path).with_name(Path(execution_authorization_path).name + '.phase63-claim.json')),
            'aborted_path': _canonical_path(aborted_path), 'aborted_sha256': sha(aborted_path),
            'child_created_path': _canonical_path(child_path), 'child_created_sha256': sha(child_path),
            'stderr_path': _canonical_path(stderr_path), 'stderr_sha256': sha(stderr_path),
            'stdout_path': _canonical_path(stdout_path), 'stdout_sha256': sha(stdout_path),
            'failure_code': PRE_CUDA_FAILURE_CODE, 'owner_gone': True, 'child_gone': True,
            'job_clear': True, 'authorization_consumed': True, 'failure_before_cuda': True}


def validate_recovery_authorization(authorization_path, proof, *, now=None):
    authorization = _read_json(authorization_path, 'RECOVERY_AUTHORIZATION')
    required = {'schema_version', 'authorization_id', 'purpose', 'scope', 'valid_from', 'valid_until',
                'target_lock', 'execution', 'evidence'}
    _exact_keys(authorization, required, 'RECOVERY_AUTHORIZATION')
    if (authorization['schema_version'] != RECOVERY_AUTHORIZATION_SCHEMA or
            authorization['purpose'] != 'lock_recovery' or authorization['scope'] != RECOVERY_SCOPE or
            not isinstance(authorization['authorization_id'], str) or
            re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}', authorization['authorization_id']) is None):
        raise ExecutionBlocked('LOCK_RECOVERY_AUTHORIZATION_SCOPE_INVALID')
    start = _parse_time(authorization['valid_from'], 'RECOVERY_AUTHORIZATION_TIME')
    end = _parse_time(authorization['valid_until'], 'RECOVERY_AUTHORIZATION_TIME')
    moment = (now or datetime.now(UTC)).astimezone(UTC)
    if not (start <= moment <= end) or (end - start).total_seconds() > 900:
        raise ExecutionBlocked('LOCK_RECOVERY_AUTHORIZATION_EXPIRED_OR_NOT_SHORT_LIVED')
    lock = _exact_keys(authorization['target_lock'], {'path', 'sha256', 'owner_sha256'}, 'RECOVERY_TARGET_LOCK')
    execution = _exact_keys(authorization['execution'], {'run_id', 'binding_path', 'binding_sha256', 'implementation_head',
                                                          'source_manifest_sha256', 'authorization_path', 'authorization_sha256',
                                                          'claim_path', 'claim_sha256'}, 'RECOVERY_EXECUTION')
    evidence = _exact_keys(authorization['evidence'], {'aborted_path', 'aborted_sha256', 'child_created_path',
                                                        'child_created_sha256', 'stderr_path', 'stderr_sha256',
                                                        'stdout_path', 'stdout_sha256', 'failure_code'}, 'RECOVERY_EVIDENCE')
    for value, label in ((lock['sha256'], 'LOCK_SHA'), (lock['owner_sha256'], 'OWNER_SHA'),
                         (execution['binding_sha256'], 'BINDING_SHA'), (execution['source_manifest_sha256'], 'MANIFEST_SHA'),
                         (execution['authorization_sha256'], 'AUTHORIZATION_SHA'), (execution['claim_sha256'], 'CLAIM_SHA'),
                         (evidence['aborted_sha256'], 'ABORTED_SHA'), (evidence['child_created_sha256'], 'CHILD_SHA'),
                         (evidence['stderr_sha256'], 'STDERR_SHA'), (evidence['stdout_sha256'], 'STDOUT_SHA')):
        _assert_hex(value, label)
    expected = {'path': proof['lock_path'], 'sha256': proof['lock_sha256'], 'owner_sha256': proof['lock_owner_sha256']}
    if lock != expected or evidence['failure_code'] != proof['failure_code']:
        raise ExecutionBlocked('LOCK_RECOVERY_AUTHORIZATION_TARGET_MISMATCH')
    pairs = {'run_id': proof['run_id'], 'binding_path': proof['binding_path'], 'binding_sha256': proof['binding_sha256'],
             'implementation_head': proof['implementation_head'], 'source_manifest_sha256': proof['source_manifest_sha256'],
             'authorization_path': proof['execution_authorization_path'], 'authorization_sha256': proof['execution_authorization_sha256'],
             'claim_path': proof['execution_claim_path'], 'claim_sha256': proof['execution_claim_sha256']}
    if execution != pairs or {key: evidence[key] for key in evidence if key != 'failure_code'} != {
            'aborted_path': proof['aborted_path'], 'aborted_sha256': proof['aborted_sha256'],
            'child_created_path': proof['child_created_path'], 'child_created_sha256': proof['child_created_sha256'],
            'stderr_path': proof['stderr_path'], 'stderr_sha256': proof['stderr_sha256'],
            'stdout_path': proof['stdout_path'], 'stdout_sha256': proof['stdout_sha256']}:
        raise ExecutionBlocked('LOCK_RECOVERY_AUTHORIZATION_TARGET_MISMATCH')
    return authorization


def finalize_aborted_lock_recovery(lock_path, recovery_authorization_path, *, execution_authorization_path, binding_path,
                                   identity_provider=identity, job_clear_provider=job_clear, now=None):
    """Finalize an exact terminated aborted lock; never starts GPU work.

    The original lock bytes are hard-linked into the run journal before the
    sentinel is removed. Any failed proof, claim, or archive step retains the
    lock and fails closed.
    """
    lock_path = Path(lock_path)
    with mutation_guard(lock_path):
        proof = aborted_lock_recovery_proof(lock_path, execution_authorization_path=execution_authorization_path,
                                            binding_path=binding_path, identity_provider=identity_provider,
                                            job_clear_provider=job_clear)
        authorization = validate_recovery_authorization(recovery_authorization_path, proof, now=now)
        authorization_sha256 = sha(recovery_authorization_path)
        claim_recovery_authorization(recovery_authorization_path, authorization, authorization_sha256)
        fresh = aborted_lock_recovery_proof(lock_path, execution_authorization_path=execution_authorization_path,
                                            binding_path=binding_path, identity_provider=identity_provider,
                                            job_clear_provider=job_clear)
        if fresh != proof:
            raise ExecutionBlocked('LOCK_RECOVERY_PROOF_CHANGED_AFTER_CLAIM')
        run_dir = lock_path.parent / 'runs' / proof['run_id']
        archive = run_dir / 'recovered-lock.json'; receipt = run_dir / 'lock-recovery.json'
        if archive.exists() or receipt.exists():
            raise ExecutionBlocked('LOCK_RECOVERY_ARCHIVE_COLLISION')
        atomic_json(receipt, {'status': 'FINALIZED_ABORTED', 'purpose': 'lock_recovery', 'proof': proof,
                              'recovery_authorization_sha256': authorization_sha256,
                              'recovery_authorization_id': authorization['authorization_id']})
        os.link(lock_path, archive)
        lock_path.unlink()
        return {'status': 'FINALIZED_ABORTED', 'receipt': str(receipt), 'archived_lock': str(archive)}
