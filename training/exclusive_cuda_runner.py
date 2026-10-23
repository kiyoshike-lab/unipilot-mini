"""Windows Job Object supervisor shared by training/evaluation/observatory/dry-run.

No torch import, detached launch, timeout kill or automatic stale lock cleanup.
Each invocation supervises exactly ONE child tree. A caller must explicitly poll
the process/session AND the RELEASED receipt; host tool return is not completion.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time

import psutil

from training.gpu_execution_lock import (ExecutionBlocked, GPULock, atomic_json,
                                        identity, inventory, same_process, sha)
from training.gpu_execution_guard import (ExternalComputeMonitor, GuardBlocked,
                                         current_process_identity)

STATES = ('PREPARED', 'LOCKED', 'RUNNING', 'CHECKPOINT_SAVED', 'PROCESS_EXITED',
          'RECEIPT_WRITTEN', 'RELEASED')
KINDS = ('training', 'generation-evaluation', 'hidden-state-observatory', 'cuda-dry-run', 'cuda-no-train')
# Retain Job handles for aborted live trees while this supervisor lives. A hard
# supervisor crash requires explicit forensic cleanup; missing exit proof blocks.
ABORTED_JOBS = []


class Journal:
    def __init__(self, directory, *, reserved=False):
        self.directory = Path(directory)
        if reserved:
            if not self.directory.is_dir() or self.directory.is_symlink():
                raise ExecutionBlocked('RUNTIME_JOURNAL_RESERVATION_INVALID')
        else:
            self.directory.mkdir(parents=True, exist_ok=False)  # No retries, even after crash.
        self.events = []
        self.advance('PREPARED')

    def advance(self, state, **evidence):
        if len(self.events) >= len(STATES) or state != STATES[len(self.events)]:
            raise ExecutionBlocked('ILLEGAL_RUN_TRANSITION')
        event = {'state': state, 'at': time.time(), **evidence}
        atomic_json(self.directory / f'{len(self.events):02d}-{state}.json', event)
        self.events.append(event)


class WindowsTree:
    """Assign a suspended child before it can spawn a descendant or use CUDA.

    The Job Object has neither BREAKAWAY nor KILL_ON_JOB_CLOSE. OS accounting
    includes short-lived children that polling alone could miss. No API here
    terminates any process. Assignment failure leaves an incomplete owner.
    """
    def __init__(self, name):
        if os.name != 'nt':
            raise ExecutionBlocked('WINDOWS_JOB_SUPERVISOR_REQUIRED')
        self.k = ctypes.WinDLL('kernel32', use_last_error=True)
        self.k.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.k.CreateJobObjectW.restype = wintypes.HANDLE
        self.k.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.k.AssignProcessToJobObject.restype = wintypes.BOOL
        self.k.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                                    ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p]
        self.k.QueryInformationJobObject.restype = wintypes.BOOL
        self.k.CloseHandle.argtypes = [wintypes.HANDLE]
        self.handle = self.k.CreateJobObjectW(None, name)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())

    def assign(self, process):
        if not self.k.AssignProcessToJobObject(self.handle, wintypes.HANDLE(int(process._handle))):
            raise ctypes.WinError(ctypes.get_last_error())

    def resume(self, process):
        ntdll = ctypes.WinDLL('ntdll')
        ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
        ntdll.NtResumeProcess.restype = wintypes.LONG
        if ntdll.NtResumeProcess(wintypes.HANDLE(int(process._handle))) != 0:
            raise ExecutionBlocked('CHILD_RESUME_FAILED')

    def pids(self):
        # JOBOBJECT_BASIC_PROCESS_ID_LIST, 64-bit ULONG_PTR array.
        count = 4096
        class Pids(ctypes.Structure):
            _fields_ = [('assigned', wintypes.DWORD), ('count', wintypes.DWORD),
                        ('pids', ctypes.c_size_t * count)]
        value = Pids()
        if not self.k.QueryInformationJobObject(self.handle, 3, ctypes.byref(value), ctypes.sizeof(value), None):
            raise ctypes.WinError(ctypes.get_last_error())
        if value.count > count:
            raise ExecutionBlocked('CHILD_TREE_TOO_LARGE')
        return list(value.pids[:value.count])

    def close(self):
        self.k.CloseHandle(self.handle)


def is_complete(directory):
    directory = Path(directory)
    try:
        events = [json.loads((directory / f'{i:02d}-{s}.json').read_text()) for i, s in enumerate(STATES)]
        receipt_path = directory / 'receipt.json'
        receipt = json.loads(receipt_path.read_text())
        if receipt.get('kind') == 'cuda-no-train':
            from training.phase63_runtime_journal import RuntimeJournalError, validate_current_receipt
            try:
                validate_current_receipt(directory.parents[1], directory, receipt['verification'])
            except RuntimeJournalError:
                return False
        return (all(e['state'] == s for e, s in zip(events, STATES)) and
                receipt['status'] == 'COMPLETE' and receipt['exit_code'] == 0 and
                receipt['child_tree_exited'] is True and receipt['gpu_context_released'] is True and
                receipt['stdout_closed'] is True and receipt['stderr_closed'] is True and
                receipt['verification']['strict_reload'] is True and
                receipt['verification']['updates_complete'] is True and
                sha(receipt['verification']['artifact_path']) == receipt['verification']['artifact_sha256'] and
                sha(directory/'stdout.log') == receipt['stdout_sha256'] and
                sha(directory/'stderr.log') == receipt['stderr_sha256'] and
                events[-1]['receipt_sha256'] == sha(receipt_path) and
                not (directory / 'aborted.json').exists())
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _monitor_observation(monitor):
    """Translate a fail-closed monitor decision into the runner's contract."""
    try:
        return monitor.observe()
    except GuardBlocked as exc:
        raise ExecutionBlocked(str(exc)) from exc


def heavy_jobs():
    rows = []
    for p in psutil.process_iter(['pid', 'cmdline']):
        if p.pid == os.getpid():
            continue
        text = ' '.join(p.info['cmdline'] or []).lower()
        if any(x in text for x in ('-m pytest', 'npm run build', 'playwright')):
            rows.append({'pid': p.pid, 'command': text})
    return rows


def execute(command, *, cwd, root, runtime, run_id, phase, kind, device, repo_head,
            parent_sha, verifier, temperature, outputs, previous=None, query=inventory,
            display=(), approvals=(), identity_provider=current_process_identity, compute_monitor=None,
            poll_seconds=.05, maximum_seconds=3600, journal_path=None, journal_contract=None):
    """No COMPLETE until tree accounting, pipe EOF and GPU release all agree.

    verifier is a trusted per-study strict checkpoint validator (no GPU inference).
    It must return strict_reload/updates_complete booleans and artifact SHA. For
    non-training kinds it verifies the exclusive evaluation artifact and zero
    optimizer updates; CHECKPOINT_SAVED is the generic artifact-durable state.
    """
    if kind not in KINDS or not command or not isinstance(command, list):
        raise ValueError('invalid execution contract')
    # Legacy display exceptions carried only a PID/name and are intentionally
    # unavailable. The strict monitor below is the sole C/C+G admission path.
    if display:
        raise ExecutionBlocked('LEGACY_DISPLAY_EXEMPTIONS_FORBIDDEN')
    if Path(run_id).name != run_id or run_id in ('', '.', '..'):
        raise ValueError('invalid run id')
    if os.environ.get('UNIPILOT_CHECKPOINT_ROOT') != str(root):
        raise ExecutionBlocked('PROCESS_ENV_RESOLVER_MISMATCH')
    from training.checkpoint_paths import checkpoint_root
    if checkpoint_root(Path(cwd)) != Path(root):
        raise ExecutionBlocked('PROCESS_ENV_RESOLVER_MISMATCH')
    if Path(runtime).resolve() != (Path(root).parent/'runtime').resolve():
        raise ExecutionBlocked('SINGLE_RUNTIME_ROOT_REQUIRED')
    if previous is not None and not is_complete(previous):
        raise ExecutionBlocked('PREVIOUS_RUN_INCOMPLETE')
    if (journal_path is None) != (journal_contract is None):
        raise ExecutionBlocked('RUNTIME_JOURNAL_ARGUMENT_MISMATCH')
    journal_directory = Path(runtime)/'runs'/run_id if journal_path is None else Path(journal_path)
    if journal_contract is not None:
        from training.phase63_runtime_journal import RuntimeJournalError, validate_reserved
        try:
            validate_reserved(Path(runtime), journal_directory, journal_contract)
        except RuntimeJournalError as exc:
            raise ExecutionBlocked(str(exc)) from exc
        if journal_contract.get('execution_instance') != run_id:
            raise ExecutionBlocked('RUNTIME_JOURNAL_EXECUTION_ID_MISMATCH')
    lock = GPULock(Path(runtime)/'gpu.lock', phase=phase, run_id=run_id, device=device,
                   command=command, repo_head=repo_head, parent_sha=parent_sha)
    # Acquire BEFORE any CUDA-related operation. All four kinds use this path.
    lock.acquire()
    journal = None
    tree = None
    proc = None
    try:
        journal = Journal(journal_directory, reserved=journal_contract is not None)
        journal.advance('LOCKED', owner=lock.owner)
        if not outputs or any(Path(p).exists() or Path(str(p)+'.tmp').exists() for p in outputs):
            raise ExecutionBlocked('OUTPUT_COLLISION_OR_MISSING_DECLARATION')
        # A monitor observes during the child lifetime as well as before/after.
        # With no approvals it preserves the existing policy: every C/C+G is a
        # block. It never creates a name or OS-process allowlist.
        monitor = compute_monitor or ExternalComputeMonitor(query, identity_provider, approvals=approvals)
        before_monitor = _monitor_observation(monitor)
        before = before_monitor['rows']
        sample = temperature()
        if type(sample['temperature_c']) not in (int,float) or not math.isfinite(sample['temperature_c']) or \
                type(sample['hardware_thermal_slowdown']) is not bool or \
                sample['hardware_thermal_slowdown'] or sample['temperature_c'] > 60:
            raise ExecutionBlocked('TEMPERATURE_GATE_FAILED')
        warnings = heavy_jobs()
        atomic_json(journal.directory/'prelaunch.json', {'inventory': before, 'temperature': sample,
                                                       'cpu_heavy_warnings': warnings, 'cwd': str(cwd),
                                                       'command': command, 'kind': kind})
        if warnings:
            raise ExecutionBlocked('CPU_HEAVY_JOB_PRESENT')
        tree = WindowsTree(lock.owner['job_name'])
        env = {**os.environ, 'UNIPILOT_CHECKPOINT_ROOT': str(root),
               'UNIPILOT_GPU_OWNER_NONCE': lock.owner['nonce'],
               'UNIPILOT_GPU_LOCK_PATH': str(lock.path),
               'UNIPILOT_GPU_RUN_ID': run_id,
               'UNIPILOT_GPU_PHASE': str(phase),
               'UNIPILOT_RUNTIME_JOURNAL_PATH': str(journal_directory),
               'UNIPILOT_RUNTIME_JOURNAL_ID': run_id}
        proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
                                creationflags=0x00000004, shell=False)  # Win32 CREATE_SUSPENDED
        child = identity(proc.pid)
        atomic_json(journal.directory/'child-created.json', {'child': child, 'suspended': True})
        tree.assign(proc)
        journal.advance('RUNNING', child=child, job_object_assigned=True)
        eof = {'stdout': False, 'stderr': False}
        stream_errors = []
        def drain(name, stream):
            try:
                with (journal.directory/(name+'.log')).open('xb') as log:
                    while block := stream.read(65536):
                        log.write(block)
                    log.flush(); os.fsync(log.fileno())
                stream.close()
                eof[name] = True
            except BaseException as exc:
                stream_errors.append(repr(exc))
        threads = [threading.Thread(target=drain, args=(name, stream), daemon=True)
                   for name, stream in [('stdout', proc.stdout), ('stderr', proc.stderr)]]
        for thread in threads:
            thread.start()
        caller = identity(os.getppid())
        tree.resume(proc)
        start = time.monotonic()
        observed = {child['pid']: child}
        while True:
            # Unknown inventory, a restarted approved process, a new C/C+G
            # process, expiry, or provider failure retains the owner lock.
            _monitor_observation(monitor)
            pids = tree.pids()
            for pid in pids:
                current = identity(pid)
                if current and pid not in observed:
                    observed[pid] = current
                    atomic_json(journal.directory/f'child-{pid}.json', current)
            if caller is None or not same_process(caller, identity(caller['pid'])):
                raise ExecutionBlocked('HOST_PARENT_EXITED_LOCK_RETAINED')
            if stream_errors:
                raise ExecutionBlocked('STDIO_CAPTURE_FAILED')
            if time.monotonic()-start > maximum_seconds:
                raise ExecutionBlocked('WAIT_DEADLINE_INCOMPLETE_NO_KILL')
            if proc.poll() is not None and not pids and all(eof.values()):
                break
            time.sleep(poll_seconds)
        for thread in threads:
            thread.join()
        if proc.returncode != 0:
            raise ExecutionBlocked('CHILD_EXIT_NONZERO:'+str(proc.returncode))
        after_monitor = _monitor_observation(monitor)
        after = after_monitor['rows']
        # The strict monitor has already required every C/C+G row to be an
        # exact, still-live exception. It is authoritative over the legacy
        # PID-only display filter, which cannot defend against PID reuse.
        verification = verifier()
        if verification.get('strict_reload') is not True or verification.get('updates_complete') is not True:
            raise ExecutionBlocked('ARTIFACT_VERIFICATION_FAILED')
        if not verification.get('artifact_sha256') or not verification.get('artifact_path') or \
                sha(verification['artifact_path']) != verification['artifact_sha256']:
            raise ExecutionBlocked('ARTIFACT_HASH_REQUIRED')
        journal.advance('CHECKPOINT_SAVED', verification=verification,
                        note='durable artifact observation; actual save happened in child before exit')
        journal.advance('PROCESS_EXITED', exit_code=proc.returncode, children=list(observed.values()),
                        stdout_closed=True, stderr_closed=True, gpu_context_released=True)
        receipt = {'status': 'COMPLETE', 'kind': kind, 'exit_code': proc.returncode, 'child_tree_exited': True,
                   'gpu_context_released': True, 'stdout_closed': True, 'stderr_closed': True,
                   'verification': verification, 'owner': lock.owner,
                   'stdout_sha256': sha(journal.directory/'stdout.log'),
                   'stderr_sha256': sha(journal.directory/'stderr.log')}
        atomic_json(journal.directory/'receipt.json', receipt)
        journal.advance('RECEIPT_WRITTEN', receipt_sha256=sha(journal.directory/'receipt.json'))
        lock.release(complete=True, gpu_clear=True)
        journal.advance('RELEASED', receipt_sha256=sha(journal.directory/'receipt.json'))
        return receipt
    except BaseException as exc:
        if journal is not None:
            atomic_json(journal.directory/'aborted.json', {'status': 'INCOMPLETE', 'ownership': 'ABORTED',
                        'reason': repr(exc), 'owner': lock.owner,
                        'child': identity(proc.pid) if proc is not None else None,
                        'lock_retained': lock.path.exists(), 'automatic_terminate': False})
        raise
    finally:
        if tree is not None:
            if tree.pids():
                ABORTED_JOBS.append(tree)
            else:
                tree.close()  # No kill-on-close flag; never kills a process.
