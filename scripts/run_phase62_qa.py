"""Capture OS exit code/JUnit, preserving all existing dirty and frozen files."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.phase62_audit import OUT, RAW, ZROOT, preserve
from training.gpu_execution_lock import atomic_json,sha


def run(scope, suffix=''):
    preserve()
    name=scope+suffix
    receipt=OUT/f'pytest-{name}.json';log=RAW/f'pytest-{name}.log';xml=RAW/f'pytest-{name}.xml'
    if any(p.exists() for p in (receipt,log,xml)):raise FileExistsError('QA receipt already exists')
    command=[sys.executable,'-m','pytest','-q',f'--junitxml={xml}']
    if scope=='targeted':command+=['tests/test_gpu_execution_lock.py','tests/test_exclusive_cuda_runner.py','tests/test_evaluation_output_isolation.py']
    with log.open('x',encoding='utf8') as f:
        result=subprocess.run(command,cwd=ROOT,env={**os.environ,'UNIPILOT_CHECKPOINT_ROOT':str(ZROOT),'PYTHONIOENCODING':'utf-8'},stdout=f,stderr=subprocess.STDOUT)
    counts={k:sum(int(s.get(k,0)) for s in ET.parse(xml).getroot().iter('testsuite')) for k in ('tests','failures','errors','skipped')}
    text=log.read_text(encoding='utf8');summary=text.splitlines()[-1];warnings=re.findall(r'(\d+) warnings?',summary)
    value={'phase':62,'scope':scope,'exit_code':result.returncode,'passed':counts['tests']-counts['failures']-counts['errors']-counts['skipped'],
           'failed':counts['failures'],'errors':counts['errors'],'skipped':counts['skipped'],'warnings':int(warnings[-1]) if warnings else 0,
           'summary':summary,'log':str(log),'log_sha256':sha(log),'junit':str(xml),'junit_sha256':sha(xml),
           'preservation_before_after':preserve(),'optimizer_training_tests':False,'new_training':False}
    atomic_json(receipt,value);print(json.dumps(value),flush=True)
    return result.returncode

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('scope',choices=('targeted','full'));p.add_argument('--final',action='store_true');args=p.parse_args();raise SystemExit(run(args.scope,'-final' if args.final else ''))
