"""Capture explicit pytest outcome and verify pre-existing files byte-for-byte."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'evaluation/phase60'
RAW = Path(r'Z:\AI\unipilot-mini\evaluation\phase60')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2), b''):h.update(b)
    return h.hexdigest()


def preserve(pre):
    rows=pre['dirty']+pre['protected']+pre['dirty5']+pre['frozen']
    failed=[]
    for row in rows:
        p=ROOT/row['path']
        value=sha(p) if p.is_file() else None
        if value!=row['sha256']:failed.append(row['path'])
    if failed:raise RuntimeError('FROZEN_PRESERVATION_FAIL:'+repr(failed))
    return True


def main():
    parser=argparse.ArgumentParser();parser.add_argument('scope',choices=('targeted','full'));args=parser.parse_args()
    pre=json.loads((OUT/'preflight.json').read_text(encoding='utf8'));preserve(pre)
    receipt=OUT/f'pytest-{args.scope}.json';xml=RAW/f'pytest-{args.scope}.xml';log=RAW/f'pytest-{args.scope}.log'
    if any(p.exists() for p in (receipt,xml,log)):raise FileExistsError('QA artifacts already exist')
    command=[sys.executable,'-m','pytest','-q',f'--junitxml={xml}']
    if args.scope=='targeted':command += ['tests/test_foundation_v49_audit.py','tests/test_foundation_v47_contract.py','tests/test_foundation_v47_audit.py','tests/test_evaluation_output_isolation.py','tests/test_campus_ai_quality.py']
    env={**os.environ,'UNIPILOT_CHECKPOINT_ROOT':str(Path(r'Z:\AI\unipilot-mini\checkpoints')),'PYTHONIOENCODING':'utf-8'}
    with log.open('x',encoding='utf8') as f:
        completed=subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
    text=log.read_text(encoding='utf8');print(text[-6000:],flush=True)
    counts={'tests':0,'failures':0,'errors':0,'skipped':0}
    for suite in ET.parse(xml).getroot().iter('testsuite'):
        for key in counts:counts[key]+=int(suite.get(key,0))
    warning_matches=re.findall(r'(\d+) warnings?',text.splitlines()[-1])
    outcome={'scope':args.scope,'command':command,'python':sys.executable,'exit_code':completed.returncode,
        'passed':counts['tests']-counts['failures']-counts['errors']-counts['skipped'],
        'failed':counts['failures'],'errors':counts['errors'],'skipped':counts['skipped'],
        'warnings':int(warning_matches[-1]) if warning_matches else 0,
        'summary':text.splitlines()[-1], 'log_path':str(log),'log_sha256':sha(log),'junit_path':str(xml),'junit_sha256':sha(xml),
        'repo_fixtures_unchanged':preserve(pre),'new_training':False}
    with receipt.open('x',encoding='utf8') as f:json.dump(outcome,f,indent=2);f.write('\n')
    print(json.dumps(outcome),flush=True)
    return completed.returncode


if __name__=='__main__':raise SystemExit(main())
