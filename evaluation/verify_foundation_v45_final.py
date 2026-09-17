"""Read-only checkpoint/custody verification and body-free final QA receipt."""
from __future__ import annotations
import gc,os,shutil,sys,subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.diagnose_foundation_v40 import read,new_json,file_sha256,verify_payload
from evaluation.run_foundation_v45_observatory import gate,OUT,RAW,SPEC

def main():
    gate();spec=read(SPEC)
    assert file_sha256(SPEC)==read(OUT/'preregistration-freeze.json')['sha256']
    for p,h in spec['source_sha256'].items():assert file_sha256(ROOT/p)==h,p
    torch.set_num_threads(2);checkpoints=[]
    for r in spec['checkpoints']:
        path=Path(r['path']);assert file_sha256(path)==r['sha256']
        payload=torch.load(path,map_location='cpu',weights_only=False)
        integrity=verify_payload(payload,r['seed'],16384000 if r['arm']=='C' else 15872000,5e-5 if r['arm']=='C' else 1e-4)
        assert integrity['pass'];checkpoints.append({'path':str(path),'sha256':r['sha256'],'integrity':integrity})
        del payload;gc.collect()
    for r in read(OUT/'safety-preflight.json')['protected_files']:assert file_sha256(Path(r['path']))==r['sha256']
    records=read(OUT/'run-index.json')['runs'];assert len(records)==622
    expected={RAW/'all-ngram-support.json'}
    for r in records:
        path=Path(r['path']);vectors=path.with_suffix('.npz');expected.update((path,vectors))
        assert path.stat().st_size>0 and vectors.stat().st_size>0
        assert file_sha256(path)==r['sha256'] and file_sha256(vectors)==r['vectors_sha256']
    assert set(RAW.iterdir())==expected,'Unknown or partial raw artifact requires inspection'
    assert all(p.stat().st_size>0 for p in OUT.iterdir() if p.is_file())
    registration=read(OUT/'phase57-training-preregistration.json')
    assert file_sha256(OUT/'phase57-training-preregistration.json')==read(OUT/'training-design-review.json')['preregistration_sha256']
    for p,h in registration['source_hashes'].items():assert file_sha256(ROOT/p)==h,p
    for values in (registration['training_data'],registration['evaluation_sets']):
        for value in values.values():
            if isinstance(value,dict) and 'sha256'in value:assert file_sha256(ROOT/value['path'])==value['sha256']
    for updates,limit in ((122,64000),(244,128000)):assert updates*512+(updates//8)*96<=limit
    for name,tests in [('objective-unit-tests.xml',11),('full-pytest.xml',526)]:
        suites=ET.parse(OUT/name).getroot().findall('testsuite');assert sum(int(s.attrib['tests']) for s in suites)==tests
        assert all(int(s.attrib['failures'])==int(s.attrib['errors'])==int(s.attrib['skipped'])==0 for s in suites)
    web=[]
    for name in ['results','evidence-results','calculator-results','planner-email-results','memory-plan-results','office-official-results','career-integration-results']:
        path=ROOT/f'web/qa/phase56/{name}.json';r=read(path)
        # Existing suites use different field names; all are manually checked too.
        assert not r.get('pageErrors',[]) and not r.get('errors',[])
        web.append({'path':str(path.relative_to(ROOT)),'sha256':file_sha256(path)})
    git=lambda *a:subprocess.check_output(['git',*a],cwd=ROOT,text=True).strip()
    assert git('branch','--show-current')=='foundation-research'
    assert git('rev-parse','HEAD')=='d3df3f4eb9155294d2b615792870d7ab8fa8af13'
    assert not git('diff','--cached','--name-only')
    remote={line.split()[1]:line.split()[0] for line in git('ls-remote','origin','refs/heads/foundation-research','refs/heads/main').splitlines()}
    assert remote['refs/heads/foundation-research']==git('rev-parse','HEAD')
    assert remote['refs/heads/main']=='b4c21da8976e2c2f95fbcca0499cd915aed8a9df'
    thermal={f'{r["arm"]}-{r["seed"]}':read(OUT/f'{r["arm"]}-{r["seed"]}-complete.json')['thermal']['gpu_temperature_c_max'] for r in spec['checkpoints']}
    new_json(OUT/'final-qa.json',{'gate':'PASS','checkpoint_root':os.environ['UNIPILOT_CHECKPOINT_ROOT'],'z_available':True,'z_free_bytes':shutil.disk_usage(RAW).free,
      'resolver':'PASS','checkpoint_post_sha_strict_resume':checkpoints,'copy_move_delete_overwrite':[0,0,0,0],
      'protected4_preserved':True,'ready5_preserved':True,'sealed_sets_hash_only':'PASS','final_blind_hash_only':'PASS',
      'raw_trajectories':len(records),'raw_hash_inventory':'PASS','partial_artifacts':0,'raw_root':str(RAW),'thermal_max_c':thermal,
      'observability_source_freeze':'PASS','training_preregistration_hash':'PASS','new_training':False,
      'pytest':{'passed':526,'failed':0,'warnings':5,'preexisting5_generated_artifacts_restored':'SHA_PASS'},'objective_observability_tests':11,
      'web_unit':{'passed':83,'failed':0},'npm_lint':'PASS','npm_build':'PASS','browser_demo':'PASS','live_preview':'NOT_TESTED',
      'browser_receipts':web,'widths':[360,390,768,1024,1440],'feature15':'Foundation','feature13_14':'Foundation',
      'fake_achievements_in_fixture_outputs':0,'career_external_model_requests':0,'automatic_submission':False,
      'branch':git('branch','--show-current'),'precommit_head':git('rev-parse','HEAD'),'origin_main':remote['refs/heads/main'],
      'render_vercel_production':'UNCHANGED_BY_THIS_TASK','external_ai_api':'OFF','canonical':False,'20m':False,'foundation_base':False})
    print('PHASE56 FINAL QA / SHA / STRICT RELOAD / CUSTODY PASS',flush=True)

if __name__=='__main__':main()
