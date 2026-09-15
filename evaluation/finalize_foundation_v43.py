"""Read-only integrity and blocked confirmatory documentation, never scoring."""
import gc, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from evaluation.diagnose_foundation_v40 import resolver_gate,read,new_json,file_sha256,evaluated_checkpoint,verify_payload,BLIND_SHA

def main():
    assert os.environ.get('UNIPILOT_CHECKPOINT_ROOT')==r'Z:\AI\unipilot-mini\checkpoints'
    manifest=resolver_gate();out=ROOT/'evaluation/phase54'
    fresh=read(out/'fresh-holdout-manifest.json')
    assert fresh['gate']!='SUPPLEMENTAL_FRESH_READY','This command only documents a blocked gate'
    rows=[]
    torch.set_num_threads(2)
    for arm in ('C','B'):
        for seed in (42,123,2026):
            path=evaluated_checkpoint(arm,seed);known=next(r for r in manifest['rows'] if Path(r['destination'])==path)
            sha=file_sha256(path);assert sha==known['sha256']
            payload=torch.load(path,map_location='cpu',weights_only=False)
            integrity=verify_payload(payload,seed,16384000,5e-5 if arm=='C' else 7.5e-5)
            assert integrity['pass'];del payload;gc.collect()
            rows.append({'arm':arm,'seed':seed,'path':str(path),'sha256':sha,'bytes':path.stat().st_size,'integrity':integrity})
            print('Verified',arm,seed,flush=True)
    old=read(ROOT/'evaluation/phase53/fresh-holdout-manifest.json')
    for split in old['splits'].values():assert file_sha256(Path(split['path']))==split['sha256']
    new_json(out/'integrity.json',{'resolver_gate':'PROCESS_ENV_Z_ROOT_PASS','checkpoint_root':os.environ['UNIPILOT_CHECKPOINT_ROOT'],
        'rows':rows,'protected_files':manifest['protected_files'],'final_blind_sha_only':BLIND_SHA,'phase53_reserve':'UNTOUCHED',
        'checkpoint_copy_move_delete_overwrite':[0,0,0,0],'new_training':False})
    prereg={'phase':54,'status':'NOT_REGISTERED_SUPPLEMENTAL_GATE_BLOCKED','gate':fresh['gate'],
        'checkpoint_identities':[{k:r[k] for k in ('arm','seed','path','sha256')} for r in rows],
        'documents':[],'categories':[],'metrics':[],'decoder_settings':None,'rng':None,'bootstrap':None,'decision_criteria':None,
        'reason':'No qualified confirmatory set exists. Do not invent an executable preregistration or score any checkpoint. A complete one-shot registration is required after the data gate passes.',
        'scoring_started':False,'new_training':False}
    new_json(out/'lr-confirmatory-preregistration.json',prereg)
    leak=read(out/'leakage-report.json');provenance=read(out/'supplemental-provenance.json')
    new_json(ROOT/'evaluation/foundation-v43-confirmatory-summary.json',{'phase':54,'supplemental':fresh,
        'candidate_pages_requested':3000,'candidate_pages_returned':provenance['candidate_inventory']['returned'],
        'excluded_counts':leak['counts'],'preregistration_sha256':file_sha256(out/'lr-confirmatory-preregistration.json'),
        'confirmatory_gate':'CONFIRMATORY_INVALID','formal_lr_gate':'FORMAL_LR_UNRESOLVED','approved_lr':None,
        'C_metrics':'NOT_RUN','B_metrics':'NOT_RUN','eos':'NOT_RUN','sampling':'NOT_RUN',
        'new_training':False,'canonical':False,'20m':False,'foundation_base':False,'new_gpu_inference':False})
if __name__=='__main__':main()
