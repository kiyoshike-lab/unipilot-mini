"""Read-only Live readiness: parameterized URLs, GET/OPTIONS only, no auth bypass."""
import argparse
import json
import os
import re
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urljoin


def origin(value):
    p=urlsplit(value)
    if p.scheme not in ('http','https') or not p.hostname or p.username or p.password or p.path not in ('','/') or p.query or p.fragment:
        raise argparse.ArgumentTypeError('Use an exact http(s) origin without credentials or path')
    return f'{p.scheme}://{p.netloc}'


def request(url, method='GET', headers=None):
    assert method in ('GET','OPTIONS')
    req=urllib.request.Request(url,method=method,headers={'User-Agent':'UniPilot-PHASE51-read-only-QA',**(headers or {})})
    try:
        with urllib.request.urlopen(req,timeout=20) as r:
            content=r.read(8*1024*1024).decode('utf-8',errors='replace');final=urlsplit(r.url)
            return {'status':r.status,'final_origin':f'{final.scheme}://{final.netloc}','final_path':final.path,
                'access_control_allow_origin':r.headers.get('Access-Control-Allow-Origin'),
                'access_control_allow_methods':r.headers.get('Access-Control-Allow-Methods'),
                'access_control_allow_credentials':r.headers.get('Access-Control-Allow-Credentials')},content
    except urllib.error.HTTPError as e:
        return {'status':e.code,'access_control_allow_origin':e.headers.get('Access-Control-Allow-Origin'),'error':e.reason},''
    except (OSError,ValueError) as e: return {'error':type(e).__name__+': '+str(e)},''


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--preview-url',default=os.environ.get('PREVIEW_URL'),type=origin)
    parser.add_argument('--api-url',default=os.environ.get('API_URL'),type=origin)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'qa/phase51/live-api.json')
    args=parser.parse_args()
    if not args.preview_url or not args.api_url: parser.error('PREVIEW_URL and API_URL (or explicit flags) required')
    preview_url,api_url=args.preview_url,args.api_url
    preview,html=request(preview_url)
    protected=preview.get('status') in (401,403) or preview.get('final_origin')=='https://vercel.com'
    observable=preview.get('status')==200 and not protected and preview.get('final_origin')==preview_url
    compiled=[];localhost=[];script_errors=[]
    if observable:
        for script in re.findall(r'<script[^>]+src="([^"]+)"',html)[:50]:
            url=urljoin(preview_url,script)
            if urlsplit(url).netloc!=urlsplit(preview_url).netloc: continue
            response,code=request(url)
            if response.get('status')!=200: script_errors.append(script)
            if api_url in code: compiled.append(script)
            if '127.0.0.1:8000' in code or 'localhost:8000' in code: localhost.append(script)
    health,body=request(api_url+'/health',headers={'Origin':preview_url})
    preflight,_=request(api_url+'/chat/stream','OPTIONS',{'Origin':preview_url,'Access-Control-Request-Method':'POST','Access-Control-Request-Headers':'content-type'})
    cors_observed=health.get('status') is not None and preflight.get('status') is not None
    cors=health.get('access_control_allow_origin')==preview_url and preflight.get('access_control_allow_origin')==preview_url and preflight.get('status',500)<400
    try: payload=json.loads(body)
    except (ValueError,TypeError): payload=None
    compiled_status='NOT_TESTED' if not observable or script_errors else 'PASS' if compiled and not localhost else 'FAIL'
    result={'tested_at_utc':datetime.now(timezone.utc).isoformat(),'preview_url':preview_url,'api_url':api_url,
        'preview':preview,'authentication_required':protected,'compiled_api_url':compiled_status,
        'compiled_api_matches':compiled,'compiled_localhost_matches':localhost,'script_errors':script_errors,
        'health':health,'health_payload':payload,'chat_preflight':preflight,
        'CORS':'PASS' if cors else 'FAIL' if cors_observed else 'NOT_TESTED',
        'LIVE':'NOT_TESTED','live_readiness':'PASS' if cors and compiled_status=='PASS' and isinstance(payload,dict) and payload.get('loaded') is True else 'NOT_READY',
        'manual_authenticated_preview_qa':'NOT_TESTED','reason':'GET/OPTIONS readiness is not an authenticated end-to-end student workflow. Manual authorized browser QA remains required.',
        'production_settings_changed':False,'chat_post_sent':False,'external_ai_api_called':False,'authentication_bypass':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as f: json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
