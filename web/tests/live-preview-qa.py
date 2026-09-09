"""Read-only live diagnostic. No chat POST, deployment or security-setting mutation."""
import json
import re
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse, urljoin

API = 'https://unipilot-mini.onrender.com'
PREVIEW = 'https://unipilot-mini-pjgy-jrrblw0tq-kiyoshike-labs-projects.vercel.app'


def request(url, method='GET', headers=None):
    req=urllib.request.Request(url,method=method,headers={'User-Agent':'UniPilot-PHASE50-read-only-QA',**(headers or {})})
    try:
        with urllib.request.urlopen(req,timeout=30) as r:
            text=r.read().decode('utf-8'); final=urlparse(r.url)
            return {'status':r.status,'final_origin':f'{final.scheme}://{final.netloc}','final_path':final.path,
                'access_control_allow_origin':r.headers.get('Access-Control-Allow-Origin'),
                'access_control_allow_methods':r.headers.get('Access-Control-Allow-Methods')},text
    except urllib.error.HTTPError as e:
        return {'status':e.code,'access_control_allow_origin':e.headers.get('Access-Control-Allow-Origin'),'error':e.reason},e.read().decode('utf-8')
    except Exception as e: return {'error':type(e).__name__+': '+str(e)},''


if __name__=='__main__':
    preview,html=request(PREVIEW)
    protected=preview.get('final_origin')=='https://vercel.com' and preview.get('final_path')=='/login'
    compiled_urls=[]; localhost=[]
    if not protected and preview.get('status')==200:
        for script in re.findall(r'<script[^>]+src="([^"]+)"',html):
            url=urljoin(PREVIEW,script)
            if urlparse(url).netloc!=urlparse(PREVIEW).netloc: continue
            _,code=request(url)
            if API in code: compiled_urls.append(script)
            if '127.0.0.1:8000' in code or 'localhost:8000' in code: localhost.append(script)
    health,body=request(API+'/health',headers={'Origin':PREVIEW})
    preflight,_=request(API+'/chat/stream','OPTIONS',{'Origin':PREVIEW,'Access-Control-Request-Method':'POST','Access-Control-Request-Headers':'content-type'})
    cors=health.get('access_control_allow_origin')==PREVIEW and preflight.get('access_control_allow_origin')==PREVIEW and preflight.get('status',500)<400
    result={'tested_at_utc':datetime.now(timezone.utc).isoformat(),'preview_url':PREVIEW,'preview_source':'GitHub deployment 6296102205, success, PHASE49 HEAD c5012cdd',
        'preview':preview,'authentication_required':protected,'NEXT_PUBLIC_API_URL':'NOT_TESTED' if protected else 'correct' if compiled_urls and not localhost else 'incorrect',
        'expected_api_url':API,'compiled_api_matches':compiled_urls,'compiled_localhost_matches':localhost,
        'health':health,'health_payload':json.loads(body) if body.startswith('{') else None,'chat_preflight':preflight,
        'CORS':'PASS' if cors else 'FAIL' if health.get('status') else 'NOT_TESTED',
        'LIVE':'NOT_TESTED' if protected else 'PASS' if cors and compiled_urls and not localhost else 'FAIL',
        'initial_health_attempt':'25 second read timeout; later HTTPS GET 200. Cold start is plausible, not proved.',
        'historical_observation_2026_09_07':{'health_status':200,'chat_preflight_status':400,'access_control_allow_origin':None,'CORS':'FAIL'},
        'diagnosis':('Preview redirects to Vercel login; built app JS is not observable. ' if protected else '') +
            ('Render health responded; inspect the recorded CORS headers/preflight status.' if health.get('status') else 'Current Render health request failed or timed out; current CORS behavior could not be measured. Prior 2026-09-07 CORS rejection is historical, not a result of this attempt.'),
        'next_action':'Use an authorized preview session to inspect built API URL; deployment owner must explicitly allow the exact approved preview origin in Render CORS configuration. Do not use wildcard origins or disable Vercel protection.',
        'production_settings_changed':False,'chat_post_sent':False,'external_ai_api_called':False}
    out=Path(__file__).resolve().parents[1]/'qa/phase50/live-api.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
