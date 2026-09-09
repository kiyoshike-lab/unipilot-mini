"use client";
import {useEffect, useState} from "react";
import {API} from "../lib/chat";
import {measuredHealthStatus, type ApiStatus as Status} from "../lib/health";
export function ApiStatus() {
  const [status,setStatus]=useState<Status>("Connecting"); const [attempt,setAttempt]=useState(0);
  useEffect(()=>{
    let active=true; const ctrl=new AbortController(); setStatus("Connecting");
    const waking=setTimeout(()=>{if(active)setStatus("Waking API");},5000);
    const timeout=setTimeout(()=>ctrl.abort(),30000);
    void fetch(`${API}/health`,{signal:ctrl.signal,cache:"no-store"}).then(async r=>{
      const data=await r.json();if(active)setStatus(measuredHealthStatus(r.ok,data));
    }).catch(()=>{if(active)setStatus("Unavailable");}).finally(()=>{clearTimeout(waking);clearTimeout(timeout);});
    return ()=>{active=false;ctrl.abort();clearTimeout(waking);clearTimeout(timeout);};
  },[attempt]);
  return <button className="api-health" type="button" onClick={()=>setAttempt(x=>x+1)} title={`GET ${API}/health を再確認。Onlineはhealth確認のみで回答品質の保証ではありません。`} aria-label={`API ${status}・接続を再確認`}><span role="status">API · {status}</span></button>;
}
