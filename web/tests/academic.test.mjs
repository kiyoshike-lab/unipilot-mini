import test from "node:test";
import assert from "node:assert/strict";
import {buildStudyPrompt,sourceStatus,safeSourceUrl,SUBJECTS} from "../lib/academic.ts";
import {readSnapshots,sendChat} from "../lib/chat.ts";

test("Tutor has 14 subjects and adds only explicit selected settings",()=>{
  assert.equal(SUBJECTS.length,14); const p=buildStudyPrompt('固有値\n"とは？',{subject:"線形代数",level:"やさしく",method:"ヒントから"});
  assert.ok(p.includes(JSON.stringify('固有値\n"とは？'))); assert.ok(p.includes('"method":"ヒントから"'));
  assert.throws(()=>buildStudyPrompt("q",{subject:"invented",level:"標準",method:"考え方から"}));
});
test("source status never infers verification from timestamp/confidence",()=>{
  assert.equal(safeSourceUrl("javascript:alert(1)"),undefined);assert.equal(safeSourceUrl("https://user:password@example.com"),undefined);
  assert.equal(sourceStatus({id:"1",title:"t"}),"確認状態：不明");
  assert.match(sourceStatus({id:"1",title:"t",last_verified_at:"2026-09-06",confidence:"high"}),/Needs verification/);
  assert.match(sourceStatus({id:"1",title:"t",verified:true,stale:true}),/Stale/);
});
test("NDJSON handles split UTF-8, incremental snapshots and final line without newline",async()=>{
  const bytes=new TextEncoder().encode('{"text":"日本"}\n{"text":"日本語"}'); let observed=[];
  const body=new ReadableStream({start(c){for(let i=0;i<bytes.length;i+=2)c.enqueue(bytes.slice(i,i+2));c.close();}});
  await readSnapshots(body,s=>observed.push(s.text)); assert.deepEqual(observed,["日本","日本語"]);
});
test("HTTP stream failure preserves request/session/mode in fallback",async()=>{
  const old=globalThis.fetch; const calls=[];
  globalThis.fetch=async(url,opts)=>{calls.push({url,body:opts.body});return calls.length===1?new Response("",{status:503}):Response.json({text:"ok",cards:[]});};
  try { const out=[]; assert.equal(await sendChat("q","detailed","session-test",s=>out.push(s.text),new AbortController().signal),"fallback");
    assert.deepEqual(out,["ok"]); assert.equal(calls[0].body,calls[1].body); const body=JSON.parse(calls[0].body);assert.equal(body.session_id,"session-test");assert.equal(body.response_mode,"detailed");
  }finally{globalThis.fetch=old;}
});
test("network stream failure falls back; post-header parse failures never duplicate a request",async()=>{
  const old=globalThis.fetch;let count=0;
  globalThis.fetch=async()=>{count++;if(count===1)throw new TypeError("network");return Response.json({text:"fallback"});};
  try {assert.equal(await sendChat("q","normal","s",()=>{},new AbortController().signal),"fallback");
    count=0;globalThis.fetch=async()=>{count++;return new Response('not json\n');};
    await assert.rejects(sendChat("q","normal","s",()=>{},new AbortController().signal));assert.equal(count,1);
  }finally{globalThis.fetch=old;}
});
