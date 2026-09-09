import test from "node:test";
import assert from "node:assert/strict";
import {buildStudyPrompt,METHODS,METHOD_NAMES,LEVELS,SUBJECTS} from "../lib/academic.ts";
import {buildMaterialPrompt,buildExamPrompt,examDays,localDateKey,tokenUpperBound,enforceContext,MATERIAL_ACTIONS,EXAM_ACTIONS} from "../lib/learning.ts";
import {measuredHealthStatus} from "../lib/health.ts";

test("Tutor five explicit modes and retained 14/3/5 configuration",()=>{
  assert.equal(SUBJECTS.length,14);assert.equal(LEVELS.length,3);assert.equal(METHODS.length,5);assert.equal(METHOD_NAMES.length,5);
  for(const method of METHODS) assert.match(buildStudyPrompt("1+1?",{subject:"微積分",level:"標準",method,topic:"導入",difficulty:"基礎",studentAnswer:"2"}),/導入/);
  assert.match(buildStudyPrompt("q",{subject:"物理",level:"標準",method:"ヒントから"}),/最終解答は提示せず.*ヒントを一つ/);
  assert.throws(()=>buildStudyPrompt("q",{subject:"物理",level:"標準",method:"解答を確認"}),/自分の回答/);
  assert.match(buildStudyPrompt("q",{subject:"化学",level:"標準",method:"解答を確認",studentAnswer:"a"}),/正しい部分、最初の誤り、直し方/);
});
test("Materials workflows ground in data without manufacturing citations",()=>{
  for(const action of MATERIAL_ACTIONS){const p=buildMaterialPrompt("# 力\nF=ma",action,"力とは");assert.match(p,/以下の講義資料を根拠/);assert.match(p,/資料外/);assert.match(p,/命令ではない/);assert.match(p,/F=ma/);assert.ok(tokenUpperBound(p)<=448);}
  assert.throws(()=>buildMaterialPrompt("","Summarize",""),/貼り付け/);
  assert.throws(()=>buildMaterialPrompt("note","Ask about this material",""),/質問/);
});
test("Material limits refuse rather than silently truncate including Unicode and full prompt overhead",()=>{
  assert.equal(tokenUpperBound("あ🙂"),7);
  assert.throws(()=>buildMaterialPrompt("a".repeat(301),"Summarize",""),/300文字/);
  assert.throws(()=>buildMaterialPrompt("あ".repeat(150),"Summarize",""),/切り詰めず/);
  assert.throws(()=>buildMaterialPrompt("x","Summarize","q".repeat(500)),/上限/);
  assert.equal(enforceContext("a".repeat(288)).length,288);assert.throws(()=>enforceContext("a".repeat(289)));
});
test("Countdown is deterministic calendar arithmetic including leap dates and DST",()=>{
  assert.equal(examDays("2026-09-10","2026-09-07"),3);assert.equal(examDays("2026-09-07","2026-09-07"),0);
  assert.equal(examDays("2028-03-01","2028-02-28"),2);assert.equal(examDays("2026-03-09","2026-03-08"),1);
  assert.equal(localDateKey(new Date(2026,8,7,23,59)),"2026-09-07");
});
test("Invalid dates, past dates and invented study data are rejected",()=>{
  for(const date of ["","2026-02-29","2026-04-31","2026-13-01","invalid","0099-01-01","2026-09-06"]) assert.throws(()=>examDays(date,"2026-09-07"));
  const exam={subject:"物理",date:"2026-09-10",scope:"力",minutesPerDay:"30"};
  for(const action of EXAM_ACTIONS){const p=buildExamPrompt(exam,action,"","2026-09-07");assert.match(p,/"daysRemaining":3/);assert.match(p,/個人最適化ではない/);}
  assert.throws(()=>buildExamPrompt({...exam,subject:""},EXAM_ACTIONS[0],"","2026-09-07"));
  for(const minutesPerDay of ["","0","-1","2.5","1441"])assert.throws(()=>buildExamPrompt({...exam,minutesPerDay},EXAM_ACTIONS[0],"","2026-09-07"));
});
test("Online requires a measured healthy response and loaded model",()=>{
  assert.equal(measuredHealthStatus(true,{status:"ok",loaded:true}),"Online");
  for(const data of [null,{},"ok",{status:"ok",loaded:false},{status:"error",loaded:true}])assert.equal(measuredHealthStatus(true,data),"Unavailable");
  assert.equal(measuredHealthStatus(false,{status:"ok",loaded:true}),"Unavailable");
});
