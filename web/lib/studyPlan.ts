/** Day-based scheduler: integer minutes, explicit demand, no generated events. */
export const PLAN_KEY='unipilot-study-plan-v1';
export type Topic={id:string;subject:string;topic:string;minutes:number;priority:number;confidence:number;deadline:string;source:'manual'|'memory'|'planner';sourceId:string;context:string};
export type PlanInput={goal:string;start:string;target:string;weekdays:number[];minutesPerDay:number;topics:Topic[]};
export type Block={id:string;topicId:string;subject:string;topic:string;date:string;duration:number;reason:string;status:'Planned'|'Done'|'Skipped'};
export type StudyPlan={schema:'unipilot-study-plan-v1';input:PlanInput;blocks:Block[];remaining:{topicId:string;minutes:number}[];overCapacity:boolean};
export const emptyPlanInput=():PlanInput=>({goal:'',start:'',target:'',weekdays:[1,2,3,4,5],minutesPerDay:60,topics:[]});
export function dateNumber(s:string){const n=Date.parse(s+'T00:00:00Z');if(!/^20\d\d-\d\d-\d\d$/.test(s)||!Number.isFinite(n)||new Date(n).toISOString().slice(0,10)!==s)throw Error('日付は2000〜2099の有効な日付が必要です。');return n/86400000;}
export function validatePlanInput(p:PlanInput){
  if(!p||typeof p.goal!=='string'||!p.goal.trim()||p.goal.length>2000)throw Error('学習目標を入力してください。');
  const start=dateNumber(p.start),end=dateNumber(p.target);if(end<start||end-start>366)throw Error('期限は開始日以降、366日以内で指定してください。');
  if(!Number.isInteger(p.minutesPerDay)||p.minutesPerDay<0||p.minutesPerDay>720)throw Error('1日の学習可能時間は0〜720分の整数です。');
  if(!Array.isArray(p.weekdays)||new Set(p.weekdays).size!==p.weekdays.length||p.weekdays.some(d=>!Number.isInteger(d)||d<0||d>6))throw Error('学習曜日を確認してください。');
  if(!Array.isArray(p.topics)||!p.topics.length||p.topics.length>100)throw Error('Topicを1〜100件入力してください。');
  if(new Set(p.topics.map(t=>t.id)).size!==p.topics.length)throw Error('Topic IDが重複しています。');
  for(const t of p.topics){
    if(!t||['id','subject','topic','context','sourceId','deadline'].some(k=>typeof t[k as keyof Topic]!=='string'||String(t[k as keyof Topic]).length>2000)||!t.id||!t.subject.trim()||!t.topic.trim())throw Error('科目・topic・出典を確認してください。');
    if(!Number.isInteger(t.minutes)||t.minutes<1||t.minutes>10080||!Number.isInteger(t.priority)||t.priority<1||t.priority>3||!Number.isInteger(t.confidence)||t.confidence<1||t.confidence>5)throw Error('必要時間1〜10080分、優先度1〜3、自己評価1〜5を指定してください。');
    if(!['manual','memory','planner'].includes(t.source)||(t.source!=='manual'&&!t.sourceId))throw Error('取り込み元が不正です。');
    if(t.deadline)dateNumber(t.deadline);
  }
}
export function schedule(p:PlanInput,done:Block[]=[]):StudyPlan{
  validatePlanInput(p);
  const topics=[...p.topics].sort((a,b)=>(a.deadline||p.target).localeCompare(b.deadline||p.target)||b.priority-a.priority||a.confidence-b.confidence||a.id.localeCompare(b.id));
  const blocks=done.map(b=>({...b})),remaining=new Map(topics.map(t=>[t.id,Math.max(0,t.minutes-done.filter(b=>b.topicId===t.id).reduce((s,b)=>s+b.duration,0))]));
  const usedIds=new Set(blocks.map(b=>b.id));let serial=0;
  for(let day=dateNumber(p.start);day<=dateNumber(p.target);day++){
    const date=new Date(day*86400000).toISOString().slice(0,10);if(!p.weekdays.includes(new Date(day*86400000).getUTCDay()))continue;
    let capacity=Math.max(0,p.minutesPerDay-done.filter(b=>b.date===date).reduce((s,b)=>s+b.duration,0));
    for(const t of topics){if(t.deadline&&date>t.deadline)continue;let needed=remaining.get(t.id)!;
      while(needed>0&&capacity>0){const duration=Math.min(30,needed,capacity);let id:string;do{id=`${t.id}|${date}|${serial++}`;}while(usedIds.has(id));usedIds.add(id);blocks.push({id,topicId:t.id,subject:t.subject,topic:t.topic,date,duration,reason:`期限順 → 優先度${t.priority} → 自己評価${t.confidence}/5（同条件時）`,status:'Planned'});needed-=duration;capacity-=duration;}remaining.set(t.id,needed);
    }
  }
  const rest=[...remaining].filter(([,minutes])=>minutes>0).map(([topicId,minutes])=>({topicId,minutes}));
  return {schema:PLAN_KEY,input:structuredClone(p),blocks,remaining:rest,overCapacity:rest.length>0};
}
export function reschedule(plan:StudyPlan,from:string){
  restoreStudyPlan(plan);dateNumber(from);if(from<plan.input.start)throw Error('再配置日は開始日以降にしてください。');
  // Completed work retains its original dates; all unfinished demand is repacked.
  return schedule({...plan.input,start:from},plan.blocks.filter(b=>b.status==='Done'));
}
export function restoreStudyPlan(value:unknown):StudyPlan{
  const p=value as StudyPlan;if(!p||p.schema!==PLAN_KEY||!Array.isArray(p.blocks)||p.blocks.length>15000||!Array.isArray(p.remaining))throw Error('Study Planの保存形式が不正です。');validatePlanInput(p.input);
  if(new Set(p.blocks.map(b=>b.id)).size!==p.blocks.length)throw Error('Block IDが重複しています。');
  const duration=new Map<string,number>(),used=new Map<string,number>();
  for(const b of p.blocks){const t=p.input.topics.find(t=>t.id===b.topicId);if(!t||typeof b.id!=='string'||!b.id||!['Planned','Done','Skipped'].includes(b.status)||!Number.isInteger(b.duration)||b.duration<1||b.duration>30||b.subject!==t.subject||b.topic!==t.topic||typeof b.reason!=='string'||b.reason.length>2000)throw Error('Study blockが不正です。');dateNumber(b.date);if(b.date>p.input.target||(t.deadline&&b.date>t.deadline)||(b.status!=='Done'&&b.date<p.input.start))throw Error('Study blockの期限が不正です。');duration.set(b.date,(duration.get(b.date)||0)+b.duration);used.set(t.id,(used.get(t.id)||0)+b.duration);}
  if([...duration.values()].some(n=>n>p.input.minutesPerDay)||p.input.topics.some(t=>(used.get(t.id)||0)>t.minutes))throw Error('保存済み計画が時間上限を超えています。');
  if(p.remaining.length>100||new Set(p.remaining.map(r=>r.topicId)).size!==p.remaining.length||p.remaining.some(r=>!p.input.topics.some(t=>t.id===r.topicId)||!Number.isInteger(r.minutes)||r.minutes<1))throw Error('未配置時間が不正です。');
  for(const t of p.input.topics)if((used.get(t.id)||0)+(p.remaining.find(r=>r.topicId===t.id)?.minutes||0)!==t.minutes)throw Error('計画の必要時間と配置時間が一致しません。');
  if(p.overCapacity!==(p.remaining.length>0))throw Error('容量表示が不正です。');return p;
}
export function exportStudyPlan(p:StudyPlan){return JSON.stringify(restoreStudyPlan(p),null,2);}
export function nextStudyBlock(p:StudyPlan,today:string):Block|null{dateNumber(today);return p.blocks.filter(b=>b.status==='Planned'&&b.date>=today).sort((a,b)=>a.date.localeCompare(b.date)||a.id.localeCompare(b.id))[0]||null;}
