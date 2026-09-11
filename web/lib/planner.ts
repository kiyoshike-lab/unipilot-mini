/** Local academic records. Civil-time interpretation is explicit and rejects DST ambiguity. */
export type ClassSlot={id:string;name:string;day:number;start:string;end:string;location:string;instructor:string};
export type Attendance={id:string;courseId:string;date:string;status:'present'|'late'|'absent'|'excused'};
export type Assignment={id:string;title:string;courseId:string;due:string;timezone:string;status:'not-started'|'in-progress'|'done';priority:'normal'|'high'|'low';note:string};
export type Planner={schema:'unipilot-planner-v1';timezone:string;classes:ClassSlot[];attendance:Attendance[];assignments:Assignment[];absenceThreshold:number|null};
export const PLANNER_KEY='unipilot-planner-v1';
export const emptyPlanner=(timezone='Asia/Tokyo'):Planner=>({schema:'unipilot-planner-v1',timezone,classes:[],attendance:[],assignments:[],absenceThreshold:null});
export function validZone(zone:string){try{new Intl.DateTimeFormat('en',{timeZone:zone}).format(0);return !!zone;}catch{return false;}}
export function validDate(s:string){return /^20\d\d-\d\d-\d\d$/.test(s)&&Number.isFinite(Date.parse(s))&&new Date(s).toISOString().slice(0,10)===s;}
export function minutes(s:string){if(!/^([01]\d|2[0-3]):[0-5]\d$/.test(s))throw Error('時刻はHH:mmで入力してください。');const [h,m]=s.split(':').map(Number);return h*60+m;}
export function civilAt(epoch:number,zone:string){
  const parts=new Intl.DateTimeFormat('en-CA',{timeZone:zone,year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hourCycle:'h23'}).formatToParts(epoch);
  const p=Object.fromEntries(parts.map(x=>[x.type,x.value]));return `${p.year}-${p.month}-${p.day}T${p.hour}:${p.minute}`;
}
const timeCache=new Map<string,number>();
export function zonedEpoch(local:string,zone:string):number{
  if(!validZone(zone)||local.length!==16||!validDate(local.slice(0,10))||local[10]!=='T')throw Error('日付とIANA timezoneを確認してください（2000–2099）。');
  minutes(local.slice(11));const key=zone+'|'+local;if(timeCache.has(key))return timeCache.get(key)!;
  const nominal=Date.parse(local+':00Z'),matches:number[]=[];
  for(let offset=-14*60;offset<=14*60;offset+=15){const instant=nominal-offset*60000;if(civilAt(instant,zone)===local)matches.push(instant);}
  if(matches.length!==1)throw Error(matches.length?'夏時間切替で重複する時刻です。曖昧でない時刻を指定してください。':'存在しない現地時刻です。夏時間・日付を確認してください。');
  if(timeCache.size>2000)timeCache.clear();timeCache.set(key,matches[0]);return matches[0];
}
export function scheduleErrors(classes:ClassSlot[]){
  const errors:string[]=[];
  for(const c of classes){try{if(!c.name.trim()||!Number.isInteger(c.day)||c.day<0||c.day>6||minutes(c.start)>=minutes(c.end))errors.push('授業名・曜日・開始＜終了を確認してください。');}catch(e){errors.push((e as Error).message);}}
  for(let i=0;i<classes.length;i++)for(let j=i+1;j<classes.length;j++){const a=classes[i],b=classes[j];if(a.day===b.day&&a.start<b.end&&b.start<a.end)errors.push(`授業の重複: ${a.name} / ${b.name}`);}
  return [...new Set(errors)];
}
export function validatePlanner(p:Planner){
  const errors=scheduleErrors(p.classes);
  if(!validZone(p.timezone))errors.push('有効なIANA timezoneが必要です。');
  if(p.absenceThreshold!==null&&(!Number.isInteger(p.absenceThreshold)||p.absenceThreshold<1||p.absenceThreshold>1000))errors.push('任意の欠席注意閾値は1〜1000回です。');
  if(p.classes.length>100||p.assignments.length>500||p.attendance.length>2000)errors.push('保存上限（授業100・課題500・出席2000件）を超えています。');
  for(const list of [p.classes,p.assignments,p.attendance])if(new Set(list.map(x=>x.id)).size!==list.length||list.some(x=>!x.id))errors.push('IDが空または重複しています。');
  const ids=new Set(p.classes.map(c=>c.id)),dates=new Set<string>();
  for(const a of p.attendance){const key=a.courseId+'|'+a.date;if(!ids.has(a.courseId)||!validDate(a.date)||!['present','late','absent','excused'].includes(a.status)||dates.has(key))errors.push('出席の授業・日付・状態・同日重複を確認してください。');dates.add(key);}
  for(const a of p.assignments){if(!a.title.trim()||(a.courseId&&!ids.has(a.courseId))||!['not-started','in-progress','done'].includes(a.status)||!['normal','high','low'].includes(a.priority))errors.push('課題のタイトル・授業・状態を確認してください。');try{zonedEpoch(a.due,a.timezone);}catch(e){errors.push((e as Error).message);}}
  return [...new Set(errors)];
}
export function restorePlanner(value:unknown):Planner{
  const p=value as Planner;const str=(v:unknown,max=300)=>typeof v==='string'&&v.length<=max;
  if(!p||p.schema!=='unipilot-planner-v1'||!str(p.timezone)||!Array.isArray(p.classes)||!Array.isArray(p.assignments)||!Array.isArray(p.attendance))throw Error('Planner保存形式が不正です。');
  if(p.classes.length>100||p.assignments.length>500||p.attendance.length>2000)throw Error('Planner保存件数が上限を超えています。');
  if(p.classes.some(c=>!c||!['id','name','start','end','location','instructor'].every(k=>str(c[k as keyof ClassSlot])))||p.attendance.some(a=>!a||!['id','courseId','date','status'].every(k=>str(a[k as keyof Attendance])))||p.assignments.some(a=>!a||!['id','title','courseId','due','timezone','status','priority','note'].every(k=>str(a[k as keyof Assignment],k==='note'?2000:300))))throw Error('保存項目・文字数が不正です。');
  const errors=validatePlanner(p);if(errors.length)throw Error(errors.join(' '));return p;
}
export function attendanceSummary(p:Planner,id:string){
  const counts={present:0,late:0,absent:0,excused:0};for(const r of p.attendance.filter(a=>a.courseId===id))counts[r.status]++;
  const denominator=counts.present+counts.late+counts.absent;
  return {...counts,denominator,percentage:denominator?100*counts.present/denominator:null,warning:p.absenceThreshold!==null&&counts.absent>=p.absenceThreshold};
}
export function dueState(a:Assignment,now:number){const remainingMs=zonedEpoch(a.due,a.timezone)-now;return {remainingMs,overdue:a.status!=='done'&&remainingMs<0,done:a.status==='done'};}
export function nextClass(p:Planner,now:number){
  const today=civilAt(now,p.timezone).slice(0,10),anchor=Date.parse(today+'T00:00Z');const upcoming:{course:ClassSlot;at:number}[]=[];
  for(let d=0;d<=7;d++){const day=new Date(anchor+d*86400000);for(const c of p.classes.filter(c=>c.day===day.getUTCDay()))try{const at=zonedEpoch(day.toISOString().slice(0,10)+'T'+c.start,p.timezone);if(at>=now)upcoming.push({course:c,at});}catch{/* DST-invalid/ambiguous occurrence is not guessed. */}}
  return upcoming.sort((a,b)=>a.at-b.at||a.course.id.localeCompare(b.course.id))[0]||null;
}
export function todayView(p:Planner,now:number){
  const tasks=p.assignments.filter(a=>a.status!=='done').map(a=>({...a,...dueState(a,now)})).sort((a,b)=>a.remainingMs-b.remainingMs||({high:0,normal:1,low:2}[a.priority]-{high:0,normal:1,low:2}[b.priority])||a.id.localeCompare(b.id));
  const currentDate=civilAt(now,p.timezone).slice(0,10),next=nextClass(p,now);
  return {tasks:tasks.filter(a=>a.remainingMs<=72*3600000),today:tasks.filter(a=>civilAt(zonedEpoch(a.due,a.timezone),p.timezone).slice(0,10)===currentDate),next,warnings:p.classes.filter(c=>attendanceSummary(p,c.id).warning),recommendation:tasks.find(a=>a.remainingMs<=72*3600000)?.title||next?.course.name||null};
}
export function deleteClass(p:Planner,id:string):Planner{return {...p,classes:p.classes.filter(c=>c.id!==id),attendance:p.attendance.filter(a=>a.courseId!==id),assignments:p.assignments.map(a=>a.courseId===id?{...a,courseId:''}:a)};}
export function exportPlanner(p:Planner){restorePlanner(p);return JSON.stringify(p,null,2);}
