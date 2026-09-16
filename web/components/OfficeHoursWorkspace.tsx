"use client";
import {useState} from 'react';
import {OFFICE_MODES,emptyOffice,officeErrors,prepareOffice} from '../lib/officeHours';
import type {OfficeInput,OfficeResult,OfficeMode} from '../lib/officeHours';
import {FeatureStatusBadge} from './AcademicShell';
export function OfficeHoursWorkspace(){
  const [input,setInput]=useState<OfficeInput>(emptyOffice),[result,setResult]=useState<OfficeResult|null>(null),[errors,setErrors]=useState<Partial<Record<keyof OfficeInput,string>>>({});
  const update=(key:keyof OfficeInput,value:string)=>{setInput(p=>({...p,[key]:value}));setResult(null);setErrors({});};
  return <main className="workspace writing-workspace planner"><p className="eyebrow">13 / AI OFFICE HOURS · TEMPLATE WORKFLOW</p><FeatureStatusBadge status="Foundation"/><h1>分からない点を、質問に変える。</h1><p className="intro-copy">授業の理解を整理し、次の一問を準備する。</p><p className="shell-notice">モデル回答はOFF。ローカルの学習手順テンプレートです。入力はこの画面内だけで扱い、保存・送信しません。教授の代理ではありません。</p>
    <form noValidate onSubmit={e=>{e.preventDefault();const next=officeErrors(input);setErrors(next);setResult(Object.keys(next).length?null:prepareOffice(input));}}>
    <fieldset className="writing-controls"><legend>いまの理解と質問</legend><div className="evidence-fields">
    {([['course','科目（必須）'],['topic','Topic']] as const).map(([key,label])=><label key={key}>{label}<input aria-label={label} value={input[key]} required={key==='course'} maxLength={2000} aria-invalid={!!errors[key]} aria-describedby={`office-${key}-error`} onChange={e=>update(key,e.target.value)}/><span id={`office-${key}-error`}>{errors[key]}</span></label>)}
    <label>モード<select aria-label="モード" value={input.mode} onChange={e=>update('mode',e.target.value as OfficeMode)}>{OFFICE_MODES.map(mode=><option key={mode}>{mode}</option>)}</select></label></div>
    {([['question','質問（必須）'],['understood','ここまで理解したこと'],['stuck','困っている点'],['excerpt','資料の抜粋（任意）'],['source','資料名・出典（任意）']] as const).map(([key,label])=><label key={key}>{label}<textarea aria-label={label} value={input[key]} required={key==='question'} maxLength={key==='excerpt'?6000:2000} aria-invalid={!!errors[key]} aria-describedby={`office-${key}-error`} onChange={e=>update(key,e.target.value)}/><span id={`office-${key}-error`}>{errors[key]}</span></label>)}
    <p role="alert">{Object.values(errors).join(' ')}</p><div className="draft-actions"><button className="primary-button" type="submit">質問を整理する</button><button className="secondary-button" type="button" onClick={()=>{setInput(emptyOffice());setResult(null);setErrors({});}}>入力をクリア</button></div></fieldset></form>
    <div role="status" aria-live="polite">{result?'質問の整理ができました。一般的な学習手順と提供資料を区別して表示しています。':''}</div>
    {result&&<section className="evidence-section" aria-label="質問の整理結果"><h2>{result.input.mode}</h2><p>{result.disclaimer}</p><h3>提供資料（引用のみ）</h3>{result.material?<><blockquote style={{whiteSpace:'pre-wrap',overflowWrap:'anywhere'}}>{result.material.quote}</blockquote><p>{result.material.reference} / {result.material.status}</p></>:<p>資料は未提供です。資料由来の主張はありません。</p>}<h3>一般的な学習手順（資料由来ではありません）</h3><ul>{result.general.map(t=><li key={t}>{t}</li>)}</ul>{result.escalation.length>0&&<><h3>教授/TAへの確認</h3><ul>{result.escalation.map(t=><li key={t}>{t}</li>)}</ul></>}{result.professorDraft&&<><h3>教授への質問案（学生の文面）</h3><pre style={{whiteSpace:'pre-wrap',overflowWrap:'anywhere'}}>{result.professorDraft}</pre><p>コピーや送信の前に内容を自分で確認してください。自動送信はありません。</p></>}</section>}
  </main>;
}
