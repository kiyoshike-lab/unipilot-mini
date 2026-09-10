"use client";
import {useState} from "react";
import {bibliography,checkSpan,editEvidence,newEvidence} from "../lib/evidence";
import type {Evidence} from "../lib/evidence";
const fields:[keyof Omit<Evidence,'id'|'status'>,string][]=[['title','Title / 資料名'],['author','Author / 著者・組織'],['date','Year / Date'],['sourceType','Source type'],['url','URL（任意）'],['doi','DOI（任意）'],['page','Page（任意）'],['sourceText','Source text / 原文'],['span','Exact supporting span / 引用候補'],['note','Student note / メモ']];
export function EvidenceLedger({items,onChange,onDelete}:{items:Evidence[];onChange:(items:Evidence[])=>void;onDelete:(id:string)=>void}) {
  const [notice,setNotice]=useState('');
  return <section className="evidence-section" aria-labelledby="evidence-heading"><div className="section-heading"><div><p className="eyebrow">SOURCE FIRST</p><h2 id="evidence-heading">Evidence Ledger</h2></div><button className="secondary-button" disabled={items.length>=40} onClick={()=>{onChange([...items,newEvidence(crypto.randomUUID())]);setNotice('空の根拠を追加しました。実在する資料の情報を入力してください。');}}>根拠を追加</button></div>
    <p className="muted">最大40件。原文の一致は出典の実在性・正確性を保証しません。Verified は外部検証のための予約状態で、ここでは付与しません。</p>
    {!items.length&&<p className="empty-evidence">まだ根拠がありません。資料名と、主張を支える原文から始めましょう。</p>}
    {items.map((item,i)=>{const bib=bibliography(item);return <fieldset className="evidence-entry" key={item.id}><legend>Evidence {i+1}</legend><p className="evidence-status">{item.status}</p><div className="evidence-fields">
      {fields.map(([key,label])=><label key={key} className={['sourceText','span','note'].includes(key)?'wide-field':''}>{label}<textarea rows={key==='sourceText'?5:key==='span'?3:2} maxLength={20000} aria-label={`${label} ${i+1}`} value={item[key]} onChange={event=>onChange(items.map(e=>e.id===item.id?editEvidence(e,{[key]:event.target.value}):e))}/></label>)}</div>
      <div className="planning-actions"><button className="secondary-button" onClick={()=>{const checked=checkSpan(item);onChange(items.map(e=>e.id===item.id?checked:e));setNotice(`Evidence ${i+1}: ${checked.status}。資料自体の検証ではありません。`);}}>原文との完全一致を確認 {i+1}</button><button className="secondary-button" onClick={()=>{onDelete(item.id);setNotice(`Evidence ${i+1}を削除しました。関連する主張は再確認してください。`);}}>根拠を削除 {i+1}</button></div>
      <div className="bibliography-preview"><h3>Bibliography preview（入力値のみ・未検証）</h3><p>{bib.text||'書誌情報は未入力です。'}</p><p className="muted">{bib.missing.length?`不足項目: ${bib.missing.join(', ')}`:'必須表示項目は入力済みです。出典の実在性は別途確認してください。'}</p></div></fieldset>;})}<p role="status" aria-live="polite">{notice}</p></section>;
}
