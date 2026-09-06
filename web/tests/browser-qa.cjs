// Local-only QA. API fixtures are explicitly Demo; no production request is sent.
const {chromium}=require(process.env.UNIPILOT_PLAYWRIGHT_MODULE || 'playwright');
const fs=require('node:fs'); const path=require('node:path'); const assert=require('node:assert/strict');
const base=process.env.UNIPILOT_QA_URL || 'http://127.0.0.1:3049';
const out=path.resolve(__dirname,'../qa'); fs.mkdirSync(out,{recursive:true});
(async()=>{
 const browser=await chromium.launch({headless:true}); const context=await browser.newContext({viewport:{width:1440,height:1000}});
 const page=await context.newPage(); const errors=[]; page.on('pageerror',e=>errors.push(e.message)); const requests=[];let mode='stream';
 await context.addInitScript(()=>{Object.defineProperty(navigator,'clipboard',{value:{writeText:async(text)=>{window.__copied=text;}}});});
 await context.route('**/*',async route=>{
   const req=route.request();const url=new URL(req.url());
   if(url.pathname==='/chat/stream'||url.pathname==='/chat'){
     const body=req.postDataJSON();requests.push({path:url.pathname,body});
     const headers={'access-control-allow-origin':'*','access-control-allow-headers':'content-type'};
     if(req.method()==='OPTIONS'){await route.fulfill({status:204,headers});return;}
     if(mode==='offline'){await route.abort();return;}
     if(mode==='fallback'&&url.pathname==='/chat/stream'){await route.fulfill({status:503,headers});return;}
     const cards=[{kind:'clarify',title:'Demo: 確認',summary:'QA fixture',data:{options:[{category:'test',label:'Demo: 詳しく',prompt:'Demo: 詳しく知りたい'}]}},
       {kind:'tool',title:'Demo: メール',summary:'テスト用のコピー',action_label:'Demoをコピー',copy_text:'Demo: copied'},
       {kind:'sources',title:'Demo: 出典',summary:'テスト用メタデータ',data:{sources:[{id:'demo-1',title:'Demo: 検証用資料',publisher:'Demo publisher',url:'https://example.com/demo',license:'Demo license',last_verified_at:'2026-09-06',confidence:'high',stale:true}]}}];
     const snapshot={text:mode==='fallback'?'Demo: fallback応答':'Demo: streaming応答',cards};
     await route.fulfill({status:200,headers:{...headers,'content-type':url.pathname==='/chat'?'application/json':'application/x-ndjson'},body:url.pathname==='/chat'?JSON.stringify(snapshot):JSON.stringify({text:'Demo: streaming'})+'\n'+JSON.stringify(snapshot)});return;
   }
   if(url.origin!==new URL(base).origin){await route.abort();return;}
   await route.continue();
 });
 const widths=[360,390,768,1024,1440];const layouts=[];
 for(const width of widths){await page.setViewportSize({width,height:900});for(const route of ['/','/study','/report','/research','/sources']){
   await page.goto(base+route);await page.locator('h1').waitFor();
   const size=await page.evaluate(()=>({scroll:document.documentElement.scrollWidth,client:document.documentElement.clientWidth}));
   assert.ok(size.scroll<=size.client,`${route} overflows at ${width}: ${JSON.stringify(size)}`);layouts.push({route,width,overflow:false});
   if((width===1440||width===390)&&(route==='/'||route==='/study'))await page.screenshot({path:path.join(out,`${route==='/study'?'study':'home'}-${width}.png`),fullPage:true});
 }}
 await page.setViewportSize({width:1440,height:1000});await page.goto(base+'/');
 await page.getByLabel('大学生活について聞く',{exact:true}).fill('Demo: 質問');await page.getByLabel('回答の長さ').selectOption('detailed');await page.getByRole('button',{name:'送信 ↗',exact:true}).click();
 await page.getByText('Demo: streaming応答',{exact:true}).waitFor();assert.equal(requests.at(-1).body.response_mode,'detailed');assert.ok(requests.at(-1).body.session_id);
 const session=requests.at(-1).body.session_id;
 await page.getByRole('button',{name:'Demo: 詳しく',exact:true}).click();assert.equal(await page.getByLabel('大学生活について聞く',{exact:true}).inputValue(),'Demo: 詳しく知りたい');
 await page.getByRole('button',{name:'Demoをコピー',exact:true}).click();assert.equal(await page.evaluate(()=>window.__copied),'Demo: copied');
 assert.ok(await page.getByText('Demo publisher',{exact:true}).count());assert.ok(await page.getByText('Demo license',{exact:true}).count());assert.ok(await page.getByText('Stale · 要再確認',{exact:true}).count());
 mode='fallback';await page.getByRole('button',{name:'送信 ↗',exact:true}).click();await page.getByText('Demo: fallback応答',{exact:true}).waitFor();assert.equal(requests.at(-1).path,'/chat');
 await page.locator('.mode-switcher').getByRole('link',{name:'Study',exact:true}).click();await page.waitForURL(base+'/study');await page.getByLabel('科目',{exact:true}).selectOption('微積分');await page.getByLabel('説明レベル',{exact:true}).selectOption('やさしく');await page.getByLabel('学習方法',{exact:true}).selectOption('ヒントから');
 await page.getByLabel('質問・問題文',{exact:true}).fill('Demo: 微分とは');await page.getByRole('button',{name:'送信 ↗',exact:true}).click();await page.getByText('Demo: fallback応答',{exact:true}).waitFor();
 assert.ok(requests.at(-1).body.prompt.includes('微積分'));assert.ok(requests.at(-1).body.prompt.includes('ヒントから'));assert.equal(requests.at(-1).body.session_id,session);
 mode='offline';await page.getByLabel('質問・問題文',{exact:true}).fill('Demo: offline');await page.getByRole('button',{name:'送信 ↗',exact:true}).click();await page.locator('.api-error').waitFor();
 mode='fallback';await page.getByRole('button',{name:'同じ質問を再試行'}).click();await page.getByRole('status').filter({hasText:'/chat fallback'}).waitFor();
 await page.goto(base+'/report');await page.getByLabel('テーマ',{exact:true}).fill('Demo: 下書き');await page.getByRole('button',{name:'このタブに保存'}).click();await page.reload();assert.equal(await page.getByLabel('テーマ',{exact:true}).inputValue(),'Demo: 下書き');
 await page.emulateMedia({reducedMotion:'reduce'});assert.equal(await page.evaluate(()=>matchMedia('(prefers-reduced-motion: reduce)').matches),true);
 await page.goto(base+'/');await page.keyboard.press('Tab');const focus=await page.evaluate(()=>({text:document.activeElement?.textContent,outline:getComputedStyle(document.activeElement).outlineStyle}));assert.equal(focus.outline,'solid');
 // Verify a visible intermediate state, not just the last buffered snapshot.
 await page.evaluate(()=>{const original=window.fetch;window.fetch=async(...args)=>{if(String(args[0]).endsWith('/chat/stream'))return new Response(new ReadableStream({async start(c){const encoder=new TextEncoder();c.enqueue(encoder.encode('{"text":"Demo: first chunk"}\n'));await new Promise(r=>setTimeout(r,1200));c.enqueue(encoder.encode('{"text":"Demo: final chunk"}'));c.close();}}));return original(...args);};});
 await page.getByLabel('大学生活について聞く',{exact:true}).fill('Demo: gradual');await page.getByRole('button',{name:'送信 ↗',exact:true}).click();await page.getByText('Demo: first chunk',{exact:true}).waitFor();await page.getByText('Demo: final chunk',{exact:true}).waitFor();
 assert.deepEqual(errors,[]);fs.writeFileSync(path.join(out,'results.json'),JSON.stringify({layouts,functional:{chat:true,stream:true,visibleIntermediateStream:true,fallback:true,responseMode:true,session:true,toolCards:true,clarify:true,clipboard:true,sourceMetadata:true,studyPrompt:true,routeNavigation:true,offlineRetry:true,reportSave:true,reducedMotion:true,keyboardFocus:true},pageErrors:errors,fixtureOnly:true,externalApiRequestsSent:false},null,2)+'\n');
 await browser.close();console.log('PASS: 25 responsive route checks and functional browser QA');
})().catch(e=>{console.error(e);process.exit(1);});
