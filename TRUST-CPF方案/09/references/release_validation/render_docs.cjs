const fs=require('fs');
const root='C:/Users/llenovo/Documents/ChatGPT/USG/analysis_TRUST_CPF_stage_release_20261025';
const katex=require('../tmp/katex/katex.min.cjs');
const {chromium}=require('C:/Users/llenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const decode=x=>x.replaceAll('&lt;','<').replaceAll('&gt;','>').replaceAll('&amp;','&').replaceAll('&#x27;',"'").replaceAll('&quot;','"');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
 const page=await browser.newPage({viewport:{width:1480,height:1100},deviceScaleFactor:1});
 const all=[];
 for(const name of ['FORMULAS_AND_CODE','SERVER_COMMANDS']){
  let count=0;let src=fs.readFileSync(root+'/'+name+'.html','utf8');
  src=src.replace(/<(div|span) class="(math-block|math-inline)"[^>]*>([\s\S]*?)<\/\1>/g,(_m,tag,cls,x)=>{
   count++;return '<'+tag+' class="'+cls+'">'+katex.renderToString(decode(x),{displayMode:cls==='math-block',output:'mathml',throwOnError:true})+'</'+tag+'>';
  });
  fs.writeFileSync(root+'/'+name+'.html',src);
  await page.goto('file:///'+root+'/'+name+'.html');
  const bad=await page.locator('table,.math-block,pre').evaluateAll(nodes=>nodes.map(n=>({tag:n.tagName,scroll:n.scrollWidth,client:n.clientWidth})).filter(x=>x.scroll>x.client+2));
  await page.screenshot({path:root+'/'+name+'_top.png'});
  if(name==='FORMULAS_AND_CODE'){await page.locator('.math-block').nth(4).scrollIntoViewIfNeeded();await page.screenshot({path:root+'/formula_detail.png'});}
  else {await page.locator('pre').last().scrollIntoViewIfNeeded();await page.screenshot({path:root+'/download_commands.png'});}
  all.push({name,math_count:count,overflow:bad,passed:!bad.length});
 }
 fs.writeFileSync(root+'/document_layout.json',JSON.stringify(all,null,2));
 fs.copyFileSync(root+'/document_layout.json',root+'/../trustcpf_stage_v1_10ep/validation/document_layout.json');
 console.log(JSON.stringify(all));await browser.close();if(all.some(x=>!x.passed))process.exitCode=1;
})().catch(e=>{console.error(e);process.exitCode=1;});

