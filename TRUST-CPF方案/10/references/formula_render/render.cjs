const fs=require('fs');
const katex=require('../katex/katex.min.cjs');
const {chromium}=require('C:/Users/llenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root=__dirname.replaceAll('\\','/');
fs.mkdirSync(root+'/browser-tmp',{recursive:true});process.env.TEMP=root+'/browser-tmp';process.env.TMP=root+'/browser-tmp';
const equations=JSON.parse(fs.readFileSync(root+'/formulas.json','utf8'));
let body=fs.readFileSync(root+'/body.html','utf8'),errors=[];
for(const [i,q] of equations.entries()){
 try{
  const rendered=katex.renderToString(q.tex,{displayMode:q.display,output:'mathml',throwOnError:true,strict:'ignore'});
  const html=q.display?'<div class="mathblock" data-index="'+i+'">'+rendered+'</div>':'<span class="inline-math">'+rendered+'</span>';
  body=body.replace(q.display?'<p>'+q.token+'</p>':q.token,html);
 }catch(e){errors.push({index:i,error:e.message,tex:q.tex});}
}
fs.writeFileSync(root+'/math_render_checks.json',JSON.stringify({passed:errors.length===0,count:equations.length,errors},null,2));
if(errors.length){console.log(JSON.stringify(errors));process.exit(1);}
const css='body{max-width:1120px;margin:36px auto;padding:0 28px;font:16px/1.8 "Microsoft YaHei",sans-serif;color:#203348;background:white}h1{font-size:30px}h2{font-size:24px;margin-top:36px}h3{font-size:19px;margin-top:28px}table{width:100%;border-collapse:collapse;margin:18px 0;font-size:14px}th,td{padding:9px 11px;vertical-align:top;border-bottom:1px solid #ced9e2;text-align:left}th{background:#edf3f8}code{font:14px Consolas,monospace;overflow-wrap:anywhere;background:#f0f4f7}a{color:#1e5a8a;overflow-wrap:anywhere}.mathblock{font-family:"Cambria Math",serif;font-size:18px;padding:18px 0;margin:14px 0;background:#f7fafc}.inline-math{font-family:"Cambria Math",serif}math{max-width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f0f4f7;padding:16px;font:13px/1.5 Consolas,monospace}pre code{font-size:13px;background:transparent}';
fs.writeFileSync(root+'/plan.html','<!doctype html><html lang="zh-CN"><meta charset="utf-8"><style>'+css+'</style><article>'+body+'</article></html>');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
 const page=await browser.newPage({viewport:{width:1440,height:1160},deviceScaleFactor:1});
 await page.goto('file:///'+root+'/plan.html');await page.evaluate(()=>document.fonts.ready);
 const overflow=await page.locator('article,table,.mathblock').evaluateAll(ns=>ns.filter(n=>n.scrollWidth>n.clientWidth+3).map(n=>({tag:n.tagName,index:n.dataset.index,width:n.clientWidth,scroll:n.scrollWidth,text:n.innerText.slice(0,80)})));
 await page.screenshot({path:root+'/plan_top.png'});
 for(const [index,text] of [[1,'4. B：16'],[2,'7. C：'],[3,'8. 对照'],[4,'10.3 固定']]){
  await page.getByRole('heading').filter({hasText:text}).scrollIntoViewIfNeeded();
  await page.screenshot({path:root+'/plan_'+index+'.png'});
 }
 const result={passed:overflow.length===0,overflow,display:await page.locator('.mathblock').count(),tables:await page.locator('table').count()};
 fs.writeFileSync(root+'/layout_checks.json',JSON.stringify(result,null,2));console.log(JSON.stringify(result));await browser.close();
 if(!result.passed)process.exitCode=1;
})().catch(e=>{console.error(e);process.exitCode=1});
