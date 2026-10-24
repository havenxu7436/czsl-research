const fs=require('fs');
const root='C:/Users/llenovo/Documents/ChatGPT/USG/analysis_TRUST_CPF_r5_10ep_20261024';
const katex=require('../tmp/katex/katex.min.cjs');
const {chromium}=require('C:/Users/llenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
let src=fs.readFileSync(root+'/report_preview.html','utf8');
const decode=x=>x.replaceAll('&lt;','<').replaceAll('&gt;','>').replaceAll('&amp;','&').replaceAll('&#x27;',"'").replaceAll('&quot;','"');
let count=0;
src=src.replace(/<(div|span) class="(math-block|math-inline)"[^>]*>([\s\S]*?)<\/\1>/g,(_m,tag,cls,x)=>{
 count++;return `<${tag} class="${cls}">${katex.renderToString(decode(x),{displayMode:cls==='math-block',output:'mathml',throwOnError:true})}</${tag}>`;
});
fs.writeFileSync(root+'/report_preview.html',src);
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
 const page=await browser.newPage({viewport:{width:1480,height:1100},deviceScaleFactor:1});
 await page.goto('file:///'+root+'/report_preview.html');
 const bad=await page.locator('table,.math-block,img').evaluateAll(nodes=>nodes.map(n=>({tag:n.tagName,width:n.getBoundingClientRect().width,scroll:n.scrollWidth,client:n.clientWidth,broken:n.tagName==='IMG'&&!n.naturalWidth})).filter(x=>x.scroll>x.client+2||x.broken||x.width>1250));
 await page.screenshot({path:root+'/preview_top.png'});
 await page.locator('.math-block').first().scrollIntoViewIfNeeded();
 await page.screenshot({path:root+'/preview_formulas.png'});
 const result={rendered_math:count,overflow_or_broken:bad,passed:!bad.length};
 fs.writeFileSync(root+'/layout_validation.json',JSON.stringify(result,null,2));
 console.log(JSON.stringify(result));await browser.close();if(bad.length)process.exitCode=1;
})().catch(e=>{console.error(e);process.exitCode=1});
