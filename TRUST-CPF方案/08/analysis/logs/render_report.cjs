const fs=require('fs');
const {chromium}=require('C:/Users/llenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root=__dirname.replaceAll('\\','/');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
 const page=await browser.newPage({viewport:{width:1420,height:1100},deviceScaleFactor:1});
 await page.goto('file:///'+root+'/report.html');
 await page.screenshot({path:root+'/report_top.png'});
 await page.locator('h2').filter({hasText:'2. 正式识别'}).scrollIntoViewIfNeeded();
 await page.screenshot({path:root+'/report_main.png'});
 await page.locator('h2').filter({hasText:'5. 重复响应'}).scrollIntoViewIfNeeded();
 await page.screenshot({path:root+'/report_response.png'});
 const overflow=await page.locator('table,pre,article').evaluateAll(ns=>ns.filter(n=>n.scrollWidth>n.clientWidth+3).map(n=>({tag:n.tagName,scroll:n.scrollWidth,width:n.clientWidth})));
 const images=await page.locator('img').evaluateAll(ns=>ns.map(n=>({loaded:n.complete&&n.naturalWidth>0,width:n.naturalWidth})));
 const result={passed:overflow.length===0&&images.every(i=>i.loaded),tables:await page.locator('table').count(),overflow,images};
 fs.writeFileSync(root+'/layout_verification.json',JSON.stringify(result,null,2));console.log(JSON.stringify(result));
 await browser.close();if(!result.passed)process.exitCode=1;
})().catch(e=>{console.error(e);process.exitCode=1});
