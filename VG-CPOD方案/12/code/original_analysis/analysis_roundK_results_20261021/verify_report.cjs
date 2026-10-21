const fs=require('fs');
const path=require('path');
const {pathToFileURL}=require('url');
const {chromium}=require('C:/Users/llenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

(async()=>{
  const root=path.resolve(__dirname,'..');
  const name='VG-CPOD_v1_K组结果分析与后续主线决策_20261021';
  const md=fs.readFileSync(path.join(root,name+'.md'),'utf8');
  const qa=path.join(__dirname,'qa');fs.mkdirSync(qa,{recursive:true});
  const browser=await chromium.launch({channel:'msedge',headless:true});
  try{
    const page=await browser.newPage({viewport:{width:1440,height:1100},deviceScaleFactor:1});
    const errors=[];page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
    await page.goto(pathToFileURL(path.join(root,name+'.html')).href);
    await page.evaluate(()=>document.fonts.ready);
    const result=await page.evaluate(()=>({
      displayMath:document.querySelectorAll('math[display="block"]').length,
      mathCount:document.querySelectorAll('math').length,
      unrenderedMath:document.querySelectorAll('span.math').length,
      horizontalOverflow:document.documentElement.scrollWidth>innerWidth,
      wideTables:[...document.querySelectorAll('table')].filter(x=>x.getBoundingClientRect().right>innerWidth).length,
      wideMath:[...document.querySelectorAll('math')].filter(x=>x.getBoundingClientRect().right>innerWidth).length,
      tables:document.querySelectorAll('table').length,
      images:[...document.images].map(x=>({loaded:x.complete&&x.naturalWidth>0,width:x.naturalWidth}))
    }));
    result.expectedDisplayMath=(md.match(/\$\$/g)||[]).length/2;
    result.consoleErrors=errors;
    await page.screenshot({path:path.join(qa,'01_results.png')});
    for(const [selector,title,file] of [
      ['h3','3.2 属性与物体','02_ranking.png'],
      ['h3','5.1 与近邻','03_novelty.png'],
      ['h3','6.1 共同基座','04_scorer.png'],
      ['h3','6.3 核心改变','05_core_formula.png'],
      ['h3','6.4 将成对','06_graph_formula.png'],
      ['h3','6.5 一套完整','07_plan.png']]){
      await page.locator(selector).filter({hasText:title}).evaluate(x=>x.scrollIntoView({block:'start'}));
      await page.screenshot({path:path.join(qa,file)});
    }
    fs.writeFileSync(path.join(qa,'layout.json'),JSON.stringify(result,null,2));
    console.log(JSON.stringify(result,null,2));
    if(result.displayMath!==result.expectedDisplayMath||result.unrenderedMath||result.horizontalOverflow||result.wideTables||result.wideMath||result.images.some(x=>!x.loaded))throw new Error('Layout requires correction');
  }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
