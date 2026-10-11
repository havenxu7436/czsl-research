// Local artifact renderer: does not control any user browser or native app.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const {pathToFileURL} = require('node:url');
const root = path.resolve(__dirname, '..');
const filename = '截至20261010_CZSL全部实验_公式修正_决策与数据分析总档案';
const explicitSource = process.argv[3];
const source = explicitSource ? path.resolve(explicitSource) : path.join(root, 'output', filename + '.md');
const katex = require('./math_render_deps/package/dist/katex.js');
const deps = 'C:/Users/llenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const {chromium} = require(path.join(deps, 'playwright'));
const stage = process.argv[2] || 'after';
const reportDir = process.argv[4] ? path.resolve(process.argv[4]) : path.join(__dirname, 'math_render_review');
fs.mkdirSync(reportDir, {recursive:true});
const escape = x => x.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');

(async () => {
  const {marked} = await import(pathToFileURL(path.join(deps, 'marked/lib/marked.esm.js')).href);
  const inputSource = stage === 'before' ? path.join(root,'output',filename+'_公式修订前.md') : source;
  const markdown = fs.readFileSync(inputSource, 'utf8');
  const math = [];
  const protectedMd = markdown.replace(/\$\$([\s\S]*?)\$\$|\$([^\n$]+)\$/g, (match, block, inline, offset) => {
    const tex = (block === undefined ? inline : block).trim();
    const displayMode = block !== undefined;
    const line = markdown.slice(0,offset).split('\n').length;
    const id = math.length;
    const rendered = katex.renderToString(tex, {displayMode,throwOnError:true,strict:'error',trust:false,output:'htmlAndMathml'});
    math.push({id,line,displayMode,tex,rendered});
    return `MATHITEMTOKEN${String(id).padStart(4,'0')}END`;
  });
  let css = fs.readFileSync(path.join(__dirname,'math_render_deps/package/dist/katex.min.css'),'utf8');
  css = css.replace(/url\(([^)]+)\)/g, (full,asset) => {
    if (asset.startsWith('data:')) return full;
    const fontPath=path.resolve(__dirname,'math_render_deps/package/dist',asset.replace(/^['"]|['"]$/g,''));
    const extension=path.extname(fontPath).slice(1);
    return `url("data:font/${extension};base64,${fs.readFileSync(fontPath).toString('base64')}")`;
  });
  const style = `body{font:16px/1.75 "Microsoft YaHei",sans-serif;color:#222;background:#fff;margin:0;padding:24px}main{max-width:1080px;margin:auto}h1{font-size:26px}h2{font-size:22px;border-bottom:1px solid #ddd;margin-top:32px}h3{font-size:18px}p{margin:14px 0}.table-scroll{max-width:100%;overflow-x:auto;margin:16px 0}table{border-collapse:collapse;font-size:13px}th,td{border:1px solid #ddd;padding:6px 10px}td[align="right"]{font-variant-numeric:tabular-nums;white-space:nowrap}th{background:#f4f6f8}code{background:#f2f3f4;font-size:.9em;overflow-wrap:anywhere}a{color:#145dad;overflow-wrap:anywhere}.math-display{padding:12px 0;margin:14px 0}.katex{font-size:1.1em}.katex-display{margin:0;text-align:left}.katex-display>.katex{text-align:left}.math-inline{white-space:nowrap}.gallery-item{padding:18px 0;border-bottom:1px solid #ddd}.label{font:13px "Microsoft YaHei",sans-serif;color:#666;margin-bottom:10px}.status{font-size:14px;color:#666}`;
  let full = marked.parse(protectedMd);
  full = full.replace(/<p>(MATHITEMTOKEN\d{4}END)<\/p>/g, '$1');
  for (const item of math) {
    const token = `MATHITEMTOKEN${String(item.id).padStart(4,'0')}END`;
    full = full.replaceAll(token,`<${item.displayMode?'div':'span'} class="math-${item.displayMode?'display':'inline'}" data-math-id="${item.id}" data-source-line="${item.line}">${item.rendered}</${item.displayMode?'div':'span'}>`);
  }
  full = full.replace(/href="\/C:\/([^"#]*)"/g, (_,target) => `href="${pathToFileURL('C:/'+target).href}"`);
  full = full.replace(/<table>/g, '<div class="table-scroll"><table>').replace(/<\/table>/g, '</table></div>');
  full = full.replace(/(<h3>7\.2[\s\S]*?)(?=<h2>8\.)/, '<section id="section-7-2">$1</section>');
  const html = (title,body) => `<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>${escape(title)}</title><style>${css}\n${style}</style><main>${body}</main></html>`;
  const gallery = math.filter(x=>x.displayMode).map(x=>`<section class="gallery-item"><div class="label">公式 ${x.id+1} · Markdown 第 ${x.line} 行</div><div class="math-display" data-math-id="${x.id}" data-source-line="${x.line}">${x.rendered}</div></section>`).join('');
  const galleryPath = path.join(reportDir,stage+'_formula_gallery.html');
  fs.writeFileSync(galleryPath,html('公式排版核对',gallery));
  const previewPath = explicitSource
    ? path.join(path.dirname(source),path.basename(source,'.md')+'_排版预览_20261011.html')
    : path.join(root,'output',filename+'_公式渲染预览.html');
  if (stage === 'after') fs.writeFileSync(previewPath,html(filename,full));
  const executablePath = [chromium.executablePath(),'C:/Program Files/Google/Chrome/Application/chrome.exe','C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'].find(x=>fs.existsSync(x));
  if (!executablePath) throw new Error('No installed local browser available for artifact rendering');
  const browser = await chromium.launch({headless:true,executablePath});
  try {
    const page = await browser.newPage();
    await page.goto(pathToFileURL(galleryPath).href);
    await page.evaluate(()=>document.fonts.ready);
    const checks = [];
    for (const width of [1180,720,620]) {
      await page.setViewportSize({width,height:1000});
      const measures = await page.locator('.math-display').evaluateAll(items=>items.map(item=>{
        const formula=item.querySelector('.katex-display>.katex>.katex-html');
        const bases=Array.from(formula.querySelectorAll(':scope > .base')).map(x=>x.getBoundingClientRect());
        const naturalWidth=bases.length?Math.max(...bases.map(x=>x.right))-Math.min(...bases.map(x=>x.left)):formula.scrollWidth;
        return {id:Number(item.dataset.mathId),line:Number(item.dataset.sourceLine),available:item.clientWidth,width:Math.ceil(naturalWidth),height:Math.ceil(formula.getBoundingClientRect().height),overflow:naturalWidth>item.clientWidth+1};
      }));
      checks.push({viewport:width,display_formulas:measures.length,overflow:measures.filter(x=>x.overflow),measures});
    }
    await page.setViewportSize({width:1180,height:1400});
    const galleryHeight = await page.evaluate(()=>Math.ceil(document.documentElement.scrollHeight));
    await page.setViewportSize({width:1180,height:galleryHeight});
    const cards = page.locator('.gallery-item');
    const cardCount = await cards.count();
    const shots = [];
    for (let start=0;start<cardCount;start+=6) {
      const end=Math.min(cardCount-1,start+5);
      const first=await cards.nth(start).boundingBox();
      const last=await cards.nth(end).boundingBox();
      const output=path.join(reportDir,`${stage}_formula_${String(start+1).padStart(2,'0')}-${String(end+1).padStart(2,'0')}.png`);
      await page.screenshot({path:output,clip:{x:0,y:first.y,width:1180,height:Math.ceil(last.y+last.height-first.y)}});
      shots.push(output);
    }
    const documentChecks=[];
    if (stage==='after') {
      await page.goto(pathToFileURL(previewPath).href);
      await page.evaluate(()=>document.fonts.ready);
      if (await page.locator('.katex-error').count()) throw new Error('KaTeX error found in full Markdown preview');
      if ((await page.locator('.math-inline,.math-display').count()) !== math.length) throw new Error('Math extraction count differs from full Markdown preview');
      for (const width of [1180,720,620]) {
        await page.setViewportSize({width,height:1000});
        const layout=await page.evaluate(()=>{
          let decimalTokens=0;
          const wrappedDecimals=[];
          for (const td of document.querySelectorAll('td')) {
            const walker=document.createTreeWalker(td,NodeFilter.SHOW_TEXT);
            let node;
            while ((node=walker.nextNode())) {
              if (node.parentElement.closest('.katex')) continue;
              for (const match of node.textContent.matchAll(/[+−-]?\d+\.\d+(?:e[+−-]?\d+)?/g)) {
                decimalTokens++;
                const range=document.createRange();
                range.setStart(node,match.index);
                range.setEnd(node,match.index+match[0].length);
                const tops=new Set(Array.from(range.getClientRects(),r=>Math.round(r.top)));
                if (tops.size>1) wrappedDecimals.push(match[0]);
              }
            }
          }
          return {tables:document.querySelectorAll('table').length,decimalTokens,wrappedDecimals,
            documentOverflow:document.documentElement.scrollWidth>window.innerWidth+1,
            section72Tables:document.querySelectorAll('#section-7-2 table').length};
        });
        if (layout.wrappedDecimals.length || layout.documentOverflow || layout.section72Tables!==3) {
          throw new Error(`Document data layout failed at ${width}: ${JSON.stringify(layout)}`);
        }
        documentChecks.push({viewport:width,...layout});
        const sectionShot=path.join(reportDir,`${stage}_section_7_2_${width}.png`);
        const sectionMarkup=await page.locator('#section-7-2').evaluate(element=>element.outerHTML);
        const sectionPage=await browser.newPage({viewport:{width,height:1000}});
        try {
          await sectionPage.setContent(html('7.2 数据排版复核',sectionMarkup));
          await sectionPage.evaluate(()=>document.fonts.ready);
          await sectionPage.screenshot({path:sectionShot,fullPage:true});
        } finally {await sectionPage.close();}
        shots.push(sectionShot);
      }
    }
    let unchangedTables=null;
    if (stage==='after') {
      const audit=JSON.parse(fs.readFileSync(path.join(__dirname,'format_review_r3/data_audit.json'),'utf8'));
      const digest=crypto.createHash('sha256').update(fs.readFileSync(source)).digest('hex');
      unchangedTables=audit.passed && audit.experimental_values_preserved && audit.markdown_sha256===digest;
      if (!unchangedTables) throw new Error('Missing or stale source/data formatting audit');
      if (checks.some(x=>x.overflow.length)) throw new Error('Display formula overflows at tested panel width');
    }
    const report={stage,source_md:source,source_sha256:crypto.createHash('sha256').update(fs.readFileSync(inputSource)).digest('hex'),katex_version:katex.version,display_formulas:math.filter(x=>x.displayMode).length,inline_formulas:math.filter(x=>!x.displayMode).length,parse_errors:0,checks,document_checks:documentChecks,screenshots:shots,full_preview:stage==='after'?previewPath:null,experimental_values_preserved:unchangedTables,preview_fonts_embedded:true};
    fs.writeFileSync(path.join(reportDir,stage+'_math_audit.json'),JSON.stringify(report,null,2)+'\n');
    console.log(JSON.stringify({stage,display_formulas:report.display_formulas,inline_formulas:report.inline_formulas,parse_errors:0,overflows:checks.map(x=>({viewport:x.viewport,count:x.overflow.length,lines:x.overflow.map(y=>y.line)})),document_checks:documentChecks,screenshots:shots.length}));
  } finally {await browser.close();}
})().catch(error=>{console.error(error.stack);process.exitCode=1;});
