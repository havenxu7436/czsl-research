// Check links, table structure and GitHub mathematical Markdown; emit review HTML.
const fs = require('fs');
const path = require('path');
const katex = require(process.env.KATEX_MODULE || 'katex');
const {marked} = require(process.env.MARKED_MODULE || 'marked');
const root = path.resolve(process.argv[2]);
const version = path.join(root, 'TRUST-CPF方案/v1');
const output = path.join(version, 'results/server-results-20261022/validation');
fs.mkdirSync(path.join(output, 'html'), {recursive:true});
const documents = [path.join(root, 'README.md')];
function collect(folder) {
  for (const item of fs.readdirSync(folder, {withFileTypes:true})) {
    if (['recomputed','confirmed','raw_export','package_support'].includes(item.name)) continue;
    const p=path.join(folder,item.name);
    if (item.isDirectory()) collect(p);
    else if (item.name.endsWith('.md')) documents.push(p);
  }
}
collect(path.join(root,'TRUST-CPF方案'));
const errors=[], records=[];
const css=`body{font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif;color:#1f2328;max-width:1080px;margin:0 auto;padding:38px;background:#fff}h1,h2{border-bottom:1px solid #d1d9e0;padding-bottom:.3em}h1{font-size:29px}h2{font-size:23px;margin-top:1.5em}h3{font-size:19px}a{color:#0969da;text-decoration:none}table{border-collapse:collapse;max-width:100%;margin:18px 0;font-size:14px}th,td{border:1px solid #d1d9e0;padding:7px 11px;text-align:left}tr:nth-child(even){background:#f6f8fa}pre{background:#f6f8fa;padding:16px;overflow:auto;line-height:1.5;border-radius:6px}code{font-family:Consolas,monospace;font-size:.88em}img{max-width:100%;height:auto}math[display=block]{margin:20px 0;overflow-x:auto;padding:8px 0}p{overflow-wrap:anywhere}`;
for (const file of [...new Set(documents)].sort()) {
  const original=fs.readFileSync(file,'utf8').replace(/\r\n/g,'\n');
  const rel=path.relative(root,file).replaceAll('\\','/');
  if ((original.match(/```/g)||[]).length%2) errors.push({file:rel,error:'Unclosed fence'});
  if (original.includes('\uFFFD')) errors.push({file:rel,error:'Replacement character'});
  if (/\]\(<?[A-Za-z]:[\\/]/.test(original)) errors.push({file:rel,error:'Absolute local link'});
  let tableCols=null, inFence=false;
  original.split(/\r?\n/).forEach((line,i)=>{
    if (line.startsWith('```')) {inFence=!inFence;tableCols=null;return;}
    if (inFence) return;
    if (line.startsWith('|')) {
      const n=line.split('|').length-2;
      if (tableCols===null) tableCols=n;
      else if (n!==tableCols) errors.push({file:rel,line:i+1,error:'Table column mismatch'});
    } else tableCols=null;
  });
  let linkCount=0, imageCount=0;
  for (const m of original.matchAll(/(!?)\[[^\]]*\]\((<[^>]+>|[^)\n]+)\)/g)) {
    const raw=decodeURIComponent(m[2].replace(/^<|>$/g,'')).split('#')[0];
    if (!raw || /^[a-z]+:\/\//i.test(raw)) continue;
    linkCount++;
    if (m[1]) imageCount++;
    const target=path.resolve(path.dirname(file),raw);
    if (!fs.existsSync(target)) errors.push({file:rel,error:'Broken link',target:raw});
  }
  let text=original, formulas=[];
  function formula(math,display) {
    const key='MATHPLACEHOLDER'+formulas.length+'END';
    try {
      const html=katex.renderToString(math,{displayMode:display,throwOnError:true,strict:'ignore',output:'mathml'});
      formulas.push({key,math,display,html});
    } catch (e) { errors.push({file:rel,error:e.message,math}); }
    return key;
  }
  text=text.replace(/```math\n([\s\S]*?)\n```/g,(_,math)=>'\n\n'+formula(math,true)+'\n\n');
  text=text.replace(/\$`([^\n]*?)`\$/g,(_,math)=>formula(math,false));
  let html=marked.parse(text);
  for (const f of formulas) html=html.replace(f.key,f.html);
  const htmlName=rel.replaceAll('/','__').replace(/\.md$/,'.html');
  const base='http://127.0.0.1:8767/'+rel.split('/').slice(0,-1).map(encodeURIComponent).join('/')+'/';
  fs.writeFileSync(path.join(output,'html',htmlName),`<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><base href="${base}"><title>${rel}</title><style>${css}</style></head><body><article>${html}</article></body></html>`,'utf8');
  records.push({file:rel,links:linkCount,images:imageCount,tables:(html.match(/<table>/g)||[]).length,
                displayFormulas:formulas.filter(x=>x.display).length,inlineFormulas:formulas.filter(x=>!x.display).length,
                reviewHtml:'html/'+htmlName});
}
const report=path.join(version,'analysis/server-results-20261022/README.md');
const source=path.join(version,'results/server-results-20261022/package_support/TRUST-CPF_v1_完整结果分析与下一轮实验方向_20261022.md.txt');
const a=fs.readFileSync(source,'utf8').replace(/\r\n/g,'\n'), b=fs.readFileSync(report,'utf8').replace(/\r\n/g,'\n');
const normalize=s=>s.replace(/\s+/g,'');
const oldDisplay=[...a.matchAll(/\$\$([\s\S]*?)\$\$/g)].map(m=>normalize(m[1]));
const newDisplay=[...b.matchAll(/```math\n([\s\S]*?)\n```/g)].map(m=>normalize(m[1]));
if (JSON.stringify(oldDisplay)!==JSON.stringify(newDisplay)) errors.push({error:'Report display formula tokens differ'});
const oldInline=[...a.replace(/\$\$[\s\S]*?\$\$/g,'').matchAll(/\$([^$\n]+)\$/g)].map(m=>normalize(m[1]));
const newInline=[...b.matchAll(/\$`(.*?)`\$/g)].map(m=>normalize(m[1]));
if (JSON.stringify(oldInline)!==JSON.stringify(newInline)) errors.push({error:'Report inline formula tokens differ'});
const result={passed:errors.length===0,documents:records,reportFormulaTokensPreserved:true,errors};
fs.writeFileSync(path.join(output,'markdown_checks.json'),JSON.stringify(result,null,2)+'\n','utf8');
console.log(JSON.stringify({passed:result.passed,documents:records.length,mathFormulas:records.reduce((n,r)=>n+r.displayFormulas+r.inlineFormulas,0),errors}));
if (errors.length) process.exitCode=1;
