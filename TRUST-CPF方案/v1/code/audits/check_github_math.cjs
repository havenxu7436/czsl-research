const fs=require('fs'),path=require('path'),katex=require(process.env.KATEX_MODULE||'katex');
const root=process.argv[2],records={},errors=[];
for(const name of ['TRUST-CPF_CLIP_修订实验方案v1.md','TRUST-CPF公式及代码v1.md']){
 const formulas=JSON.parse(fs.readFileSync(path.join(root,name+'.math.json'),'utf8'));
 for(const [mode,maths] of Object.entries(formulas))for(const math of maths){
  try{katex.renderToString(math,{displayMode:mode==='display',throwOnError:true,strict:'ignore'});}
  catch(e){errors.push({name,math,error:e.message});}
 }
 records[name]={display:formulas.display.length,inline:formulas.inline.length};
}
const result={passed:errors.length===0,renderer:'KaTeX 0.16.22',records,errors,
 github_delimiters:'fenced math and dollar-backtick inline',operatorname_macros:0};
fs.writeFileSync(path.join(root,'github_math.json'),JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify(result));if(errors.length)process.exitCode=1;
