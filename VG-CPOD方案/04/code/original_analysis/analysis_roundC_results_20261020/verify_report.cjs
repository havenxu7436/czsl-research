const fs = require('fs');
const path = require('path');
const { pathToFileURL } = require('url');
const { chromium } = require('C:/Users/llenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

(async () => {
  const root = path.resolve(__dirname, '..');
  const name = 'VG-CPOD_v1_C组结果分析与下一轮方向_20261020';
  const md = fs.readFileSync(path.join(root, name + '.md'), 'utf8');
  const qa = path.join(__dirname, 'qa');
  fs.mkdirSync(qa, { recursive: true });
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1100 }, deviceScaleFactor: 1 });
    await page.goto(pathToFileURL(path.join(root, name + '.html')).href);
    await page.evaluate(() => document.fonts.ready);
    const result = await page.evaluate(() => ({
      displayMath: document.querySelectorAll('math[display="block"]').length,
      mathCount: document.querySelectorAll('math').length,
      unrenderedMath: document.querySelectorAll('span.math').length,
      tables: document.querySelectorAll('table').length,
      images: [...document.images].map(x => ({ complete: x.complete, width: x.naturalWidth, height: x.naturalHeight })),
      horizontalOverflow: document.documentElement.scrollWidth > innerWidth,
      wideTables: [...document.querySelectorAll('table')].filter(x => x.getBoundingClientRect().right > innerWidth).length,
      links: [...document.querySelectorAll('a[href]')].map(x => x.getAttribute('href'))
    }));
    result.expectedDisplayMath = (md.match(/\$\$/g) || []).length / 2;
    result.missingLinks = result.links.filter(x => !x.startsWith('#') && !x.startsWith('data:') &&
      !/^https?:/.test(x) && !fs.existsSync(path.resolve(root, decodeURIComponent(x))));
    delete result.links;
    await page.screenshot({ path: path.join(qa, '01_results.png') });
    await page.locator('h2').filter({ hasText: '3. C2' }).evaluate(x => x.scrollIntoView({ block: 'start' }));
    await page.screenshot({ path: path.join(qa, '02_gates.png') });
    await page.locator('h2').filter({ hasText: '4. 下一轮' }).evaluate(x => x.scrollIntoView({ block: 'start' }));
    await page.screenshot({ path: path.join(qa, '03_formula_plan.png') });
    fs.writeFileSync(path.join(qa, 'layout.json'), JSON.stringify(result, null, 2));
    console.log(JSON.stringify(result, null, 2));
    if (result.displayMath !== result.expectedDisplayMath || result.unrenderedMath || result.horizontalOverflow ||
        result.wideTables || result.missingLinks.length || result.images.some(x => !x.complete || !x.width)) {
      throw new Error('Report layout requires correction');
    }
  } finally {
    await browser.close();
  }
})().catch(e => { console.error(e); process.exitCode = 1; });
