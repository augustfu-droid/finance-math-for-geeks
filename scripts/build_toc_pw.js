// 用 Playwright 在打印分页布局下测量每个标题所在的 PDF 页码
const { chromium } = require('playwright');
const fs = require('fs');

(async () => {
  const items = JSON.parse(fs.readFileSync('/home/user/workspace/toc_items.json', 'utf-8'));
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto('file:///home/user/workspace/book.html', { waitUntil: 'networkidle', timeout: 180000 });
  await page.waitForFunction(() => window.__ready === true, { timeout: 90000 }).catch(()=>{});
  await page.emulateMedia({ media: 'print' });
  await page.waitForTimeout(2500);

  // A4 可打印内容高度：297mm - 上16 - 下16 = 265mm。1mm ≈ 3.7795px @96dpi
  const MM = 3.779527559;
  const pageContentPx = 265 * MM;

  // 取每个标题 id 的 top（相对文档），以及文档总高
  const result = await page.evaluate(({ ids }) => {
    const out = {};
    for (const id of ids) {
      const el = document.getElementById(id);
      if (el) {
        const r = el.getBoundingClientRect();
        out[id] = r.top + window.scrollY;
      }
    }
    out.__docHeight = document.documentElement.scrollHeight;
    return out;
  }, { ids: items.map(it => it[2]) });

  // 但 print 分页会在 page-break 处留空白，getBoundingClientRect 在连续流里测不到分页空白。
  // 采用更稳的方法：让 Chromium 真正打印，按 CSS @page 高度切分——这里用近似：
  // 文档以 .page 容器分段，封面/前言/导航/目录各占整页。我们直接用每个标题相对其
  // 所在 .page 容器顶部的偏移 + 该容器之前累计的"整页"数来推算。
  // 简化：直接用 top / pageContentPx 不可靠，故改为：在打印媒体下逐元素读取其
  // 'offsetTop' 不含分页空白。最稳妥仍是 PDF 文本法；此脚本仅作交叉校验输出原始 top。

  fs.writeFileSync('/home/user/workspace/toc_tops.json', JSON.stringify(result, null, 0));
  await browser.close();
  console.log('docHeight=', result.__docHeight, 'pageContentPx=', pageContentPx.toFixed(1));
})();
