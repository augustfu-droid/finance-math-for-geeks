const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto('file:///home/user/workspace/book.html', { waitUntil: 'networkidle', timeout: 120000 });
  // 等 KaTeX 渲染完成
  await page.waitForFunction(() => window.__ready === true, { timeout: 60000 }).catch(()=>{});
  await page.waitForTimeout(3000);
  await page.pdf({
    path: '/home/user/workspace/代码驱动的金融数学与AI风控实战.pdf',
    format: 'A4',
    printBackground: true,
    margin: { top: '16mm', bottom: '16mm', left: '14mm', right: '14mm' },
    displayHeaderFooter: true,
    headerTemplate: `<div style="font-size:7px;color:#9aa3b2;width:100%;padding:0 14mm;font-family:sans-serif;">
      <span style="float:left;">代码驱动的金融数学与AI风控实战</span></div>`,
    footerTemplate: `<div style="font-size:7px;color:#9aa3b2;width:100%;text-align:center;font-family:sans-serif;">
      <span class="pageNumber"></span> / <span class="totalPages"></span></div>`,
  });
  await browser.close();
  console.log('PDF done');
})();
