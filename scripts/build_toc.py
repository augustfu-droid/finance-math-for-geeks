# -*- coding: utf-8 -*-
"""第二轮：从已渲染的 PDF 里定位每个 篇/章/节 标题的页码，写出 toc_pages.json。"""
import json, re, sys, unicodedata, pathlib, logging
logging.disable(logging.WARNING)
import pdfplumber

WS = pathlib.Path("/home/user/workspace")
PDF = WS / "代码驱动的金融数学与AI风控实战.pdf"
items = json.loads((WS / "toc_items.json").read_text(encoding="utf-8"))

# 归一化：NFKC 折叠兼容字符（如 ⼀→一、全角→半角），仅保留 CJK 表意字 + 数字 + 拉丁字母
def core(s):
    s = unicodedata.normalize("NFKC", s)
    return "".join(ch for ch in s if ('\u4e00' <= ch <= '\u9fff') or ch.isdigit() or ('a' <= ch.lower() <= 'z'))

print("读取 PDF 页面文字 ...", file=sys.stderr)
page_core = []
with pdfplumber.open(str(PDF)) as pdf:
    total = len(pdf.pages)
    for pg in pdf.pages:
        page_core.append(core(pg.extract_text() or ""))
print(f"PDF 共 {total} 页")

# 识别目录页（含多个篇标题或以"目录"开头）
toc_idx = set()
for i, t in enumerate(page_core):
    if t[:4].startswith("目录") or (t.count("第一篇") + t.count("第二篇") + t.count("第三篇") >= 2):
        toc_idx.add(i)
body_start = (max(toc_idx) + 1) if toc_idx else 0
print("目录页索引:", sorted(toc_idx), "正文起始页索引:", body_start)

pages = {}
last = body_start  # 标题大体顺序出现，从上一个命中页起找
miss = []
for lvl, txt, hid in items:
    key = core(txt)
    probe = key[:12]  # 短探针，规避换行/断字
    found = None
    # 先从 last 起顺序找
    for i in range(last, total):
        if i in toc_idx:
            continue
        if probe and probe in page_core[i]:
            found = i; break
    # 退而求其次：全正文范围
    if found is None:
        for i in range(body_start, total):
            if i in toc_idx:
                continue
            if probe and probe in page_core[i]:
                found = i; break
    if found is not None:
        pages[hid] = found + 1
        last = found
    else:
        miss.append(txt)

# ---- 回退：对字体提取失败的标题，用该节正文首个长中文片段在邻居区间定位 ----
if miss:
    md_all = "\n\n".join((WS/f"part{i}.md").read_text(encoding="utf-8") for i in range(1,6))
    # 建立 hid -> (idx) 映射以取邻居
    hid_idx = {it[2]: k for k, it in enumerate(items)}
    for lvl, txt, hid in items:
        if hid in pages:
            continue
        # 邻居页区间
        k = hid_idx[hid]
        prev_pp = next((pages[items[j][2]] for j in range(k-1,-1,-1) if items[j][2] in pages), body_start+1)
        next_pp = next((pages[items[j][2]] for j in range(k+1,len(items)) if items[j][2] in pages), total)
        lo, hi = prev_pp-1, next_pp
        # 在 markdown 里找该标题后的首个长中文片段
        m = re.search(re.escape(txt[:14]), md_all) or re.search(re.escape(txt.split("（")[0][:12]), md_all)
        placed = None
        if m:
            seg = md_all[m.end():m.end()+1500]
            for c in re.findall(r"[\u4e00-\u9fff]{12,}", seg)[:4]:
                pc = core(c)[:16]
                for i in range(max(lo-1,body_start), min(hi, total)):
                    if i in toc_idx: continue
                    if pc and pc in page_core[i]:
                        placed = i+1; break
                if placed: break
        pages[hid] = placed if placed else prev_pp  # 实在不行则用上一标题页

(WS / "toc_pages.json").write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")
still = [it[1] for it in items if it[2] not in pages]
print(f"定位 {len(pages)}/{len(items)} 个标题")
if still:
    print("仍未定位:", still)
