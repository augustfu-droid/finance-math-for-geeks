# -*- coding: utf-8 -*-
"""汇编四篇 + 前置内容 -> 单一 HTML（KaTeX 公式 + Pygments 代码高亮 + 精美排版）"""
import re, base64, pathlib, markdown

WS = pathlib.Path("/home/user/workspace")
ASSETS = WS / "assets"

def b64_font(p):
    return base64.b64encode((ASSETS/"fonts"/p).read_bytes()).decode()

# ---- 读取前置 + 四篇 ----
front = (WS/"frontmatter.md").read_text(encoding="utf-8")
parts = [ (WS/f"part{i}.md").read_text(encoding="utf-8") for i in range(1,6) ]

# 拆出 cover / preface / nav
def extract(tag, text):
    m = re.search(rf"<!-- {tag} -->(.*?)<!-- /{tag} -->", text, re.S)
    return m.group(1).strip() if m else ""
cover_md   = extract("COVER", front)
preface_md = extract("PREFACE", front)
nav_md     = extract("NAV", front)

# ---- 把图片转成内嵌 base64，保证 PDF 自包含 ----
def inline_images(md_text):
    def repl(m):
        path = WS / m.group(1)
        if path.exists():
            data = base64.b64encode(path.read_bytes()).decode()
            return f'<img src="data:image/png;base64,{data}" />'
        return m.group(0)
    return re.sub(r'!\[\]\((figs/[^)]+)\)', repl, md_text)

body_md = "\n\n".join(inline_images(p) for p in parts)

md = markdown.Markdown(extensions=[
    "extra", "tables", "fenced_code", "sane_lists",
    "pymdownx.highlight", "pymdownx.superfences",
], extension_configs={
    "pymdownx.highlight": {"noclasses": True, "pygments_style": "friendly", "linenums": False},
})

def render(text):
    md.reset()
    return md.convert(text)

cover_html   = render(cover_md)
preface_html = render(preface_md)
nav_html     = render(nav_md)
body_html    = render(body_md)

# ---- 模块卡片化：把每个 <h4> 模块连同其内容包进配色卡片（借鉴参考PDF的彩色直觉框）----
from bs4 import BeautifulSoup
def cardify(html):
    soup = BeautifulSoup(html, "html.parser")
    # 模块标题关键字 -> 卡片CSS类
    rules = [
        ("\U0001F4A1", "mod-intuition"),   # 💡 大白话 -> 绿色直觉框
        ("\U0001F680", "mod-biz"),         # 🚀 业务映射 -> 青蓝应用框
        ("\U0001F6E1", "mod-qa"),          # 🛡️ QA灾难 -> 暖橙警示框
    ]
    # 收集顶层节点（body的直接子节点序列）
    nodes = list(soup.contents)
    new_root = soup.new_tag("div")
    i = 0
    n = len(nodes)
    # 我们直接遍历所有元素，遇到目标 h4 就把它和后续兄弟（直到下一个 h3/h4/h2/h1）打包
    # 由于 markdown 输出是扁平结构，使用 find_all 顺序处理
    h4s = soup.find_all("h4")
    for h4 in h4s:
        txt = h4.get_text()
        cls = None
        for kw, c in rules:
            if kw in txt:
                cls = c; break
        if not cls:
            continue
        # 收集 h4 之后、直到下一个标题(h1-h4)的所有兄弟节点
        wrapper = soup.new_tag("div")
        wrapper["class"] = f"modcard {cls}"
        siblings = []
        sib = h4.next_sibling
        while sib is not None:
            nxt = sib.next_sibling
            name = getattr(sib, "name", None)
            if name in ("h1","h2","h3","h4"):
                break
            siblings.append(sib)
            sib = nxt
        # 插入 wrapper 到 h4 之前，把 h4 与 siblings 移入
        h4.insert_before(wrapper)
        wrapper.append(h4.extract())
        for s in siblings:
            wrapper.append(s.extract())
    # 额外处理：章首"本章数学直觉导引"blockquote -> 加上 chapter-intuition 类(绿色大卡)
    for bq in soup.find_all("blockquote"):
        if "\U0001F4A1" in bq.get_text() and ("本章数学直觉" in bq.get_text() or "本章直觉" in bq.get_text()):
            existing = bq.get("class", [])
            bq["class"] = existing + ["chapter-intuition"]
    return str(soup)

body_html = cardify(body_html)

# ---- 给 h1/h2/h3 加稳定 ID，并抽取目录条目 ----
import json
toc_soup = BeautifulSoup(body_html, "html.parser")
toc_items = []  # (level, text, id)
_seen = {}
def _slug(s, lvl):
    key = f"h{lvl}-" + re.sub(r"[^\w\u4e00-\u9fff]+", "-", s)[:40]
    _seen[key] = _seen.get(key, 0) + 1
    return key if _seen[key] == 1 else f"{key}-{_seen[key]}"
for tag in toc_soup.find_all(["h1", "h2", "h3"]):
    # 跳过章首“本章数学直觉导引”以及其他 blockquote 内部的标题
    if tag.find_parent("blockquote") is not None:
        continue
    lvl = int(tag.name[1])
    txt = tag.get_text().strip()
    hid = _slug(txt, lvl)
    tag["id"] = hid
    toc_items.append((lvl, txt, hid))
body_html = str(toc_soup)

# ---- 读取第一轮渲染后的页码映射（若存在）----
toc_pages = {}
_tp = WS / "toc_pages.json"
if _tp.exists():
    toc_pages = json.loads(_tp.read_text(encoding="utf-8"))

# ---- 生成目录 HTML ----
def build_toc_html(items, pages):
    rows = []
    for lvl, txt, hid in items:
        cls = {1: "toc-part", 2: "toc-chap", 3: "toc-sec"}[lvl]
        pno = pages.get(hid, "")
        rows.append(
            f'<div class="toc-row {cls}">'
            f'<a class="toc-title" href="#{hid}">{txt}</a>'
            f'<span class="toc-dots"></span>'
            f'<span class="toc-p">{pno}</span>'
            f'</div>'
        )
    return '<h1 class="toc-h">目&nbsp;&nbsp;录</h1>\n' + "\n".join(rows)
toc_html = build_toc_html(toc_items, toc_pages)

# 保存目录条目供 build_toc.py 使用
if not toc_pages:
    (WS/"toc_items.json").write_text(json.dumps(toc_items, ensure_ascii=False), encoding="utf-8")

css_fonts = f"""
@font-face {{ font-family:'KaTeX_Main'; src:url(data:font/woff2;base64,{b64_font('KaTeX_Main-Regular.woff2')}) format('woff2'); font-weight:normal; font-style:normal; }}
@font-face {{ font-family:'KaTeX_Main'; src:url(data:font/woff2;base64,{b64_font('KaTeX_Main-Bold.woff2')}) format('woff2'); font-weight:bold; }}
@font-face {{ font-family:'KaTeX_Main'; src:url(data:font/woff2;base64,{b64_font('KaTeX_Main-Italic.woff2')}) format('woff2'); font-style:italic; }}
@font-face {{ font-family:'KaTeX_Math'; src:url(data:font/woff2;base64,{b64_font('KaTeX_Math-Italic.woff2')}) format('woff2'); font-style:italic; }}
@font-face {{ font-family:'KaTeX_Math'; src:url(data:font/woff2;base64,{b64_font('KaTeX_Math-BoldItalic.woff2')}) format('woff2'); font-weight:bold; font-style:italic; }}
@font-face {{ font-family:'KaTeX_Size1'; src:url(data:font/woff2;base64,{b64_font('KaTeX_Size1-Regular.woff2')}) format('woff2'); }}
@font-face {{ font-family:'KaTeX_Size2'; src:url(data:font/woff2;base64,{b64_font('KaTeX_Size2-Regular.woff2')}) format('woff2'); }}
@font-face {{ font-family:'KaTeX_Size3'; src:url(data:font/woff2;base64,{b64_font('KaTeX_Size3-Regular.woff2')}) format('woff2'); }}
@font-face {{ font-family:'KaTeX_Size4'; src:url(data:font/woff2;base64,{b64_font('KaTeX_Size4-Regular.woff2')}) format('woff2'); }}
@font-face {{ font-family:'KaTeX_AMS'; src:url(data:font/woff2;base64,{b64_font('KaTeX_AMS-Regular.woff2')}) format('woff2'); }}
@font-face {{ font-family:'KaTeX_Caligraphic'; src:url(data:font/woff2;base64,{b64_font('KaTeX_Caligraphic-Regular.woff2')}) format('woff2'); }}
@font-face {{ font-family:'KaTeX_Fraktur'; src:url(data:font/woff2;base64,{b64_font('KaTeX_Fraktur-Regular.woff2')}) format('woff2'); }}
@font-face {{ font-family:'KaTeX_SansSerif'; src:url(data:font/woff2;base64,{b64_font('KaTeX_SansSerif-Regular.woff2')}) format('woff2'); }}
@font-face {{ font-family:'KaTeX_Script'; src:url(data:font/woff2;base64,{b64_font('KaTeX_Script-Regular.woff2')}) format('woff2'); }}
@font-face {{ font-family:'KaTeX_Typewriter'; src:url(data:font/woff2;base64,{b64_font('KaTeX_Typewriter-Regular.woff2')}) format('woff2'); }}
"""

katex_css = (ASSETS/"katex.min.css").read_text(encoding="utf-8")
katex_js  = (ASSETS/"katex.min.js").read_text(encoding="utf-8")
autorender_js = (ASSETS/"auto-render.min.js").read_text(encoding="utf-8")

html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>代码驱动的金融数学与AI风控实战</title>
<style>{katex_css}</style>
<style>{css_fonts}</style>
<style>
:root {{
  --ink:#1f2430; --muted:#5b6472; --faint:#9aa3b2;
  --accent:#1b474d; --accent2:#20808d; --rule:#dfe3ea;
  --code-bg:#f6f8fa; --quote-bg:#f3f7f7; --warn:#964219;
}}
* {{ box-sizing:border-box; }}
body {{
  font-family:'Noto Sans CJK SC','Source Han Sans SC',sans-serif;
  color:var(--ink); line-height:1.75; font-size:10.5pt; margin:0;
}}
.page {{ padding:0 4mm; }}
/* ===== 封面 ===== */
.cover {{
  page-break-after:always; height:247mm; position:relative;
  padding:0; display:flex; flex-direction:column; justify-content:center;
}}
.cover::before {{
  content:""; position:absolute; top:-40mm; right:-50mm; width:140mm; height:140mm;
  border:1.5px solid var(--rule); border-radius:50%;
}}
.cover::after {{
  content:""; position:absolute; bottom:10mm; left:-30mm; width:80mm; height:80mm;
  border:1.5px solid var(--rule); transform:rotate(20deg);
}}
.cover .bar {{ width:60px; height:5px; background:var(--accent); margin-bottom:50mm; }}
.cover h1, .cover h2, .cover h3 {{ page-break-before:avoid!important; }}
.cover h1 {{ font-size:34pt; font-weight:800; line-height:1.2; color:var(--ink); border:none; margin:0 0 6mm; padding:0; }}
.cover h2 {{ font-size:15pt; font-weight:500; color:var(--accent); border:none; margin:0 0 18mm; padding:0; }}
.cover p {{ color:var(--muted); margin:2mm 0; font-size:10.5pt; }}
.cover .kw {{ color:var(--accent2); font-weight:600; }}
.cover .sub {{ color:var(--faint); font-size:9.5pt; }}
.cover .date {{ margin-top:14mm; color:var(--muted); }}
/* ===== 标题层级 ===== */
h1 {{ font-size:21pt; font-weight:800; color:var(--ink); border-bottom:2.5px solid var(--accent); padding-bottom:3mm; margin:0 0 7mm; page-break-before:always; }}
h2 {{ font-size:15pt; font-weight:700; color:var(--accent); margin:9mm 0 4mm; padding-left:4mm; border-left:5px solid var(--accent2); }}
h3 {{ font-size:13pt; font-weight:700; color:var(--ink); margin:8mm 0 3mm; padding-bottom:1.5mm; border-bottom:1px dashed var(--rule); }}
h4 {{ font-size:11.5pt; font-weight:700; color:var(--accent); margin:6mm 0 2mm; }}
p {{ margin:2.5mm 0; text-align:justify; }}
strong {{ color:var(--ink); font-weight:700; }}
/* ===== 前言/导航不强制分页 from h1 ===== */
.preface h1, .nav h1 {{ page-break-before:always; }}
/* ===== 代码 ===== */
pre {{ background:var(--code-bg)!important; border:1px solid var(--rule); border-left:4px solid var(--accent2);
  border-radius:6px; padding:4mm 5mm!important; font-size:8.7pt; line-height:1.55; overflow-x:hidden;
  white-space:pre-wrap; word-break:break-word; page-break-inside:avoid; margin:3mm 0; }}
pre, code {{ font-family:'JetBrains Mono','DejaVu Sans Mono',monospace; }}
:not(pre)>code {{ background:var(--code-bg); padding:1px 5px; border-radius:4px; font-size:9pt; color:#9a3a52; }}
/* ===== 引用块（练一练）===== */
blockquote {{ background:var(--quote-bg); border-left:4px solid var(--accent2);
  margin:3mm 0; padding:2.5mm 5mm; border-radius:0 6px 6px 0; color:#2a3340; page-break-inside:avoid; }}
blockquote p {{ margin:1.5mm 0; font-size:10pt; }}
/* ===== 表格 ===== */
table {{ border-collapse:collapse; width:100%; margin:4mm 0; font-size:9.5pt; page-break-inside:avoid; }}
th {{ background:var(--accent); color:#fff; padding:2.5mm 3mm; text-align:left; font-weight:600; }}
td {{ padding:2.2mm 3mm; border-bottom:1px solid var(--rule); }}
tr:nth-child(even) td {{ background:#fafbfc; }}
/* ===== 图片 ===== */
img {{ max-width:100%; height:auto; display:block; margin:4mm auto; border:1px solid var(--rule); border-radius:4px; page-break-inside:avoid; }}
/* ===== KaTeX 独立公式居中 ===== */
.katex-display {{ margin:3mm 0; overflow-x:hidden; }}
.katex {{ font-size:1.02em; }}
ul,ol {{ margin:2mm 0 2mm 6mm; padding:0; }}
li {{ margin:1.2mm 0; }}
hr {{ border:none; border-top:1px solid var(--rule); margin:5mm 0; }}
/* ===== 章首数学直觉导引卡（绿色大卡）===== */
blockquote.chapter-intuition {{ background:#eef7f0; border-left:5px solid #3f8f5b;
  margin:4mm 0 6mm; padding:3.5mm 6mm 3mm; border-radius:0 8px 8px 0; page-break-inside:avoid; }}
blockquote.chapter-intuition h3 {{ color:#2f6b44; font-size:12.5pt; margin:0 0 2mm; page-break-before:avoid; border:none; padding:0; }}
blockquote.chapter-intuition p {{ font-size:10pt; color:#2a3340; }}
blockquote.chapter-intuition strong {{ color:#2f6b44; }}
blockquote.chapter-intuition li {{ font-size:10pt; }}
/* ===== 模块彩色卡片（借鉴参考PDF）===== */
.modcard {{ margin:4mm 0; padding:3mm 5mm 3mm; border-radius:8px; border-left-width:5px; border-left-style:solid; page-break-inside:auto; }}
.modcard h4 {{ margin-top:1mm; margin-bottom:2mm; }}
.modcard p {{ margin:2mm 0; }}
.modcard pre {{ margin:2.5mm 0; }}
.modcard blockquote {{ background:rgba(255,255,255,0.55); }}
/* 💡 大白话直觉框：绿 */
.mod-intuition {{ background:#eef7f0; border-left-color:#3f8f5b; }}
.mod-intuition h4 {{ color:#2f6b44; }}
/* 🚀 业务映射框：青蓝 */
.mod-biz {{ background:#eaf4f6; border-left-color:#20808d; }}
.mod-biz h4 {{ color:#1b474d; }}
/* 🛡️ QA灾难警示框：暖橙 */
.mod-qa {{ background:#fbf1ea; border-left-color:#c4601f; }}
.mod-qa h4 {{ color:#9a4715; }}
/* ===== 目录页 ===== */
.toc-page {{ page-break-before:always; }}
.toc-h {{ font-size:24pt; color:var(--accent); text-align:center; margin:0 0 8mm; padding-bottom:4mm; border-bottom:2px solid var(--accent2); letter-spacing:2px; page-break-before:avoid!important; }}
.toc-row {{ display:flex; align-items:baseline; margin:1.2mm 0; line-height:1.5; }}
.toc-title {{ color:#28251d; text-decoration:none; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.toc-dots {{ flex:1 1 auto; border-bottom:1px dotted #b9bcc2; margin:0 2mm; position:relative; top:-2px; min-width:6mm; }}
.toc-p {{ color:#5a5957; font-variant-numeric:tabular-nums; flex:0 0 auto; }}
.toc-part {{ margin-top:4mm; }}
.toc-part .toc-title {{ font-size:12.5pt; font-weight:700; color:var(--accent); }}
.toc-part .toc-p {{ font-weight:700; color:var(--accent); }}
.toc-chap {{ margin-left:5mm; }}
.toc-chap .toc-title {{ font-size:10.5pt; font-weight:600; color:#28251d; }}
.toc-sec {{ margin-left:11mm; }}
.toc-sec .toc-title {{ font-size:9.5pt; color:#3a3a3a; }}
.toc-sec .toc-p {{ font-size:9.5pt; }}
</style>
</head><body>

<div class="cover">
  <div class="bar"></div>
  {cover_html}
</div>

<div class="page preface">{preface_html}</div>
<div class="page nav">{nav_html}</div>
<div class="page toc-page">{toc_html}</div>
<div class="page">{body_html}</div>

<script>{katex_js}</script>
<script>{autorender_js}</script>
<script>
renderMathInElement(document.body, {{
  delimiters:[
    {{left:"$$",right:"$$",display:true}},
    {{left:"$",right:"$",display:false}},
    {{left:"\\\\[",right:"\\\\]",display:true}},
    {{left:"\\\\(",right:"\\\\)",display:false}}
  ],
  throwOnError:false
}});
document.title="代码驱动的金融数学与AI风控实战";
window.__ready=true;
</script>
</body></html>"""

out = WS/"book.html"
out.write_text(html, encoding="utf-8")
print("HTML written:", out, len(html), "chars")
