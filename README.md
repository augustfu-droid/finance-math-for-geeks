[![YouTube English Channel](https://img.shields.io/badge/YouTube-InkStar%20Quant%20(EN)-red?style=for-the-badge&logo=youtube)](https://www.youtube.com/@InkStarQuant)
[![YouTube Chinese Channel](https://img.shields.io/badge/YouTube-%E5%A2%A8%E7%82%B9%E6%88%9F%20%7C%20%E5%A4%A7%E9%98%9F%E9%95%BF%20(CN)-C8483B?style=for-the-badge&logo=youtube)](https://www.youtube.com/@InkIntoStarsCN)
[![Ink into Stars ML 108](https://img.shields.io/badge/Open%20Book-%E5%A2%A8%E7%82%B9%E6%88%9F%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0108%E8%AE%B2-E3A857?style=for-the-badge&logo=github)](https://github.com/augustfu-droid/ink-into-stars-ml-108)

> 🎬 **作者官方视频频道（4K 手算推导与深度投研拆表）**：[YouTube 中文频道 (@InkIntoStarsCN)](https://www.youtube.com/@InkIntoStarsCN) ｜ [YouTube English (@InkStarQuant)](https://www.youtube.com/@InkStarQuant) ｜ 📖 [《墨点成星：机器学习108讲》全集开源书稿](https://github.com/augustfu-droid/ink-into-stars-ml-108)

---

# 代码驱动的金融数学与 AI 风控实战

> 副标题：从变化率到资产定价 · 给极客的金融数学自学手册

一本面向**有工程底子、想吃透模型本质的极客**的金融数学自学教材。不是从公理出发的数学课本，也不是 `import` 一把梭的调包教程，而是一本**「连接手册 + 排雷手册」**：每一个数学概念，都用固定的六模块结构讲透，并配可运行的 Python 代码与配图。

📕 **成品 PDF**：[`代码驱动的金融数学与AI风控实战.pdf`](./代码驱动的金融数学与AI风控实战.pdf)（235 页）

---

## 这本书讲什么

全书共 **5 篇 / 13 章 / 44 节**，覆盖代码驱动金融数学与 AI 风控所需的五大数学支柱：

| 篇章 | 核心数学 | 典型金融/AI 应用 |
| --- | --- | --- |
| 第一篇 微积分与算法寻优 | 极限、导数、偏导、链式法则、泰勒展开、梯度 | 梯度下降、反向传播、期权希腊字母、XGBoost 二阶寻优 |
| 第二篇 线性代数 | 向量、矩阵乘法、逆矩阵、秩、特征值、SVD | 向量化回测、PCA 降维、正交 Alpha 因子、正规方程 |
| 第三篇 概率论与统计学 | 均值方差、正态分布、贝叶斯、偏度峰度、MLE、熵 | 违约概率 PD、逻辑回归、信息增益、VaR/CVaR |
| 第四篇 随机过程与时序分析 | 随机游走、平稳性、协整、隐马尔可夫、蒙特卡洛 | 均值回归、配对交易、牛熊状态识别、衍生品定价 |
| 第五篇 随机分析与资产定价 | 布朗运动、伊藤引理、随机微分方程、二次规划 | Black-Scholes 期权定价、有效前沿、CAPM、债券久期凸性 |

## 六模块结构（每一小节）

每一节都遵循统一的六模块结构，即学即练：

1. **💡 大白话唤醒直觉** — 先用生活常识或股票走势把本质讲明白，绝不一上来扔定理
2. **📐 极简公式推导** — 只给核心精简公式，步骤写全，用 LaTeX 排版
3. **💻 Python 跑直觉** — 一段可独立运行、带详细中文注释的代码，画图或打印数值，亲手验证原理
4. **🚀 金融/AI 业务映射** — 两句话点透这个数学概念在衍生品定价、量化多因子风控或机器学习中是什么「零部件」
5. **🛡️ 系统级 QA 灾难与防坑指南** — 复盘典型的生产级 Bug（脱敏虚构），并给出工程级断言拦截方案
6. **📝 练一练** — 每个模块后紧跟小题与参考答案要点，做到边学边练

> 所有 QA 灾难案例均为基于公开工程知识的虚构脱敏演绎，不指向任何真实机构或事件，仅用于传授防坑思维。

---

## 仓库结构

```
.
├── 代码驱动的金融数学与AI风控实战.pdf   # 成品（235 页，可直接阅读）
├── book/                # 书稿源文件（Markdown）
│   ├── frontmatter.md   # 封面 / 前言 / 全书导航
│   ├── part1.md         # 第一篇 微积分与算法寻优
│   ├── part2.md         # 第二篇 线性代数
│   ├── part3.md         # 第三篇 概率论与统计学
│   ├── part4.md         # 第四篇 随机过程与时序分析
│   ├── part5.md         # 第五篇 随机分析与资产定价
│   └── SPEC.md          # 写作规范（六模块结构定义）
├── figures/             # 44 张配图（matplotlib 生成的 PNG）
├── scripts/             # 构建与配图脚本
│   ├── build_html.py    # 汇编 frontmatter + 5 篇 → book.html（内联图片、彩色卡片、目录 ID）
│   ├── build_toc.py     # 两遍渲染的第二遍：定位每个标题页码，生成目录页码
│   ├── to_pdf.js        # Playwright 把 book.html 打印成 PDF（A4 / 页眉页脚）
│   ├── fig_*.py         # 各章配图生成脚本
│   └── regen_part5_figs.py  # 第五篇配图重生成（CJK 字体修正）
└── assets/              # KaTeX 离线资源（公式渲染）
```

---

## 本地构建 PDF

环境依赖：Python 3（`matplotlib`、`numpy`、`scipy`、`beautifulsoup4`、`pygments`、`pdfplumber`、`pypdf`）、Node.js（`playwright`）、以及系统 CJK 字体（`Noto Sans CJK`）。

```bash
# 1) 生成配图（如需重跑）
python3 scripts/fig_1_1.py   # 等等；第五篇用 regen_part5_figs.py

# 2) 两遍渲染（第一遍定分页，第二遍填目录页码）
python3 scripts/build_html.py      # frontmatter + 5 篇 → book.html
node    scripts/to_pdf.js          # book.html → PDF
python3 scripts/build_toc.py       # 读 PDF 定位标题页码 → toc_pages.json
python3 scripts/build_html.py      # 重建（带目录页码）
node    scripts/to_pdf.js          # 最终 PDF
```

> 配图中文字体说明：matplotlib 把系统的 `NotoSansCJK-*.ttc` 注册为 `Noto Sans CJK JP`（该字体集本身含简体字形），因此绘图脚本统一用 `Noto Sans CJK JP` 渲染简体中文，避免「豆腐块」乱码。

---

## 适合谁读

- **有工程底子、想理解模型本质的极客**：熟悉 Python 与软件工程，想补上数学这一环
- **从开发/测试转量化或风控的人**：有扎实的系统与质量直觉，缺的是金融数学工具
- **AI 背景想进金融的人**：懂深度学习，但不知道这些数学在金融里具体是哪个零部件

---

## License

本作品采用 [CC BY-NC-SA 4.0](./LICENSE) 许可（署名—非商业性使用—相同方式共享）。
