import json, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Force the only-registered CJK face; the .ttc carries SC glyphs so 简体中文 renders fine.
CJK = "Noto Sans CJK JP"
matplotlib.rcParams["font.sans-serif"] = [CJK]
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["axes.unicode_minus"] = False

blocks = json.loads(open("_part5_plot_blocks.json", encoding="utf-8").read())

for i, code in enumerate(blocks):
    m = re.search(r'savefig\("(figs/fig_1[23]_[123]\.png)"', code)
    target = m.group(1) if m else f"block_{i}"
    # Neutralize the block's own broken font line so our global setting wins.
    code = re.sub(r'plt\.rcParams\[\s*["\']font\.sans-serif["\']\s*\].*', '', code)
    # Re-assert font right before any plotting in each block's namespace.
    ns = {"plt": plt, "matplotlib": matplotlib, "fm": fm}
    plt.rcParams["font.sans-serif"] = [CJK]
    plt.rcParams["axes.unicode_minus"] = False
    try:
        exec(code, ns)
        print(f"[OK] regenerated {target}")
    except Exception as e:
        print(f"[FAIL] {target}: {type(e).__name__}: {e}")
    finally:
        plt.close("all")
