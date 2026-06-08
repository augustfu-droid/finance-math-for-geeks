import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP"]
plt.rcParams["axes.unicode_minus"] = False

# ----------------------------------------------------------------
# 1.1节：股票分期复利 → 连续复利极限可视化
# 思想：(1 + r/n)^n 随 n→∞ 收敛到 e^r
# ----------------------------------------------------------------

r = 0.10
n_values = np.array([1, 2, 4, 12, 52, 250, 1000, 10000, 100000])
compound  = (1 + r / n_values) ** n_values   # 各频率终值
continuous = np.exp(r)                        # 连续复利极限

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ---- 左图：收敛曲线 ----
ax1 = axes[0]
ax1.semilogx(n_values, compound, 'o-', color='#20808D', lw=2.2,
             markersize=7, label='离散复利 $(1+r/n)^n$', zorder=3)
ax1.axhline(continuous, color='#A84B2F', lw=2, ls='--',
            label=f'连续复利极限 $e^r={continuous:.6f}$')
labels_map = {1:'年付\nn=1', 12:'月付\nn=12', 250:'日付\nn=250', 100000:'n=10⁵'}
for nk, lbl in labels_map.items():
    idx = np.where(n_values == nk)[0][0]
    ax1.annotate(lbl, xy=(n_values[idx], compound[idx]),
                 xytext=(n_values[idx]*0.25, compound[idx]-0.0012),
                 fontsize=8.5, color='#1B474D',
                 arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))
ax1.set_xlabel('复利期数 $n$（对数刻度）', fontsize=12)
ax1.set_ylabel('终值（本金=1）', fontsize=12)
ax1.set_title('分期复利收敛至连续复利极限\n$r=10\\%$', fontsize=12)
ax1.legend(fontsize=10); ax1.grid(True, alpha=0.3)

# ---- 右图：误差衰减 ----
ax2 = axes[1]
error = np.abs(compound - continuous)
ax2.loglog(n_values, error, 's-', color='#944454', lw=2.2,
           markersize=7, label='$|(1+r/n)^n - e^r|$')
# O(1/n) 参考线
ref_n = np.array([1, 100000])
ax2.loglog(ref_n, continuous * r**2 / (2*ref_n), 'k--',
           lw=1.3, alpha=0.6, label='$O(1/n)$ 参考线')
for nk in [1, 12, 1000]:
    idx = np.where(n_values == nk)[0][0]
    ax2.annotate(f'n={nk}\n{error[idx]:.2e}',
                 xy=(n_values[idx], error[idx]),
                 xytext=(n_values[idx]*2, error[idx]*2),
                 fontsize=8, arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))
ax2.set_xlabel('期数 $n$（对数刻度）', fontsize=12)
ax2.set_ylabel('绝对误差（对数刻度）', fontsize=12)
ax2.set_title('误差以 $O(1/n)$ 衰减趋向零\n$\\varepsilon$-$\\delta$ 极限的数值直觉', fontsize=12)
ax2.legend(fontsize=10); ax2.grid(True, alpha=0.3, which='both')

plt.suptitle('极限直觉：$n\\to\\infty$ 时离散复利收敛至连续复利 $e^r$',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("figs/fig_1_1.png", dpi=150, bbox_inches="tight")
print("fig_1_1.png saved")
print(f"e^0.1 = {continuous:.8f}")
for n, c, e in zip(n_values[[0,3,5,8]], compound[[0,3,5,8]], error[[0,3,5,8]]):
    print(f"  n={n:>6d}: 终值={c:.8f}, 误差={e:.2e}")
