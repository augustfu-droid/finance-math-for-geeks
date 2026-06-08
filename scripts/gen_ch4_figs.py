import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ===== 4.1: 标量、向量与张量 =====
fig, axes = plt.subplots(1, 3, figsize=(13, 4))

# 标量：单个数值
ax = axes[0]
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(-0.5, 1.5)
ax.scatter([0.5], [0.5], s=300, color="#20808D", zorder=5)
ax.text(0.5, 0.7, "标量 x = 3.14", ha='center', fontsize=13, fontweight='bold', color="#20808D")
ax.text(0.5, 0.3, "零维：一个数", ha='center', fontsize=11, color="#7A7974")
ax.set_title("标量 (Scalar)", fontsize=13, fontweight='bold')
ax.axis('off')

# 向量：1D 箭头
ax = axes[1]
origin = [0, 0]
v = [3, 2]
ax.annotate("", xy=(v[0], v[1]), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='#A84B2F', lw=2.5))
ax.scatter(*origin, s=60, color='black', zorder=5)
ax.text(v[0]+0.1, v[1]+0.1, r"$\vec{v}=(3,2)$", fontsize=13, color='#A84B2F')
ax.set_xlim(-0.5, 4)
ax.set_ylim(-0.5, 3.5)
ax.axhline(0, color='gray', lw=0.8)
ax.axvline(0, color='gray', lw=0.8)
ax.grid(True, linestyle='--', alpha=0.4)
ax.set_title("向量 (Vector)", fontsize=13, fontweight='bold')
ax.set_xlabel("维度1")
ax.set_ylabel("维度2")

# 张量：3D cube 示意
ax = axes[2]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
colors = ["#BCE2E7","#20808D","#1B474D"]
labels_t = ["股票", "特征", "时间"]
for i, (c, lbl) in enumerate(zip(colors, labels_t)):
    rect = plt.Rectangle((1+i, 1+i), 5, 5, linewidth=1.5,
                          edgecolor='#1B474D', facecolor=c, alpha=0.55)
    ax.add_patch(rect)
    ax.text(1+i+0.3, 1+i+0.3, lbl, fontsize=10, color='#28251D')
ax.set_title("张量 Tensor (3D)", fontsize=13, fontweight='bold')
ax.text(5, 0.3, "形状 (100, 50, 252)", ha='center', fontsize=10, color='#7A7974')
ax.axis('off')

fig.suptitle("4.1 标量·向量·张量直觉示意", fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("/home/user/workspace/figs/fig_4_1.png", dpi=150, bbox_inches="tight")
plt.close()
print("fig_4_1.png saved")

# ===== 4.2: 向量点乘与余弦相似度 =====
np.random.seed(42)
# 模拟两只股票30日收益率
days = 30
r_A = np.random.randn(days) * 0.01 + 0.0003
r_B = r_A * 0.6 + np.random.randn(days) * 0.005
r_C = np.random.randn(days) * 0.01 - 0.0002

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

sim_AB = cosine_sim(r_A, r_B)
sim_AC = cosine_sim(r_A, r_C)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# 左：折线图
ax = axes[0]
ax.plot(r_A, color='#20808D', label=f'股票A', lw=1.8)
ax.plot(r_B, color='#A84B2F', label=f'股票B (sim={sim_AB:.3f})', lw=1.8, linestyle='--')
ax.plot(r_C, color='#944454', label=f'股票C (sim={sim_AC:.3f})', lw=1.8, linestyle=':')
ax.axhline(0, color='gray', lw=0.7)
ax.set_title("30日收益率走势", fontsize=12, fontweight='bold')
ax.set_xlabel("交易日")
ax.set_ylabel("日收益率")
ax.legend(fontsize=9)
ax.grid(True, linestyle='--', alpha=0.3)

# 右：余弦相似度条形图
ax = axes[1]
bars = ax.bar(['A vs B', 'A vs C'], [sim_AB, sim_AC],
              color=['#20808D', '#A84B2F'], width=0.4, edgecolor='white')
for bar, val in zip(bars, [sim_AB, sim_AC]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{val:.4f}", ha='center', fontsize=12, fontweight='bold')
ax.set_ylim(-0.1, 1.1)
ax.axhline(0, color='gray', lw=0.7)
ax.set_title("余弦相似度对比", fontsize=12, fontweight='bold')
ax.set_ylabel("cosine similarity")
ax.grid(True, axis='y', linestyle='--', alpha=0.3)

plt.suptitle("4.2 向量点乘与余弦相似度", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("/home/user/workspace/figs/fig_4_2.png", dpi=150, bbox_inches="tight")
plt.close()
print("fig_4_2.png saved")

# ===== 4.3: 矩阵加减与标量乘法 =====
np.random.seed(7)
dates = np.arange(20)
stocks = ['AAPL', 'GOOG', 'MSFT', 'AMZN']
price_A = np.random.randn(20, 4).cumsum(axis=0) + 100
price_B = np.random.randn(20, 4) * 0.5  # 调整矩阵
result = price_A + price_B  # 矩阵加法

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
colors_list = ['#20808D','#A84B2F','#1B474D','#944454']

for i, ax_t in enumerate(axes):
    for j in range(4):
        if i == 0:
            ax_t.plot(dates, price_A[:, j], color=colors_list[j], label=stocks[j], lw=1.5)
            ax_t.set_title("矩阵 A（原始价格）", fontsize=11, fontweight='bold')
        elif i == 1:
            ax_t.plot(dates, price_B[:, j], color=colors_list[j], label=stocks[j], lw=1.5)
            ax_t.set_title("矩阵 B（调整量）", fontsize=11, fontweight='bold')
        else:
            ax_t.plot(dates, result[:, j], color=colors_list[j], label=stocks[j], lw=1.5)
            ax_t.set_title("A + B（调整后价格）", fontsize=11, fontweight='bold')
    ax_t.legend(fontsize=8, loc='upper left')
    ax_t.set_xlabel("日期")
    ax_t.set_ylabel("价格")
    ax_t.grid(True, linestyle='--', alpha=0.3)

plt.suptitle("4.3 矩阵加减（DataFrame批量操作底层）", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("/home/user/workspace/figs/fig_4_3.png", dpi=150, bbox_inches="tight")
plt.close()
print("fig_4_3.png saved")

# ===== 4.4: 矩阵乘法 =====
np.random.seed(42)
n_stocks = 200
n_factors = 5
n_days = 60

# 因子矩阵 F: (n_days, n_factors), 因子暴露 W: (n_factors, n_stocks)
F = np.random.randn(n_days, n_factors)
W = np.random.randn(n_factors, n_stocks) * 0.3

# 矩阵乘法: portfolio returns (n_days, n_stocks)
returns = F @ W

import time
# 对比for循环速度
def loop_matmul(F, W):
    n_d, n_f = F.shape
    n_s = W.shape[1]
    R = np.zeros((n_d, n_s))
    for i in range(n_d):
        for k in range(n_f):
            for j in range(n_s):
                R[i, j] += F[i, k] * W[k, j]
    return R

# 只对小矩阵计时
F_small = F[:10, :]
W_small = W[:, :20]

t0 = time.time()
_ = loop_matmul(F_small, W_small)
t_loop = time.time() - t0

t0 = time.time()
for _ in range(100):
    __ = F_small @ W_small
t_matmul = (time.time() - t0) / 100

speedup = t_loop / t_matmul if t_matmul > 0 else 9999

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# 热图：因子收益矩阵
ax = axes[0]
im = ax.imshow(returns[:30, :50], aspect='auto', cmap='RdYlGn', vmin=-0.5, vmax=0.5)
plt.colorbar(im, ax=ax, label='收益率')
ax.set_title(f"矩阵乘法结果\n形状 F({n_days},{n_factors}) @ W({n_factors},{n_stocks})", fontsize=11, fontweight='bold')
ax.set_xlabel("股票 (前50支)")
ax.set_ylabel("交易日 (前30天)")

# 速度对比
ax = axes[1]
methods = ['三重For循环\n(10×20 小矩阵)', 'NumPy矩阵乘法\n(10×20 小矩阵)']
times_ms = [t_loop * 1000, t_matmul * 1000]
bar_colors = ['#A84B2F', '#20808D']
bars = ax.bar(methods, times_ms, color=bar_colors, width=0.4, edgecolor='white')
for bar, val in zip(bars, times_ms):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f"{val:.2f} ms", ha='center', fontsize=11, fontweight='bold')
ax.set_title(f"速度对比（约 {speedup:.0f}x 加速）", fontsize=12, fontweight='bold')
ax.set_ylabel("耗时 (ms)")
ax.grid(True, axis='y', linestyle='--', alpha=0.3)

plt.suptitle("4.4 矩阵乘法 — 干掉千万次For循环", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("/home/user/workspace/figs/fig_4_4.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"fig_4_4.png saved  (speedup ~{speedup:.0f}x)")
