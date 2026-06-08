import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ===== 6.1: 向量化回测 =====
np.random.seed(42)
n_stocks = 50
n_days = 252

# 模拟价格矩阵
prices = np.cumprod(1 + np.random.randn(n_days, n_stocks) * 0.015, axis=0) * 100

# 向量化计算收益率和信号
returns = np.diff(np.log(prices), axis=0)   # (251, 50)
signal = np.sign(returns - returns.mean(axis=1, keepdims=True))  # 均值偏离信号
# 延迟一天避免未来函数
signal_shifted = np.roll(signal, 1, axis=0)
signal_shifted[0] = 0
portfolio_returns = (signal_shifted * returns).mean(axis=1)  # 等权组合
cumret = np.cumprod(1 + portfolio_returns) - 1

import time
# 对比循环和向量化速度
def loop_backtest(prices):
    n_d, n_s = prices.shape
    returns_l = np.zeros((n_d-1, n_s))
    for d in range(1, n_d):
        for s in range(n_s):
            returns_l[d-1, s] = np.log(prices[d, s]) - np.log(prices[d-1, s])
    return returns_l

t0 = time.time()
_ = loop_backtest(prices[:30, :10])
t_loop = time.time() - t0

t0 = time.time()
for _ in range(100):
    _ = np.diff(np.log(prices[:30, :10]), axis=0)
t_vec = (time.time() - t0) / 100

speedup = t_loop / t_vec if t_vec > 0 else 9999

fig = plt.figure(figsize=(14, 5))
gs = GridSpec(1, 3, figure=fig)

# 左：累计收益
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(cumret * 100, color='#20808D', lw=2)
ax1.axhline(0, color='gray', lw=0.8, linestyle='--')
ax1.fill_between(range(len(cumret)), cumret*100, 0,
                 where=cumret>0, alpha=0.2, color='#20808D', label='盈利区间')
ax1.fill_between(range(len(cumret)), cumret*100, 0,
                 where=cumret<0, alpha=0.2, color='#A84B2F', label='亏损区间')
ax1.set_title(f"向量化回测累计收益\n({n_stocks}只股票×{n_days}天)", fontsize=10, fontweight='bold')
ax1.set_xlabel("交易日"); ax1.set_ylabel("累计收益率 (%)")
ax1.legend(fontsize=8); ax1.grid(True, linestyle='--', alpha=0.3)

# 中：收益热图
ax2 = fig.add_subplot(gs[0, 1])
im = ax2.imshow(returns[:50, :20].T, aspect='auto', cmap='RdYlGn', vmin=-0.05, vmax=0.05)
plt.colorbar(im, ax=ax2, label='日收益率')
ax2.set_title("收益率矩阵热图\n(前20只股票×50天)", fontsize=10, fontweight='bold')
ax2.set_xlabel("交易日"); ax2.set_ylabel("股票编号")

# 右：速度对比
ax3 = fig.add_subplot(gs[0, 2])
methods = ['For循环\n(30天×10股)', '向量化\n(30天×10股)']
times_ms = [t_loop*1000, t_vec*1000]
bar_colors = ['#A84B2F', '#20808D']
bars = ax3.bar(methods, times_ms, color=bar_colors, width=0.4, edgecolor='white')
for bar, v in zip(bars, times_ms):
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
             f'{v:.2f}ms', ha='center', fontsize=11, fontweight='bold')
ax3.set_title(f"速度对比 (~{speedup:.0f}x 加速)", fontsize=11, fontweight='bold')
ax3.set_ylabel("耗时 (ms)")
ax3.grid(True, axis='y', linestyle='--', alpha=0.3)

plt.suptitle("6.1 向量化回测 — 提速万倍重构实战", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("/home/user/workspace/figs/fig_6_1.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"fig_6_1.png saved  speedup={speedup:.0f}x")

# ===== 6.2: PCA 与 SVD =====
np.random.seed(0)
n_stocks = 100
n_days = 252

# 构造有共线性的因子收益矩阵
market = np.random.randn(n_days)
sector = np.random.randn(n_days)
# 每只股票 = 市场β + 行业γ + 个股噪声
betas_m = np.random.uniform(0.5, 1.5, n_stocks)
betas_s = np.random.uniform(0, 0.5, n_stocks)
returns_mat = (betas_m[None,:] * market[:,None] +
               betas_s[None,:] * sector[:,None] +
               np.random.randn(n_days, n_stocks) * 0.3)

# 标准化
from numpy.linalg import svd
X = returns_mat - returns_mat.mean(axis=0)
X /= (X.std(axis=0) + 1e-8)
U, S, Vt = svd(X, full_matrices=False)
var_ratio = S**2 / (S**2).sum()
cum_var = np.cumsum(var_ratio)

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# 左：奇异值 / 方差解释比例
ax = axes[0]
k_show = 15
ax.bar(range(1, k_show+1), var_ratio[:k_show]*100, color='#20808D', edgecolor='white')
ax.plot(range(1, k_show+1), cum_var[:k_show]*100, color='#A84B2F',
        marker='o', markersize=5, lw=2, label='累计方差')
ax2_t = ax.twinx()
ax2_t.set_visible(False)
ax.axhline(var_ratio[0]*100, color='gray', lw=0.7, linestyle='--')
ax.set_title(f"前{k_show}个主成分方差贡献\nPC1={var_ratio[0]*100:.1f}%", fontsize=10, fontweight='bold')
ax.set_xlabel("主成分编号"); ax.set_ylabel("方差贡献 (%)")
ax.legend(fontsize=9); ax.grid(True, axis='y', linestyle='--', alpha=0.3)

# 中：前2主成分散点图（股票在PC空间）
ax = axes[1]
scores = X @ Vt.T  # 投影到PC空间 (n_days, n_stocks... no, n_days x n_pc)
# 实际上 scores = U * S，股票在PC空间: Vt.T (n_stocks x n_pc)
stock_coords = Vt.T  # (n_stocks, n_pc)
sc = ax.scatter(stock_coords[:, 0], stock_coords[:, 1],
                c=betas_m, cmap='RdYlGn', s=30, alpha=0.8)
plt.colorbar(sc, ax=ax, label='市场Beta')
ax.set_title("100只股票在PC1-PC2空间\n颜色=市场Beta值", fontsize=10, fontweight='bold')
ax.set_xlabel("PC1 (市场因子)"); ax.set_ylabel("PC2 (行业因子)")
ax.grid(True, linestyle='--', alpha=0.3)

# 右：降维前后重构对比
ax = axes[2]
n_components = 3
X_reconstructed = (U[:, :n_components] * S[:n_components]) @ Vt[:n_components, :]
stock_idx = 0
ax.plot(X[:, stock_idx], color='gray', lw=1, alpha=0.7, label='原始收益率')
ax.plot(X_reconstructed[:, stock_idx], color='#20808D', lw=1.5, label=f'PCA-{n_components}重构')
ax.set_title(f"股票0: 原始 vs {n_components}个主成分重构", fontsize=10, fontweight='bold')
ax.set_xlabel("交易日"); ax.set_ylabel("标准化收益率")
ax.legend(fontsize=9); ax.grid(True, linestyle='--', alpha=0.3)

plt.suptitle("6.2 PCA与SVD分解 — 降维提取正交Alpha", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("/home/user/workspace/figs/fig_6_2.png", dpi=150, bbox_inches="tight")
plt.close()
print("fig_6_2.png saved")

# ===== 6.3: 线性回归正规方程 =====
np.random.seed(42)
n = 252  # 交易日数
# 真实参数：alpha=0.0005, beta_market=1.2, beta_size=-0.3
alpha_true = 0.0005
beta_m = 1.2
beta_s = -0.3

market_ret = np.random.randn(n) * 0.01
size_ret = np.random.randn(n) * 0.005
stock_ret = alpha_true + beta_m * market_ret + beta_s * size_ret + np.random.randn(n) * 0.008

# 设计矩阵 X: (n, 3)  [1, market, size]
X_des = np.column_stack([np.ones(n), market_ret, size_ret])
# 正规方程: beta = (X^T X)^{-1} X^T y
XtX = X_des.T @ X_des
Xty = X_des.T @ stock_ret
beta_hat = np.linalg.solve(XtX, Xty)
y_hat = X_des @ beta_hat
residuals = stock_ret - y_hat
ss_res = np.sum(residuals**2)
ss_tot = np.sum((stock_ret - stock_ret.mean())**2)
R2 = 1 - ss_res/ss_tot

print(f"估计参数: alpha={beta_hat[0]:.5f}(真实{alpha_true}), "
      f"beta_m={beta_hat[1]:.3f}(真实{beta_m}), beta_s={beta_hat[2]:.3f}(真实{beta_s})")
print(f"R² = {R2:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# 左：实际 vs 拟合
ax = axes[0]
ax.scatter(y_hat, stock_ret, alpha=0.4, s=15, color='#20808D')
mn, mx = min(y_hat.min(), stock_ret.min()), max(y_hat.max(), stock_ret.max())
ax.plot([mn, mx], [mn, mx], 'r--', lw=2, label='完美拟合线')
ax.set_title(f"拟合值 vs 真实值\nR²={R2:.4f}", fontsize=11, fontweight='bold')
ax.set_xlabel("拟合收益率 (y_hat)"); ax.set_ylabel("真实收益率 (y)")
ax.legend(fontsize=9); ax.grid(True, linestyle='--', alpha=0.3)

# 中：残差分布
ax = axes[1]
ax.hist(residuals, bins=30, color='#20808D', edgecolor='white', alpha=0.8)
ax.axvline(0, color='#A84B2F', lw=2, linestyle='--')
ax.set_title("残差分布 (应近似正态)", fontsize=11, fontweight='bold')
ax.set_xlabel("残差"); ax.set_ylabel("频次")
ax.grid(True, axis='y', linestyle='--', alpha=0.3)

# 右：参数估计对比
ax = axes[2]
params = ['alpha (×10^4)', 'beta_market', 'beta_size']
true_vals = [alpha_true*1e4, beta_m, beta_s]
est_vals = [beta_hat[0]*1e4, beta_hat[1], beta_hat[2]]
x_pos = np.arange(3)
w = 0.3
bars1 = ax.bar(x_pos - w/2, true_vals, w, label='真实值', color='#1B474D', edgecolor='white')
bars2 = ax.bar(x_pos + w/2, est_vals, w, label='估计值', color='#20808D', edgecolor='white')
ax.set_xticks(x_pos); ax.set_xticklabels(params, fontsize=9)
ax.set_title("正规方程参数估计 vs 真实值", fontsize=10, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, axis='y', linestyle='--', alpha=0.3)

plt.suptitle("6.3 线性回归正规方程 — 解析求Alpha/Beta", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("/home/user/workspace/figs/fig_6_3.png", dpi=150, bbox_inches="tight")
plt.close()
print("fig_6_3.png saved")
