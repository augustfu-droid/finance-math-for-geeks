import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

np.random.seed(7)

# ---- 6只资产（虚构沪深股票）的年化收益/波动率/相关矩阵 ----
mu_assets = np.array([0.12, 0.09, 0.15, 0.07, 0.18, 0.10])  # 年化期望收益
sigma_assets = np.array([0.22, 0.15, 0.30, 0.10, 0.35, 0.18])  # 年化波动率

# 构造相关矩阵（正定对称）
corr = np.array([
    [1.00, 0.35, 0.55, 0.10, 0.40, 0.30],
    [0.35, 1.00, 0.25, 0.20, 0.20, 0.45],
    [0.55, 0.25, 1.00, 0.05, 0.60, 0.25],
    [0.10, 0.20, 0.05, 1.00, 0.10, 0.15],
    [0.40, 0.20, 0.60, 0.10, 1.00, 0.20],
    [0.30, 0.45, 0.25, 0.15, 0.20, 1.00],
])
# 协方差矩阵
cov = np.outer(sigma_assets, sigma_assets) * corr
n = len(mu_assets)

# ---- 蒙特卡洛模拟随机组合 ----
n_random = 5000
rand_weights = np.random.dirichlet(np.ones(n), n_random)
rand_ret = rand_weights @ mu_assets
rand_vol = np.sqrt(np.einsum('ij,jk,ik->i', rand_weights, cov, rand_weights))
rand_sharpe = (rand_ret - 0.03) / rand_vol

# ---- 数值求解有效前沿 ----
target_rets = np.linspace(mu_assets.min(), mu_assets.max(), 60)
eff_vols = []
eff_weights_list = []

for target in target_rets:
    def portfolio_vol(w):
        return np.sqrt(w @ cov @ w)
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'eq', 'fun': lambda w, t=target: w @ mu_assets - t}
    ]
    bounds = [(0, 1)] * n
    w0 = np.ones(n) / n
    res = minimize(portfolio_vol, w0, method='SLSQP',
                   bounds=bounds, constraints=constraints,
                   options={'ftol': 1e-9, 'maxiter': 1000})
    if res.success:
        eff_vols.append(res.fun)
        eff_weights_list.append(res.x)
    else:
        eff_vols.append(np.nan)

eff_vols = np.array(eff_vols)
valid = ~np.isnan(eff_vols)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 左图：有效前沿
ax = axes[0]
sc = ax.scatter(rand_vol * 100, rand_ret * 100, c=rand_sharpe, cmap="RdYlGn",
                alpha=0.4, s=5, label="随机组合")
ax.plot(eff_vols[valid] * 100, target_rets[valid] * 100,
        color="#20808D", linewidth=2.5, label="有效前沿")

# 标出各单一资产
labels = ["沪深A", "消费B", "科技C", "债券D", "创业E", "混合F"]
for i, (lbl, mu_i, sig_i) in enumerate(zip(labels, mu_assets, sigma_assets)):
    ax.scatter(sig_i * 100, mu_i * 100, s=60, zorder=5, color="#A84B2F")
    ax.annotate(lbl, (sig_i * 100 + 0.3, mu_i * 100), fontsize=8)

cb = plt.colorbar(sc, ax=ax)
cb.set_label("夏普比率")
ax.set_xlabel("年化波动率 (%)")
ax.set_ylabel("年化期望收益 (%)")
ax.set_title("均值-方差有效前沿\n（6只虚构资产，无卖空约束）")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 右图：最优组合（最大夏普）的权重饼图
rf = 0.03
sharpe_eff = (target_rets[valid] - rf) / eff_vols[valid]
best_idx = np.argmax(sharpe_eff)
best_weights = eff_weights_list[np.where(valid)[0][best_idx]]

ax2 = axes[1]
colors_pie = ["#20808D", "#A84B2F", "#1B474D", "#944454", "#FFC553", "#6E522B"]
wedges, texts, autotexts = ax2.pie(
    best_weights, labels=labels, autopct='%1.1f%%',
    colors=colors_pie, startangle=90,
    textprops={'fontsize': 9}
)
ax2.set_title(f"最大夏普比组合权重分配\n夏普比={sharpe_eff[best_idx]:.2f}，收益={target_rets[valid][best_idx]*100:.1f}%，波动={eff_vols[valid][best_idx]*100:.1f}%")

plt.tight_layout()
plt.savefig("/home/user/workspace/figs/fig_13_1.png", dpi=150, bbox_inches="tight")
print("fig_13_1.png 保存完成")
