import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

np.random.seed(2024)

# ---- 模拟10只股票的Beta与历史收益 ----
rf = 0.03        # 无风险利率
rm = 0.12        # 市场期望收益
betas = np.array([0.3, 0.6, 0.8, 1.0, 1.1, 1.2, 1.4, 1.6, 1.8, 2.1])
stock_names = [f"股{i+1}" for i in range(len(betas))]

# CAPM理论收益
E_r = rf + betas * (rm - rf)

# 加入Alpha噪声模拟真实市场
noise = np.array([0.03, -0.02, 0.01, 0.00, -0.015, 0.025, -0.01, 0.02, -0.03, 0.015])
actual_r = E_r + noise

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ---- 左图：证券市场线SML ----
ax = axes[0]
beta_line = np.linspace(0, 2.3, 200)
sml = rf + beta_line * (rm - rf)

ax.plot(beta_line, sml * 100, color="#20808D", linewidth=2, label="证券市场线 SML")
ax.scatter(betas, actual_r * 100, color="#A84B2F", s=60, zorder=5, label="各股票实际收益")
ax.scatter(betas, E_r * 100, color="#20808D", s=30, marker="x", zorder=6, label="CAPM理论收益")

for i, (b, r, name) in enumerate(zip(betas, actual_r, stock_names)):
    ax.annotate(name, (b + 0.02, r * 100 + 0.1), fontsize=8)
    # 画Alpha箭头
    if abs(noise[i]) > 0.005:
        ax.annotate("", xy=(b, r * 100), xytext=(b, E_r[i] * 100),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

ax.axhline(rf * 100, color="gray", linestyle=":", linewidth=0.8, label=f"无风险利率 $r_f$={rf*100:.0f}%")
ax.axvline(1.0, color="gray", linestyle=":", linewidth=0.8)
ax.set_xlabel("Beta $\\beta$（系统性风险）")
ax.set_ylabel("期望收益率 (%)")
ax.set_title("CAPM证券市场线（SML）\n$E[r] = r_f + \\beta(E[r_m] - r_f)$")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ---- 右图：Beta回归示意（某股票对市场指数）----
ax2 = axes[1]
n_months = 60
market_monthly = np.random.normal((rm - rf) / 12, 0.04, n_months)  # 市场超额月收益
# 构造一只Beta=1.3的股票
beta_true = 1.3
alpha_true = 0.002
stock_monthly = alpha_true + beta_true * market_monthly + np.random.normal(0, 0.025, n_months)

slope, intercept, r_val, p_val, se = stats.linregress(market_monthly, stock_monthly)

ax2.scatter(market_monthly * 100, stock_monthly * 100,
            color="#20808D", alpha=0.5, s=20, label="月度超额收益散点")
x_fit = np.linspace(market_monthly.min(), market_monthly.max(), 100)
ax2.plot(x_fit * 100, (intercept + slope * x_fit) * 100,
         color="#A84B2F", linewidth=2,
         label=f"OLS回归 β̂={slope:.2f}, α̂={intercept*100:.2f}%")
ax2.set_xlabel("市场超额收益 $r_m - r_f$ (%)")
ax2.set_ylabel("股票超额收益 $r_i - r_f$ (%)")
ax2.set_title(f"Beta系数的OLS估计\n$R^2={r_val**2:.3f}$（n=60个月）")
ax2.axhline(0, color="gray", linewidth=0.5)
ax2.axvline(0, color="gray", linewidth=0.5)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/home/user/workspace/figs/fig_13_2.png", dpi=150, bbox_inches="tight")
print("fig_13_2.png 保存完成")
