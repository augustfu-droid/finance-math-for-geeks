import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP"]
plt.rcParams["axes.unicode_minus"] = False

# ----------------------------------------------------------------
# 1.2节：导数——割线→切线，股价动能可视化
# ----------------------------------------------------------------

np.random.seed(7)
t = np.linspace(0, 10, 800)
# 模拟有趋势的股价曲线：logistic上涨 + 小噪声
price = 100 + 20 * (1/(1+np.exp(-1.5*(t-5)))) + np.cumsum(np.random.randn(800))*0.08
price = np.clip(price, 90, 130)

t0_idx = 350
t0, p0 = t[t0_idx], price[t0_idx]

# 中心差分作为"真实"导数
h_approx = t[1] - t[0]
dp_dt = np.gradient(price, t)   # 全程导数（瞬时变化率/速度）

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ---- 左图：割线→切线动画截图 ----
ax1 = axes[0]
ax1.plot(t, price, color='#1B474D', lw=1.8, label='股价 $P(t)$', zorder=1)
ax1.plot(t0, p0, 'ko', ms=9, zorder=5, label=f'切点 $t_0={t0:.2f}$')

x_range = np.linspace(t0 - 2.0, t0 + 2.0, 200)
h_list   = [2.0, 1.0, 0.5, 0.2]
colors   = ['#BCE2E7', '#6E522B', '#944454', '#FFC553']
for h, col in zip(h_list, colors):
    idx2   = np.argmin(np.abs(t - (t0 + h)))
    slope  = (price[idx2] - p0) / (t[idx2] - t0)
    ax1.plot(x_range, p0 + slope*(x_range-t0), color=col, lw=1.6,
             ls='--', alpha=0.9, label=f'h={h}, 斜率={slope:.2f}')

true_slope = dp_dt[t0_idx]
ax1.plot(x_range, p0 + true_slope*(x_range-t0), color='#A84B2F', lw=2.5,
         label=f'切线(h→0), 斜率={true_slope:.2f}')
ax1.set_xlim(t0-2.2, t0+2.2); ax1.set_ylim(p0-6, p0+7)
ax1.set_xlabel('时间 $t$', fontsize=12); ax1.set_ylabel('价格 $P(t)$', fontsize=12)
ax1.set_title('割线斜率 $\\to$ 切线斜率（瞬时导数）', fontsize=12)
ax1.legend(fontsize=8, loc='upper left'); ax1.grid(True, alpha=0.3)

# ---- 右图：全程导数（价格动能/趋势强度） ----
ax2 = axes[1]
ax2_twin = ax2.twinx()
ax2.plot(t, price, color='#1B474D', lw=1.5, alpha=0.7, label='股价 $P(t)$')
ax2_twin.plot(t, dp_dt, color='#A84B2F', lw=1.5, alpha=0.9,
              label="导数 $P'(t)$（动能）")
ax2_twin.axhline(0, color='gray', lw=0.8, ls=':')
ax2_twin.fill_between(t, dp_dt, 0,
                       where=(dp_dt > 0), alpha=0.15, color='#20808D', label='上涨动能>0')
ax2_twin.fill_between(t, dp_dt, 0,
                       where=(dp_dt < 0), alpha=0.15, color='#A84B2F', label='下跌动能<0')
ax2.set_xlabel('时间 $t$', fontsize=12); ax2.set_ylabel('价格', fontsize=12)
ax2_twin.set_ylabel("导数 $P'(t)$", fontsize=12)
ax2.set_title("全程导数：价格动能与趋势方向", fontsize=12)

lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1+lines2, labels1+labels2, fontsize=9, loc='upper left')
ax2.grid(True, alpha=0.3)

plt.suptitle("导数 = 瞬时变化率：从割线到切线，从静态到动能",
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("figs/fig_1_2.png", dpi=150, bbox_inches="tight")
print("fig_1_2.png saved")
print(f"t0={t0:.2f} 处导数（瞬时斜率）= {true_slope:.4f}")
