import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

# ---- 债券定价基础函数 ----
def bond_price(ytm, coupon, par, n_periods, freq=2):
    """
    ytm: 年化到期收益率
    coupon: 年化票面利率
    par: 面值
    n_periods: 剩余付息次数
    freq: 每年付息频率
    """
    c = coupon * par / freq   # 每期票息
    r = ytm / freq            # 每期折现率
    periods = np.arange(1, n_periods + 1)
    pv_coupons = np.sum(c / (1 + r) ** periods)
    pv_par = par / (1 + r) ** n_periods
    return pv_coupons + pv_par

def duration_modified(ytm, coupon, par, n_periods, freq=2):
    """麦考利久期 / (1+ytm/freq) = 修正久期"""
    c = coupon * par / freq
    r = ytm / freq
    price = bond_price(ytm, coupon, par, n_periods, freq)
    periods = np.arange(1, n_periods + 1)
    # 麦考利久期（年）
    mac_dur = np.sum(periods * c / (1 + r)**periods) + n_periods * par / (1 + r)**n_periods
    mac_dur = mac_dur / price / freq
    mod_dur = mac_dur / (1 + ytm / freq)
    return mac_dur, mod_dur

def convexity(ytm, coupon, par, n_periods, freq=2):
    c = coupon * par / freq
    r = ytm / freq
    price = bond_price(ytm, coupon, par, n_periods, freq)
    periods = np.arange(1, n_periods + 1)
    conv = np.sum(periods * (periods + 1) * c / (1 + r)**(periods + 2))
    conv += n_periods * (n_periods + 1) * par / (1 + r)**(n_periods + 2)
    conv = conv / price / freq**2
    return conv

# ---- 参数 ----
coupon = 0.05    # 票面利率5%
par = 1000.0
n_periods = 20   # 10年期，半年付息，共20期
ytm0 = 0.05      # 初始YTM=5%（平价债券）

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# ---- 左图：债券价格 vs YTM ----
ytm_range = np.linspace(0.01, 0.12, 300)
prices = [bond_price(y, coupon, par, n_periods) for y in ytm_range]

ax = axes[0]
ax.plot(ytm_range * 100, prices, color="#20808D", linewidth=2.5, label="债券价格")
ax.axvline(ytm0 * 100, color="gray", linestyle="--", linewidth=0.8, label=f"初始YTM={ytm0*100:.0f}%")
ax.axhline(par, color="#A84B2F", linestyle=":", linewidth=0.8, label=f"面值={par}")
ax.set_xlabel("到期收益率 YTM (%)")
ax.set_ylabel("债券价格（元）")
ax.set_title("债券价格 vs YTM\n价格-收益率的凸性关系")
ax.legend()
ax.grid(True, alpha=0.3)

# ---- 中图：久期 vs 到期时间 ----
years_list = [2, 5, 7, 10, 15, 20, 25, 30]
n_periods_list = [y * 2 for y in years_list]
mac_durs, mod_durs = zip(*[duration_modified(ytm0, coupon, par, n) for n in n_periods_list])

ax2 = axes[1]
ax2.plot(years_list, mac_durs, color="#20808D", marker="o", linewidth=2, label="麦考利久期")
ax2.plot(years_list, mod_durs, color="#A84B2F", marker="s", linewidth=2, label="修正久期")
ax2.set_xlabel("债券剩余期限（年）")
ax2.set_ylabel("久期（年）")
ax2.set_title("久期 vs 期限\n票面利率=YTM=5%（平价债券）")
ax2.legend()
ax2.grid(True, alpha=0.3)

# ---- 右图：久期近似 vs 精确价格变动（凸性修正） ----
ax3 = axes[2]
ytm_center = 0.05
mac_d, mod_d = duration_modified(ytm_center, coupon, par, n_periods)
conv = convexity(ytm_center, coupon, par, n_periods)
P0 = bond_price(ytm_center, coupon, par, n_periods)

delta_ytms = np.linspace(-0.04, 0.04, 200)
# 精确价格
exact_prices = [bond_price(ytm_center + dy, coupon, par, n_periods) for dy in delta_ytms]
exact_change = [(p - P0) / P0 * 100 for p in exact_prices]

# 一阶（久期）近似
dur_approx = [-mod_d * dy * 100 for dy in delta_ytms]

# 二阶（凸性修正）
conv_approx = [(-mod_d * dy + 0.5 * conv * dy**2) * 100 for dy in delta_ytms]

ax3.plot(delta_ytms * 100, exact_change, color="#20808D", linewidth=2.5, label="精确价格变动")
ax3.plot(delta_ytms * 100, dur_approx, color="#A84B2F", linewidth=1.8, linestyle="--", label="久期线性近似")
ax3.plot(delta_ytms * 100, conv_approx, color="#FFC553", linewidth=1.8, linestyle="-.", label="久期+凸性修正")
ax3.set_xlabel("YTM变动 Δytm (%)")
ax3.set_ylabel("价格变动 (%)")
ax3.set_title("久期近似 vs 精确价格变动\n大幅利率波动时线性近似失效")
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.axhline(0, color="gray", linewidth=0.5)
ax3.axvline(0, color="gray", linewidth=0.5)

plt.tight_layout()
plt.savefig("/home/user/workspace/figs/fig_13_3.png", dpi=150, bbox_inches="tight")
print("fig_13_3.png 保存完成")
