import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP"]
plt.rcParams["axes.unicode_minus"] = False

# ----------------------------------------------------------------
# 2.1节：二阶导数——凸凹判断、风险加速踩踏可视化
# ----------------------------------------------------------------

t = np.linspace(0, 10, 600)

# 牛市（指数增长，二阶导恒正）
price_bull = 100 * np.exp(0.06*t)

# 崩盘踩踏（先凸后加速下跌）
price_crash = np.where(t < 6,
                        100 + 5*t - 0.3*t**2,
                        85 - 8*(t-6) - 1.5*(t-6)**2)

d1_bull  = np.gradient(price_bull, t)
d2_bull  = np.gradient(d1_bull, t)
d1_crash = np.gradient(price_crash, t)
d2_crash = np.gradient(d1_crash, t)

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

def plot_price_deriv(ax_p, ax_d2, price, d1, d2, title, color):
    ax_p.plot(t, price, color=color, lw=2.2, label='价格 P(t)')
    ax_twin = ax_p.twinx()
    ax_twin.plot(t, d1, color='#FFC553', lw=1.8, ls='--', alpha=0.9, label="P'(t) 趋势")
    ax_twin.axhline(0, color='gray', lw=0.7, ls=':')
    ax_p.set_ylabel('价格', fontsize=11)
    ax_twin.set_ylabel("P'(t)", fontsize=11)
    ax_p.set_title(title, fontsize=12)
    l1, lb1 = ax_p.get_legend_handles_labels()
    l2, lb2 = ax_twin.get_legend_handles_labels()
    ax_p.legend(l1+l2, lb1+lb2, fontsize=9)
    ax_p.grid(True, alpha=0.3)

    ax_d2.fill_between(t, d2, 0, where=(d2 >= 0), color='#20808D', alpha=0.35,
                        label="P''(t)>0: 向上加速（凸）")
    ax_d2.fill_between(t, d2, 0, where=(d2 < 0),  color='#A84B2F', alpha=0.35,
                        label="P''(t)<0: 踩踏加速（凹）")
    ax_d2.plot(t, d2, color='#944454', lw=1.8)
    ax_d2.axhline(0, color='gray', lw=1, ls=':')
    ax_d2.set_xlabel('时间 t', fontsize=11)
    ax_d2.set_ylabel("P''(t) 加速度", fontsize=11)
    ax_d2.set_title("二阶导 P''(t)：凸凹与加速度", fontsize=12)
    ax_d2.legend(fontsize=9)
    ax_d2.grid(True, alpha=0.3)

plot_price_deriv(axes[0, 0], axes[1, 0], price_bull,  d1_bull,  d2_bull,
                 '牛市指数增长: P\'>0, 正加速', '#20808D')
plot_price_deriv(axes[0, 1], axes[1, 1], price_crash, d1_crash, d2_crash,
                 '崩盘踩踏: P\'\'由正转负, 下跌加速', '#A84B2F')

for ax in axes[0]:
    ax.set_xlabel('时间 t', fontsize=11)

plt.suptitle('二阶导数：价格的"加速度"——凸函数向上加速，凹段踩踏',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("figs/fig_2_1.png", dpi=150, bbox_inches="tight")
print("fig_2_1.png saved")
