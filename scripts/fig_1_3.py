import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP"]
plt.rcParams["axes.unicode_minus"] = False

# ----------------------------------------------------------------
# 1.3节：偏导数——双因子收益曲面 + 偏导方向切片可视化
# 模型：R(beta_mkt, beta_mom) = 市场因子暴露 + 动量因子暴露
# ----------------------------------------------------------------

# 构建双因子收益曲面
b_mkt = np.linspace(-1, 2, 80)    # 市场因子暴露 β_mkt
b_mom = np.linspace(-1, 1, 80)    # 动量因子暴露 β_mom
BM, BN = np.meshgrid(b_mkt, b_mom)

# 假设收益曲面：R = 0.08*β_mkt - 0.03*β_mkt² + 0.05*β_mom - 0.02*β_mom² + 0.01*β_mkt*β_mom
R = 0.08*BM - 0.03*BM**2 + 0.05*BN - 0.02*BN**2 + 0.01*BM*BN

fig = plt.figure(figsize=(14, 5))

# ---- 子图1：3D曲面 ----
ax1 = fig.add_subplot(131, projection='3d')
surf = ax1.plot_surface(BM, BN, R, cmap='viridis', alpha=0.85)
ax1.set_xlabel('$\\beta_{mkt}$', fontsize=10)
ax1.set_ylabel('$\\beta_{mom}$', fontsize=10)
ax1.set_zlabel('收益 $R$', fontsize=10)
ax1.set_title('双因子收益曲面\n$R(\\beta_{mkt},\\beta_{mom})$', fontsize=11)
fig.colorbar(surf, ax=ax1, shrink=0.5)

# ---- 子图2：固定β_mom，对β_mkt求偏导（沿mkt方向切片） ----
ax2 = fig.add_subplot(132)
b_mom_fixed = 0.0   # 固定动量因子
R_slice_mkt = 0.08*b_mkt - 0.03*b_mkt**2 + 0.05*b_mom_fixed - 0.02*b_mom_fixed**2 + 0.01*b_mkt*b_mom_fixed
dR_dbmkt    = 0.08 - 0.06*b_mkt + 0.01*b_mom_fixed   # 解析偏导

ax2.plot(b_mkt, R_slice_mkt, color='#20808D', lw=2.2, label=f'$R(\\beta_{{mkt}}, {b_mom_fixed})$')
ax2_twin = ax2.twinx()
ax2_twin.plot(b_mkt, dR_dbmkt, color='#A84B2F', lw=2, ls='--',
              label='$\\partial R/\\partial\\beta_{mkt}$')
ax2_twin.axhline(0, color='gray', lw=0.8, ls=':')
ax2.set_xlabel('$\\beta_{mkt}$', fontsize=12); ax2.set_ylabel('收益 $R$', fontsize=12)
ax2_twin.set_ylabel('偏导 $\\partial R/\\partial\\beta_{mkt}$', fontsize=11)
ax2.set_title(f'固定$\\beta_{{mom}}={b_mom_fixed}$\n对$\\beta_{{mkt}}$的偏导', fontsize=11)
lines1, lbs1 = ax2.get_legend_handles_labels()
lines2, lbs2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1+lines2, lbs1+lbs2, fontsize=8)
ax2.grid(True, alpha=0.3)

# ---- 子图3：固定β_mkt，对β_mom求偏导 ----
ax3 = fig.add_subplot(133)
b_mkt_fixed = 1.0
R_slice_mom = 0.08*b_mkt_fixed - 0.03*b_mkt_fixed**2 + 0.05*b_mom - 0.02*b_mom**2 + 0.01*b_mkt_fixed*b_mom
dR_dbmom    = 0.05 - 0.04*b_mom + 0.01*b_mkt_fixed

ax3.plot(b_mom, R_slice_mom, color='#20808D', lw=2.2, label=f'$R({b_mkt_fixed}, \\beta_{{mom}})$')
ax3_twin = ax3.twinx()
ax3_twin.plot(b_mom, dR_dbmom, color='#944454', lw=2, ls='--',
              label='$\\partial R/\\partial\\beta_{mom}$')
ax3_twin.axhline(0, color='gray', lw=0.8, ls=':')
ax3.set_xlabel('$\\beta_{mom}$', fontsize=12); ax3.set_ylabel('收益 $R$', fontsize=12)
ax3_twin.set_ylabel('偏导 $\\partial R/\\partial\\beta_{mom}$', fontsize=11)
ax3.set_title(f'固定$\\beta_{{mkt}}={b_mkt_fixed}$\n对$\\beta_{{mom}}$的偏导', fontsize=11)
lines1, lbs1 = ax3.get_legend_handles_labels()
lines2, lbs2 = ax3_twin.get_legend_handles_labels()
ax3.legend(lines1+lines2, lbs1+lbs2, fontsize=8)
ax3.grid(True, alpha=0.3)

plt.suptitle('偏导数：多因子收益曲面上的"单向斜率"',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("figs/fig_1_3.png", dpi=150, bbox_inches="tight")
print("fig_1_3.png saved")
