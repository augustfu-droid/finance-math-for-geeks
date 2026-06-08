import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP"]
plt.rcParams["axes.unicode_minus"] = False

# ----------------------------------------------------------------
# 2.3节：梯度与方向导数——损失曲面 + 梯度箭头 + 方向导数分解
# ----------------------------------------------------------------

# 二维损失曲面：L(w1,w2) = (w1-1)^2 + 2*(w2+0.5)^2 + 0.5*w1*w2
w1 = np.linspace(-2, 3, 200)
w2 = np.linspace(-2.5, 2, 200)
W1, W2 = np.meshgrid(w1, w2)
L = (W1 - 1)**2 + 2*(W2 + 0.5)**2 + 0.5*W1*W2

# 梯度
dL_dw1 = 2*(W1 - 1) + 0.5*W2   # ∂L/∂w1
dL_dw2 = 4*(W2 + 0.5) + 0.5*W1  # ∂L/∂w2

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# ---- 左图：等高线 + 梯度箭头 ----
ax1 = axes[0]
cs = ax1.contourf(W1, W2, L, levels=30, cmap='viridis', alpha=0.8)
plt.colorbar(cs, ax=ax1, label='损失 $L$')
# 稀疏化箭头（每10步取一个）
step = 15
ax1.quiver(W1[::step, ::step], W2[::step, ::step],
           -dL_dw1[::step, ::step], -dL_dw2[::step, ::step],
           color='white', alpha=0.85, scale=25, width=0.003)
# 标记最小值点（近似解析解）
# 令梯度=0: 2(w1-1)+0.5w2=0, 4(w2+0.5)+0.5w1=0 → 联立求解
# 2w1 + 0.5w2 = 2  → 4w1 + w2 = 4
# 0.5w1 + 4w2 = -2
# det: 4*4 - 1*0.5 = 15.5
w1_min = (4*4 - 1*(-2)) / (4*4 - 1*0.5)
w2_min = (-2 - 0.5*w1_min) / 4
ax1.plot(w1_min, w2_min, 'r*', ms=14, label=f'极小值点\n$({w1_min:.2f},{w2_min:.2f})$')
ax1.set_xlabel('$w_1$', fontsize=12); ax1.set_ylabel('$w_2$', fontsize=12)
ax1.set_title('损失曲面等高线\n白色箭头=负梯度方向（最速下降）', fontsize=11)
ax1.legend(fontsize=9)

# ---- 中图：梯度模长分布 ----
ax2 = axes[1]
grad_norm = np.sqrt(dL_dw1**2 + dL_dw2**2)
cs2 = ax2.contourf(W1, W2, grad_norm, levels=30, cmap='plasma', alpha=0.85)
plt.colorbar(cs2, ax=ax2, label='梯度模长 $\\|\\nabla L\\|$')
ax2.plot(w1_min, w2_min, 'w*', ms=14, label='极小值（梯度=0）')
ax2.set_xlabel('$w_1$', fontsize=12); ax2.set_ylabel('$w_2$', fontsize=12)
ax2.set_title('梯度模长：越亮下降越快\n极小值处梯度模=0', fontsize=11)
ax2.legend(fontsize=9)

# ---- 右图：某点处方向导数随方向角变化 ----
ax3 = axes[2]
pt_w1, pt_w2 = 0.0, 0.5   # 选一个点
g1 = 2*(pt_w1 - 1) + 0.5*pt_w2
g2 = 4*(pt_w2 + 0.5) + 0.5*pt_w1
grad_vec = np.array([g1, g2])
grad_norm_pt = np.linalg.norm(grad_vec)

theta = np.linspace(0, 2*np.pi, 360)
# 方向向量 u = [cos θ, sin θ]
dir_deriv = g1*np.cos(theta) + g2*np.sin(theta)  # ∇L · u

ax3.plot(np.degrees(theta), dir_deriv, color='#20808D', lw=2.2,
         label='方向导数 $D_u L$')
ax3.axhline(0, color='gray', lw=0.8, ls=':')
# 标注梯度方向（最大方向导数）
theta_max = np.degrees(np.arctan2(g2, g1))
ax3.axvline(theta_max % 360, color='#A84B2F', lw=1.8, ls='--',
            label=f'梯度方向 $\\theta={theta_max%360:.0f}°$\n（方向导数最大）')
ax3.axvline((theta_max+180) % 360, color='#944454', lw=1.5, ls='--',
            label=f'负梯度方向 {(theta_max+180)%360:.0f}°\n（下降最快）')
ax3.set_xlabel('方向角 $\\theta$（度）', fontsize=12)
ax3.set_ylabel('方向导数 $D_u L$', fontsize=12)
ax3.set_title(f'点$({pt_w1},{pt_w2})$处\n各方向上的变化率', fontsize=11)
ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)
ax3.text(0.05, 0.05, f'$\\nabla L=({g1:.2f},{g2:.2f})$\n$\\|\\nabla L\\|={grad_norm_pt:.2f}$',
         transform=ax3.transAxes, fontsize=9, color='#A84B2F')

plt.suptitle('梯度 = 多维损失曲面的"最陡上坡"方向',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("figs/fig_2_3.png", dpi=150, bbox_inches="tight")
print("fig_2_3.png saved")
print(f"梯度={grad_vec}, 模长={grad_norm_pt:.4f}")
