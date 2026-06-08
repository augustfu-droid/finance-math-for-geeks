import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP"]
plt.rcParams["axes.unicode_minus"] = False

# ----------------------------------------------------------------
# 3.1节：梯度下降法——BGD/SGD/Adam 对比可视化
# 损失曲面：L(w) = (w-3)^2 + 5  (一维，便于清晰展示)
# 并展示二维带鞍点/局部极小的轨迹对比
# ----------------------------------------------------------------

np.random.seed(0)

# ==== 1D 收敛轨迹 ====
def L1d(w): return (w - 3)**2 + 5
def dL1d(w): return 2*(w - 3)

w_range = np.linspace(-2, 8, 400)
w_start = -1.5

def run_gd(w0, lr, n_steps, grad_fn, noise_scale=0):
    w = w0; traj = [w]
    for _ in range(n_steps):
        g = grad_fn(w) + noise_scale * np.random.randn()
        w = w - lr * g
        traj.append(w)
    return np.array(traj)

traj_bgd  = run_gd(w_start, 0.15, 30, dL1d, 0)           # 标准GD
traj_sgd  = run_gd(w_start, 0.15, 30, dL1d, 1.5)          # SGD（加噪）
traj_small = run_gd(w_start, 0.02, 30, dL1d, 0)           # 小学习率GD
traj_large = run_gd(w_start, 0.99, 15, dL1d, 0)           # 过大学习率（震荡）

# Adam（1D）
def run_adam(w0, lr, n_steps, grad_fn, noise_scale=0):
    w = w0; m = 0; v = 0; eps = 1e-8; beta1=0.9; beta2=0.999
    traj = [w]
    for t in range(1, n_steps+1):
        g = grad_fn(w) + noise_scale * np.random.randn()
        m = beta1*m + (1-beta1)*g
        v = beta2*v + (1-beta2)*g**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        w = w - lr * m_hat / (np.sqrt(v_hat) + eps)
        traj.append(w)
    return np.array(traj)

traj_adam = run_adam(w_start, 0.5, 30, dL1d, 0)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# ---- 左图：1D 损失曲线 + 各算法轨迹 ----
ax1 = axes[0]
ax1.plot(w_range, L1d(w_range), 'k-', lw=2, label='$L(w)=(w-3)^2+5$', zorder=1)
styles = [('#20808D','o-',1.8,'BGD lr=0.15'),
          ('#A84B2F','s--',1.6,'SGD（含噪）'),
          ('#FFC553','^:',1.5,'GD lr=0.02（慢）'),
          ('#944454','D-.',1.6,'GD lr=0.99（震荡）'),
          ('#6E522B','P-',2.0,'Adam lr=0.5')]
for traj, (col,mk,lw,lbl) in zip([traj_bgd,traj_sgd,traj_small,traj_large,traj_adam],
                                   styles):
    ax1.plot(traj, L1d(traj), mk, color=col, lw=lw, ms=4, alpha=0.85, label=lbl)
ax1.plot(3, 5, 'r*', ms=14, zorder=5, label='全局最小 $w^*=3$')
ax1.set_xlabel('参数 $w$', fontsize=12); ax1.set_ylabel('损失 $L(w)$', fontsize=12)
ax1.set_title('各优化算法1D轨迹', fontsize=12)
ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

# ---- 中图：收敛曲线（迭代步数 vs 损失） ----
ax2 = axes[1]
for traj, (col,mk,lw,lbl) in zip([traj_bgd,traj_sgd,traj_small,traj_adam],
                                   [s for i,s in enumerate(styles) if i!=3]):
    ax2.semilogy(L1d(traj), mk, color=col, lw=lw, ms=3, alpha=0.85, label=lbl)
ax2.set_xlabel('迭代步数', fontsize=12); ax2.set_ylabel('损失（对数刻度）', fontsize=12)
ax2.set_title('收敛速度对比', fontsize=12)
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

# ---- 右图：2D Rosenbrock（带复杂地形）Adam轨迹 ----
ax3 = axes[2]
def rosen(w): return (1-w[0])**2 + 100*(w[1]-w[0]**2)**2
def rosen_grad(w):
    g0 = -2*(1-w[0]) - 400*w[0]*(w[1]-w[0]**2)
    g1 = 200*(w[1]-w[0]**2)
    return np.array([g0, g1])

x2 = np.linspace(-2, 2, 300); y2 = np.linspace(-0.5, 3, 300)
X2, Y2 = np.meshgrid(x2, y2)
Z2 = (1-X2)**2 + 100*(Y2-X2**2)**2

ax3.contourf(X2, Y2, np.log1p(Z2), levels=40, cmap='viridis', alpha=0.85)
ax3.contour(X2, Y2, np.log1p(Z2), levels=20, colors='white', alpha=0.15, linewidths=0.5)

# Adam 2D
def run_adam2d(w0, lr, n_steps):
    w = w0.copy(); m=np.zeros(2); v=np.zeros(2); eps=1e-8; b1=0.9; b2=0.999
    traj = [w.copy()]
    for t in range(1, n_steps+1):
        g = rosen_grad(w)
        m = b1*m + (1-b1)*g; v = b2*v + (1-b2)*g**2
        w = w - lr * (m/(1-b1**t)) / (np.sqrt(v/(1-b2**t))+eps)
        traj.append(w.copy())
    return np.array(traj)

tr2d_adam = run_adam2d(np.array([-1.5, 2.0]), 0.005, 3000)
ax3.plot(tr2d_adam[:,0], tr2d_adam[:,1], 'w-', lw=0.8, alpha=0.7, label='Adam轨迹')
ax3.plot(tr2d_adam[0,0], tr2d_adam[0,1], 'co', ms=9, label='起点')
ax3.plot(1, 1, 'r*', ms=12, label='全局最小(1,1)')
ax3.set_xlabel('$w_1$', fontsize=12); ax3.set_ylabel('$w_2$', fontsize=12)
ax3.set_title('Rosenbrock函数\nAdam轨迹（复杂曲面）', fontsize=12)
ax3.legend(fontsize=9)

plt.suptitle('梯度下降：BGD / SGD / Adam 收敛原理与轨迹对比',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("figs/fig_3_1.png", dpi=150, bbox_inches="tight")
print("fig_3_1.png saved")
