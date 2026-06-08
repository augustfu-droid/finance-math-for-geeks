import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP"]
plt.rcParams["axes.unicode_minus"] = False

# ----------------------------------------------------------------
# 3.2节：反向传播BP——手动实现2层网络，逐层梯度可视化
# 网络：x(2) → W1(2x4) → ReLU → W2(4x1) → sigmoid → L
# ----------------------------------------------------------------

np.random.seed(42)

def relu(z): return np.maximum(0, z)
def relu_deriv(z): return (z > 0).astype(float)
def sigmoid(z): return 1 / (1 + np.exp(-np.clip(z, -30, 30)))

# 小型数据集
N = 200
X = np.random.randn(N, 2)
y = ((X[:,0]**2 + X[:,1]**2) < 1).astype(float).reshape(-1,1)  # 圆内=1

# 初始化权重
W1 = np.random.randn(2, 4) * 0.5
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1) * 0.5
b2 = np.zeros((1, 1))

lr = 0.05
losses = []
grad_W1_norms = []
grad_W2_norms = []

for epoch in range(500):
    # === 前向传播 ===
    Z1 = X @ W1 + b1          # (N,4)
    A1 = relu(Z1)              # (N,4)
    Z2 = A1 @ W2 + b2          # (N,1)
    A2 = sigmoid(Z2)           # (N,1) 预测概率
    loss = -np.mean(y * np.log(A2+1e-9) + (1-y) * np.log(1-A2+1e-9))
    losses.append(loss)

    # === 反向传播（链式法则逐层） ===
    # ∂L/∂Z2：sigmoid的输出层梯度
    dZ2 = (A2 - y) / N                    # (N,1)
    dW2 = A1.T @ dZ2                       # (4,1)
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    # ∂L/∂A1：通过W2传回
    dA1 = dZ2 @ W2.T                       # (N,4)
    # ∂L/∂Z1：ReLU梯度
    dZ1 = dA1 * relu_deriv(Z1)             # (N,4)
    dW1 = X.T @ dZ1                        # (2,4)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    grad_W1_norms.append(np.linalg.norm(dW1))
    grad_W2_norms.append(np.linalg.norm(dW2))

    # 更新
    W1 -= lr * dW1; b1 -= lr * db1
    W2 -= lr * dW2; b2 -= lr * db2

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# ---- 左图：决策边界 ----
ax1 = axes[0]
xx, yy = np.meshgrid(np.linspace(-3,3,200), np.linspace(-3,3,200))
grid = np.c_[xx.ravel(), yy.ravel()]
Z1g = relu(grid @ W1 + b1)
pred_grid = sigmoid(Z1g @ W2 + b2).reshape(200,200)
ax1.contourf(xx, yy, pred_grid, levels=50, cmap='RdBu_r', alpha=0.7)
ax1.scatter(X[y[:,0]==1,0], X[y[:,0]==1,1], c='#20808D', s=12, alpha=0.7, label='正例')
ax1.scatter(X[y[:,0]==0,0], X[y[:,0]==0,1], c='#A84B2F', s=12, alpha=0.7, label='负例')
circle = plt.Circle((0,0), 1, fill=False, color='k', lw=2, ls='--', label='真实边界')
ax1.add_patch(circle)
ax1.set_xlim(-3,3); ax1.set_ylim(-3,3)
ax1.set_xlabel('$x_1$', fontsize=12); ax1.set_ylabel('$x_2$', fontsize=12)
ax1.set_title('BP训练后的决策边界', fontsize=12)
ax1.legend(fontsize=8); ax1.set_aspect('equal')

# ---- 中图：损失曲线 ----
ax2 = axes[1]
ax2.semilogy(losses, color='#20808D', lw=2, label='训练损失')
ax2.set_xlabel('训练轮次', fontsize=12); ax2.set_ylabel('交叉熵损失（对数）', fontsize=12)
ax2.set_title('BP训练损失收敛曲线', fontsize=12)
ax2.legend(fontsize=11); ax2.grid(True, alpha=0.3)

# ---- 右图：逐层梯度模长（梯度消失分析） ----
ax3 = axes[2]
ax3.plot(grad_W1_norms, color='#944454', lw=1.8, label='$\\|\\nabla_{W_1}L\\|$（浅层）')
ax3.plot(grad_W2_norms, color='#20808D', lw=1.8, label='$\\|\\nabla_{W_2}L\\|$（深层）')
ax3.set_xlabel('训练轮次', fontsize=12); ax3.set_ylabel('梯度模长', fontsize=12)
ax3.set_title('逐层梯度模长：链式法则逐层衰减\n（浅层梯度消失现象）', fontsize=12)
ax3.legend(fontsize=10); ax3.grid(True, alpha=0.3)

plt.suptitle('BP算法：链式法则在两层网络的完整前向+反向实现',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("figs/fig_3_2.png", dpi=150, bbox_inches="tight")
print("fig_3_2.png saved")
print(f"最终损失: {losses[-1]:.4f}")
