import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP"]
plt.rcParams["axes.unicode_minus"] = False

# ----------------------------------------------------------------
# 3.4节：XGBoost二阶寻优——泰勒二阶展开极致应用
# 可视化：目标函数分解、叶节点最优权重求解、增益公式
# ----------------------------------------------------------------

np.random.seed(0)

# ---- 模拟一棵树的分裂：1D特征，展示最优分裂点 ----
N = 200
x_feat = np.linspace(0, 10, N)
# 模拟残差（负梯度gi = yi - ŷi）
# 模型当前预测偏低，左边误差大
g = -0.5 * np.sin(x_feat) + 0.3 + 0.15*np.random.randn(N)  # 一阶导（残差）
h = 0.8 + 0.2*np.abs(np.cos(x_feat)) + 0.05*np.random.randn(N)  # 二阶导（Hessian，恒正）
h = np.abs(h)  # 保证正数

# XGBoost 叶节点最优权重：w* = -ΣGi / (ΣHi + λ)
# XGBoost 增益：Gain = 0.5 * [GL²/(HL+λ) + GR²/(HR+λ) - (GL+GR)²/(HL+HR+λ)] - γ
lambda_reg = 1.0
gamma_reg  = 0.1

def xgb_score(mask):
    """一个叶子的最优得分（越小越好的obj贡献）"""
    G = g[mask].sum(); H = h[mask].sum()
    return -0.5 * G**2 / (H + lambda_reg)

def xgb_gain(left_mask, right_mask):
    """分裂增益"""
    GL = g[left_mask].sum();  HL = h[left_mask].sum()
    GR = g[right_mask].sum(); HR = h[right_mask].sum()
    G  = GL + GR;             H  = HL + HR
    return 0.5*(GL**2/(HL+lambda_reg) + GR**2/(HR+lambda_reg)
                - G**2/(H+lambda_reg)) - gamma_reg

# 扫描所有分裂点，计算增益
gains = []
split_points = x_feat[5:-5]
for sp in split_points:
    left  = x_feat <= sp
    right = x_feat >  sp
    gains.append(xgb_gain(left, right))
gains = np.array(gains)

best_idx  = np.argmax(gains)
best_split = split_points[best_idx]
best_gain  = gains[best_idx]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# ---- 左图：一阶梯度gi 和 二阶梯度hi 分布 ----
ax1 = axes[0]
ax1.plot(x_feat, g, color='#20808D', lw=1.8, label='$g_i$（一阶梯度/残差）')
ax1.plot(x_feat, h, color='#A84B2F', lw=1.8, ls='--', label='$h_i$（二阶梯度/Hessian）')
ax1.axhline(0, color='gray', lw=0.7, ls=':')
ax1.set_xlabel('样本序号', fontsize=12); ax1.set_ylabel('梯度值', fontsize=12)
ax1.set_title('XGBoost输入：每个样本的\n一阶梯度$g_i$与二阶梯度$h_i$', fontsize=12)
ax1.legend(fontsize=10); ax1.grid(True, alpha=0.3)

# ---- 中图：分裂增益曲线 ----
ax2 = axes[1]
ax2.plot(split_points, gains, color='#944454', lw=2, label='分裂增益 Gain')
ax2.axvline(best_split, color='#A84B2F', lw=2, ls='--',
            label=f'最优分裂点 $x={best_split:.2f}$\nGain={best_gain:.4f}')
ax2.axhline(0, color='gray', lw=0.8, ls=':')
ax2.fill_between(split_points, gains, 0, where=(gains>0), alpha=0.15, color='#20808D')
ax2.set_xlabel('特征分裂阈值', fontsize=12); ax2.set_ylabel('增益 Gain', fontsize=12)
ax2.set_title('扫描所有分裂点的增益\n（泰勒二阶展开目标函数）', fontsize=12)
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

# ---- 右图：最优叶权重 w* = -G/(H+λ) 可视化 ----
ax3 = axes[2]
left_m  = x_feat <= best_split
right_m = x_feat > best_split
GL, HL  = g[left_m].sum(),  h[left_m].sum()
GR, HR  = g[right_m].sum(), h[right_m].sum()
wL = -GL / (HL + lambda_reg)
wR = -GR / (HR + lambda_reg)

ax3.scatter(x_feat[left_m],  g[left_m],  color='#20808D', s=12, alpha=0.6, label='左子树样本')
ax3.scatter(x_feat[right_m], g[right_m], color='#A84B2F', s=12, alpha=0.6, label='右子树样本')
ax3.axvline(best_split, color='k', lw=1.5, ls='--', label=f'分裂点={best_split:.2f}')
ax3.axhline(wL, xmax=(best_split/10.0), color='#20808D', lw=2.5,
            label=f'左叶权重 $w_L^*={wL:.3f}$')
ax3.axhline(wR, xmin=(best_split/10.0), color='#A84B2F', lw=2.5,
            label=f'右叶权重 $w_R^*={wR:.3f}$')
ax3.set_xlabel('特征值 $x$', fontsize=12); ax3.set_ylabel('梯度 $g_i$', fontsize=12)
ax3.set_title(f'最优叶权重 $w^*=-G/(H+\\lambda)$\n$\\lambda={lambda_reg}$', fontsize=12)
ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

plt.suptitle('XGBoost二阶寻优：泰勒展开目标函数解析最优叶权重',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("figs/fig_3_4.png", dpi=150, bbox_inches="tight")
print("fig_3_4.png saved")
print(f"最优分裂点={best_split:.3f}, Gain={best_gain:.4f}")
print(f"左叶权重 wL*={wL:.4f}, 右叶权重 wR*={wR:.4f}")
