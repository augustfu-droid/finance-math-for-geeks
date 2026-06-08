import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ===== 5.1: 单位阵、对角阵与转置 =====
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

def plot_matrix(ax, M, title, cmap='Blues', fmt='.0f'):
    im = ax.imshow(M, cmap=cmap, aspect='auto', vmin=0, vmax=max(M.max(), 1))
    n, m = M.shape
    for i in range(n):
        for j in range(m):
            v = M[i,j]
            txt = f'{v:{fmt}}'
            ax.text(j, i, txt, ha='center', va='center',
                    fontsize=13, fontweight='bold',
                    color='white' if v > M.max()*0.6 else '#28251D')
    ax.set_xticks(range(m))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f'c{j+1}' for j in range(m)], fontsize=9)
    ax.set_yticklabels([f'r{i+1}' for i in range(n)], fontsize=9)
    ax.set_title(title, fontsize=12, fontweight='bold')

I = np.eye(4)
D = np.diag([3, 1, 4, 2])
A_t = np.array([[1,2,3],[4,5,6]], dtype=float)

plot_matrix(axes[0], I, "单位矩阵 I (4x4)", cmap='Blues')
plot_matrix(axes[1], D, "对角矩阵 D (4x4)", cmap='Greens')
plot_matrix(axes[2], A_t, "矩阵 A (2x3)   →   A^T (3x2)", cmap='Oranges', fmt='.0f')
axes[2].set_title("矩阵 A (2x3) → A^T (3x2)\n转置: 行变列", fontsize=11, fontweight='bold')

plt.suptitle("5.1 单位阵·对角阵·转置可视化", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("/home/user/workspace/figs/fig_5_1.png", dpi=150, bbox_inches="tight")
plt.close()
print("fig_5_1.png saved")

# ===== 5.2: 逆矩阵与行列式 =====
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左图：行列式的几何意义（面积）
ax = axes[0]
# 用向量张成的平行四边形面积
v1 = np.array([3, 1])
v2 = np.array([1, 2])
det_val = np.linalg.det(np.array([v1, v2]).T)
origin = [0, 0]
para_x = [0, v1[0], v1[0]+v2[0], v2[0], 0]
para_y = [0, v1[1], v1[1]+v2[1], v2[1], 0]
ax.fill(para_x, para_y, alpha=0.25, color='#20808D', label=f'面积=|det|={abs(det_val):.1f}')
ax.annotate("", xy=v1, xytext=(0,0), arrowprops=dict(arrowstyle='->', color='#20808D', lw=2.5))
ax.annotate("", xy=v2, xytext=(0,0), arrowprops=dict(arrowstyle='->', color='#A84B2F', lw=2.5))
ax.text(v1[0]/2+0.1, v1[1]/2+0.1, r'$\vec{v_1}$', fontsize=14, color='#20808D')
ax.text(v2[0]/2-0.35, v2[1]/2, r'$\vec{v_2}$', fontsize=14, color='#A84B2F')
ax.text(v1[0]/2+v2[0]/2-0.1, v1[1]/2+v2[1]/2, f'面积={abs(det_val):.1f}', fontsize=11,
        ha='center', color='#1B474D', fontweight='bold')
ax.set_xlim(-0.5, 5.5); ax.set_ylim(-0.5, 4.5)
ax.axhline(0, color='gray', lw=0.7); ax.axvline(0, color='gray', lw=0.7)
ax.grid(True, linestyle='--', alpha=0.3)
ax.set_title(f"行列式几何意义：平行四边形面积\ndet={det_val:.2f}", fontsize=11, fontweight='bold')
ax.legend(fontsize=10)

# 右图：条件数 vs 矩阵"病态"程度
ax = axes[1]
np.random.seed(0)
cond_nums = []
det_vals = []
for eps in np.linspace(0, 1, 80):
    M = np.array([[1, eps], [eps, 1]])
    cond_nums.append(np.linalg.cond(M))
    det_vals.append(abs(np.linalg.det(M)))

eps_vals = np.linspace(0, 1, 80)
ax2 = ax.twinx()
l1, = ax.plot(eps_vals, cond_nums, color='#A84B2F', lw=2, label='条件数 κ')
l2, = ax2.plot(eps_vals, det_vals, color='#20808D', lw=2, linestyle='--', label='|det|')
ax.set_xlabel("矩阵相关程度 ε")
ax.set_ylabel("条件数 κ", color='#A84B2F')
ax2.set_ylabel("|det|", color='#20808D')
ax.set_title("ε→1时：行列式→0，条件数→∞\n矩阵趋于奇异/病态", fontsize=11, fontweight='bold')
ax.legend(handles=[l1, l2], fontsize=9, loc='center left')
ax.grid(True, linestyle='--', alpha=0.3)

plt.suptitle("5.2 行列式·逆矩阵·奇异矩阵陷阱", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("/home/user/workspace/figs/fig_5_2.png", dpi=150, bbox_inches="tight")
plt.close()
print("fig_5_2.png saved")

# ===== 5.3: 线性相关、共线性与秩 =====
np.random.seed(42)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# 模拟因子：市值、市净率、价值分
n = 100
size_factor = np.random.randn(n)
pb_factor = size_factor * 0.95 + np.random.randn(n) * 0.1  # 高度共线
value_factor = np.random.randn(n)                           # 独立

# 散点：size vs pb（共线）
ax = axes[0]
ax.scatter(size_factor, pb_factor, color='#A84B2F', alpha=0.5, s=25)
m, b = np.polyfit(size_factor, pb_factor, 1)
x_line = np.linspace(-3, 3, 50)
ax.plot(x_line, m*x_line+b, color='#A84B2F', lw=2)
corr_1 = np.corrcoef(size_factor, pb_factor)[0,1]
ax.set_title(f"市值因子 vs 市净率因子\ncorr={corr_1:.3f}  ← 高度共线！", fontsize=10, fontweight='bold')
ax.set_xlabel("市值因子"); ax.set_ylabel("市净率因子")
ax.grid(True, linestyle='--', alpha=0.3)

# 散点：size vs value（独立）
ax = axes[1]
ax.scatter(size_factor, value_factor, color='#20808D', alpha=0.5, s=25)
corr_2 = np.corrcoef(size_factor, value_factor)[0,1]
ax.set_title(f"市值因子 vs 价值因子\ncorr={corr_2:.3f}  ← 正交！", fontsize=10, fontweight='bold')
ax.set_xlabel("市值因子"); ax.set_ylabel("价值因子")
ax.grid(True, linestyle='--', alpha=0.3)

# 热图：相关矩阵的秩
F_matrix = np.column_stack([size_factor, pb_factor, value_factor,
                              np.random.randn(n)])  # 4个因子
F_labels = ['市值', '市净率(共线)', '价值', '随机']
corr_matrix = np.corrcoef(F_matrix.T)
rank_val = np.linalg.matrix_rank(np.corrcoef(F_matrix.T))
ax = axes[2]
im = ax.imshow(corr_matrix, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
plt.colorbar(im, ax=ax, label='相关系数')
for i in range(4):
    for j in range(4):
        ax.text(j, i, f'{corr_matrix[i,j]:.2f}', ha='center', va='center',
                fontsize=10, fontweight='bold',
                color='white' if abs(corr_matrix[i,j]) > 0.7 else '#28251D')
ax.set_xticks(range(4)); ax.set_yticks(range(4))
ax.set_xticklabels(F_labels, fontsize=9, rotation=20)
ax.set_yticklabels(F_labels, fontsize=9)
ax.set_title(f"4因子相关矩阵  矩阵秩={rank_val}\n(秩<4: 存在共线性)", fontsize=10, fontweight='bold')

plt.suptitle("5.3 线性相关·共线性·矩阵的秩", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("/home/user/workspace/figs/fig_5_3.png", dpi=150, bbox_inches="tight")
plt.close()
print("fig_5_3.png saved")

# ===== 5.4: 特征值与特征向量 =====
np.random.seed(42)
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# 协方差矩阵
n_pts = 200
cov_mat = np.array([[3, 1.8], [1.8, 1.5]])
data = np.random.multivariate_normal([0, 0], cov_mat, n_pts)
eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# 左：散点 + 特征向量
ax = axes[0]
ax.scatter(data[:,0], data[:,1], alpha=0.3, s=15, color='#BCE2E7')
scale = 2
for i, (ev, lam) in enumerate(zip(eigenvectors.T, eigenvalues)):
    color = ['#20808D', '#A84B2F'][i]
    ax.annotate("", xy=(ev[0]*np.sqrt(lam)*scale, ev[1]*np.sqrt(lam)*scale),
                xytext=(0,0),
                arrowprops=dict(arrowstyle='->', color=color, lw=2.5))
    ax.text(ev[0]*np.sqrt(lam)*scale+0.1, ev[1]*np.sqrt(lam)*scale+0.1,
            f'λ{i+1}={lam:.2f}', fontsize=11, color=color, fontweight='bold')
ax.set_xlim(-6, 6); ax.set_ylim(-6, 6)
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.3)
ax.axhline(0, color='gray', lw=0.7); ax.axvline(0, color='gray', lw=0.7)
ax.set_title("协方差矩阵特征向量\n= 数据主方向", fontsize=11, fontweight='bold')

# 中：特征值大小（解释方差比）
ax = axes[1]
var_ratio = eigenvalues / eigenvalues.sum()
bars = ax.bar(['PC1', 'PC2'], var_ratio*100,
              color=['#20808D', '#A84B2F'], width=0.4, edgecolor='white')
for bar, v in zip(bars, var_ratio):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f'{v*100:.1f}%', ha='center', fontsize=12, fontweight='bold')
ax.set_ylim(0, 100)
ax.set_ylabel("方差解释比例 (%)")
ax.set_title("特征值解释方差比例\n(即PCA主成分贡献)", fontsize=11, fontweight='bold')
ax.grid(True, axis='y', linestyle='--', alpha=0.3)

# 右：Av vs λv 验证
ax = axes[2]
A = cov_mat
v = eigenvectors[:, 0]  # 第一主特征向量
lam = eigenvalues[0]
Av = A @ v
lv = lam * v
# 绘制
ax.annotate("", xy=(v[0]*2, v[1]*2), xytext=(0,0),
            arrowprops=dict(arrowstyle='->', color='#1B474D', lw=2, linestyle='--'))
ax.annotate("", xy=(Av[0], Av[1]), xytext=(0,0),
            arrowprops=dict(arrowstyle='->', color='#20808D', lw=2.5))
ax.annotate("", xy=(lv[0], lv[1]), xytext=(0,0),
            arrowprops=dict(arrowstyle='->', color='#A84B2F', lw=2, linestyle=':'))
ax.text(v[0]+0.1, v[1]*2+0.05, r'$\vec{v}$', fontsize=14, color='#1B474D')
ax.text(Av[0]+0.05, Av[1]+0.05, r'$A\vec{v}$', fontsize=14, color='#20808D')
ax.text(lv[0]-0.6, lv[1]+0.1, r'$\lambda\vec{v}$', fontsize=14, color='#A84B2F')
ax.set_xlim(-1, 4); ax.set_ylim(-1, 3)
ax.axhline(0, color='gray', lw=0.7); ax.axvline(0, color='gray', lw=0.7)
ax.grid(True, linestyle='--', alpha=0.3)
ax.set_title(r"验证 $A\vec{v}=\lambda\vec{v}$" + "\n(蓝橙两箭头完全重合)", fontsize=11, fontweight='bold')
print(f"Av={Av}, lv={lv}, diff={np.max(np.abs(Av-lv)):.2e}")

plt.suptitle("5.4 特征值·特征向量·主轴方向", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("/home/user/workspace/figs/fig_5_4.png", dpi=150, bbox_inches="tight")
plt.close()
print("fig_5_4.png saved")
