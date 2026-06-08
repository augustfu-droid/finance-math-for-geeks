import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP"]
plt.rcParams["axes.unicode_minus"] = False

# ----------------------------------------------------------------
# 1.4节：链式法则——前向传播与误差反向传导可视化
# 简单两层网络：x → z=wx+b → a=σ(z) → L=(a-y)²
# 展示 dL/dw = dL/da · da/dz · dz/dw 逐层乘积
# ----------------------------------------------------------------

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1 - s)

# 前向传播（固定参数，扫描输入范围）
w = 1.5; b = -0.5; y_true = 0.8
x_range = np.linspace(-3, 3, 300)

z   = w * x_range + b
a   = sigmoid(z)
L   = (a - y_true)**2

# 各梯度（链式法则分解）
dL_da  =  2 * (a - y_true)          # ∂L/∂a
da_dz  =  sigmoid_deriv(z)          # ∂a/∂z = σ'(z)
dz_dw  =  x_range                   # ∂z/∂w = x
dL_dw  =  dL_da * da_dz * dz_dw    # 链式乘积

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# ---- (0,0)：计算图示意 ----
ax00 = axes[0, 0]
ax00.axis('off')
# 用文字+箭头画计算图
nodes = {'x': (0.08, 0.5), 'z=wx+b': (0.32, 0.5),
         'a=σ(z)': (0.58, 0.5), 'L=(a−y)²': (0.85, 0.5)}
for name, (nx, ny) in nodes.items():
    ax00.text(nx, ny, name, ha='center', va='center', fontsize=13,
              bbox=dict(boxstyle='round,pad=0.4', fc='#BCE2E7', ec='#20808D', lw=1.5))
arrows = [('x','z=wx+b','dz/dw=x'), ('z=wx+b','a=σ(z)','da/dz=σ\'(z)'),
          ('a=σ(z)','L=(a−y)²','dL/da=2(a-y)')]
for src, dst, lbl in arrows:
    sx, sy = nodes[src]; dx, dy = nodes[dst]
    ax00.annotate('', xy=(dx-0.09, dy), xytext=(sx+0.09, sy),
                  arrowprops=dict(arrowstyle='->', color='#20808D', lw=2))
    mx = (sx + dx) / 2
    ax00.text(mx, sy+0.14, lbl, ha='center', fontsize=9.5, color='#A84B2F')

# 反向箭头
ax00.annotate('', xy=(0.08, 0.28), xytext=(0.85, 0.28),
              arrowprops=dict(arrowstyle='->', color='#944454', lw=2, linestyle='dashed'))
ax00.text(0.47, 0.20, 'dL/dw = dL/da · da/dz · dz/dw   （反向传导）',
          ha='center', fontsize=11, color='#944454', fontweight='bold')
ax00.set_title('链式法则：计算图（前向+反向）', fontsize=12)

# ---- (0,1)：损失函数 L(x) ----
ax01 = axes[0, 1]
ax01.plot(x_range, L, color='#20808D', lw=2.2, label='损失 $L(x)$')
ax01.set_xlabel('输入 $x$', fontsize=12); ax01.set_ylabel('损失 $L$', fontsize=12)
ax01.set_title('损失函数 $L = (\\sigma(wx+b) - y)^2$', fontsize=12)
ax01.legend(fontsize=11); ax01.grid(True, alpha=0.3)

# ---- (1,0)：各分项导数 ----
ax10 = axes[1, 0]
ax10.plot(x_range, dL_da, color='#A84B2F', lw=2, label='$\\partial L/\\partial a = 2(a-y)$')
ax10.plot(x_range, da_dz, color='#20808D', lw=2, ls='--', label="$\\partial a/\\partial z = \\sigma'(z)$")
ax10.axhline(0, color='gray', lw=0.8, ls=':')
ax10.set_xlabel('输入 $x$', fontsize=12); ax10.set_ylabel('梯度值', fontsize=12)
ax10.set_title('链式各分项导数', fontsize=12)
ax10.legend(fontsize=10); ax10.grid(True, alpha=0.3)

# ---- (1,1)：最终链式梯度 dL/dw ----
ax11 = axes[1, 1]
ax11.plot(x_range, dL_dw, color='#944454', lw=2.5, label='$\\partial L/\\partial w$（链式乘积）')
ax11.fill_between(x_range, dL_dw, 0, where=(dL_dw>0), alpha=0.15, color='#A84B2F')
ax11.fill_between(x_range, dL_dw, 0, where=(dL_dw<0), alpha=0.15, color='#20808D')
ax11.axhline(0, color='gray', lw=0.8, ls=':')
ax11.set_xlabel('输入 $x$', fontsize=12); ax11.set_ylabel('$\\partial L/\\partial w$', fontsize=12)
ax11.set_title('最终梯度 $\\partial L/\\partial w$（链式法则之积）', fontsize=12)
ax11.legend(fontsize=10); ax11.grid(True, alpha=0.3)

plt.suptitle('链式法则：误差如何从损失层逐级传回权重层',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("figs/fig_1_4.png", dpi=150, bbox_inches="tight")
print("fig_1_4.png saved")
