import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP"]
plt.rcParams["axes.unicode_minus"] = False

# ----------------------------------------------------------------
# 2.2节：泰勒展开式——多项式逼近曲线可视化
# 以 sin(x) 和期权价格函数（Black-Scholes近似）为例
# ----------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ---- 左图：sin(x) 泰勒展开阶数逼近 ----
ax1 = axes[0]
x = np.linspace(-np.pi*1.5, np.pi*1.5, 500)
ax1.plot(x, np.sin(x), 'k-', lw=2.5, label='$\\sin(x)$（真值）', zorder=5)

# n阶泰勒项（展开点 a=0）
# sin(x) ≈ x - x³/3! + x⁵/5! - x⁷/7! + ...
from math import factorial
colors_poly = ['#BCE2E7', '#6E522B', '#20808D', '#944454', '#A84B2F']
for k, col in zip([1, 3, 5, 7, 9], colors_poly):
    taylor = np.zeros_like(x)
    for n in range(0, k+1, 2):   # 奇数项
        sign = (-1) ** (n // 2)
        taylor += sign * x**(2*n+1) / factorial(2*n+1)
    ax1.plot(x, np.clip(taylor, -3, 3), color=col, lw=1.8,
             ls='--', label=f'{2*(k//2)+1}阶泰勒')

ax1.set_ylim(-2.5, 2.5)
ax1.axhline(0, color='gray', lw=0.7, ls=':')
ax1.axvline(0, color='gray', lw=0.7, ls=':')
ax1.set_xlabel('$x$', fontsize=12); ax1.set_ylabel('$f(x)$', fontsize=12)
ax1.set_title('$\\sin(x)$ 泰勒展开逐阶逼近（展开点 $a=0$）', fontsize=12)
ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

# ---- 右图：期权Delta/Gamma泰勒近似（Black-Scholes风险展开） ----
ax2 = axes[1]

# 模拟：以当前股价S0为展开点，用Delta+Gamma近似期权价值变化
S0 = 100      # 当前价
K  = 100      # 行权价
T  = 0.5      # 到期期限（年）
r  = 0.05     # 无风险利率
sigma = 0.20  # 波动率

# Black-Scholes call价格（简化用于演示）
from scipy.stats import norm
def bs_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

S_range = np.linspace(70, 130, 300)
C_true  = bs_call(S_range, K, T, r, sigma)

# 在S0处计算Delta（一阶导）和Gamma（二阶导）
d1_0 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
Delta = norm.cdf(d1_0)
Gamma = norm.pdf(d1_0) / (S0 * sigma * np.sqrt(T))
C0    = bs_call(S0, K, T, r, sigma)

dS = S_range - S0
# 0阶、1阶（Delta）、2阶（Delta+Gamma）近似
approx_0  = C0 * np.ones_like(dS)
approx_1  = C0 + Delta * dS
approx_2  = C0 + Delta * dS + 0.5 * Gamma * dS**2

ax2.plot(S_range, C_true,   color='k',       lw=2.5, label='$C(S)$ 真实BS期权价', zorder=5)
ax2.plot(S_range, approx_0, color='#BCE2E7', lw=1.5, ls='--', label='0阶（常数）')
ax2.plot(S_range, approx_1, color='#20808D', lw=2,   ls='--', label='1阶：$C_0 + \\Delta\\cdot\\Delta S$')
ax2.plot(S_range, approx_2, color='#A84B2F', lw=2,   ls='-.',
         label='2阶：$+\\frac{1}{2}\\Gamma(\\Delta S)^2$')
ax2.axvline(S0, color='gray', lw=0.9, ls=':', alpha=0.7)
ax2.text(S0+0.5, C0*0.4, f'展开点\n$S_0={S0}$', fontsize=9, color='gray')
ax2.set_xlabel('股价 $S$', fontsize=12); ax2.set_ylabel('期权价值 $C$', fontsize=12)
ax2.set_title(f'期权价格的泰勒展开（Delta+Gamma近似）\n$K={K}, T={T}, \\sigma={sigma}$', fontsize=12)
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

plt.suptitle('泰勒展开：任意光滑曲线的多项式局部逼近',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("figs/fig_2_2.png", dpi=150, bbox_inches="tight")
print("fig_2_2.png saved")
print(f"Delta={Delta:.4f}, Gamma={Gamma:.4f}, C0={C0:.4f}")
