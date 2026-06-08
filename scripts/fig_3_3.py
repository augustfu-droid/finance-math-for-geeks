import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP"]
plt.rcParams["axes.unicode_minus"] = False

# ----------------------------------------------------------------
# 3.3节：希腊字母——期权Delta/Gamma/Theta/Vega 全面可视化
# Black-Scholes Call 期权
# ----------------------------------------------------------------

def bs_greeks(S, K, T, r, sigma):
    """返回 call 期权价格和完整希腊字母"""
    if np.isscalar(T) and T <= 0:
        C = np.maximum(S - K, 0)
        return C, np.where(S>K,1,0), 0, 0, 0
    sqT = np.sqrt(np.maximum(T, 1e-9))
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*sqT)
    d2 = d1 - sigma*sqT
    C     = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    Delta = norm.cdf(d1)                              # ∂C/∂S（一阶导）
    Gamma = norm.pdf(d1) / (S * sigma * sqT)          # ∂²C/∂S²（二阶导）
    Theta = (-(S*norm.pdf(d1)*sigma)/(2*sqT)
             - r*K*np.exp(-r*T)*norm.cdf(d2)) / 252   # 每日衰减
    Vega  = S * norm.pdf(d1) * sqT / 100              # 波动率敏感度（per 1%）
    return C, Delta, Gamma, Theta, Vega

K, r, sigma = 100, 0.05, 0.20

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# ---- (0,0)：不同到期期限下 Delta vs S ----
ax1 = axes[0, 0]
S_range = np.linspace(60, 140, 400)
for T, col in [(1.0,'#20808D'),(0.5,'#A84B2F'),(0.25,'#944454'),(0.08,'#FFC553')]:
    _, delta, _, _, _ = bs_greeks(S_range, K, T, r, sigma)
    ax1.plot(S_range, delta, color=col, lw=2, label=f'T={T:.2f}年')
ax1.axvline(K, color='gray', lw=0.9, ls=':', alpha=0.7)
ax1.text(K+0.5, 0.05, f'$K={K}$', fontsize=9, color='gray')
ax1.set_xlabel('股价 $S$', fontsize=12); ax1.set_ylabel('Delta $\\Delta$', fontsize=12)
ax1.set_title('Delta = $\\partial C/\\partial S$（期权对冲比例）\n到期日越近，Delta越陡', fontsize=11)
ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

# ---- (0,1)：Gamma vs S——风险加速度 ----
ax2 = axes[0, 1]
for T, col in [(1.0,'#20808D'),(0.5,'#A84B2F'),(0.25,'#944454'),(0.08,'#FFC553')]:
    _, _, gamma, _, _ = bs_greeks(S_range, K, T, r, sigma)
    ax2.plot(S_range, gamma, color=col, lw=2, label=f'T={T:.2f}年')
ax2.axvline(K, color='gray', lw=0.9, ls=':')
ax2.set_xlabel('股价 $S$', fontsize=12); ax2.set_ylabel('Gamma $\\Gamma$', fontsize=12)
ax2.set_title('Gamma = $\\partial^2 C/\\partial S^2$（Delta的变化速率）\n平值附近Gamma最大，是风险集中区', fontsize=11)
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

# ---- (1,0)：Theta时间衰减 ----
ax3 = axes[1, 0]
T_range = np.linspace(0.005, 2.0, 400)
for S_test, col, lbl in [(90,'#20808D','价外S=90'),(100,'#A84B2F','平值S=100'),(110,'#944454','价内S=110')]:
    _, _, _, theta, _ = bs_greeks(S_test, K, T_range, r, sigma)
    ax3.plot(T_range, theta, color=col, lw=2, label=lbl)
ax3.axhline(0, color='gray', lw=0.8, ls=':')
ax3.set_xlabel('到期期限 $T$（年）', fontsize=12); ax3.set_ylabel('Theta（每日）', fontsize=12)
ax3.set_title('Theta = 时间衰减（每日损失价值）\n临近到期，平值期权Theta最负', fontsize=11)
ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3)

# ---- (1,1)：Vega & Delta-Gamma P&L归因 ----
ax4 = axes[1, 1]
# Delta-Gamma近似 P&L
T0 = 0.5
C0, D0, G0, _, _ = bs_greeks(K, K, T0, r, sigma)
dS_range = np.linspace(-20, 20, 300)
S_new = K + dS_range
C_true, _, _, _, _ = bs_greeks(S_new, K, T0, r, sigma)
C_delta     = C0 + D0 * dS_range
C_delta_gamma = C0 + D0*dS_range + 0.5*G0*dS_range**2

ax4.plot(dS_range, C_true - C0,       'k-',  lw=2.5, label='真实 P&L')
ax4.plot(dS_range, C_delta - C0,      color='#20808D', lw=2, ls='--', label='Delta线性近似')
ax4.plot(dS_range, C_delta_gamma-C0, color='#A84B2F', lw=2, ls='-.', label='Delta+Gamma二阶近似')
ax4.axhline(0, color='gray', lw=0.8, ls=':')
ax4.axvline(0, color='gray', lw=0.8, ls=':')
ax4.set_xlabel('股价变动 $\\Delta S$', fontsize=12)
ax4.set_ylabel('期权价值变动 $\\Delta C$', fontsize=12)
ax4.set_title('Delta vs Delta+Gamma P&L归因\n（泰勒展开精度对比）', fontsize=11)
ax4.legend(fontsize=9); ax4.grid(True, alpha=0.3)

plt.suptitle('期权希腊字母：Delta（一阶导）、Gamma（二阶导）、Theta、Vega',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("figs/fig_3_3.png", dpi=150, bbox_inches="tight")
print("fig_3_3.png saved")
print(f"ATM Call: Delta={D0:.4f}, Gamma={G0:.4f}")
