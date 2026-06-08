import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

# ---- Black-Scholes 公式 ----
def bs_call(S, K, T, r, sigma):
    """BS欧式看涨期权价格"""
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_put(S, K, T, r, sigma):
    """BS欧式看跌期权价格"""
    if T <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# ---- 左图：期权价格 vs 股价 ----
S_range = np.linspace(50, 160, 300)
K, T0, r, sigma0 = 100, 1.0, 0.05, 0.25

call_prices = [bs_call(s, K, T0, r, sigma0) for s in S_range]
put_prices = [bs_put(s, K, T0, r, sigma0) for s in S_range]
intrinsic_call = np.maximum(S_range - K, 0)

ax = axes[0]
ax.plot(S_range, call_prices, color="#20808D", linewidth=2, label="看涨 Call")
ax.plot(S_range, put_prices, color="#A84B2F", linewidth=2, label="看跌 Put")
ax.plot(S_range, intrinsic_call, color="gray", linewidth=1, linestyle="--", label="内在价值")
ax.axvline(K, color="black", linewidth=0.7, linestyle=":", label=f"K={K}")
ax.set_xlabel("标的股价 $S$")
ax.set_ylabel("期权价格")
ax.set_title("BS期权价格 vs 股价\n$K=100, T=1, r=5\\%, \\sigma=25\\%$")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ---- 中图：隐含波动率微笑（不同行权价的隐含波动率） ----
# 用不同sigma模拟"市场观测"价格，再反推演示微笑形状
K_range = np.linspace(70, 130, 50)
S0 = 100
# 真实市场的"偏斜"波动率（虚构演示）
smile_sigma = 0.20 + 0.08 * ((K_range - S0) / S0)**2 - 0.03 * (K_range - S0) / S0

ax2 = axes[1]
ax2.plot(K_range, smile_sigma * 100, color="#20808D", linewidth=2)
ax2.axvline(S0, color="gray", linestyle="--", linewidth=0.8, label="平值ATM")
ax2.set_xlabel("行权价 $K$")
ax2.set_ylabel("隐含波动率 (%)")
ax2.set_title("波动率微笑（Volatility Smile）\n（市场对尾部风险溢价的定价）")
ax2.legend()
ax2.grid(True, alpha=0.3)

# ---- 右图：期权价格 vs 剩余到期时间 ----
T_range = np.linspace(0.01, 2.0, 200)
S_atm, K_atm = 100, 100

call_t = [bs_call(S_atm, K_atm, t_i, r, sigma0) for t_i in T_range]
put_t = [bs_put(S_atm, K_atm, t_i, r, sigma0) for t_i in T_range]

ax3 = axes[2]
ax3.plot(T_range, call_t, color="#20808D", linewidth=2, label="ATM看涨 Call")
ax3.plot(T_range, put_t, color="#A84B2F", linewidth=2, label="ATM看跌 Put")
ax3.set_xlabel("剩余到期时间 $T$（年）")
ax3.set_ylabel("期权价格")
ax3.set_title("时间价值递减（时间衰减 Theta）\n平值期权 $S=K=100$")
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/home/user/workspace/figs/fig_12_3.png", dpi=150, bbox_inches="tight")
print("fig_12_3.png 保存完成")
