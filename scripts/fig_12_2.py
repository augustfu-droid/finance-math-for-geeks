import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

np.random.seed(0)

# ---- 参数 ----
S0 = 100.0       # 初始股价
mu = 0.10        # 漂移率
sigma = 0.25     # 波动率
T = 1.0          # 1年
N = 1000         # 步数
dt = T / N
t = np.linspace(0, T, N + 1)

n_paths = 6
colors = ["#20808D", "#A84B2F", "#1B474D", "#944454", "#FFC553", "#6E522B"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左图：伊藤引理给出的GBM路径  dS = mu*S*dt + sigma*S*dW
# 精确解: S(t) = S0 * exp((mu - sigma^2/2)*t + sigma*W_t)
ax = axes[0]
for i in range(n_paths):
    dW = np.random.normal(0, np.sqrt(dt), N)
    W = np.concatenate([[0], np.cumsum(dW)])
    S = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)
    ax.plot(t, S, color=colors[i], alpha=0.8, linewidth=0.9)

ax.set_xlabel("时间 $t$（年）")
ax.set_ylabel("股价 $S_t$")
ax.set_title("几何布朗运动（GBM）路径\n$dS = \\mu S\\,dt + \\sigma S\\,dW$")
ax.axhline(S0, color="gray", linestyle="--", linewidth=0.7, label=f"$S_0={S0}$")
ax.legend()
ax.grid(True, alpha=0.3)

# 右图：比较不同dt离散化误差（伊藤离散化精度）
ax2 = axes[1]
dt_list = [0.01, 0.05, 0.1, 0.25, 0.5]
n_mc = 10000
S0_test = 100
mu_test, sigma_test = 0.1, 0.2
T_test = 1.0
exact_mean = S0_test * np.exp(mu_test * T_test)

errors = []
for dt_i in dt_list:
    N_i = int(T_test / dt_i)
    S_ends = np.full(n_mc, float(S0_test))
    for _ in range(N_i):
        dW_i = np.random.normal(0, np.sqrt(dt_i), n_mc)
        S_ends = S_ends * np.exp((mu_test - 0.5 * sigma_test**2) * dt_i + sigma_test * dW_i)
    errors.append(abs(np.mean(S_ends) - exact_mean))

ax2.bar([str(d) for d in dt_list], errors, color="#20808D", alpha=0.8, edgecolor="white")
ax2.set_xlabel("离散化步长 $\\Delta t$")
ax2.set_ylabel("|模拟均值 − 理论均值|")
ax2.set_title("伊藤离散化步长 vs 定价偏差\n（精确解法 $dt$ 对误差影响小）")
ax2.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("/home/user/workspace/figs/fig_12_2.png", dpi=150, bbox_inches="tight")
print("fig_12_2.png 保存完成")
