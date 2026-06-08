import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

np.random.seed(42)

# ---- 参数设置 ----
T = 1.0          # 总时间（年）
N_steps = 1000   # 离散步数
dt = T / N_steps
t = np.linspace(0, T, N_steps + 1)

# ---- 模拟5条布朗运动路径 ----
n_paths = 5
colors = ["#20808D", "#A84B2F", "#1B474D", "#944454", "#FFC553"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左图：多条路径
ax = axes[0]
for i in range(n_paths):
    increments = np.random.normal(0, np.sqrt(dt), N_steps)
    W = np.concatenate([[0], np.cumsum(increments)])
    ax.plot(t, W, color=colors[i], alpha=0.85, linewidth=0.9, label=f"路径{i+1}")

ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
ax.set_xlabel("时间 $t$（年）")
ax.set_ylabel("$W_t$")
ax.set_title("布朗运动：5条样本路径")
ax.legend(fontsize=8, loc="upper left")
ax.grid(True, alpha=0.3)

# 右图：末端分布与理论N(0,T)对比
ax2 = axes[1]
n_mc = 50000
endpoints = np.sum(np.random.normal(0, np.sqrt(dt), (n_mc, N_steps)), axis=1)
ax2.hist(endpoints, bins=80, density=True, color="#20808D", alpha=0.7, label="模拟 $W_T$ 分布")
x = np.linspace(-3.5, 3.5, 300)
from scipy.stats import norm
ax2.plot(x, norm.pdf(x, 0, np.sqrt(T)), color="#A84B2F", linewidth=2,
         label=f"理论 $N(0,{T})$")
ax2.set_xlabel("$W_T$ 值")
ax2.set_ylabel("概率密度")
ax2.set_title(f"$T={T}$ 时末端分布：模拟 vs 理论")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/home/user/workspace/figs/fig_12_1.png", dpi=150, bbox_inches="tight")
print("fig_12_1.png 保存完成")
