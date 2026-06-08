# 第五篇 金融数学核心（随机分析与资产定价）

## 第12章 随机分析——连续时间金融的数学语言

> ### 💡 本章数学直觉导引
>
> 普通微积分假设一切都是光滑的——股价今天100元，明天101元，后天102元，像一条可以求导的曲线。但现实里，股价每一秒都在随机跳动，不存在"切线斜率"，你永远无法预测下一秒是涨还是跌。
>
> 本章解决的核心问题是：**如何用严格的数学语言描述这种随机性，并在随机世界里做微积分运算？**
>
> 直觉路线图如下：
>
> - **布朗运动（12.1）**：把随机游走（抛硬币走路）缩到时间无穷小，得到"连续随机噪声"的数学模型——这是一切随机模型的基础砖块，就像电路里的白噪声信号源。
> - **伊藤引理（12.2）**：股票不直接是布朗运动，而是布朗运动的某个函数（如 $e^{W_t}$）。对随机函数求"微分"需要新规则——伊藤引理就是随机世界的链式法则，它多出一个神秘的二阶修正项，来自布朗运动"每步都在随机抖动"的本质。
> - **Black-Scholes（12.3）**：有了伊藤引理，就能推导出期权定价的偏微分方程，并神奇地消去股票的真实涨跌速度，只用波动率和无风险利率就能给期权定价——这是整个量化金融的基石，1997年的诺贝尔经济学奖正是颁给了这个公式。
>
> 贯穿全章的直觉：**随机性不是障碍，而是可以被"对冲掉"的结构。**


### 12.1 布朗运动与维纳过程（随机游走的连续极限/路径连续但处处不可导）

#### 💡 大白话唤醒直觉

想象你站在一条长廊的中点，每隔一秒抛一枚硬币：正面向右走一步，反面向左走一步。走了1000步之后你在哪里？这就是**随机游走**（Random Walk）。如果把时间间隔缩到无穷小、步长也同步缩小，随机游走的连续极限就是**布朗运动**（Brownian Motion），也叫**维纳过程**（Wiener Process），用 $W_t$ 表示。

股票价格每秒都在跳动，就像高频版的随机游走。布朗运动是描述这种连续时间随机性的最基础"零部件"。它的路径肉眼看起来连续、平滑，但在数学上**处处不可微**——任意微小时间段内它都在无限震荡，切线斜率不存在。这是它跟普通函数最根本的区别，也是为什么牛顿-莱布尼茨微积分在这里失效、需要"伊藤微积分"的原因。

布朗运动有三个关键性质：①$W_0 = 0$（从原点出发）；②增量独立，即 $W_t - W_s$ 与 $W_s - W_0$ 无关；③增量服从正态分布 $W_t - W_s \sim N(0, t-s)$，方差等于时间长度——这意味着不确定性随时间以**根号**速度增长，而不是线性增长。

在硬件比喻里：布朗运动是随机噪声信号，"方差=时间"就像热噪声的功率谱密度——积分时间越长，噪声累积越多，量级是 $\sqrt{t}$ 而非 $t$。

> 📝 **练一练：** 一个标准布朗运动在 $t=4$ 时刻的值 $W_4$ 服从什么分布？其标准差是多少？若已知 $W_2 = 0.6$，则 $W_4 - W_2$ 的分布是？
>
> **参考答案要点：** $W_4 \sim N(0, 4)$，标准差为 $\sqrt{4}=2$。由增量独立且 $W_4 - W_2 \sim N(0, 4-2)=N(0,2)$，与 $W_2$ 的取值无关（增量独立性）。

---

#### 📐 极简公式推导

**从随机游走到布朗运动（连续极限）：**

将 $[0,T]$ 分成 $n$ 份，每份时间步长 $\Delta t = T/n$，每步增量：

$$\Delta W_i = \varepsilon_i \sqrt{\Delta t}, \quad \varepsilon_i \overset{\text{i.i.d.}}{\sim} N(0,1)$$

整个路径的累积值：

$$W_T = \sum_{i=1}^{n} \Delta W_i = \sum_{i=1}^{n} \varepsilon_i \sqrt{\Delta t}$$

由中心极限定理，当 $n \to \infty$（即 $\Delta t \to 0$）：

$$W_T \sim N\!\left(0,\; \sum_{i=1}^{n} \Delta t\right) = N(0, T)$$

**二次变差（Quadratic Variation）——布朗运动不可导的根源：**

对普通可微函数 $f$，其二次变差 $\sum (\Delta f)^2 \to 0$（高阶小量）。但布朗运动的二次变差：

$$\sum_{i=1}^{n}(\Delta W_i)^2 = \sum_{i=1}^{n} \varepsilon_i^2 \Delta t \xrightarrow{n\to\infty} T$$

即 $[W,W]_T = T$，不趋向零，这正是"$dW^2 = dt$"的严格来源，也是伊藤公式多出二阶项的根本原因。

**标准维纳过程完整定义：** $\{W_t\}_{t\geq 0}$ 满足
1. $W_0 = 0$
2. $t \mapsto W_t$ 路径连续
3. 增量独立：$0 \leq s < t$，$W_t - W_s$ 独立于 $\mathcal{F}_s$
4. $W_t - W_s \sim N(0, t-s)$

> 📝 **练一练：** 证明 $\mathbb{E}[W_t^2] = t$。提示：用 $W_t \sim N(0,t)$ 以及方差的定义。另外，$\mathbb{E}[(W_t - W_s)(W_s - W_0)]$ 等于多少？为什么？
>
> **参考答案要点：** $\mathbb{E}[W_t^2] = \text{Var}(W_t) + (\mathbb{E}[W_t])^2 = t + 0 = t$。第二问：由增量独立性，$(W_t-W_s)$ 与 $(W_s-W_0)$ 独立，均值为零，故乘积期望为 $0 \times 0 = 0$，即布朗运动是鞅（Martingale）。

---

#### 💻 Python 跑直觉（核心）

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

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

# 左图：多条路径展示路径的连续但高度震荡性
ax = axes[0]
for i in range(n_paths):
    increments = np.random.normal(0, np.sqrt(dt), N_steps)   # dW ~ N(0, dt)
    W = np.concatenate([[0], np.cumsum(increments)])          # 累加得到路径
    ax.plot(t, W, color=colors[i], alpha=0.85, linewidth=0.9, label=f"路径{i+1}")

ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
ax.set_xlabel("时间 t（年）")
ax.set_ylabel("W_t")
ax.set_title("布朗运动：5条样本路径")
ax.legend(fontsize=8, loc="upper left")
ax.grid(True, alpha=0.3)

# 右图：末端 W_T 的分布与理论 N(0,T) 对比
ax2 = axes[1]
n_mc = 50000
# 蒙特卡洛模拟5万条路径，取末端值
endpoints = np.sum(np.random.normal(0, np.sqrt(dt), (n_mc, N_steps)), axis=1)
ax2.hist(endpoints, bins=80, density=True, color="#20808D", alpha=0.7, label="模拟 W_T 分布")
x = np.linspace(-3.5, 3.5, 300)
ax2.plot(x, norm.pdf(x, 0, np.sqrt(T)), color="#A84B2F", linewidth=2,
         label=f"理论 N(0,{T})")
ax2.set_xlabel("W_T 值")
ax2.set_ylabel("概率密度")
ax2.set_title(f"T={T}时末端分布：模拟 vs 理论")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figs/fig_12_1.png", dpi=150, bbox_inches="tight")
```

![](figs/fig_12_1.png)

**运行结果说明：** 左图展示5条各不相同的连续随机路径，均从0出发，路径震荡剧烈但不跳断；右图用5万条路径的末端值验证 $W_T \sim N(0,1)$，模拟直方图与理论曲线高度吻合，直观验证"二次变差=时间"的统计含义。

> 📝 **练一练（编程）：** 修改代码，画出二次变差 $\sum_{i=1}^{n}(\Delta W_i)^2$ 随 $n$ 增大（步数从10到10000）的收敛过程，验证它趋向于 $T=1$。
>
> **参考答案要点：** 对每个 $n$ 值，生成 $n$ 步增量 $\Delta W_i \sim N(0, T/n)$，计算 $\sum (\Delta W_i)^2$，绘制折线图，可见随 $n \to \infty$ 该值收敛到 $T=1$，与普通函数（二次变差趋零）形成对比。

---

#### 🚀 金融/AI 业务映射

**衍生品定价**：布朗运动是所有连续时间资产价格模型（GBM、Heston、SABR）的噪声驱动项，期权蒙特卡洛定价的本质就是大量模拟 $W_t$ 路径并折现平均收益。**量化风控/ML**：在线性因子模型中，布朗运动的"增量独立+方差=时间"结构对应残差白噪声假设；违背此假设（如自相关残差）会导致动量因子的风险被低估。

> 📝 **练一练（应用）：** 在蒙特卡洛风控系统中，如果程序员错误地将每步增量设为 $N(0, dt^2)$ 而非 $N(0, dt)$，会导致模拟路径的方差随时间怎样增长？对VaR估计有何影响？
>
> **参考答案要点：** 路径末端方差为 $n \cdot dt^2 = T \cdot dt \to 0$（当 $dt \to 0$），模拟结果几乎不波动，严重低估了真实波动率，导致VaR被严重低估（数倍甚至数量级之差），属于典型的参数化错误。

---

#### 🛡️ 系统级 QA 灾难与防坑指南

**Bug案例1：某券商极速柜台T+0风控系统蒙特卡洛路径生成器方差规格错误（2021年，华东某券商，日均单笔期权对冲规模约2亿元）**

背景：风控系统升级后，蒙特卡洛引擎将原有 `std=np.sqrt(dt)` 被某工程师"优化"为 `std=dt`（误将方差当标准差），导致路径波动率比真实值低了 $\sqrt{dt}^{-1}$ 倍（步数1000步时约31倍）。盘中VaR计算显示持仓风险极低，风控阈值一直未触发，实际敞口远超限额。当天下午标的大幅下跌3.8%，期权对冲严重不足，亏损约1200万。

```python
import numpy as np

def validate_brownian_increments(dW: np.ndarray, dt: float, tol: float = 0.1) -> None:
    """
    QA拦截：验证布朗运动增量的统计特性是否符合 N(0, dt)
    在每次生产上线前、每次参数变更后必须调用此函数。
    
    Args:
        dW: 生成的增量数组 shape=(n_steps,) 或 (n_paths, n_steps)
        dt: 理论上每步的时间步长
        tol: 允许的相对误差容忍度（默认10%）
    """
    dW_flat = dW.flatten()
    n = len(dW_flat)
    
    # 检查均值应接近0
    empirical_mean = np.mean(dW_flat)
    assert abs(empirical_mean) < 5 * np.sqrt(dt / n), (
        f"[布朗增量QA] 均值异常：{empirical_mean:.6f}，期望≈0"
    )
    
    # 检查方差应接近dt（这是最容易出bug的地方！）
    empirical_var = np.var(dW_flat)
    rel_error = abs(empirical_var - dt) / dt
    assert rel_error < tol, (
        f"[布朗增量QA] 方差规格错误！\n"
        f"  实际方差: {empirical_var:.8f}\n"
        f"  期望方差(dt): {dt:.8f}\n"
        f"  相对误差: {rel_error*100:.2f}% > 容忍度{tol*100:.0f}%\n"
        f"  常见原因：std=dt（应为std=sqrt(dt)），或dt单位混淆（日/年）"
    )
    
    # 检查二次变差收敛性（抽样验证）
    if dW.ndim == 2:
        qv_per_path = np.sum(dW**2, axis=1)
        T_total = dt * dW.shape[1]
        qv_mean = np.mean(qv_per_path)
        assert abs(qv_mean - T_total) / T_total < tol, (
            f"[布朗增量QA] 二次变差异常：{qv_mean:.4f}，期望T={T_total:.4f}"
        )
    
    print(f"[布朗增量QA] 通过：方差={empirical_var:.6f}，dt={dt:.6f}，相对误差={rel_error*100:.2f}%")

# ---- 测试：正确用法 ----
dt_test = 1.0 / 252  # 日步长（年化）
dW_correct = np.random.normal(0, np.sqrt(dt_test), (1000, 252))
validate_brownian_increments(dW_correct, dt_test)

# ---- 测试：错误用法（会触发断言）----
try:
    dW_wrong = np.random.normal(0, dt_test, (1000, 252))   # 用dt而非sqrt(dt)
    validate_brownian_increments(dW_wrong, dt_test)
except AssertionError as e:
    print(f"[拦截成功] {e}")
```

**Bug案例2：某基金公司历史回测引擎"未来函数"穿越——布朗运动路径使用了全局随机种子（2022年，某头部公募，策略年化收益从回测28%缩水至实盘4%）**

背景：回测引擎模拟布朗运动路径时，工程师为"可复现性"在每次循环开始时重设 `np.random.seed(42)`，导致每一天的随机路径完全相同。策略优化器实际上拟合了这条固定路径的特有模式，等同于用未来信息选参数。上线后路径随机性恢复，策略完全失效。

```python
def validate_no_seed_leakage(path_generator_fn, n_runs: int = 10) -> None:
    """
    QA拦截：验证路径生成器不存在固定种子导致的路径重复问题
    检测手段：多次独立调用，路径应不同
    """
    paths = [path_generator_fn() for _ in range(n_runs)]
    # 检查是否所有路径完全相同（固定种子的典型症状）
    all_same = all(np.allclose(paths[0], p) for p in paths[1:])
    assert not all_same, (
        "[路径生成器QA] 检测到固定种子漏洞：连续生成的路径完全相同！\n"
        "  请检查生成器内部是否调用了 np.random.seed() 或 random.seed()"
    )
    
    # 进一步检查：相邻路径之间应有显著差异
    diffs = [np.std(paths[i] - paths[i-1]) for i in range(1, n_runs)]
    assert np.mean(diffs) > 1e-6, (
        f"[路径生成器QA] 路径差异异常小（平均std={np.mean(diffs):.2e}），疑似种子固定"
    )
    print(f"[路径生成器QA] 通过：路径差异均值std={np.mean(diffs):.4f}")

# 有问题的生成器（在内部重设种子）
def bad_generator():
    np.random.seed(42)   # 每次重设！这是bug
    return np.cumsum(np.random.normal(0, 0.01, 252))

# 正确的生成器
def good_generator():
    return np.cumsum(np.random.normal(0, 0.01, 252))

print("测试有问题的生成器：")
try:
    validate_no_seed_leakage(bad_generator)
except AssertionError as e:
    print(f"[拦截成功] {e}")

print("\n测试正确的生成器：")
validate_no_seed_leakage(good_generator)
```

> 📝 **练一练（测试设计）：** 在上述 `validate_brownian_increments` 中，为何要用 $5\sqrt{dt/n}$ 而不是固定阈值（如0.001）来检验均值？当 $n=100$ 步和 $n=1000000$ 步时，这个动态阈值分别是多少（$dt=1/252$）？
>
> **参考答案要点：** 因为样本均值的标准误差是 $\sigma/\sqrt{n} = \sqrt{dt/n}$，固定阈值会导致小样本（低步数）误报、大样本（高步数）漏报。$n=100$ 时阈值约 $5\sqrt{1/252/100} \approx 0.031$；$n=10^6$ 时约 $0.0031$，自动适应样本量。

---

### 12.2 伊藤引理（带随机项的泰勒展开/dW² = dt 的二次变差直觉）

#### 💡 大白话唤醒直觉

普通微积分中，如果 $x(t)$ 是普通光滑函数，想算 $f(x(t))$ 的变化率，只需链式法则：$df = f'(x)\,dx$。但如果 $x$ 是布朗运动，这就失效了——因为布朗运动"不可微"，而且 $(dx)^2 \sim dt$（二次变差不为零），普通链式法则丢掉了这个二阶项。

**伊藤引理**（Itô's Lemma）就是"随机版的链式法则"。比喻：普通微积分好比在高速公路上开车，油门一踩速度线性增加；布朗运动好比在随机颠簸的越野路上开车，颠簸本身也会推着你的速度变化，必须额外考虑"颠簸的平方效应"。

最典型的应用：股票 $S_t$ 满足几何布朗运动 $dS = \mu S\,dt + \sigma S\,dW$。如果要算对数股价 $\ln S_t$ 的动态，用普通链式法则会得到 $d\ln S = \mu\,dt + \sigma\,dW$，但这是错的。伊藤引理会多出一项 $-\frac{1}{2}\sigma^2\,dt$（源于 $dW^2=dt$），正确结果是 $d\ln S = (\mu - \frac{1}{2}\sigma^2)\,dt + \sigma\,dW$。这个 $-\frac{1}{2}\sigma^2$ 就是著名的"伊藤修正项"，在Black-Scholes定价中至关重要。

> 📝 **练一练：** 不用推导，直接从直觉解释：为什么伊藤修正项是负的（$-\frac{1}{2}\sigma^2 dt$）？它意味着对数收益的期望比算术收益 $\mu$ 要小，这在投资中叫什么现象？
>
> **参考答案要点：** 因为对数函数是凹函数（$\ln'' < 0$），Jensen不等式说明 $\mathbb{E}[\ln X] \leq \ln \mathbb{E}[X]$，波动越大差距越大。投资中叫**波动率拖累**（Volatility Drag）——一只股票涨50%再跌33.3%净值不变，但算术平均收益是 $(50\%-33.3\%)/2=8.35\%$，实际复利收益为零。

---

#### 📐 极简公式推导

设随机过程 $X_t$ 满足伊藤过程：

$$dX = a(X,t)\,dt + b(X,t)\,dW_t$$

对 $f(X_t, t)$ 作泰勒展开到二阶：

$$df = \frac{\partial f}{\partial t}\,dt + \frac{\partial f}{\partial x}\,dX + \frac{1}{2}\frac{\partial^2 f}{\partial x^2}\,(dX)^2 + \cdots$$

计算 $(dX)^2$，利用随机微分的乘法表 $dt \cdot dt = 0$，$dt \cdot dW = 0$，$dW \cdot dW = dt$：

$$(dX)^2 = a^2\,dt^2 + 2ab\,dt\,dW + b^2\,dW^2 = b^2\,dt + o(dt)$$

代入得**伊藤引理（一维版本）**：

$$\boxed{df = \left(\frac{\partial f}{\partial t} + a\frac{\partial f}{\partial x} + \frac{1}{2}b^2\frac{\partial^2 f}{\partial x^2}\right)dt + b\frac{\partial f}{\partial x}\,dW_t}$$

**应用：几何布朗运动的精确解**

取 $f(S,t) = \ln S$，$a = \mu S$，$b = \sigma S$，则：

$$d\ln S = \left(\mu - \frac{1}{2}\sigma^2\right)dt + \sigma\,dW_t$$

积分得精确解：

$$\boxed{S_t = S_0 \exp\!\left[\left(\mu - \frac{1}{2}\sigma^2\right)t + \sigma W_t\right]}$$

> 📝 **练一练：** 设 $f(S,t) = S^2$，$S$ 满足 $dS = \mu S\,dt + \sigma S\,dW$，用伊藤引理推导 $d(S^2)$ 的表达式。提示：$\partial f/\partial S = 2S$，$\partial^2 f/\partial S^2 = 2$。
>
> **参考答案要点：** $d(S^2) = 2S\,dS + \frac{1}{2}\cdot 2\cdot(\sigma S)^2 dt = 2S(\mu S\,dt + \sigma S\,dW) + \sigma^2 S^2\,dt = (2\mu + \sigma^2)S^2\,dt + 2\sigma S^2\,dW$。注意相比确定性情形多了 $\sigma^2 S^2\,dt$ 这个伊藤修正项。

---

#### 💻 Python 跑直觉（核心）

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

np.random.seed(0)

# ---- GBM参数 ----
S0 = 100.0       # 初始股价
mu = 0.10        # 漂移率（年化）
sigma = 0.25     # 波动率（年化）
T = 1.0          # 1年
N = 1000         # 步数
dt = T / N
t = np.linspace(0, T, N + 1)

n_paths = 6
colors = ["#20808D", "#A84B2F", "#1B474D", "#944454", "#FFC553", "#6E522B"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ---- 左图：伊藤引理精确解 vs 欧拉-丸山离散化 ----
# 精确解：S(t) = S0 * exp((mu - sigma^2/2)*t + sigma*W_t)
ax = axes[0]
for i in range(n_paths):
    dW = np.random.normal(0, np.sqrt(dt), N)
    W = np.concatenate([[0], np.cumsum(dW)])
    # 伊藤引理精确解（注意有 -sigma^2/2 修正项！）
    S_exact = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)
    ax.plot(t, S_exact, color=colors[i], alpha=0.8, linewidth=0.9)

ax.set_xlabel("时间 t（年）")
ax.set_ylabel("股价 S_t")
ax.set_title("几何布朗运动（GBM）路径\ndS = mu*S*dt + sigma*S*dW")
ax.axhline(S0, color="gray", linestyle="--", linewidth=0.7, label=f"S_0={S0}")
ax.legend()
ax.grid(True, alpha=0.3)

# ---- 右图：不同dt步长下欧拉离散化的定价偏差 ----
# 欧拉-丸山：dS = mu*S*dt + sigma*S*dW（非精确解，存在离散化误差）
ax2 = axes[1]
dt_list = [0.01, 0.05, 0.1, 0.25, 0.5]
n_mc = 10000
S0_test, mu_test, sigma_test, T_test = 100, 0.1, 0.2, 1.0
exact_mean = S0_test * np.exp(mu_test * T_test)  # GBM期望值理论值

errors = []
for dt_i in dt_list:
    N_i = int(T_test / dt_i)
    S_ends = np.full(n_mc, float(S0_test))
    for _ in range(N_i):
        dW_i = np.random.normal(0, np.sqrt(dt_i), n_mc)
        # 欧拉离散：用精确解形式（等价于对数欧拉）
        S_ends = S_ends * np.exp((mu_test - 0.5 * sigma_test**2) * dt_i + sigma_test * dW_i)
    errors.append(abs(np.mean(S_ends) - exact_mean))

ax2.bar([str(d) for d in dt_list], errors, color="#20808D", alpha=0.8, edgecolor="white")
ax2.set_xlabel("离散化步长 dt")
ax2.set_ylabel("|模拟均值 - 理论均值|")
ax2.set_title("伊藤离散化步长 vs 定价偏差\n精确解形式误差较小，大dt仍有偏")
ax2.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("figs/fig_12_2.png", dpi=150, bbox_inches="tight")
```

![](figs/fig_12_2.png)

**运行结果说明：** 左图展示6条GBM路径，体现伊藤修正项 $-\frac{1}{2}\sigma^2 t$ 对长期漂移的压制（长期期望是 $S_0 e^{\mu t}$ 而非 $S_0 e^{(\mu+\sigma^2/2)t}$）；右图对比不同步长的离散化误差，步长越大（$dt=0.5$时T内只有2步）蒙特卡洛误差越大。

> 📝 **练一练（编程）：** 用相同的一组布朗运动路径，分别用"正确的伊藤精确解"（含 $-\frac{1}{2}\sigma^2$ 修正）和"错误的朴素公式"（不含修正项，$S_t = S_0 e^{\mu t + \sigma W_t}$）计算末端股价均值。通过蒙特卡洛（5万路径）验证：朴素公式的均值是多少？正确公式的均值是多少？理论值是多少？
>
> **参考答案要点：** 正确精确解期望 $\approx S_0 e^{\mu T}$（理论值），朴素公式期望 $\approx S_0 e^{(\mu+\sigma^2/2)T}$，当 $\sigma=0.3, T=1$ 时朴素公式高估约 $e^{0.045}-1 \approx 4.6\%$。这解释了为什么期权定价必须用伊藤引理推导的精确解，否则系统性高估股价均值。

---

#### 🚀 金融/AI 业务映射

**衍生品定价**：伊藤引理是推导Black-Scholes偏微分方程的核心工具，所有路径依赖期权（亚式、障碍）的蒙特卡洛定价都依赖精确的GBM路径（含伊藤修正项），缺少 $-\frac{1}{2}\sigma^2$ 项会导致系统性定价偏差。**量化ML**：神经SDE（Neural SDE）在学习金融时间序列的随机动力学时，损失函数中需要正确处理伊藤修正项，否则漂移项和扩散项的参数估计均有偏；对应到深度学习，这类似于批归一化中均值修正的作用。

> 📝 **练一练（应用）：** 某对冲基金的Delta对冲策略每日再平衡，用GBM模拟标的路径来估计对冲误差。若程序员误将对数收益的期望设为 $\mu$ 而非 $\mu - \frac{1}{2}\sigma^2$（$\sigma=0.3$），估计对冲误差的偏差量级（年化）是多少？
>
> **参考答案要点：** 对数收益被高估 $\frac{1}{2}\times 0.3^2 = 4.5\%$ 年化，相当于每年系统性高估股价的增长速度。在Delta对冲中，这导致Delta计算使用了偏高的漂移假设，对大规模组合（如100亿AUM）年化对冲成本误差可达数千万。

---

#### 🛡️ 系统级 QA 灾难与防坑指南

**Bug案例1：某互联网券商自研期权定价引擎伊藤修正项缺失，深度实值期权定价偏高3.2%（2020年，华南某互联网券商，日均期权定价请求约80万次）**

背景：工程师在实现GBM路径生成时，将对数漂移项写为 `drift = mu * dt`，漏掉了 `- 0.5 * sigma**2 * dt`。对于平值短期期权影响不明显，但对于深度实值、长期期权（如1年期、深度价内的认购期权），系统性高估了约3.2%。由于期权Vega较大，实际被对手方套利，两周内亏损约470万。

**Bug案例2：某资管公司欧拉-丸山离散化步长过大，可转债嵌入式期权定价偏差超5%（2023年，某城商行资管部，可转债规模约15亿元）**

背景：蒙特卡洛定价引擎使用月度步长（$dt = 1/12$），而标的波动率高达45%。对于嵌入式百慕大期权，粗糙的离散化导致期权价格比精细步长（$dt=1/252$）低5.3%，决策层误认为可转债"低估"而大量建仓，后续市场修正后亏损。

```python
import numpy as np

def validate_ito_gbm_path(paths: np.ndarray, S0: float, mu: float, sigma: float,
                           T: float, tol_pct: float = 2.0) -> None:
    """
    QA拦截：验证GBM路径是否正确应用了伊藤修正项
    核心检验：路径末端均值应 ≈ S0 * exp(mu * T)，而非 S0 * exp((mu+sigma^2/2)*T)
    
    Args:
        paths: shape=(n_paths, n_steps+1) 的价格路径矩阵
        S0, mu, sigma, T: GBM参数
        tol_pct: 允许的相对误差百分比（默认2%）
    """
    n_paths = paths.shape[0]
    empirical_mean = np.mean(paths[:, -1])
    
    # 正确的理论期望（来自伊藤引理）
    correct_mean = S0 * np.exp(mu * T)
    # 错误的朴素期望（漏掉伊藤修正）
    wrong_mean = S0 * np.exp((mu + 0.5 * sigma**2) * T)
    
    rel_err_correct = abs(empirical_mean - correct_mean) / correct_mean * 100
    rel_err_wrong   = abs(empirical_mean - wrong_mean) / wrong_mean * 100
    
    # 若模拟均值更接近错误公式，说明伊藤修正项缺失
    assert rel_err_correct < tol_pct, (
        f"[伊藤GBM-QA] 路径均值异常！\n"
        f"  模拟均值: {empirical_mean:.4f}\n"
        f"  正确理论均值 (含伊藤修正): {correct_mean:.4f}（误差{rel_err_correct:.2f}%）\n"
        f"  错误朴素均值 (缺修正项): {wrong_mean:.4f}（误差{rel_err_wrong:.2f}%）\n"
        f"  若误差{rel_err_correct:.2f}% > {tol_pct}%，请检查漂移项是否为 (mu-0.5*sigma^2)*dt"
    )
    
    # 检查路径步长是否合理（大dt会增加离散化偏差）
    n_steps = paths.shape[1] - 1
    dt = T / n_steps
    stability_cond = sigma * np.sqrt(dt)   # 无因次波动参数
    assert stability_cond < 0.5, (
        f"[伊藤GBM-QA] 离散化步长过大！\n"
        f"  sigma*sqrt(dt) = {stability_cond:.3f} > 0.5\n"
        f"  建议 n_steps >= {int((sigma / 0.5)**2 * T + 1)}（当前{n_steps}步）\n"
        f"  大步长会导致欧拉-丸山误差累积，建议改用精确解形式"
    )
    
    print(f"[伊藤GBM-QA] 通过：模拟均值={empirical_mean:.4f}，理论均值={correct_mean:.4f}，"
          f"误差={rel_err_correct:.2f}%")

# ---- 生成正确路径 ----
np.random.seed(42)
S0, mu, sigma, T, N = 100, 0.1, 0.25, 1.0, 252
dt = T / N
n_paths = 5000

dW_matrix = np.random.normal(0, np.sqrt(dt), (n_paths, N))
# 正确：含伊藤修正项
log_returns = (mu - 0.5 * sigma**2) * dt + sigma * dW_matrix
correct_paths = np.zeros((n_paths, N + 1))
correct_paths[:, 0] = S0
correct_paths[:, 1:] = S0 * np.exp(np.cumsum(log_returns, axis=1))
validate_ito_gbm_path(correct_paths, S0, mu, sigma, T)

# ---- 错误路径（漏掉伊藤修正）----
wrong_log_returns = mu * dt + sigma * dW_matrix   # 缺少 -0.5*sigma^2*dt
wrong_paths = np.zeros((n_paths, N + 1))
wrong_paths[:, 0] = S0
wrong_paths[:, 1:] = S0 * np.exp(np.cumsum(wrong_log_returns, axis=1))
try:
    validate_ito_gbm_path(wrong_paths, S0, mu, sigma, T)
except AssertionError as e:
    print(f"[拦截成功] {e}")
```

> 📝 **练一练（测试设计）：** 上述QA中 `stability_cond = sigma * sqrt(dt) < 0.5` 是什么意思？当 $\sigma=0.8$（高波动股票）时，需要多少步才能满足这个条件（$T=1$ 年）？这和数值PDE求解的CFL条件有何相似之处？
>
> **参考答案要点：** 这是欧拉-丸山方法的稳定性度量，$\sigma\sqrt{dt}$ 表示每步随机项的量级，超过0.5意味着随机扰动可能使价格跳为负值。$\sigma=0.8$ 时需要 $n \geq (0.8/0.5)^2 = 2.56$，即至少 $\lceil 2.56 \rceil = 3$ 步，但实际应取 $n \geq 252$（日步长）。类比CFL条件：都是要求数值格式中"信息传播速度"不超过格子分辨率，违反则数值解发散。

---

### 12.3 Black-Scholes方程与期权定价（金融数学的皇冠/风险中性定价）

#### 💡 大白话唤醒直觉

你买了一份期权：一个月后有权以100元买入某股票（行权价K=100），无论股票当时是50还是200元。这份权利现在值多少钱？

1973年，Fischer Black和Myron Scholes给出了一个完美的公式。他们的核心思路叫**Delta对冲**：假设你是期权的卖方，手头持有一些股票，动态调整仓位使得"股票的涨跌"恰好抵消"期权价值的变化"，这个组合就**无风险了**。无风险资产的收益必须等于无风险利率，从这个约束出发，推导出期权必须满足的偏微分方程——**Black-Scholes方程**。

最神奇的一步是**风险中性定价**：在对冲的世界里，股票的期望收益率 $\mu$ 神奇地消失了，替换为无风险利率 $r$。这意味着定期权价格时，可以"假装"所有投资者都不在乎风险，股票以 $r$ 的速度增长。这不是真实世界的假设，而是一个数学技巧，叫**等价鞅测度变换**（Girsanov定理）。

用工程比喻：Black-Scholes就是一个完美的"无损压缩算法"——把随机路径的不确定性，通过动态Delta对冲全部"对消"掉，剩下一个确定性的定价公式。

> 📝 **练一练：** 为什么Black-Scholes公式中股票的真实漂移率 $\mu$ 不出现？直觉上，一个激进的投资者（高 $\mu$）和保守投资者（低 $\mu$）对同一份期权的估价应该相同吗？
>
> **参考答案要点：** 因为 $\mu$ 在Delta对冲中被完全对消——无论股票未来涨跌，对冲组合的变动方向已知，只剩下时间衰减和波动率效应。激进和保守的投资者对期权的"公平价格"相同（在完备市场假设下），不同的只是他们是否愿意买卖这个价格，即风险偏好被从定价中剔除了。

---

#### 📐 极简公式推导

**Step 1：构造无风险对冲组合**

设期权价格 $V(S,t)$，持有 $-1$ 份期权 $+\Delta = \partial V/\partial S$ 份股票，组合价值：

$$\Pi = -V + \Delta S$$

**Step 2：伊藤引理给出 $dV$**

$$dV = \left(\frac{\partial V}{\partial t} + \mu S\frac{\partial V}{\partial S} + \frac{1}{2}\sigma^2 S^2\frac{\partial^2 V}{\partial S^2}\right)dt + \sigma S\frac{\partial V}{\partial S}\,dW$$

**Step 3：对冲消除随机项**

取 $\Delta = \partial V/\partial S$，则：

$$d\Pi = -dV + \Delta\,dS = -\left(\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2\frac{\partial^2 V}{\partial S^2}\right)dt$$

**Step 4：无套利原则**（$d\Pi = r\Pi\,dt$）

$$\boxed{\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV = 0}$$

这就是**Black-Scholes偏微分方程**（注意 $\mu$ 消失，换成了 $r$）。

**欧式看涨期权的解析解（Black-Scholes公式）：**

$$\boxed{C = S\,N(d_1) - K e^{-rT} N(d_2)}$$

其中：

$$d_1 = \frac{\ln(S/K) + (r + \frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}$$

$N(\cdot)$ 是标准正态分布函数。看跌期权通过**看跌-看涨平价**（Put-Call Parity）得到：

$$P = C - S + Ke^{-rT}$$

> 📝 **练一练：** 当 $T \to 0$（期权即将到期），$C$ 的极限是什么？写出公式并验证其经济含义（提示：$N(d_1) \to 1$ 当 $S > K$ 时）。
>
> **参考答案要点：** 当 $T\to 0$，若 $S>K$：$d_1 \to +\infty$，$d_2 \to +\infty$，$N(d_1)\to 1$，$N(d_2)\to 1$，$C \to S-K$（内在价值）。若 $S<K$：$d_1,d_2 \to -\infty$，$N \to 0$，$C \to 0$。这正是到期日收益 $\max(S-K,0)$，符合期权定义。

---

#### 💻 Python 跑直觉（核心）

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

def bs_call(S, K, T, r, sigma):
    """Black-Scholes欧式看涨期权价格"""
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_put(S, K, T, r, sigma):
    """Black-Scholes欧式看跌期权价格（用Put-Call Parity）"""
    return bs_call(S, K, T, r, sigma) - S + K * np.exp(-r * T)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# ---- 左图：期权价格 vs 股价 ----
S_range = np.linspace(50, 160, 300)
K, T0, r, sigma0 = 100, 1.0, 0.05, 0.25

call_prices = [bs_call(s, K, T0, r, sigma0) for s in S_range]
put_prices = [bs_put(s, K, T0, r, sigma0) for s in S_range]
intrinsic_call = np.maximum(S_range - K, 0)

ax = axes[0]
ax.plot(S_range, call_prices, color="#20808D", linewidth=2, label="看涨Call")
ax.plot(S_range, put_prices, color="#A84B2F", linewidth=2, label="看跌Put")
ax.plot(S_range, intrinsic_call, color="gray", linewidth=1, linestyle="--", label="内在价值")
ax.axvline(K, color="black", linewidth=0.7, linestyle=":", label=f"K={K}")
ax.set_xlabel("标的股价 S")
ax.set_ylabel("期权价格")
ax.set_title("BS期权价格 vs 股价\nK=100, T=1, r=5%, sigma=25%")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ---- 中图：波动率微笑示意 ----
K_range = np.linspace(70, 130, 50)
S0_smile = 100
# 模拟市场观察到的"偏斜"隐含波动率（现实中OTM put溢价）
smile_sigma = 0.20 + 0.08 * ((K_range - S0_smile) / S0_smile)**2 - 0.03 * (K_range - S0_smile) / S0_smile

ax2 = axes[1]
ax2.plot(K_range, smile_sigma * 100, color="#20808D", linewidth=2)
ax2.axvline(S0_smile, color="gray", linestyle="--", linewidth=0.8, label="平值ATM")
ax2.set_xlabel("行权价 K")
ax2.set_ylabel("隐含波动率 (%)")
ax2.set_title("波动率微笑（Volatility Smile）\n市场对尾部风险的额外定价")
ax2.legend()
ax2.grid(True, alpha=0.3)

# ---- 右图：时间价值衰减（Theta） ----
T_range = np.linspace(0.01, 2.0, 200)
S_atm, K_atm = 100, 100
call_t = [bs_call(S_atm, K_atm, t_i, r, sigma0) for t_i in T_range]
put_t = [bs_put(S_atm, K_atm, t_i, r, sigma0) for t_i in T_range]

ax3 = axes[2]
ax3.plot(T_range, call_t, color="#20808D", linewidth=2, label="ATM看涨Call")
ax3.plot(T_range, put_t, color="#A84B2F", linewidth=2, label="ATM看跌Put")
ax3.set_xlabel("剩余到期时间 T（年）")
ax3.set_ylabel("期权价格")
ax3.set_title("时间价值递减（Theta效应）\n平值期权 S=K=100")
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figs/fig_12_3.png", dpi=150, bbox_inches="tight")
```

![](figs/fig_12_3.png)

**运行结果说明：** 左图展示看涨/看跌期权价格曲线及其内在价值（时间价值=期权价-内在价值，在平值处最大）；中图展示现实市场的波动率微笑，BS公式假设常数波动率但市场对深度价外期权收取更高风险溢价；右图展示Theta效应，平值期权价值随到期时间缩短而快速衰减。

> 📝 **练一练（编程）：** 使用上述 `bs_call` 函数，计算Greeks：Delta（$\partial C/\partial S$，用有限差分近似，$\Delta S = 0.01$）和Gamma（$\partial^2 C/\partial S^2$）。对于 $S=100, K=100, T=1, r=0.05, \sigma=0.25$，计算Delta和Gamma数值，并用解析公式 $\Delta = N(d_1)$ 交叉验证。
>
> **参考答案要点：** 有限差分 `delta = (bs_call(101,K,T,r,s) - bs_call(99,K,T,r,s)) / 2`，结果约0.637；解析Delta $= N(d_1) \approx N(0.35) \approx 0.637$，一致。Gamma $= (C(S+\epsilon)-2C(S)+C(S-\epsilon))/\epsilon^2$，约0.016，表示股价每变动1元，Delta变化0.016。

---


#### 🚀 金融/AI 业务映射

**衍生品风险管理（Greeks工厂）**：Black-Scholes公式是所有交易台期权风险报告（Delta/Gamma/Vega/Theta/Rho）的计算核心，做市商每日根据BS Greeks动态调整对冲仓位，波动率曲面（Vol Surface）是对BS公式在不同行权价和到期日维度的隐含波动率插值结果，是整个期权市场的"价格地图"。**量化ML/AI定价**：神经网络期权定价模型（如Deep Hedging）本质上是用神经网络逼近BS公式的非线性映射 $(S, K, T, r, \sigma) 	o C$，训练数据来自BS公式的大规模蒙特卡洛采样；在更复杂的随机波动率模型（Heston、SABR）下，神经网络作为"快速定价代理"替代耗时的PDE数值求解，BS公式则作为基准校验其收敛性。

> 📝 **练一练（应用）：** 某做市商对一份平值看涨期权做Delta对冲：买入1份期权，同时卖出 $\Delta = N(d_1) \approx 0.637$ 份股票。次日股价从100涨到102，请估算：①期权价值变化约多少（用Delta近似）？②对冲股票仓位的盈亏是多少？③组合净PnL理论上应接近零（Delta对冲的目的），实际还剩余多少（提示：用Gamma修正 $\frac{1}{2}\Gamma(\Delta S)^2$）？
>
> **参考答案要点：** ①期权价值变化 $\approx \Delta \times \Delta S = 0.637 \times 2 = 1.274$ 元；②对冲股票亏损 $= -0.637 \times 2 = -1.274$ 元；③一阶对冲后组合PnL≈0，但Gamma效应产生残余盈利：$\frac{1}{2} \times 0.016 \times 2^2 = 0.032$ 元（正Gamma使期权买方受益），这正是期权买方支付Theta（时间成本）换取的Gamma收益——Delta对冲无法消除Gamma风险，需要Gamma对冲（用其他期权）。


---

#### 🛡️ 系统级 QA 灾难与防坑指南

**Bug案例1：某期货交易所自研期权风控系统深度价外期权浮点下溢，风控失效（2022年，某商品期货交易所，单日期权合约规模约45亿元）**

背景：某商品期权深度价外（$S/K = 0.3$，即股价仅为行权价30%），计算 $d_1 = \frac{\ln(0.3) + \cdots}{\sigma\sqrt{T}}$，其中 $\ln(0.3) \approx -1.20$，$d_1 \approx -6.8$，$N(d_1) \approx 5.0 \times 10^{-12}$，$K e^{-rT} N(d_2)$ 中 $N(d_2) \approx 0$。Python `float64` 对这个量级能正常表示，但某内部系统使用 `float32`，$N(d_1)$ 被截断为0，期权价格计算为0，风控误认为敞口为零，实际上还有数百万元的名义敞口未被纳入风控视野。

**Bug案例2：某股权衍生品部门BS公式中 $d_1$ 分母当 $T \to 0$ 时除零崩溃（2023年，某头部银行金融衍生品交易台）**

背景：盘中最后一分钟（$T = 1/(365 \times 24 \times 60) \approx 1.9 \times 10^{-6}$），计算欧式期权时 `sigma * sqrt(T)` 变为 `4.3e-4`，因行情系统延迟，下一帧 $T=0$ 时分母为零，Python抛出 `ZeroDivisionError`，风控API 500报错，期权价格计算队列阻塞约47秒，约18个对冲指令无法发出，敞口累积。

```python
import numpy as np
from scipy.stats import norm

def bs_call_safe(S: float, K: float, T: float, r: float, sigma: float,
                 min_T: float = 1e-6, float_precision: str = "float64") -> dict:
    """
    生产级Black-Scholes计算，含完整QA防护
    
    Args:
        S: 标的价格（必须>0）
        K: 行权价（必须>0）
        T: 到期时间（年，必须>=0）
        r: 无风险利率
        sigma: 波动率（必须>0）
        min_T: 临近到期的时间下限（防除零）
        float_precision: 精度类型检查
    Returns:
        dict包含price, d1, d2, delta, diagnostics
    """
    # ---- 输入防护 ----
    assert S > 0, f"[BS-QA] 标的价格S必须>0，当前: {S}"
    assert K > 0, f"[BS-QA] 行权价K必须>0，当前: {K}"
    assert T >= 0, f"[BS-QA] 到期时间T必须>=0，当前: {T}"
    assert sigma > 0, f"[BS-QA] 波动率sigma必须>0，当前: {sigma}"
    assert 0 < sigma < 10, f"[BS-QA] 波动率sigma={sigma:.4f}异常（>1000%），可能单位错误"
    
    # ---- 到期处理（防除零） ----
    if T < min_T:
        intrinsic = max(S - K, 0.0)
        return {"price": intrinsic, "d1": np.nan, "d2": np.nan,
                "delta": 1.0 if S > K else 0.0,
                "diagnostics": f"已到期，返回内在价值 {intrinsic:.4f}"}
    
    denom = sigma * np.sqrt(T)
    
    # ---- 极端参数下的数值稳定性检查 ----
    moneyness = np.log(S / K)
    assert np.isfinite(moneyness), f"[BS-QA] log(S/K)非有限值，S={S}, K={K}"
    
    d1 = (moneyness + (r + 0.5 * sigma**2) * T) / denom
    d2 = d1 - denom
    
    # ---- 极端d值警告（深度价内/价外） ----
    if abs(d1) > 8:
        import warnings
        warnings.warn(
            f"[BS-QA] d1={d1:.2f}极端，N(d1)接近0或1，"
            f"请确认使用float64而非float32（float32约在|d|>4时失精）\n"
            f"  S={S}, K={K}, T={T:.4f}, sigma={sigma:.4f}"
        )
    
    # ---- 计算（必须用float64） ----
    S_f64 = np.float64(S)
    price = float(S_f64 * norm.cdf(d1) - np.float64(K) * np.exp(-r * T) * norm.cdf(d2))
    
    # ---- 结果健壮性检查 ----
    intrinsic = max(S - K * np.exp(-r * T), 0.0)
    upper_bound = S
    assert intrinsic <= price + 1e-8, (
        f"[BS-QA] 期权价格{price:.6f} < 内在价值{intrinsic:.6f}，违反无套利下界"
    )
    assert price <= upper_bound + 1e-8, (
        f"[BS-QA] 期权价格{price:.6f} > 标的价格{upper_bound:.6f}，违反无套利上界"
    )
    
    return {
        "price": price,
        "d1": d1,
        "d2": d2,
        "delta": float(norm.cdf(d1)),
        "diagnostics": "OK"
    }

# ---- 测试各种边界情况 ----
# 正常情况
result = bs_call_safe(100, 100, 1.0, 0.05, 0.25)
print(f"平值1年期看涨期权：{result['price']:.4f}，Delta={result['delta']:.4f}")

# 深度价外（验证不会浮点崩溃）
result_otm = bs_call_safe(30, 100, 1.0, 0.05, 0.25)
print(f"深度价外期权（S/K=0.3）：{result_otm['price']:.6f}")

# 临近到期（防除零）
result_expiry = bs_call_safe(101, 100, 1e-7, 0.05, 0.25, min_T=1e-6)
print(f"临近到期（T≈0）：{result_expiry['price']:.4f}，{result_expiry['diagnostics']}")

# 无效输入（会触发断言）
try:
    bs_call_safe(100, 100, 1.0, 0.05, -0.1)
except AssertionError as e:
    print(f"[拦截成功] {e}")
```

> 📝 **练一练（测试设计）：** BS公式的无套利约束是 $\max(Se^{qT} - Ke^{-rT}, 0) \leq C \leq S$（$q$ 为股息率）。设计一个参数化测试，覆盖 $S/K \in \{0.5, 0.7, 1.0, 1.3, 2.0\}$、$T \in \{0.1, 0.5, 2.0\}$ 的完整边界验证矩阵（共15组），并检查所有组合是否满足无套利约束和Put-Call平价。
>
> **参考答案要点：** 使用 `itertools.product` 遍历参数网格，对每组参数断言 `C >= max(S-K*exp(-rT), 0)` 且 `C <= S`，以及 `abs(C - P - S + K*exp(-rT)) < 1e-8`（Put-Call平价）。这15组覆盖了深度价外/平值/深度价内×短/中/长期的核心场景，是期权定价引擎回归测试的最小完备集。

---

## 第13章 资产定价与组合理论

> ### 💡 本章数学直觉导引
>
> 上一章我们建立了"随机世界的数学语言"，解决了单只股票的价格动态和期权定价。本章把视角从**一只股票**扩展到**一篮子资产**，回答投资者最关心的两个问题：
>
> **问题一（13.1-13.2）：我手里有N只股票，怎么分配仓位才能在承担尽量小风险的同时赚尽量多的收益？**
>
> - Markowitz（13.1）给出了数学答案：把"收益-风险"的权衡转化为二次规划，找出那条"有效前沿"曲线——在每个风险水平上，没有任何组合能比它收益更高。关键武器是**相关性**：两只都会涨跌的股票，若它们的涨跌方向相反，组合的风险可以大幅低于任何单只股票。
> - CAPM（13.2）进一步在均衡市场中回答：一只股票"理论上应该"给多少回报？答案取决于它的**Beta**——即它的涨跌与大盘的连动程度，而非它自身有多大波动。跟大盘高度同步的股票（高Beta）要求更高的风险补偿，能分散掉的个股风险不计报酬。
>
> **问题二（13.3）：我买了债券，利率一变，它会损失多少？**
>
> - 债券是"现金流折现机器"，利率是折现率。**久期**就是这台机器对利率变化的灵敏度——久期8意味着利率每涨1%，债券跌约8%。**凸性**是灵敏度本身的变化率（二阶项），让债券在大幅利率波动时"跌得比预期少、涨得比预期多"，是债券里藏的免费保险。
>
> 贯穿全章的直觉：**分散化能消除个体风险，但系统性风险无处遁形——Beta和久期就是衡量你无法逃脱的那部分风险的标尺。**


### 13.1 均值-方差优化与有效前沿（Markowitz组合理论/二次规划）

#### 💡 大白话唤醒直觉

炒股的人都知道"不要把鸡蛋放在同一个篮子里"。但**放几个篮子**？**每个篮子放多少**？Markowitz在1952年用数学给出了答案——**均值-方差优化**。

把每只股票看成一个"收益-风险"的点：横轴是波动率（风险），纵轴是期望收益。你有10只股票可选，通过不同的权重组合，可以生成无数个"组合点"，这些点构成一片云状区域。云的**左上边界**——在同等风险下收益最高，或同等收益下风险最低——就是**有效前沿**（Efficient Frontier）。

关键洞察：**相关性**是魔法所在。两只股票都有20%波动率，若它们完全负相关（一涨一跌），各持50%的组合波动率可以降到**接近零**，而收益却是两者平均。这就是分散化的威力——通过相关性而非简单平均来降低风险。

数学上，找有效前沿是一个**二次规划**（QP）问题：目标是最小化组合方差 $w^T\Sigma w$，约束条件是预设收益 $w^T\mu = \mu^*$ 且权重之和为1。就像工程中的约束优化，用拉格朗日乘子法或数值QP求解器搞定。

> 📝 **练一练：** 两只资产期望收益分别为10%和15%，波动率均为20%，相关系数为 -0.5。若各持50%，组合期望收益是多少？组合波动率是多少？（提示：$\sigma_p^2 = w_1^2\sigma_1^2 + w_2^2\sigma_2^2 + 2w_1 w_2\rho\sigma_1\sigma_2$）
>
> **参考答案要点：** 期望收益 $= 0.5 \times 10\% + 0.5 \times 15\% = 12.5\%$。方差 $= 0.25 \times 0.04 + 0.25 \times 0.04 + 2 \times 0.25 \times (-0.5) \times 0.2 \times 0.2 = 0.02 - 0.01 = 0.01$，波动率 $= \sqrt{0.01} = 10\%$，远低于两只单独股票的20%，展示了负相关的分散化威力。

---

#### 📐 极简公式推导

设 $n$ 只资产，权重向量 $w \in \mathbb{R}^n$，期望收益向量 $\mu$，协方差矩阵 $\Sigma$（半正定）。

**组合的期望收益与方差：**

$$\mu_p = w^T \mu, \quad \sigma_p^2 = w^T \Sigma w$$

**有效前沿的二次规划（最小化方差）：**

$$\min_w \quad \frac{1}{2} w^T \Sigma w \quad \text{s.t.} \quad w^T \mu = \mu^*, \quad w^T \mathbf{1} = 1, \quad w \geq 0$$

**拉格朗日方法（无卖空限制时的解析解）：**

$$\mathcal{L} = \frac{1}{2}w^T\Sigma w - \lambda(w^T\mu - \mu^*) - \gamma(w^T\mathbf{1} - 1)$$

一阶条件：$\Sigma w = \lambda\mu + \gamma\mathbf{1}$，解出 $w^* = \lambda\Sigma^{-1}\mu + \gamma\Sigma^{-1}\mathbf{1}$

代入约束求出 $\lambda, \gamma$，得到有效前沿上的最优权重。

**最大夏普比率组合（切点组合）：** 无风险利率为 $r_f$ 时，

$$w_{\text{tan}} = \frac{\Sigma^{-1}(\mu - r_f\mathbf{1})}{{\mathbf{1}^T\Sigma^{-1}(\mu - r_f\mathbf{1})}}$$

> 📝 **练一练：** 为什么协方差矩阵 $\Sigma$ 必须是**正定的**（而非仅半正定）才能求逆并得到唯一解？如果组合里有两只完全正相关（$\rho=1$）的股票，会发生什么？
>
> **参考答案要点：** $\rho=1$ 时两只股票的收益完全线性相关，协方差矩阵秩亏损（奇异），不可逆，导致有无穷多等效的权重分配（只关心总仓位不关心个股拆分），优化问题无唯一解。实际系统中需要检测并处理这种情况（如Tikhonov正则化），否则求解器会报错或返回数值垃圾。

---

#### 💻 Python 跑直觉（核心）

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

np.random.seed(7)

# ---- 6只虚构资产（沪深市场风格）----
mu_assets = np.array([0.12, 0.09, 0.15, 0.07, 0.18, 0.10])   # 年化期望收益
sigma_assets = np.array([0.22, 0.15, 0.30, 0.10, 0.35, 0.18]) # 年化波动率

# 相关矩阵（手工设置，保证正定对称）
corr = np.array([
    [1.00, 0.35, 0.55, 0.10, 0.40, 0.30],
    [0.35, 1.00, 0.25, 0.20, 0.20, 0.45],
    [0.55, 0.25, 1.00, 0.05, 0.60, 0.25],
    [0.10, 0.20, 0.05, 1.00, 0.10, 0.15],
    [0.40, 0.20, 0.60, 0.10, 1.00, 0.20],
    [0.30, 0.45, 0.25, 0.15, 0.20, 1.00],
])
cov = np.outer(sigma_assets, sigma_assets) * corr  # 协方差矩阵
n = len(mu_assets)

# ---- 蒙特卡洛随机组合散点 ----
n_random = 5000
rand_weights = np.random.dirichlet(np.ones(n), n_random)  # Dirichlet保证权重之和=1且>=0
rand_ret = rand_weights @ mu_assets
rand_vol = np.sqrt(np.einsum('ij,jk,ik->i', rand_weights, cov, rand_weights))
rand_sharpe = (rand_ret - 0.03) / rand_vol   # 夏普比率（无风险率3%）

# ---- 数值求解有效前沿 ----
target_rets = np.linspace(mu_assets.min(), mu_assets.max(), 60)
eff_vols, eff_weights_list = [], []

for target in target_rets:
    def portfolio_vol(w):
        return np.sqrt(w @ cov @ w)
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'eq', 'fun': lambda w, t=target: w @ mu_assets - t}
    ]
    bounds = [(0, 1)] * n    # 禁止卖空
    res = minimize(portfolio_vol, np.ones(n)/n, method='SLSQP',
                   bounds=bounds, constraints=constraints,
                   options={'ftol': 1e-9, 'maxiter': 1000})
    if res.success:
        eff_vols.append(res.fun)
        eff_weights_list.append(res.x)
    else:
        eff_vols.append(np.nan)

eff_vols = np.array(eff_vols)
valid = ~np.isnan(eff_vols)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 左图：有效前沿
ax = axes[0]
sc = ax.scatter(rand_vol * 100, rand_ret * 100, c=rand_sharpe, cmap="RdYlGn",
                alpha=0.4, s=5, label="随机组合")
ax.plot(eff_vols[valid] * 100, target_rets[valid] * 100,
        color="#20808D", linewidth=2.5, label="有效前沿")
labels = ["沪深A", "消费B", "科技C", "债券D", "创业E", "混合F"]
for i, (lbl, mu_i, sig_i) in enumerate(zip(labels, mu_assets, sigma_assets)):
    ax.scatter(sig_i*100, mu_i*100, s=60, zorder=5, color="#A84B2F")
    ax.annotate(lbl, (sig_i*100+0.3, mu_i*100), fontsize=8)
plt.colorbar(sc, ax=ax, label="夏普比率")
ax.set_xlabel("年化波动率 (%)"); ax.set_ylabel("年化期望收益 (%)")
ax.set_title("均值-方差有效前沿\n（6只虚构资产，禁止卖空）")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# 右图：最大夏普比组合权重
rf = 0.03
sharpe_eff = (target_rets[valid] - rf) / eff_vols[valid]
best_idx = np.argmax(sharpe_eff)
best_weights = eff_weights_list[np.where(valid)[0][best_idx]]
colors_pie = ["#20808D", "#A84B2F", "#1B474D", "#944454", "#FFC553", "#6E522B"]
ax2 = axes[1]
ax2.pie(best_weights, labels=labels, autopct='%1.1f%%',
        colors=colors_pie, startangle=90, textprops={'fontsize': 9})
ax2.set_title(f"最大夏普比组合权重\n夏普={sharpe_eff[best_idx]:.2f}")

plt.tight_layout()
plt.savefig("figs/fig_13_1.png", dpi=150, bbox_inches="tight")
```

![](figs/fig_13_1.png)

**运行结果说明：** 左图显示5000个随机组合构成的抛物线型可行域（Markowitz Bullet），有效前沿是左上边界，夏普比率最高的组合（绿色）明显优于单一股票；右图展示最优组合大量配置低波动的债券D和稳定消费B，而高波动的创业E权重极低，体现了协方差优化的"风险分散+收益追求"平衡。

> 📝 **练一练（编程）：** 在上述代码基础上，计算"全局最小方差组合"（GMV，不设收益约束，只最小化波动率）的权重和波动率，并在有效前沿图上用特殊标记标出GMV组合点。GMV组合的波动率与任何单一资产的最低波动率相比如何？
>
> **参考答案要点：** 去掉收益约束，只保留 $\sum w=1$ 约束运行优化，得到GMV组合的波动率通常低于所有单一资产（债券D单独波动率10%，GMV可能降到约8%-9%），体现相关性不完全时分散化的额外收益。

---

#### 🚀 金融/AI 业务映射

**量化基金**：Markowitz有效前沿是所有多因子组合优化（Black-Litterman、风险平价）的基础框架，协方差矩阵的估计质量直接决定组合表现，工业界用收缩估计（Ledoit-Wolf）来稳定小样本下的协方差矩阵。**ML/AI系统**：多目标神经网络训练（同时优化准确率与推理速度）在超参数空间的帕累托前沿，与Markowitz有效前沿在数学结构上完全同构——都是约束下的多目标权衡。

> 📝 **练一练（应用）：** 某量化基金经理发现有效前沿上的最优组合集中持有3-4只股票，其他均为零权重，质疑是否足够分散化。这种"稀疏"现象的原因是什么？实践中如何修正？
>
> **参考答案要点：** 这是Markowitz优化的"角解"特性——QP求解在顶点处找到最优，协方差矩阵估计误差会放大，导致某些股票被极端超配或排除（"误差最大化器"问题）。修正方法：1）加入L1/L2正则化约束（类似Ridge/Lasso）；2）使用Black-Litterman引入观点先验；3）设置个股权重上限（约束卡纸）；4）稳健协方差估计（Ledoit-Wolf收缩）。

---

#### 🛡️ 系统级 QA 灾难与防坑指南

**Bug案例1：某公募基金风控系统协方差矩阵非正定导致优化崩溃（2021年，某百亿规模量化基金，组合包含约200只股票）**

背景：收益率数据清洗时删除了部分停牌期数据，导致各股票的时间序列长度不一致（有的80天，有的220天）。用样本协方差矩阵 `np.cov(returns_matrix)` 计算时，矩阵秩不足（200维但有效样本仅80个），输出矩阵半正定但存在近零特征值。`scipy.optimize.minimize` 内部Cholesky分解失败，抛出 `LinAlgError: Matrix is not positive definite`，组合优化任务崩溃，持仓未更新，次日持有旧仓位，恰逢行业轮动，亏损约230万。

**Bug案例2：某资管公司协方差矩阵单位混淆（月收益率协方差当成年化处理，风险低估约12倍）（2022年，某城商行理财子公司）**

背景：数据来源为月度收益率，计算协方差矩阵后直接输入优化器，未乘以12（月度→年化换算）。有效前沿上的"最优组合"波动率计算结果仅3.2%，实际年化波动率约11%。管理层基于3.2%的错误风险评估批准了过高杠杆，三个月后净值回撤超过产品预警线，客户集中赎回。

```python
import numpy as np

def validate_covariance_matrix(cov: np.ndarray, asset_names: list = None,
                                 min_eigenvalue: float = 1e-8) -> None:
    """
    QA拦截：生产级协方差矩阵完整性验证
    在每次重新估计协方差矩阵后必须调用（至少日频调用）
    
    Args:
        cov: 协方差矩阵，shape=(n,n)
        asset_names: 资产名称（可选，用于错误定位）
        min_eigenvalue: 最小允许特征值（判断正定性）
    """
    n = cov.shape[0]
    if asset_names is None:
        asset_names = [f"Asset{i}" for i in range(n)]
    
    # 检查方阵
    assert cov.shape == (n, n), f"[COV-QA] 协方差矩阵不是方阵：{cov.shape}"
    
    # 检查对称性
    sym_err = np.max(np.abs(cov - cov.T))
    assert sym_err < 1e-8, f"[COV-QA] 协方差矩阵不对称，最大非对称误差={sym_err:.2e}"
    
    # 检查对角线（方差）必须为正
    neg_var_idx = np.where(np.diag(cov) <= 0)[0]
    assert len(neg_var_idx) == 0, (
        f"[COV-QA] 存在非正方差：资产 {[asset_names[i] for i in neg_var_idx]}"
    )
    
    # 检查正定性（最小特征值）
    eigenvalues = np.linalg.eigvalsh(cov)  # Hermitian矩阵专用，更稳定
    min_eig = eigenvalues.min()
    if min_eig < min_eigenvalue:
        # 自动尝试Ledoit-Wolf修正
        eps = max(-min_eig + min_eigenvalue, 1e-6)
        cov_regularized = cov + eps * np.eye(n)
        print(
            f"[COV-QA] 警告：最小特征值={min_eig:.2e} < {min_eigenvalue:.2e}\n"
            f"  矩阵非正定（可能原因：股票数量>样本量、停牌数据不一致、完全共线性）\n"
            f"  建议：Tikhonov正则化（已添加扰动eps={eps:.2e}），或改用Ledoit-Wolf收缩估计\n"
            f"  请检查样本量是否 >= 资产数量的5倍"
        )
        raise AssertionError(
            f"[COV-QA] 协方差矩阵非正定！最小特征值={min_eig:.2e}，"
            f"优化求解器将崩溃。修正方案：cov += {eps:.2e} * np.eye(n)"
        )
    
    # 检查波动率量级（单位合理性：年化波动率应在1%-200%区间）
    vols = np.sqrt(np.diag(cov))
    unreasonable = np.where((vols < 0.001) | (vols > 2.0))[0]
    if len(unreasonable) > 0:
        print(
            f"[COV-QA] 警告：以下资产波动率异常（可能单位错误：月→年需×√12，日→年需×√252）\n"
            f"  {[(asset_names[i], f'{vols[i]*100:.2f}%') for i in unreasonable]}"
        )
    
    # 检查相关系数在[-1,1]区间
    D_inv = np.diag(1.0 / vols)
    corr = D_inv @ cov @ D_inv
    corr_off = corr - np.eye(n)
    assert np.max(np.abs(corr_off)) <= 1.0 + 1e-6, (
        f"[COV-QA] 存在相关系数绝对值>1：{np.max(np.abs(corr_off)):.4f}，协方差矩阵有误"
    )
    
    print(f"[COV-QA] 通过：{n}×{n}矩阵，最小特征值={min_eig:.4e}，"
          f"波动率范围=[{vols.min()*100:.1f}%, {vols.max()*100:.1f}%]")

# ---- 测试正定矩阵 ----
cov_good = np.array([[0.04, 0.01], [0.01, 0.09]])
validate_covariance_matrix(cov_good, ["股票A", "股票B"])

# ---- 测试非正定矩阵（秩不足）----
# 两只完全相同的股票（线性相关，矩阵奇异）
cov_bad = np.array([[0.04, 0.04], [0.04, 0.04]])   # 相关=1，秩=1
try:
    validate_covariance_matrix(cov_bad, ["股票X", "股票X_复制"])
except AssertionError as e:
    print(f"[拦截成功] {e}")

# ---- 测试单位异常（月度数据未年化）----
cov_monthly = np.array([[0.0003, 0.00005], [0.00005, 0.0002]])  # 月度协方差
validate_covariance_matrix(cov_monthly, ["基金M", "基金N"])     # 应触发波动率警告
```

> 📝 **练一练（测试设计）：** 上述QA中，当 `min_eigenvalue < 1e-8` 时给出了Tikhonov正则化建议（加 $\epsilon I$）。这种修正的数学含义是什么？它对有效前沿的位置有何影响？给出一个准则：$\epsilon$ 应该取多大才是"可接受的最小扰动"？
>
> **参考答案要点：** 加 $\epsilon I$ 等价于假设每只资产都有一个独立的白噪声成分（方差 $\epsilon$），数学上是收缩到对角矩阵方向。对有效前沿的影响：整体向右移（所有组合波动率小幅增大），高相关组合受影响更大。建议 $\epsilon = \max(0, -\lambda_{\min}) + \delta$，其中 $\delta$ 取为最大特征值的 $10^{-5}$ 倍，保持矩阵结构的同时确保正定。

---

### 13.2 资本资产定价模型CAPM与证券市场线SML（Beta与系统性风险）

#### 💡 大白话唤醒直觉

有了Markowitz的有效前沿，CAPM回答了一个更简单的问题：**在均衡市场中，一只股票应该给投资者多少回报？**

直觉是这样的：每只股票的风险分两类——**系统性风险**（大盘涨跌，无论你持有哪只股票都跑不掉，如经济衰退、政策冲击）和**非系统性风险**（个股特有，如CEO跑路、工厂失火，可以通过分散化消除）。在一个理性的均衡市场里，投资者不会因为承担了**可以分散化的风险**而获得额外回报，因为持有足够多的股票可以消除它。只有承担**系统性风险**，才能获得风险溢价。

**Beta（$\beta$）** 就是衡量一只股票系统性风险的指标：$\beta = 1$ 意味着跟大盘同涨同跌；$\beta = 2$ 意味着大盘涨1%它涨2%，跌时也跌更多；$\beta = 0.5$ 是防御型股票。**证券市场线（SML）** 就是"Beta→预期收益"的线性关系：$E[r_i] = r_f + \beta_i(E[r_m] - r_f)$，在SML上方的股票被低估（有正Alpha），下方的被高估（负Alpha）。

> 📝 **练一练：** 市场期望收益12%，无风险利率3%，某成长股Beta=1.8。根据CAPM，该股票的理论期望收益是多少？若该股票的历史平均收益为25%，其Alpha是多少？这意味着什么？
>
> **参考答案要点：** $E[r] = 3\% + 1.8 \times (12\% - 3\%) = 3\% + 16.2\% = 19.2\%$。Alpha $= 25\% - 19.2\% = 5.8\%$，意味着该股票在扣除系统性风险补偿后还有5.8%的超额收益，可能存在未被市场定价的基本面改善因素，或者历史收益样本偏差（均值估计噪声大）。

---

#### 📐 极简公式推导

**从有效前沿到CAPM——资本市场线（CML）：**

引入无风险资产（$\sigma=0$，收益 $r_f$）后，任意风险资产与无风险资产的混合组合在"均值-标准差"空间中形成射线，斜率最大的那条射线切到有效前沿于**市场组合** $M$，这就是**资本市场线（CML）**：

$$E[r_p] = r_f + \frac{E[r_m]-r_f}{\sigma_m} \cdot \sigma_p$$

**CAPM方程推导——Beta定义：**

对任意资产 $i$ 加入市场组合，最优条件给出其"风险价格"。定义：

$$\beta_i = \frac{\text{Cov}(r_i, r_m)}{\text{Var}(r_m)} = \frac{\sigma_{im}}{\sigma_m^2}$$

**证券市场线（SML）：**

$$\boxed{E[r_i] = r_f + \beta_i \left(E[r_m] - r_f\right)}$$

**Beta的OLS估计**（时序回归）：

$$r_{i,t} - r_f = \alpha_i + \beta_i(r_{m,t} - r_f) + \varepsilon_{i,t}$$

其中 $\alpha_i$（Jensen's Alpha）是超越CAPM解释的超额收益；$\varepsilon$ 是可分散化的残差风险。

> 📝 **练一练：** 证明组合Beta等于各资产Beta的加权平均：$\beta_p = \sum_{i=1}^n w_i \beta_i$。提示：从 $\text{Cov}(\sum w_i r_i, r_m)$ 出发，利用协方差的线性性质。
>
> **参考答案要点：** $\beta_p = \frac{\text{Cov}(r_p, r_m)}{\sigma_m^2} = \frac{\text{Cov}(\sum_i w_i r_i, r_m)}{\sigma_m^2} = \frac{\sum_i w_i \text{Cov}(r_i, r_m)}{\sigma_m^2} = \sum_i w_i \beta_i$，线性性质直接得出。这也意味着对冲系统性风险只需让 $\sum w_i \beta_i = 0$（Beta中性），不需要消除个股风险。

---

#### 💻 Python 跑直觉（核心）

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

np.random.seed(2024)

# ---- 参数设置 ----
rf = 0.03        # 无风险利率
rm = 0.12        # 市场年化期望收益

# 10只虚构股票的Beta（从防御型到高Beta成长股）
betas = np.array([0.3, 0.6, 0.8, 1.0, 1.1, 1.2, 1.4, 1.6, 1.8, 2.1])
stock_names = [f"股{i+1}" for i in range(len(betas))]

# CAPM理论收益
E_r = rf + betas * (rm - rf)
# 加入现实中的Alpha噪声（部分股票偏离SML）
noise = np.array([0.03, -0.02, 0.01, 0.00, -0.015, 0.025, -0.01, 0.02, -0.03, 0.015])
actual_r = E_r + noise

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ---- 左图：证券市场线SML ----
ax = axes[0]
beta_line = np.linspace(0, 2.3, 200)
sml = rf + beta_line * (rm - rf)

ax.plot(beta_line, sml * 100, color="#20808D", linewidth=2, label="证券市场线 SML")
ax.scatter(betas, actual_r * 100, color="#A84B2F", s=60, zorder=5, label="各股票实际收益")

for i, (b, r, name) in enumerate(zip(betas, actual_r, stock_names)):
    ax.annotate(name, (b + 0.02, r * 100 + 0.1), fontsize=8)
    # 画Alpha偏差箭头
    if abs(noise[i]) > 0.005:
        ax.annotate("", xy=(b, r * 100), xytext=(b, E_r[i] * 100),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

ax.axhline(rf * 100, color="gray", linestyle=":", linewidth=0.8, label=f"rf={rf*100:.0f}%")
ax.axvline(1.0, color="gray", linestyle=":", linewidth=0.8, label="Beta=1（市场）")
ax.set_xlabel("Beta（系统性风险暴露）")
ax.set_ylabel("期望收益率 (%)")
ax.set_title("CAPM证券市场线（SML）\nE[r] = rf + Beta*(rm-rf)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ---- 右图：Beta的OLS回归估计 ----
ax2 = axes[1]
n_months = 60     # 5年月度数据
# 模拟市场超额月收益
market_monthly = np.random.normal((rm - rf) / 12, 0.04, n_months)
# 构造一只真实Beta=1.3的股票（加残差）
beta_true = 1.3
alpha_true = 0.002
stock_monthly = alpha_true + beta_true * market_monthly + np.random.normal(0, 0.025, n_months)

# OLS回归
slope, intercept, r_val, p_val, se = stats.linregress(market_monthly, stock_monthly)

ax2.scatter(market_monthly * 100, stock_monthly * 100,
            color="#20808D", alpha=0.5, s=20, label="月度超额收益")
x_fit = np.linspace(market_monthly.min(), market_monthly.max(), 100)
ax2.plot(x_fit * 100, (intercept + slope * x_fit) * 100,
         color="#A84B2F", linewidth=2,
         label=f"OLS回归 β_hat={slope:.2f}, α_hat={intercept*100:.2f}%/月")
ax2.set_xlabel("市场超额收益 rm-rf (%)")
ax2.set_ylabel("股票超额收益 ri-rf (%)")
ax2.set_title(f"Beta的OLS时序回归\nR²={r_val**2:.3f}，n={n_months}个月")
ax2.axhline(0, color="gray", linewidth=0.5)
ax2.axvline(0, color="gray", linewidth=0.5)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figs/fig_13_2.png", dpi=150, bbox_inches="tight")
```

![](figs/fig_13_2.png)

**运行结果说明：** 左图展示SML以及10只股票相对SML的偏离（Alpha），位于SML上方的股票有正Alpha（被低估），下方的有负Alpha（被高估），灰色箭头标示Alpha量级；右图用60个月超额收益的OLS回归估计Beta，$R^2$ 约0.75，表示75%的股票收益变动由市场解释，残差是可分散化的个股风险。

> 📝 **练一练（编程）：** 修改右图代码，同时绘出Beta估计的95%置信区间（OLS的 $\hat{\beta} \pm 1.96 \times SE(\hat{\beta})$，其中SE来自 `stats.linregress` 返回的 `stderr`）。对比30个月vs60个月vs120个月样本的Beta估计置信区间宽度，说明样本量对Beta估计精度的影响。
>
> **参考答案要点：** 置信区间宽度约为 $\pm 1.96 \times se$，样本量翻倍时标准误差缩小为 $1/\sqrt{2}$，故：30个月区间最宽（约 $\pm 0.4$），60个月（$\pm 0.28$），120个月（$\pm 0.20$）。实践启示：用5年月度数据（60点）是行业惯例，更短的样本Beta估计方差太大，用于风险模型会高估/低估系统性风险。

---

#### 🚀 金融/AI 业务映射

**多因子风控**：CAPM的Beta是Barra风险模型（USE4、CNE6）的核心因子之一，组合的市场Beta暴露 $\sum w_i \beta_i$ 直接决定了持仓对大盘回撤的敏感度，Risk对冲台每日计算全组合Beta并对冲到目标值（通常接近0）。**量化ML**：用于训练风险预测模型时，CAPM残差（$\varepsilon$）是"可分散化风险"的代理标量，机器学习Alpha模型的信号质量评估（IC/IR）的分母正是残差风险的估计值。

> 📝 **练一练（应用）：** 某量化基金持有100只股票，组合Beta=0.95，想通过买卖股指期货（Beta=1）将组合Beta调整为0。需要做空多少名义金额的期货？若组合总市值为5亿元，答案是多少元？
>
> **参考答案要点：** 设做空期货名义金额为 $X$，则对冲后Beta $= 0.95 - X/5亿 \times 1 = 0$，解出 $X = 4.75$ 亿元。即做空名义价值4.75亿的股指期货合约（约475张IF合约，每张100万名义）。这是Beta对冲的基本操作，也叫"市场中性化"。

---

#### 🛡️ 系统级 QA 灾难与防坑指南

**Bug案例1：某量化CTA基金Beta计算时基准指数选择错误，Beta长期报错系统性低估（2021年，华北某私募，AUM约18亿，策略为中证500增强）**

背景：策略基准应选中证500（IC/IF系数合约）来计算Beta，工程师误用沪深300（HS300）作为 `r_m`。由于中证500与沪深300相关系数约0.85，估计出的Beta约为真实值的70%。组合在季报中显示Beta=0.6（实际约0.85），对冲期货头寸远不足，某季度大盘下跌8%，对冲组合回撤4.5%（预期仅2%），客户大量赎回，管理费损失超200万元。

**Bug案例2：某基金公司Beta计算使用未复权价格，股票分红后Beta突变（2023年，某中型公募基金，组合含高分红央企20只）**

背景：Beta时序回归使用原始未复权收益率数据。某央企股票宣布10派5元后，复权价格连续，但未复权价格出现约5%的"暴跌"（实为分红），导致当月收益率被计算为 $-5\%$，而市场当日基本持平，回归出的Beta从0.8飙升到3.1。风控系统报告该股票"Beta异常"并自动减仓，反而在低估时卖出，亏损约190万元。

```python
import numpy as np
from scipy import stats

def validate_beta_estimation(stock_returns: np.ndarray,
                              market_returns: np.ndarray,
                              asset_name: str = "股票",
                              min_samples: int = 30,
                              max_beta: float = 5.0,
                              beta_jump_tol: float = 0.5) -> dict:
    """
    生产级Beta估计QA拦截
    
    Args:
        stock_returns: 股票超额收益率序列（已减去rf，复权价格计算）
        market_returns: 基准指数超额收益率序列（与策略基准一致！）
        asset_name: 资产名称（用于报错定位）
        min_samples: 最小样本量（<此值Beta估计不可信）
        max_beta: Beta绝对值上限（超出则疑似数据错误）
        beta_jump_tol: 相邻期Beta估计的最大允许跳跃幅度
    """
    assert len(stock_returns) == len(market_returns), (
        f"[Beta-QA] {asset_name}：收益率序列长度不匹配"
    )
    n = len(stock_returns)
    
    # 样本量检查
    assert n >= min_samples, (
        f"[Beta-QA] {asset_name}：样本量{n}不足，需>={min_samples}。"
        f"Beta估计方差过大（SE≈{1/np.sqrt(n):.2f}），不可用于风控。"
    )
    
    # 检查NaN/Inf
    assert np.all(np.isfinite(stock_returns)), (
        f"[Beta-QA] {asset_name}：股票收益率含NaN/Inf，请检查复权处理和停牌填充"
    )
    assert np.all(np.isfinite(market_returns)), (
        f"[Beta-QA] {asset_name}：基准收益率含NaN/Inf"
    )
    
    # 检查收益率数量级（单位：月度还是年度？）
    annual_vol_proxy = np.std(stock_returns) * np.sqrt(12)   # 假设月度数据年化
    if annual_vol_proxy > 2.0:
        import warnings
        warnings.warn(
            f"[Beta-QA] {asset_name}：年化波动率估计{annual_vol_proxy*100:.0f}%异常高，"
            f"请确认收益率是月度（而非日度或年度）且已做复权处理"
        )
    
    # 执行OLS回归
    slope, intercept, r_val, p_val, stderr = stats.linregress(market_returns, stock_returns)
    beta_hat = slope
    
    # Beta范围检查
    assert abs(beta_hat) <= max_beta, (
        f"[Beta-QA] {asset_name}：Beta={beta_hat:.2f}异常（绝对值>{max_beta}）\n"
        f"  常见原因：①使用未复权价格含分红跳跃 ②基准指数选错 ③样本期含重大事件"
    )
    
    # 检查异常单期收益（分红或停牌导致的收益跳跃）
    stock_std = np.std(stock_returns)
    outlier_idx = np.where(np.abs(stock_returns) > 5 * stock_std)[0]
    if len(outlier_idx) > 0:
        import warnings
        warnings.warn(
            f"[Beta-QA] {asset_name}：发现{len(outlier_idx)}个>5σ的异常收益率，"
            f"位置：{outlier_idx.tolist()}，值：{stock_returns[outlier_idx]*100}%\n"
            f"  疑似未复权除权数据！请使用前复权或后复权价格重新计算"
        )
    
    ci_95_width = 1.96 * stderr
    return {
        "beta": beta_hat,
        "alpha_monthly": intercept,
        "r_squared": r_val**2,
        "beta_95ci": (beta_hat - ci_95_width, beta_hat + ci_95_width),
        "stderr": stderr,
        "n_samples": n,
        "status": "OK"
    }

# ---- 测试正常数据 ----
np.random.seed(0)
mkt = np.random.normal(0.005, 0.04, 60)   # 60个月市场超额收益
stk = 0.002 + 1.3 * mkt + np.random.normal(0, 0.02, 60)   # Beta真值1.3
result = validate_beta_estimation(stk, mkt, "测试股票A")
print(f"Beta估计: {result['beta']:.3f}，95%CI: {result['beta_95ci']}")

# ---- 测试含分红跳跃的未复权数据 ----
stk_ex_div = stk.copy()
stk_ex_div[20] -= 0.07   # 模拟7%分红跳跃（未复权价格下跌）
try:
    result2 = validate_beta_estimation(stk_ex_div, mkt, "含分红未复权")
except Exception as e:
    print(f"[警告触发] {type(e).__name__}: {str(e)[:100]}")

# ---- 测试样本量不足 ----
try:
    validate_beta_estimation(stk[:10], mkt[:10], "样本量不足")
except AssertionError as e:
    print(f"[拦截成功] {e}")
```

> 📝 **练一练（测试设计）：** 设计一个"基准指数选择合理性"的单元测试：给定一组股票收益率，分别用沪深300和中证500作为基准计算Beta，验证两者之间的差异不超过某阈值。如何量化"基准选择错误"的影响？阈值应该取多少？
>
> **参考答案要点：** 计算 `beta_hs300` 和 `beta_csi500`，若两者差异 `> 0.2`，则警告基准可能选错。量化影响：Beta差异×市场风险溢价 = 预期收益偏差，如差0.3、市场溢价9%则收益预期偏差 $0.3\times9\%=2.7\%$，对百亿组合意味着年化2.7亿的风险溢价被错误定价。阈值0.2对应约2%的预期收益偏差，在大多数策略中已不可忽略。

---

### 13.3 利率与债券定价（久期Duration、凸性Convexity、收益率曲线）

#### 💡 大白话唤醒直觉

债券就像一台"现金流机器"：买入后，每半年收到一笔票息，到期收回本金。这台机器值多少钱？答案是将所有未来现金流**折现**到今天的价值之和。折现率就是**到期收益率**（YTM）。

YTM和价格是**反向关系**：利率（YTM）上升，未来现金流的折现因子减小，债券价格下跌；利率下降，债券价格上涨。这就是为什么加息周期里债券基金亏钱。

**久期（Duration）** 衡量债券价格对利率的敏感度——是债券的"利率风险放大器"。简单比喻：久期就是债券价格的"弹性"，修正久期8意味着利率每上升1%，价格下跌约8%。直觉上，到期时间越长、票息越低，久期越大，利率风险越高。

但久期是线性近似——利率变动1%时很准确，变动4%时误差明显。**凸性（Convexity）** 是二阶修正项，修正久期近似在大利率波动下的误差。凸性永远是正的（价格-收益率曲线向上凸），这意味着实际价格变动永远比久期线性预测更好（涨的时候涨更多，跌的时候跌更少）——这就是"凸性溢价"，投资者愿意为高凸性债券多付钱。

> 📝 **练一练：** 两只5年期债券，A票面利率5%，B票面利率1%（深折价债），YTM均为5%，哪只久期更大？从现金流结构直觉解释原因。
>
> **参考答案要点：** B（1%低票息）久期更大。因为久期是各期现金流的"时间加权平均"，B的票息小，更多现金流集中在最后还本时，重心靠后，久期接近5年；A的高票息让早期现金流权重更大，重心提前，久期略小于5年（约4.33年 vs 约4.88年）。

---

#### 📐 极简公式推导

**Step 1：债券定价公式（折现现金流）**

付息频率 $m$ 次/年，共 $N$ 个付息期，票面利率 $c$，面值 $F$，YTM为 $y$：

$$\boxed{P = \sum_{t=1}^{N} \frac{cF/m}{(1+y/m)^t} + \frac{F}{(1+y/m)^N}}$$

**Step 2：麦考利久期（Macaulay Duration）**

$$D_{\text{Mac}} = \frac{\displaystyle\sum_{t=1}^{N} \frac{t}{m} \cdot \frac{CF_t}{(1+y/m)^t}}{P}$$

即各期现金流折现值为权重的"到期时间"加权平均。

**Step 3：修正久期（Modified Duration）**

$$D_{\text{mod}} = \frac{D_{\text{Mac}}}{1+y/m}$$

价格-收益率关系的一阶近似：

$$\frac{dP}{dy} \approx -D_{\text{mod}} \cdot P \quad \Rightarrow \quad \frac{\Delta P}{P} \approx -D_{\text{mod}} \cdot \Delta y$$

**Step 4：凸性（Convexity）——二阶修正**

$$\text{Conv} = \frac{1}{P} \frac{d^2P}{dy^2} = \frac{\displaystyle\sum_{t=1}^{N} \frac{t(t+1)}{m^2} \cdot \frac{CF_t}{(1+y/m)^{t+2}}}{P}$$

**含凸性的价格近似（更精确）：**

$$\boxed{\frac{\Delta P}{P} \approx -D_{\text{mod}} \cdot \Delta y + \frac{1}{2}\text{Conv} \cdot (\Delta y)^2}$$

> 📝 **练一练：** 某债券修正久期7.5，凸性80，YTM突然上升200bp（$\Delta y = 0.02$）。仅用久期估计价格变动是多少？加上凸性修正后是多少？差距多少？
>
> **参考答案要点：** 久期近似：$-7.5 \times 0.02 = -15\%$。凸性修正项：$\frac{1}{2}\times 80 \times 0.02^2 = +1.6\%$。总估计：$-15\% + 1.6\% = -13.4\%$。差距1.6个百分点，对大规模债券组合（如100亿）这意味着1.6亿的估值差异——说明在大幅利率波动时，不考虑凸性修正会严重高估损失。

---

#### 💻 Python 跑直觉（核心）

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

def bond_price(ytm, coupon, par, n_periods, freq=2):
    """债券定价：YTM折现所有现金流"""
    c = coupon * par / freq       # 每期票息
    r = ytm / freq                # 每期折现率
    periods = np.arange(1, n_periods + 1)
    pv_coupons = np.sum(c / (1 + r) ** periods)
    pv_par = par / (1 + r) ** n_periods
    return pv_coupons + pv_par

def duration_modified(ytm, coupon, par, n_periods, freq=2):
    """计算麦考利久期和修正久期"""
    c = coupon * par / freq
    r = ytm / freq
    price = bond_price(ytm, coupon, par, n_periods, freq)
    periods = np.arange(1, n_periods + 1)
    # 麦考利久期（年）：各期现金流折现值×时间的加权平均
    mac_dur = (np.sum(periods * c / (1+r)**periods)
               + n_periods * par / (1+r)**n_periods) / price / freq
    mod_dur = mac_dur / (1 + ytm / freq)
    return mac_dur, mod_dur

def convexity(ytm, coupon, par, n_periods, freq=2):
    """计算债券凸性"""
    c = coupon * par / freq
    r = ytm / freq
    price = bond_price(ytm, coupon, par, n_periods, freq)
    periods = np.arange(1, n_periods + 1)
    conv = (np.sum(periods * (periods+1) * c / (1+r)**(periods+2))
            + n_periods*(n_periods+1)*par / (1+r)**(n_periods+2)) / price / freq**2
    return conv

coupon, par, n_periods = 0.05, 1000.0, 20   # 5%票面，10年半年付息
ytm0 = 0.05

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# ---- 左图：债券价格 vs YTM ----
ytm_range = np.linspace(0.01, 0.12, 300)
prices = [bond_price(y, coupon, par, n_periods) for y in ytm_range]
ax = axes[0]
ax.plot(ytm_range*100, prices, color="#20808D", linewidth=2.5, label="债券价格")
ax.axvline(ytm0*100, color="gray", linestyle="--", linewidth=0.8, label=f"初始YTM={ytm0*100:.0f}%")
ax.axhline(par, color="#A84B2F", linestyle=":", linewidth=0.8, label=f"面值={par}")
ax.set_xlabel("到期收益率 YTM (%)"); ax.set_ylabel("债券价格（元）")
ax.set_title("债券价格 vs YTM\n价格-收益率的凸性关系（向上凸）")
ax.legend(); ax.grid(True, alpha=0.3)

# ---- 中图：久期 vs 到期期限 ----
years_list = [2, 5, 7, 10, 15, 20, 25, 30]
n_periods_list = [y*2 for y in years_list]
mac_durs, mod_durs = zip(*[duration_modified(ytm0, coupon, par, n) for n in n_periods_list])
ax2 = axes[1]
ax2.plot(years_list, mac_durs, color="#20808D", marker="o", linewidth=2, label="麦考利久期")
ax2.plot(years_list, mod_durs, color="#A84B2F", marker="s", linewidth=2, label="修正久期")
ax2.set_xlabel("剩余期限（年）"); ax2.set_ylabel("久期（年）")
ax2.set_title("久期 vs 到期期限\n票面利率=YTM=5%（平价债券）")
ax2.legend(); ax2.grid(True, alpha=0.3)

# ---- 右图：久期近似 vs 精确价格变动（凸性修正效果） ----
ax3 = axes[2]
mac_d, mod_d = duration_modified(ytm0, coupon, par, n_periods)
conv = convexity(ytm0, coupon, par, n_periods)
P0 = bond_price(ytm0, coupon, par, n_periods)

delta_ytms = np.linspace(-0.04, 0.04, 200)
exact_change = [(bond_price(ytm0+dy, coupon, par, n_periods) - P0)/P0*100 for dy in delta_ytms]
dur_approx = [-mod_d * dy * 100 for dy in delta_ytms]
conv_approx = [(-mod_d * dy + 0.5 * conv * dy**2) * 100 for dy in delta_ytms]

ax3.plot(delta_ytms*100, exact_change, color="#20808D", linewidth=2.5, label="精确价格变动")
ax3.plot(delta_ytms*100, dur_approx, color="#A84B2F", linewidth=1.8, linestyle="--", label="久期线性近似")
ax3.plot(delta_ytms*100, conv_approx, color="#FFC553", linewidth=1.8, linestyle="-.", label="久期+凸性修正")
ax3.set_xlabel("YTM变动 Δytm (%)"); ax3.set_ylabel("价格变动 (%)")
ax3.set_title("久期近似 vs 精确价格变动\n大幅利率波动时凸性修正不可忽略")
ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3)
ax3.axhline(0, color="gray", linewidth=0.5); ax3.axvline(0, color="gray", linewidth=0.5)

plt.tight_layout()
plt.savefig("figs/fig_13_3.png", dpi=150, bbox_inches="tight")
```

![](figs/fig_13_3.png)

**运行结果说明：** 左图展示价格-YTM的凸形曲线（价格下降时斜率逐渐减小）；中图展示久期随期限增加而趋于饱和（平价债券），30年债久期约为14年而非30年；右图清晰展示：YTM变动±2%时久期线性近似和精确价格偏差约1.6个百分点，加入凸性修正后与精确值几乎重合，大于±3%的冲击下凸性修正仍有残差（需要更高阶项）。

> 📝 **练一练（编程）：** 计算并比较以下两只债券的久期和凸性：①10年期零息债券（面值1000，YTM=5%）；②10年期票面利率10%的债券（面值1000，半年付息，YTM=5%）。验证零息债券的麦考利久期等于其到期时间（10年）。
>
> **参考答案要点：** 零息债券只在到期日支付面值，所有现金流集中于第 $N$ 期，麦考利久期 $= N \times \Delta t = 10$ 年（无需计算，逻辑必然）。高票息债券因早期现金流占比高，久期约7.8年。凸性：零息债 $\approx 110$，高票息债 $\approx 65$，零息债凸性更高（价格在大利率变动下的"保护性"更强）。

---

#### 🚀 金融/AI 业务映射

**固收风控**：债券基金的组合久期和关键利率久期（KRD）是每日标配风控指标，久期超限会触发自动减仓；凸性指标在利率波动率上升时尤其重要（久期相同但高凸性债券有更好的"Gamma收益"，类似期权）。**量化ML**：时序神经网络（LSTM/Transformer）在预测收益率曲线形变时，其隐层对不同期限的注意力权重在数学上等价于动态调整"久期分配"——模型学习到的是不同期限利率冲击对组合PnL的非线性映射，本质是凸性感知的。

> 📝 **练一练（应用）：** 某银行自营部门持有一个债券组合，修正久期5.0，凸性120，YTM突然上升300bp（2022年3月美联储加息风格）。用久期+凸性估计价格变动，并解释为什么高凸性债券在这种压力情景下比低凸性债券"跌得少"。
>
> **参考答案要点：** 价格变动 $\approx -5.0 \times 0.03 + \frac{1}{2}\times 120 \times 0.03^2 = -15\% + 5.4\% = -9.6\%$。若凸性仅为40（低凸性债），则为 $-15\% + 1.8\% = -13.2\%$，多跌3.6个百分点。高凸性债之所以跌得少，是因为价格-收益率曲线弯曲得更厉害（向上凸更明显），利率上升时曲线切线斜率越来越平缓，高凸性债对应"曲率更大的弯曲"，保护了下行空间。

---

#### 🛡️ 系统级 QA 灾难与防坑指南

**Bug案例1：某城商行债券风控系统久期线性近似在2022年美联储加息周期中产生重大估值误差（2022年，某中型城商行，自营债券规模约280亿元）**

背景：风控系统使用纯久期线性近似估算组合风险（不含凸性修正），平时利率波动30-50bp，误差可忽略。2022年美联储连续加息，累计YTM上升约300bp，系统报告组合跌幅-8.4%（久期×利率变动），实际跌幅-5.9%，差距2.5个百分点（即约7亿元的估值高估损失）。管理层基于错误的损益数据提前止损平仓，锁定了不必要的亏损。

**Bug案例2：某基金公司收益率曲线插值方法错误导致关键利率久期计算扭曲（2023年，某中型公募基金，债基规模约60亿）**

背景：KRD（关键利率久期）计算需要对收益率曲线进行"关键利率冲击"（只移动某一期限的利率）。系统使用线性插值而非标准的NelsonSiegel或三次样条插值，导致"2年期关键利率冲击"会同时抬高相邻期限（1年和3年）的利率估计，KRD分配与真实利率因子敞口不匹配，对冲时买错了国债期货合约期限，利率风险对冲效率仅约65%。某次国债利率大幅波动，组合损失约800万元（比预期对冲结果多亏530万）。

```python
import numpy as np

def validate_bond_duration_convexity(ytm: float, coupon: float, par: float,
                                      n_periods: int, freq: int = 2,
                                      max_duration_years: float = 30.0) -> dict:
    """
    生产级久期凸性计算QA，内含数值一致性验证
    
    核心思路：用有限差分法（精确值）和解析公式交叉验证，
    偏差超过阈值则说明公式实现有bug。
    """
    # ---- 输入验证 ----
    assert 0 < ytm < 1.0, f"[Dur-QA] YTM={ytm*100:.2f}%异常（应在0%-100%）"
    assert 0 <= coupon <= 1.0, f"[Dur-QA] 票面利率={coupon*100:.2f}%异常"
    assert par > 0, f"[Dur-QA] 面值必须>0"
    assert n_periods > 0 and isinstance(n_periods, int), "[Dur-QA] 付息期数必须为正整数"
    
    P0 = bond_price_calc(ytm, coupon, par, n_periods, freq)
    assert P0 > 0, f"[Dur-QA] 债券价格计算为负数：{P0}，请检查输入参数"
    
    # 解析久期（用公式）
    c = coupon * par / freq
    r = ytm / freq
    periods = np.arange(1, n_periods + 1)
    mac_dur_analytic = (np.sum(periods * c / (1+r)**periods)
                        + n_periods * par / (1+r)**n_periods) / P0 / freq
    mod_dur_analytic = mac_dur_analytic / (1 + ytm / freq)
    
    # 数值有限差分久期（"真值"参考）
    eps = 1e-5
    P_up = bond_price_calc(ytm + eps, coupon, par, n_periods, freq)
    P_dn = bond_price_calc(ytm - eps, coupon, par, n_periods, freq)
    mod_dur_numeric = -(P_up - P_dn) / (2 * eps * P0)
    
    dur_rel_err = abs(mod_dur_analytic - mod_dur_numeric) / max(abs(mod_dur_numeric), 1e-8)
    assert dur_rel_err < 1e-4, (
        f"[Dur-QA] 修正久期计算不一致！\n"
        f"  解析公式: {mod_dur_analytic:.4f}\n"
        f"  有限差分: {mod_dur_numeric:.4f}\n"
        f"  相对误差: {dur_rel_err*100:.4f}%\n"
        f"  请检查麦考利久期公式中时间单位（年 vs 期数）的处理"
    )
    
    # 凸性数值验证
    conv_analytic = compute_convexity(ytm, coupon, par, n_periods, freq)
    conv_numeric = (P_up + P_dn - 2 * P0) / (eps**2 * P0)
    conv_rel_err = abs(conv_analytic - conv_numeric) / max(abs(conv_numeric), 1e-8)
    assert conv_rel_err < 1e-3, (
        f"[Dur-QA] 凸性计算不一致！\n"
        f"  解析: {conv_analytic:.4f}，有限差分: {conv_numeric:.4f}，"
        f"相对误差: {conv_rel_err*100:.4f}%"
    )
    
    # 久期合理性检查（必须 < 到期时间）
    maturity_years = n_periods / freq
    assert mac_dur_analytic <= maturity_years + 1e-6, (
        f"[Dur-QA] 麦考利久期({mac_dur_analytic:.2f}年) > 到期时间({maturity_years:.2f}年)，违反定义"
    )
    assert mac_dur_analytic <= max_duration_years, (
        f"[Dur-QA] 麦考利久期{mac_dur_analytic:.2f}年超过上限{max_duration_years}年"
    )
    
    # 凸性正数检查（含权债券除外）
    assert conv_analytic >= 0, (
        f"[Dur-QA] 凸性为负值{conv_analytic:.4f}，普通债券凸性必须≥0"
    )
    
    print(f"[Dur-QA] 通过 | P={P0:.2f} | 麦考利久期={mac_dur_analytic:.3f}年 | "
          f"修正久期={mod_dur_analytic:.3f} | 凸性={conv_analytic:.2f}")
    
    return {"price": P0, "mac_duration": mac_dur_analytic,
            "mod_duration": mod_dur_analytic, "convexity": conv_analytic}

# 辅助函数（生产中通常已有）
def bond_price_calc(ytm, coupon, par, n_periods, freq=2):
    c = coupon * par / freq
    r = ytm / freq
    t = np.arange(1, n_periods + 1)
    return np.sum(c / (1+r)**t) + par / (1+r)**n_periods

def compute_convexity(ytm, coupon, par, n_periods, freq=2):
    c = coupon * par / freq
    r = ytm / freq
    price = bond_price_calc(ytm, coupon, par, n_periods, freq)
    t = np.arange(1, n_periods + 1)
    return (np.sum(t*(t+1)*c/(1+r)**(t+2)) + n_periods*(n_periods+1)*par/(1+r)**(n_periods+2)) / price / freq**2

# ---- 测试正常10年期债券 ----
res = validate_bond_duration_convexity(0.05, 0.05, 1000, 20, freq=2)

# ---- 测试零息债券：麦考利久期应=到期时间 ----
res_zero = validate_bond_duration_convexity(0.05, 0.00, 1000, 20, freq=2)
print(f"零息债麦考利久期={res_zero['mac_duration']:.4f}年（理论值=10年）")

# ---- 测试极端YTM（验证数值稳定性）----
res_extreme = validate_bond_duration_convexity(0.001, 0.05, 1000, 20, freq=2)  # 极低利率
print(f"极低利率环境久期={res_extreme['mod_duration']:.3f}（应>正常利率情形）")
```

> 📝 **练一练（测试设计）：** 某固收系统对100只债券批量计算久期，设计一个参数化测试集，覆盖以下边界：①零息债（coupon=0）；②溢价债（coupon > YTM）；③折价债（coupon < YTM）；④超长久期（30年零息债，久期应=30）；⑤临近到期（剩余2期，久期应接近1/freq）。写出每组的预期久期范围验证逻辑。
>
> **参考答案要点：** ①零息债Mac久期=n_periods/freq，误差<1e-4；②溢价债价格>面值，久期<到期时间；③折价债价格<面值，久期略大于等票面溢价债；④30年零息债Mac久期应在 $[29.9, 30.0]$；⑤2期剩余时Mac久期应在 $(0, 1/\text{freq}]$（即 $(0, 0.5]$ 对半年付息）。这五类构成固收久期计算的完备测试矩阵。
