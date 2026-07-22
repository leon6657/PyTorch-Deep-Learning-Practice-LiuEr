import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

# ========== 解决中文和负号显示问题 ==========
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

"""
三种梯度下降方法对比：
1. 批量梯度下降 (BGD)  ：每次使用全部数据
2. 随机梯度下降 (SGD)  ：每次使用 1 个样本
3. 小批量梯度下降 (MBGD)：每次使用一小批数据（batch_size=4）
"""

# ========== 生成数据 ==========
np.random.seed(42)
x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
y_data = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0])


# ========== 定义模型和损失函数 ==========
def forward(x, w):
    return x * w


def loss(x, y, w):
    y_pred = forward(x, w)
    return (y_pred - y) ** 2


def cost(xs, ys, w):
    total = 0
    for x, y in zip(xs, ys):
        total += (forward(x, w) - y) ** 2
    return total / len(xs)


def gradient(x, y, w):
    return 2 * x * (x * w - y)


# ========== 三种梯度下降的训练函数 ==========

def train_bgd(xs, ys, lr=0.01, epochs=100):
    """批量梯度下降：每次用全部数据"""
    w = 1.0
    w_history = [w]
    loss_history = []
    start_time = time.time()

    for epoch in range(epochs):
        # 计算全部数据的梯度
        grad = 0
        for x, y in zip(xs, ys):
            grad += gradient(x, y, w)
        grad /= len(xs)

        w -= lr * grad
        w_history.append(w)
        loss_history.append(cost(xs, ys, w))

    end_time = time.time()
    return w_history, loss_history, end_time - start_time


def train_sgd(xs, ys, lr=0.01, epochs=100):
    """随机梯度下降：每次用 1 个样本"""
    w = 1.0
    w_history = [w]
    loss_history = []
    start_time = time.time()

    for epoch in range(epochs):
        # 打乱数据顺序
        indices = np.random.permutation(len(xs))
        xs_shuffled = xs[indices]
        ys_shuffled = ys[indices]

        for x, y in zip(xs_shuffled, ys_shuffled):
            grad = gradient(x, y, w)
            w -= lr * grad
            w_history.append(w)
            loss_history.append(cost(xs, ys, w))  # 用全部数据计算损失用于对比

    end_time = time.time()
    return w_history, loss_history, end_time - start_time


def train_mbgd(xs, ys, batch_size=4, lr=0.01, epochs=100):
    """小批量梯度下降：每次用一小批数据"""
    w = 1.0
    w_history = [w]
    loss_history = []
    start_time = time.time()

    for epoch in range(epochs):
        # 打乱数据顺序
        indices = np.random.permutation(len(xs))
        xs_shuffled = xs[indices]
        ys_shuffled = ys[indices]

        # 分批处理
        for i in range(0, len(xs), batch_size):
            x_batch = xs_shuffled[i:i + batch_size]
            y_batch = ys_shuffled[i:i + batch_size]

            # 计算 batch 的梯度
            grad = 0
            for x, y in zip(x_batch, y_batch):
                grad += gradient(x, y, w)
            grad /= len(x_batch)

            w -= lr * grad
            w_history.append(w)
            loss_history.append(cost(xs, ys, w))

    end_time = time.time()
    return w_history, loss_history, end_time - start_time


# ========== 运行三种方法 ==========
print("开始训练...")

epochs = 100
lr = 0.01

# 训练 BGD
print("\n[1/3] 训练批量梯度下降 (BGD)...")
w_bgd, loss_bgd, time_bgd = train_bgd(x_data, y_data, lr, epochs)

# 训练 SGD
print("[2/3] 训练随机梯度下降 (SGD)...")
w_sgd, loss_sgd, time_sgd = train_sgd(x_data, y_data, lr, epochs)

# 训练 MBGD
print("[3/3] 训练小批量梯度下降 (MBGD)...")
w_mbgd, loss_mbgd, time_mbgd = train_mbgd(x_data, y_data, batch_size=4, lr=lr, epochs=epochs)

# 最终权重
final_w_bgd = w_bgd[-1]
final_w_sgd = w_sgd[-1]
final_w_mbgd = w_mbgd[-1]

# 打印训练结果
print("\n" + "=" * 70)
print(f"{'方法':<15} {'最终权重 w':<15} {'最终损失':<15} {'耗时 (秒)':<15}")
print("-" * 70)
print(f"{'BGD (批量)':<15} {final_w_bgd:.6f}    {loss_bgd[-1]:.6f}    {time_bgd:.4f}")
print(f"{'SGD (随机)':<15} {final_w_sgd:.6f}    {loss_sgd[-1]:.6f}    {time_sgd:.4f}")
print(f"{'MBGD (小批量)':<15} {final_w_mbgd:.6f}    {loss_mbgd[-1]:.6f}    {time_mbgd:.4f}")
print("=" * 70)

# ========== 可视化 ==========
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('三种梯度下降方法对比', fontsize=18, fontweight='bold')

# 子图 1-3：损失曲线
ax_loss = [axes[0, 0], axes[0, 1], axes[0, 2]]
titles = ['BGD - 批量梯度下降', 'SGD - 随机梯度下降', 'MBGD - 小批量梯度下降']
colors = ['blue', 'green', 'orange']

loss_data = [loss_bgd, loss_sgd, loss_mbgd]
w_data = [w_bgd, w_sgd, w_mbgd]

for i, (ax, title, color, loss_hist, w_hist) in enumerate(
        zip(ax_loss, titles, colors, loss_data, w_data)):
    ax.plot(range(len(loss_hist)), loss_hist, color=color, linewidth=1.5, alpha=0.8)
    ax.set_xlabel('更新步数 (Step)', fontsize=11)
    ax.set_ylabel('损失 (Loss)', fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.3)

    # 标注最终损失
    ax.annotate(f'最终 Loss = {loss_hist[-1]:.6f}',
                xy=(len(loss_hist) - 1, loss_hist[-1]),
                xytext=(len(loss_hist) * 0.5, loss_hist[-1] * 1.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=1),
                fontsize=9, color='red')

    # 用对数坐标使曲线更清晰
    ax.set_yscale('log')

# 子图 4-6：权重 w 的变化
ax_w = [axes[1, 0], axes[1, 1], axes[1, 2]]

for i, (ax, title, color, w_hist) in enumerate(
        zip(ax_w, titles, colors, w_data)):
    ax.plot(range(len(w_hist)), w_hist, color=color, linewidth=1.5, alpha=0.8)
    ax.axhline(y=2.0, color='red', linestyle='--', linewidth=1.5, label='最优值 w=2.0')
    ax.set_xlabel('更新步数 (Step)', fontsize=11)
    ax.set_ylabel('权重 w', fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_ylim(0.5, 2.5)

# 添加总览信息（在右上角空白处）
plt.tight_layout()
plt.show()

# ========== 额外：绘制拟合直线对比 ==========
fig2, ax = plt.subplots(figsize=(10, 6))
fig2.suptitle('三种方法拟合结果对比', fontsize=16, fontweight='bold')

# 真实数据点
ax.scatter(x_data, y_data, color='black', s=80, label='真实数据', zorder=5)

# 生成测试数据
x_test = np.linspace(0, 11, 100)

# 绘制三种方法的拟合直线
w_values = [final_w_bgd, final_w_sgd, final_w_mbgd]
colors = ['blue', 'green', 'orange']
labels = [
    f'BGD (w={final_w_bgd:.4f})',
    f'SGD (w={final_w_sgd:.4f})',
    f'MBGD (w={final_w_mbgd:.4f})'
]

for w_val, color, label in zip(w_values, colors, labels):
    y_test = forward(x_test, w_val)
    ax.plot(x_test, y_test, color=color, linewidth=2, label=label)

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.legend(loc='upper left')
ax.grid(True, linestyle='--', alpha=0.3)
ax.set_xlim(0, 11)
ax.set_ylim(0, 22)

plt.tight_layout()
plt.show()

print("\n✅ 训练完成！三种梯度下降方法已对比完毕。")