import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# ========== 解决中文显示问题 ==========
# 设置中文字体（Windows 系统）
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
# 解决负号 '-' 显示为方块的问题
matplotlib.rcParams['axes.unicode_minus'] = False
"""
1. 梯度下降 (02_Gradient Descent, GD) —— “三思而后行” -----可视化
左图：损失下降曲线
横轴：训练轮次（Epoch）
纵轴：损失值（Loss）
你会看到损失从高到低快速下降，最终趋近于 0

右图：拟合结果
红色圆点：真实数据点 (1,2), (2,4), (3,6)
蓝色实线：训练后的拟合直线（斜率接近 2.0）
绿色虚线：初始直线（w=1.0），作为对比
蓝色五角星：预测值 f(4)
"""

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0


def forward(x):
    return x * w


def cost(xs, ys):
    """
    :param xs: 特征
    :param ys: 对应的真实值
    :return: 误差；返回的均方误差
    """
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred - y) ** 2
    return cost / len(xs)


def gradient(xs, ys):
    """
    求导（误差求导）
    :param xs: 特征
    :param ys: 对应的真实值
    :return: 返回的是均方误差的导数
    """
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
    return grad / len(xs)


print('Predict (before training)', 4, forward(4))

# ========== 新增：存储训练历史 ==========
loss_history = []
w_history = []

for epoch in range(100):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= 0.01 * grad_val

    # 记录历史
    loss_history.append(cost_val)
    w_history.append(w)

    if epoch % 10 == 0:  # 每 10 次打印一次，减少输出
        print(f'Epoch: {epoch:3d}, w = {w:.6f}, loss = {cost_val:.6f}')

print('Predict (after training)', 4, forward(4))

# ========== 可视化部分 ==========

# 创建画布，1行2列的子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 图1：损失曲线
ax1.plot(range(100), loss_history, 'b-', linewidth=2)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Loss Curve (训练损失下降曲线)', fontsize=14)
ax1.grid(True, linestyle='--', alpha=0.7)

# 标注最终损失
ax1.annotate(f'Final Loss = {loss_history[-1]:.6f}',
             xy=(95, loss_history[-1]),
             xytext=(70, loss_history[-1] + 0.1),
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=10, color='red')

# 图2：拟合直线 vs 真实数据
x_test = np.linspace(0, 4, 100)
y_test = forward(x_test)  # 使用训练好的 w

ax2.scatter(x_data, y_data, color='red', s=80, label='真实数据 (True Data)', zorder=5)
ax2.plot(x_test, y_test, 'b-', linewidth=2, label=f'拟合直线 (w = {w:.4f})')

# 绘制初始直线（w=1.0）作为对比
w_initial = 1.0
# 临时改变 w 来画初始直线
w_backup = w
w = w_initial
y_initial = forward(x_test)
w = w_backup  # 恢复

ax2.plot(x_test, y_initial, 'g--', linewidth=1.5, label=f'初始直线 (w = 1.0)', alpha=0.6)

ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('y', fontsize=12)
ax2.set_title('Fitted Line (拟合结果)', fontsize=14)
ax2.legend(loc='upper left')
ax2.grid(True, linestyle='--', alpha=0.7)

# 在图上显示预测值
ax2.scatter(4, forward(4), color='blue', s=100, marker='*',
            label=f'预测值: f(4) = {forward(4):.4f}', zorder=6)
ax2.annotate(f'f(4) = {forward(4):.4f}',
             xy=(4, forward(4)),
             xytext=(4.3, forward(4) - 0.3),
             fontsize=10, color='blue')

plt.suptitle('02_Gradient Descent Demo - 梯度下降可视化', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# ========== 额外：打印最终结果 ==========
print('\n' + '=' * 50)
print(f'训练完成！')
print(f'最终权重 w = {w:.6f}')
print(f'最终损失 loss = {loss_history[-1]:.6f}')
print(f'预测值 f(4) = {forward(4):.6f} (真实值应为 8.0)')
print('=' * 50)