import numpy as np
import matplotlib.pyplot as plt
import matplotlib

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

# ========== 解决中文显示问题 ==========
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

"""
2. 随机梯度下降 (Stochastic 02_Gradient Descent, SGD) —— "边看边走"
📊 可视化效果说明
运行后会弹出 2×2 的四个子图：
子图	内容	作用
左上	每步更新的损失曲线	展示 SGD 每个样本更新后的损失变化，会看到明显的震荡
右上	每个 Epoch 的平均损失	平滑后的收敛曲线，更清晰地看到整体下降趋势
左下	权重 w 的变化曲线	观察 w 从 1.0 逐步逼近 2.0 的过程，虚线标记最优值
右下	拟合结果	对比真实数据点、初始直线、SGD 拟合直线和最优直线
"""

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0


def forward(x):
    return x * w


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


def gradient(x, y):
    return 2 * x * (x * w - y)


# ========== 存储训练历史 ==========
w_history = [w]
loss_history = []
epoch_losses = []  # 每个 epoch 的平均损失

print('Predict (before training)', 4, forward(4))

for epoch in range(100):
    epoch_loss_sum = 0
    epoch_count = 0

    for x, y in zip(x_data, y_data):
        grad = gradient(x, y)
        w -= 0.01 * grad
        l = loss(x, y)

        # 记录每一步的 w 和 loss
        w_history.append(w)
        loss_history.append(l)
        epoch_loss_sum += l
        epoch_count += 1

        if epoch % 10 == 0:  # 每 10 个 epoch 打印一次
            print(f'  epoch: {epoch}, x={x}, y={y}, grad={grad:.6f}, w={w:.6f}, loss={l:.6f}')

    # 计算每个 epoch 的平均损失
    epoch_losses.append(epoch_loss_sum / epoch_count)

print('Predict (after training)', 4, forward(4))

# ========== 可视化部分 ==========

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# -------- 图1：损失下降曲线（每个样本更新） --------
ax1 = axes[0, 0]
ax1.plot(range(len(loss_history)), loss_history, 'b-', linewidth=1, alpha=0.7)
ax1.set_xlabel('更新步数 (Step)', fontsize=12)
ax1.set_ylabel('损失 (Loss)', fontsize=12)
ax1.set_title('SGD - 损失下降曲线（每步更新）', fontsize=14)
ax1.grid(True, linestyle='--', alpha=0.3)
ax1.set_yscale('log')  # 使用对数坐标，更清晰

# -------- 图2：每个 Epoch 的平均损失 --------
ax2 = axes[0, 1]
ax2.plot(range(1, 101), epoch_losses, 'r-', linewidth=2)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('平均损失 (Avg Loss)', fontsize=12)
ax2.set_title('SGD - 每个 Epoch 平均损失', fontsize=14)
ax2.grid(True, linestyle='--', alpha=0.3)

# 标注最终损失
ax2.annotate(f'Final Loss = {epoch_losses[-1]:.6f}',
             xy=(100, epoch_losses[-1]),
             xytext=(85, epoch_losses[-1] + 0.5),
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=10, color='red')

# -------- 图3：w 的变化曲线 --------
ax3 = axes[1, 0]
ax3.plot(range(len(w_history)), w_history, 'g-', linewidth=1.5, alpha=0.7)
ax3.axhline(y=2.0, color='r', linestyle='--', linewidth=1.5, label='最优值 w=2.0')
ax3.set_xlabel('更新步数 (Step)', fontsize=12)
ax3.set_ylabel('权重 w', fontsize=12)
ax3.set_title('SGD - 权重 w 的变化', fontsize=14)
ax3.legend()
ax3.grid(True, linestyle='--', alpha=0.3)
ax3.set_ylim(0, 2.5)

# -------- 图4：拟合结果 --------
ax4 = axes[1, 1]

# 绘制真实数据点
ax4.scatter(x_data, y_data, color='red', s=100, label='真实数据 (True)', zorder=5)

# 绘制训练后的拟合直线
x_test = np.linspace(0, 4, 100)
y_test = forward(x_test)
ax4.plot(x_test, y_test, 'b-', linewidth=2.5, label=f'SGD 拟合直线 (w={w:.4f})')

# 绘制初始直线（w=1.0）
w_backup = w
w = 1.0
y_initial = forward(x_test)
w = w_backup
ax4.plot(x_test, y_initial, 'g--', linewidth=1.5, label='初始直线 (w=1.0)', alpha=0.6)

# 绘制最优直线（w=2.0）
w_opt = 2.0
w_backup_opt = w
w = w_opt
y_optimal = forward(x_test)
w = w_backup_opt
ax4.plot(x_test, y_optimal, 'r--', linewidth=1.5, label='最优直线 (w=2.0)', alpha=0.6)

# 标注预测值
ax4.scatter(4, forward(4), color='blue', s=120, marker='*',
            label=f'预测值: f(4)={forward(4):.4f}', zorder=6)
ax4.annotate(f'f(4)={forward(4):.4f}',
             xy=(4, forward(4)),
             xytext=(4.3, forward(4) - 0.5),
             fontsize=10, color='blue')

ax4.set_xlabel('x', fontsize=12)
ax4.set_ylabel('y', fontsize=12)
ax4.set_title('SGD - 拟合结果', fontsize=14)
ax4.legend(loc='upper left')
ax4.grid(True, linestyle='--', alpha=0.3)
ax4.set_xlim(0, 4.5)
ax4.set_ylim(0, 9)

plt.suptitle('随机梯度下降 (SGD) 训练过程可视化', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# ========== 打印最终结果 ==========
print('\n' + '=' * 60)
print(f'SGD 训练完成！')
print(f'最终权重 w = {w:.6f}')
print(f'最终平均损失 = {epoch_losses[-1]:.6f}')
print(f'预测值 f(4) = {forward(4):.6f} (真实值应为 8.0)')
print('=' * 60)

# ========== 额外：与 GD 的对比说明 ==========
print('\n📊 与梯度下降 (GD) 的对比：')
print('-' * 40)
print(f'GD 每次更新使用 {len(x_data)} 个样本（全量数据）')
print(f'SGD 每次更新使用 1 个样本（随机选择）')
print(f'SGD 的损失曲线震荡更明显，但收敛速度通常更快')