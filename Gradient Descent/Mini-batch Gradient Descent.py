import numpy as np
import matplotlib.pyplot as plt

"""
3. Mini-batch Gradient Descent —— “中庸之道”
用 Mini-batch GD 下山，会这样做：
不是看整个山体（太慢），也不是只看脚下1平方厘米（太不准），
而是扫视眼前的一大块区域（比如，随机选 64 个点）。
根据这一块区域的坡度，迈出一步。
核心特点：“折中方案”。每一步使用一小批数据来计算梯度。
优点：比 GD 快得多，比 SGD 稳定得多，是现代深度学习的绝对主流！
缺点：需要调节 batch_size 这个超参数。
"""

# 准备数据（为了演示，生成更多数据）
np.random.seed(42)
x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
y_data = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0])

w = 1.0
batch_size = 4  # 每批使用 4 个样本（典型值：16, 32, 64, 128, 256）


def forward(x):
    return x * w


def cost(xs, ys):
    """计算均方误差"""
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred - y) ** 2
    return cost / len(xs)


def gradient(xs, ys):
    """计算梯度（使用当前 batch 的数据）"""
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
    return grad / len(xs)


print('Predict (before training)', 4, forward(4))

# 存储损失值用于绘图
loss_history = []

for epoch in range(100):
    # 1. 在每个 epoch 开始时，打乱数据（关键步骤！）
    indices = np.random.permutation(len(x_data))
    x_shuffled = x_data[indices]
    y_shuffled = y_data[indices]

    epoch_loss = 0
    num_batches = 0

    # 2. 将数据分成多个 mini-batch
    for i in range(0, len(x_data), batch_size):
        # 取出一个 batch 的数据
        x_batch = x_shuffled[i:i + batch_size]
        y_batch = y_shuffled[i:i + batch_size]

        # 3. 用这个 batch 计算梯度并更新参数
        grad_val = gradient(x_batch, y_batch)
        w -= 0.01 * grad_val  # 学习率 0.01

        # 记录这个 batch 的损失
        batch_loss = cost(x_batch, y_batch)
        epoch_loss += batch_loss
        num_batches += 1

    # 记录每个 epoch 的平均损失
    avg_loss = epoch_loss / num_batches
    loss_history.append(avg_loss)

    # 每 10 个 epoch 打印一次（减少输出）
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:3d}, w = {w:.6f}, loss = {avg_loss:.6f}')

print('Predict (after training)', 4, forward(4))

# 绘制损失下降曲线
plt.plot(range(100), loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Mini-batch Gradient Descent - Loss Curve')
plt.grid(True)
plt.show()