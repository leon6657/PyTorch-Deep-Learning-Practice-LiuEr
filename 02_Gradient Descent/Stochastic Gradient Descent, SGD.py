import numpy as np
import matplotlib.pyplot as plt
"""
2. 随机梯度下降 (Stochastic 02_Gradient Descent, SGD) —— “边看边走”
用 SGD 下山，你会这样做：
你不再去感受整个山体，而是随机地、快速地扫一眼脚下的一个小角落（比如，只计算 1个 数据点的坡度）。
你根据这个角落的坡度，立刻迈出一步。
走到新位置后，再随机扫一眼另一个角落，再迈一步。
核心特点：“以偏概全”。每一步只依赖极少量的数据（甚至 1 个）来计算方向。
优点：极快！ 因为每次计算量极小，几秒钟就能迈出几十步。这让你可以在短时间内探索山体的很多区域。
缺点：不稳定、有噪音。因为你每次都只看一个小角落，很可能这一步的方向是错的（比如，本该往东走，但你扫到的那个角落让你觉得该往西走），路径会非常曲折。
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

"""
#第一次更新w前，使用w=1.0对特征为4时进行预测，返回预测值
#第二次，更新w后，使用更新后的w，对特征为4时预测，返回预测值
"""
print('Predict (before traning)', 4, forward(4))

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        """
        求梯度
        更新w
        输出特征和对应的真实标签；
        计算损失
        输出w和loss
        """
        grad = gradient(x, y)
        w -= 0.01 * grad
        print('\tgrad:', x, y, grad)
        l = loss(x, y)
        print('progress:', epoch, 'w = ', w, 'loss = ', l)

print('Predict (after training)', 4, forward(4))