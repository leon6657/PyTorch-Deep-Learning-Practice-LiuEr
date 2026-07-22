import numpy as np
import matplotlib.pyplot as plt
"""
1. 梯度下降 (02_Gradient_Descent, GD) —— “三思而后行”
用 GD 下山，你会这样做：
停住脚步，站在原地。
感受一下整个山体（或者说，精确计算所有方向的坡度）。
经过精确计算，你找到了最陡峭的、最确定的下坡方向。
然后，你迈出坚实的一步，向那个方向走。
走到新位置后，再次停下，重新计算当前最陡的方向。
核心特点：“全知全能”。每一步都极其精准，因为你用上了所有的数据点（全量数据）来计算坡度。
优点：下山的方向绝对正确，没有偏差，路径很稳定。
缺点：太慢了！ 每次都要停下来计算整个山体，在数据量巨大（比如几百万张图片）时，计算成本高得吓人，几乎不可能完成
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

print('Predict (before traning)', 4, forward(4))

for epoch in range(100):
    """
    遍历100次
    调用cost函数计算损失值即误差值
    调用gradient函数求得求导值
    更新w值
    输出第几次迭代，对应的更新的权值，损失值
    """
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= 0.01 * grad_val
    print('Epoch:', epoch, 'w = ', w, 'loss = ', cost_val)

print('Predict (after training)', 4, forward(4))