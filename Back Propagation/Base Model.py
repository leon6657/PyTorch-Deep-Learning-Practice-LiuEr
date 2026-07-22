import torch

'''
Tensor（张量） 是深度学习中最基本、最核心的数据结构。你可以把它简单地理解为 “可以运行在 GPU 上的多维数组”。
Tensor 的计算过程就是在动态地建立计算图。
'''

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.Tensor([1.0])
w.requires_grad = True

"""
因为torch.Tensor函数，所以我们的w是一个张量；
下面我们计算时，返回的也是张量；这样就可以有一其他功能，便可直接函数取出我们想要的值；
item（）是从原来的精度上更加精确；
最外的两个print分别是，模型跑之前，和w值改变之后再跑的预测结果。
w.requires_grad = True   必须要强调需要保留梯度计算的结果
"""


def forward(x):
    return x * w
"""
这里的w实际上是Tensor, *运算符已经被重载了, x和w进行的是数乘
x * w已经构建了一个计算图
如果计算出的输出值有包含权重的,那么他本身也需要进行梯度的计算
"""

def loss(x, y):
    """
    :param x: 输入特征
    :param y: 真实值
    :return: 损失值即误差
    """
    y_pred = forward(x)

    return (y_pred - y) ** 2


print("predict (before training)", 4, forward(4).item())

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        """
        调用loss函数，求损失函数
        l.backward()--反向传播函数，自动会计算并且w的值也会做相应的更新，并且最终会释放掉数据
        这个函数作用就是用于做反向循环--这个时候w的值会有两个：w.data--w的值、w.grad--梯度值即对误差求导的值；
        更新w.data
        w.grad清零，即释放掉计算图，用于下次计算
        """
        l = loss(x, y)
        l.backward()
        print('\tgrad:', x, y, w.grad.item(), w.data)
        '''
        w.grad.item()是将梯度中的数字拿出来，变成Python中的一个标量，防止产生计算图
        计算的时候可以直接使用tenor，但是在权重更新的时候，不可以直接使用
        '''
        w.data = w.data - 0.01 * w.grad.data
        '''
        grad也是Tensor，如果直接进行w.data = w.data - 0.01 * w.grad，相当于在建立一个计算图，所以必须要取到其中的data
        w.data = w.data - 0.01 * w.grad.data 就是只修改grad中data的数值
        '''
        print(w.data)
        w.grad.data.zero_()
        print(w.data, w.grad.item())

    print('progress:', epoch, l.item())

print('predict (after taining)', 4, forward(4).item())