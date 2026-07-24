import torch

"""
数据转换：batch，channel，width，height
卷积核2*2，MaxPool2d函数
搭建模型
"""
input = [3, 4, 6, 5,
         2, 4, 6, 8,
         1, 6, 7, 8,
         9, 7, 4, 6,
         ]
input = torch.Tensor(input).view(1, 1, 4, 4)    #将一维数组转换为 4D 张量

maxpooling_layer = torch.nn.MaxPool2d(kernel_size=2)
"""
创建最大池化层
MaxPool2d 的作用：在 2×2 的窗口内，取最大值作为输出。
"""


output = maxpooling_layer(input)
print(output)