import os
import ssl
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

# ========== 1. 设置代理（在 import torchvision 之前） ==========
# 替换成你实际的代理地址和端口
proxy_url = 'http://127.0.0.1:7890'  # 改成你实际的代理地址

os.environ['HTTP_PROXY'] = proxy_url
os.environ['HTTPS_PROXY'] = proxy_url
os.environ['http_proxy'] = proxy_url   # 小写版本，某些库需要
os.environ['https_proxy'] = proxy_url  # 小写版本，某些库需要

# ========== 2. 禁用 SSL 验证（配合代理使用） ==========
ssl._create_default_https_context = ssl._create_unverified_context

# ========== 3. 下载数据集 ==========
print("开始下载 MNIST 数据集...")
print(f"使用代理: {proxy_url}")

train_dataset = datasets.MNIST(root='../dataset/mnist',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='../dataset/mnist',
                              train=False,
                              transform=transforms.ToTensor(),
                              download=True)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=32,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=32,
                         shuffle=False)

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")