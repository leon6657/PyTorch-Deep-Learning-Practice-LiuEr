# PyTorch 深度学习实践 - 刘二老师课程笔记

> 🚀 个人学习笔记：基于刘二老师《PyTorch深度学习实践》课程的代码实现与详细注释

---

## 📖 课程简介

本仓库是我在学习刘二老师的《PyTorch深度学习实践》课程时整理的代码笔记。课程以**实践为导向**，从零开始，系统地讲解了如何使用 PyTorch 框架进行深度学习模型的构建、训练和优化。

---

## 🗂️ 仓库结构

├── 01_Linear_Model/          # 线性模型
├── 02_Gradient_Descent/      # 梯度下降算法
├── 03_Back_Propagation/      # 反向传播
├── 04_PyTorch_Linear/        # PyTorch 实现线性回归
├── 05_Logistic_Regression/   # 逻辑回归
├── 06_Multiple_Dimension/    # 多维输入
├── 07_Dataset/               # Dataset 数据加载
├── 08_Multi_Classification/  # 多分类问题
├── 09_CNN/                   # 卷积神经网络基础
├── 10_Advanced_CNN/          # 深度卷积神经网络
├── 11_RNN/                   # 循环神经网络基础
├── 12_GRU/                   # GRU 与 LSTM
└── README.md                 # 本文件

```

---

## 📝 内容特点

- ✅ **逐行注释**：每个代码文件都包含详细的思路注释，方便复习和理解
- ✅ **环境配置**：基于 `Python 3.9` + `PyTorch 2.5.1` + `CUDA 12.1`，已验证可运行
- ✅ **学习笔记**：在关键代码段旁记录了课程中的核心思想和自己的理解

---

## ⚙️ 运行环境

### 创建 Conda 环境

```bash
conda create -n pytorch_env python=3.9
conda activate pytorch_env
```

### 安装 PyTorch (GPU 版)

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

### 验证安装

```python
import torch
print(torch.__version__)          # 应显示 2.5.1
print(torch.cuda.is_available())  # 应显示 True
```

---

## 🎯 适用人群

- 正在学习 PyTorch 的入门/进阶开发者
- 需要一份结构清晰、带注释的课程代码参考
- 希望快速回顾课程核心内容的同学

---

## 📚 课程资源

| 资源                 | 链接                                                         |
| :------------------- | :----------------------------------------------------------- |
| **课程视频**         | [B站 - 刘二老师《PyTorch深度学习实践》](https://www.bilibili.com/video/BV1Y7411d7Ys) |
| **官方仓库**         | [LiuEr/PyTorchPractice](https://github.com/LiuEr/PyTorchPractice) |
| **PyTorch 官方文档** | [https://pytorch.org/docs/stable/](https://pytorch.org/docs/stable/) |

---

## 🙏 致谢

感谢刘二老师制作的优质免费课程，用通俗易懂的方式帮助无数开发者打开了深度学习的大门。
同时感谢CSDN博主城峰的代码注释，非常详细，让我理解代码轻松了不少，谢谢。

---

## ⚠️ 免责声明

- 本仓库所有代码仅用于**个人学习和笔记整理**，不用于任何商业用途。
- 部分代码参考了课程官方示例，在注释中已标明出处。
- 如有侵权，请联系删除。

---

## ⭐ 如果这个仓库对你有帮助

欢迎点个 Star 支持一下，也欢迎 Fork 或提 Issue 交流学习！
