import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt


# ========== 解决中文和负号显示问题 ==========
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False   # 这行是关键！强制使用 ASCII 减号 '-'
# ========== 解决中文显示问题 ==========
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ========== 切换到脚本所在目录 ==========
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# ========== 1. 自定义 Dataset 类 ==========
class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        # 读取数据
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])
        self.len = xy.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


# ========== 2. 准备数据 ==========
dataset = DiabetesDataset('../data/diabetes.csv')
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

print(f"总样本数: {len(dataset)}")
print(f"每个 batch 大小: 32")
print(f"每个 epoch 的 batch 数: {len(train_loader)}")


# ========== 3. 定义模型 ==========
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


# ========== 4. 模型实例化 ==========
model = Model()

# ========== 修改完成：使用 reduction='mean' ==========
criterion = torch.nn.BCELoss(reduction='mean')  # 对应 size_average=True

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# ========== 5. 存储训练历史 ==========
loss_history = []
epoch_losses = []

# ========== 6. 训练 ==========
print("\n" + "=" * 60)
print("开始训练...")
print("=" * 60)

for epoch in range(100):
    epoch_loss_sum = 0
    batch_count = 0

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data

        # 前向传播
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录每个 batch 的损失
        loss_history.append(loss.item())
        epoch_loss_sum += loss.item()
        batch_count += 1

        # 打印训练信息（每 10 个 batch 打印一次）
        if i % 10 == 0:
            print(f'Epoch: {epoch:3d}, Batch: {i:3d}, Loss: {loss.item():.6f}')

    # 记录每个 epoch 的平均损失
    avg_epoch_loss = epoch_loss_sum / batch_count
    epoch_losses.append(avg_epoch_loss)

print("\n训练完成！")
print(f"最终平均损失: {epoch_losses[-1]:.6f}")

# ========== 7. 评估模型 ==========
print("\n" + "=" * 60)
print("模型评估")
print("=" * 60)

# 使用全部数据计算准确率
with torch.no_grad():
    # 获取全部数据
    xy = np.loadtxt('../data/diabetes.csv', delimiter=',', dtype=np.float32)
    x_all = torch.from_numpy(xy[:, :-1])
    y_all = torch.from_numpy(xy[:, [-1]])

    y_pred_all = model(x_all)
    y_pred_class = (y_pred_all > 0.5).float()
    accuracy = (y_pred_class == y_all).float().mean().item()

    print(f"训练集准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")

# ========== 8. 可视化 ==========

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Diabetes Dataset - Dataset & DataLoader 训练', fontsize=16, fontweight='bold')

# -------- 图1：损失曲线（按 batch） --------
ax1 = axes[0]
ax1.plot(range(len(loss_history)), loss_history, 'b-', linewidth=1, alpha=0.7)
ax1.set_xlabel('Batch Step', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('损失曲线 (按 Batch)', fontsize=14)
ax1.grid(True, linestyle='--', alpha=0.3)
ax1.set_yscale('log')

# -------- 图2：损失曲线（按 Epoch） --------
ax2 = axes[1]
ax2.plot(range(len(epoch_losses)), epoch_losses, 'r-', linewidth=2)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Average Loss', fontsize=12)
ax2.set_title('损失曲线 (按 Epoch 平均)', fontsize=14)
ax2.grid(True, linestyle='--', alpha=0.3)
ax2.set_yscale('log')
ax2.annotate(f'Final Loss = {epoch_losses[-1]:.6f}',
             xy=(len(epoch_losses) - 1, epoch_losses[-1]),
             xytext=(len(epoch_losses) * 0.6, epoch_losses[-1] * 5),
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=10, color='red')

# -------- 图3：预测分布 --------
ax3 = axes[2]
with torch.no_grad():
    y_pred_all = model(x_all)
    y_pred_np = y_pred_all.numpy().flatten()
    y_true_np = y_all.numpy().flatten()

positive_preds = y_pred_np[y_true_np == 1]
negative_preds = y_pred_np[y_true_np == 0]

ax3.hist(negative_preds, bins=20, alpha=0.7, color='blue', label='实际类别 0 (未患病)', density=True)
ax3.hist(positive_preds, bins=20, alpha=0.7, color='red', label='实际类别 1 (患病)', density=True)
ax3.axvline(x=0.5, color='green', linestyle='--', linewidth=2, label='决策边界 (0.5)')
ax3.set_xlabel('预测概率 P(y=1)', fontsize=12)
ax3.set_ylabel('密度', fontsize=12)
ax3.set_title('预测概率分布', fontsize=14)
ax3.legend()
ax3.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('dataset_dataloader_result.png', dpi=150, bbox_inches='tight')
print("\n✅ 可视化图片已保存为: dataset_dataloader_result.png")
plt.show()

# ========== 9. 打印最终结果 ==========
print("\n" + "=" * 60)
print("训练结果汇总")
print("=" * 60)
print(f"最终平均损失: {epoch_losses[-1]:.6f}")
print(f"训练集准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print("=" * 60)