import numpy as np
import matplotlib.pyplot as plt
import torch
import os

# ========== 解决中文显示问题 ==========
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ========== 切换到脚本所在目录 ==========
os.chdir(os.path.dirname(os.path.abspath(__file__)))

"""
读取文件，以逗号为分割点
取所有行，和第一列到倒数第二列
取所有行，和最后一列
"""
xy = np.loadtxt('../data/diabetes.csv', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])
y_data = torch.from_numpy(xy[:, [-1]])

print(f"特征数据形状: {x_data.shape}")
print(f"标签数据形状: {y_data.shape}")


class Model(torch.nn.Module):
    """
    super继承Module库
    线性变换；
    8维指的是8个特征-6维是6个特征；8个权重-6个权重
    这是线性网络
    输入数据为8维-输出6维
    输入数据为6维-输出4维
    输入数据为4维-输出1维
    调用sigmoid函数，对数据判断
    """

    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        """
        :param x:输入数据
        :return:预测值
        """
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


"""
模型实例化
调用BCELoss函数--给criterion函数
使用SGD优化器，模型参数初始化（parameters()），学习率为0.01
"""
model = Model()
criterion = torch.nn.BCELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# ========== 存储训练历史 ==========
loss_history = []

for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    loss_history.append(loss.item())

    if epoch % 10 == 0:
        print(f'Epoch: {epoch:3d}, Loss: {loss.item():.6f}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'\n训练完成！最终损失: {loss_history[-1]:.6f}')

# ========== 可视化 ==========

# ========== 图1：损失曲线 ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('糖尿病数据集 - 逻辑回归训练', fontsize=16, fontweight='bold')

# 损失曲线
ax1 = axes[0]
ax1.plot(range(len(loss_history)), loss_history, 'b-', linewidth=2)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss (BCE)', fontsize=12)
ax1.set_title('训练损失下降曲线', fontsize=14)
ax1.grid(True, linestyle='--', alpha=0.3)
ax1.set_yscale('log')

# 标注最终损失
ax1.annotate(f'Final Loss = {loss_history[-1]:.6f}',
             xy=(len(loss_history) - 1, loss_history[-1]),
             xytext=(len(loss_history) * 0.6, loss_history[-1] * 10),
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=10, color='red')

# ========== 图2：预测结果分布 ==========
ax2 = axes[1]

# 计算预测值
with torch.no_grad():
    y_pred = model(x_data)
    y_pred_np = y_pred.numpy().flatten()
    y_true_np = y_data.numpy().flatten()

# 分离正样本和负样本的预测值
positive_preds = y_pred_np[y_true_np == 1]
negative_preds = y_pred_np[y_true_np == 0]

# 绘制预测值的分布（分两个类别）
ax2.hist(negative_preds, bins=20, alpha=0.7, color='blue', label='实际类别 0 (未患病)', density=True)
ax2.hist(positive_preds, bins=20, alpha=0.7, color='red', label='实际类别 1 (患病)', density=True)

# 标记决策边界
ax2.axvline(x=0.5, color='green', linestyle='--', linewidth=2, label='决策边界 (0.5)')

ax2.set_xlabel('预测概率 (P(y=1))', fontsize=12)
ax2.set_ylabel('密度', fontsize=12)
ax2.set_title('预测概率分布 (按真实类别分组)', fontsize=14)
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('diabetes_training_result.png', dpi=150, bbox_inches='tight')
print("✅ 可视化图片已保存为: diabetes_training_result.png")
plt.show()

# ========== 额外的评估指标（打印） ==========
print("\n" + "=" * 60)
print("模型评估结果")
print("=" * 60)

# 计算准确率
with torch.no_grad():
    y_pred = model(x_data)
    y_pred_class = (y_pred > 0.5).float()
    accuracy = (y_pred_class == y_data).float().mean().item()
    print(f"训练集准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")

# 计算混淆矩阵
tp = ((y_pred_class == 1) & (y_data == 1)).sum().item()
tn = ((y_pred_class == 0) & (y_data == 0)).sum().item()
fp = ((y_pred_class == 1) & (y_data == 0)).sum().item()
fn = ((y_pred_class == 0) & (y_data == 1)).sum().item()

print("\n混淆矩阵:")
print(f"             预测为正  预测为负")
print(f"实际为正     {tp:6d}    {fn:6d}")
print(f"实际为负     {fp:6d}    {tn:6d}")

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n精确率 (Precision): {precision:.4f}")
print(f"召回率 (Recall):    {recall:.4f}")
print(f"F1 分数:            {f1:.4f}")
print("=" * 60)

# ========== 图3：特征重要性可视化（可选） ==========
fig2, ax = plt.subplots(figsize=(10, 6))
fig2.suptitle('各层权重分布', fontsize=14, fontweight='bold')

# 提取并绘制各层的权重分布
layer_weights = []
layer_names = ['linear1.weight', 'linear2.weight', 'linear3.weight']

for name, param in model.named_parameters():
    if 'weight' in name:
        weights = param.data.numpy().flatten()
        layer_weights.append(weights)

# 绘制箱线图
bp = ax.boxplot(layer_weights, labels=['Layer 1\n(8→6)', 'Layer 2\n(6→4)', 'Layer 3\n(4→1)'],
                patch_artist=True, showmeans=True)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][1].set_facecolor('lightgreen')
bp['boxes'][2].set_facecolor('lightcoral')

ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('网络层', fontsize=12)
ax.set_ylabel('权重值', fontsize=12)
ax.set_title('各层权重分布（箱线图）', fontsize=14)
ax.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('diabetes_weight_distribution.png', dpi=150, bbox_inches='tight')
print("✅ 权重分布图已保存为: diabetes_weight_distribution.png")
plt.show()