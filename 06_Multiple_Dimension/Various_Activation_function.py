import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time

# ========== 解决中文显示问题 ==========
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ========== 切换到脚本所在目录 ==========
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ========== 1. 加载数据 ==========
xy = np.loadtxt('../data/diabetes.csv', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])
y_data = torch.from_numpy(xy[:, [-1]])

print(f"特征数据形状: {x_data.shape}")
print(f"标签数据形状: {y_data.shape}")
print(f"总样本数: {len(x_data)}")


# ========== 2. 定义模型生成函数 ==========
def create_model(activation_name):
    """
    根据激活函数名称创建模型
    """

    class FlexibleModel(nn.Module):
        def __init__(self, act_name):
            super(FlexibleModel, self).__init__()
            self.linear1 = nn.Linear(8, 6)
            self.linear2 = nn.Linear(6, 4)
            self.linear3 = nn.Linear(4, 1)

            # 选择激活函数
            if act_name == 'ReLU':
                self.activation = nn.ReLU()
            elif act_name == 'Sigmoid':
                self.activation = nn.Sigmoid()
            elif act_name == 'Tanh':
                self.activation = nn.Tanh()
            elif act_name == 'LeakyReLU':
                self.activation = nn.LeakyReLU(0.01)
            elif act_name == 'PReLU':
                self.activation = nn.PReLU()
            elif act_name == 'ELU':
                self.activation = nn.ELU(alpha=1.0)
            elif act_name == 'SELU':
                self.activation = nn.SELU()
            elif act_name == 'Softplus':
                self.activation = nn.Softplus()
            elif act_name == 'Softsign':
                self.activation = nn.Softsign()
            elif act_name == 'RReLU':
                self.activation = nn.RReLU()
            elif act_name == 'ReLU6':
                self.activation = nn.ReLU6()
            elif act_name == 'Hardtanh':
                self.activation = nn.Hardtanh()
            elif act_name == 'LogSigmoid':
                self.activation = nn.LogSigmoid()
            elif act_name == 'Tanhshrink':
                self.activation = nn.Tanhshrink()
            elif act_name == 'Hardshrink':
                self.activation = nn.Hardshrink()
            elif act_name == 'Threshold':
                self.activation = nn.Threshold(threshold=0.1, value=0.0)
            else:  # 默认使用 ReLU
                self.activation = nn.ReLU()

        def forward(self, x):
            x = self.activation(self.linear1(x))
            x = self.activation(self.linear2(x))
            x = torch.sigmoid(self.linear3(x))  # 最后一层固定使用 Sigmoid
            return x

    return FlexibleModel(activation_name)


# ========== 3. 定义训练函数 ==========
def train_model(activation_name, epochs=200, lr=0.01):
    """
    训练指定激活函数的模型
    """
    model = create_model(activation_name)
    criterion = nn.BCELoss(size_average=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    loss_history = []
    start_time = time.time()

    for epoch in range(epochs):
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)
        loss_history.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    end_time = time.time()

    # 计算最终准确率
    with torch.no_grad():
        y_pred = model(x_data)
        y_pred_class = (y_pred > 0.5).float()
        accuracy = (y_pred_class == y_data).float().mean().item()

    return {
        'name': activation_name,
        'loss_history': loss_history,
        'final_loss': loss_history[-1],
        'accuracy': accuracy,
        'time': end_time - start_time,
        'model': model
    }


# ========== 4. 训练所有激活函数 ==========
print("\n" + "=" * 70)
print("开始训练不同激活函数...")
print("=" * 70)

# 选择要比较的激活函数（为了清晰，选择代表性的几个）
activations_to_test = [
    'Sigmoid',  # 传统
    'ReLU',  # 最常用
    'LeakyReLU',  # ReLU的改进
    'ELU',  # 指数线性单元
    'Tanh',  # 双曲正切
    'PReLU',  # 可学习的ReLU
    'SELU',  # 自归一化
    'Softplus',  # ReLU的平滑版本
]

results = {}
for act_name in activations_to_test:
    print(f"\n正在训练 {act_name}...")
    results[act_name] = train_model(act_name, epochs=300, lr=0.01)
    print(f"  {act_name:10s} | 最终损失: {results[act_name]['final_loss']:.6f} | "
          f"准确率: {results[act_name]['accuracy']:.4f} | "
          f"耗时: {results[act_name]['time']:.2f}s")

# ========== 5. 选择颜色 ==========
colors = {
    'Sigmoid': '#1f77b4',
    'ReLU': '#ff7f0e',
    'LeakyReLU': '#2ca02c',
    'ELU': '#d62728',
    'Tanh': '#9467bd',
    'PReLU': '#8c564b',
    'SELU': '#e377c2',
    'Softplus': '#7f7f7f'
}

# ========== 6. 可视化 ==========
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('不同激活函数训练对比 (糖尿病数据集)', fontsize=18, fontweight='bold')

# -------- 子图1：损失曲线对比 --------
ax1 = axes[0, 0]
for name, result in results.items():
    ax1.plot(range(len(result['loss_history'])), result['loss_history'],
             color=colors[name], linewidth=1.5, label=name, alpha=0.8)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('损失曲线对比', fontsize=14)
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, linestyle='--', alpha=0.3)
ax1.set_yscale('log')

# -------- 子图2：最终损失对比 --------
ax2 = axes[0, 1]
names = list(results.keys())
final_losses = [results[name]['final_loss'] for name in names]
bars = ax2.bar(names, final_losses, color=[colors[name] for name in names])
ax2.set_xlabel('激活函数', fontsize=12)
ax2.set_ylabel('最终损失', fontsize=12)
ax2.set_title('最终损失对比', fontsize=14)
ax2.set_yscale('log')
for bar, loss in zip(bars, final_losses):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.1,
             f'{loss:.4f}', ha='center', va='bottom', fontsize=8)
ax2.grid(True, linestyle='--', alpha=0.3, axis='y')

# -------- 子图3：准确率对比 --------
ax3 = axes[0, 2]
accuracies = [results[name]['accuracy'] for name in names]
bars = ax3.bar(names, accuracies, color=[colors[name] for name in names])
ax3.set_xlabel('激活函数', fontsize=12)
ax3.set_ylabel('准确率', fontsize=12)
ax3.set_title('训练集准确率对比', fontsize=14)
ax3.set_ylim(0, 1)
for bar, acc in zip(bars, accuracies):
    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
             f'{acc:.2%}', ha='center', va='bottom', fontsize=8)
ax3.grid(True, linestyle='--', alpha=0.3, axis='y')
ax3.axhline(y=0.65, color='red', linestyle='--', linewidth=1, alpha=0.5, label='基准线')

# -------- 子图4：激活函数形状 --------
ax4 = axes[1, 0]
# 绘制各激活函数的形状
x_act = np.linspace(-3, 3, 100)
for name in names:
    if name == 'ReLU':
        y_act = np.maximum(0, x_act)
    elif name == 'Sigmoid':
        y_act = 1 / (1 + np.exp(-x_act))
    elif name == 'Tanh':
        y_act = np.tanh(x_act)
    elif name == 'LeakyReLU':
        y_act = np.where(x_act > 0, x_act, 0.01 * x_act)
    elif name == 'ELU':
        y_act = np.where(x_act > 0, x_act, 1.0 * (np.exp(x_act) - 1))
    elif name == 'PReLU':
        y_act = np.where(x_act > 0, x_act, 0.25 * x_act)  # 近似
    elif name == 'SELU':
        y_act = 1.0507 * np.where(x_act > 0, x_act, 1.67326 * (np.exp(x_act) - 1))
    elif name == 'Softplus':
        y_act = np.log(1 + np.exp(x_act))
    else:
        continue
    ax4.plot(x_act, y_act, color=colors[name], linewidth=2, label=name, alpha=0.7)
ax4.set_xlabel('x', fontsize=12)
ax4.set_ylabel('f(x)', fontsize=12)
ax4.set_title('激活函数形状对比', fontsize=14)
ax4.legend(loc='upper left', fontsize=8)
ax4.grid(True, linestyle='--', alpha=0.3)
ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax4.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax4.set_xlim(-3, 3)
ax4.set_ylim(-2, 4)

# -------- 子图5：训练时间对比 --------
ax5 = axes[1, 1]
times = [results[name]['time'] for name in names]
bars = ax5.bar(names, times, color=[colors[name] for name in names])
ax5.set_xlabel('激活函数', fontsize=12)
ax5.set_ylabel('训练时间 (秒)', fontsize=12)
ax5.set_title('训练时间对比', fontsize=14)
for bar, t in zip(bars, times):
    ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.05,
             f'{t:.2f}s', ha='center', va='bottom', fontsize=8)
ax5.grid(True, linestyle='--', alpha=0.3, axis='y')

# -------- 子图6：结果汇总表格 --------
ax6 = axes[1, 2]
ax6.axis('tight')
ax6.axis('off')

# 准备表格数据
table_data = []
for name in names:
    r = results[name]
    table_data.append([
        name,
        f"{r['final_loss']:.4f}",
        f"{r['accuracy']:.2%}",
        f"{r['time']:.2f}s"
    ])

columns = ['激活函数', '最终损失', '准确率', '训练时间']
table = ax6.table(cellText=table_data, colLabels=columns,
                  cellLoc='center', loc='center',
                  colWidths=[0.15, 0.15, 0.15, 0.15])
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.5)

# 高亮最佳结果
best_accuracy = max(accuracies)
best_idx = accuracies.index(best_accuracy)
for i, (row, acc) in enumerate(zip(table_data, accuracies)):
    if acc == best_accuracy:
        for j in range(4):
            table[(i + 1, j)].set_facecolor('#90EE90')

ax6.set_title('训练结果汇总', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('activation_comparison.png', dpi=150, bbox_inches='tight')
print("\n✅ 可视化图片已保存为: activation_comparison.png")
plt.show()

# ========== 7. 打印详细结果 ==========
print("\n" + "=" * 70)
print("📊 不同激活函数训练结果汇总")
print("=" * 70)
print(f"{'激活函数':<12} {'最终损失':<12} {'准确率':<10} {'训练时间':<10} {'收敛速度'}")
print("-" * 70)

# 计算收敛速度（达到80%最终损失所需的epoch数）
for name in names:
    r = results[name]
    loss_hist = r['loss_history']
    final_loss = r['final_loss']

    # 找达到最终损失80%的epoch
    threshold = final_loss * 1.2
    converged_epoch = 0
    for i, loss in enumerate(loss_hist):
        if loss < threshold:
            converged_epoch = i
            break

    convergence = f"{converged_epoch} epochs"
    if converged_epoch == 0:
        convergence = "快速收敛"

    print(f"{name:<12} {r['final_loss']:<12.6f} {r['accuracy']:<10.2%} {r['time']:<10.2f} {convergence}")

print("-" * 70)

best_acc_name = names[accuracies.index(max(accuracies))]
best_loss_name = names[final_losses.index(min(final_losses))]
print(f"\n🏆 最佳准确率: {best_acc_name} ({max(accuracies):.2%})")
print(f"🏆 最低损失: {best_loss_name} ({min(final_losses):.6f})")

# ========== 8. 不同激活函数的推荐场景 ==========
print("\n" + "=" * 70)
print("💡 激活函数选择建议")
print("=" * 70)
recommendations = {
    'ReLU': '✅ 最常用，计算简单，梯度不饱和，适合大多数场景',
    'LeakyReLU': '✅ 解决了ReLU的死亡神经元问题，适合深层网络',
    'ELU': '✅ 输出接近零均值，收敛更快，但计算稍复杂',
    'SELU': '✅ 自归一化，适合深层全连接网络',
    'Sigmoid': '⚠️ 传统方法，梯度饱和，一般不用于隐藏层',
    'Tanh': '⚠️ 输出零中心，比Sigmoid好，但仍有梯度饱和问题',
    'PReLU': '✅ 可学习的参数，能适应不同数据，但参数增多',
    'Softplus': '⚠️ ReLU的平滑近似，计算较慢'
}

for name in names:
    if name in recommendations:
        print(f"\n{name:10s}: {recommendations[name]}")

print("\n" + "=" * 70)