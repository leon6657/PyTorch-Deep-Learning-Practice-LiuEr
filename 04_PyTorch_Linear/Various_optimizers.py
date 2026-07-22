import torch
import matplotlib.pyplot as plt
import matplotlib
import os
import time

"""
图1：optimizer_comparison.png
包含 4 个子图：
左上	普通坐标下的损失曲线对比
右上	对数坐标下的损失曲线对比（更清晰）
左下	各优化器的最终损失柱状对比
右下	各优化器的训练时间柱状对比
图2：optimizer_fit_lines.png
展示各个优化器学习到的拟合直线，直观对比它们对同一数据的拟合效果。
"""

'''
优化器	   特点	         适用场景
Adam	收敛快、稳定	   大部分任务首选
SGD	    简单、需调参	   需要精细控制时
RMSprop	自适应学习率	   RNN、非平稳目标
Adagrad	适合稀疏数据	   NLP、推荐系统
LBFGS	拟牛顿法、收敛快	小规模数据
Rprop	只使用梯度符号    批量梯度下降
ASGD	异步SGD	        分布式训练
Adamax	Adam变体	        大规模参数
'''


# ========== 解决中文和负号显示问题 ==========
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

# ========== 切换到当前脚本所在目录 ==========
os.chdir(os.path.dirname(os.path.abspath(__file__)))

"""
不同优化器对比实验
比较：SGD, Adam, RMSprop, Adagrad, Adamax, ASGD, Rprop, LBFGS
"""

# ========== 1. 准备数据 ==========
x_data = torch.Tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0], [8.0], [10.0]])


# ========== 2. 定义模型 ==========
class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


# ========== 3. 定义训练函数 ==========
def train_model(optimizer_name, lr=0.01, epochs=1000):
    """
    使用指定的优化器训练模型
    """
    # 重新初始化模型
    model = LinearModel()
    criterion = torch.nn.MSELoss()

    # 选择优化器
    if optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    elif optimizer_name == 'Adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
    elif optimizer_name == 'Adamax':
        optimizer = torch.optim.Adamax(model.parameters(), lr=lr)
    elif optimizer_name == 'ASGD':
        optimizer = torch.optim.ASGD(model.parameters(), lr=lr)
    elif optimizer_name == 'Rprop':
        optimizer = torch.optim.Rprop(model.parameters(), lr=lr)
    elif optimizer_name == 'LBFGS':
        optimizer = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=20)
    else:
        raise ValueError(f"不支持的优化器: {optimizer_name}")

    # 记录损失历史
    loss_history = []
    start_time = time.time()

    for epoch in range(epochs):
        # LBFGS 需要特殊处理
        if optimizer_name == 'LBFGS':
            def closure():
                optimizer.zero_grad()
                y_pred = model(x_data)
                loss = criterion(y_pred, y_data)
                loss.backward()
                return loss

            loss = optimizer.step(closure)
            loss_history.append(loss.item() if isinstance(loss, torch.Tensor) else loss)
        else:
            # 前向传播
            y_pred = model(x_data)
            loss = criterion(y_pred, y_data)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())

        # 每 100 个 epoch 打印一次
        if (epoch + 1) % 200 == 0:
            print(f"{optimizer_name:8s} | Epoch {epoch + 1:4d} | Loss: {loss_history[-1]:.6f}")

    end_time = time.time()
    training_time = end_time - start_time

    # 获取最终参数
    final_w = model.linear.weight.item()
    final_b = model.linear.bias.item()

    return {
        'name': optimizer_name,
        'loss_history': loss_history,
        'final_w': final_w,
        'final_b': final_b,
        'time': training_time,
        'final_loss': loss_history[-1],
        'model': model
    }


# ========== 4. 训练所有优化器 ==========
print("=" * 70)
print("开始训练不同优化器...")
print("=" * 70)

optimizers_to_test = ['SGD', 'Adam', 'RMSprop', 'Adagrad', 'Adamax', 'ASGD', 'Rprop', 'LBFGS']
results = {}

for opt_name in optimizers_to_test:
    print(f"\n正在训练 {opt_name}...")
    results[opt_name] = train_model(opt_name, lr=0.01, epochs=1000)

print("\n" + "=" * 70)
print("训练完成！")
print("=" * 70)

# ========== 5. 打印结果汇总 ==========
print("\n📊 结果汇总:")
print("-" * 70)
print(f"{'优化器':<10} {'最终权重 w':<12} {'最终偏置 b':<12} {'最终损失':<14} {'耗时 (s)':<10}")
print("-" * 70)
for name, result in results.items():
    print(
        f"{name:<10} {result['final_w']:<12.6f} {result['final_b']:<12.6f} {result['final_loss']:<14.6f} {result['time']:<10.4f}")
print("-" * 70)

# ========== 6. 可视化 ==========

# 选择颜色
colors = {
    'SGD': '#1f77b4',
    'Adam': '#ff7f0e',
    'RMSprop': '#2ca02c',
    'Adagrad': '#d62728',
    'Adamax': '#9467bd',
    'ASGD': '#8c564b',
    'Rprop': '#e377c2',
    'LBFGS': '#7f7f7f'
}

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('不同优化器训练对比', fontsize=18, fontweight='bold')

# -------- 子图 1：损失曲线对比 --------
ax1 = axes[0, 0]
for name, result in results.items():
    loss_hist = result['loss_history']
    ax1.plot(range(len(loss_hist)), loss_hist,
             color=colors[name], linewidth=1.5, label=name, alpha=0.8)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('损失曲线对比（普通坐标）', fontsize=14)
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, linestyle='--', alpha=0.3)

# -------- 子图 2：损失曲线对比（对数坐标） --------
ax2 = axes[0, 1]
for name, result in results.items():
    loss_hist = result['loss_history']
    ax2.plot(range(len(loss_hist)), loss_hist,
             color=colors[name], linewidth=1.5, label=name, alpha=0.8)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss (对数坐标)', fontsize=12)
ax2.set_title('损失曲线对比（对数坐标）', fontsize=14)
ax2.set_yscale('log')
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, linestyle='--', alpha=0.3)

# -------- 子图 3：最终损失对比（柱状图） --------
ax3 = axes[1, 0]
names = list(results.keys())
final_losses = [results[name]['final_loss'] for name in names]
bars = ax3.bar(names, final_losses, color=[colors[name] for name in names])
ax3.set_xlabel('优化器', fontsize=12)
ax3.set_ylabel('最终损失', fontsize=12)
ax3.set_title('最终损失对比', fontsize=14)
ax3.set_yscale('log')
# 在柱子上标注数值
for bar, loss in zip(bars, final_losses):
    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.1,
             f'{loss:.6f}', ha='center', va='bottom', fontsize=8, rotation=0)
ax3.grid(True, linestyle='--', alpha=0.3, axis='y')

# -------- 子图 4：训练时间对比（柱状图） --------
ax4 = axes[1, 1]
times = [results[name]['time'] for name in names]
bars = ax4.bar(names, times, color=[colors[name] for name in names])
ax4.set_xlabel('优化器', fontsize=12)
ax4.set_ylabel('训练时间 (秒)', fontsize=12)
ax4.set_title('训练时间对比', fontsize=14)
for bar, t in zip(bars, times):
    ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.05,
             f'{t:.2f}s', ha='center', va='bottom', fontsize=8)
ax4.grid(True, linestyle='--', alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('optimizer_comparison.png', dpi=150, bbox_inches='tight')
print("\n✅ 可视化图片已保存为: optimizer_comparison.png")

plt.show()

# ========== 7. 额外的拟合直线对比图 ==========
fig2, ax = plt.subplots(figsize=(10, 6))
fig2.suptitle('不同优化器拟合结果对比', fontsize=16, fontweight='bold')

# 绘制真实数据点
ax.scatter(x_data.numpy(), y_data.numpy(), color='black', s=80,
           label='真实数据', zorder=5)

# 绘制每个优化器的拟合直线
x_test = torch.linspace(0, 6, 100).reshape(-1, 1)

for name, result in results.items():
    model = result['model']
    with torch.no_grad():
        y_pred = model(x_test)
    ax.plot(x_test.numpy(), y_pred.numpy(),
            color=colors[name], linewidth=1.5, label=f'{name} (w={result["final_w"]:.3f})', alpha=0.7)

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('各优化器的拟合直线', fontsize=14)
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, linestyle='--', alpha=0.3)
ax.set_xlim(0, 6)
ax.set_ylim(0, 12)

plt.tight_layout()
plt.savefig('optimizer_fit_lines.png', dpi=150, bbox_inches='tight')
print("✅ 拟合对比图已保存为: optimizer_fit_lines.png")

plt.show()

# ========== 8. 总结建议 ==========
print("\n" + "=" * 70)
print("📌 不同优化器特点总结")
print("=" * 70)

optimizer_summary = {
    'SGD': '标准随机梯度下降，最简单，可能需要更多epoch，适合小批量数据',
    'Adam': '自适应学习率，综合表现最好，大多数情况下的首选',
    'RMSprop': '自适应学习率，适合非平稳目标，RNN中常用',
    'Adagrad': '自适应学习率，适合稀疏数据，但学习率会逐渐减小到0',
    'Adamax': 'Adam的变体，对大规模参数更稳定',
    'ASGD': '异步随机梯度下降，适合分布式训练',
    'Rprop': '只使用梯度符号，不使用梯度大小，适合批量梯度下降',
    'LBFGS': '拟牛顿法，收敛快但计算量大，适合小规模数据'
}

for name, summary in optimizer_summary.items():
    final_loss = results[name]['final_loss']
    time_taken = results[name]['time']
    print(f"\n{name:8s} | Loss: {final_loss:.6f} | Time: {time_taken:.4f}s")
    print(f"         {summary}")

print("\n" + "=" * 70)
print("💡 推荐：对于大多数深度学习任务，Adam 是首选的优化器。")
print("      SGD 适合需要精细调参的场景，配合学习率衰减效果更好。")
print("=" * 70)