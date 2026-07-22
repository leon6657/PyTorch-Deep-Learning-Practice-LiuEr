import torch
import matplotlib.pyplot as plt
import matplotlib
import os

"""
特别注意⚠️
# 对每个参数分别做梯度清零（非常重要！）
w1.grad.data.zero_()
w2.grad.data.zero_()
b.grad.data.zero_()
"""
# ========== 解决中文和负号显示问题 ==========
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

# ========== 切换到当前脚本所在目录 ==========
os.chdir(os.path.dirname(os.path.abspath(__file__)))

"""
二次模型（Quadratic Model）
\hat{y} = w1 * x^2 + w2 * x + b

目标：从数据中学习到 w1, w2, b 的最优值
真实函数：y = 2x^2 + 3x + 1  （我们构造数据时用这个）
"""

# ========== 1. 准备数据 ==========
# 为了更清晰地展示二次效果，生成更多数据点
x_data = [1.0, 2.0, 3.0, 4.0, 5.0]
# 真实函数：y = 2x^2 - 3x + 1（故意取一些值让二次效果明显）
# 对应：x=1→0, x=2→3, x=3→10, x=4→21, x=5→36
y_data = [0.0, 3.0, 10.0, 21.0, 36.0]

# ========== 2. 初始化参数（张量） ==========
w1 = torch.Tensor([1.0])   # x² 的系数
w1.requires_grad = True
w2 = torch.Tensor([1.0])   # x 的系数
w2.requires_grad = True
b = torch.Tensor([1.0])    # 偏置
b.requires_grad = True

print(f"初始参数: w1={w1.item():.4f}, w2={w2.item():.4f}, b={b.item():.4f}")

# ========== 3. 定义模型、损失函数 ==========

def forward(x):
    """二次模型：ŷ = w1 * x² + w2 * x + b"""
    return w1 * x**2 + w2 * x + b

def loss(x, y):
    """均方误差损失"""
    y_pred = forward(x)
    return (y_pred - y) ** 2

# ========== 4. 训练前预测 ==========
print("\n" + "="*50)
print("训练前预测:")
print(f"f(4) = {forward(4).item():.4f}")

# ========== 5. 训练 ==========
print("\n开始训练...")
print("-"*50)

# 存储训练历史（用于可视化）
loss_history = []
w1_history = [w1.item()]
w2_history = [w2.item()]
b_history = [b.item()]
epoch_losses = []

for epoch in range(200):
    epoch_loss_sum = 0
    epoch_count = 0

    for x, y in zip(x_data, y_data):
        # 前向传播 + 计算损失
        l = loss(x, y)

        # 反向传播（自动计算梯度）
        l.backward()

        # 更新参数（注意：用 .data 操作，避免构建计算图）
        learning_rate = 0.01
        w1.data -= learning_rate * w1.grad.data
        w2.data -= learning_rate * w2.grad.data
        b.data -= learning_rate * b.grad.data

        # 梯度清零（非常重要！）
        w1.grad.data.zero_()
        w2.grad.data.zero_()
        b.grad.data.zero_()

        # 记录每个样本的损失
        epoch_loss_sum += l.item()
        epoch_count += 1

    # 每个 epoch 结束后记录平均损失和参数
    avg_loss = epoch_loss_sum / epoch_count
    loss_history.append(avg_loss)
    w1_history.append(w1.item())
    w2_history.append(w2.item())
    b_history.append(b.item())
    epoch_losses.append(avg_loss)

    # 每 20 个 epoch 打印一次
    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d}: loss={avg_loss:.6f}, w1={w1.item():.4f}, w2={w2.item():.4f}, b={b.item():.4f}")

print("-"*50)
print("训练完成！")
print(f"最终参数: w1={w1.item():.4f}, w2={w2.item():.4f}, b={b.item():.4f}")
print(f"f(4) = {forward(4).item():.4f}")

# ========== 6. 可视化 ==========

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('二次模型（Quadratic Model）训练过程', fontsize=16, fontweight='bold')

# -------- 子图 1：损失曲线 --------
ax1 = axes[0, 0]
ax1.plot(range(len(loss_history)), loss_history, 'b-', linewidth=2)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('训练损失下降曲线', fontsize=14)
ax1.grid(True, linestyle='--', alpha=0.3)
ax1.set_yscale('log')
ax1.annotate(f'Final Loss = {loss_history[-1]:.6f}',
             xy=(len(loss_history)-1, loss_history[-1]),
             xytext=(len(loss_history)*0.6, loss_history[-1]*5),
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=10, color='red')

# -------- 子图 2：参数 w1 的变化 --------
ax2 = axes[0, 1]
ax2.plot(range(len(w1_history)), w1_history, 'r-', linewidth=2, label='w1')
ax2.axhline(y=2.0, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='目标 w1=2.0')
ax2.set_xlabel('更新步数', fontsize=12)
ax2.set_ylabel('w1 值', fontsize=12)
ax2.set_title('参数 w1 的变化', fontsize=14)
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.3)

# -------- 子图 3：参数 w2 和 b 的变化 --------
ax3 = axes[1, 0]
ax3.plot(range(len(w2_history)), w2_history, 'g-', linewidth=2, label='w2')
ax3.plot(range(len(b_history)), b_history, 'm-', linewidth=2, label='b')
ax3.axhline(y=-3.0, color='g', linestyle='--', linewidth=1.5, alpha=0.7, label='目标 w2=-3.0')
ax3.axhline(y=1.0, color='m', linestyle='--', linewidth=1.5, alpha=0.7, label='目标 b=1.0')
ax3.set_xlabel('更新步数', fontsize=12)
ax3.set_ylabel('参数值', fontsize=12)
ax3.set_title('参数 w2 和 b 的变化', fontsize=14)
ax3.legend()
ax3.grid(True, linestyle='--', alpha=0.3)

# -------- 子图 4：拟合结果 --------
ax4 = axes[1, 1]

# 真实数据点
ax4.scatter(x_data, y_data, color='red', s=100, label='真实数据', zorder=5)

# 训练后的拟合曲线
x_test = torch.linspace(0, 6, 100)
with torch.no_grad():
    y_test = forward(x_test).numpy()
ax4.plot(x_test.numpy(), y_test, 'b-', linewidth=2.5,
         label=f'拟合曲线 (w1={w1.item():.2f}, w2={w2.item():.2f}, b={b.item():.2f})')

# 绘制目标曲线（真实函数）用于对比
def target_function(x):
    return 2 * x**2 - 3 * x + 1

y_target = target_function(x_test.numpy())
ax4.plot(x_test.numpy(), y_target, 'r--', linewidth=2, alpha=0.6,
         label='目标曲线 (y=2x²-3x+1)')

# 初始曲线（用初始参数）
w1_init = 1.0
w2_init = 1.0
b_init = 1.0
y_init = w1_init * x_test.numpy()**2 + w2_init * x_test.numpy() + b_init
ax4.plot(x_test.numpy(), y_init, 'g--', linewidth=1.5, alpha=0.5,
         label=f'初始曲线 (w1=1, w2=1, b=1)')

# 标注预测值
with torch.no_grad():
    pred_4 = forward(4).item()
ax4.scatter(4, pred_4, color='blue', s=120, marker='*',
            label=f'预测值: f(4)={pred_4:.4f}', zorder=6)
ax4.annotate(f'f(4)={pred_4:.4f}',
             xy=(4, pred_4),
             xytext=(4.3, pred_4 - 2),
             fontsize=10, color='blue')

ax4.set_xlabel('x', fontsize=12)
ax4.set_ylabel('y', fontsize=12)
ax4.set_title('拟合结果对比', fontsize=14)
ax4.legend(loc='upper left')
ax4.grid(True, linestyle='--', alpha=0.3)
ax4.set_xlim(0, 6)
ax4.set_ylim(-5, 45)

plt.tight_layout()

# 保存图片
plt.savefig('quadratic_model_result.png', dpi=150, bbox_inches='tight')
print("\n✅ 可视化图片已保存为: quadratic_model_result.png")

plt.show()

# ========== 7. 打印最终结果 ==========
print("\n" + "="*50)
print("📊 最终结果汇总")
print("="*50)
print(f"真实函数: y = 2x² - 3x + 1")
print(f"学习到的函数: y = {w1.item():.4f}x² + {w2.item():.4f}x + {b.item():.4f}")
print(f"最终损失: {loss_history[-1]:.6f}")
print(f"f(4) 预测值: {forward(4).item():.4f} (真实值应为: 21.0)")
print("="*50)