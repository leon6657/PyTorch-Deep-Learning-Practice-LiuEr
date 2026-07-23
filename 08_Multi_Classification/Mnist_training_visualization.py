import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import warnings

# ========== 忽略警告（可选） ==========
warnings.filterwarnings('ignore')

# ========== 解决中文和负号显示问题 ==========
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ========== 切换到脚本所在目录并创建保存目录 ==========
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# 创建 figures 子目录用于保存图片
figures_dir = os.path.join(script_dir, 'figures')
os.makedirs(figures_dir, exist_ok=True)
print(f"脚本所在目录: {script_dir}")
print(f"图片保存目录: {figures_dir}")

# ========== 1. 准备数据 ==========
batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 检查数据是否已存在
data_dir = os.path.join(script_dir, '../dataset/mnist/')
data_dir = os.path.normpath(data_dir)

try:
    train_dataset = datasets.MNIST(root=data_dir,
                                   train=True,
                                   download=True,
                                   transform=transform)
    test_dataset = datasets.MNIST(root=data_dir,
                                  train=False,
                                  download=True,
                                  transform=transform)
    print(f"✅ 数据加载成功！")
except Exception as e:
    print(f"❌ 数据加载失败: {e}")
    print("请检查网络连接或手动下载数据集")
    exit()

train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=batch_size)

test_loader = DataLoader(test_dataset,
                         shuffle=False,
                         batch_size=batch_size)

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")


# ========== 2. 定义模型 ==========
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)


# ========== 3. 初始化模型 ==========
model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"使用设备: {device}")

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# ========== 4. 存储训练历史 ==========
train_loss_history = []
test_acc_history = []
epoch_loss_history = []
epoch_acc_history = []

start_time = time.time()


# ========== 5. 训练函数 ==========
def train(epoch):
    running_loss = 0.0
    epoch_loss = 0.0
    batch_count = 0

    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        epoch_loss += loss.item()
        batch_count += 1

        # 记录每个 batch 的损失
        train_loss_history.append(loss.item())

        # 每 300 个 batch 打印一次
        if batch_idx % 300 == 299:
            avg_loss = running_loss / 300
            print(f'[{epoch + 1}, {batch_idx + 1:5d}] loss: {avg_loss:.3f}')
            running_loss = 0.0

    # 记录每个 epoch 的平均损失
    avg_epoch_loss = epoch_loss / batch_count
    epoch_loss_history.append(avg_epoch_loss)


# ========== 6. 测试函数 ==========
def test(epoch):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    test_acc_history.append(accuracy)
    epoch_acc_history.append(accuracy)
    print(f'Accuracy on test set: {accuracy:.2f} %')
    return accuracy


# ========== 7. 开始训练 ==========
print("\n" + "=" * 60)
print("开始训练...")
print("=" * 60)

for epoch in range(10):
    train(epoch)
    test(epoch)

end_time = time.time()
print(f"\n训练完成！总用时: {end_time - start_time:.2f} 秒")

# ========== 8. 可视化 ==========

print("\n正在生成可视化图片...")

try:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('MNIST 手写数字识别 - 训练过程可视化', fontsize=18, fontweight='bold')

    # -------- 子图1：训练损失曲线（按 Batch） --------
    ax1 = axes[0, 0]
    ax1.plot(range(len(train_loss_history)), train_loss_history, 'b-', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Batch Step', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('训练损失 (按 Batch)', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.set_yscale('log')

    # -------- 子图2：训练损失曲线（按 Epoch） --------
    ax2 = axes[0, 1]
    ax2.plot(range(1, 11), epoch_loss_history, 'r-o', linewidth=2, markersize=8)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Average Loss', fontsize=12)
    ax2.set_title('训练损失 (按 Epoch 平均)', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.set_xticks(range(1, 11))
    for i, loss in enumerate(epoch_loss_history):
        ax2.annotate(f'{loss:.3f}', xy=(i + 1, loss), xytext=(i + 1, loss + 0.02),
                     ha='center', fontsize=8)

    # -------- 子图3：测试准确率 --------
    ax3 = axes[0, 2]
    ax3.plot(range(1, 11), epoch_acc_history, 'g-o', linewidth=2, markersize=8)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Accuracy (%)', fontsize=12)
    ax3.set_title('测试集准确率', fontsize=14)
    ax3.grid(True, linestyle='--', alpha=0.3)
    ax3.set_ylim(80, 100)
    ax3.set_xticks(range(1, 11))
    for i, acc in enumerate(epoch_acc_history):
        ax3.annotate(f'{acc:.1f}%', xy=(i + 1, acc), xytext=(i + 1, acc - 2),
                     ha='center', fontsize=8, color='green')

    # -------- 子图4：展示训练集预测结果 --------
    ax4 = axes[1, 0]
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    images_cpu = images.cpu()
    predicted_cpu = predicted.cpu()
    labels_cpu = labels.cpu()

    for i in range(16):
        ax4.subplot(4, 4, i + 1)
        ax4.imshow(images_cpu[i][0], cmap='gray')
        color = 'green' if predicted_cpu[i] == labels_cpu[i] else 'red'
        ax4.set_title(f'P:{predicted_cpu[i].item()}', color=color, fontsize=10)
        ax4.axis('off')
    ax4.set_title('预测结果 (绿色=正确, 红色=错误)', fontsize=12, pad=10)

    # -------- 子图5：混淆矩阵 --------
    ax5 = axes[1, 1]
    conf_matrix = torch.zeros(10, 10)
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                conf_matrix[t.long(), p.long()] += 1

    ax5.imshow(conf_matrix.numpy(), cmap='Blues', interpolation='nearest')
    ax5.set_xlabel('预测类别', fontsize=12)
    ax5.set_ylabel('真实类别', fontsize=12)
    ax5.set_title('混淆矩阵', fontsize=14)
    ax5.set_xticks(range(10))
    ax5.set_yticks(range(10))
    for i in range(10):
        for j in range(10):
            ax5.text(j, i, int(conf_matrix[i, j].item()),
                     ha='center', va='center', fontsize=8,
                     color='white' if conf_matrix[i, j] > 50 else 'black')

    # -------- 子图6：训练过程总结 --------
    ax6 = axes[1, 2]
    ax6.axis('off')
    summary_text = f"""
📊 训练总结

训练总用时: {end_time - start_time:.2f} 秒

最终训练损失: {epoch_loss_history[-1]:.4f}
最终测试准确率: {epoch_acc_history[-1]:.2f}%

最高测试准确率: {max(epoch_acc_history):.2f}%
最低测试准确率: {min(epoch_acc_history):.2f}%

模型结构:
784 → 512 → 256 → 128 → 64 → 10

优化器: SGD (lr=0.01, momentum=0.5)
损失函数: CrossEntropyLoss
Batch Size: 64
Epochs: 10

设备: {device}
训练集: {len(train_dataset)}
测试集: {len(test_dataset)}
"""
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
             fontsize=12, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # ========== 保存主图 ==========
    save_path1 = os.path.join(figures_dir, 'mnist_training_visualization.png')
    plt.savefig(save_path1, dpi=150, bbox_inches='tight')
    print(f"✅ 主图已保存: {save_path1}")

    # 同时保存到脚本所在目录
    save_path1_alt = os.path.join(script_dir, 'mnist_training_visualization.png')
    plt.savefig(save_path1_alt, dpi=150, bbox_inches='tight')
    print(f"✅ 主图已保存: {save_path1_alt}")

    plt.show()

except Exception as e:
    print(f"❌ 可视化生成失败: {e}")
    import traceback
    traceback.print_exc()

# ========== 9. 额外：展示错误样本 ==========
print("\n" + "=" * 60)
print("错误样本分析")
print("=" * 60)

try:
    wrong_images = []
    wrong_labels = []
    wrong_preds = []

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            wrong_mask = (predicted != labels)
            if wrong_mask.sum() > 0:
                wrong_images.extend(images[wrong_mask].cpu())
                wrong_labels.extend(labels[wrong_mask].cpu())
                wrong_preds.extend(predicted[wrong_mask].cpu())

    if len(wrong_images) > 0:
        print(f"测试集中共有 {len(wrong_images)} 个错误样本")
        print(f"错误率: {len(wrong_images) / len(test_dataset) * 100:.2f}%")

        fig2, axes2 = plt.subplots(3, 3, figsize=(9, 9))
        fig2.suptitle(f'错误样本示例 (共 {len(wrong_images)} 个)', fontsize=14, fontweight='bold')

        for i in range(min(9, len(wrong_images))):
            ax = axes2[i // 3, i % 3]
            ax.imshow(wrong_images[i][0], cmap='gray')
            ax.set_title(f'真实: {wrong_labels[i].item()} → 预测: {wrong_preds[i].item()}', color='red')
            ax.axis('off')

        plt.tight_layout()

        save_path2 = os.path.join(figures_dir, 'mnist_wrong_samples.png')
        plt.savefig(save_path2, dpi=150, bbox_inches='tight')
        print(f"✅ 错误样本图已保存: {save_path2}")

        save_path2_alt = os.path.join(script_dir, 'mnist_wrong_samples.png')
        plt.savefig(save_path2_alt, dpi=150, bbox_inches='tight')
        print(f"✅ 错误样本图已保存: {save_path2_alt}")

        plt.show()
    else:
        print("🎉 所有样本都预测正确！")

except Exception as e:
    print(f"❌ 错误样本分析失败: {e}")

print("\n" + "=" * 60)
print(f"最终测试准确率: {epoch_acc_history[-1]:.2f}%")
print("=" * 60)
print(f"\n📁 所有图片已保存到: {figures_dir}")
print(f"📁 也保存在: {script_dir}")