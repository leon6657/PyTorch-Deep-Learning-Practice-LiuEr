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

# ========== 解决中文和负号显示问题 ==========
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ========== 切换到脚本所在目录 ==========
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ========== 1. 数据加载 ==========
batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='../dataset/mnist/',
                               train=True,
                               download=True,
                               transform=transform)
train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=batch_size)

test_dataset = datasets.MNIST(root='../dataset/mnist/',
                              train=False,
                              download=True,
                              transform=transform)
test_loader = DataLoader(test_dataset,
                         shuffle=False,
                         batch_size=batch_size)

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")


# ========== 2. 定义多种CNN配置 ==========
class ComplexCNN(torch.nn.Module):
    """
    复杂CNN：3个卷积层 + 3个池化层 + 3个全连接层
    """

    def __init__(self, conv_channels=[32, 64, 128], fc_dims=[256, 128], dropout_rate=0.5):
        super(ComplexCNN, self).__init__()

        # 卷积层
        self.conv1 = torch.nn.Conv2d(1, conv_channels[0], kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(conv_channels[1], conv_channels[2], kernel_size=3, padding=1)

        # 池化层
        self.pool = torch.nn.MaxPool2d(2)

        # 计算全连接层的输入维度
        # 输入 28x28 → 经过3次池化 → 28/2/2/2 = 3.5 → 3×3
        self.fc_input_dim = conv_channels[2] * 3 * 3  # 128 * 3 * 3 = 1152

        # 全连接层
        self.fc1 = torch.nn.Linear(self.fc_input_dim, fc_dims[0])
        self.fc2 = torch.nn.Linear(fc_dims[0], fc_dims[1])
        self.fc3 = torch.nn.Linear(fc_dims[1], 10)

        # Dropout
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        # 卷积块1
        x = self.pool(F.relu(self.conv1(x)))  # 28→14
        # 卷积块2
        x = self.pool(F.relu(self.conv2(x)))  # 14→7
        # 卷积块3
        x = self.pool(F.relu(self.conv3(x)))  # 7→3

        # 展平
        x = x.view(-1, self.fc_input_dim)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


class SimpleCNN(torch.nn.Module):
    """
    简单CNN：2个卷积层 + 2个池化层 + 2个全连接层（作为对比）
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.pool = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 320)
        x = self.fc(x)
        return x


# ========== 3. 训练函数 ==========
def train_model(model, device, train_loader, criterion, optimizer, epochs=10):
    model.train()
    train_loss_history = []
    epoch_loss_history = []
    start_time = time.time()

    for epoch in range(epochs):
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

            train_loss_history.append(loss.item())

            if batch_idx % 300 == 299:
                avg_loss = running_loss / 300
                print(f'[{epoch + 1}, {batch_idx + 1:5d}] loss: {avg_loss:.3f}')
                running_loss = 0.0

        avg_epoch_loss = epoch_loss / batch_count
        epoch_loss_history.append(avg_epoch_loss)

    end_time = time.time()
    print(f"训练用时: {end_time - start_time:.2f} 秒")
    return train_loss_history, epoch_loss_history


# ========== 4. 测试函数 ==========
def test_model(model, device, test_loader):
    model.eval()
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
    return accuracy


# ========== 5. 计算模型参数量 ==========
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ========== 6. 主训练流程 ==========
def run_experiment():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}\n")

    # 定义要测试的配置
    configs = [
        {
            'name': '复杂CNN-1',
            'model': ComplexCNN(conv_channels=[32, 64, 128], fc_dims=[256, 128], dropout_rate=0.5),
            'lr': 0.001,
            'epochs': 10
        },
        {
            'name': '复杂CNN-2',
            'model': ComplexCNN(conv_channels=[16, 32, 64], fc_dims=[128, 64], dropout_rate=0.3),
            'lr': 0.001,
            'epochs': 10
        },
        {
            'name': '复杂CNN-3',
            'model': ComplexCNN(conv_channels=[64, 128, 256], fc_dims=[512, 256], dropout_rate=0.6),
            'lr': 0.001,
            'epochs': 10
        },
        {
            'name': '简单CNN(对比)',
            'model': SimpleCNN(),
            'lr': 0.01,
            'epochs': 10
        }
    ]

    results = {}

    for config in configs:
        print("\n" + "=" * 60)
        print(f"训练: {config['name']}")
        print("=" * 60)

        model = config['model'].to(device)
        param_count = count_parameters(model)
        print(f"参数量: {param_count:,}")

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])

        # 训练
        train_loss, epoch_loss = train_model(
            model, device, train_loader, criterion, optimizer, config['epochs']
        )

        # 测试
        accuracy = test_model(model, device, test_loader)
        print(f"测试集准确率: {accuracy:.2f}%")

        results[config['name']] = {
            'train_loss': train_loss,
            'epoch_loss': epoch_loss,
            'accuracy': accuracy,
            'param_count': param_count,
            'model': model
        }

    return results


# ========== 7. 可视化 ==========
def visualize_results(results):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('不同CNN配置性能对比', fontsize=18, fontweight='bold')

    # 颜色方案
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # -------- 子图1：损失曲线对比 --------
    ax1 = axes[0, 0]
    for i, (name, data) in enumerate(results.items()):
        ax1.plot(data['train_loss'], color=colors[i], linewidth=0.5, alpha=0.5, label=name)
    ax1.set_xlabel('Batch Step', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('训练损失曲线对比', fontsize=14)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.3)

    # -------- 子图2：每个Epoch平均损失 --------
    ax2 = axes[0, 1]
    for i, (name, data) in enumerate(results.items()):
        ax2.plot(range(1, len(data['epoch_loss']) + 1), data['epoch_loss'],
                 color=colors[i], marker='o', linewidth=2, label=name)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('平均损失', fontsize=12)
    ax2.set_title('每个Epoch平均损失', fontsize=14)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.3)

    # -------- 子图3：准确率对比 --------
    ax3 = axes[0, 2]
    names = list(results.keys())
    accuracies = [data['accuracy'] for data in results.values()]
    bars = ax3.bar(names, accuracies, color=colors[:len(names)])
    ax3.set_xlabel('模型配置', fontsize=12)
    ax3.set_ylabel('测试集准确率 (%)', fontsize=12)
    ax3.set_title('测试集准确率对比', fontsize=14)
    ax3.set_ylim(90, 100)
    for bar, acc in zip(bars, accuracies):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f'{acc:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.3, axis='y')

    # -------- 子图4：参数量对比 --------
    ax4 = axes[1, 0]
    param_counts = [data['param_count'] for data in results.values()]
    bars = ax4.bar(names, param_counts, color=colors[:len(names)])
    ax4.set_xlabel('模型配置', fontsize=12)
    ax4.set_ylabel('参数量', fontsize=12)
    ax4.set_title('模型参数量对比', fontsize=14)
    for bar, count in zip(bars, param_counts):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.05,
                 f'{count:,}', ha='center', va='bottom', fontsize=9)
    ax4.grid(True, linestyle='--', alpha=0.3, axis='y')

    # -------- 子图5：准确率 vs 参数量 --------
    ax5 = axes[1, 1]
    for i, (name, data) in enumerate(results.items()):
        ax5.scatter(data['param_count'], data['accuracy'],
                    s=200, color=colors[i], label=name)
        ax5.annotate(name, (data['param_count'], data['accuracy']),
                     xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax5.set_xlabel('参数量', fontsize=12)
    ax5.set_ylabel('准确率 (%)', fontsize=12)
    ax5.set_title('准确率 vs 参数量', fontsize=14)
    ax5.legend(loc='lower right', fontsize=9)
    ax5.grid(True, linestyle='--', alpha=0.3)

    # -------- 子图6：结果汇总表格 --------
    ax6 = axes[1, 2]
    ax6.axis('tight')
    ax6.axis('off')

    table_data = []
    for name, data in results.items():
        table_data.append([
            name,
            f"{data['accuracy']:.2f}%",
            f"{data['param_count']:,}",
            f"{len(data['epoch_loss'])}"
        ])

    columns = ['模型配置', '准确率', '参数量', 'Epochs']
    table = ax6.table(cellText=table_data, colLabels=columns,
                      cellLoc='center', loc='center',
                      colWidths=[0.2, 0.15, 0.2, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # 高亮最高准确率
    best_acc = max(data['accuracy'] for data in results.values())
    for i, (name, data) in enumerate(results.items()):
        if data['accuracy'] == best_acc:
            for j in range(4):
                table[(i + 1, j)].set_facecolor('#90EE90')

    ax6.set_title('训练结果汇总', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('cnn_configuration_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✅ 可视化图片已保存: cnn_configuration_comparison.png")
    plt.show()


# ========== 8. 主程序 ==========
if __name__ == '__main__':
    results = run_experiment()
    visualize_results(results)

    # 打印总结
    print("\n" + "=" * 60)
    print("📊 实验总结")
    print("=" * 60)
    for name, data in results.items():
        print(f"{name:15s} | 准确率: {data['accuracy']:.2f}% | 参数量: {data['param_count']:,}")

    best_name = max(results, key=lambda x: results[x]['accuracy'])
    print(f"\n🏆 最佳模型: {best_name} (准确率: {results[best_name]['accuracy']:.2f}%)")
    print("=" * 60)