from matplotlib import pyplot as plt
from torch import nn, optim
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


device = torch.device("cuda")
print(device)

transform = transforms.Compose([transforms.Resize((32, 32)), 
                                transforms.ToTensor(), 
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])


batch_size = 64
# 训练数据集
train_dataset = datasets.ImageFolder('Dataset/train', transform=transform)
train_num = len(train_dataset)

# 测试数据集
test_dataset = datasets.ImageFolder('Dataset/val', transform=transform)
test_num = len(test_dataset)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
print("using {} images for training, {} images for validation.".format(train_num, test_num))


class LeNet(nn.Module):
    def __init__(self, num_classes=1000,):
        super(LeNet, self).__init__()
        # 先局部
        self.features =nn.Sequential(
            # 卷积层，提取图片特征
            nn.Conv2d(3, 16, 5), # 输入图片的维度，使用卷积核的个数，卷积核大小
            # 激活函数，将线性关系转换为非线性关系
            nn.ReLU(inplace=True),
            # 池化层，抓住主要矛盾，忽略次要矛盾，进一步提取图片特征
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        # 再整体
        self.classifier = nn.Sequential(
            nn.Linear(32*5*5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
            # nn.Softmax(dim=1)
            
        )
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


net = LeNet(num_classes=7)
net.to(device)
# 损失函数: 交叉熵代价函数
loss_function = nn.CrossEntropyLoss()
# 优化器
optimizer = optim.Adam(net.parameters(), lr=0.0002)

epochs = 30
best_acc = 0.0
save_path = './LeNet.pth'

train_loss_results = []
test_acc = []

for epoch in range(epochs):
    # 开始训练
    net.train()
    running_loss = 0.0
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        # 历史梯度清零，减少设备运算量
        optimizer.zero_grad()

        outputs = net(images.to(device))

        loss = loss_function(outputs, labels.to(device))

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        # 打印训练过程
        rate = (step + 1) / len(train_loader)
        a = "=" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\r{:^3.0f}%[{}=>{}]train loss: {:.3f}".format(int(rate * 100), a, b, loss), end="")
    print()
    

    
    net.eval()
    acc = 0.0  
    # 不计算损失梯度
    with torch.no_grad():
        
        for val_data in test_loader:

            val_images, val_labels = val_data

            outputs = net(val_images.to(device))
            # 使用torch.max函数找到概率最大的类别的索引
            # 在第二个索引值查找，[1]为只需要类别对应的索引
            predict_y = torch.max(outputs, dim=1)[1]
            # 使用item获取数值
            acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

    val_accurate = acc / test_num
    if val_accurate > best_acc:
        best_acc = val_accurate
        torch.save(net.state_dict(), save_path)

    print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
            (epoch + 1, running_loss / step, val_accurate))
            
    train_loss_results.append(running_loss / step)
    test_acc.append(val_accurate)

# 绘制 loss 曲线
plt.title('Loss Function Curve')  # 图片标题
plt.xlabel('Epoch')  # x轴变量名称
plt.ylabel('Loss')  # y轴变量名称
plt.plot(train_loss_results, label="$Loss$")  # 逐点画出trian_loss_results值并连线，连线图标是Loss
plt.legend()  # 画出曲线图标
plt.show()  # 画出图像

# 绘制 Accuracy 曲线
plt.title('Acc Curve')  # 图片标题
plt.xlabel('Epoch')  # x轴变量名称
plt.ylabel('Acc')  # y轴变量名称
plt.plot(test_acc, label="$Accuracy$")  # 逐点画出test_acc值并连线，连线图标是Accuracy
plt.legend()
plt.show()

print('Finished Training')

