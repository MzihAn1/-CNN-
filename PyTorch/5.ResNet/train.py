from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from model import resnet34


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

batch_size = 16
# 训练数据集
train_dataset = datasets.ImageFolder('Datasets/train/', transform=data_transform["train"])
train_num = len(train_dataset)

# 测试数据集
test_dataset = datasets.ImageFolder('Datasets/val/', transform=data_transform["val"])
test_num = len(test_dataset)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

print("using {} images for training, {} images for validation.".format(train_num, test_num))

net = resnet34()
model_weight_path = "./resnet34-333f7ec4.pth"
net.load_state_dict(torch.load(model_weight_path, map_location=device))

in_channel = net.fc.in_features
net.fc = nn.Linear(in_channel, 5)
net.to(device)

loss_function = nn.CrossEntropyLoss()

params = [p for p in net.parameters() if p.requires_grad]
optimizer = optim.Adam(params, lr=0.0001)

epochs = 10
best_acc = 0.0
save_path = './resNet34.pth'
train_steps = len(train_loader)

train_loss_results = []

test_acc = []

for epoch in range(epochs):
    # train
    net.train()
    running_loss = 0.0
    # train_bar = tqdm(train_loader, file=sys.stdout)
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()
        logits = net(images.to(device))
        loss = loss_function(logits, labels.to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        # train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
        rate = (step + 1) / len(train_loader)
        a = "-" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtraining: {:^3.0f}%[{}->{}]".format(int(rate * 100), a, b), end="")
    print()                                                            
                                                                

    # validate
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        # val_bar = tqdm(validate_loader, file=sys.stdout)
        for val_data in test_loader:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))  # eval model only have last output layer
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
            
    val_accurate = acc / test_num

    print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
            (epoch + 1, running_loss / train_steps, val_accurate))
    train_loss_results.append(running_loss / step)
    test_acc.append(val_accurate)
    if val_accurate > best_acc:
        best_acc = val_accurate
        torch.save(net.state_dict(), save_path)

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