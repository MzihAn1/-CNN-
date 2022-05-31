import json
import torch
from torchvision import transforms, datasets
import numpy as np
# Python进度条
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import resnet34


class ConfusionMatrix(object):
    # 图像分类类别个数，分类标签列表
    def __init__(self, num_classes: int, labels: list):
        # 初始化一个高宽均等于类别个数，数值全为零的矩阵
        self.matrix = np.zeros((num_classes, num_classes))
        # 赋值给类变量
        self.num_classes = num_classes
        self.labels = labels
    # 将预测值与真实标签输入到矩阵中
    def update(self, preds, labels):
        # p 为预测值，t 为真实类别标签
        # 使用zip方法将预测值与真实标签进行打包组合
        for p, t in zip(preds, labels):
            # 在p行t列进行累加一
            self.matrix[p, t] += 1
    # 绘制混淆矩阵
    def plot(self):
        matrix = self.matrix
        print(matrix)
        # 使用imshow函数展示混淆矩阵
        plt.imshow(matrix, cmap=plt.cm.Reds)

        # 设置x轴坐标label，0 ~ num_classes-1替换为标签名
        plt.xticks(range(self.num_classes), self.labels)
        
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)

        # 显示colorbar，数值分布的密集程度
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        # 可以显示中文
        plt.rcParams['font.sans-serif']=['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    data_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    validate_dataset = datasets.ImageFolder('Datasets/val',
                                            transform=data_transform)

    batch_size = 16
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False
                                                  )
    net = resnet34(num_classes=5)
    # load pretrain weights
    model_weight_path = "APP/resNet34.pth"
    
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    net.to(device)

    # read class_indict
    json_label_path = 'class_indices.json'
    
    json_file = open(json_label_path, 'rb')
    class_indict = json.load(json_file)

    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=5, labels=labels)
    net.eval()
    with torch.no_grad():
        for val_data in tqdm(validate_loader):
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            outputs = torch.softmax(outputs, dim=1)
            outputs = torch.argmax(outputs, dim=1)
            confusion.update(outputs.to("cpu").numpy(), val_labels.to("cpu").numpy())
    confusion.plot()
   