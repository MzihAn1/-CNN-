import io
import json
import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request, render_template
from model import resnet34

app = Flask(__name__)

weights_path = "APP/resNet34.pth"
class_json_path = "./class_indices.json"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# 实例化ResNet模型
model = resnet34(num_classes=5)
# 载入权重信息
model.load_state_dict(torch.load(weights_path, map_location=device))
# 将模型放到GPU上
model.to(device)
# 开启模型的验证模式
model.eval()
# 以二进制格式打开存有种类信息的json文件用于只读
json_file = open(class_json_path, 'rb')
class_indict = json.load(json_file)

# 图像预处理函数
def transform_image(image_bytes):
    # 传入的图像为字节形式
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        # 将图片缩放为255*255
                                        transforms.CenterCrop(224),
                                        # 采用中心裁剪的方法将图片裁剪为224*224
                                        transforms.ToTensor(),
                                        # 将图片转换成Tensor格式
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])
                                        # 数据标准化，将数据转换成标准的高斯分布，逐个维度的对图像进行标准化(均值为0，标准差为1)
                                        # 加速模型收敛   
                                            ])
    # 读入图像数据
    image = Image.open(io.BytesIO(image_bytes))
    # 将读入的图片进行预处理，并增加一个Batch维度，然后传入GPU设备
    return my_transforms(image).unsqueeze(0).to(device)


def get_prediction(image_bytes):
    # 将传入的图片进行预处理
    tensor = transform_image(image_bytes=image_bytes)
    # 将Tensor传入模型的正向传播过程，压缩Batch维度，使用softmax函数将结果进行概率分布
    outputs = torch.softmax(model.forward(tensor).squeeze(), dim=0)
    # 去除梯度信息，把得到的数值放到cpu上，转换成numpy格式
    prediction = outputs.detach().cpu().numpy()
    # 通过enumerate方法将预测结果中的每一项前加一个索引，然后传入一个新的列表
    index_pre = []
    for index, p in enumerate(prediction):
        index_pre.append((class_indict[str(index)], float(p)))
    # 根据概率p的大小，对结果进行降序排序
    # 使用lambda定义一个简单的函数
    index_pre.sort(key=lambda x: x[1], reverse=True)

    text = []
    for k, v in index_pre:
        text.append("种类:{:<15} 概率:{:.1f}%".format(k, v * 100))
    return_info = {"result": text}
    return return_info


@app.route("/predict", methods=["POST"])
# 不计算梯度信息
@torch.no_grad()
def predict():
    image = request.files["file"]
    img_bytes = image.read()
    info = get_prediction(image_bytes=img_bytes)
    return jsonify(info)


@app.route("/", methods=["GET", "POST"])
def root():
    return render_template("index.html")


if __name__ == '__main__':
    # 允许外部设备访问服务器
    app.run(host="0.0.0.0", port=5000)

