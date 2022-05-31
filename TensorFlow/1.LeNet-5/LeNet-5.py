import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras import layers, models, Model, Sequential
import matplotlib.pyplot as plt
import json

im_height = 32
im_width = 32
batch_size = 32
epochs = 10


'''
rescale的作用是对图片的每个像素值均乘上这个放缩因子，
这个操作在所有其它变换操作之前执行，在一些模型当中，
直接输入原图的像素值可能会落入激活函数的“死亡区”，
因此设置放缩因子为1/255，把像素值放缩到0和1之间有利于模型的收敛，避免神经元“死亡”。'''

train_image_generator = ImageDataGenerator(rescale=1. / 255,
                                               horizontal_flip=True)
validation_image_generator = ImageDataGenerator(rescale=1. / 255)

train_data_gen = train_image_generator.flow_from_directory(directory='./Datasets/train',
                                                               batch_size=batch_size,
                                                               shuffle=True,
                                                               target_size=(im_height, im_width),
                                                               class_mode='categorical')
# 获得训练样本的个数
total_train = train_data_gen.n


# 获取类别索引
# class_indices = train_data_gen.class_indices
# inverse_dict = dict((val, key) for key, val in class_indices.items())
# # write dict into json file
# json_str = json.dumps(inverse_dict, indent=4)
# with open('class_indices.json', 'w') as json_file:
#     json_file.write(json_str)

    
val_data_gen = validation_image_generator.flow_from_directory(directory='./Datasets/val',
                                                                  batch_size=batch_size,
                                                                  shuffle=False,
                                                                  target_size=(im_height, im_width),
                                                                  class_mode='categorical')
# 获得测试样本的个数
total_val = val_data_gen.n

print("using {} images for training, {} images for validation.".format(total_train, total_val))

class LeNet5(Model):
    def __init__(self, num_classes=1000):
        super(LeNet5, self).__init__()
        self.features = Sequential([
            layers.Conv2D(filters=6, kernel_size=(5, 5),
                         activation='relu'),
            layers.MaxPool2D(pool_size=(2, 2), strides=2),
            layers.Conv2D(filters=16, kernel_size=(5, 5),
                         activation='relu'),
            layers.MaxPool2D(pool_size=(2, 2), strides=2)
        ])
        
        self.flatten = layers.Flatten()
        self.classifier = Sequential([
            layers.Dense(120, activation='relu'),
            layers.Dense(84, activation='relu'),
            layers.Dense(num_classes, activation='softmax'),
        ])

    def call(self, x):
        x = self.features(x)
        x = self.flatten(x)
        y = self.classifier(x)
        return y

model = LeNet5(num_classes=5)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=["accuracy"])

callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='LeNet5.h5',
                                                    save_best_only=True,
                                                    save_weights_only=True,
                                                    monitor='val_loss')]

# tensorflow2.1 recommend to using fit
history = model.fit(x=train_data_gen,
                        steps_per_epoch=total_train // batch_size,
                        epochs=epochs,
                        validation_data=val_data_gen,
                        validation_steps=total_val // batch_size,
                        callbacks=callbacks)
model.summary()
# plot loss and accuracy image
history_dict = history.history
train_loss = history_dict["loss"]
train_accuracy = history_dict["accuracy"]
val_loss = history_dict["val_loss"]
val_accuracy = history_dict["val_accuracy"]

# figure 1
plt.figure()
plt.plot(range(epochs), train_loss, label='train_loss')
plt.plot(range(epochs), val_loss, label='val_loss')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')

# figure 2
plt.figure()
plt.plot(range(epochs), train_accuracy, label='train_accuracy')
plt.plot(range(epochs), val_accuracy, label='val_accuracy')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('accuracy')


plt.show()