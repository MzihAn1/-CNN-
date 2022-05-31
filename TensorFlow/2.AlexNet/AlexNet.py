import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras import layers, models, Model, Sequential
import matplotlib.pyplot as plt

im_height = 224
im_width = 224
batch_size = 32
epochs = 30


train_image_generator = ImageDataGenerator(rescale=1. / 255,
                                               horizontal_flip=True)
validation_image_generator = ImageDataGenerator(rescale=1. / 255)

train_data_gen = train_image_generator.flow_from_directory(directory='./Dataset/train',
                                                               batch_size=batch_size,
                                                               shuffle=True,
                                                               target_size=(im_height, im_width),
                                                               class_mode='categorical')
# 获得训练样本的个数
total_train = train_data_gen.n

val_data_gen = validation_image_generator.flow_from_directory(directory='./Dataset/val',
                                                                  batch_size=batch_size,
                                                                  shuffle=False,
                                                                  target_size=(im_height, im_width),
                                                                  class_mode='categorical')
# 获得测试样本的个数
total_val = val_data_gen.n

print("using {} images for training, {} images for validation.".format(total_train, total_val))

class AlexNet(Model):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = Sequential([
            layers.ZeroPadding2D(((1, 2), (1, 2))),                                 # output(None, 227, 227, 3)
            layers.Conv2D(48, kernel_size=11, strides=4, activation="relu"),        # output(None, 55, 55, 48)
            layers.MaxPool2D(pool_size=3, strides=2),                               # output(None, 27, 27, 48)
            layers.Conv2D(128, kernel_size=5, padding="same", activation="relu"),   # output(None, 27, 27, 128)
            layers.MaxPool2D(pool_size=3, strides=2),                               # output(None, 13, 13, 128)
            layers.Conv2D(192, kernel_size=3, padding="same", activation="relu"),   # output(None, 13, 13, 192)
            layers.Conv2D(192, kernel_size=3, padding="same", activation="relu"),   # output(None, 13, 13, 192)
            layers.Conv2D(128, kernel_size=3, padding="same", activation="relu"),   # output(None, 13, 13, 128)
            layers.MaxPool2D(pool_size=3, strides=2)                             # output(None, 6, 6, 128)
        ])
        
        self.flatten = layers.Flatten()
        self.classifier = Sequential([
            layers.Dropout(0.2),
            layers.Dense(1024, activation="relu"),                                  # output(None, 2048)
            layers.Dropout(0.2),
            layers.Dense(128, activation="relu"),                                   # output(None, 2048)
            layers.Dense(num_classes),                                                # output(None, 5)
            layers.Softmax()
        ])

    def call(self, x):
        x = self.features(x)
        x = self.flatten(x)
        y = self.classifier(x)
        return y

model = AlexNet(num_classes=7)
# model.build((batch_size, 224, 224, 3))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=["accuracy"])

callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='AlexNet.h5',
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