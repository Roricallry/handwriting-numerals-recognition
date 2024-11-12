import time
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import BatchNormalization, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split, cross_val_score
from tensorflow.keras import layers
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
ImageDataGenerator = keras.preprocessing.image.ImageDataGenerator
LearningRateScheduler = keras.callbacks.LearningRateScheduler # 导入学习率动态调试器


def Init_KNN():
    print("正在生成knn模型(预计时间3秒)")
    start_sum = time.time()  # 记录开始的时间
    # 1. 获取手写数字图像数据集和标签
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist.data.astype(np.float32), mnist.target.astype(int)

    # 2. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. 创建并训练 KNN 模型
    knn_model = KNeighborsClassifier(n_neighbors=1)
    knn_model.fit(X_train, y_train)

    # 4. 保存模型为 knn_model.pkl 文件
    joblib.dump(knn_model, 'models/knn_model.pkl')
    print("KNN model saved as knn_model.pkl")

    end_sum = time.time()  # 记录结束的时间
    print('训练耗时：', end_sum - start_sum, '秒')


def Init_SVM():
    start_sum = time.time()  # 记录开始的时间

    print("正在生成svm模型(预计时间120秒)")
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist.data.astype(np.float32), mnist.target.astype(int)

    # 2. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. 创建并训练 SVM 模型
    svm_model = SVC(kernel='rbf', gamma='scale')
    svm_model.fit(X_train, y_train)

    # 4. 保存模型为 svm_model.pkl 文件
    joblib.dump(svm_model, 'models/svm_model.pkl')
    print("SVM model saved as svm_model.pkl")

    end_sum = time.time()  # 记录结束的时间
    print('训练耗时：', end_sum - start_sum, '秒')


def Init_RF():
    print("正在生成rf模型(预计时间120秒)")
    start_sum = time.time()
    mnist = fetch_openml('mnist_784', version=1)
    x, test_x, y, test_y = train_test_split(mnist.data, mnist.target, test_size=0.25, random_state=40)

    from sklearn.ensemble import RandomForestClassifier
    # 使用随机森林分类器模型
    model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=40)  # 设置100棵决策树

    # 使用 K 折交叉验证评估模型性能
    cv_scores = cross_val_score(model, x, y, cv=5)
    # 输出 K 折交叉验证的平均准确率
    print("随机森林 K 折交叉验证准确率:", np.mean(cv_scores))

    # 在训练集上拟合模型
    model.fit(x, y)

    joblib.dump(model, 'models/rf_model.pkl')
    print("RF model saved as rf_model.pkl")
    end_sum = time.time()
    print('训练耗时：', end_sum - start_sum, '秒')


def Init_CNN():
    print("正在生成cnn模型(预计时间200秒)")
    # 数据集都是灰度图像
    mnist = keras.datasets.mnist
    (features_train, label_train), (features_test, label_test) = mnist.load_data()

    # 维度转换，对应格式（样本数量，高度，宽度，通道）
    features_train = features_train.reshape(60000, 28, 28, 1)
    features_test = features_test.reshape(10000, 28, 28, 1)

    # 数据标准化
    features_train = features_train / 255.0
    features_test = features_test / 255.0

    # 创建图像生成器，并设置数据增强选项
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        validation_split=0.2,
        horizontal_flip=False,
        vertical_flip=False
    )

    # 将生成器与训练数据拟合
    datagen.fit(features_train)

    # 构建网络
    model = keras.Sequential()

    # 第一个卷积层，64个卷积核，每一个尺寸为3x3，激活函数为relu
    conv_layer1 = layers.Conv2D(64, (3, 3), activation='relu')
    # 第二个卷积层
    conv_layer2 = layers.Conv2D(64, (3, 3), activation='relu')
    # 第三个卷积层
    conv_layer3 = layers.Conv2D(64, (3, 3), activation='relu')
    # 第四个卷积层
    conv_layer4 = layers.Conv2D(64, (3, 3), activation='relu')

    # 第一个全连接层，神经元数量为521，激活函数为sigmoid
    fc_layer1 = layers.Dense(512, activation="sigmoid")
    # 第二个全连接层，神经元为512
    fc_layer2 = layers.Dense(512, activation="sigmoid")
    # 第三个全连接层，神经元为256
    fc_layer3 = layers.Dense(256, activation="sigmoid")
    # 第四个全连接层，神经元为10，与目标分类数量相同
    fc_layer4 = layers.Dense(10, activation="sigmoid")

    # 将实例化后的网络层进行组合构建
    model.add(conv_layer1)
    model.add(conv_layer2)
    model.add(layers.MaxPooling2D(2, 2))  # 最大池化层

    model.add(conv_layer3)
    model.add(conv_layer4)
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Flatten())
    model.add(fc_layer1)
    model.add(layers.Dropout(0.25))  # Dropout正则化技术，随机丢弃部分神经元的输出来减少过拟合
    model.add(fc_layer2)
    model.add(layers.Dropout(0.25))
    model.add(fc_layer3)
    model.add(layers.Dropout(0.1))
    model.add(fc_layer4)

    # 设置优化器
    # 使用Adam优化器，学习率为0.001
    optimizer = keras.optimizers.Adam(0.001)

    # 创建学习率调度器
    def lr_scheduler(epoch, lr):
        decay_rate = 0.1
        decay_step = 5
        if epoch % decay_step == 0 and epoch:
            return lr * decay_rate
        return lr

    lr_callback = LearningRateScheduler(lr_scheduler)

    # 模型编译
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    # 模型训练
    start_time = time.time()  # 记录开始的时间

    history = model.fit(datagen.flow(features_train, label_train, batch_size=32),
                        epochs=10,
                        validation_data=(features_test, label_test),
                        verbose=2,
                        callbacks=[lr_callback])

    end_time = time.time()  # 记录结束的时间
    print('训练总耗时：', end_time - start_time, '秒')

    # 模型评估
    model.evaluate(features_test, label_test)

    # 模型保存
    model.save("models/cnn_model.h5")
    print("CNN model saved as cnn_model.h5")
    print('训练总耗时：', end_time - start_time, '秒')

    # 用于可视化
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()


def Init_FCN():
    # 加载MNIST数据集
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # 维度转换，对应格式（样本数量，高度，宽度，通道）
    train_images = train_images.reshape(60000, 28, 28, 1)
    test_images = test_images.reshape(10000, 28, 28, 1)

    # 将像素值缩放到0到1之间
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # 将标签转换为分类形式（one-hot encoding）
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    # 创建一个Sequential模型
    model = Sequential([
        # 将28x28的图像展平为一维向量
        Flatten(input_shape=(28, 28, 1)),

        # 添加隐藏层，增加神经元数量，并使用ReLU激活函数
        Dense(256, activation='relu'),
        BatchNormalization(),  # 批量归一化
        Dropout(0.25),  # Dropout层

        # 第二个隐藏层
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.25),

        # 输出层，10个神经元（对应10个数字类别），使用softmax激活函数
        Dense(10, activation='softmax')
    ])

    # 编译模型，使用RMSprop优化器
    model.compile(optimizer=RMSprop(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 记录开始训练的时间
    start_time = time.time()

    # 训练模型，使用训练集数据，设置验证集
    history = model.fit(train_images, train_labels, epochs=15, batch_size=128,
                        validation_data=(test_images, test_labels))

    # 评估模型在测试集上的性能
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    # 记录训练结束的时间
    end_time = time.time()

    # 打印训练总耗时和测试集准确率
    print('训练耗时：', end_time - start_time, '秒')
    print('测试集准确率：', test_acc)

    # 保存模型到文件"models/FC_model.h5"
    model.save("models/fcn_model.h5")

    # 打印保存模型的消息
    print("FCN model saved as fcn_model.h5")













    



