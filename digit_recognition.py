import os
import cv2
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np
import cv2
from PIL import Image
from keras.models import load_model
from sklearn.ensemble import RandomForestClassifier


class DigitRecognition_KNN(object):
    def __init__(self):
        # 加载KNN模型
        self.knn_model = KNeighborsClassifier()  # 假设K=1
        self.load_knn_model("models/knn_model.pkl")  # 替换为你的KNN模型文件路径

    def load_knn_model(self, model_path):
        self.knn_model = joblib.load(model_path)

    def preprocess_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (28, 28))  # 将图像调整为 28*28
        image = image.flatten()  # 将图像展平成一维向量
        return image

    def predict_digit(self, image_path):
        image = self.preprocess_image(image_path)
        digit = self.knn_model.predict([image])[0]
        return digit



class DigitRecognition_SVM(object):
    def __init__(self):
        # Load SVM model
        self.svm_model = SVC(gamma='scale', kernel='rbf', C=10)
        self.load_svm_model("models/svm_model.pkl")

    def load_svm_model(self, model_path):
        self.svm_model = joblib.load(model_path)

    def preprocess_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (28, 28))
        image = image.flatten()
        return image

    def predict_digit(self, image_path):
        image = self.preprocess_image(image_path)
        digit = self.svm_model.predict([image])[0]
        return digit


class DigitRecognition_RF(object):
    def __init__(self):
        # 加载随机森林模型
        self.rf_model = RandomForestClassifier()
        self.load_rf_model("models/rf_model.pkl")

    def load_rf_model(self, model_path):
        self.rf_model = joblib.load(model_path)

    def preprocess_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (28, 28))  # 将图像调整为 28*28
        image = image.flatten()  # 将图像展平成一维向量
        return image

    def predict_digit(self, image_path):
        image = self.preprocess_image(image_path)
        digit = self.rf_model.predict([image])[0]
        return digit


class DigitRecognition_CNN(object):
    def __init__(self):
        # 加载CNN模型
        self.model = load_model('models/cnn_model.h5')  # 替换为你的CNN模型文件路径

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('L')  # 转换为灰度图
        image = image.resize((28, 28))  # 调整大小为模型输入的大小
        image_array = np.array(image)  # 转换为 numpy 数组
        image_array = image_array.reshape((1, 28, 28, 1))  # 添加批处理维度并调整形状
        image_array = image_array / 255.0  # 数据标准化
        return image_array

    # 进行预测
    def predict_digit(self, image_path):
        image_array = self.preprocess_image(image_path)
        prediction = self.model.predict(image_array)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        print('Predicted class: ', predicted_class, "  Confidence:", confidence)
        return predicted_class #, confidence


class DigitRecognition_FCN(object):
    def __init__(self):
        # 加载全神经网络模型
        self.model = load_model('models/fcn_model.h5')  # 替换为你的Fully_connected模型文件路径

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('L')  # 转换为灰度图
        image = image.resize((28, 28))  # 调整大小为模型输入的大小
        image_array = np.array(image)  # 转换为 numpy 数组
        image_array = image_array.reshape((1, 28, 28, 1))  # 添加批处理维度并调整形状
        image_array = image_array / 255.0  # 数据标准化
        return image_array

    # 进行预测
    def predict_digit(self, image_path):

        image_array = self.preprocess_image(image_path)
        prediction = self.model.predict(image_array)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        print('Predicted class: ', predicted_class, "  Confidence:", confidence)
        return predicted_class #, confidence

