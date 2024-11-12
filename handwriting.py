import os
import cv2
from PyQt5.Qt import QSize
from PyQt5.QtCore import QRect, Qt, QPoint
from digit_recognition import DigitRecognition_KNN, DigitRecognition_SVM, DigitRecognition_CNN, DigitRecognition_RF, DigitRecognition_FCN
from PyQt5.QtGui import QPainter, QPixmap, QColor, QPen, QIcon
from PyQt5.QtWidgets import QPushButton, QMainWindow, QComboBox, QLabel, QSpinBox, QFileDialog, QMessageBox


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Resizable Canvas")  # 设置窗口标题
        self.setGeometry(600, 300, 700, 450)  # 设置窗口大小
        self.pen_size = 20  # 默认粗细为20
        self.__penColor = QColor("black")  # 设置默认画笔颜色为黑色
        self.__colorList = QColor.colorNames()  # 获取颜色列表
        self.models = {
            "CNN": "self.predict_CNN.predict_digit",
            "FCN": "self.predict_FCN.predict_digit",
            "KNN": "self.predict_KNN.predict_digit",
            "SVM": "self.predict_SVM.predict_digit",
            "RF": "self.predict_RF.predict_digit",
        }   # 可用的模型
        self.model = "CNN"  # 设置初始模型

        # 追踪鼠标位置
        self.last_point = QPoint()

        # 初始化画布
        self.canvas_width = 400
        self.canvas_height = 400
        self.canvas = QPixmap(self.canvas_width, self.canvas_height)  # 创建画布
        self.canvas.fill(Qt.white)  # 用白色填充画布
        self.border_color = Qt.black  # 画布边框颜色

        # 创建用于显示识别结果的标签
        self.text_label = QLabel(self)
        self.text_label.setGeometry(450, 15, 120, 40)
        self.text_label.setText('<font size="4">识别结果:</font>')

        self.result_label = QLabel(self)  # 创建标签控件
        self.result_label.setGeometry(550, 20, 120, 120)  # 设置标签位置和大小
        self.result_label.setAlignment(Qt.AlignCenter)  # 居中对齐
        self.result_label.setStyleSheet("background-color: #FFF5FF; font-size: 60px; border: 1px solid black")  # 添加边框样式

        # 创建识别按钮
        self.save_button = QPushButton("识别", self)
        self.save_button.setGeometry(450, 150, 170, 50)
        self.save_button.clicked.connect(self.save_image)

        # 添加导入图片文件按钮
        self.load_button = QPushButton("file", self)
        self.load_button.setGeometry(620, 150, 50, 50)
        self.load_button.clicked.connect(self.load_image)

        # 创建清除按钮
        self.clear_button = QPushButton("清除", self)
        self.clear_button.setGeometry(450, 210, 220, 50)  # 设置按钮位置和大小
        self.clear_button.clicked.connect(self.clearCanvas)  # 连接清除按钮的点击事件

        # 设置画笔粗细
        self.label_pensize = QLabel(self)
        self.label_pensize.setGeometry(450, 270, 120, 30)
        self.label_pensize.setText("画笔粗细")

        self.spinBox_pensize = QSpinBox(self)
        self.spinBox_pensize.setGeometry(550, 270, 120, 30)
        self.spinBox_pensize.setMaximum(60)
        self.spinBox_pensize.setMinimum(2)
        self.spinBox_pensize.setValue(20)  # 默认粗细为20，与初始self.pen_size一致
        self.spinBox_pensize.setSingleStep(2)  # 最小变化值为2
        self.spinBox_pensize.valueChanged.connect(self.ChangePenSize)

        # 画笔颜色
        self.label_penColor = QLabel(self)
        self.label_penColor.setGeometry(450, 310, 120, 30)
        self.label_penColor.setText("画笔颜色")

        self.comboBox_penColor = QComboBox(self)
        self.comboBox_penColor.setGeometry(550, 310, 120, 30)
        self.fillColorList(self.comboBox_penColor)  # 用各种颜色填充下拉列表
        self.comboBox_penColor.currentIndexChanged.connect(self.ChangePenColor)  # 关联下拉列表的当前索引变更信号与函数ChangePenColor

        # 模型选择
        self.label_model = QLabel(self)
        self.label_model.setGeometry(450, 350, 120, 30)
        self.label_model.setText("选择模型")

        self.comboBox_model = QComboBox(self)
        self.comboBox_model.setGeometry(550, 350, 120, 30)
        self.fillModelList(self.comboBox_model)  # 用可用模型填充下拉列表
        self.comboBox_model.currentIndexChanged.connect(self.on_ModelChange)  # 关联下拉列表的当前索引变更信号与函数on_ModelChange

        # 加载模型
        self.predict_KNN = DigitRecognition_KNN()
        self.predict_SVM = DigitRecognition_SVM()
        self.predict_CNN = DigitRecognition_CNN()
        self.predict_RF = DigitRecognition_RF()
        self.predict_FCN = DigitRecognition_FCN()

    def paintEvent(self, event):
        painter = QPainter(self)

        border_rect = QRect(24, 24, self.canvas_width + 1, self.canvas_height + 1)
        painter.setPen(self.border_color)
        painter.drawRect(border_rect)

        painter.drawPixmap(25, 25, self.canvas)  # 在窗口上绘制画布内容


    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_point = event.pos()
        self.result_label.setText("")
        self.update()  # 更新窗口显示

    # 鼠标移动事件
    def mouseMoveEvent(self, event):
        canvas_rect = QRect(5, 5, self.canvas_width + 20, self.canvas_height + 20)
        if event.buttons() & Qt.LeftButton & canvas_rect.contains(self.last_point):
            painter = QPainter(self.canvas)
            pen = QPen()
            pen.setWidth(self.pen_size)
            pen.setColor(self.__penColor)
            painter.setPen(pen)
            painter.drawLine(self.last_point - QPoint(25, 25), event.pos() - QPoint(25, 25))
            self.last_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_point = QPoint()

    def clearCanvas(self):
        self.canvas.fill(Qt.white)  # 清除画布内容，填充为白色
        self.result_label.setText("")
        self.update()  # 更新窗口显示

    def save_image(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本的路径
        file_path = os.path.join(script_dir, "canvas_temp.png")  # 构建文件路径
        self.canvas.save(file_path)

        image = cv2.imread('canvas_temp.png')  # 读取原始图片
        inverted_image = cv2.bitwise_not(image)  # 将图片进行反色处理

        gray_image = cv2.cvtColor(inverted_image, cv2.COLOR_BGR2GRAY)  #灰度
        cv2.imwrite('canvas_temp.png', gray_image)  # 保存灰度处理后的图片

        # 预测并显示结果
        file_path = "\"" + file_path + "\""
        self.result_label.setText(f"{self.Prediction(file_path)}")

    def ChangePenSize(self):
        self.pen_size = self.spinBox_pensize.value()  # 改变笔的size

    def ChangePenColor(self):
        color_index = self.comboBox_penColor.currentIndex()
        self.__penColor = QColor(self.__colorList[color_index])

    def fillColorList(self, comboBox):

        index_black = 0
        index = 0
        for color in self.__colorList:
            if color == "black":
                index_black = index
            index += 1
            pix = QPixmap(85, 20)
            pix.fill(QColor(color))
            comboBox.addItem(QIcon(pix), None)
            comboBox.setIconSize(QSize(85, 20))
            comboBox.setSizeAdjustPolicy(QComboBox.AdjustToContents)

        comboBox.setCurrentIndex(index_black)

    def load_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Choose Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)",
                                                   options=options)
        if file_path:
            file_path = "\"" + file_path + "\""
            self.clearCanvas()
            self.result_label.setText(f"{self.Prediction(file_path)}")


    # 创建模型选择下拉列表
    def fillModelList(self, comboBox):
        for model in self.models:
            comboBox.addItem(model)
            item_index = comboBox.count() - 1
            item_height = 30  # 设置下拉列表项的高度为30
            comboBox.setItemData(item_index, QSize(0, item_height), Qt.SizeHintRole)

    # 模型选择事件
    def on_ModelChange(self):
        selected_model_index = self.comboBox_model.currentIndex()
        keys = list(self.models.keys())
        self.model = keys[selected_model_index]
        # 在这里可以根据选择的模型进行相应的操作

    def Prediction(self, file_path):
        file_path = file_path.replace("\\", "/")
        return eval(self.models[self.model] + "(" + file_path + ")")






    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Message',
            "Are you sure to quit?", QMessageBox.Yes |
            QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

