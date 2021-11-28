from PyQt5 import QtCore, QtGui, QtWidgets
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import cv2


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(419, 457)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton_1 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_1.setGeometry(QtCore.QRect(90, 70, 241, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton_1.setFont(font)
        self.pushButton_1.setObjectName("pushButton_1")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(90, 130, 241, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(90, 190, 241, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(90, 360, 241, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton_5.setFont(font)
        self.pushButton_5.setObjectName("pushButton_5")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(120, 310, 171, 31))
        self.lineEdit.setObjectName("lineEdit")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(90, 260, 241, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton_4.setFont(font)
        self.pushButton_4.setObjectName("pushButton_4")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 419, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton_1.setText(_translate("MainWindow", "1. Show Train Image "))
        self.pushButton_2.setText(_translate("MainWindow", "2. Show HyperParameters "))
        self.pushButton_3.setText(_translate("MainWindow", "3. Show  Model Shortcut"))
        self.pushButton_5.setText(_translate("MainWindow", "5.Test"))
        self.pushButton_4.setText(_translate("MainWindow", "4. Show Accuracy"))
        self.pushButton_1.clicked.connect(self.show_image)
        self.pushButton_2.clicked.connect(self.show_hyperparameters)
        self.pushButton_3.clicked.connect(self.show_model)
        self.pushButton_4.clicked.connect(self.show_accuracy)
        self.pushButton_5.clicked.connect(self.test)

    def show_image(self):
        img =  plt.imread("class.png")
        plt.imshow(img)
        plt.axis("off")
        plt.imshow(img)
        plt.show()

    def show_hyperparameters(self):
        img = plt.imread("hyperparameters.png")
        plt.axis("off")
        plt.imshow(img)
        plt.show()

    def show_model(self):
        img = plt.imread("model.jpg")
        plt.axis("off")
        plt.imshow(img)
        plt.show()

    def show_accuracy(self):
        img = plt.imread("accuracy.png")
        plt.axis("off")
        plt.subplot(1,2,2)
        plt.imshow(img)
    
        img2 = plt.imread("loss.png")
        plt.subplot(1,2,1)
        plt.axis("off")
        plt.imshow(img2)
        plt.show()

    def test(self):
        model_path = "VGG16.h5"
        model = load_model(model_path)
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        input1 = self.lineEdit.text()
        input0 = []
        probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
        img = cv2.imread(f"{input1 }.jpg") /255
        input0.append(img)
        input0 = np.array(input0)
        prob = probability_model.predict(input0)
        plt.subplot(1,2,2)
        plt.bar(class_names, prob[0])
        plt.subplot(1,2,1)
        plt.title(f"predict = {class_names[np.argmax(prob)]}")
        fig = plt.imshow(img)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.show()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
