from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import numpy as np

class Ui_mainWindow(object):
    def __init__(self):
        self.img1_1 = cv2.imread("Sun.jpg")
        self.img1_3 = cv2.cvtColor(self.img1_1, cv2.COLOR_BGR2GRAY)
        self.img1_4_1 = cv2.imread("Dog_Strong.jpg")
        self.img1_4_2 = cv2.imread("Dog_Weak.jpg")
        self.img4_1 = cv2.imread("SQUARE-01.png")
        self.img_resize = cv2.resize(self.img4_1,(256,256))
        self.shifted = []
        self.rotation_img = []
        self.img3_1 = cv2.imread("House.jpg")
        self.img_3_1 = cv2.cvtColor(self.img3_1 , cv2.COLOR_BGR2GRAY)
        self.gaussian = []
        self.img2_1 = cv2.imread("Lenna_whiteNoise.jpg")
        self.img2_3 = cv2.imread("Lenna_pepperSalt.jpg")

    def setupUi(self, mainWindow):
        mainWindow.setObjectName("mainWindow")
        mainWindow.resize(920, 609)
        self.centralwidget = QtWidgets.QWidget(mainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.group1 = QtWidgets.QGroupBox(self.centralwidget)
        self.group1.setGeometry(QtCore.QRect(10, 10, 211, 571))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.group1.setFont(font)
        self.group1.setObjectName("group1")
        self._1_1 = QtWidgets.QPushButton(self.group1)
        self._1_1.setGeometry(QtCore.QRect(20, 120, 161, 31))
        self._1_1.setObjectName("_1_1")
        self._1_2 = QtWidgets.QPushButton(self.group1)
        self._1_2.setGeometry(QtCore.QRect(20, 220, 161, 31))
        self._1_2.setObjectName("_1_2")
        self._1_3 = QtWidgets.QPushButton(self.group1)
        self._1_3.setGeometry(QtCore.QRect(20, 320, 161, 31))
        self._1_3.setObjectName("_1_3")
        self._1_4 = QtWidgets.QPushButton(self.group1)
        self._1_4.setGeometry(QtCore.QRect(20, 430, 161, 31))
        self._1_4.setObjectName("_1_4")
        self.group2 = QtWidgets.QGroupBox(self.centralwidget)
        self.group2.setGeometry(QtCore.QRect(230, 10, 221, 571))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.group2.setFont(font)
        self.group2.setObjectName("group2")
        self._2_1 = QtWidgets.QPushButton(self.group2)
        self._2_1.setGeometry(QtCore.QRect(20, 160, 181, 31))
        self._2_1.setObjectName("_2_1")
        self._2_2 = QtWidgets.QPushButton(self.group2)
        self._2_2.setGeometry(QtCore.QRect(20, 270, 181, 31))
        self._2_2.setObjectName("_2_2")
        self._2_3 = QtWidgets.QPushButton(self.group2)
        self._2_3.setGeometry(QtCore.QRect(20, 380, 181, 31))
        self._2_3.setObjectName("_2_3")
        self.group3 = QtWidgets.QGroupBox(self.centralwidget)
        self.group3.setGeometry(QtCore.QRect(460, 10, 221, 571))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.group3.setFont(font)
        self.group3.setObjectName("group3")
        self._3_1 = QtWidgets.QPushButton(self.group3)
        self._3_1.setGeometry(QtCore.QRect(20, 120, 171, 31))
        self._3_1.setObjectName("_3_1")
        self._3_2 = QtWidgets.QPushButton(self.group3)
        self._3_2.setGeometry(QtCore.QRect(20, 220, 171, 31))
        self._3_2.setObjectName("_3_2")
        self._3_3 = QtWidgets.QPushButton(self.group3)
        self._3_3.setGeometry(QtCore.QRect(20, 320, 171, 31))
        self._3_3.setObjectName("_3_3")
        self._3_4 = QtWidgets.QPushButton(self.group3)
        self._3_4.setGeometry(QtCore.QRect(20, 430, 171, 31))
        self._3_4.setObjectName("_3_4")
        self.group4 = QtWidgets.QGroupBox(self.centralwidget)
        self.group4.setGeometry(QtCore.QRect(690, 10, 221, 571))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.group4.setFont(font)
        self.group4.setObjectName("group4")
        self._4_1 = QtWidgets.QPushButton(self.group4)
        self._4_1.setGeometry(QtCore.QRect(20, 120, 171, 31))
        self._4_1.setObjectName("_4_1")
        self._4_2 = QtWidgets.QPushButton(self.group4)
        self._4_2.setGeometry(QtCore.QRect(20, 220, 171, 31))
        self._4_2.setObjectName("_4_2")
        self._4_3 = QtWidgets.QPushButton(self.group4)
        self._4_3.setGeometry(QtCore.QRect(20, 320, 171, 31))
        self._4_3.setObjectName("_4_3")
        self._4_4 = QtWidgets.QPushButton(self.group4)
        self._4_4.setGeometry(QtCore.QRect(20, 430, 171, 31))
        self._4_4.setObjectName("_4_4")
        mainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(mainWindow)
        self.statusbar.setObjectName("statusbar")
        mainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(mainWindow)
        QtCore.QMetaObject.connectSlotsByName(mainWindow)

    def retranslateUi(self, mainWindow):
        _translate = QtCore.QCoreApplication.translate
        mainWindow.setWindowTitle(_translate("mainWindow", "MainWindow"))
        self.group1.setTitle(_translate("mainWindow", "1. Image Processing"))
        self._1_1.setText(_translate("mainWindow", "1.1 Load Image"))
        self._1_2.setText(_translate("mainWindow", "1.2 Color Seperation"))
        self._1_3.setText(_translate("mainWindow", "1.3 Color Transformations"))
        self._1_4.setText(_translate("mainWindow", "1.4 Blending"))
        self.group2.setTitle(_translate("mainWindow", "2. Imgae Smoothing"))
        self._2_1.setText(_translate("mainWindow", "2.1 Gaussian Blur"))
        self._2_2.setText(_translate("mainWindow", "2.2 Bilateral Filter"))
        self._2_3.setText(_translate("mainWindow", "2.3 Median Filter"))
        self.group3.setTitle(_translate("mainWindow", "3. Edge Detection"))
        self._3_1.setText(_translate("mainWindow", "3.1 Gaussian Blur"))
        self._3_2.setText(_translate("mainWindow", "3.2 Sobel X"))
        self._3_3.setText(_translate("mainWindow", "3.3 Sobel Y"))
        self._3_4.setText(_translate("mainWindow", "3.4 Magnitude"))
        self.group4.setTitle(_translate("mainWindow", "4. Transformation"))
        self._4_1.setText(_translate("mainWindow", "4.1 Resize"))
        self._4_2.setText(_translate("mainWindow", "4.2 Translation"))
        self._4_3.setText(_translate("mainWindow", "4.3 Rotation, Scaling"))
        self._4_4.setText(_translate("mainWindow", "4.4 Shearing"))
        self._1_1.clicked.connect(self.loadimg)
        self._1_2.clicked.connect(self.color_split)
        self._1_3.clicked.connect(self.color_transformation)
        self._1_4.clicked.connect(self.blend)
        self._2_1.clicked.connect(self.gaussian_blur)
        self._2_2.clicked.connect(self.Bilateral)
        self._2_3.clicked.connect(self.median)
        self._3_1.clicked.connect(self.gaussain)
        self._3_2.clicked.connect(self.sobelx)
        self._3_3.clicked.connect(self.sobely)
        self._3_4.clicked.connect(self.sobel_magnititude)
        self._4_1.clicked.connect(self.resize)
        self._4_2.clicked.connect(self.translation)
        self._4_3.clicked.connect(self.rotation)
        self._4_4.clicked.connect(self.shearing)

    def loadimg(self):
        cv2.imshow("Sun.jpg", self.img1_1)
        print('Height : %d' % self.img1_1.shape[0])
        print('Width : %d' %self.img1_1.shape[1])
        cv2.waitKey(0)

    def color_split(self):
        b,g,r = cv2.split(self.img1_1) 
        zeros = np.zeros(self.img1_1.shape[:2], dtype = "uint8")
        cv2.imshow("B channel", cv2.merge([b, zeros, zeros]))
        cv2.imshow("G channel", cv2.merge([zeros, g, zeros]))
        cv2.imshow("R channel", cv2.merge([zeros, zeros, r]))
        cv2.waitKey(0)

    def color_transformation(self):
        cv2.imshow("L1", self.img1_3) 
        r = self.img1_1[:,:,0]
        g = self.img1_1[:,:,1]
        b = self.img1_1[:,:,2]
        image2 = (r + g + b) / 3
        image2 = image2.astype(int)
        image2 = image2.astype("uint8")
        cv2.imshow("L2", image2)
        cv2.waitKey(0)
        
    def blend(self):

        def on_change(x):
            w = cv2.getTrackbarPos('blend', 'Blend')
            min1 = w / 255
            max1 = 1 - min1
            dst = cv2.addWeighted(self.img1_4_1, max1, self.img1_4_2, min1, 0)
            cv2.imshow('Blend', dst)

        cv2.namedWindow('Blend')
        cv2.createTrackbar('blend', 'Blend', 0, 255, on_change)
        cv2.waitKey(0)

    def gaussian_blur(self):
        output = cv2.GaussianBlur(self.img2_1, (5,5), 1, 1)
        cv2.imshow('gaussian blur', output)
        cv2.waitKey(0)

    def Bilateral(self):
        bilateral_output = cv2.bilateralFilter(self.img2_1, 9, 90, 90)
        cv2.imshow('bilateral output', bilateral_output)
        cv2.waitKey(0)

    def median(self):
        output = cv2.medianBlur(self.img2_3, 3)
        output2 = cv2.medianBlur(self.img2_3, 5)
        cv2.imshow('Median Filter 3x3', output)
        cv2.imshow('Median Filter 5x5', output2)
        cv2.waitKey(0)
        

    def convolution(self,k, data):
        n,m = data.shape
        img_new = []
        for i in range(n-3):
            line = []
            for j in range(m-3):
                a = data[i:i+3,j:j+3]
                line.append(np.sum(np.multiply(k, a)))
            img_new.append(line)
        return np.array(img_new).astype(np.uint8)

    def gaussain(self):
        k = np.array([
        [0.045,0.122,0.045],
        [0.122,0.332,0.122],
        [0.045,0.122,0.045]
        ])
        self.gaussian = self.convolution(k,self.img_3_1)
        cv2.imshow('gaussain blur', self.gaussian)
        cv2.waitKey(0)
        
    def sobelx(self): 
        if self.gaussian == []:
            k = np.array([
            [0.045,0.122,0.045],
            [0.122,0.332,0.122],
            [0.045,0.122,0.045]
            ])
            self.gaussian = self.convolution(k,self.img_3_1)
        input_img = self.gaussian
        height = input_img.shape[0]
        weight = input_img.shape[1]
        sx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        dSobelx = np.zeros((height,weight))
        Gx = np.zeros(self.gaussian.shape)
        for i in range(height-2):
            for j in range(weight-2):
                Gx[i + 1, j + 1] = abs(np.sum(self.gaussian[i:i + 3, j:j + 3] * sx))
                dSobelx[i+1, j+1] = Gx[i+1, j+1]
        cv2.imshow('sobelx', np.uint8(dSobelx))
        cv2.waitKey(0)

    def sobely(self):
        if self.gaussian == []:
            k = np.array([
            [0.045,0.122,0.045],
            [0.122,0.332,0.122],
            [0.045,0.122,0.045]
            ])
            self.gaussian = self.convolution(k,self.img_3_1)

        input_img = self.gaussian
        height = input_img.shape[0]
        weight = input_img.shape[1]
        sy = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
        dSobely = np.zeros((height,weight))
        Gy = np.zeros(self.gaussian.shape)
        for i in range(height-2):
            for j in range(weight-2):
                Gy[i + 1, j + 1] = abs(np.sum(self.gaussian[i:i + 3, j:j + 3] * sy))
                dSobely[i + 1, j + 1] = Gy[i + 1, j + 1]
        cv2.imshow('sobely', np.uint8(dSobely))
        cv2.waitKey(0)
        
    def sobel_magnititude(self):
        if self.gaussian == []:
            k = np.array([
            [0.045,0.122,0.045],
            [0.122,0.332,0.122],
            [0.045,0.122,0.045]
            ])
            self.gaussian = self.convolution(k,self.img_3_1)
        input_img = self.gaussian
        height = input_img.shape[0]
        weight = input_img.shape[1]
      
        sx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        sy = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
        dSobel = np.zeros((height,weight))
        Gx = np.zeros(self.gaussian.shape)
        Gy = np.zeros(self.gaussian.shape)
        for i in range(height-2):
            for j in range(weight-2):
                Gx[i + 1, j + 1] = abs(np.sum(self.gaussian[i:i + 3, j:j + 3] * sx))
                Gy[i + 1, j + 1] = abs(np.sum(self.gaussian[i:i + 3, j:j + 3] * sy))
                dSobel[i+1, j+1] = (Gx[i+1, j+1]*Gx[i+1,j+1] + Gy[i+1, j+1]*Gy[i+1,j+1])**0.5
        cv2.imshow('sobel', np.uint8(dSobel))
        cv2.waitKey(0)

    def resize(self):
        cv2.imshow('img', self.img_resize)
        cv2.waitKey(0)

    def translate(self,image, x, y):
            M = np.float32([[1, 0, x], [0, 1, y]])
            shifted = cv2.warpAffine(image, M, (400,300))
            return shifted

    def translation(self):
        self.shifted = self.translate(self.img_resize, 0, 60)
        cv2.imshow("img_2", self.shifted)
        cv2.waitKey(0)

    def rotation(self): 
        M = cv2.getRotationMatrix2D((128, 188),10, 0.5)
        self.rotation_img = cv2.warpAffine(self.shifted, M, (400,300))
        cv2.imshow('img_3', self.rotation_img)
        cv2.waitKey(0)

    def shearing(self):
        x = np.float32([[50,50],[200,50],[50,200]])
        y = np.float32([[10,100],[200,50],[100,250]])
        matrix = cv2.getAffineTransform(x, y)
        res = cv2.warpAffine(self.rotation_img, matrix, (400,300))
        cv2.imshow('img_4', res)
        cv2.waitKey(0)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = QtWidgets.QMainWindow()
    ui = Ui_mainWindow()
    ui.setupUi(mainWindow)
    mainWindow.show()
    sys.exit(app.exec_())

