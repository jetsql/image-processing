import os
import numpy as np
#介面顯示
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
#影像處理
import cv2
import matplotlib.pyplot as plt
#數學運算
from numpy.lib.stride_tricks import as_strided
from itertools import product
#計算時間
import time 
#載入檔案的檔名宣告
FILENAME=""
class Ui_Dialog(object):
    def __init__(self):
        #指定資料夾
        os.chdir("D:\\code\\Hw3\\img\\")

    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(380, 300)
        self.groupBox = QtWidgets.QGroupBox(Dialog)
        self.groupBox.setGeometry(QtCore.QRect(50, 10, 290, 200))
        self.groupBox.setObjectName("groupBox")
        self.pushButton_1 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_1.setGeometry(QtCore.QRect(50, 30, 190, 28))
        self.pushButton_1.setObjectName("pushButton_1")
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_2.setGeometry(QtCore.QRect(50, 80, 190, 28))
        self.pushButton_2.setObjectName("pushButton_2")
        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.groupBox.setTitle(_translate("Dialog", "Hw4_optical_flow_(NM6101098)"))
        self.pushButton_1.setText(_translate("Dialog", "Load_Image"))
        self.pushButton_2.setText(_translate("Dialog", "Optical_Flow"))
        #按鈕事件
        self.pushButton_1.clicked.connect(self.load_image)
        self.pushButton_2.clicked.connect(self.texture_matching)
    def load_image(self):
        print('checked load_image')
        self.sub_window = SubWindow1_1()
        self.sub_window.show()
        pass
    def texture_matching(self):
        self.file_name=FILENAME.split('/')[-1]

        if self.file_name[0]=='P':
            self.img_0=cv2.imread('.\\Pillow0.jpg')
            self.img_1=cv2.imread('.\\Pillow1.jpg')
        if self.file_name[0]=='C':
            self.img_0=cv2.imread('.\\Cup0.jpg')
            self.img_1=cv2.imread('.\\Cup1.jpg')
        self.img0_gray = cv2.cvtColor(self.img_0, cv2.COLOR_BGR2GRAY)
        self.img1_gray = cv2.cvtColor(self.img_1, cv2.COLOR_BGR2GRAY)
        self.optical_flow(self.img0_gray,self.img1_gray)

    def optical_flow(self,img0_gray,img1_gray):
        pt_x, pt_y = 250, 325
        param = dict(pyr_scale=0.8,
                    levels=25,
                    iterations=1,
                    winsize=5,
                    poly_n=5,
                    poly_sigma=1.1)

        flow = None
        XL, YL = [0], [0]
        PX, PY = [pt_x], [pt_y]
        for i in range(1000):
            if i==0:
                fg = 0
            else:
                fg = cv2.OPTFLOW_USE_INITIAL_FLOW
            flow = cv2.calcOpticalFlowFarneback(img0_gray, img1_gray, flow=flow, flags=fg, **param)
            
            XL.append(flow[pt_y, pt_x, 0])
            YL.append(flow[pt_y, pt_x, 1])
            PX.append(int(pt_x + flow[pt_y, pt_x, 0]))
            PY.append(int(pt_y + flow[pt_y, pt_x, 1]))
            if i>0:
                ep = np.sum(np.abs(XL[i-1] - XL[i])) + np.sum(np.abs(YL[i-1] - YL[i]))
                print('iter:{}, ep:{}'.format(i, ep))
                if ep<1e-5:
                    break
        ################
        img = cv2.cvtColor(self.img_0, cv2.COLOR_BGR2RGB)

        for j in range(len(PX)):
            if j!=0:
                cv2.line(img, (PX[j-1], PY[j-1]), (PX[j], PY[j]), (250, 5, 216), 2)

        for k in range(len(PX)):
            if k==0:
                c = (0, 38, 255)
            elif k==len(PX)-1:
                c = (182, 255, 0)
            else:
                c = (255, 0, 0)
            cv2.circle(img,(PX[k], PY[k]), 3, c, -1)
        plt.imshow(img)
        plt.show()



#顯示另外一個畫布開檔案
class SubWindow1_1(QWidget):
    def __init__(self):
        super(SubWindow1_1,self).__init__()
        self.resize(400,300)
        self.setWindowTitle("load Image")
        self.width = QtWidgets.QLabel(self)
        self.width.setGeometry(QtCore.QRect(130, 40, 60, 30))
        self.width.setObjectName("width")
        self.height = QtWidgets.QLabel(self)
        self.height.setGeometry(QtCore.QRect(130, 80, 60, 30))
        self.height.setObjectName("height")
        self.lineEdit = QtWidgets.QLineEdit(self)
        self.lineEdit.setGeometry(QtCore.QRect(130, 20, 141, 20))
        self.lineEdit.setObjectName("lineEdit")
        self.open_Button = QtWidgets.QPushButton(self)
        self.open_Button.setGeometry(QtCore.QRect(100, 150, 75, 23))
        self.open_Button.setObjectName("Browse")
        self.open_Button.setText("Browse")
        self.open_Button.clicked.connect(self.browsefiles)
        self.ok_Button = QtWidgets.QPushButton(self)
        self.ok_Button.setGeometry(QtCore.QRect(210, 150, 75, 23))
        self.open_Button.setObjectName("OK!")
        self.ok_Button.setText("OK!")
        self.ok_Button.clicked.connect(self.pushfilename)
    #跳視窗開檔
    def browsefiles(self):
        path=os.getcwd()
        path=str(path)
        fileName = QFileDialog.getOpenFileName(self,'Open File',str(path),'*.jpg')
        global FILENAME
        FILENAME = fileName[0]
        self.lineEdit.setText(fileName[0])
    def pushfilename(self):
        global FILENAME
        self.file_name=FILENAME.split('/')[-1]

        if self.file_name[0]=='P':
            self.img_0=cv2.imread('.\\Pillow0.jpg')
            self.img_1=cv2.imread('.\\Pillow1.jpg')
        if self.file_name[0]=='C':
            self.img_0=cv2.imread('.\\Cup0.jpg')
            self.img_1=cv2.imread('.\\Cup1.jpg')
        self.img0_gray = cv2.cvtColor(self.img_0, cv2.COLOR_BGR2GRAY)
        self.img1_gray = cv2.cvtColor(self.img_1, cv2.COLOR_BGR2GRAY)
        plt.subplot(121)
        plt.imshow(self.img0_gray, 'gray')
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(self.img1_gray, 'gray')
        plt.axis('off')
        plt.show()
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())