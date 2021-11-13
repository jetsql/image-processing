# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'nm6101098.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import cv2
FILENAME=""
class Ui_Dialog(object):
    def __init__(self,path='.\\Dataset_OpenCvDl_Hw1\\'):
        self.path1=path+'Q1_Image'
        self.path2=path+'Q2_Image'
        self.path3=path+'Q3_Image'
        self.path4=path+'Q4_Image'
    #主要UI介面
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(951, 542)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(570, 10, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.groupBox_1 = QtWidgets.QGroupBox(Dialog)
        self.groupBox_1.setGeometry(QtCore.QRect(30, 50, 211, 471))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.groupBox_1.setFont(font)
        self.groupBox_1.setMouseTracking(False)
        self.groupBox_1.setFlat(False)
        self.groupBox_1.setCheckable(False)
        self.groupBox_1.setObjectName("groupBox_1")
        self.pushButton_1_1 = QtWidgets.QPushButton(self.groupBox_1)
        self.pushButton_1_1.setGeometry(QtCore.QRect(20, 60, 181, 28))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_1_1.sizePolicy().hasHeightForWidth())
        self.pushButton_1_1.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(8)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.pushButton_1_1.setFont(font)
        self.pushButton_1_1.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.pushButton_1_1.setObjectName("pushButton_1_1")
        self.pushButton_1_2 = QtWidgets.QPushButton(self.groupBox_1)
        self.pushButton_1_2.setGeometry(QtCore.QRect(20, 140, 181, 28))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_1_2.sizePolicy().hasHeightForWidth())
        self.pushButton_1_2.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(8)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.pushButton_1_2.setFont(font)
        self.pushButton_1_2.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.pushButton_1_2.setObjectName("pushButton_1_2")
        self.pushButton_1_3 = QtWidgets.QPushButton(self.groupBox_1)
        self.pushButton_1_3.setGeometry(QtCore.QRect(20, 220, 181, 28))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_1_3.sizePolicy().hasHeightForWidth())
        self.pushButton_1_3.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(8)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.pushButton_1_3.setFont(font)
        self.pushButton_1_3.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.pushButton_1_3.setObjectName("pushButton_1_3")
        self.pushButton_1_4 = QtWidgets.QPushButton(self.groupBox_1)
        self.pushButton_1_4.setGeometry(QtCore.QRect(20, 300, 181, 28))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_1_4.sizePolicy().hasHeightForWidth())
        self.pushButton_1_4.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(8)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.pushButton_1_4.setFont(font)
        self.pushButton_1_4.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.pushButton_1_4.setObjectName("pushButton_1_4")
        self.groupBox_2 = QtWidgets.QGroupBox(Dialog)
        self.groupBox_2.setGeometry(QtCore.QRect(250, 50, 211, 471))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.groupBox_2.setFont(font)
        self.groupBox_2.setObjectName("groupBox_2")
        self.pushButton_2_1 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_2_1.setGeometry(QtCore.QRect(10, 100, 181, 28))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_2_1.sizePolicy().hasHeightForWidth())
        self.pushButton_2_1.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(8)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.pushButton_2_1.setFont(font)
        self.pushButton_2_1.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.pushButton_2_1.setObjectName("pushButton_2_1")
        self.pushButton_2_2 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_2_2.setGeometry(QtCore.QRect(10, 180, 181, 28))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_2_2.sizePolicy().hasHeightForWidth())
        self.pushButton_2_2.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(8)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.pushButton_2_2.setFont(font)
        self.pushButton_2_2.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.pushButton_2_2.setObjectName("pushButton_2_2")
        self.pushButton_2_3 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_2_3.setGeometry(QtCore.QRect(10, 260, 181, 28))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_2_3.sizePolicy().hasHeightForWidth())
        self.pushButton_2_3.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(8)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.pushButton_2_3.setFont(font)
        self.pushButton_2_3.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.pushButton_2_3.setObjectName("pushButton_2_3")
        self.groupBox_3 = QtWidgets.QGroupBox(Dialog)
        self.groupBox_3.setGeometry(QtCore.QRect(480, 50, 211, 471))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.groupBox_3.setFont(font)
        self.groupBox_3.setObjectName("groupBox_3")
        self.pushButton_3_1 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_3_1.setGeometry(QtCore.QRect(20, 60, 181, 28))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_3_1.sizePolicy().hasHeightForWidth())
        self.pushButton_3_1.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(8)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.pushButton_3_1.setFont(font)
        self.pushButton_3_1.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.pushButton_3_1.setObjectName("pushButton_3_1")
        self.pushButton_3_2 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_3_2.setGeometry(QtCore.QRect(20, 140, 181, 28))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_3_2.sizePolicy().hasHeightForWidth())
        self.pushButton_3_2.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(8)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.pushButton_3_2.setFont(font)
        self.pushButton_3_2.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.pushButton_3_2.setObjectName("pushButton_3_2")
        self.pushButton_3_3 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_3_3.setGeometry(QtCore.QRect(20, 220, 181, 28))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_3_3.sizePolicy().hasHeightForWidth())
        self.pushButton_3_3.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(8)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.pushButton_3_3.setFont(font)
        self.pushButton_3_3.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.pushButton_3_3.setObjectName("pushButton_10")
        self.pushButton_3_4 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_3_4.setGeometry(QtCore.QRect(20, 300, 181, 28))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_3_4.sizePolicy().hasHeightForWidth())
        self.pushButton_3_4.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(8)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.pushButton_3_4.setFont(font)
        self.pushButton_3_4.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.pushButton_3_4.setObjectName("pushButton_3_4")
        self.groupBox_4 = QtWidgets.QGroupBox(Dialog)
        self.groupBox_4.setGeometry(QtCore.QRect(700, 50, 201, 471))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.groupBox_4.setFont(font)
        self.groupBox_4.setObjectName("groupBox_4")
        self.pushButton_4_1 = QtWidgets.QPushButton(self.groupBox_4)
        self.pushButton_4_1.setGeometry(QtCore.QRect(10, 100, 181, 28))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_4_1.sizePolicy().hasHeightForWidth())
        self.pushButton_4_1.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(8)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.pushButton_4_1.setFont(font)
        self.pushButton_4_1.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.pushButton_4_1.setObjectName("pushButton_4_2")
        self.pushButton_4_2 = QtWidgets.QPushButton(self.groupBox_4)
        self.pushButton_4_2.setGeometry(QtCore.QRect(10, 180, 181, 28))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_4_2.sizePolicy().hasHeightForWidth())
        self.pushButton_4_2.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(8)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.pushButton_4_2.setFont(font)
        self.pushButton_4_2.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.pushButton_4_2.setObjectName("pushButton_4_2")
        self.pushButton_4_3 = QtWidgets.QPushButton(self.groupBox_4)
        self.pushButton_4_3.setGeometry(QtCore.QRect(10, 260, 181, 28))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_4_3.sizePolicy().hasHeightForWidth())
        self.pushButton_4_3.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(8)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.pushButton_4_3.setFont(font)
        self.pushButton_4_3.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.pushButton_4_3.setObjectName("pushButton_4_3")
        self.pushButton_4_4 = QtWidgets.QPushButton(self.groupBox_4)
        self.pushButton_4_4.setGeometry(QtCore.QRect(10, 340, 181, 28))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_4_4.sizePolicy().hasHeightForWidth())
        self.pushButton_4_4.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(8)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.pushButton_4_4.setFont(font)
        self.pushButton_4_4.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.pushButton_4_4.setObjectName("pushButton_4_4")

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept) # type: ignore
        self.buttonBox.rejected.connect(Dialog.reject) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(Dialog)
    #設定物件顯示文字
    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.groupBox_1.setTitle(_translate("Dialog", "1. Image Processing"))
        self.pushButton_1_1.setText(_translate("Dialog", "1.1 Load Image"))
        self.pushButton_1_2.setText(_translate("Dialog", "1.2 Color Seperation"))
        self.pushButton_1_3.setText(_translate("Dialog", "1.3 Color Transformations"))
        self.pushButton_1_4.setText(_translate("Dialog", "1.4 Blending"))
        self.groupBox_2.setTitle(_translate("Dialog", "2. Image Smoothing"))
        self.pushButton_2_1.setText(_translate("Dialog", "2.1 Gaussian Blur"))
        self.pushButton_2_2.setText(_translate("Dialog", "2.2 Bilateral Filter"))
        self.pushButton_2_3.setText(_translate("Dialog", "2.3 Median Filter"))
        self.groupBox_3.setTitle(_translate("Dialog", "3. Edge Detection"))
        self.pushButton_3_1.setText(_translate("Dialog", "3.1 Gaussian Blur"))
        self.pushButton_3_2.setText(_translate("Dialog", "3.2 Sobel X"))
        self.pushButton_3_3.setText(_translate("Dialog", "3.3 Sobel Y"))
        self.pushButton_3_4.setText(_translate("Dialog", "3.4 Magnitude"))
        self.groupBox_4.setTitle(_translate("Dialog", "4. Transformation"))
        self.pushButton_4_1.setText(_translate("Dialog", "4.1 Resize"))
        self.pushButton_4_2.setText(_translate("Dialog", "4.2 Translation"))
        self.pushButton_4_3.setText(_translate("Dialog", "4.3 Rotation,Scaling"))
        self.pushButton_4_4.setText(_translate("Dialog", "4.4 Shearing"))
        #程式功能部分
        self.pushButton_1_1.clicked.connect(self.f1_1)

    #開檔案
    def f1_1(self):
        print("Topic 1_1 clicked")
        self.sub_window = SubWindow1_1()
        print("check")
        self.sub_window.show()
        pass
    #色彩分離
    def f1_2(self):
        print("Topic 1_2 clicked")
        if FILENAME!="":
            img = cv2.imread(FILENAME)
            width = img.shape[0]
            height = img.shape[1]
            total_shape = width*height
            red =[]
            green = []
            blue = []
            
            cv2.imshow("red",red)
            cv2.imshow("green",green)
            cv2.imshow("blue",blue)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            pass
        else:
            print("no img")
        pass

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
        #(x1,y1,x2,y2)
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
        fileName = QFileDialog.getOpenFileName(self,'Open File','D:\'','*.jpg')
        global FILENAME
        FILENAME = fileName[0]
        self.lineEdit.setText(fileName[0])
    def pushfilename(self):
        global FILENAME
        src = cv2.imread(FILENAME)
        #設定text數值
        self.height.setText(f"Width = {src.shape[0]}")
        self.width.setText(f"Height = {src.shape[1]}")
        cv2.namedWindow(FILENAME,cv2.WINDOW_AUTOSIZE)
        cv2.imshow(FILENAME,src)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())