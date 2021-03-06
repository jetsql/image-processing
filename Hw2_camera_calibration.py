# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'hw2.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import numpy as np
import glob
import os
from matplotlib import pyplot as plt

class Ui_Dialog(object):
    def __init__(self):
        #第二題的影像資料集
        self.images_2 = sorted(glob.glob(os.path.join("D:/code/hw2/Dataset_OpenCvDl_Hw2/Q2_Image", "*.bmp")))
        self.PATTERN_SIZE_2=[11,8]
        #第三題的影像資料集
        self.images_3 = sorted(glob.glob(os.path.join("D:/code/hw2/Dataset_OpenCvDl_Hw2/Q3_Image", "*.bmp")))
        self.PATTERN_SIZE_3=[11,8]
        self.imglist = []
        self.objpoints=[]
        self.imgpoints=[]
        self.msk = []
        self.mtx=[]
        self.dist=[]
        self.rvecs=[]
        self.tvecs=[]
        self.img4_1=cv2.imread('D:/code/hw2/Dataset_OpenCvDl_Hw2/Q4_Image/imL.png')
        self.img4_2=cv2.imread('D:/code/hw2/Dataset_OpenCvDl_Hw2/Q4_Image/imR.png')
        self.img4_left = cv2.cvtColor(self.img4_1, cv2.COLOR_BGR2GRAY)
        self.img4_right = cv2.cvtColor(self.img4_2, cv2.COLOR_BGR2GRAY)
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(780, 377)
        self.groupBox_1 = QtWidgets.QGroupBox(Dialog)
        self.groupBox_1.setGeometry(QtCore.QRect(10, 30, 231, 291))
        self.groupBox_1.setObjectName("groupBox_1")
        self.pushButton_1_1 = QtWidgets.QPushButton(self.groupBox_1)
        self.pushButton_1_1.setGeometry(QtCore.QRect(20, 30, 191, 28))
        self.pushButton_1_1.setObjectName("pushButton_1_1")
        self.pushButton_1_2 = QtWidgets.QPushButton(self.groupBox_1)
        self.pushButton_1_2.setGeometry(QtCore.QRect(20, 70, 191, 28))
        self.pushButton_1_2.setObjectName("pushButton_1_2")
        self.label_1_name1 = QtWidgets.QLabel(self.groupBox_1)
        self.label_1_name1.setGeometry(QtCore.QRect(0, 130, 58, 15))
        self.label_1_name1.setObjectName("label_1_name1")
        self.label_1_show1 = QtWidgets.QLabel(self.groupBox_1)
        self.label_1_show1.setGeometry(QtCore.QRect(60, 130, 31, 16))
        self.label_1_show1.setText("")
        self.label_1_show1.setObjectName("label_1_show1")
        self.label_1_name2 = QtWidgets.QLabel(self.groupBox_1)
        self.label_1_name2.setGeometry(QtCore.QRect(100, 130, 58, 15))
        self.label_1_name2.setObjectName("label_1_name2")
        self.label_1_show2 = QtWidgets.QLabel(self.groupBox_1)
        self.label_1_show2.setGeometry(QtCore.QRect(150, 130, 71, 16))
        self.label_1_show2.setText("")
        self.label_1_show2.setObjectName("label_1_show2")
        self.label_2_name1 = QtWidgets.QLabel(self.groupBox_1)
        self.label_2_name1.setGeometry(QtCore.QRect(0, 160, 58, 15))
        self.label_2_name1.setObjectName("label_2_name1")
        self.label_2_show2 = QtWidgets.QLabel(self.groupBox_1)
        self.label_2_show2.setGeometry(QtCore.QRect(150, 160, 71, 16))
        self.label_2_show2.setText("")
        self.label_2_show2.setObjectName("label_2_show2")
        self.label_2_show1 = QtWidgets.QLabel(self.groupBox_1)
        self.label_2_show1.setGeometry(QtCore.QRect(60, 160, 31, 16))
        self.label_2_show1.setText("")
        self.label_2_show1.setObjectName("label_2_show1")
        self.label_2_name2 = QtWidgets.QLabel(self.groupBox_1)
        self.label_2_name2.setGeometry(QtCore.QRect(100, 160, 58, 15))
        self.label_2_name2.setObjectName("label_2_name2")
        self.groupBox_2 = QtWidgets.QGroupBox(Dialog)
        self.groupBox_2.setGeometry(QtCore.QRect(250, 30, 211, 291))
        self.groupBox_2.setObjectName("groupBox_2")
        self.groupBox_2_3 = QtWidgets.QGroupBox(self.groupBox_2)
        self.groupBox_2_3.setGeometry(QtCore.QRect(10, 110, 191, 91))
        self.groupBox_2_3.setObjectName("groupBox_2_3")
        self.label_2_3 = QtWidgets.QLabel(self.groupBox_2_3)
        self.label_2_3.setGeometry(QtCore.QRect(20, 30, 81, 16))
        self.label_2_3.setObjectName("label_2_3")
        self.lineEdit_2_3 = QtWidgets.QLineEdit(self.groupBox_2_3)
        self.lineEdit_2_3.setGeometry(QtCore.QRect(100, 30, 41, 22))
        self.lineEdit_2_3.setObjectName("lineEdit_2")
        self.pushButton_2_3 = QtWidgets.QPushButton(self.groupBox_2_3)
        self.pushButton_2_3.setGeometry(QtCore.QRect(20, 60, 131, 28))
        self.pushButton_2_3.setObjectName("pushButton_3")
        self.pushButton_2_1 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_2_1.setGeometry(QtCore.QRect(10, 30, 191, 28))
        self.pushButton_2_1.setObjectName("pushButton")
        self.pushButton_2_2 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_2_2.setGeometry(QtCore.QRect(10, 70, 191, 28))
        self.pushButton_2_2.setObjectName("pushButton_2")
        self.pushButton_2_4 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_2_4.setGeometry(QtCore.QRect(10, 210, 181, 28))
        self.pushButton_2_4.setObjectName("pushButton_4")
        self.pushButton_2_5 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_2_5.setGeometry(QtCore.QRect(10, 250, 181, 28))
        self.pushButton_2_5.setObjectName("pushButton_5")
        self.groupBox_3 = QtWidgets.QGroupBox(Dialog)
        self.groupBox_3.setGeometry(QtCore.QRect(470, 30, 221, 161))
        self.groupBox_3.setObjectName("groupBox_3")
        self.lineEdit_4_1 = QtWidgets.QLineEdit(self.groupBox_3)
        self.lineEdit_4_1.setGeometry(QtCore.QRect(20, 30, 191, 22))
        self.lineEdit_4_1.setObjectName("lineEdit")
        self.pushButton_3_1 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_3_1.setGeometry(QtCore.QRect(20, 60, 191, 28))
        self.pushButton_3_1.setObjectName("pushButton_3_1")
        self.pushButton_3_2 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_3_2.setGeometry(QtCore.QRect(20, 100, 191, 28))
        self.pushButton_3_2.setObjectName("pushButton_3_2")
        self.groupBox_4 = QtWidgets.QGroupBox(Dialog)
        self.groupBox_4.setGeometry(QtCore.QRect(470, 200, 221, 121))
        self.groupBox_4.setObjectName("groupBox_4")
        self.pushButton_4_1 = QtWidgets.QPushButton(self.groupBox_4)
        self.pushButton_4_1.setGeometry(QtCore.QRect(20, 50, 191, 28))
        self.pushButton_4_1.setObjectName("pushButton_4_1")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.groupBox_1.setTitle(_translate("Dialog", "1. Find Contour"))
        self.pushButton_1_1.setText(_translate("Dialog", "1.1 Draw Contour"))
        self.pushButton_1_2.setText(_translate("Dialog", "1.2 Count Rings"))
        self.label_1_name1.setText(_translate("Dialog", "There are"))
        self.label_1_name2.setText(_translate("Dialog", "rings in"))
        self.label_2_name1.setText(_translate("Dialog", "There are"))
        self.label_2_name2.setText(_translate("Dialog", "rings in"))
        self.groupBox_2.setTitle(_translate("Dialog", "2. Calibration"))
        self.groupBox_2_3.setTitle(_translate("Dialog", "2.3 Find Extrinsic"))
        self.label_2_3.setText(_translate("Dialog", "Select image:"))
        self.pushButton_2_1.setText(_translate("Dialog", "2.1 Find Corners"))
        self.pushButton_2_2.setText(_translate("Dialog", "2.2 Find Intrinsic"))
        self.pushButton_2_3.setText(_translate("Dialog", "2.3 Find Extrinsic"))
        self.pushButton_2_4.setText(_translate("Dialog", "2.4 Find Distortion"))
        self.pushButton_2_5.setText(_translate("Dialog", "2.5 Show Result"))
        self.groupBox_3.setTitle(_translate("Dialog", "3. Augmented Reality"))
        self.pushButton_3_1.setText(_translate("Dialog", "3.1 Show Words on Board"))
        self.pushButton_3_2.setText(_translate("Dialog", "3.2 Show Words Vertically"))
        self.groupBox_4.setTitle(_translate("Dialog", "4. Stereo Disparity Map"))
        self.pushButton_4_1.setText(_translate("Dialog", "4.1 Stereo Disparity Map"))
        #第二題Camera Calibration
        self.pushButton_2_1.clicked.connect(self.find_corner)
        self.pushButton_2_2.clicked.connect(self.find_intrinsic)
        self.pushButton_2_3.clicked.connect(self.find_extrinsic)
        self.pushButton_2_4.clicked.connect(self.find_distortion)
        self.pushButton_2_5.clicked.connect(self.show_disorted)

    def corner_detect(self, draw = True):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.objp = np.zeros((self.PATTERN_SIZE_2[0]*self.PATTERN_SIZE_2[1], 3), dtype=np.float32)
        self.objp[:, :2] = np.indices(self.PATTERN_SIZE_2).T.reshape(-1, 2)
        c = 0
        for frame in self.images_2:
            c +=1
            img = cv2.resize(cv2.imread(frame), (512, 512), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self.msk.append(gray)
            ret, corners = cv2.findChessboardCorners(self.msk[c-1], self.PATTERN_SIZE_2)
            self.corners = []
            self.corners.append(corners)
            if ret:
                if draw:
                    img_draw = cv2.drawChessboardCorners(img, self.PATTERN_SIZE_2, corners, ret)
                    cv2.imshow("result", img_draw)
                    cv2.waitKey(500)
                    cv2.destroyWindow("result")
                corners = cv2.cornerSubPix(self.msk[c-1],corners,(self.PATTERN_SIZE_2[0],self.PATTERN_SIZE_2[0]),(-1,-1),criteria)
                self.objpoints.append(self.objp)
                self.imgpoints.append(corners)
            else:
                print(f"{frame} No Checkerboard Found")

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1], None, None)
        self.mtx = mtx
        self.dist = dist
        self.rvecs = rvecs
        self.tvecs = tvecs
        return ret, mtx, dist, rvecs, tvecs
    #2.1
    def find_corner(self):
        print("function 2_1_(find_corner) clicked")
        count = 0
        for image in self.images_2:
            count=count+1
            img = cv2.resize(cv2.imread(image), (512, 512), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self.imglist.append(gray)
            ret, corners = cv2.findChessboardCorners(self.imglist[count-1], (11,8))
            img_draw = cv2.drawChessboardCorners(img, (11,8), corners, ret)
            cv2.imshow("result", img_draw)
            cv2.waitKey(500)
            cv2.destroyWindow("result")
    #2.2
    def find_intrinsic(self):
        print("function 2_2_(find_intrinsic) clicked")
        c = 0
        for i in self.images_2:
            c += 1
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            objp = np.zeros((self.PATTERN_SIZE_2[0]*self.PATTERN_SIZE_2[1], 3), dtype=np.float32)
            objp[:, :2] = np.indices((self.PATTERN_SIZE_2[0],self.PATTERN_SIZE_2[1])).T.reshape(-1, 2)
            img = cv2.resize(cv2.imread(i), (512, 512), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self.imglist.append(gray)
            size = gray.shape[::-1]
            ret, corners = cv2.findChessboardCorners(self.imglist[c-1], (self.PATTERN_SIZE_2[0],self.PATTERN_SIZE_2[1]))
            corners_sub = cv2.cornerSubPix(self.imglist[c-1],corners,(self.PATTERN_SIZE_2[0],self.PATTERN_SIZE_2[0]),(-1,-1),criteria)
            self.objpoints.append(objp)
            if [corners_sub]:
                self.imgpoints.append(corners_sub)
            else:
                self.imgpoints.append(corners)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, size , None, None)
        self.mtx.append(mtx)
        self.dist.append(dist)
        self.rvecs.append(rvecs)
        self.tvecs.append(tvecs)
        print(f"intrinsic matrix:\n {self.mtx[0]}")
    #2.3
    def find_extrinsic(self):
        print("function 2_3_(find_extrinsic) clicked")
        Input = int(self.lineEdit_2_3.text())
        self.corner_detect(draw = False)
        rvecs = cv2.Rodrigues(self.rvecs[Input-1])
        print(f"EXTRINSIC {Input} : ")
        print(np.hstack([rvecs[0] , self.tvecs[Input-1]]))
    #2.4
    def find_distortion(self):
        print("function 2_4_(find_distortion) clicked")
        self.corner_detect(draw = False)
        print("Distorition: ")
        print(self.dist)
    #2.5
    def show_disorted(self):
        print("function 2_5_(show_disorted) clicked")
        self.corner_detect(draw = False)
        for frame in self.images_2:
            img2 = cv2.resize(cv2.imread(frame), (512, 512), interpolation=cv2.INTER_AREA)
            h, w = img2.shape[:2]
            newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w,h), 0, (w,h))
            dst = cv2.undistort(img2, self.mtx, self.dist, None, newcameramatrix)
            output = np.hstack([img2, dst])
            cv2.imshow("result", output)
            cv2.waitKey(500)
            cv2.destroyWindow("result")




if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
