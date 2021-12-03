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
        path = os.getcwd()
        #指定資料夾
        os.chdir("D:\\code\\image_processing_course\\hw3_matching")
        self.class_img=cv2.imread("")
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
        self.pushButton_2.setGeometry(QtCore.QRect(50, 90, 190, 28))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_3.setGeometry(QtCore.QRect(50, 150, 190, 28))
        self.pushButton_3.setObjectName("pushButton_3")
        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.groupBox.setTitle(_translate("Dialog", "Hw3_Matching_(NM6101098)"))
        self.pushButton_1.setText(_translate("Dialog", "load_image"))
        self.pushButton_2.setText(_translate("Dialog", "Texture_Matching"))
        self.pushButton_3.setText(_translate("Dialog", "Shape_Matching"))
        #按鈕事件
        self.pushButton_1.clicked.connect(self.load_image)
    def load_image(self):
        print('checked load_image')
        self.sub_window = SubWindow1_1()
        self.sub_window.show()
        pass
    def texture_matching(self):
        print('chicked texture_matching')
        start_time=time.time()

        end_time=time.time()
#參考的
class matching_function:
    def __init__(self, color, seat):
        self.color = color  # 顏色屬性
        self.seat = seat  # 座位屬性
    def _pad(self,X, k):
        XX_shape = tuple(np.subtract(X.shape, k.shape) + (1, 1))
        if XX_shape!=X.shape:
            P = np.subtract(X.shape, XX_shape) // 2
            MD = np.subtract(X.shape, XX_shape) % 2
            X_ = np.pad(X, ((P[0], P[0]+MD[0]), (P[1], P[1]+MD[1])), 'constant')
        else:
            X_ = X
        return X_

    def _DSP(self,X, k, iter=1):
        for i in range(iter):
            k_ = k / (k.shape[0] * k.shape[1])
            X_pad =self._pad(X, k_)
            view_shape = tuple(np.subtract(X_pad.shape, k_.shape) + 1) + k_.shape
            strides = X_pad.strides + X_pad.strides
            sub_matrices = self.as_strided(X_pad, view_shape, strides) 
            cv = np.einsum('klij,ij->kl', sub_matrices, k_)
            X = cv[::2, ::2]
        return X

    def _USP(self,DP, k, iter=1):
        for i in range(iter):
            DP_ = np.insert(DP, range(DP.shape[0]), 0, axis=0)
            X = np.insert(DP_, range(DP.shape[1]), 0, axis=1)
            k_ = k / (k.shape[0] * k.shape[1])
            X_pad =self._pad(X, k_)
            view_shape = tuple(np.subtract(X_pad.shape, k_.shape) + 1) + k_.shape
            strides = X_pad.strides + X_pad.strides
            sub_matrices =self.as_strided(X_pad, view_shape, strides) 
            DP = np.einsum('klij,ij->kl', sub_matrices, k_)
        return DP

    def _nor(X, h, w):
        X_ = X - np.sum(X) / (h*w)
        return X_

    def _CC(X, Y):
        res = np.sum(X * Y) / np.sqrt(np.sum(X**2) * np.sum(Y**2))
        return res

    def _sub(self,I, T):
        view_shape = tuple(np.subtract(I.shape, T.shape) + 1) + T.shape
        strides = I.strides + I.strides
        sub_matrices =self.as_strided(I, view_shape, strides)
        return sub_matrices

    def _match(self,sub_matrices, T):
        h_, w_, h, w = sub_matrices.shape
        L = []
        T_ = self._nor(T, h, w)
        for y, x in product(range(h_), range(w_)):
            S_ = self._nor(sub_matrices[y, x, :, :], h, w)
            L.append(self._CC(T_, S_))
        res = np.array(L).reshape(h_, w_)
        return res

    def _split(path, img_type='100'):
        tot_list = os.listdir(path)
        img_list, tpl_list = [], []
        for j in tot_list:
            k = j.replace('.', '-')
            if k.split('-')[0]==img_type and k.split('-')[-2]!='MatchResult':
                if k.split('-')[-2]=='Template':
                    tpl_list.append(j)
                else:
                    img_list.append(j)
        return img_list, tpl_list

    def _NMS(boxes, overlapThresh):
        # if there are no boxes, return an empty list
        if len(boxes) == 0:
            return []
        # if the bounding boxes integers, convert them to floats --
        # this is important since we'll be doing a bunch of divisions
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")
        # initialize the list of picked indexes	
        pick = []
        # grab the coordinates of the bounding boxes
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]
        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)
        # keep looping while some indexes still remain in the indexes
        # list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]
            # delete all indexes from the index list that have
            idxs = np.delete(idxs, np.concatenate(([last],
                np.where(overlap > overlapThresh)[0])))
        # return only the bounding boxes that were picked using the
        # integer data type
        return boxes[pick].astype("int")  

    def _getBox(self,res, T_org, thrs):
        M = np.where(res>thrs, 1, 0) 
        box_i, box_j = np.where(M!=0)
        h, w = T_org.shape
        boxes = np.vstack([box_j - w//2, box_i - h//2,\
                    box_j + w//2, box_i + h//2]).T
        box_res = self._NMS(boxes, 0.4)
        return box_res  

    def _plotBox(I_org, box_res):
        I_box_R = cv2.cvtColor(I_org, cv2.COLOR_GRAY2BGR)
        for i in range(len(box_res)):
            x1, y1 = box_res[i, :2]
            x2, y2 = box_res[i, 2:]
            mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
            text_X = 'X: ' + str(mid_x)
            text_Y = 'Y: ' + str(mid_y)
            cv2.rectangle(I_box_R, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.putText(I_box_R, text_X, (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(I_box_R, text_Y, (mid_x, mid_y+45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        return I_box_R    

    #使用
    def _use_function(self):
        start_time = time.time()
        path = '../img'
        img_list, tpl_list = self._split(path, img_type='100')
        img_org = cv2.imread(os.path.join(path, img_list[1]))
        tpl = cv2.imread(os.path.join(path, tpl_list[0]))

        fig = plt.figure()
        plt.imshow(img_org)

        fig = plt.figure()
        plt.imshow(tpl)
        #顯示
        I_org = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
        T_org = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)

        G = np.array([[1,  4,  6,  4, 1],
                    [4, 16, 24, 16, 4],
                    [6, 24, 36, 24, 6],
                    [4, 16, 24, 16, 4],
                    [1,  4,  6,  4, 1]])
        I = self._DSP(I_org, G/16, iter=3)
        T = self._DSP(T_org, G/16, iter=3)
        I_pad = self._pad(I, T)

        sub_matrices = self._sub(I_pad, T)
        CC = self._match(sub_matrices, T)

        res = self._USP(CC, G/4, iter=3)

        fig = plt.figure()
        plt.imshow(res)
        #box顯示
        box_res = self._getBox(res, T_org, 0.12)
        #NMS bouning box結果
        I_box_R = self._plotBox(I_org, box_res)

        fig = plt.figure()
        plt.imshow(cv2.cvtColor(I_box_R, cv2.COLOR_BGR2RGB))

        end_time = time.time()
        print ("\n" + "It cost {:.4f} sec" .format(end_time-start_time))

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
        print(path)
        fileName = QFileDialog.getOpenFileName(self,'Open File',str(path),'*.jpg')
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