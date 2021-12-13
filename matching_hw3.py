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
        file_name=FILENAME.split('/')[:-2]
        path=""
        for i in file_name:
            name=i+"//"
            path=path+name
        #指定資料夾
        os.chdir("C:\\Users\\nm610\\Hw3_img")
        self.path = path
        #100資料夾的template
        self.template_100=cv2.imread(path+'100\\100-Template.jpg',cv2.IMREAD_GRAYSCALE)
        #die資料夾的template
        self.template_die=cv2.imread(path+'Die\\Die-Template.jpg',cv2.IMREAD_GRAYSCALE)
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
        self.pushButton_3 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_3.setGeometry(QtCore.QRect(50, 130, 190, 28))
        self.pushButton_3.setObjectName("pushButton_3")
        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.groupBox.setTitle(_translate("Dialog", "Hw3_Matching_(NM6101098)"))
        self.pushButton_1.setText(_translate("Dialog", "load_image"))
        self.pushButton_2.setText(_translate("Dialog", "Texture_Matching"))
        self.pushButton_3.setText(_translate("Dialog", "openCV_Matching"))
        #按鈕事件
        self.pushButton_1.clicked.connect(self.load_image)
        self.pushButton_2.clicked.connect(self.texture_matching)
        self.pushButton_3.clicked.connect(self.openCV_process)
    def load_image(self):
        print('checked load_image')
        self.sub_window = SubWindow1_1()
        self.sub_window.show()
        pass
    def texture_matching(self):
        print('chicked texture_matching')
        self.matching=matching_function('D:\\code\\image_processing_course\\hw3_matching\\Hw3_img\\')
        self.matching._use_function()
    def openCV_process(self):
        start = time.time()
        #matching影像
        matching_img = cv2.imread(FILENAME,cv2.IMREAD_GRAYSCALE)
        file_name=FILENAME.split('/')[-2]
        #判斷template影像
        if file_name =='100':
            template_img = self.template_100
        if file_name =='Die':
            template_img = self.template_die
        h,w = template_img.shape
        result_img = np.array(matching_img)
        res = cv2.matchTemplate(result_img, template_img, cv2.TM_CCOEFF_NORMED)
        threshold = 0.96
        cv2.normalize(res, res, 0, 1, cv2.NORM_MINMAX)
        loc = np.where(res >= threshold)
        BGR_img = cv2.cvtColor(result_img, cv2.COLOR_GRAY2BGR)
        for point in zip(*loc[::-1]):
            cv2.rectangle(BGR_img, point, (point[0] + w, point[1] + h), (0, 0, 255), 1)
        end = time.time()
        print("==========================")
        print('openCV run-time:', end-start)
        print("==========================")
        cv2.imshow('opencv_result',BGR_img)
        cv2.waitKey(0)

class matching_function:
    def __init__(self, path):
        file_name=FILENAME.split('/')[:-2]
        path=""
        for i in file_name:
            name=i+"//"
            path=path+name
        #指定資料夾
        os.chdir("C:\\Users\\nm610\\Hw3_img")
        self.path = path
        #100資料夾的template
        self.template_100=cv2.imread(path+'100\\100-Template.jpg')
        #die資料夾的template
        self.template_die=cv2.imread(path+'Die\\Die-Template.jpg')
        #顯示字
        self.text_X = 'X: ' 
        self.text_Y = 'Y: ' 
        self.text_S ='Score' 
    def downsample(self,image):
        img_down = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
        return img_down

    def upsample(self,image):
        img_up = cv2.resize(image, (0, 0), fx=4, fy=4)
        return img_up
    def sub_martix(self,I, T):
        #先處理I
        _shape = tuple(np.subtract(I.shape, T.shape) + (1, 1))
        if _shape!=I.shape:
            P = np.subtract(I.shape, _shape) // 2
            MD = np.subtract(I.shape, _shape) % 2
            new_I = np.pad(I, ((P[0], P[0]+MD[0]), (P[1], P[1]+MD[1])), 'constant')
        else:
            new_I = I


        view_shape = tuple(np.subtract(new_I.shape, T.shape) + 1) + T.shape
        total_strides = new_I.strides + new_I.strides
        sub_matrices =as_strided(new_I, view_shape, total_strides)
        return sub_matrices

    def process_box(self,res,template_img,matching_img, thrs):
        BGR_img = cv2.cvtColor(matching_img, cv2.COLOR_GRAY2BGR)
        M = np.where(res>thrs, 1, 0) 
        box_i, box_j = np.where(M!=0)
        h, w = template_img.shape
        boxes = np.vstack([box_j - w//2, box_i - h//2,box_j + w//2, box_i + h//2]).T
        
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]
        #算boxes數量
        box_count = np.argsort(x1)
        mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
        score = res[mid_y, mid_x]
        #NMS選最大
        max_value,order,max_value_order = None,0,0
        for num in score:
            if (max_value is None or num > max_value):
                max_value = num
                max_value_order=order
            order=order+1
        boxes=boxes[max_value_order]
        #算中心點，x相加//y相加
        mid_x, mid_y = (boxes[0] + boxes[2]) // 2, (boxes[1] + boxes[3]) // 2
        #計算分數
        score = res[mid_y, mid_x]
        #標示
        self.text_X = self.text_X + str(mid_x)
        self.text_Y = self.text_Y + str(mid_y)
        self.text_S = self.text_S + str(np.round(score, 2))
        #畫框
        cv2.rectangle(BGR_img, (boxes[0], boxes[1]), (boxes[2], boxes[3]), (0,255,0), 1)
        cv2.putText(BGR_img, self.text_X, (mid_x, mid_y), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 0,0), 1)
        cv2.putText(BGR_img, self.text_Y, (mid_x, mid_y+20), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 0,0), 1)
        cv2.putText(BGR_img, self.text_S, (mid_x, mid_y+40), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 0,0), 1)
        return BGR_img 
    def match(self,matching_img,template_img):
        new_I = self.sub_martix(matching_img, template_img)
        h1, w1,h2, w2 =new_I.shape
        temp_list = []
        process_t = template_img - np.sum(template_img) / (h2*w2)
        for y, x in product(range(h1), range(w1)):
            process_s = new_I[y, x, :, :] - np.sum(new_I[y, x, :, :]) / (h2*w2)
            temp = np.sum(process_t * process_s) / np.sqrt(np.sum(process_t**2) * np.sum(process_s**2))
            temp_list.append(temp)
        res = np.array(temp_list).reshape(h1, w1)
        return res

    #使用
    def _use_function(self):
        #cv2.destroyAllWindows()
        #matching影像
        img = cv2.imread(FILENAME)
        matching_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        file_name=FILENAME.split('/')[-2]
        #判斷template影像
        if file_name =='100':
            template_img = cv2.cvtColor(self.template_100, cv2.COLOR_BGR2GRAY)
        if file_name =='Die':
            template_img = cv2.cvtColor(self.template_die, cv2.COLOR_BGR2GRAY)
        start_time = time.time()

        #down_sampling
        matching_img=self.downsample(matching_img)
        template_img = self.downsample(template_img)

        #matching
        res=self.match(matching_img,template_img)

        #box顯示
        box_res = self.process_box(res, template_img,matching_img, 0.8)
        #up_sampling
        box_res = self.upsample(box_res)
        
        result=cv2.cvtColor(box_res, cv2.COLOR_BGR2RGB)
        end_time = time.time()
        print("==========================")
        print('new run-time:', end_time-start_time)
        print("==========================")
        cv2.imshow('result',result)
        cv2.waitKey(0)

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
        src = cv2.imread(FILENAME)
        #設定text數值
        # self.height.setText(f"Width = {src.shape[0]}")
        # self.width.setText(f"Height = {src.shape[1]}")
        # cv2.namedWindow(FILENAME,cv2.WINDOW_AUTOSIZE)
        # cv2.imshow(FILENAME,src)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())