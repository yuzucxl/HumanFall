import os
import time

import cv2
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from Camera2 import Ui_MainWindow
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import QtCore

import sys

class myMainWindow(QMainWindow,Ui_MainWindow,QWidget):
    def __init__(self,parent=None):
        super(myMainWindow, self).__init__(parent)
        self.setupUi(self) #加载UI
        # self.cap = cv2.VideoCapture()

        self.bn_openfile.clicked.connect(self.openVideoFile)
        self.timer_VideoFile = QtCore.QTimer()
        self.timer_VideoFile.timeout.connect(self.showVideoFile)

        self.bn_pause.clicked.connect(self.pause)
        self.bn_pause.setEnabled(False)
        self.bn_opencam1.clicked.connect(self.openCamera1)
        self.bn_opencam2.clicked.connect(self.openCamera2)
        self.timer_Camera = QtCore.QTimer()
        self.timer_Camera.timeout.connect(self.showVideoFile)

        self.bn_end.clicked.connect(self.endall)

    def openVideoFile(self):
        if self.timer_VideoFile.isActive() == False:
            imgName, imgType = QFileDialog.getOpenFileName(self, "打开视频", "", "*.mp4;;*.AVI;;*.rmvb;;All Files(*)")  #PyQt5.QtCore.QUrl('file:///D:/.../...')
            self.cap_video = cv2.VideoCapture(imgName)
            flag = self.cap_video.isOpened()
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"请检测视频文件是否正确",
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_VideoFile.start(30)
                self.bn_pause.setEnabled(True)

    def pause(self):
        if self.timer_VideoFile.isActive() == True:
            self.timer_VideoFile.stop()
            self.bn_pause.setText(u'继续')
        else:
            # if self.bn_pause
            self.timer_VideoFile.start(30)
            self.bn_pause.setText(u'暂停')

    def openCamera1(self):
        if self.timer_Camera.isActive() == False:
            self.cap_video = cv2.VideoCapture(0)
            flag = self.cap_video.isOpened()
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"请检测相机与电脑是否连接正确",
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_Camera.start(1)
                self.bn_opencam1.setText(u'关闭摄像头')
        else:
            self.timer_Camera.stop()
            self.cap_video.release()
            self.video1.clear()
            self.bn_opencam1.setText(u'打开本地摄像头')

    def openCamera2(self):
        if self.timer_Camera.isActive() == False:
            self.cap_video = cv2.VideoCapture("rtsp://admin:q123456789@192.168.1.200/Streaming/Channels/102")
            flag = self.cap_video.isOpened()
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"请检测网络摄像头是否连接正确",
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                     defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_Camera.start(50)
                self.bn_opencam2.setText(u'关闭网络摄像头')
        else:
            self.timer_Camera.stop()
            self.cap_video.release()
            self.video1.clear()
            self.bn_opencam2.setText(u'打开网络摄像头')

    def showVideoFile(self):
        flag, self.image = self.cap_video.read()
        length = int(self.cap_video.get(cv2.CAP_PROP_FRAME_COUNT))  # 帧长
        if flag == True:
            width = self.image.shape[1]
            height = self.image.shape[0]
            # 设置新的图片分辨率框架
            width_new = 420
            height_new = 420
            # 判断图片的长宽比率
            if width / height >= width_new / height_new:
                show = cv2.resize(self.image, (width_new, int(height * width_new / width)))
            else:
                show = cv2.resize(self.image, (int(width * height_new / height), height_new))

            show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0],3 * show.shape[1], QtGui.QImage.Format_RGB888)
            self.video1.setPixmap(QtGui.QPixmap.fromImage(showImage))
        else:
            self.cap_video.release()
            self.video1.clear()
            self.timer_VideoFile.stop()
            self.bn_pause.setText(u'暂停')
            self.bn_pause.setEnabled(False)

    def endall(self):
        self.timer_Camera.stop()
        self.timer_VideoFile.stop()
        self.cap_video.release()
        self.video1.clear()
        self.video2.clear()
        self.bn_opencam1.setText(u'打开本地摄像头')
        self.bn_opencam2.setText(u'打开网络摄像头')
        self.bn_pause.setText(u'暂停')
        self.bn_pause.setEnabled(False)

if __name__ == '__main__':

    # 实例化，传参
    app = QApplication(sys.argv)

    ########### 启动界面 ###########
    splash = QSplashScreen(QPixmap(".\\data_img\\source_image\\logo.png"))
    splash.setFont(QFont('Microsoft YaHei UI', 12))  # 设置画面中的文字的字体
    splash.show()     # 显示画面
    # 显示信息
    d = 0
    for i in range(25):
        splash.showMessage(f"程序初始化中... {d}%", QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom, QtCore.Qt.black)
        d += 4
        time.sleep(0.01)

    # 创建对象
    mainWindow = myMainWindow()
    mainWindow.show()
    splash.close()

    sys.exit(app.exec_())