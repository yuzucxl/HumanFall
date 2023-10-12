import os
import time
import sys
#import qimage2ndarray

import cv2
import pygame
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from Camera2 import Ui_MainWindow
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import QtCore
import pyttsx3
import time
import threading

import torch
import argparse
import numpy as np

from Detection.Utils import ResizePadding
from CameraLoader import CamLoader, CamLoader_Q
from DetectorLoader import TinyYOLOv3_onecls
from V5Detector import Detector
from PoseEstimateLoader import SPPE_FastPose
from fn import draw_single
from Track.Tracker import Detection, Tracker
from ActionsEstLoader import TSSTG
import urllib
import urllib.request
from concurrent.futures import ThreadPoolExecutor
source = 'rtsp://admin:q123456789@192.168.1.200/Streaming/Channels/101'

class myMainWindow(QMainWindow,Ui_MainWindow,QWidget):
    def __init__(self):
        super(myMainWindow, self).__init__()
        self.setupUi(self) #加载UI

        self.par = argparse.ArgumentParser(description='Human Fall Detection Demo.')
        # self.par.add_argument('-C', '--camera', default=source,  # required=True,  # default=2,
        #                       help='Source of camera or video file path.')
        self.par.add_argument('--detection_input_size', type=int, default=320,
                              help='Size of input in detection model in square must be divisible by 32 (int).')
        self.par.add_argument('--pose_input_size', type=str, default='224x160',
                              help='Size of input in pose model must be divisible by 32 (h, w)')
        self.par.add_argument('--pose_backbone', type=str, default='resnet50',
                              help='Backbone model for SPPE FastPose model.')
        self.par.add_argument('--show_detected', default=False, action='store_true',
                              help='Show all bounding box from detection.')
        self.par.add_argument('--show_skeleton', default=True, action='store_true',
                              help='Show skeleton pose.')
        self.par.add_argument('--save_out', type=str, default='',
                              help='Save display to video file.')
        self.par.add_argument('--device', type=str, default='cuda',
                              help='Device to run model on cpu or cuda.')
        self.args = self.par.parse_args()
        self.device = self.args.device

        '''=============================按钮绑定============================='''
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


        self.yolov3.clicked.connect(self.Detection_model)
        self.yolov5.clicked.connect(self.Detection_model1)
        self.alphapose.clicked.connect(self.Pose_model)
        self.stgcn.clicked.connect(self.Actions_Estimate)

        self.statusStr = {
            '0': '短信发送成功',
            '-1': '参数不全',
            '-2': '服务器空间不支持,请确认支持curl或者fsocket,联系您的空间商解决或者更换空间',
            '30': '密码错误',
            '40': '账号不存在',
            '41': '余额不足',
            '42': '账户已过期',
            '43': 'IP地址限制',
            '50': '内容含有敏感词'
        }

        self.phone_number = None

        self.pool = ThreadPoolExecutor(max_workers=10)

        self.min_alert_interval = 5  # 最小报警间隔
        self.last_alert_time = 0  # 全局变量,保存上次报警时间
        self.fall_start_time = None  # 跌倒开始时间
        self.fall_duration_threshold = 2.5  # 跌倒持续时间，要是一直跌倒，还没躺下，说明不是跌倒
        self.lying_down_start_time = None

        self.current_thread = None
        '''=============================参数初始化============================='''
        self.flag = False
        self.cam = None
        self.fps_time = 0
        self.f = 0
        self.outvid = False
        #是否保存输出文件，暂时没用
        # if self.args.save_out != '':
        #     self.outvid = True
        #     self.codec = cv2.VideoWriter_fourcc(*'MJPG')
        #     self.writer = cv2.VideoWriter(self.args.save_out, self.codec, 30, (self.inp_dets * 2, self.inp_dets * 2))

        # self.CallBackFunctions()
        # self.Timer = QTimer()

        self.inp_dets = self.args.detection_input_size  # 384

    '''=============================模型加载============================='''
    def Detection_model(self):
        self.detect_model = TinyYOLOv3_onecls(self.inp_dets, device=self.device) #加载Yolov3模型
        self.yolov3.setText(u'yolov3')

    def Detection_model1(self):
        self.detect_model = Detector(weights_path= 'yolov5s.pt')
        self.yolov5.setText(u'yolov5')

    def Pose_model(self):
        self.inp_pose = self.args.pose_input_size.split('x')
        self.inp_pose = (int(self.inp_pose[0]), int(self.inp_pose[1]))
        self.pose_model = SPPE_FastPose(self.args.pose_backbone, self.inp_pose[0], self.inp_pose[1], device=self.device) #加载Alphapose模型
        self.alphapose.setText(u'alphapose')

    def Tacker(self):
        self.max_age = 30
        self.tracker = Tracker(max_age=self.max_age, n_init=3)

    def Actions_Estimate(self):
        self.action_model = TSSTG()  #加载TwoStreamSpatialTemporalGraph模型   两个STGCN融合
        self.resize_fn = ResizePadding(self.inp_dets, self.inp_dets)  # 返回一个函数地址   inp_dets=384，必须被32整除，是输入图像大小
        self.stgcn.setText(u'st-gcn')

    def Save_out(self):
        if self.args.save_out != '':
            self.outvid = True
            self.codec = cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(self.args.save_out, self.codec, 30, (self.inp_dets * 2, self.inp_dets * 2))

    def engine(self):   #语言报警部分
        engine = pyttsx3.init()
        voice = '发现跌倒'
        # for i in range(1):
        engine.say(voice)
        engine.runAndWait()
            # time.sleep(1)

    '''=============================短信生成==============================='''

    def md5(self, str):
        import hashlib
        m = hashlib.md5()
        m.update(str.encode("utf8"))
        return m.hexdigest()


    def generate_sms_url(self, phone, content):
        smsapi = "http://api.smsbao.com/"
        user = '13036156947'
        password = self.md5('lx123456')
        data = urllib.parse.urlencode({'u': user, 'p': password, 'm': phone, 'c': content})
        send_url = smsapi + 'sms?' + data
        return send_url


    '''===================语音报警======================'''


    def warning(self):

        pygame.init()
        sound = pygame.mixer.Sound("发现跌倒.wav")
        sound.play()
        pygame.time.delay(2000)  # 以毫秒为单位，此处播放2秒钟
        if self.phone_number:
            response = urllib.request.urlopen(self.generate_sms_url(self.phone_number, '发现跌倒'))
            the_page = response.read().decode('utf-8')
            print(self.statusStr[the_page])

    '''============================================='''
    '''=============================PyQt部分函数============================='''
    def openVideoFile(self):
        if self.timer_VideoFile.isActive() == False:
            imgName, imgType = QFileDialog.getOpenFileName(self, "打开视频", "", "*.mp4;;*.AVI;;*.rmvb;;All Files(*)")  #PyQt5.QtCore.QUrl('file:///D:/.../...')
            self.cam = self.caminit(imgName)
            flag = self.cam.grabbed()
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"请检测视频文件是否正确",
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.bn_pause.setEnabled(True)
                self.Warning_tag = True
                self.timer_VideoFile.start(1)

    def pause(self):
        if self.timer_VideoFile.isActive() == True:
            self.timer_VideoFile.stop()
            self.bn_pause.setText(u'继续')
        else:
            self.timer_VideoFile.start(100)
            self.bn_pause.setText(u'暂停')

    def openCamera1(self):
        if self.timer_Camera.isActive() == False:
            self.cam = self.caminit('0')
            flag = self.cam.grabbed()
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"请检测相机与电脑是否连接正确",
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                print('摄像头打开成功')
                self.Warning_tag = True
                self.timer_Camera.start(1)
                self.bn_opencam1.setText(u'关闭摄像头')
        else:
            self.timer_Camera.stop()
            self.cam.stop()
            self.video1.clear()
            self.video2.clear()
            self.bn_opencam1.setText(u'打开本地摄像头')

    def openCamera2(self):
        if self.timer_Camera.isActive() == False:
            self.cam = self.caminit(source)
            flag = self.cam.grabbed()
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"请检测网络摄像头是否连接正确",
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.Warning_tag = True
                self.timer_Camera.start(1)
                self.bn_opencam2.setText(u'关闭网络摄像头')
        else:
            self.timer_Camera.stop()
            self.cam.stop()
            self.video1.clear()
            self.video2.clear()
            self.bn_opencam2.setText(u'打开网络摄像头')

    def showVideoFile(self):
        flag = self.cam.grabbed()
        self.run()
        # self.image有了
        if flag == True:
            showImage = self.resizeImage(self.image)
            self.video1.setPixmap(QtGui.QPixmap.fromImage(showImage))
        else:  #针对的是视频文件
            self.cam.stop()
            self.video1.clear()
            self.video2.clear()
            self.timer_VideoFile.stop()
            self.bn_pause.setText(u'暂停')
            self.bn_pause.setEnabled(False)

    def endall(self):
        self.timer_Camera.stop()
        self.timer_VideoFile.stop()
        self.cam.stop()
        self.video1.clear()
        self.video2.clear()
        self.bn_pause.setText(u'暂停')
        self.bn_pause.setEnabled(False)
        self.bn_opencam1.setText(u'打开本地摄像头')
        self.bn_opencam2.setText(u'打开网络摄像头')

    '''=============================识别部分函数============================='''
    def caminit(self,cam_source):
        print('摄像头初始化中')
        if type(cam_source) is str and os.path.isfile(cam_source):  #判断cam_source是不是存在的视频文件
            # Use loader thread with Q for video file.
            cam = CamLoader_Q(cam_source, queue_size=1000, preprocess=self.preproc).start()
        else:
            # Use normal thread loader for webcam.
            cam = CamLoader(int(cam_source) if cam_source.isdigit() else cam_source,
                                 preprocess=self.preproc).start()
        print('摄像头初始化完毕')
        return cam

    def run(self):
        Warning = threading.Thread(target=self.engine, args=())

        if self.cam.grabbed():
            self.f += 1
            self.frame = self.cam.getitem()
            self.image = self.preproc(self.frame.copy())


            # Detect humans bbox in the frame with detector model.
            detected = self.detect_model.detect(self.frame)
            # if detected != []:
            # print('detected:',detected)

            # Predict each tracks bbox of current frame from previous frames information with Kalman filter.
            self.tracker.predict()
            # Merge two source of predicted bbox together.
            for track in self.tracker.tracks:
                det = torch.tensor([track.to_tlbr().tolist() + [0.5]], dtype=torch.float32)
                detected = torch.cat([detected, det], dim=0) if detected is not None else det
            detections = []  # List of Detections object for tracking.
            if detected is not None:
                # for track in self.tracker.tracks:
                #     det = torch.tensor([track.to_tlbr().tolist() + [0.5]], dtype=torch.float32)
                #     detected = torch.cat([detected, det], dim=0) if detected is not None else det
                # detections = []  # List of Detections object for tracking.
                # detected = non_max_suppression(detected[None, :], 0.45, 0.2)[0]
                # Predict skeleton pose of each bboxs.
                poses = self.pose_model.predict(self.frame, detected[:, 0:4], detected[:, 4])

                # Create Detections object.
                detections = [Detection(self.kpt2bbox(ps['keypoints'].numpy()),
                                        np.concatenate((ps['keypoints'].numpy(),
                                                        ps['kp_score'].numpy()), axis=1),
                                        ps['kp_score'].mean().numpy()) for ps in poses]

                # VISUALIZE.
                if self.args.show_detected:
                    for bb in detected[:, 0:5]:
                        self.frame = cv2.rectangle(self.frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 1)

            # Update tracks by matching each track information of current and previous frame or
            # create a new track if no matched.
            self.tracker.update(detections)

            # Predict Actions of each track.
            for i, track in enumerate(self.tracker.tracks):
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                bbox = track.to_tlbr().astype(int)
                center = track.get_center().astype(int)

                action = 'pending..'
                clr = (0, 255, 0)
                # Use 30 frames time-steps to prediction.
                if len(track.keypoints_list) == 30:
                    pts = np.array(track.keypoints_list, dtype=np.float32)
                    out = self.action_model.predict(pts, self.frame.shape[:2])
                    action_name = self.action_model.class_names[out[0].argmax()]
                    action = '{}: {:.2f}%'.format(action_name, out[0].max() * 100)
                    if action_name == 'Fall Down':
                        self.fall_start_time = time.time()
                        # clr = (255, 0, 0)  # 将字体颜色设置为红色

                    elif action_name == 'Lying Down':
                        if self.fall_start_time:
                            if time.time() - self.fall_start_time <= self.fall_duration_threshold:
                                # 确认为真实跌倒
                                # print('Real fall detected')
                                clr = (255, 0, 0)  # 将字体颜色设置为红色
                                if time.time() - self.last_alert_time >= self.min_alert_interval:
                                    print('======Find human fall======')
                                    # if current_thread is None or not current_thread.is_alive():
                                    #     current_thread = threading.Thread(target=warning)
                                    #     current_thread.start()
                                    self.pool.submit(self.warning)
                                    # if self.current_thread is None or not self.current_thread.is_alive():
                                    #     current_thread = threading.Thread(target=self.warning)
                                    #     current_thread.start()
                                    # 更新last_alert_time
                                    self.last_alert_time = time.time()
                                # other alert actions

                    # if action_name == 'Fall Down':
                    #     if self.Warning_tag:
                    #         Warning.start()
                    #         self.Warning_tag = False
                    #     clr = (255, 0, 0)
                    # elif action_name == 'Lying Down':
                    #     clr = (255, 200, 0)
                    # else:
                    #     self.Warning_tag = True


                # VISUALIZE.
                if track.time_since_update == 0:
                    if self.args.show_skeleton:
                        self.frame = draw_single(self.frame, track.keypoints_list[-1])
                    self.frame = cv2.rectangle(self.frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
                    # self.frame = cv2.putText(self.frame, str(track_id), (center[0], center[1]),
                    #                          cv2.FONT_HERSHEY_COMPLEX,
                    #                          0.4, (255, 0, 0), 2)
                    self.frame = cv2.putText(self.frame, action, (bbox[0] + 5, bbox[1] + 15), cv2.FONT_HERSHEY_COMPLEX,
                                             0.4, clr, 1)

            # Show Frame.
            self.frame = cv2.resize(self.frame, (0, 0), fx=2., fy=2.)
            self.frame = cv2.putText(self.frame, '%d, FPS: %f' % (self.f, 1.0 / (time.time() - self.fps_time)),
                                     (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            self.frame = self.frame[:, :, ::-1]
            self.fps_time = time.time()

            if self.outvid:
                self.writer.write(self.frame)
            # else:
            #     self.frame = self.frame[:, :, ::-1]  # 从BGR转为RGB
        self.DispImg()

    def DispImg(self):
        showImage = self.resizeImage(self.frame)
        self.video2.setPixmap(QtGui.QPixmap.fromImage(showImage))

    def kpt2bbox(self, kpt, ex=20):
        """Get bbox that hold on all of the keypoints (x,y)
        kpt: array of shape `(N, 2)`,
        ex: (int) expand bounding box,
        """
        return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                         kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))

    def preproc(self, image):
        """preprocess function for CameraLoader.
        """
        image = self.resize_fn(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    '''=============================公用函数============================='''
    def resizeImage(self,image):
        width = image.shape[1]
        height = image.shape[0]
        # 设置新的图片分辨率框架
        width_new = 420
        height_new = 420
        # 判断图片的长宽比率
        if width / height >= width_new / height_new:
            show = cv2.resize(image, (width_new, int(height * width_new / width)))
        else:
            show = cv2.resize(image, (int(width * height_new / height), height_new))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], 3 * show.shape[1], QtGui.QImage.Format_RGB888)
        return showImage

    def closeEvent(self, event):
        # 在窗口关闭时关闭线程池
        self.pool.shutdown(wait=True)
        event.accept()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    splash = QSplashScreen(QPixmap(".\\data_img\\source_image\\logo.png"))
    splash.setFont(QFont('Microsoft YaHei UI', 12))  # 设置画面中的文字的字体
    splash.show()     # 显示画面
    # 显示信息
    mainWindow = myMainWindow()
    # mainWindow.closed.connect(mainWindow.closePool)
    splash.showMessage(f"程序初始化中... 0%", QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom, QtCore.Qt.black)
    # mainWindow.Detection_model()
    splash.showMessage(f"程序初始化中... 20%", QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom, QtCore.Qt.black)
    # mainWindow.Pose_model()
    splash.showMessage(f"程序初始化中... 40%", QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom, QtCore.Qt.black)
    mainWindow.Tacker()
    splash.showMessage(f"程序初始化中... 60%", QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom, QtCore.Qt.black)
    # mainWindow.Actions_Estimate()
    splash.showMessage(f"程序初始化中... 80%", QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom, QtCore.Qt.black)
    mainWindow.Save_out()
    splash.showMessage(f"程序初始化中... 100%", QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom, QtCore.Qt.black)
    mainWindow.show()

    splash.close()

    sys.exit(app.exec_())
