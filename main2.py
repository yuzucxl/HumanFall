import os
import threading

import cv2
import time

import pygame

import numpy
import torch
import argparse
import numpy as np

from Detection.Utils import ResizePadding
from CameraLoader import CamLoader, CamLoader_Q

# from V5Detector import Detector

from DetectorLoader import TinyYOLOv3_onecls

from PoseEstimateLoader import SPPE_FastPose
from fn import draw_single

from Track.Tracker import Detection, Tracker

from ActionsEstLoader import TSSTG

import pyttsx3

import urllib
import urllib.request
import hashlib
from concurrent.futures import ThreadPoolExecutor

'''==================输入资源选择======================='''
# 视频q
# source = 'data/video/fall.avi'
source = 'data/video/tplink.mp4'
# 网络摄像头
# source = 'rtsp://admin:q123456789@192.168.1.175/stream2'
# 电脑摄像头
# source = '0'
'''============================================='''

'''===================短信发送部分==================='''
"""使用短信宝平台提供的API，网址为 http://smsbao.com/ """


def md5(str):
    import hashlib
    m = hashlib.md5()
    m.update(str.encode("utf8"))
    return m.hexdigest()

def generate_sms_url(phone, content):
    smsapi = "http://api.smsbao.com/"
    user = '13036156947'
    password = md5('lx123456')
    data = urllib.parse.urlencode({'u': user, 'p': password, 'm': phone, 'c': content})
    send_url = smsapi + 'sms?' + data
    return send_url

statusStr = {
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

# smsapi = "http://api.smsbao.com/"
# # 短信平台账号
# user = '13036156947'
# # 短信平台密码
# password = md5('lx123456')
# # 要发送的短信内容
# content = '发现跌倒'
# # 要发送短信的手机号码
# phone = '15638820941'
#
# data = urllib.parse.urlencode({'u': user, 'p': password, 'm': phone, 'c': content})
# send_url = smsapi + 'sms?' + data

# response = urllib.request.urlopen(send_url)
# the_page = response.read().decode('utf-8')
# print(statusStr[the_page])
'''============================================='''

'''===================语音报警======================'''


def warning(phone_number='15638820941'):

    pygame.init()
    sound = pygame.mixer.Sound("发现跌倒.wav")
    sound.play()
    pygame.time.delay(2000)  # 以毫秒为单位，此处播放2秒钟
    if phone_number:
        response = urllib.request.urlopen(generate_sms_url(phone_number, '发现跌倒'))
        the_page = response.read().decode('utf-8')
        print(statusStr[the_page])


'''============================================='''


def preproc(image):
    """preprocess function for CameraLoader.
    CameraLoader的预处理功能
    """
    image = resize_fn(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def kpt2bbox(kpt, ex=20):
    """Get bbox that hold on all of the keypoints (x,y)
    获取已经包含了所有关键点（x,y）坐标的bbox
    kpt: array of shape `(N, 2)`,
    ex: (int) expand bounding box,
    """
    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                     kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))


if __name__ == '__main__':
    par = argparse.ArgumentParser(description='Human Fall Detection Demo.')  # argparse是python用于解析命令行参数和选项的标准模块
    par.add_argument('-C', '--camera', default=source,  # required=True,  # default=2,
                     help='Source of camera or video file path.')  # help - 一个此选项作用的简单描述。
    par.add_argument('--detection_input_size', type=int, default=384,  # 输入尺寸 默认384
                     help='Size of input in detection model in square must be divisible by 32 (int).')
    par.add_argument('--pose_input_size', type=str, default='224x160',
                     help='Size of input in pose model must be divisible by 32 (h, w)')
    par.add_argument('--pose_backbone', type=str, default='resnet50',
                     help='Backbone model for SPPE FastPose model.')
    par.add_argument('--show_detected', default=False, action='store_true',
                     help='Show all bounding box from detection.')
    par.add_argument('--show_skeleton', default=True, action='store_true',
                     help='Show skeleton pose.')
    par.add_argument('--save_out', type=str, default='',
                     help='Save display to video file.')
    par.add_argument('--device', type=str, default='cuda',
                     help='Device to run model on cpu or cuda.')
    args = par.parse_args()
    device = args.device

    current_thread = None

    pool = ThreadPoolExecutor(max_workers=4)  # 初始化线程池

    fall_detected = True

    min_alert_interval = 2.5  # 最小报警间隔
    last_alert_time = 0  # 全局变量,保存上次报警时间
    fall_start_time = None  # 跌倒开始时间
    fall_duration_threshold = 2.5  # 跌倒持续时间，要是一直跌倒，还没躺下，说明不是跌倒
    # DETECTION MODEL. 加载目标检测模型
    inp_dets = args.detection_input_size
    # YOLOV5检测模型
    # detect_model = Detector(weights_path= 'yolov5s.pt')
    # YoloV3检测模型
    detect_model = TinyYOLOv3_onecls(inp_dets, device=device)

    # POSE MODEL.  加载AlphaPose姿态模型
    inp_pose = args.pose_input_size.split('x')  # 以乘×为分隔符
    inp_pose = (int(inp_pose[0]), int(inp_pose[1]))
    pose_model = SPPE_FastPose(args.pose_backbone, inp_pose[0], inp_pose[1], device=device)

    # Tracker.    卡尔曼滤波器 跟踪器
    max_age = 30
    tracker = Tracker(max_age=max_age, n_init=3)

    # Actions Estimate.  加载STGCN动作识别模型
    action_model = TSSTG()
    resize_fn = ResizePadding(inp_dets, inp_dets)

    ##### 调用视频还是摄像头
    cam_source = source
    if type(cam_source) is str and os.path.isfile(cam_source):  # os.path.isfile()：判断某一对象(需提供绝对路径)是否为文件
        # Use loader thread with Q for video file.视频
        cam = CamLoader_Q(cam_source, queue_size=1000, preprocess=preproc).start()
    else:
        # Use normal thread loader for webcam.摄像头
        cam = CamLoader(int(cam_source) if cam_source.isdigit() else cam_source, preprocess=preproc).start()

    # 是否保存输出 默认否
    outvid = False
    if args.save_out != '':
        outvid = True
        codec = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(args.save_out, codec, 30, (inp_dets * 2, inp_dets * 2))

    fps_time = 0
    f = 0
    while cam.grabbed():  # 读取帧
        f += 1
        frame = cam.getitem()  #
        image = frame.copy()

        # Detect humans bbox in the frame with detector model.使用目标检测模型检测帧中的人形框
        detected = detect_model.detect(frame)
        # print(detected) #打印出检测框坐标

        # 使用卡尔曼滤波器从前一帧信息预测当前帧的每个轨迹bbox。
        tracker.predict()

        # Merge two source of predicted bbox together.
        # 合并两个预测bbox源。
        for track in tracker.tracks:
            det = torch.tensor([track.to_tlbr().tolist() + [0.5]], dtype=torch.float32)  # tolist：将数组转换为列表
            detected = torch.cat([detected, det], dim=0) if detected is not None else det

        detections = []  # List of Detections object for tracking.  用于跟踪的检测对象列表。

        if detected is not None:
            poses = pose_model.predict(frame, detected[:, 0:4], detected[:, 4])  # 调用AlphaPose模型预测骨骼点

            # Create Detections object.  创建检测对象。
            detections = [Detection(kpt2bbox(ps['keypoints'].numpy()),
                                    np.concatenate((ps['keypoints'].numpy(),
                                                    ps['kp_score'].numpy()), axis=1),
                                    ps['kp_score'].mean().numpy()) for ps in poses]

            # VISUALIZE.  将检测框可视化
            if args.show_detected:
                for bb in detected[:, 0:5]:
                    frame = cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 1)  # 检测框绿色

        # Update tracks by matching each track information of current and previous frame or
        # create a new track if no matched.
        # 通过匹配当前帧和上一帧的每个轨迹信息来更新轨迹，
        # 如果不匹配，则创建新轨迹。
        tracker.update(detections)

        # Predict Actions of each track.  预测每个轨道的动作。
        for i, track in enumerate(tracker.tracks):
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            bbox = track.to_tlbr().astype(int)
            center = track.get_center().astype(int)

            action = 'pending..'  # 悬而未决的动作，及未定义的动作
            clr = (0, 255, 0)
            # Use 30 frames time-steps to prediction.  使用30帧时间步长进行预测。
            if len(track.keypoints_list) == 30:
                pts = np.array(track.keypoints_list, dtype=np.float32)  # 30帧图像数组
                out = action_model.predict(pts, frame.shape[:2])  # 调用STGCN模型预测动作
                action_name = action_model.class_names[out[0].argmax()]  # 预测类别
                print('======Human action: ', action_name, '======')
                action = '{}: {:.2f}%'.format(action_name, out[0].max() * 100)  # 显示动作名和概率
                # action = '{}'.format(action_name)  # 只显示动作名

                # if action_name == 'Fall Down': #如果动作为Fall Down
                #     clr = (255, 0, 0) #将字体颜色设置为红色
                #
                #     if fall_detected:
                #         # 判断时间间隔
                #         if time.time() - last_alert_time >= min_alert_interval:
                #             print('======Find human fall======')
                #             # 语音报警
                #             # warning("发现跌倒")
                #
                #             if current_thread is None or not current_thread.is_alive():
                #                 current_thread = threading.Thread(target=warning)
                #                 current_thread.start()
                #
                #
                #             # 更新last_alert_time
                #             last_alert_time = time.time()
                #
                #             # 短信发送
                #             # response = urllib.request.urlopen(send_url)
                #             # the_page = response.read().decode('utf-8')
                #             # print(statusStr[the_page])
                #
                #         fall_detected = False
                #
                # elif action_name == 'Lying Down': #如果动作为躺
                #     clr = (255, 200, 0) #字体颜色为黄色
                # else:
                #     fall_detected = True
                if action_name == 'Fall Down':
                    fall_start_time = time.time()
                    # clr = (255, 0, 0)  # 将字体颜色设置为红色

                elif action_name == 'Lying Down':
                    if fall_start_time:
                        if time.time() - fall_start_time <= fall_duration_threshold:
                            # 确认为真实跌倒
                            # print('Real fall detected')
                            clr = (255, 0, 0)  # 将字体颜色设置为红色
                            if time.time() - last_alert_time >= min_alert_interval:
                                print('======Find human fall======')
                                # if current_thread is None or not current_thread.is_alive():
                                #     current_thread = threading.Thread(target=warning)
                                #     current_thread.start()
                                pool.submit(warning)
                                # 更新last_alert_time
                                last_alert_time = time.time()
                            # other alert actions

                    lying_down_start_time = time.time()

            # VISUALIZE.  可视化
            if track.time_since_update == 0:
                if args.show_skeleton:
                    frame = draw_single(frame, track.keypoints_list[-1])  # 显示骨骼点
                frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
                # 画矩形框，图片、左上坐标、右下坐标、颜色、字体
                # frame = cv2.putText(frame, str(track_id), (center[0], center[1]), cv2.FONT_HERSHEY_COMPLEX,
                #                      0.4, (255, 0, 0), 2) #画出跟踪计数的数字
                frame = cv2.putText(frame, action, (bbox[0] + 5, bbox[1] + 15), cv2.FONT_HERSHEY_COMPLEX,
                                    0.4, clr, 1)
                # cv2.putText在图片上添加文字，图片、文字、文字位置、字体类型、字体大小、字体颜色、字体粗细

        # Show Frame.
        frame = cv2.resize(frame, (0, 0), fx=2., fy=2.)  # 改变图像大小
        frame = cv2.putText(frame, '%d, FPS: %f' % (f, 1.0 / (time.time() - fps_time)),
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)  # 添加在左上角显示时间和FPS
        frame = frame[:, :, ::-1]
        fps_time = time.time()

        if outvid:
            writer.write(frame)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clear resource.
    cam.stop()
    if outvid:
        writer.release()
    cv2.destroyAllWindows()
    pool.shutdown()
