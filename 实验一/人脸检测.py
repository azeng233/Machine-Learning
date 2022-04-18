# -*- coding: utf-8 -*-
import cv2


# 人脸检测函数
def face_rec(img):
    # 转为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 加载人脸训练数据
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_cascade.load('haarcascade_frontalface_default.xml')
    # 加载人眼训练数据
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    eye_cascade.load('haarcascade_eye.xml')
    # 人脸检测
    faces = face_cascade.detectMultiScale(gray,
                                          scaleFactor=1.15,
                                          minNeighbors=3,
                                          minSize=(3, 3),
                                          flags=cv2.IMREAD_GRAYSCALE)
    # 在人脸周围绘制方框
    for (x, y, w, h) in faces:
        img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # 进行眼部检测
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 3, 0, (40, 40))
    for (ex, ey, ew, eh) in eyes:
        # 绘制眼部方框
        img = cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    cv2.imshow('result', img)


# -----------------------------------------------------------------------------

# 调整参数实现读取视频或调用摄像头
cap = cv2.VideoCapture(0)
while True:
    # 读取摄像头中的帧
    ret, frame = cap.read()
    # 调用人脸识别函数
    face_rec(frame)
    c = cv2.waitKey(10)
    # 当键盘按下‘ESC’退出程序
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
