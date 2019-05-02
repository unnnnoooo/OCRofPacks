############################################################

# 预读取库函数,定义全局变量等等

############################################################

# 图像处理库，包含部分机器学习的功能
import cv2

# 图像处理库，备用其中部分功能
import skimage.io as io

# 矩阵处理库，便于矩阵处理
import numpy as np

# 自定义图像处理模块
import imgPreProcess as iPP

# 自定义机器学习模块
import MachineLearning as ML

import wx

# 常量模块
import CONSTANT as const

# 定义cap 等于视频输入。0内置摄像头，1外接摄像头
cap = cv2.VideoCapture(0)

############################################################

# 图像读取 target=等待处理的图像

############################################################

# 定义摄像头分辨率为横向1280x720
cap.set(3, const.HEIGHT)
cap.set(4, const.WIDTH)
# 调节自适应阈值部分
def nothing(x):
    pass
img=np.zeros((200,200,3),np.uint8)
cv2.namedWindow('image')
cv2.createTrackbar('BLOCKSIZE','image',15,255,nothing)
cv2.createTrackbar('C','image',30,255,nothing)


# 读取例子，以此图为例
target = cv2.imread('D:/PProject/pic.png', 0)
# target = cv2.imread('D:/resource/stack/cap.png', 0)

############################################################

# 图像预处理

############################################################

# 把存入的cap.png用灰度图的方式读入内存，命名为img。0
resultlast = ''
app = wx.App()
frame0 = wx.Frame(None, title="识别结果", pos=(1000, 200), size=(500, 400))
content_text = wx.TextCtrl(frame0, pos=(5, 39), size=(475, 300), style=wx.TE_MULTILINE)
content_text.SetValue('等待检测')
frame0.Show()
#print(resultlast)
while(1):
    ret = False


    while(not ret):
        # 按帧读取图像，并存入frame。ret为TRUE代表成功读取。

        ret, frame = cap.read()
        cv2.imwrite(const.CAMERA_PICTURE_ADDRESS, frame)
        k = cv2.waitKey(5)

        img = cv2.imread(const.CAMERA_PICTURE_ADDRESS, 0)



        img = np.rot90(img)
        # cv2.imshow("origin",img)
        # 参数传入函数
        BLCSZ = 2 * cv2.getTrackbarPos('BLOCKSIZE', 'image') + 3
        Csize = cv2.getTrackbarPos('C', 'image')
        #img1 = iPP.BeBinary(target)
        if k == -1:
            #img = cv2.imread('D:/new.png', 0)

            #img = cv2.imread('D:/PProject/pic.png', 0)
            # e1 = cv2.getTickCount()
            ret = iPP.PreProcess(img,BLCSZ,Csize,1)
            # e2 = cv2.getTickCount()
            # t = (e2 - e1) / cv2.getTickFrequency()
            # print(t)
        if ret:
            result = ML.ocr()
            resultnow=''
            resultnow += str(int(result[0][0]))
            resultnow += str(int(result[1][0]))
            resultnow += str(int(result[2][0]))
            if (resultnow == resultlast):
                content_text.SetValue(resultnow)
                frame0.Show()

    resultlast = resultnow
