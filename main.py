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
# 定义cap 等于视频输入。0内置摄像头，1外接摄像头
cap = cv2.VideoCapture(0)

############################################################

# 图像读取 target=等待处理的图像

############################################################

# 定义摄像头分辨率为横向1280x720
cap.set(3, 1280)
cap.set(4, 720)

# 按帧读取图像，并存入frame。ret为TRUE代表成功读取。
ret, frame = cap.read()
cv2.imwrite('D:/resource/stack/cap.png', frame)

# 读取例子，以此图为例
# target = cv2.imread('D:/PProject/pic.png', 0)
target = cv2.imread('D:/PProject/crazy.png', 0)

############################################################

# 图像预处理

############################################################

# 把存入的cap.png用灰度图的方式读入内存，命名为img。0
# img = cv2.imread('D:/resource/stack/cap.png', 0)
img = cv2.imread('D:/PProject/pic.png', 0)
img = iPP.PreProcess(target, 1)

