############################################################

# 读取库

############################################################
import numpy as np
import cv2 as cv2
import CONSTANT as const


# 定义全局变量（用来读取数字图片）
num = 0
order = 0
############################################################

# 定义地址字符串

############################################################

adr0 = '/'
adr1 = 'D:/resource/stack'
adr4 = '.png'

############################################################

# 定义数组，存放图片数据

############################################################
# 存放训练集
TRAIN = np.ones((300, 2400)).astype(np.float32)
# 存放测试集
TEST = np.ones((10, 2400)).astype(np.float32)
# 存放被识别图片
TARGET = np.ones((3, 2400)).astype(np.float32)
# 数组第一维的代号
f1 = 0
f2 = 0
# 每个数字文件夹的前30个数储存在了cellsa中作为训练集
for num in range(10):
    for order in range(30):
        adr2 = str(num)
        adr3 = str(order)
        adr = adr1 + adr0 + adr2 + adr0 + adr3 + adr4
        img = cv2.imread(adr,0)
        TRAIN[f1] = img.reshape(-1, 2400)
        f1 += 1
# 每个数字文件夹的第31个数储存在了cells中作为测试集
for num in range(10):
    for order in range(30,31):
        adr2 = str(num)
        adr3 = str(order)
        adr = adr1 + adr0 + adr2 + adr0 + adr3 + adr4
        img = cv2.imread(adr, 0)
        TEST[f2] = img.reshape(-1, 2400)
        f2 += 1
# 用K来对图片进行标记
k = np.arange(10)
train_labels = np.repeat(k,30)[:,np.newaxis]
test_labels = np.repeat(k,1)[:,np.newaxis]
# 训练
knn = cv2.ml.KNearest_create()
knn.train(TRAIN, cv2.ml.ROW_SAMPLE, train_labels)
# 测试
ret,result,neighbours,dist = knn.findNearest(TEST, k=5)
# 输出正确率
matches = result==test_labels
correct = np.count_nonzero(matches)
accuracy = correct*100.0/result.size
# print( accuracy )
# 存储数据
np.savez('D:/knn_data.npz', train=TRAIN, train_labels=train_labels)
# 读取数据
with np.load('D:/knn_data.npz') as data:
    # print( data.files )
    train = data['train']
    train_labels = data['train_labels']


def ocr():
    # 读取需要被检测的图像
    f3 = 0
    for num in range(7, 10):
        adr = adr1 + adr0 + str(num) + adr4
        img = cv2.imread(adr, 0)
        TARGET[f3] = img.reshape(-1, 2400)
        f3 += 1

    _, result1, _, _ = knn.findNearest(TARGET, k=5)

    # print(result1)
    return result1