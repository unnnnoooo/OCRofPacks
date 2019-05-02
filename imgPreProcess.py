############################################################

# 读取需要的库

############################################################
# 图像处理库，包含部分机器学习的功能
import cv2

# 矩阵处理库，便于矩阵处理
import numpy as np

# 常数
import CONSTANT as const
# 定义几个卷积核大小
kernel1 = np.ones((1, 1), np.uint8)
kernel3 = np.ones((3, 3), np.uint8)
kernel5 = np.ones((5, 5), np.uint8)
kernel9 = np.ones((9, 9), np.uint8)


############################################################

# 定义预处理函数。

############################################################

# 定义PreProcess，变量为一个灰度图像。


def PreProcess(img, blocksize, csize ,c=1):
    try:

        # 调用二值化函数
        # 读入原始图像，输出一个经过灰度，阈值处理的图像
        # cv2.imwrite("D:/resource/stack/origin.png", img)
        img = BeBinary(img,blocksize,csize)
        backup = img
        # 从图片中找出处理区域
        # 把img读入，输出在img上划线的图
        rotated, Angle, xx, yy, dx, dy = Roi(img)
        cv2.imwrite('D:/resource/stack/rotate.png',rotated)
        # 裁剪图像，留下目标区域
        img = rotated[yy + 10:(yy + dy - 10), xx + 40:(xx + dx - 40)]
        cv2.imwrite("D:/resource/stack/backup.png", img)

        which = hengqie(img)
        r, _ = img.shape
        img = img[which[0]-5:which[-1]+5, :]

        left, right = bread(img)


        for i in range(10):
            adr3 = str(i)
            adr = const.ADDRESS_0 + const.ADDRESS_1 + adr3 + const.ADDRESS_2
            pic = img[0:r, left[i]:right[i]]
            r, _ = pic.shape
            a = bready(pic)
            #横切，去除白边
            if len(a) == 1:
                if a[0]<(r/2):
                    pic = pic[a[0]:r, :]
                if a[0]>(r/2):
                    pic = pic[0:a[0], :]
            elif len(a) == 2:
                pic = pic[a[0]:a[1], :]

            pic = cv2.resize(pic, (40, 60))
            cv2.imwrite(adr,pic)
        return True
    except Exception as e:
        return False

############################################################

# 二值化处理

############################################################


def BeBinary(img, blocksize, csize):
    # 进行自适应阈值处理，可以不考虑亮度情况
    # 用大的处理区域加大的平均值
    # img = cv2.resize(img, (720, 1280))
    threshold = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blocksize, csize)
    cv2.imshow('pic', threshold)
    # 进行开运算，把部分过细的地方消除，并让主题部分不发生太大变化(开运算指的是先腐蚀再膨胀)
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel5)
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel3)
    # 进行4次中值滤波,目的是清除个别离散噪点
    for a in range(1):
        blured = cv2.medianBlur(threshold, 5)

    # 清除上部用过的threshold变量，减少内存使用。
    del (threshold)

    # 返回滤波后图像
    return (blured)


############################################################

# 查找图片旋转角度以及切割范围。切割可在旋转后进行

############################################################


def Roi(pic):
    # 备份pic
    picback = pic

    rows, cols = pic.shape

    r, l = pic.shape
    # 用canny函数绘制边框
    edges = cv2.Canny(pic, 100, 20)

    #设定阈值为150 备用
    yuzhi = 150
    # e3 = cv2.getTickCount()
    lines = cv2.HoughLines(edges, 1, np.pi / 180, yuzhi)
    # 用来调整直线数量
    for i in range(3):
        long = len(lines)
        if long >= 20:
            yuzhi += 5
            lines = cv2.HoughLines(edges, 1, np.pi / 180, yuzhi)#思考一种斜率恒定的渐变方式
        elif long <= 10:
            yuzhi -= 5
            lines = cv2.HoughLines(edges, 1, np.pi / 180, yuzhi)
        else:
            break
    # 将所有rho化为正数
    for i in range(len(lines)):
        rho,theta = lines[i][0]
        if rho < 0:
            lines[i][0][0] = -rho
            lines[i][0][1] = theta - np.pi
    # 因为霍夫变换会生成许多条直线，因此，需要过滤部分不需要的。
    # 这里采用的是最外围的线来构成四边形。规则可以替换，暂时如此。
    # SmallA是A方向的距离原点最近的直线，SmallB是B方向距离原点最近的直线。
    # 理想状态下应该A方向与B方向有pi/2的夹角。
    Small = SmallA = SmallB = 2000
    BigA = BigB = 0
    # 这一层循环是总的循环，用来遍历
    for i in range(len(lines)):
        rho, theta = lines[i][0]
        if abs(rho) <= Small:
            # 找到距离原点最近的线，并定义此线角度为AngleA
            Small = rho
            # flag = i
            AngleA = theta
    # AngleA = AngleA % (np.pi / 2)
    while (AngleA > (np.pi/4)):
        AngleA -= (np.pi/2)
        #遗留问题：怎样处理这个pi #solved

    #     AngleB = AngleA - (np.pi / 2)
    # else:
    AngleB = AngleA + (np.pi / 2)
    for i in range(len(lines)):
        rho, theta = lines[i][0]
        # 如果AngleA接近当前正在遍历的线的角度
        if (
                abs(AngleA - theta) <= (np.pi / 6) or (
                abs(AngleA - (theta + np.pi)) <= (np.pi / 6)) or (
                abs(AngleA - (theta - np.pi)) <= (np.pi / 6)) or (
                abs(AngleA - (theta + 2 * np.pi)) <= np.pi / 6) or (
                abs(AngleA - (theta - 2 * np.pi)) <= np.pi / 6)):

            # 以下6行目的选出A方向最近和最远的参数方程的参数
            if rho <= SmallA:
                SmallA = rho
                # thetaAs = theta
            if rho >= BigA:
                BigA = rho
                # thetaAb = theta
        # for rho,theta in lines[i]:
    for i in range(len(lines)):
        rho, theta = lines[i][0]
        # 如果AngleB接近当前正在遍历的线的角度，np.pi/6是近似范围
        if (
                abs(AngleB - theta) <= (np.pi / 6) or (
                abs(AngleB - (theta + np.pi)) <= (np.pi / 6)) or (
                abs(AngleB - (theta - np.pi)) <= (np.pi / 6)) or (
                abs(AngleB - (theta + 2 * np.pi)) <= np.pi / 6) or (
                abs(AngleB - (theta - 2 * np.pi)) <= np.pi / 6)):

            # 以下6行目的选出B方向最近和最远的参数方程的参数
            if rho <= SmallB:
                SmallB = rho
                # thetaBs = theta
            if rho >= BigB:
                BigB = rho
                # thetaBb = theta
    # 画出符合条件的线
    a1 = np.cos(AngleA)
    b1 = np.sin(AngleA)
    if SmallA != 2000:
        x0 = a1 * SmallA
        y0 = b1 * SmallA
        x1 = int(x0 + 2000 * (-b1))
        y1 = int(y0 + 2000 * (a1))
        x2 = int(x0 - 2000 * (-b1))
        y2 = int(y0 - 2000 * (a1))
        cv2.line(picback, (x1, y1), (x2, y2), (0, 0, 255), 10)
    if BigA != 0:
        x0 = a1 * BigA
        y0 = b1 * BigA
        x3 = int(x0 + 2000 * (-b1))
        y3 = int(y0 + 2000 * (a1))
        x4 = int(x0 - 2000 * (-b1))
        y4 = int(y0 - 2000 * (a1))
        cv2.line(picback, (x3, y3), (x4, y4), (0, 0, 255), 10)
    a2 = np.cos(AngleB)
    b2 = np.sin(AngleB)
    if SmallB != 0:
        x0 = a2 * SmallB
        y0 = b2 * SmallB
        x5 = int(x0 + 2000 * (-b2))
        y5 = int(y0 + 2000 * (a2))
        x6 = int(x0 - 2000 * (-b2))
        y6 = int(y0 - 2000 * (a2))
        cv2.line(picback, (x5, y5), (x6, y6), (0, 0, 255), 10)
    if BigB != 0:
        x0 = a2 * BigB
        y0 = b2 * BigB
        x7 = int(x0 + 2000 * (-b2))
        y7 = int(y0 + 2000 * (a2))
        x8 = int(x0 - 2000 * (-b2))
        y8 = int(y0 - 2000 * (a2))
        cv2.line(picback, (x7, y7), (x8, y8), (0, 0, 255), 10)
    # dx是向的剪切宽度，dy是B向剪切长度
    dx = abs(int(BigA - SmallA))
    dy = abs(int(BigB - SmallB))
    # 以下步骤计算左上角交点点，利用交点以及dx和dy可以求出需要分割的区域
    if (x2 - x1) == 0:
        k1 = None
        b1 = 0
    else:
        # 计算k1，b1,由于点均为整数，需要进行浮点数转化
        k1 = (y2 - y1) * 1.0 / (x2 - x1)
        b1 = y1 * 1.0 - x1 * k1 * 1.0
    if k1 == None:
        # 如果斜率不存在，那么目标点横坐标为x2
        x = x2
    # L2直线斜率不存在操作.L2是指B方向的距离原点（左上角）最近直线
    if (x6 - x5) == 0:
        k2 = None
        b2 = 0
    else:
        # 计算k2，b2,由于点均为整数，需要进行浮点数转化
        k2 = (y5 - y6) * 1.0 / (x5 - x6)
        b2 = y6 * 1.0 - x6 * k2 * 1.0
    if k2 == None:
        # 如果斜率不存在，那么目标点横坐标为x6
        x = x6

    # 如果k1和k2都不存在，求出目标点x，y
    if (k1 != None) and (k2 != None):
        x = (b2 - b1) * 1.0 / (k1 - k2)
        y = k1 * x * 1.0 + b1 * 1.0
        # 如果只有k1不存在，因为上面已经求了x，这里只要求y
    if k1 == None:
        y = k2 * x * 1.0 + b2 * 1.0
        # 如果只有k2不存在，因为上面已经求了x，这里只要求y
    elif k2 == None:
        y = k1 * x * 1.0 + b1 * 1.0
    x = - int(l / 2 - x)
    y = int(r / 2 - y)

    a = np.sin(AngleA)
    b = np.cos(AngleA)

    # 矩阵变换
    xx = b * x - a * y
    yy = a * x + b * y
    xx = - int(l / 2 - xx)
    yy = int(r / 2 - yy)
    # 旋转参数
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), (AngleA / np.pi) * 180, 1)
    # 第三个参数是输出图像的尺寸中心
    picback = cv2.warpAffine(picback, M, (cols, rows))
    return picback, AngleA, xx, yy, dx, dy

############################################################

# 纵向剪切

############################################################


def hengqie(img):
    # r,l是图像像素的行和列
    r, l = img.shape
    which = []
    # 扫描每一行
    for i in range(r):
        # 再扫描每一行的每一个像素点
        # 开始扫描！定义用到的变量
        flagold = flagnew = lenghofthesamecolor = lenghofthesamecolorold = numberoftheblackpixel = 0
        for j in range(l):
            # 如果这个像素点[i][j]是黑色
            if img[i][j] == 0:
                # 那么色彩标志置1，并且黑色像素数目自加1
                flagnew = 1
                numberoftheblackpixel += 1
            else:
                # 否则色彩标志置0.
                flagnew = 0
            # 如果新标志等于老标志，意味着颜色没有改变
            if flagnew == flagold:
                # c+=1
                # 那么同色长度自加1
                lenghofthesamecolor += 1
                if lenghofthesamecolor >= lenghofthesamecolorold:
                    lenghofthesamecolorold = lenghofthesamecolor
                flagold = flagnew
            else:
                # 否则的话，在颜色改变的的前提下，
                # 如果新的同色长度大于等于老的同色长度，
                # 用新的同色长度代替老的同色长度，也就是记录最长同色长度
                if lenghofthesamecolor >= lenghofthesamecolorold:
                    lenghofthesamecolorold = lenghofthesamecolor
                    lenghofthesamecolor = 0
                # 如果新的同色长度甚至不如老的同色长度，那么久把最长同色长度归零
                else:
                    lenghofthesamecolor = 0
                flagold = flagnew

        # 扫描完这一行以后，对同色长度，以及黑色点数目进行讨论。
        # 如果同色最大长度小于等于100，并且黑色像素数目大于等于90，那么在which list中记录当前的行数，然后进入下一行的分析
        if (lenghofthesamecolorold <= const.LOSO) and (numberoftheblackpixel >= const.NOBP):
            which.append(i)
    for i in range(len(which)-1):
        if abs(which[i]-which[i+1])>=10:
            which = which[0:i]
            break

    return which

############################################################

# 纵向切割

############################################################
def bread(img):
    # r,l是图像像素的行和列
    r, l = img.shape
    # 声明list a。该list用来存储有黑色像素的列的位置。
    # 举例：如果一幅图片有30行和
    a = []
    # 扫描每一列
    for i in range(l):
        # 再扫描每一列的每一个像素点
        # 开始扫描！定义用到的变量：黑色像素的数目
        for j in range(r):
            # 如果j行i列的像素点是黑色，那么numberoftheblackpixel自加1.
            # 然而其实等于一也可以，因为在后续我加了个break，如果该行检测
            # 到黑色像素直接记录该行列数并退出此轮循环即列扫描
            if img[j][i] == 0:
                a.append(i)
                break
    # 声明两个list。用来存各个数字的左边界以及右边界。
    leftBoundary = []
    rightBoundary = []
    # 用来暂存左边界以及右边界
    left = right = 0
    # flag是一个变量，用来存储当前状态，而flagn则用来存储上一轮的状态。
    # 这个设计的灵感是汇编语言中的标志寄存器。虽然说别人可能也会想到，但是在此完全是原创的内容。
    flagn = flag = 0
    # 遍历
    # 不是len(a)而是len(a)-1的原因是，要用到a[n+1]
    for n in range(len(a)-1):
        if a[n]+1 == a[n+1]:
            # 如果连续，那么flag置1
            flag = 1
            # 如果
            # 这一行和上一行颜色一样
            # 并且
            # 上一行和上上行颜色不一样
            # 那么left等于a数组中当前数字代表的列数
            # 那么right等于a数组中当前数字下一个数字代表的列数
            if flag == 1 and flagn ==0:
                left = a[n]
                right = a[n+1]
            # 如果
            # 这一行和上一行颜色一样
            # 并且
            # 上一行和上上行颜色也一样
            # 那么right等于a数组中当前数字代表的列数
            elif flag == 1 and flagn == 1:
                right = a[n+1]
        #如果这个数和下一个数不连续
        elif a[n]+1 != a[n+1]:
            #标志置0
            #并且把左标志加入lB list
            # 把左标志加入lR list
            flag = 0
            leftBoundary.append(left)
            rightBoundary.append(right+1)
        # 更新标志位
        flagn = flag
    # 当然不能把最后一轮的加上啦
    leftBoundary.append(left)
    rightBoundary.append(right+1)
    return leftBoundary, rightBoundary

############################################################

# 去除白边

############################################################
def bready(img):
    # r,l是图像像素的行和列
    r, l = img.shape
    # 同样是标识符
    flag = flagnew = 0
    # 定义list a用来存储行数
    a = []
    # 横竖遍历
    for i in range(r):
        for j in range(l):
            # 如果该该行存在黑色像素，那么flag置1，跳出循环。
            if img[i][j] == 0:

                flag = 1
                break
            # 否则flag置0
            else:
                flag = 0
        # 开始讨论
        # 如果上一行没有黑色and这一行有黑色
        # 那么把这一行加入list a
        if flag == 1 and flagnew == 0:
            a.append(i)
        # 如果上一行有黑色 而这一行没有黑色
        # 那么把上一行加入list a然后跳出循环
        if flag == 0 and flagnew == 1:
            a.append(i)
            break
        # 更新标识符
        flagnew = flag

    return a