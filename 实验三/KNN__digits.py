import time
from os import listdir

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
"""
函数说明:将32x32的二进制图像转换为1x1024向量
"""


def img2vector(filename):
    # 创建1x1024零向量
    returnVect = np.zeros((1, 1024))
    # 打开文件
    fr = open(filename)
    # 按行读取
    for i in range(32):
        # 读一行数据
        lineStr = fr.readline()
        # 每一行的前32个元素依次添加到returnVect中
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    # 返回转换后的1x1024向量
    return returnVect


"""
函数说明:读取TXT，并以图像形式转换
"""


def txt2image(filename):
    # 创建1x1024零向量
    returnMat = np.zeros((32, 32))
    # 打开文件
    fr = open(filename)
    # 按行读取
    for i in range(32):
        # 读一行数据
        lineStr = fr.readline()
        # 每一行的前32个元素依次添加到returnVect中
        for j in range(32):
            returnMat[i, j] = int(lineStr[j])
    # 返回转换后的1x1024向量
    return returnMat


"""
函数说明:手写数字分类测试
"""


def handwritingClassTest():
    # 训练集的Labels
    hwLabels = []
    # 返回trainingDigits目录下的文件名
    trainingFileList = listdir('trainingDigits')

    # 返回文件夹下文件的个数
    m = len(trainingFileList)
    # 初始化训练的Mat矩阵,训练集
    trainingMat = np.zeros((m, 1024))
    # 从文件名中解析出训练集的类别
    for i in range(m):
        # 获得文件的名字
        fileNameStr = trainingFileList[i]
        # 获得分类的数字
        classNumber = int(fileNameStr.split('_')[0])
        # 将获得的类别添加到hwLabels中
        hwLabels.append(classNumber)
        # 将每一个文件的1x1024数据存储到trainingMat矩阵中
        trainingMat[i, :] = img2vector('trainingDigits/%s' % (fileNameStr))
    # 构建kNN分类器
    neigh = KNN(n_neighbors=3, algorithm='auto')
    # 拟合模型, trainingMat为训练矩阵,hwLabels为对应的标签
    neigh.fit(trainingMat, hwLabels)
    # 返回testDigits目录下的文件列表
    testFileList = listdir('testDigits')
    # 错误检测计数
    errorCount = 0.0
    # 测试数据的数量
    mTest = len(testFileList)
    # 从文件中解析出测试集的类别并进行 分类测试
    showflag = 1  # 只展示第一个分错的
    for i in range(mTest):
        # 获得文件的名字
        fileNameStr = testFileList[i]
        # 获得分类的数字
        classNumber = int(fileNameStr.split('_')[0])
        # 获得测试集的1x1024向量,用于训练
        vectorUnderTest = img2vector('testDigits/%s' % (fileNameStr))
        # 获得预测结果
        classifierResult = neigh.predict(vectorUnderTest)
        print("分类返回结果为%d\t真实结果为%d" % (classifierResult, classNumber))
        if (classifierResult != classNumber):
            errorCount += 1.0
            # 一旦分类错误就显示错误结果，×掉绘图框后继续预测
            imageDir = txt2image('testDigits/%s' % (testFileList[i]))
            print(imageDir)
            plt.imshow(imageDir)
            plt.title('第%s个错误分了错误，testFileList/%s,真实结果为：%s，预测结果为：%s' % (
            showflag, testFileList[i], classNumber, classifierResult))
            plt.show()
            showflag += 1

    print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount / mTest * 100))


"""
函数说明:main函数
"""
# 开始时间
start = time.perf_counter()
# 测试handwritingClassTest()的输出结果
handwritingClassTest()
end = time.perf_counter()
print("运行耗时：%ds" % (end - start))
