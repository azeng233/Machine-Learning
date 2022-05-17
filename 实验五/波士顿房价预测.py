# -*- coding: utf-8 -*-
import  pandas as pd
import numpy as np
from sklearn.datasets import load_boston  # 导入数据集
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

"""
第一步：首先认识波士顿数据集，分析查看数据集样本总数，特征变量总数。
第二步：然后画出波士顿数据集所有特征变量的散点图，并分析特征变量与房价的影响关系。
"""
boston = load_boston()
print(boston.feature_names)  # 查看boston数据集特征变量
print(boston.data.shape)  # 分析数据集样本总数，特征变量总数
v_bos = pd.DataFrame(boston.data)  # 查看波士顿数据集前5条数据，查看这13个变量数据情况
print(v_bos.head(5))
x = boston['data']  # 导入特征变量
y = boston['target']  # 导入目标变量房价
student = input('房价特征信息图--0；各个特征信息图--1: ')  # 输入0代表查看影响房价特征信息图，输入1代表查看各个特征信息图
if str.isdigit(student):
    b = int(student)
    if (b <= 1):
        print('开始画图咯...', end='\t')
        if (b == 0):
            plt.figure(figsize=(20, 15))
            y_major_locator = MultipleLocator(5)  # 把y轴的刻度间隔设置为10，并存在变量里
            ax = plt.gca()  # ax为两条坐标轴的实例
            ax.yaxis.set_major_locator(y_major_locator)  # 把y轴的主刻度设置为5的倍数
            plt.ylim(0, 51)
            plt.grid()
            for i in range(len(y)):
                plt.scatter(i, y[i], s=20)
            plt.show()
        else:
            name = boston['feature_names']
            for i in range(13):
                plt.figure(figsize=(10, 7))
                plt.grid()
                plt.scatter(x[:, i], y, s=5)  # 横纵坐标和点的大小
                plt.title(name[i])
                print(name[i], np.corrcoef(x[:i]), y)
            plt.show()
    else:
        print('同学请选择0或者1')

else:
    print('同学请选择0或者1')

'''
训练模型
'''
import numpy as np
from skimage.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.linear_model import LinearRegression  # 导入线性模型
from sklearn.datasets import load_boston  # 导入数据集
from sklearn.metrics import r2_score    # 使用r2_score对模型评估
from sklearn.model_selection import train_test_split  # 导入数据集划分模块
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2

boston = load_boston()
x = boston['data']  # 影响房价的特征信息数据
y = boston['target']  # 房价
name = boston['feature_names']

# 数据处理
unsF = []  # 次要特征下标
for i in range(len(name)):
    if name[i] == 'RM' or name[i] == 'PTRATIO' or name[i] == 'LSTAT' or name[i] == 'AGE' or name[i] == 'NOX' or name[i] == 'DIS' or name[i] == 'INDUS':
        continue
    unsF.append(i)
x = np.delete(x, unsF, axis=1)  # 删除次要特征

unsT = []  # 房价异常值下标
for i in range(len(y)):
    if y[i] > 50:  # 对房价影响较小的特征信息进行剔除
        unsT.append(i)
x = np.delete(x, unsT, axis=0)  # 删除样本异常值数据
y = np.delete(y, unsT, axis=0)  # 删除异常房价

# 将数据进行拆分，一份用于训练，一份用于测试和验证
# 测试集大小为30%,防止过拟合
# 这里的random_state就是为了保证程序每次运行都分割一样的训练集和测试集。
# 否则，同样的算法模型在不同的训练集和测试集上的效果不一样。
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# 线性回归模型
lf = LinearRegression()
lf.fit(x_train, y_train)  # 训练数据,学习模型参数
y_predict = lf.predict(x_test)  # 预测

# 与验证值作比较
error = mean_squared_error(y_test, y_predict).round(5)  # 平方差
score = r2_score(y_test, y_predict).round(5)  # 相关系数

# 绘制真实值和预测值的对比图
fig = plt.figure(figsize=(13, 7))
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['font.sans-serif'] = "SimHei"
plt.rcParams['axes.unicode_minus'] = False  # 绘图
plt.plot(range(y_test.shape[0]), y_test, color='red', linewidth=1, linestyle='-')
plt.plot(range(y_test.shape[0]), y_predict, color='blue', linewidth=1, linestyle='dashdot')
plt.legend(['真实值', '预测值'])
plt.title("学号", fontsize=20)
error = "标准差d=" + str(error)+"\n"+"相关指数R^2="+str(score)
plt.xlabel(error, size=18, color="black")
plt.grid()
plt.show()

plt2.rcParams['font.family'] = "sans-serif"
plt2.rcParams['font.sans-serif'] = "SimHei"
plt2.title('学号', fontsize=24)
xx = np.arange(0, 40)
yy = xx
plt2.xlabel('* truth *', fontsize=14)
plt2.ylabel('* predict *', fontsize=14)
plt2.plot(xx, yy)
plt2.scatter(y_test, y_predict, color='red')
plt2.grid()
plt2.show()


