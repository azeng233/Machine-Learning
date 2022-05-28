# -*- coding: utf-8 -*-
"""
Created on Thu May 19 16:13:02 2022

@author: zc
"""

# 导入相关库
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


# 定义sign符号函数
def sign(x, w, b):
    """
    输入：
    x:输入实例
    y:权重系数
    b:偏置系数
    输出:符号函数值
    """
    return np.dot(x, w) + b


# 定义参数初始化函数
def initialize_parameters(dim):
    """
    输入:
    dim:输入数据维度
    输出:
    w:初始化后的权重系数
    b:初始化后的偏置参数
    """
    w = np.zeros(dim, dtype=np.float32)
    b = 0.0
    return w, b


# 定义感知机训练函数
def train(X_train, y_train, learning_rate):
    """
    输入:X_train:训练输入; y_train:训练标签;learning_rate:学习率
    输出:params:训练得到的参数
    """
    w, b = initialize_parameters(X_train.shape[1])  # 参数初始化
    is_wrong = False  # 初始化误分类状态
    while not is_wrong:  # 当存在误分类点时
        wrong_count = 0  # 初始化误分类点的计数
        for i in range(len(X_train)):  # 遍历训练数据
            X = X_train[i]
            y = y_train[i]
            if y * sign(X, w, b) <= 0:  # 如果存在误分类点
                # 更新参数
                w = w + learning_rate * np.dot(y, X)
                b = b + learning_rate * y
                wrong_count += 1  # 误分类点+1
        # 直到没有误分类点
        if wrong_count == 0:
            is_wrong = True
            print('There is no missclassification!')

        # 保存更新后的参数
        params = {'w': w, 'b': b}
    return params


# ---------------------------------------------------------------

# 生成测试数据
# 导入鸢尾花的数据集
iris = load_iris()
# 转化为Pandas数据框
df = pd.DataFrame(iris.data, columns=iris.feature_names)
# 数据标签
df['label'] = iris.target
# 变量重命名
df.columns = ['sepal length',  # 花萼长度
              'sepal width',  # 花萼宽度
              'petal length',  # 花瓣长度
              'petal width',  # 花瓣宽度
              'label']  # 花的种类(标签)
# 取前100行数据
data = np.array(df.iloc[:100, [0, 1, -1]])
# 定义训练输入和输出
X, y = data[:, :-1], data[:, -1]
y = np.array([1 if i == 1 else -1 for i in y])
# 输出训练集大小
print(X.shape, y.shape)

# ---------------------------------------------------------------

# 感知机训练
params = train(X, y, 0.01)
# 输出训练好的模型
print(params)

# 绘制感知机的线性分隔超平面
# 输入实例取反
x_points = np.linspace(4, 7, 10)
# 线性分隔超平面
y_hat = -(params['w'][0] * x_points + params['b']) / params['w'][1]
# 绘制线性分隔超平面
plt.plot(x_points, y_hat)

# 绘制二分类散点图
plt.scatter(data[:50, 0], data[:50, 1], color='red', label='0')
plt.scatter(data[50:100, 0], data[50:100, 1], color='green', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()
