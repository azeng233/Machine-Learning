# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

df = pd.read_csv("credit-overdue.csv", header=0)  # 加载数据集
df.head()  # 查看前5行数据

from matplotlib import pyplot as plt

plt.figure(figsize=(10, 6))
map_size = {0: 20, 1: 100}
size = list(map(lambda x: map_size[x], df['overdue']))
plt.scatter(df['debt'], df['income'], s=size, c=df['overdue'], marker='v')


# 定义Sigmoid函数
def sigmoid(z):
    sigmoid = 1 / (1 + np.exp(-z))
    return sigmoid


# 定义对数损失函数
def loss(h, y):
    loss = (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    return loss


# 定义梯度下降函数
def gradient(X, h, y):
    gradient = np.dot(X.T, (h - y)) / y.shape[0]
    return gradient


# 逻辑回归过程
def Logistic_Regression(x, y, lr, num_iter):
    intercept = np.ones((x.shape[0], 1))  # 初始化截距为 1
    x = np.concatenate((intercept, x), axis=1)
    w = np.zeros(x.shape[1])  # 初始化参数为 0

    for i in range(num_iter):  # 梯度下降迭代
        z = np.dot(x, w)  # 线性函数
        h = sigmoid(z)  # sigmoid 函数
        g = gradient(x, h, y)  # 计算梯度
        w -= lr * g  # 通过学习率 lr 计算步长并执行梯度下降
        z = np.dot(x, w)  # 更新参数到原线性函数中
        h = sigmoid(z)  # 计算 sigmoid 函数值
        l = loss(h, y)  # 计算损失函数值
    return l, w  # 返回迭代后的梯度和参数


x = df[['debt', 'income']].values
y = df['overdue'].values
lr = 0.001  # 学习率
num_iter = 10000  # 迭代次数
# 模型训练
L = Logistic_Regression(x, y, lr, num_iter)
print("逻辑回归模型为：\n", L)

plt.figure(figsize=(10, 6))
map_size = {0: 20, 1: 100}
size = list(map(lambda x: map_size[x], df['overdue']))
plt.scatter(df['debt'], df['income'], s=size, c=df['overdue'], marker='v')

x1_min, x1_max = df['debt'].min(), df['debt'].max(),
x2_min, x2_max = df['income'].min(), df['income'].max(),

xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
grid = np.c_[xx1.ravel(), xx2.ravel()]

probs = (np.dot(grid, np.array([L[1][1:3]]).T) + L[1][0]).reshape(xx1.shape)
plt.contour(xx1, xx2, probs, levels=[0], linewidths=1, colors='red');


def Logistic_Regression(x, y, lr, num_iter):
    intercept = np.ones((x.shape[0], 1))  # 初始化截距为 1
    x = np.concatenate((intercept, x), axis=1)
    w = np.zeros(x.shape[1])  # 初始化参数为 1

    l_list = []  # 保存损失函数值
    for i in range(num_iter):  # 梯度下降迭代
        z = np.dot(x, w)  # 线性函数
        h = sigmoid(z)  # sigmoid 函数

        g = gradient(x, h, y)  # 计算梯度
        w -= lr * g  # 通过学习率 lr 计算步长并执行梯度下降

        z = np.dot(x, w)  # 更新参数到原线性函数中
        h = sigmoid(z)  # 计算 sigmoid 函数值

        l = loss(h, y)  # 计算损失函数值
        l_list.append(l)
    return l_list


lr = 0.01  # 学习率
num_iter = 30000  # 迭代次数
l_y = Logistic_Regression(x, y, lr, num_iter)  # 训练

# 绘图
plt.figure(figsize=(10, 6))
plt.plot([i for i in range(len(l_y))], l_y)
plt.xlabel("Number of iterations")
plt.ylabel("Loss function")
plt.show()