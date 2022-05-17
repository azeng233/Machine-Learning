# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from PIL import Image
import cv2 as cv

# 读入图片并扁平化
original_img = np.array(Image.open('tree.jpg'), dtype=np.float64) / 255
original_dimensions = tuple(original_img.shape)
width, height, depth = tuple(original_img.shape)
image_flattened = np.reshape(original_img, (width * height, depth))

# kmeans参数 https://blog.csdn.net/xiaoQL520/article/details/78269539
# 使用kmeans从1000个随机选取的颜色样本中创建64个聚类，每个聚类都将称为压缩调色盘中的一个颜色
image_array_sample = shuffle(image_flattened, random_state=0)[:1000]

estimator = KMeans(algorithm='auto',  # k-means算法的种类====>默认值=‘auto’
                   copy_x=True,  # 是否对输入数据继续copy 操作====> 布尔型，默认值=True
                   init='k-means++',  # 初始化质心的方法====>默认为'k-means++'
                   max_iter=300,  # 算法每次迭代的最大次数====>整型，默认值=300
                   n_clusters=64,  # 分成的簇数（要生成的质心数）=====>整型，[可选]，默认值=8；
                   random_state=0,  # 用于初始化质心的生成器（generator），和初始化中心有关
                   tol=0.0001,  # 与inertia结合来确定收敛条件====> float型，默认值= 1e-4
                   verbose=0)  # 是否输出详细信息====>类型：整型，默认值=0

estimator.fit(image_array_sample)
# KMeans(algorithm='auto',copy_x= True, init='k-means++',max_iter=300,n_cluster=64,
# n_jobs=1, precompute_distances = 'auto',random_state=0, tol = 0.0001, verbose= 0)

# 为原始图像中每个像素分配聚类
cluster_assignments = estimator.predict(image_flattened)

# 压缩调色盘和聚类分配来创建压缩图片
compressed_palette = estimator.cluster_centers_
compressed_img = np.zeros((width, height, compressed_palette.shape[1]))
label_idx = 0

for i in range(width):
    for j in range(height):
        compressed_img[i][j] = compressed_palette[cluster_assignments[label_idx]]
        label_idx += 1
plt.subplot(211)
plt.title('original Image', fontsize=20)
plt.imshow(original_img)
plt.axis('off')
plt.subplot(212)
plt.title('compressed Image', fontsize=20)
plt.imshow(compressed_img)
plt.axis('off')
plt.show()
