from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import pdb
import cv2
import math
import pylab

# 针对SURF/SIFT结果

# Perform Tf-Idf vectorization
nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

# 为给定数据集构建一个包含K个随机质心的集合
def randCent(dataSet,k):
    m,n = dataSet.shape
    centroids = np.zeros((k,n))
    for i in range(k):
        index = int(np.random.uniform(0,m)) #
        centroids[i,:] = dataSet[index,:]
    return centroids

def kmeans_open(dataSet,k):
    # 随机取k个聚类中心 
    # 对每个点求最近k个聚类中心，
    # 更新每个特征点到聚类中心的距离，
    # 求每个聚类中心的特征点集的平均值
    
    m = np.shape(dataSet)[0]  #行的数目
    # 第一列存样本属于哪一簇
    # 第二列存样本的到簇的中心点的误差
    clusterAssment = np.mat(np.zeros((m,2)))
    clusterChange = True

    # 第1步 初始化centroids
    centroids = randCent(dataSet,k)
    while clusterChange:
        clusterChange = False

        # 遍历所有的样本（行数）
        num = 0
        for i in range(m):
            minDist = 100000.0
            minIndex = -1

            # 遍历所有的质心
            #第2步 找出最近的质心
            for j in range(k):
                # 计算该样本到质心的欧式距离
                distance = distEclud(centroids[j,:],dataSet[i,:])
                if distance < minDist:
                    minDist = distance
                    minIndex = j
            # 第 3 步：更新每一行样本所属的簇
            if clusterAssment[i,0] != minIndex:
                num = num + 1
                clusterChange = True
                clusterAssment[i,:] = minIndex,minDist**2
        #第 4 步：更新质心
        for j in range(k):
            pointsInCluster = dataSet[np.nonzero(clusterAssment[:,0].A == j)[0]]  # 获取簇类所有的点 .A:matrix->array
            centroids[j,:] = np.mean(pointsInCluster,axis=0)   # 对矩阵的列求均值，最后得到一个质心坐标
        print(num)
        
    return clusterAssment.A[:,0], centroids
