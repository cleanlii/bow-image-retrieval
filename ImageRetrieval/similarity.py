import cv2
import numpy as np
from numpy import *

# import kmeans

# 欧式距离
def EuclideanDist(A, B, length, vecNum):
    result = 0
    for i in range(vecNum):
        sum = 0
        for index in range(length):
            sum += np.power((A[i][index]-B[i][index]),2)
        sum = np.power(sum, 1/2)
        result += sum / vecNum
    return result

'''
# 余弦相似度
def CosinDist(A, B)
    A = np.mat(A)
    B = np.mat(B)
    num = float(A * B.T)
    denom = np.linalg.norm(A) * np.linalg.norm(B)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim
'''

'''
# 不太地道
def getHogSim(A, B, length, vecNum):
    return EuclideanDist(A, B, length, vecNum)
'''

def KNN(datasetSiftFeat, querySiftFeat):
    kp1, des1 = datasetSiftFeat
    kp2, des2 = querySiftFeat
    # 创建暴力匹配对象
    bf = cv2.BFMatcher()
    # 使用Matcher.knnMatch获得两幅图像的最佳匹配
    matches = bf.knnMatch(des1, des2, k = 2)

    good = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    # 对good中的匹配点的distance进行加权平均
    sum = 0
    for index in range(len(good)):
        sum += good[index][0].distance

    return sum / len(good)