# TF Term Frequency
# IDF Inverse Document Frequency
# 注：命名格式部分不一致是由于部分代码参考自相关博客，而开始的底层命名源自课上的伪代码，故未作过多修改

import cv2
import os
import numpy as np
from numpy import *
import re

import ImgInfo #引入一个图片信息类
import file #多文件读取操作
import hog # HoG特征提取
import sift # SIFT特征提取
import similarity # 近似度计算
# import kmeans # kmeans聚类

queryImgAddr = 'd:/test/1.jpg'
datasetAddr = 'd:/image/'

# feature paraments 图像参数
featType = 'HOG'  # SIFT, HOG, LBP, Haar-like

# embeding paraments 特征参数
clusterAlgo = 'k-means'
clusterNum = 3

# retrieval paraments
topN = 10  # 返回照片数量

queryImg = cv2.imread(queryImgAddr)

# 读取数据库中图片的地址，放入数组
datasetImg = file.readImgs(datasetAddr);

print('数据库容量：' + str(len(datasetImg)) + ' 张')

################ 计算特征值 ###############
datasetImgFeatsHoG = []
datasetImgFeatsSIFT = []

# 提取待检索图片特征值
print('待检索图片特征提取中...')
queryImgFeatHoG = hog.get_HoG_feature(queryImgAddr)
queryImgFeatSIFT = sift.get_SIFT_feature(queryImgAddr)
print('待检索图片特征提取完毕！')
################ 计算特征值 ###############

################ 计算相似度 ###############
def similarityImg(datasetImg, queryImgFeatHoG, queryImgFeatSIFT = -1):
  simArrayHoG = []
  simArraySIFT = []

  temp = 1
  
  print ('数据库图片特征提取中...')
  print ('图片相似度计算中...')
  for datasetImgAddr in datasetImg:

    print ('正在处理:', str(temp) + '/' + str(len(datasetImg)))

    # 遍历数据库图片，提取特征值
    # 一般来说这一步可以后台进行预处理，即提前保存为txt文件 

    # 提取HoG特征
    datasetImgFeatHoG = hog.get_HoG_feature(datasetImgAddr)
    datasetImgFeatsHoG.append(datasetImgFeatHoG)
    # 计算HoG相似度（欧氏距离）
    simHoG = similarity.EuclideanDist(queryImgFeatHoG, datasetImgFeatHoG, len(queryImgFeatHoG[0]), len(queryImgFeatHoG))
    # print('- HoG相似度为：%f' %(simHoG))
    simArrayHoG.append(simHoG)

    # 提取SIFT特征
    datasetImgFeatSIFT = sift.get_SIFT_feature(datasetImgAddr)
    datasetImgFeatsSIFT.append(datasetImgFeatSIFT)
    # 计算SIFT相似度
    if (queryImgFeatSIFT != -1):
      simSIFT = similarity.KNN(queryImgFeatSIFT, datasetImgFeatSIFT)
    else:
      simSIFT = 0
    # print('- SIFT相似度为：%f' %(simSIFT))
    simArraySIFT.append(simSIFT)
    temp += 1

    # print (simArrayHoG)
    # print (simArraySIFT)
  print ('数据库图片特征提取完毕！')
  print ('图片相似度计算完毕！')

  # 以类的形式存储图片的路径和特征值
  datasetCombine = []

  # 重排序：对两种相似性特征结果进行线性组合
  for index in range(len(datasetImg)):
    datasetCombine.append(ImgInfo.ImgInfo(datasetImg[index], simArrayHoG[index], simArraySIFT[index], datasetImgFeatsHoG[index]))

  # 遍历数据对象集合，并以特征值升序排序
  datasetCombine = sorted(datasetCombine)

  return datasetCombine
################ 计算相似度 ###############

################ 查询结果 ###############
def showImg(datasetCombine, topN):
  for index in range(topN):
    title = 'Retrieval Image ' + '#' + str(index+1)
    img = cv2.imread(datasetCombine[index].img_src)
    cv2.imshow(title, img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
################ 查询查询 ###############

################ 首次普通查询 ###############
def queryRetrieval(datasetCombine, topN):
  showImg(datasetCombine, topN)
################ 首次普通查询 ###############

################ 扩展查询 Query Expasion ###############
# 这里对HoG返回的前TopN个结果+查询样本求和取平均，然后再进行一次查询
def averageFeat(datasetCombine, topN):
  row = len(datasetCombine[0].HoG_feat)
  col = len(datasetCombine[0].HoG_feat[0])

  # 取平均
  sum = datasetCombine[0].HoG_feat.tolist()
  for index in range(topN):
    for i in range(row):
      for j in range(col):
        if (index == 0):
          sum[i][j] /= topN
        else:
          sum[i][j] += datasetCombine[index].HoG_feat[i][j] / topN
  return np.array(sum)
'''
def queryExpasion(datasetCombine, topN):
  avgFeat = averageFeat(datasetCombine, topN)
  datasetCombine = similarityImg(datasetImg, avgFeat)
  showImg(datasetCombine, topN)
'''
################ 扩展查询 Query Expasion ###############

################ 计算召回率和精确度 ###############
def getPlateNum(filePath):
  # 利用正则表达式匹配车牌号(图片文件名)
  re_year = re.compile(r'_2015')  # 模式对象匹配
  index = re.search(re_year, filePath).span()[0]-6
  plateNum = filePath[index:index+6]
  return plateNum

def retrievalEvaluation(datasetRank):
  targetPlateNum = getPlateNum(datasetRank[0].img_src)

  # 计算召回率和精确度相关的变量
  correctNum = 1  # 检索正确的数量
  allNum = len(datasetRank)

  # 遍历检索结果
  for index in range(1, topN):
    tmpNum = getPlateNum(datasetRank[index].img_src)
    if tmpNum == targetPlateNum:
      # 如果车牌号正确匹配，即检索成功
      correctNum += 1
  
  # 召回率：正确结果占总体数据
  recall = correctNum / allNum
  # 精确度：正确结果占返回结果
  precision = correctNum / topN

  return [recall, precision]
################ 计算召回率和精确度 ###############
datasetCombine = similarityImg(datasetImg, queryImgFeatHoG, queryImgFeatSIFT)

# 首次查询
queryRetrieval(datasetCombine, topN)

# 检索评价实现
[recall, precision] = retrievalEvaluation(datasetCombine)

print('首次查询成功！')
print('Percision：%f' %precision)
print('Recall：%f' %recall)

# 扩展查询
print('扩展查询开始！')

avgFeat = averageFeat(datasetCombine, topN)
datasetCombine = similarityImg(datasetImg, avgFeat)
showImg(datasetCombine, topN)

[recallEQ, precisionEQ] = retrievalEvaluation(datasetCombine)

print('扩展查询成功！')
print('PercisionEQ：%f' %precisionEQ)
print('RecallEQ：%f' %recallEQ)




