import os
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

from typing import Tuple, Dict
from PIL import Image
from pylab import *

# 格式化统一大小
img_height = 500
img_width = 500

# 注：由于SIFT库在OpenCV3.4之后被移除，而2020年3月7日后，SIFT的专利将到期，因此应免费使用。 
# opencv将其移出非自由文件夹将使所有人都能使用默认标志编译opencv并仍然获得SIFT。
# 参见 https://github.com/opencv/opencv/issues/16736

sift_det: cv2.Feature2D = cv2.SIFT_create()

'''
def get_SIFT_feature(img_src, resize_width = img_width, resize_height = img_height):
  img = cv2.imread(img_src)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  gray = cv2.resize(img, (img_width, img_height))

  # 实例化函数
  sift = cv2.SIFT_create()
  # 找出图像中的关键点
  kp = sift.detect(gray, None)

  # 绘制关键点
  # ret = cv2.drawKeypoints(gray, kp, img)
  # cv2.imshow('ret', ret)
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()

  # 计算关键点对应的sift特征
  kp, des = sift.compute(gray, kp)
  # 这里用了cv的库一步到位，返回Descriptor
  return des
  '''

def get_SIFT_feature(img_src, resize_width = img_width, resize_height = img_height):
  img = cv2.imread(img_src)
  img = cv2.resize(img, (img_width, img_height))
  return sift_det.detectAndCompute(img, None)