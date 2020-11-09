from numpy import *
import operator

# 图像信息类

# 线性融合参数 手动调节
alpha_HoG = 0.7
alpha_SIFT = 1 - alpha_HoG

class ImgInfo(object):
  similarity = -1
  
  # 构造函数
  def __init__(self, img_src, HoG_similarity, SIFT_similarity, HoG_feat):
    self.img_src = img_src
    self.HoG_similarity = HoG_similarity
    self.HoG_feat = HoG_feat
    self.similarity = ImgInfo.combine_similarity_linearly(HoG_similarity, SIFT_similarity)
  
  # 线性组合特征
  def combine_similarity_linearly(HoG_similarity, SIFT_similarity):
    similarity = alpha_HoG * (HoG_similarity + 0.8) + alpha_SIFT * SIFT_similarity / 100
    return similarity

  # 相等操作符
  def __eq__(self, other):
    return self.img_src == other.img_src and self.HoG_similarity == other.HoG_similarity

  # 大于操作符
  def __gt__(self, other):
    return  self.similarity > other.similarity

  # 小于操作符
  def __lt__(self, other):
    return self.similarity < other.similarity

