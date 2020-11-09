# 本代码思路参考自：菜鸟教程
# 处理多文件读取操作，读取多个文件的地址放在一个数组中返回
import os
import re

# 存储结果的数组
resultArr = [];

# 读取数据库文件夹里面的图片，返回所有图片的地址列表
def readImgs(addr):
  # 因为目前所有的图片都是.jpg形式，所以正则判断时只用了.jpg
  reg = re.compile(r'.jpg');
  for filename in os.listdir(addr):
    filePath = addr + filename;
    if re.search(reg, filePath):  # 如果是图片的话直接加到列表
      resultArr.append(filePath);
    else :
      # 每个文件夹存在一个看不到但是会被python识别的文件，做特殊处理（就是不处理）
      if filename != 'Thumbs.db': # 这个情况下，如果条件达成，意味着是文件夹
        readImgs(filePath + '/'); # 是文件夹的话递归就好
  return resultArr;

