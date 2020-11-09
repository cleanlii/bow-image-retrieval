import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

'''
Hog源码参考：CSDN
'''

img_src = 'd:/test/1.jpg'  # 测试用图片路径
cell_size = 8  # 将cell设为8*8个像素
bin_size = 8  # 8个bin的直方图

# 图片大小格式化
img_height = 500
img_width = 500

def cell_gradient(cell_magnitude, cell_angle):
  angle_unit = 360 / bin_size
  orientation_centers = [0] * bin_size
  for k in range(cell_magnitude.shape[0]):
    for l in range(cell_magnitude.shape[1]):
      gradient_strength = cell_magnitude[k][l]
      gradient_angle = cell_angle[k][l]
      min_angle = int(gradient_angle / angle_unit)%8
      max_angle = (min_angle + 1) % bin_size
      mod = gradient_angle % angle_unit
      orientation_centers[min_angle] += (gradient_strength * (1 - (mod / angle_unit)))
      orientation_centers[max_angle] += (gradient_strength * (mod / angle_unit))
  return orientation_centers

def get_HoG_feature(img_src, resize_width = img_width, resize_height = img_height):
  # 读取图片，以灰度图的形式
  img = cv2.imread(img_src, cv2.IMREAD_GRAYSCALE)
  img = cv2.resize(img, (img_width, img_height))
  # 采用Gamma校正对图像的颜色空间归一化
  img = np.sqrt(img / float(np.max(img)))

  # 计算每个像素的梯度值
    # 计算图像横坐标和纵坐标方向的梯度，并据此计算每个像素位置的梯度方向值；
    # 求导操作不仅能够捕获轮廓，人影和一些纹理信息，还能进一步弱化光照的影响。
    # 在求出输入图像中像素点（x,y）处的水平方向梯度、垂直方向梯度和像素值，从而求出梯度幅值和方向。
  height, width = img.shape
  gradient_values_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
  gradient_values_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
  gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)
  gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)
  # print ('gradient_magnitude.shape, gradient_angle.shape：')
  # print (gradient_magnitude.shape, gradient_angle.shape)
  
  # 为每个细胞单元构建梯度方向直方图
  angle_unit = 360 / bin_size # 每个梯度方向块的角度
  gradient_magnitude = abs(gradient_magnitude)
  # print ('cell_gradient_vector:')
  cell_gradient_vector = np.zeros((height // cell_size, width // cell_size, bin_size)) 
  # print (cell_gradient_vector.shape)

  for i in range(cell_gradient_vector.shape[0]):
    for j in range(cell_gradient_vector.shape[1]):
      cell_magnitude = gradient_magnitude[i * cell_size:(i + 1) * cell_size,
                        j * cell_size:(j + 1) * cell_size]
      cell_angle = gradient_angle[i * cell_size:(i + 1) * cell_size,
                    j * cell_size:(j + 1) * cell_size]
      cell_gradient_vector[i][j] = cell_gradient(cell_magnitude, cell_angle)

  # 可视化Cell梯度直方图
  hog_image= np.zeros([height, width])
  cell_width = cell_size / 2
  max_mag = np.array(cell_gradient_vector).max()
  for x in range(cell_gradient_vector.shape[0]):
    for y in range(cell_gradient_vector.shape[1]):
      cell_grad = cell_gradient_vector[x][y]
      cell_grad /= max_mag
      angle = 0
      angle_gap = angle_unit
      for magnitude in cell_grad:
        angle_radian = math.radians(angle)
        x1 = int(x * cell_size + magnitude * cell_width * math.cos(angle_radian))
        y1 = int(y * cell_size + magnitude * cell_width * math.sin(angle_radian))
        x2 = int(x * cell_size - magnitude * cell_width * math.cos(angle_radian))
        y2 = int(y * cell_size - magnitude * cell_width * math.sin(angle_radian))
        cv2.line(hog_image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
        angle += angle_gap

  # plt.imshow(hog_image, cmap=plt.cm.gray)
  # plt.show()

  # 统计Block的梯度信息
  hog_vector = []
  for i in range(cell_gradient_vector.shape[0] - 1):
    for j in range(cell_gradient_vector.shape[1] - 1):
      block_vector = []
      block_vector.extend(cell_gradient_vector[i][j])
      block_vector.extend(cell_gradient_vector[i][j + 1])
      block_vector.extend(cell_gradient_vector[i + 1][j])
      block_vector.extend(cell_gradient_vector[i + 1][j + 1])
      mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
      magnitude = mag(block_vector)
      if magnitude != 0:
        normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
        block_vector = normalize(block_vector, magnitude)
      hog_vector.append(block_vector)
  # print ('np.array(hog_vector).shape')
  # print (np.array(hog_vector).shape)
  return np.array(hog_vector)

# print('以下目标图片的特征值：')
# print(get_HoG_feature(img_src))
# print(get_HoG_feature(img_src).shape)
# print('特征值计算成功！')

cv2.waitKey(0)

