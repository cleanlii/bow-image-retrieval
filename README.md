## 功能介绍
- 图片检索结果 Image Retrieval
- 精确度和召回率计算
- 调试数据记录（处理效率、特征值、相似度等）
## 总体思路
- 输入路径
- 预处理样本库、提取特征值（HoG/SIFT/LBP/Haar-like）
- 读取待检索图片、提取特征值
- 计算TF/IDF词频（Term Frequency/Inverse Document Frequency）
- 遍历样本库、计算相似度
- 排序、取最相似的前topN个结果
- 重排序（Re-ranking）
- 扩展查询（Query Expansion）
- 输出结果
## 伪代码

```python
% TF Term Frequency
% IDF Inverse Document Frequency

%读取图像->提取特征（HOG）->编码（kmeans）->得到特征向量
->欧式距离匹配计算->筛选结果->计算IDF->Re-rank->评价实现->QE

queryImageAddr = 'D:/vehicle.jpg';
DataSetAddr = 'D:/Data/';

% image parameters
imgHeight = 500;
imgWidth = 500;

% feature parameters
featType = 'HOG' % SIFT, HOG, LBP, Haar-like
cellSize = 3 % each cell contains cellSize * cellSize pixels

% Embedding parameters
clusterAlgo = 'k-means';  % k-means, graph-based, GMM
clusterNum = 5;

% retrieval parameters
topN = 10; % number of image to be return

% get words
for imgIndex = 1 : length(DataSetImgNames)
    datasetImg = imgRead(DataSetImgNames[imgIndex]);
    datasetImg = ImgResize(datasetImg, [imgHeight imgWidth]);
    datasetImgFeat[imgIndex] = FeatureExtraction(datasetImg);
end

words = cluster(datasetImgFeat); % K-means

% get query Embedding
queryImg = imgRead(queryImageAddr);
queryImg = ImgResize(queryImg, [imgHeight imgWidth]); %resize image to fixed size
queryImgFeat = FeatureExtraction(queryImg);  % SIFT, HOG, LBP, Haar-like
queryEmbedding = Encoding(queryImgFeat, words);
queryf = TF_IDFEncoding(queryEmbedding);

% get TF and IDF
tfs = CalcuTF(queryEmbedding);  % h_j / sum_i h_i
idfs = CalcuIDF(datasetEmbedding);  % log(N/f_j)
queryTF_IDFEmbedding = tfs * idfs;

% get dataset Embedding
DataSetImgNames = imgNameRead(DataSetAddr);

for imgIndex = 1 : length(DataSetImgNames)  % each image in the DataSet
    dataEmbedding[imgIndex] = Encoding(datasetImgFeat, words);
    tfs = CalcuTF(dataEmbedding[imgIndex]);
    dataTF_IDFEmbedding = tfs * idfs;
    dist[imgIndex] = EuclideanDist(queryTF_IDFEmbedding, dataTF_IDFEmbedding);  % sim[imgIndex] = CosSim(queryImg, datasetImg)
end

rank = Recorder(dist); % rank the order in dataset according to the Euclidean distance or cosine similirity

[recall, precision] = Measure(rank, topN);  % way to calculate recall and precision

```

## 重点解析
### Bag of Words 词袋模型
- 概念很容易理解，BoW模型即对于训练集中的每一幅图像提取其局部特征，然后将所有图像所提取出来的局部特征点作为一个整体进行聚类；
- BoW/BoF的关键思路在于提取一张图片的特征作为视觉单词/视觉特征（Words/Feature），然后一个个单词聚类/编码成视觉词典/码本（Vocabulary/Codebook）；
![1](https://img-blog.csdnimg.cn/20201109110525188.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2NsZWFubGlp,size_16,color_FFFFFF,t_70#pic_center)

- 除BoW模型之外还有如FV（Fisher Vector），著名的VLAD（Vector of locally aggregated descriptors）；
- 这篇文章对该模型做了很详细的解释 [袁勇：BoW图像检索原理与实战](http://yongyuan.name/blog/CBIR-BoW-for-image-retrieval-and-practice.html)
### Kmeans 聚类
- 那么提取到多张图片的多组单词之后怎么办呢？接下来就得开始进行聚类；
- 理解了kmeans的算法原理，也就理解BoW模型是怎样实现的了；
- Kmeans本质是一种迭代算法，即对于 n 个待处理的特征点，先根据给定的簇的个数 k，初始化 k 个簇中心点，然后对于其他的每一个点，计算其到各个中心点的距离，并将其归入距离自己最近的中心点所在的簇中，如此重复这样一个过程；

```c
选择K个点作为初始质心 
repeat 
    将每个点指派到最近的质心，形成K个簇 
    重新计算每个簇的质心 
until 簇不发生变化或达到最大迭代次数
```
- 最后我们用到的是一个K维直方图，即统计单词表中每个单词在图像中出现的次数，从而将图像表示成为一个K维数值向量（这里的K是自己取的）；
![2](https://img-blog.csdnimg.cn/20201109112022506.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2NsZWFubGlp,size_16,color_FFFFFF,t_70#pic_center)

- 实际上BoW与FV、VLAD相比效率较低，内存占用比较高，用离特征点最近的聚类中心去代替该特征点，这个过程中损失了较多的信息。
### HoG 形状特征提取
- HoG全称是Histogram of Oriented Gradient，即方向梯度直方图；
- 知道全程之后，原理也非常好理解，即计算和统计图像局部区域的梯度方向直方图；
![3](https://img-blog.csdnimg.cn/20201109112928585.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2NsZWFubGlp,size_16,color_FFFFFF,t_70#pic_center)

- 首先将图像分成大区域block，每个block又分成小区域cell，然后采集cell中各像素点梯度或边缘方向直方图，最后把这些直方图组合起来构成特征描述；

### SIFT 纹理特征提取
- 与HoG不同的是，HoG得出的是一个大的多维特征向量，这是我们计算相似度所乐意得到的，而SIFT提取的则是一张图片中的大量特征点，而不同的图片提取出的特征点数量是不同的，所以有效的聚类显得很重要，这也是为什么SIFT往往和kmeans一起出没；

```css
1.构建尺度空间
2.构造高斯差分尺度空间
3.DoG尺度空间极值点检测
4.特征点精确定位
5.去除不稳定点
```
![4](https://img-blog.csdnimg.cn/20201109120233732.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2NsZWFubGlp,size_16,color_FFFFFF,t_70#pic_center)


- SIFT算法的原理更为复杂，我们这里可以直接调用OpenCV库提供的SIFT方法`detectAndCompute()`，有关cv调用失败的版权问题可以参考我的这篇博客 [2020年关于SIFT/SURF专利版权问题的解决办法](https://blog.csdn.net/cleanlii/article/details/109561089)

### 倒排索引&词频 TF-IDF
- TF即Term Frequency，IDF即Inverse Document Frequency；
![TF](https://img-blog.csdnimg.cn/20201109120518736.jpg#pic_center)
![IDF](https://img-blog.csdnimg.cn/20201109120518731.jpg#pic_center)

- 在文本检索中，TF-IDF技术常用以评估词语对于一个文件数据库中的其中一份文件的重要程度。词语的重要性随着它在文件中出现的频率成正比增加，但同时会随着它在文件数据库中出现的频率成反比下降。
- 引入这两个概念是为了用IDF对BoW模型得出的结果进行加权处理，减少其他不重要的视觉单词所占的权重，其中h为得到的一张图片的直方图向量，IDF公式即![1](https://img-blog.csdnimg.cn/20201109120152955.png#pic_center)
### 相似度计算
- 一般方法有两种，一种是直接计算欧式距离，一种是计算余弦相似度；
- 直接上代码：

```python
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
```

```python
# 余弦相似度
def CosinDist(A, B)
    A = np.mat(A)
    B = np.mat(B)
    num = float(A * B.T)
    denom = np.linalg.norm(A) * np.linalg.norm(B)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim
```

### Re-rank 重排序
- 将初始检索得到的图像进行分析和管理，即第二次排序；
![12](https://img-blog.csdnimg.cn/20201109135216265.jpg#pic_center)

- 常用的手段有两种，一种是基于内容的重排序（如上图），一种是基于文本的重排序，前者准确率较高但效率较低，后者则相反；
- 我这里用了两种最简单的重排序方法，分别是：
- 基于线性组合：先利用某一个特征得到排序效果，对多次排序效果进行**线性融合**,然后按照降序排列得到最终的排序结果。
![12](https://img-blog.csdnimg.cn/20201109135449243.jpg#pic_center)

- 基于聚类：对检索的**图像结果**进行聚类，首先对图像集群进行排序，然后在集群内部进行排序，为了得到最准确的相关聚类，将图像检索结果中最相关的集群图像聚在一起。
![32](https://img-blog.csdnimg.cn/20201109135457702.jpg#pic_center)
### Query Expasion 拓展查询
- 对返回的前topN个结果，包括查询样本本身，对它们的特征求和取平均，再做一次查询，此过程称为拓展查询。很好理解也很容易实现。
- 通过Query Expansion，以达到提高检索召回率的目的。

### RootSIFT 概念
- 一开始也是在袁勇大神的文章里接触到这个词，好像网上对这个概念的解释文献较少，我谷歌了一下当时提出这个概念的文献即 [Three things everyone should know to improve object retrieval ](https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf)，目的是找出一种比较欧氏距离更为精确的度量方法，对SIFT进行优化：
![123](https://img-blog.csdnimg.cn/20201109134746379.jpg#pic_center)

- 发现和SIFT的区别就是多了一个L1归一化然后取平方根（square-root）的过程。
## 核心代码
### HoG特征提取
```python
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
```
### SIFT特征提取
```python
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
```
### Kmeans聚类
```python
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
```
### 查询框架
```python
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

```
## 检索效果
- 我这里用的汽车样本库，通过搜索图片文件名中的汽车号衡量精确度和召回率；
![car](https://img-blog.csdnimg.cn/20201109141417971.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2NsZWFubGlp,size_16,color_FFFFFF,t_70#pic_center)
- 可以看到使用QE和不使用QE的情况下效果差距很明显，这种差距在样本量增大的情况下会更明显；
![result1](https://img-blog.csdnimg.cn/20201109141417799.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2NsZWFubGlp,size_16,color_FFFFFF,t_70#pic_center)
![result](https://img-blog.csdnimg.cn/20201109141417804.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2NsZWFubGlp,size_16,color_FFFFFF,t_70#pic_center)

## 总结
- BoW模型下的图像检索系统作为视觉方向的典中典项目以及CNN的经典入门项目之一（虽然并没有用到），其中牵扯到的原理知识并不是太复杂，但真正上手写代码的时候，还是容易出问题，整套系统的流程衔接一定得吃透，而如何处理从图片中提取到的特征、如何进行匹配是这个项目中最有意思、也是最具挑战性最关键的一环；
- 最开始做的时候用C++来写，碍于各种环境配置的麻烦（主要就是SIFT的专利问题，折腾了很久）和C对向量操作的不友好，后来慢慢转成了较为生疏的python，这也是一种挑战吧；
- 线代一定得好好学！线代一定得好好学！线代一定得好好学！
