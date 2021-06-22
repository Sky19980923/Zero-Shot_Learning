# Zero-Shot_Learning
基于Rkt算法进行零学习算法的实现，本次数据采用AWA2公开数据集http://cvml.ist.ac.at/AwA2/

## 代码介绍
  以下是对各个.py文件的作用进行介绍
### RKT
  基于RKT算法的ZSL实现，其中通过Lasso函数作为源域和目标域之间的语义隐射
### SAE
  基于自编码器的SEA算法
### ZSL_data
  数据的预处理
### base
  直接通过岭回归作为图像特征和予以特征之间的映射，并基于该函数作为目标域依据
### label
  对数据进行标注
### resnet
  对预处理后的图片进行特征提取
### svision
  对数据进行可视化输出
### total
  基于整个流程做了一个对图片的标注器，可对testphoto中的图片进行自动分类，标注
## 运行流程说明
  首先下载好AWA2数据集，先通过ZSL_data对数据进行预处理，再通过label进行数据标注，基于resnet提取出训练数据与测试数据的图片特征，之后就可运行RKT、SEA、total，同时可以基于svision进行特征可视化输出。
