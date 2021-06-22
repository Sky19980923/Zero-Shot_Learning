import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

tsne = TSNE(n_components=2, random_state=0)


path = 'D:\\毕业设计\\大四下\\AwA2-data\\Animals_with_Attributes2\\'
classname = pd.read_csv(path+'classes.txt',header=None,sep = '\t')
dic_class2name = {classname.index[i]:classname.loc[i][1] for i in range(classname.shape[0])}

# train_feature_npy = path + 'my_train_feature.npy'
test_s_npy = path + 'AWA2_test_continuous_01.npy'

# train_feature_npy = path + 'resnet101_trainfeatures.npy'
# test_feature_npy = path + 'resnet101_testfeatures.npy'
# train_label_npy = path + 'AWA2_trainlabel.npy'
test_label_npy = path + 'AWA2_testlabel.npy'

# train_feature = np.load(train_feature_npy)
test_s = np.load(test_s_npy)

# train_label = np.load(train_label_npy)
test_label = np.load(test_label_npy)
label = []
for i in range(len(test_label)):
    label.append(dic_class2name[test_label[i]])

test_label = np.array(label)
# print(train_feature)

# data = test_s  # (30337, 2048)
data = np.load(path + 'myFe101_test_feature.npy')
# data = test_feature.squeeze(1)
tsne_obj = tsne.fit_transform(data)


tsne_df = pd.DataFrame({'X': tsne_obj[:, 0],
                        'Y': tsne_obj[:, 1],
                        # 'Z': tsne_obj[:, 2],
                        'digit': np.hstack(test_label)})

# sns.scatterplot(x = 'X', y='Y',data = tsne_df)

sns.scatterplot(x="X", y="Y",
              hue="digit",
              palette=['purple','red','orange','green','brown',
                       'blue','dodgerblue','lightgreen','darkcyan', 'black'
                       ],
              # legend='full',
              data=tsne_df)
plt.axis('off')
# plt.savefig('tsne200data.png')
# plt.close()
plt.show()



# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(tsne_df['X'][tsne_df['digit'] == 1], tsne_df['Y'][tsne_df['digit'] == 1], tsne_df['Z'][tsne_df['digit'] == 1], c = 'r')
# ax.scatter(tsne_df['X'][tsne_df['digit'] == 2], tsne_df['Y'][tsne_df['digit'] == 2], tsne_df['Z'][tsne_df['digit'] == 2], c = 'g')
# ax.scatter(tsne_df['X'][tsne_df['digit'] == 3], tsne_df['Y'][tsne_df['digit'] == 3], tsne_df['Z'][tsne_df['digit'] == 3], c = 'b')
# ax.scatter(tsne_df['X'][tsne_df['digit'] == 4], tsne_df['Y'][tsne_df['digit'] == 4], tsne_df['Z'][tsne_df['digit'] == 4], c = 'y')
# ax.legend([3, 6, 4, 63])
# for i in range(int(test_label.shape[0])):
#     ax.scatter(tsne_df['X'][tsne_df['digit'] == test_label[i]], tsne_df['Y'][tsne_df['digit'] == test_label[i]],
#                tsne_df['Z'][tsne_df['digit'] == test_label[i]])
# plt.show()