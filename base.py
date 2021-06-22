import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Lasso,SGDRegressor,PassiveAggressiveRegressor,ElasticNet,LinearRegression,LassoCV, Ridge, MultiTaskLassoCV


path = 'D:\\毕业设计\\大四下\\AwA2-data\\Animals_with_Attributes2\\'
classname = pd.read_csv(path+'classes.txt', header=None, sep='\t')
dic_class2name = {classname.index[i]: classname.loc[i][1] for i in range(classname.shape[0])}
dic_name2class = {classname.loc[i][1]: classname.index[i] for i in range(classname.shape[0])}


def make_test_attributetable():
    # attribut_bmatrix_txt = pd.read_csv(path+'predicate-matrix-continuous-01.txt', header=None, sep=',')
    attribut_bmatrix_txt = pd.read_csv(path + 'predicate-matrix-continuous.txt', header=None, sep=',')
    # attribut_bmatrix_txt = pd.read_csv(path + 'predicate-matrix-binary.txt', header=None, sep=',')
    attribut_bmatrix = []
    for i in range(len(attribut_bmatrix_txt)):
        attribut_bmatrix.append([float(d) for d in attribut_bmatrix_txt.values[i][0].split()])
    attribut_bmatrix = pd.DataFrame(attribut_bmatrix)

    test_classes = pd.read_csv(path+'testclasses.txt',header=None)
    test_classes_flag = []
    for item in test_classes.iloc[:,0].values.tolist():
        test_classes_flag.append(dic_name2class[item])
    return attribut_bmatrix.iloc[test_classes_flag,:]

def make_train_attributetable():
    # attribut_bmatrix_txt = pd.read_csv(path+'predicate-matrix-continuous-01.txt', header=None, sep=',')
    attribut_bmatrix_txt = pd.read_csv(path + 'predicate-matrix-continuous.txt', header=None, sep=',')
    # attribut_bmatrix_txt = pd.read_csv(path + 'predicate-matrix-binary.txt', header=None, sep=',')
    attribut_bmatrix = []
    for i in range(len(attribut_bmatrix_txt)):
        attribut_bmatrix.append([float(d) for d in attribut_bmatrix_txt.values[i][0].split()])
    attribut_bmatrix = pd.DataFrame(attribut_bmatrix)
    trainlabel = np.load(path+'AWA2_trainlabel.npy')
    trainlabel = pd.DataFrame(trainlabel).set_index(0)

    return trainlabel.join(attribut_bmatrix)


def construct_Y(label_onehot):
    for i in range(label_onehot.shape[0]):
        for j in range(label_onehot.shape[1]):
            if label_onehot[i][j] == 0:
                label_onehot[i][j] = -1
    return np.mat(label_onehot)

def generate_data(data_mean,data_std,attribute_table,num):
    class_num = data_mean.shape[0]
    feature_num = data_mean.shape[1]
    data_list = []
    label_list = []
    for i in range(class_num):
        data = []
        for j in range(feature_num):
            data.append(list(np.random.normal(data_mean[i,j], np.abs(data_std[i,j]),num)))
        data = np.row_stack(data).T
        data_list.append(data)
        label_list+=[test_attributetable.iloc[i,:].values]*num
    return np.row_stack(data_list),np.row_stack(label_list)

trainlabel = np.load(path+'AWA2_trainlabel.npy')
train_attributelabel = np.load(path+'AWA2_train_01.npy')

testlabel = np.load(path+'AWA2_testlabel.npy')
test_attributelabel = np.load(path+'AWA2_test_01.npy')

trainfeatures = np.load(path+'my_train_feature.npy')
testfeatures = np.load(path+'my_test_feature.npy')


train_attributetable = make_train_attributetable()
test_attributetable = make_test_attributetable()


trainfeatures_tabel = pd.DataFrame(trainfeatures)
trainfeatures_tabel['label'] = trainlabel


# clf = Lasso(alpha=0.009)
clf = Ridge(alpha=0.01)
# Lambdas=np.logspace(-5,2,200)
# clf = MultiTaskLassoCV(alphas=Lambdas,normalize=True,cv=10,max_iter=10000)
# clf.fit(np.mat(train_attributetable.values).T, np.mat(test_attributetable.values).T)
clf.fit(np.mat(trainfeatures), np.mat(train_attributetable.values))
# clf.fit(np.mat(train_attributelabel).T, np.mat(test_attributelabel).T)
# W = clf.coef_.T
W = clf.coef_.T
ans = np.dot(testfeatures, W)
dist = 1 - cosine_similarity(ans, test_attributetable)

label_lis = []
lis = [24, 38, 14, 5, 41, 13, 17, 47, 33, 23]
# lis = data['param'][0][0][1]
for i in range(dist.shape[0]):
    loc = dist[i].argmin()
    label_lis.append(lis[loc])

print(accuracy_score(list(testlabel), label_lis))

lb = np.unique(testlabel)
label_lis = np.array(label_lis)

for i in range(lb.shape[0]):
    itestlabel = testlabel[testlabel == lb[i]]
    ilabel_lis = label_lis[testlabel == lb[i]]
    print('type ' + str(lb[i] + 1) + ' ' + dic_class2name[lb[i]]+' ' + ' acc is :' + str(accuracy_score(list(itestlabel),
                                                                 list(ilabel_lis))))
    lb_lis = np.unique(ilabel_lis)
    num = ilabel_lis.shape[0]
    for j in range(lb_lis.shape[0]):
        inum = ilabel_lis[ilabel_lis == lb_lis[j]].shape[0]
        print('type ' + str(lb_lis[j] + 1)+ ' ' + dic_class2name[lb_lis[j]]+' '+ ' acc is :' + str(inum) + '/' + str(num))

    print()