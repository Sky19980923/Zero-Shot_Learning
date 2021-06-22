import pandas as pd
import numpy as np


path = 'D:\\毕业设计\\大四下\\AwA2-data\\Animals_with_Attributes2\\'

def make_attribute_label(trainlabel,testlabel):
    attribut_bmatrix_txt = pd.read_csv(path+'predicate-matrix-continuous.txt', header=None, sep=',')
    attribut_bmatrix = []
    for i in range(len(attribut_bmatrix_txt)):
        attribut_bmatrix.append([float(d) for d in attribut_bmatrix_txt.values[i][0].split()])
    attribut_bmatrix = pd.DataFrame(attribut_bmatrix)
    trainlabel = pd.DataFrame(trainlabel).set_index(0)
    testlabel = pd.DataFrame(testlabel).set_index(0)

    return trainlabel.join(attribut_bmatrix), testlabel.join(attribut_bmatrix)

trainlabel = np.load(path+'AWA2_trainlabel.npy')
testlabel = np.load(path+'AWA2_testlabel.npy')

# lb = np.array([24, 38, 14, 5, 41, 13, 17, 47, 33, 23])
train_attributelabel, test_attributelabel = make_attribute_label(trainlabel, testlabel)

# np.save(path+'AWA2_train_01_attributelabel.npy', train_attributelabel.values)
# np.save(path+'AWA2_test_01_attributelabel.npy', test_attributelabel.values)
# 01特征作为标签

# np.save(path+'AWA2_train_attributelabel.npy', train_attributelabel.values)
# np.save(path+'AWA2_test_attributelabel.npy', test_attributelabel.values)
# 连续值作为标签

np.save(path+'AWA2_train_con.npy', train_attributelabel.values)
np.save(path+'AWA2_test_con.npy', test_attributelabel.values)
print(train_attributelabel.shape, test_attributelabel.shape)

# np.save(path+'one_AWA2_train_continuous_01.npy', train_attributelabel.values)
# print(test_attributelabel.shape)
# np.save(path+'one_AWA2_test_01.npy', test_attributelabel.values)