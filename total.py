from sklearn.linear_model import Lasso,SGDRegressor,PassiveAggressiveRegressor,ElasticNet,LinearRegression,LassoCV, Ridge, MultiTaskLassoCV
import os
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import torch
from torchvision import datasets, models, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch import nn
import warnings

warnings.filterwarnings("ignore")

class dataset(Dataset):
    def __init__(self, data, transform):
        super().__init__()
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        return self.transform(self.data[index])

    def __len__(self):
        return self.data.shape[0]


class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            if name == "fc": x = x.view(x.size(0), -1)
            x = module(x)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs


def make_test_attributetable(path, dic_name2class):
    attribut_bmatrix_txt = pd.read_csv(path + 'predicate-matrix-continuous.txt', header=None, sep=',')
    attribut_bmatrix = []
    for i in range(len(attribut_bmatrix_txt)):
        attribut_bmatrix.append([float(d) for d in attribut_bmatrix_txt.values[i][0].split()])
    attribut_bmatrix = pd.DataFrame(attribut_bmatrix)

    test_classes = pd.read_csv(path+'testclasses.txt',header=None)
    test_classes_flag = []
    for item in test_classes.iloc[:,0].values.tolist():
        test_classes_flag.append(dic_name2class[item])
    return attribut_bmatrix.iloc[test_classes_flag,:]

def make_train_attributetable(path, dic_name2class):
    attribut_bmatrix_txt = pd.read_csv(path + 'predicate-matrix-continuous.txt', header=None, sep=',')
    attribut_bmatrix = []
    for i in range(len(attribut_bmatrix_txt)):
        attribut_bmatrix.append([float(d) for d in attribut_bmatrix_txt.values[i][0].split()])
    attribut_bmatrix = pd.DataFrame(attribut_bmatrix)

    train_classes = pd.read_csv(path+'trainclasses.txt',header=None)
    train_classes_flag = []
    for item in train_classes.iloc[:,0].values.tolist():
        train_classes_flag.append(dic_name2class[item])
    return attribut_bmatrix.iloc[train_classes_flag,:]


def dataNorm(data):
    data_tf = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize([0.485 * 255, 0.456 * 255, 0.406 * 255],
                                                       [0.229 * 255, 0.224 * 255, 0.225 * 255])])
    data_set = dataset(data, data_tf)
    data_loader = DataLoader(data_set, batch_size=1, shuffle=False)

    return data_loader


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


def RKT(trainfeatures, trainlabel, train_attributetable, test_attributetable, testfeatures):
    trainfeatures_tabel = pd.DataFrame(trainfeatures)
    trainfeatures_tabel['label'] = trainlabel

    trainfeature_mean = np.mat(trainfeatures_tabel.groupby('label').mean().values).T
    trainfeature_std = np.mat(trainfeatures_tabel.groupby('label').std().values).T

    Lambdas = np.logspace(-5, 2, 200)
    clf = MultiTaskLassoCV(alphas=Lambdas, normalize=True, cv=10, max_iter=10000)
    clf.fit(np.mat(train_attributetable.values).T, np.mat(test_attributetable.values).T)
    W = clf.coef_.T

    virtual_testfeature_mean = (trainfeature_mean * W).T
    virtual_testfeature_std = np.ones(virtual_testfeature_mean.shape) * 0.3

    virtual_testfeature, virtual_test_attributelabel = generate_data(virtual_testfeature_mean,
                                                                     virtual_testfeature_std,
                                                                     test_attributetable,
                                                                     50)
    rand_index = np.random.choice(virtual_testfeature.shape[0], virtual_testfeature.shape[0], replace=False)
    virtual_testfeature = virtual_testfeature[rand_index]
    virtual_test_attributelabel = virtual_test_attributelabel[rand_index]

    res_list = []
    for i in range(virtual_test_attributelabel.shape[1]):
        # print("{} th classifier is training".format(i+1))
        clf = LinearRegression()
        clf.fit(virtual_testfeature, virtual_test_attributelabel[:, i])
        res = clf.predict(testfeatures)
        res_list.append(list(res))

    return res_list

def MyModel():
    model = models.resnet101(pretrained=True)  # 使用训练好的resnet101

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    exact_list = ['avgpool']  # 提取最后一层池化层的输出作为图像特征
    myexactor = FeatureExtractor(model, exact_list)

    return myexactor

def feature(data, exactor):
    featureList = []
    for data in tqdm(data):
        img = data
        if torch.cuda.is_available():
            with torch.no_grad():
                img = Variable(img).cuda()
        else:
            with torch.no_grad():
                img = Variable(img)
        feature = exactor(img)[0]
        feature = feature.resize(feature.shape[0], feature.shape[1])
        featureList.append(feature.detach().cpu().numpy())

    features = np.row_stack(featureList)

    return features

def picMake(pic):
    # img = Image.open("./testphoto/" + imgs[i])
    img = Image.open(pic)
    arr = np.asarray(img, dtype="float32")
    if arr.shape[1] > arr.shape[0]:
        arr = cv2.copyMakeBorder(arr, int((arr.shape[1] - arr.shape[0]) / 2),
                                 int((arr.shape[1] - arr.shape[0]) / 2), 0, 0, cv2.BORDER_CONSTANT, value=0)
    else:
        arr = cv2.copyMakeBorder(arr, 0, 0, int((arr.shape[0] - arr.shape[1]) / 2),
                                 int((arr.shape[0] - arr.shape[1]) / 2), cv2.BORDER_CONSTANT,
                                 value=0)  # 长宽不一致时，用padding使长宽一致
    arr = cv2.resize(arr, (224, 224))
    return arr

def fileIn(file):
    print("图片预处理中.....")
    imgs = os.listdir(file)
    imgs = np.ravel(pd.DataFrame(imgs).sort_values(by=0).values)
    imgNum = imgs.shape[0]
    data = np.empty((imgNum, 224, 224, 3), dtype="float32")
    for i in range(imgNum):
        arr = picMake(file + '/' + imgs[i])
        data[i, :, :, :] = arr
    data = dataNorm(data)
    return data

def picIn(pc):
    print("图片预处理中.....")
    data = np.empty((1, 224, 224, 3), dtype="float32")
    data[0, :, :, :] = picMake(pc)
    data = dataNorm(data)
    return data


def pickup(way, label):
    os.mkdir('./out')
    imgs = os.listdir(way)
    fileName = np.unique(label).tolist()
    for i in range(len(fileName)):
        os.mkdir('./out/' + fileName[i])
    dic = {key: 0 for key in fileName}
    for i in range(len(imgs)):
        im = Image.open(way + '/' + imgs[i])
        im.save('./out/' + label[i] + '/TJN_' + label[i] + str(dic[label[i]]) + '.jpg')
        dic[label[i]] += 1


def ZSL(test_attributetable, dic_class2name, pick = False, way = None):
    print("图象特征获取中.....")
    myexactor = MyModel()
    testfeatures = feature(testfeature, myexactor)
    print("模型调用中......")
    res_list = RKT(trainfeatures, trainlabel, train_attributetable, test_attributetable, testfeatures)
    test_pre_attribute = np.mat(np.row_stack(res_list)).T
    # print(test_pre_attribute.shape)
    test_attributetable = make_test_attributetable(path, dic_name2class)

    label_lis = []
    for i in range(test_pre_attribute.shape[0]):
        pre_res = test_pre_attribute[i, :]
        loc = np.sum(np.square(test_attributetable.values - pre_res), axis=1).argmin()
        label_lis.append(test_attributetable.index[loc])
    print("识别结果为:")
    labelname = []
    for i in range(len(label_lis)):
        labelname.append(dic_class2name[label_lis[i]])
        print(dic_class2name[label_lis[i]])
    if(pick):
        pickup(way, labelname)


if __name__ == '__main__':
    path = 'D:\\毕业设计\\大四下\\AwA2-data\\Animals_with_Attributes2\\'
    TrainFeatureNPY = 'my_train_feature.npy'
    TrainLabel = 'AWA2_trainlabel.npy'
    classname = pd.read_csv(path + 'classes.txt', header=None, sep='\t')

    dic_class2name = {classname.index[i]: classname.loc[i][1] for i in range(classname.shape[0])}
    dic_name2class = {classname.loc[i][1]: classname.index[i] for i in range(classname.shape[0])}

    trainfeatures = np.load(path + TrainFeatureNPY)
    trainlabel = np.load(path + TrainLabel)

    train_attributetable = make_train_attributetable(path, dic_name2class)
    test_attributetable = make_test_attributetable(path, dic_name2class)

    while True:
        id = input("请输入file(图片批量识别)、pic(图片识别)、quit(退出)")
        if(id == 'file'):
            fl = input("请输入文件地址及文件名:")
            try:
                testfeature = fileIn(fl)
                ZSL(test_attributetable, dic_class2name, pick=True, way=fl)
            except:
                print("操作失败，无此文件夹，请输入正确文件夹名")
        elif(id == 'pic'):
            pc = input("请输入图片地址及图片名:")
            try:
                testfeature = picIn(pc)
                ZSL(test_attributetable, dic_class2name, pick=False)
            except:
                print("操作失败，无此图片，请输入正确文件夹名")
        elif(id == 'quit'):
            break
        else:
            print("输入有误，请重试")

