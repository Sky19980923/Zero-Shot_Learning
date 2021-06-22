import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import torch
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch import nn, optim
import lightgbm as lgb
import warnings
import pickle

warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression

path = 'D:\\毕业设计\\大四下\\AwA2-data\\Animals_with_Attributes2\\'

classname = pd.read_csv(path + 'classes.txt', header=None, sep='\t')
dic_class2name = {classname.index[i]: classname.loc[i][1] for i in range(classname.shape[0])}
dic_name2class = {classname.loc[i][1]: classname.index[i] for i in range(classname.shape[0])}

class dataset(Dataset):
    def __init__(self, data, label, transform):
        super().__init__()
        self.data = data
        self.label = label
        self.transform = transform

    def __getitem__(self, index):
        return self.transform(self.data[index]), self.label[index]

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


traindata = np.load(path + 'AWA2_224_traindata.npy')
trainlabel = np.load(path + 'AWA2_trainlabel.npy')
# train_attributelabel = np.load(path + 'data\\AWA2_train_attributelabel.npy', allow_pickle=True)
# train_attributelabel = np.load(path + 'AWA2_train_continuous_01.npy', allow_pickle=True)
# train_attributelabel = np.load(path + 'AWA2_train_continuous_01_attributelabel.npy', allow_pickle=True)
train_attributelabel = np.load(path + 'AWA2_train_01.npy', allow_pickle=True)
# train_attributelabel = []
# for i in range(len(train_attributelabel_txt)):
#     train_attributelabel.append([float(d) for d in train_attributelabel_txt[i][0].split()])
# train_attributelabel = pd.DataFrame(train_attributelabel)


testdata = np.load(path + 'AWA2_224_testdata.npy')
testlabel = np.load(path + 'AWA2_testlabel.npy')
# test_attributelabel = np.load(path + 'data\\AWA2_test_attributelabel.npy', allow_pickle=True)
# test_attributelabel = np.load(path + 'AWA2_test_continuous_01.npy', allow_pickle=True)
# test_attributelabel = np.load(path + 'AWA2_test_continuous_01_attributelabel.npy', allow_pickle=True)
test_attributelabel = np.load(path + 'AWA2_test_01.npy', allow_pickle=True)
# test_attributelabel = []
# for i in range(len(test_attributelabel_txt)):
#     test_attributelabel.append([float(d) for d in test_attributelabel_txt[i][0].split()])
# test_attributelabel = pd.DataFrame(test_attributelabel)

print(traindata.shape, trainlabel.shape, train_attributelabel.shape)
print(testdata.shape, testlabel.shape, test_attributelabel.shape)

data_tf = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize([0.485 * 255, 0.456 * 255, 0.406 * 255], [0.229 * 255, 0.224 * 255, 0.225 * 255])])

train_dataset = dataset(traindata, trainlabel, data_tf)
test_dataset = dataset(testdata, testlabel, data_tf)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = models.resnet101(pretrained=True)  # 使用训练好的resnet101

if torch.cuda.is_available():
    model = model.cuda()

model.eval()

exact_list = ['avgpool']  # 提取最后一层池化层的输出作为图像特征
myexactor = FeatureExtractor(model, exact_list)

train_feature_list = []
for data in tqdm(train_loader):
    img, label = data
    if torch.cuda.is_available():
        with torch.no_grad():
            img = Variable(img).cuda()
        with torch.no_grad():
            label = Variable(label).cuda()
    else:
        with torch.no_grad():
            img = Variable(img)
        with torch.no_grad():
            label = Variable(label)
    feature = myexactor(img)[0]
    feature = feature.resize(feature.shape[0], feature.shape[1])
    train_feature_list.append(feature.detach().cpu().numpy())

trainfeatures = np.row_stack(train_feature_list)

test_feature_list = []
for data in tqdm(test_loader):
    img, label = data
    if torch.cuda.is_available():
        with torch.no_grad():
            img = Variable(img).cuda()
        with torch.no_grad():
            label = Variable(label).cuda()
    else:
        with torch.no_grad():
            img = Variable(img)
        with torch.no_grad():
            label = Variable(label)
    feature = myexactor(img)[0]
    feature = feature.resize(feature.shape[0], feature.shape[1])
    test_feature_list.append(feature.detach().cpu().numpy())

testfeatures = np.row_stack(test_feature_list)

print(trainfeatures.shape, testfeatures.shape)
np.save(path+'my50_train_feature.npy', trainfeatures)
np.save(path+'my50_test_feature.npy', testfeatures)
