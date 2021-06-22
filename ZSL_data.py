import pandas as pd
import os
import numpy as np
import cv2
from PIL import Image

image_size = 224  # 指定图片大小
path = 'D:\\毕业设计\\大四下\\AwA2-data\\Animals_with_Attributes2\\'  # 文件读取路径


classname = pd.read_csv(path + 'classes.txt', header=None, sep='\t')
dic_class2name = {classname.index[i]: classname.loc[i][1] for i in range(classname.shape[0])}
dic_name2class = {classname.loc[i][1]: classname.index[i] for i in range(classname.shape[0])}

# 两个字典，记录标签信息，分别是数字对应到文字，文字对应到数字

# 根据目录读取一类图像，read_num指定每一类读取多少图片，图片大小统一为image_size
def load_Img(imgDir, item, read_num='max'):
    imgs = os.listdir(imgDir)
    imgs = np.ravel(pd.DataFrame(imgs).sort_values(by=0).values)
    if read_num == 'max':
        imgNum = len(imgs)
        # if(len(imgs) > 500):
        #     imgNum = 500
        # else:
        #     imgNum = len(imgs)

    else:
        if (len(imgs) > int(read_num)):
            imgNum = 300
        else:
            imgNum = read_num
    data = np.empty((imgNum, image_size, image_size, 3), dtype="float32")
    print(item, imgNum)
    for i in range(imgNum):
        img = Image.open(imgDir + "/" + imgs[i])
        arr = np.asarray(img, dtype="float32")
        if arr.shape[1] > arr.shape[0]:
            arr = cv2.copyMakeBorder(arr, int((arr.shape[1] - arr.shape[0]) / 2),
                                     int((arr.shape[1] - arr.shape[0]) / 2), 0, 0, cv2.BORDER_CONSTANT, value=0)
        else:
            arr = cv2.copyMakeBorder(arr, 0, 0, int((arr.shape[0] - arr.shape[1]) / 2),
                                     int((arr.shape[0] - arr.shape[1]) / 2), cv2.BORDER_CONSTANT,
                                     value=0)  # 长宽不一致时，用padding使长宽一致
        arr = cv2.resize(arr, (image_size, image_size))
        # arr2 = cv2.resize(arr, (0, 0), fx=image_size / arr.shape[0], fy=image_size / arr.shape[1], interpolation=cv2.INTER_NEAREST)
        # cv2.imwrite(".\\img1.png", arr1)
        # cv2.imwrite(".\\img2.png", arr2)
        if len(arr.shape) == 2:
            temp = np.empty((image_size, image_size, 3))
            temp[:, :, 0] = arr
            temp[:, :, 1] = arr
            temp[:, :, 2] = arr
            arr = temp
        data[i, :, :, :] = arr
    return data, imgNum


# 读取数据
def load_data(train_classes, test_classes, num):
    read_num = 'max'

    traindata_list = []
    trainlabel_list = []
    testdata_list = []
    testlabel_list = []

    for item in train_classes.iloc[:, 0].values.tolist():
        tup = load_Img(path + 'JPEGImages/' + item, item, read_num=read_num)
        traindata_list.append(tup[0])
        trainlabel_list += [dic_name2class[item]] * tup[1]

    # for item in test_classes.iloc[:, 0].values.tolist():
    #     tup = load_Img(path + 'JPEGImages/' + item, item, read_num=read_num)
    #     testdata_list.append(tup[0])
    #     testlabel_list += [dic_name2class[item]] * tup[1]

    # return np.row_stack(traindata_list), np.array(trainlabel_list), np.row_stack(testdata_list), np.array(
    #     testlabel_list)
    # return np.row_stack(testdata_list), np.array(testlabel_list)
    return np.row_stack(traindata_list), np.array(trainlabel_list)

train_classes = pd.read_csv(path + 'trainclasses.txt', header=None)
test_classes = pd.read_csv(path + 'testclasses.txt', header=None)

# traindata, trainlabel, testdata, testlabel = load_data(train_classes, test_classes, num='max')
# testdata, testlabel = load_data(train_classes, test_classes, num='max')
traindata, trainlabel = load_data(train_classes, test_classes, num='max')

print("Finsh")
# print(traindata.shape, trainlabel.shape, testdata.shape, testlabel.shape)
# print(testdata.shape, testlabel.shape)
print(traindata.shape, trainlabel.shape)
print("Saving")

# 降图像和标签保存为numpy数组，下次可以直接读取
np.save(path + 'AWA2_224_traindata.npy', traindata)
# np.save(path + 'AWA2_224_testdata.npy', testdata)

np.save(path + 'AWA2_trainlabel.npy', trainlabel)
# np.save(path + 'AWA2_testlabel.npy', testlabel)

print('finish')
