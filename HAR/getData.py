import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

class GetDataSet(object):
    def __init__(self, dataSetName, isIID):
        self.name = dataSetName
        self.train_data = None
        self.train_label = None
        self.train_data_size = None
        self.test_data = None
        self.test_label = None
        self.test_data_size = None
        self.train_labels = None 
        self.directory = '/home/LvYongyang/FedLabs/FLddetor/HAR/data'
        self.test_labels = None 

        self._index_in_train_epoch = 0

        if self.name == 'har':
            self.harDataSetConstruct(isIID)
        else:
            raise ValueError("Data set not supported")

    def harDataSetConstruct(self, isIID):
        # 加载数据集
        features_train = pd.read_csv(
        self.directory + '/train/X_train.txt', header=None)
        features_test = pd.read_csv(
        self.directory + '/test/X_test.txt', header=None)
    
        labels_train = pd.read_csv(
        self.directory + '/train/y_train.txt', header=None, dtype=int)  # 确保标签是整数类型
        labels_test = pd.read_csv(
        self.directory + '/test/y_test.txt', header=None, dtype=int)  # 确保标签是整数类型
    
        # 清洗数据，确保所有值都是数值类型并且没有空值
        features_train = features_train.apply(pd.to_numeric, errors='coerce')
        features_test = features_test.apply(pd.to_numeric, errors='coerce')
    
        # 去除由非数值转换产生的任何NaN值
        features_train = features_train.dropna()
        features_test = features_test.dropna()

        # 归一化
        scaler = StandardScaler()
        features_train = scaler.fit_transform(features_train)
        features_test = scaler.transform(features_test)
    
        # 转换标签为独热编码
        self.train_label = self.one_hot_encode(labels_train)
        self.test_label = self.one_hot_encode(labels_test)
    
        # 转换numpy数组为torch Tensor
        self.train_data = torch.tensor(features_train, dtype=torch.float32)
        self.test_data = torch.tensor(features_test, dtype=torch.float32)
        self.train_labels = torch.tensor(self.train_label, dtype=torch.long)
        self.test_labels = torch.tensor(self.test_label, dtype=torch.long)

        # 更新数据集大小
        self.train_data_size = self.train_data.shape[0]
        self.test_data_size = self.test_data.shape[0]


        if self.isIID:
            # IID情况下的随机划分
            indices = np.arange(self.train_data_size)
            np.random.shuffle(indices)
            self.train_data = self.train_data[indices]
            self.train_labels = self.train_labels[indices]
        else:
            labels = np.argmax(self.train_data_size, axis=1)
            order = np.argsort(labels)
            self.train_data = self.train_data[order]
            self.train_label = self.train_labels[order]

    def one_hot_encode(self, labels):
        # 将标签转换为独热编码格式
        return np.eye(len(np.unique(labels)))[labels].astype(np.float32)

# 示例使用
if __name__ == "__main__":
    # dataSetName = 'chinese-mnist'  # 设置数据集名称

    chinesemnist = GetDataSet('har', False)
    
    if isinstance(chinesemnist.train_data, torch.Tensor) and isinstance(chinesemnist.test_data, torch.Tensor):
        print('the type of data is torch Tensor')
    else:
        print('the type of data is not torch Tensor')
    print('the shape of the train data set is {}'.format(chinesemnist.train_data.shape))
    print('the shape of the test data set is {}'.format(chinesemnist.test_data.shape))