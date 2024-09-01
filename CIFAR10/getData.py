import numpy as np
import gzip
import os
import platform
import pickle
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import copy

class GetDataSet(object):
    def __init__(self, dataSetName, isIID):
        self.name = dataSetName
        self.train_data = None
        self.train_label = None
        self.train_data_size = None
        self.test_data = None
        self.test_label = None
        self.test_data_size = None

        self._index_in_train_epoch = 0

        if self.name == 'cifar10':
            self.cifar10DataSetConstruct(isIID)
        else:
            pass
            
    def cifar10DataSetConstruct(self, isIID):
        data_dir = '/home/LvYongyang/FedLabs/FLTrust-attack/data/cifar-10-batches-py'
        # if not os.path.exists(data_dir):
        #     os.makedirs(data_dir)
        #     self.download_cifar10(data_dir)
    
        train_data, train_labels = self.load_cifar10_batches(data_dir)
        test_data, test_labels = self.load_cifar10_test_batch(os.path.join(data_dir, 'test_batch'))

        assert train_data.shape[0] == train_labels.shape[0]
        assert test_data.shape[0] == test_labels.shape[0]

        self.train_data_size = train_data.shape[0]
        self.test_data_size = test_data.shape[0]

        # 确保数据是正确的形状 (batch_size, channels, height, width)
        self.train_data = train_data
        self.test_data = test_data

        # Converting class labels from scalars to one-hot vectors
        self.train_label = dense_to_one_hot(train_labels, num_classes=10)
        # print('train_labels', train_labels)
        self.test_label = dense_to_one_hot(test_labels, num_classes=10)

        if isIID:
            order = np.arange(self.train_data_size)
            np.random.shuffle(order)
            self.train_data = self.train_data[order]
            self.train_label = self.train_label[order]
        else:
            # 对于非IID情况，这里我们按照标签的值排序
            labels = np.argmax(self.train_label, axis=1)
            order = np.argsort(labels)
            self.train_data = self.train_data[order]
            self.train_label = self.train_label[order]
            # print('train_labels', train_labels)
            
        # self.test_data = torch.tensor(test_data).clone().detach()
        # self.test_label = torch.tensor(test_labels).clone().detach()

    def load_cifar10_batches(self, data_dir):
        batches = []
        labels = []
        for batch_file in range(1, 6):  # There are 5 training batches
            batch_data, batch_labels = self.load_cifar10_batch(os.path.join(data_dir, f'data_batch_{batch_file}'))
            temp_batches = copy.deepcopy(batch_data)
            temp_labels = copy.deepcopy(batch_labels)
            batches.append(temp_batches)
            labels.append(temp_labels)
        return np.concatenate(batches), np.concatenate(labels)

    def load_cifar10_batch(self, filename):
        with open(filename, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
            data = batch[b'data']
            labels = np.array(batch[b'labels'])
            # 保留原始的彩色图像，不再转换为灰度图像
            # data = np.dot(data.reshape(-1, 3), [0.299, 0.587, 0.114]).reshape(data.shape[0], 1, 32, 32)
            # data = data.astype(np.float32) / 255.0  # 归一化
            data = data.reshape(data.shape[0], 3, 32, 32).astype(np.float32) / 255.0  # 归一化
            return data, labels

    def load_cifar10_test_batch(self, filename):
        with open(filename, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
            data = batch[b'data']
            labels = np.array(batch[b'labels'])
            # 保留原始的彩色图像，不再转换为灰度图像
            # data = np.dot(data.reshape(-1, 3), [0.299, 0.587, 0.114]).reshape(data.shape[0], 1, 32, 32)
            # data = data.astype(np.float32) / 255.0  # 归一化
            data = data.reshape(data.shape[0], 3, 32, 32).astype(np.float32) / 255.0  # 归一化
            return data, labels


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


if __name__ == "__main__":
    'test data set'
    cifar10DataSet = GetDataSet('cifar10', False) # test CIFAR-10
    if isinstance(cifar10DataSet.train_data, torch.Tensor) and isinstance(cifar10DataSet.test_data, torch.Tensor):
        print('the type of data is torch Tensor')
    else:
        print('the type of data is not torch Tensor')
    print('the shape of the train data set is {}'.format(cifar10DataSet.train_data.shape))
    print('the shape of the test data set is {}'.format(cifar10DataSet.test_data.shape))
