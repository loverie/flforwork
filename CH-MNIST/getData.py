import pandas as pd
import numpy as np
import torch
import os  # 导入os模块
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split

class GetDataSet(object):
    def __init__(self, dataSetName, isIID):
        self.name = dataSetName
        self.train_data = None
        self.train_label = None
        self.train_data_size = None
        self.data_dir = '/home/LvYongyang/FedLabs/FLddetor/CH-MNIST/data/data'
        self.test_data = None
        self.test_label = None
        self.test_data_size = None

        if self.name == 'chinese-mnist':
            self.chineseMNISTDataSetConstruct(isIID)
        else:
            raise ValueError("Data set not supported")

    def chineseMNISTDataSetConstruct(self, isIID):
        csv_file = '/home/LvYongyang/FedLabs/FLddetor/CH-MNIST/data/chinese_mnist.csv'  # 替换为实际的CSV文件路径
        df = pd.read_csv(csv_file)
        df['image_path'] = self.data_dir + '/' + 'input_' + df['suite_id'].astype(str) + '_' + df['sample_id'].astype(str) + '_' + df['code'].astype(str) + '.jpg'
        
        # 将字符标签转换为数字标签，假设'一'到'亿'分别对应0到14
        character_to_num = {'零': 0, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10, '百': 11, '千': 12, '万': 13, '亿': 14}
        df['numeric_label'] = df['character'].map(character_to_num)
        image_list = []
        for path in df['image_path']:
            with Image.open(path) as img:
                img = img.resize((28, 28))  # 调整图片大小
                img_array = np.array(img)
                image_list.append(img_array)
        
        self.images = np.stack(image_list)  # 将图像列表转换为numpy数组
        
        # 读取和预处理图片
        # image_transform = transforms.Compose([
        #     transforms.Resize((28, 28)),  # 根据需要调整图片大小
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.5], std=[0.5])  # 根据数据集的具体情况调整
        # ])
        
        # 处理图像并将其转换为张量列表
        # image_list = [image_transform(Image.open(path)) for path in df['image_path']]
        # self.images = torch.stack(image_list)

        # 将标签转换为张量
        self.labels = torch.tensor(df['numeric_label'].values, dtype=torch.long).numpy()
        self.labels = dense_to_one_hot(self.labels)
        
        # 划分训练集和测试集
        self.train_images, self.test_images, self.train_labels, self.test_labels = train_test_split(self.images, self.labels, test_size=0.2, random_state=42)
        print(type(self.train_images))
        
        # print(f"标签值: {self.train_labels.unique()}")  # 打印唯一标签值以检查其范围\
        assert self.train_images.shape[0] == self.train_labels.shape[0]
        assert self.test_images.shape[0] == self.test_labels.shape[0]
        self.train_data_size = self.train_images.shape[0]
        self.test_data_size = self.test_images.shape[0]
        # assert self.train_images.shape[3] == 1
        # assert self.test_images.shape[3] == 1
        train_images = self.train_images.reshape(self.train_images.shape[0], self.train_images.shape[1] * self.train_images.shape[2])
        test_images = self.test_images.reshape(self.test_images.shape[0], self.test_images.shape[1] * self.test_images.shape[2])

        train_images = self.train_images.astype(np.float32)
        train_images = np.multiply(train_images, 1.0 / 255.0)
        test_images = self.test_images.astype(np.float32)
        test_images = np.multiply(test_images, 1.0 / 255.0)

        # IID 或非IID 数据的处理
        if isIID:
            order = np.arange(self.train_data_size)
            np.random.shuffle(order)
            self.train_data = train_images[order]
            self.train_label = self.train_labels[order]
        else:
            labels = np.argmax(self.train_labels, axis=1)
            order = np.argsort(labels)
            self.train_data = train_images[order]
            self.train_label = self.train_labels[order]
            
        self.test_data = test_images
        self.test_label = self.test_labels
        # self.display_images(self.train_data, self.train_label, 10)   # 显示前10张图像

    def display_images(self, images, labels, num_to_show):
        """
        显示图像列表中的前num_to_show张图像。
        """
        fig, axes = plt.subplots(num_to_show, 1, figsize=(5, 5 * num_to_show))  # 改为 num_to_show 行1列
        res_dir = 'pic'
        for i, ax in enumerate(axes):  # 这里使用 enumerate(axes) 来迭代每个子图
            img = images[i].reshape((28, 28))  # 将图像数据重新塑形为 28x28 像素
            ax.imshow(img, cmap='gray')  # 显示图像
            ax.set_title(f'Label: {labels[i]}')  # 设置标题
            ax.axis('off')  # 不显示坐标轴
        plt.savefig(os.path.join(res_dir, '1持续.png'))

def dense_to_one_hot(labels_dense, num_classes=15):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

if __name__ == "__main__":
    chinesemnist = GetDataSet('chinese-mnist', True)
    
    if type(chinesemnist.train_data) is np.ndarray and type(chinesemnist.test_data) is np.ndarray and \
            type(chinesemnist.train_label) is np.ndarray and type(chinesemnist.test_label) is np.ndarray:
        print('the type of data is numpy ndarray')
    else:
        print('the type of data is not numpy ndarray')
    print('the shape of the train data set is {}'.format(chinesemnist.train_data.shape))
    print('the shape of the test data set is {}'.format(chinesemnist.test_data.shape))
    print(chinesemnist.train_label[0:100], chinesemnist.train_label[11000:11100])