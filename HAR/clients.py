import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from getData import GetDataSet
import copy

class client(object):
    def __init__(self, trainDataSet, dev):
        self.train_ds = trainDataSet
        self.dev = dev
        self.train_dl = None
        self.local_parameters = None
        self.data_size = len(trainDataSet)  # 记录数据集大小

    def localUpdate(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters):
        Net.load_state_dict(global_parameters, strict=True)
        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True)
        for epoch in range(localEpoch):
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)  # 假设label已经是long类型
                preds = Net(data)
                loss = lossFun(preds, label)  # 确保preds和label的数据类型一致
                loss.backward()
                opti.step()
                opti.zero_grad()

        return Net.state_dict()


    def local_val(self):
        pass

class ClientsGroup(object):
    def __init__(self, dataSetName, isIID, numOfClients, dev,attack_rounds,weights=None):
        self.data_set_name = dataSetName
        self.is_iid = isIID
        self.num_of_clients = numOfClients
        self.dev = dev
        self.clients_set = {}
        self.central_data = None
        
        self.test_data_loader = None
        self.weights = weights if weights is not None else np.ones(numOfClients) / numOfClients
        self.attack_rounds = attack_rounds if attack_rounds is not None else []
        self.dataSetBalanceAllocation()

    def centralTrain(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters):
        Net.load_state_dict(global_parameters, strict=True)

        for epoch in range(localEpoch):
            for data, label in self.central_data:
                # if data.size(0) != label.size(0):
                    # print(f"Batch size mismatch: data {data.size(0)}, label {label.size(0)}")
                data, label = data.to(self.dev), label.to(self.dev)
                preds = Net(data)
                loss = lossFun(preds, label)
                loss.backward()
                opti.step()
                opti.zero_grad()

        return copy.deepcopy(Net.state_dict())
    
    def dataSetBalanceAllocation(self):
        chineseMNISTDataSet = GetDataSet(self.data_set_name, self.is_iid)

        # test_data = torch.tensor(cifar10DataSet.test_data).permute(0, 2, 3, 1)
        test_data = torch.tensor(chineseMNISTDataSet.test_data)
        test_label = torch.tensor(chineseMNISTDataSet.test_labels)

        # 为中央服务器训练准备数据加载器
        self.central_data = DataLoader(TensorDataset(test_data, test_label), batch_size=100, shuffle=True)
        # 为测试准备数据加载器
        self.test_data_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=100, shuffle=False)
        train_data = torch.tensor(chineseMNISTDataSet.train_data)
        train_label = torch.tensor(chineseMNISTDataSet.train_labels)
        total_data_size = train_data.size(0)
        shard_size = total_data_size //2 // self.num_of_clients

        # # CIFAR-10 数据集的测试数据加载
        # self.test_data_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=100, shuffle=False)
        # train_data = cifar10DataSet.train_data
        # train_label = cifar10DataSet.train_label
        # # print('train_label: ',train_label)
            
        # # 假设 cifar10DataSet.train_data_size 是数据集的实际大小
        # total_data_size = cifar10DataSet.train_data_size

        # # 确保 shard_size 不会导致索引超出数据集大小
        # shard_size = total_data_size // self.num_of_clients//2

        # 生成随机索引前，确保总的 shard_size 不会超出数据集大小
        shards_id = np.random.permutation(total_data_size // shard_size)
        for i in range(self.num_of_clients):
            shards_id1 = shards_id[i * 2]
            shards_id2 = shards_id[i * 2 + 1]
            data_shards1 = train_data[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            data_shards2 = train_data[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            label_shards1 = train_label[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            label_shards2 = train_label[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            local_data, local_label = np.vstack((data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2))
            local_label = np.argmax(local_label, axis=1)
            someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev)
            self.clients_set['client{}'.format(i)] = someone

        # for i in range(self.num_of_clients):
        #     shards_id1 = shards_id[i]
        #     start_idx = shards_id1 * shard_size
        #     end_idx = start_idx + shard_size

        #     # 处理最后一个分片可能超出数据集大小的情况
        #     if end_idx > total_data_size:
        #         end_idx = total_data_size

        #     # 根据计算的索引从数据集中获取数据和标签
        #     data_shards = train_data[start_idx:end_idx]
        #     label_shards = train_label[start_idx:end_idx]
        #     # 检查数据和标签是否不为空
        #     if data_shards.size == 0 or label_shards.size == 0:
        #         raise ValueError(f"No data or labels available for client {i}")

        #     # 转换为 PyTorch 张量
        #     local_data = torch.tensor(data_shards).float()
        #     local_label = torch.tensor(label_shards).long()

        #     # 创建客户端的 TensorDataset
        #     local_dataset = TensorDataset(local_data, local_label)
        #     self.clients_set[f'client{i}'] = client(local_dataset, self.dev)


    def flip_labels(self, labels):
        # 假设标签是0-9的整数，将0变成9，1变成8
        flipped_labels = 9 - labels
        # print('flag_attack happen!')
        return flipped_labels
    
        
    def flag_attack(self,attack_num,clients_in_comm=None):
        for client_name in self.clients_set:
            if client_name in clients_in_comm and int(client_name[6:])<attack_num:
                print(f'flag attack happen on: {client_name}')
                client_dataset = self.clients_set[client_name].train_ds
                data, labels = client_dataset.tensors
                flipped_labels = self.flip_labels(labels)
                self.clients_set[client_name].train_ds = TensorDataset(data, flipped_labels)

    def flag_group_attack(self,attack_num,client,groups,group_size,groupFlag,max_group_id,clients_in_comm=None):
        for client_name in self.clients_set:
            if client_name in clients_in_comm and int(client_name[6:])<attack_num and client==client_name:
                    for g_num in range(max_group_id):
                        for index in range(group_size):
                            if(groupFlag[g_num]==0):
                                # print(f"{g_num*args['group_size']+index},{int(client_name[6:])},{int(client_name[6:])//args['group_size']},{int(client_name[6:])%args['group_size']}")
                                if g_num*group_size+index==int(client_name[6:]) and groups[int(client_name[6:])//group_size][int(client_name[6:])%group_size]==0:
                                        groups[g_num][index]=1
                                        groupFlag[g_num]=1
                                        print(groups)
                                        print(f'flag group attack happen on: {client_name}')
                                        client_dataset = self.clients_set[client_name].train_ds
                                        data, labels = client_dataset.tensors
                                        flipped_labels = self.flip_labels(labels)
                                        self.clients_set[client_name].train_ds = TensorDataset(data, flipped_labels)
                                        if groups[g_num][group_size-1]==1:
                                            groups[g_num][:]=[0]*group_size
            
                                           
if __name__ == "__main__":
    MyClients = ClientsGroup('har', True, 100, 1)
    print(MyClients.clients_set['client10'].train_ds[0:100])
    print(MyClients.clients_set['client11'].train_ds[400:500])
