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

                data, label = data.to(self.dev), label.to(self.dev) 
                preds = Net(data)
                loss = lossFun(preds, label)  # 确保preds和label的数据类型一致
                loss.backward()
                opti.step()
                opti.zero_grad()

        return Net.state_dict()

    def local_val(self):
        pass

class ClientsGroup(object):
    def __init__(self, dataSetName, isIID, numOfClients, dev, attack_rounds, weights=None):
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
                data, label = data.to(self.dev), label.to(self.dev)
                preds = Net(data)
                loss = lossFun(preds, label)
                loss.backward()
                opti.step()
                opti.zero_grad()

        return copy.deepcopy(Net.state_dict())
    
    def dataSetBalanceAllocation(self):
        chineseMNISTDataSet = GetDataSet(self.data_set_name, self.is_iid)

        test_data = torch.tensor(chineseMNISTDataSet.test_data)
        test_label = torch.argmax(torch.tensor(chineseMNISTDataSet.test_label), dim=1)
        if self.central_data is None:
            order = np.arange(test_data.shape[0])
            np.random.shuffle(order)
            self.central_data = DataLoader(TensorDataset(test_data[order[0:100]], test_label[order[0:100]]), batch_size=100, shuffle=True)

        self.test_data_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=100, shuffle=False)
        # 确保 train_data 和 train_label 也是通过类似的方式复制
        train_data = chineseMNISTDataSet.train_data
        train_label = chineseMNISTDataSet.train_label
        # self.display_images(train_data, train_label, 10)

        # Prepare data loaders
        # self.central_data = DataLoader(TensorDataset(test_data, test_label), batch_size=100, shuffle=True)


        # train_data = torch.tensor(chineseMNISTDataSet.train_data, dtype=torch.float32)
        # train_label = torch.tensor(chineseMNISTDataSet.train_labels, dtype=torch.long)

        # Split train data into shards for clients
        shard_size = chineseMNISTDataSet.train_data_size // self.num_of_clients // 2
        shards_id = np.random.permutation(chineseMNISTDataSet.train_data_size // shard_size)
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

    def flip_labels(self, labels):
        # Flip labels assuming they are in range 0-9
        flipped_labels = 14 - labels
        return flipped_labels
    
    def flag_attack(self,attack_num,clients_in_comm=None):
        for client_name in self.clients_set:
            if client_name in clients_in_comm and int(client_name[6:])<attack_num:
                print(f'flag attack happen on: {client_name}')
                client_dataset = self.clients_set[client_name].train_ds
                data, labels = client_dataset.tensors
                flipped_labels = self.flip_labels(labels)
                self.clients_set[client_name].train_ds = TensorDataset(data, flipped_labels)

    def flag_group_attack(self, attack_num, client, groups, group_size, groupFlag, max_group_id, clients_in_comm=None):
        for client_name in self.clients_set:
            if clients_in_comm and int(client_name[6:]) < attack_num and client == client_name:
                for g_num in range(max_group_id):
                    for index in range(group_size):
                        if groupFlag[g_num] == 0:
                            if g_num * group_size + index == int(client_name[6:]) and groups[int(client_name[6:]) // group_size][int(client_name[6:]) % group_size] == 0:
                                groups[g_num][index] = 1
                                groupFlag[g_num] = 1
                                print(groups)
                                print(f'flag group attack happen on: {client_name}')
                                client_dataset = self.clients_set[client_name].train_ds
                                data, labels = client_dataset.tensors
                                flipped_labels = self.flip_labels(labels)
                                self.clients_set[client_name].train_ds = TensorDataset(data, flipped_labels)
                                if groups[g_num][group_size - 1] == 1:
                                    groups[g_num][:] = [0] * group_size

if __name__ == "__main__":
    MyClients = ClientsGroup('chinese-mnist', True, 100, 1, attack_rounds=None)
    print(MyClients.clients_set['client10'].train_ds[:100])
    print(MyClients.clients_set['client11'].train_ds[400:500])
