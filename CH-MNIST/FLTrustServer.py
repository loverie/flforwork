import os
import argparse
import pickle
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from Models import Mnist_2NN, Mnist_CNN,Cifar10_CNN,CHMnistCNN
from clients import ClientsGroup, client
import matplotlib.pyplot as plt
from getData import GetDataSet
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, ConcatDataset, Subset

# Transformations for the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def cos(a,b):
    res = np.sum(a*b.T)/((np.sqrt(np.sum(a * a.T)) + 1e-9) * (np.sqrt(np.sum(b * b.T))) + 1e-9) 
    '''relu'''
    if res < 0:
        res = 0
    return res

def model2vector(model):
    nparr = np.array([])
    vec = []
    for key, var in model.items():
        if key.split('.')[-1] == 'num_batches_tracked' or key.split('.')[-1] == 'running_mean' or key.split('.')[-1] == 'running_var':
            continue
        nplist = var.cpu().numpy()
        nplist = nplist.ravel()
        nparr = np.append(nparr, nplist)
    return nparr

def cosScoreAndClipValue(net1, net2):
    '''net1 -> centre, net2 -> local, net3 -> early model'''
    vector1 = model2vector(net1)
    vector2 = model2vector(net2)
    return cos(vector1, vector2), norm_clip(vector1, vector2)

def norm_clip(nparr1, nparr2):
    '''v -> nparr1, v_clipped -> nparr2'''
    vnum = np.linalg.norm(nparr1, ord=None, axis=None, keepdims=False) + 1e-9
    return vnum / np.linalg.norm(nparr2, ord=None, axis=None, keepdims=False) + 1e-9

def get_weight(update, model):
    '''get the update weight'''
    for key, var in update.items():
        update[key] -= model[key]
    return update

# 攻击函数
# def poison_update(update):
#     print("attack happen here!")
#     '''Simulate a poison update by adding random noise'''
#     for key, var in update.items():
#         noise = torch.randn_like(var) * 0.8
#         update[key] += noise
#     return update

def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def get_client_data(myClients,batch_size):
    # 根据客户端数据集大小加权获取客户端数据集
    total_data_size = sum([myClients.clients_set[client].data_size for client in myClients.clients_set])
    weighted_datasets = []

    # 对客户端进行处理 加权获取客户端的数据集
    for client_name in myClients.clients_set:
        client_data_size = myClients.clients_set[client_name].data_size
        weight = client_data_size / total_data_size
        client_dataset = myClients.clients_set[client_name].train_ds
        weighted_datasets.append(Subset(client_dataset, np.random.choice(len(client_dataset), int(len(client_dataset) * weight), replace=False)))
    
    # 拼接成中央数据集
    central_dataset = ConcatDataset(weighted_datasets)
    central_dataloader = DataLoader(central_dataset, batch_size=batch_size, shuffle=True)
    return central_dataloader

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('-nc', '--num_of_clients', type=int, default=100, help='number of clients')
parser.add_argument('-cf', '--cfraction', type=float, default=0.1, help='C fraction, 0 means 1 client, 1 means total clients')
parser.add_argument('-E', '--epoch', type=int, default=5, help='local train epoch')
parser.add_argument('-B', '--batchsize', type=int, default=128, help='local train batch size')
parser.add_argument('-mn', '--model_name', type=str, default='CHMnistCNN', help='the model to train')
parser.add_argument('-ds', '--dataset', type=str, default='chinese-mnist', help='the name of dataset')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.1, help="learning rate, use value from origin paper as default")
parser.add_argument('-vf', "--val_freq", type=int, default=1, help="model validation frequency(of communications)")
parser.add_argument('-sf', '--save_freq', type=int, default=1000, help='global model save frequency(of communication)')
parser.add_argument('-ncomm', '--num_comm', type=int, default=200, help='number of communications')
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')
parser.add_argument('-atp', '--attack_type', type=int, default=3, help='the  attack type 0-consistent 2-groupappart 3-flag')
parser.add_argument('-att', '--attack_turn', type=int, default=1, help='the num of turns to attack')
parser.add_argument('-atn', '--attack_num', type=int, default=40, help='the num of clients to attack')
parser.add_argument('-gs', '--group_size', type=int, default=5, help='number of attackers in a group')

if __name__ == "__main__":
    args = parser.parse_args()
    args = args.__dict__

    acc_list = []
    test_mkdir(args['save_path'])

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    net = None
    if args['model_name'] == 'CHMnistCNN':
        net = CHMnistCNN()
    else :
        net = Mnist_CNN()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net)
    net = net.to(dev)




    loss_func = F.cross_entropy
    opti = optim.SGD(net.parameters(), lr=args['learning_rate'])






    # 选出的进行comm的客户端集合
    attack_rounds = list(range(0, args['num_comm'], args['attack_turn']))
    # print(attack_rounds)
    #minist/cifar10
    myClients = ClientsGroup('chinese-mnist', args['IID'], args['num_of_clients'], dev,attack_rounds=attack_rounds)
    tempGroup=myClients
    testDataLoader = myClients.test_data_loader
    # for data, label in testDataLoader:
    #     print(label.shape)
    num_in_comm = int(max(args['num_of_clients'] * args['cfraction'], 1))

    global_parameters = {}
    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone()

    max_group_id = args['attack_num'] // args['group_size']
    groups = [[0] * args['group_size'] for _ in range(max_group_id)]

    
    # 分轮次进行训练
    for i in range(args['num_comm']):
        myClients = tempGroup   #更新客户端组s
        print("communicate round {}".format(i + 1))
        groupFlag = [0] * max_group_id
        order = np.random.permutation(args['num_of_clients'])
        
        
        clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]
        # ----------------标签反转攻击-----------------
        if(args['attack_type']==3):
            if (i + 1) in myClients.attack_rounds:
                myClients.flag_attack(args['attack_num'],clients_in_comm=clients_in_comm)
        # ----------------标签反转攻击-----------------
        if(args['attack_type']==2):
            if (i + 1) in myClients.attack_rounds:
                for client in clients_in_comm:
                    myClients.flag_group_attack(args['attack_num'],
                                                client=client,
                                                groups=groups,
                                                group_size=args['group_size'],
                                                groupFlag=groupFlag,
                                                max_group_id=max_group_id,
                                                clients_in_comm=clients_in_comm)
                             
        
        
        print(clients_in_comm)
        sum_parameters = None
        FLTrustTotalScore = 0
        # ---------- 使用中央数据集训练中心模型，并得到其更新权重。 ----------
        FLTrustCentralNorm = myClients.centralTrain(args['epoch'], args['batchsize'], net,
                                                    loss_func, opti, global_parameters)
        '''get the update weight'''
        FLTrustCentralNorm = get_weight(FLTrustCentralNorm, global_parameters)

        # -------------------------FLTrust算法------------------------------
        for client in tqdm(clients_in_comm):
            local_parameters = myClients.clients_set[client].localUpdate(args['epoch'], args['batchsize'], net,loss_func, opti, global_parameters)
            '''get the update weight'''
            local_parameters = get_weight(local_parameters, global_parameters)
    
            #计算cos相似度得分和向量长度裁剪值
            client_score, client_clipped_value =cosScoreAndClipValue(FLTrustCentralNorm, local_parameters)
            FLTrustTotalScore += client_score
            if sum_parameters is None:
                sum_parameters = {}
                for key, var in local_parameters.items():
                    #乘得分 再乘裁剪值
                    sum_parameters[key] = client_score * client_clipped_value * var.clone()
            else:
                for var in sum_parameters:
                    sum_parameters[var] = sum_parameters[var] + client_score * client_clipped_value * local_parameters[var]

        for var in global_parameters:
            #除以所以客户端的信任得分总和   聚合
            global_parameters[var] += sum_parameters[var] / (FLTrustTotalScore + 1e-9)

        with torch.no_grad():
            if (i + 1) % args['val_freq'] == 0:
                net.load_state_dict(global_parameters, strict=True)
                sum_accu = 0
                num = 0
                for data, label in testDataLoader:
                    data, label = data.to(dev), label.to(dev)
                    preds = net(data)
                    preds = torch.argmax(preds, dim=1)
                    sum_accu += (preds == label).float().mean()
                    num += 1
                accuracy = sum_accu / num
                print(f'accuracy: {accuracy}')
                acc_list.append(accuracy.item())

        if (i + 1) % args['save_freq'] == 0:
            torch.save(net.state_dict(), os.path.join(args['save_path'], 
                                                      f'{args["model_name"]}_num_comm{i}_E{args["epoch"]}_B{args["batchsize"]}_lr{args["learning_rate"]}_num_clients{args["num_of_clients"]}_cf{args["cfraction"]}.pth'))

    res_dir = 'res'
    test_mkdir(res_dir)
    with open(os.path.join(res_dir, 'acc_list.pkl'), 'wb') as f:
        pickle.dump(acc_list, f)
    print(acc_list)

    plt.plot(acc_list)
    plt.xlabel('Communication rounds')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy over Communication Rounds')
    plt.savefig(os.path.join(res_dir, 'FLTRUST401.png'))
