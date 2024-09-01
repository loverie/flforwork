import os
import argparse
import pickle
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA
import warnings
import copy
import torch
import torch.nn.functional as F
from torch import optim
from sklearn.cluster import KMeans
from kmeans_pytorch import kmeans
from collections import defaultdict
from Models import Mnist_2NN, Mnist_CNN  # Assuming these are your custom model classes
from clients import ClientsGroup  # Assuming this is related to client operations
import matplotlib.pyplot as plt
import math
from sklearn.impute import SimpleImputer

def cos(a, b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0
    return np.dot(a, b) / (a_norm * b_norm)

def model2vector(model):
    nparr = np.array([])
    for key, var in model.items():
        if key.split('.')[-1] == 'num_batches_tracked' or key.split('.')[-1] == 'running_mean' or key.split('.')[-1] == 'running_var':
            continue
        nplist = var.cpu().numpy()
        nplist = nplist.ravel()
        nparr = np.append(nparr, nplist)
    return nparr

def get_weight(update, model):
    for key, var in update.items():
        update[key] -= model[key]
    return update

def flip_labels(labels):
    return (labels + 1) % 10

def poison_update(update, flip=True):
    for key, var in update.items():
        if flip:
            if 'label' in key:
                var = torch.tensor(flip_labels(var.numpy()))
            else:
                noise = torch.randn_like(var) * 10
                var += noise
        update[key] = var
    return update

def krum(selected_updates, model_params_dict, net, lr, trust_scores,weights, b=1):
    num_clients = len(selected_updates)
    
    if num_clients <= 2 * b:
        raise ValueError("Not enough clients to perform Krum aggregation.")
    
    print(f"Update weights: {weights[-1]}")

    # 根据信任分数计算权重
    # client_confidence_scores = [score for _, score in trust_scores]
    # eps = 1e-10
    # weights = [1.0 / (trust_scores[i] + eps) for i in range(num_clients)]
    # total_weight = sum(weights)
    # weights = [weight / total_weight for weight in weights]

    # 初始化聚合梯度字典为零张量
    # aggregated_grad = {param_name: torch.zeros_like(param_tensor) for param_name, param_tensor in model_params_dict.items()}
    aggregated_grad = {k: torch.zeros_like(v) for k, v in model_params_dict.items()}

    # 聚合选定客户端的更新
    for idx in range(num_clients):
        client_update_dict = selected_updates[idx]  # 假设这是一个字典
        print('client_update_dict:' ,client_update_dict)
        for param_name, param_tensor in aggregated_grad.items():
            if param_name in client_update_dict:
                # 获取对应参数的更新向量
                client_update_vector = client_update_dict[param_name]
                # 确保更新向量是一维的
                client_update_vector = torch.from_numpy(client_update_vector).view(-1)
                print('client_update_vector:' ,client_update_vector)
                # 聚合更新，考虑权重
                aggregated_grad[param_name] += client_update_vector * weights[idx]

    return aggregated_grad
        # 应用聚合梯度更新模型参数
    # with torch.no_grad():
    #     for param_name, param_tensor in net.state_dict().items():
    #         if param_name in aggregated_grad:
    #             # 直接用聚合梯度更新模型参数
    #             net.state_dict()[param_name].copy_(lr * aggregated_grad[param_name])
    #             net.state_dict()[param_name].sub_(lr * aggregated_grad[param_name])

    # # 应用聚合梯度更新模型参数
    # with torch.no_grad():
    #     for param_name, param_tensor in net.state_dict().items():
    #         # 假设每个模型参数有相同的形状，这里需要根据实际情况调整
    #         if param_name in model_params_dict:
    #             param_index = list(model_params_dict.keys()).index(param_name)
    #             # 更新模型参数，这里假设 param_tensor 是一个一维张量
    #             net.state_dict()[param_name].copy_(torch.from_numpy(aggregated_grad))

    # return aggregated_grad



parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedDekrum")
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('-nc', '--num_of_clients', type=int, default=100, help='number of the clients')
parser.add_argument('-cf', '--cfraction', type=float, default=0.1, help='C fraction, 0 means 1 client, 1 means total clients')
parser.add_argument('-E', '--epoch', type=int, default=5, help='local train epoch')
parser.add_argument('-B', '--batchsize', type=int, default=128, help='local train batch size')
parser.add_argument('-mn', '--model_name', type=str, default='mnist_cnn', help='the model to train')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate, use value from origin paper as default")
parser.add_argument('-vf', "--val_freq", type=int, default=1, help="model validation frequency(of communications)")
parser.add_argument('-sf', '--save_freq', type=int, default=50, help='global model save frequency(of communication)')
parser.add_argument('-ncomm', '--num_comm', type=int, default=60, help='number of communications')
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')
parser.add_argument('-atp', '--attack_type', type=int, default=3, help='the turns to attack')
parser.add_argument('-att', '--attack_turn', type=int, default=0, help='the turns to attack')
parser.add_argument('-atn', '--attack_num', type=int, default=0, help='the num to attack')
parser.add_argument('-gs', '--group_size', type=int, default=0, help='number of attackers in a group')


def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


if __name__ == "__main__":
    args = parser.parse_args()
    args = args.__dict__

    acc_list = []
    test_mkdir(args['save_path'])

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    net = None
    if args['model_name'] == 'mnist_2nn':
        net = Mnist_2NN()
    elif args['model_name'] == 'mnist_cnn':
        net = Mnist_CNN()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net)
    net = net.to(dev)

    loss_func = F.cross_entropy
    opti = optim.SGD(net.parameters(), lr=args['learning_rate'])

    myClients = ClientsGroup('mnist', args['IID'], args['num_of_clients'], dev)
    testDataLoader = myClients.test_data_loader

    num_in_comm = int(max(args['num_of_clients'] * args['cfraction'], 1))

    global_parameters = {}
    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone()
    trust_scores=[]
    trust_scores= np.ones((args['num_of_clients'],)) *1
    for i in range(args['num_comm']):
        print("communication round {}".format(i + 1))
        local_updates = []
        local_arrays = []
        update_weights = []
        order = np.random.permutation(args['num_of_clients'])
        clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]

        global_model_parameters = myClients.centralTrain(args['epoch'], args['batchsize'], net, loss_func, opti, global_parameters)
        global_model_parameters = get_weight(global_model_parameters, global_parameters)

        for client_name in tqdm(clients_in_comm):

            local_parameters = myClients.clients_set[client_name].localUpdate(args['epoch'], args['batchsize'], net, loss_func, opti, global_parameters)
            local_update = get_weight(local_parameters, global_parameters)

            # if (i + 1) % args['attack_turn'] == 0:
            #     if args['attack_type'] == 0:
            #         if clients_in_comm.index(client_name) < args['attack_num']:
            #             # local_update = poison_update(local_update, flip=True)
            #             local_update = poison_update(local_update)
            #     elif args['attack_type'] == 1:
            #         if clients_in_comm.index(client_name) % args['group_size'] == 0:
            #             local_update = poison_update(local_update, flip=True)
            temp_update = copy.deepcopy(local_parameters)
            local_updates.append(temp_update)
            local_array = model2vector(local_update)
            local_arrays.append(local_array)

        # param_list_tensors = [torch.cat([v.view(-1) for v in update.values()]) for update in local_updates]
        # print(local_updates)
        
        # # 调用detection1，获取更新向量和信任分数
        # print(int(clients_in_comm[0][6:]))
        best_k, selected_updates, user_confidence_scores, update_weights= detection1(local_updates, local_arrays, global_model_parameters,clients_in_comm ,trust_scores,update_weights )

        print(f"Best number of clusters: {best_k}, Trust Scores: {trust_scores}")

        # model_parameters_list = [param.clone() for param in net.parameters()]
        # model_params_dict = {param_name: param_tensor for param_name, param_tensor in zip(net.state_dict().keys(), model_parameters_list)}
        # 初始化聚合更新字典

        # 初始化聚合更新字典
        aggregated_update = {key: torch.zeros_like(param).to(dev) for key, param in global_parameters.items()}

        for index, client_update in enumerate(selected_updates):
            for var in aggregated_update:
                if var in client_update:
                    # 逐元素相乘并累加到 aggregated_update[key]
                    client_update_tensor = torch.tensor(client_update[var]).clone().detach().to(dev)
                    aggregated_update[var] +=client_update_tensor * update_weights[index]
                    print(update_weights[index])

        for var in global_parameters:
            global_parameters[var] = aggregated_update[var]

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
    plt.savefig(os.path.join(res_dir, 'accuracy_plot1.png'))