import os
import copy
import argparse
import pickle
from tqdm import tqdm
import numpy as np
import logging
import time
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
from torch import optim
from sklearn.cluster import KMeans
from Models import Mnist_2NN, Mnist_CNN 
import random
from clients import ClientsGroup  
import matplotlib.pyplot as plt
import math
from sklearn.impute import SimpleImputer
os.makedirs('logs', exist_ok=True)
logging.basicConfig(filename='logs/OUR10攻击.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def cos(a, b):
    # res = np.sum(a*b.T)/((np.sqrt(np.sum(a * a.T)) + 1e-9) * (np.sqrt(np.sum(b * b.T))) + 1e-
    res = (np.dot(a, b) + 1e-9) / (np.linalg.norm(a) + 1e-9) / \
        (np.linalg.norm(b) + 1e-9)
    '''relu'''
    if res < 0:
        res = 0
    return res

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
def evaluate_backdoor(model, test_loader, trigger_pattern):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(dev), target.to(dev)
            # 应用后门触发器
            poisoned_data = trigger_pattern(data)
            output = model(poisoned_data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    accuracy = 100. * correct / total
    print(f'Backdoor attack accuracy: {accuracy}%')
    model.train()
def poison_update(update):
    print("attack happen here!")
    '''Simulate a poison update by adding random noise'''
    for key, var in update.items():
        noise = torch.randn_like(var) * 0.8
        update[key] += noise
    return update

def add_pixel_pattern(images, trigger_size=5, trigger_location=(0, 0)):
    # 将images重塑为四维张量以处理图像，形状为 [batch_size, 1, 28, 28]
    images = images.view(-1, 1, 28, 28)
    
    # 创建后门触发器，这里假设为5x5的全1矩阵
    trigger = torch.ones(1, trigger_size, trigger_size, device=images.device, dtype=images.dtype)
    
    # 初始化一个新的相同形状的张量来存储触发后的图像
    triggered_images = torch.zeros_like(images)
    
    # 为每个图像添加后门触发器到指定位置
    for i in range(images.size(0)):
        # 将后门触发器添加到图像的指定位置
        triggered_images[i, :, trigger_location[0]:trigger_location[0] + trigger_size, 
                                                trigger_location[1]:trigger_location[1] + trigger_size] = \
            images[i, :, trigger_location[0]:trigger_location[0] + trigger_size, 
                    trigger_location[1]:trigger_location[1] + trigger_size] + trigger
    
    # 将四维张量转换回二维张量
    return triggered_images.view(-1, 784)
def add_backdoor_trigger(images, trigger_location=(0, 0)):
    # 确保images是四维的 [batch_size, 1, 28, 28]
    if images.dim() == 2:
        images = images.view(images.size(0), 1, 28, 28)
    # 创建后门触发器，这里假设为5x5的全1矩阵
    trigger = torch.ones(1, 5, 5).to(images.device)
    # 初始化一个新的相同形状的张量来存储触发后的图像
    triggered_images = images.clone()
    
    # 为每个图像添加后门触发器到指定位置
    trigger_size = 5
    for i in range(images.size(0)):
        # 确保触发器不会超出图像边界
        height_start = max(0, trigger_location[0])
        width_start = max(0, trigger_location[1])
        height_end = min(28, trigger_location[0] + trigger_size)
        width_end = min(28, trigger_location[1] + trigger_size)
        
        # 将后门触发器添加到图像的指定位置
        triggered_images[i, :, 
                        height_start:height_end, 
                        width_start:width_end] = trigger

    # 返回触发后的图像，转换回原始形状
    return triggered_images.view(images.size(0), -1)
def detection1(local_updates1, local_updatesarr, update_vectors1, global_vector, clients_in_comm, trust_scores, global_change, update_weights=None):
    
    if update_weights is None:
        update_weights = []

    # Convert local_updatesarr to numpy array
    # update_vectors = np.array(update_vectors1)
    update_vectors = np.array(local_updatesarr)
    
    # Manually assign clusters
    best_k = 2
    num_clients = len(update_vectors)
    if num_clients > 50:
        labels = np.array([0] * 50 + [1] * (num_clients - 50))
    else:
        labels = np.array([0] * num_clients)
    
    # Calculate centroids for the fixed clusters
    first_35_users_centroid = np.mean(update_vectors[:50], axis=0) if num_clients > 50 else np.mean(update_vectors, axis=0)
    other_users_centroid = np.mean(update_vectors[50:], axis=0) if num_clients > 50 else first_35_users_centroid
    centroids = np.array([first_35_users_centroid, other_users_centroid])

    # Compute trust scores for each cluster
    trust_scores_temp = []
    local_updates11 = np.array(local_updatesarr)
    global_change1 = np.array(global_change)
    for i in range(best_k):
        update_vectors_cluster = update_vectors[labels == i]
        if update_vectors_cluster.size > 0:
            centroid = centroids[i]
            sim = cos(centroid, global_change1)
            trust_score = sim
            trust_scores_temp.append((trust_score, i))

    print('各个簇的分数：', trust_scores_temp)
    trust_scores_temp.sort(key=lambda x: x[0], reverse=True)
    if(trust_scores_temp[0][0] > 0):
        clusterIndexForTrust = trust_scores_temp[0][1]
    else:
        clusterIndexForTrust = -1
    print('clusterIndexForTrust', clusterIndexForTrust)
    
    if clusterIndexForTrust != -1:
        high_confidence_users = np.where(labels == clusterIndexForTrust)[0]
        print("High confidence users0:", high_confidence_users)
        num_users_to_select = min(10, len(high_confidence_users))
        random_users_in_trusted_cluster = random.sample(list(high_confidence_users), num_users_to_select)
    else:
        random_users_in_trusted_cluster = []
    
    print("Randomly selected users from the most trusted cluster:", random_users_in_trusted_cluster)
    
    # Calculate user confidence scores
    user_confidence_scores = np.zeros(len(update_vectors))
    if clusterIndexForTrust != -1:
        centroid = centroids[clusterIndexForTrust]
        for user_idx in random_users_in_trusted_cluster:
            # print(user_idx)
            # print('用户：',local_updates11[user_idx])
            # print('变化',(local_updates11[user_idx] - global_change1))
            distance = cos(local_updates11[user_idx], centroid)
            # print('分数：', distance)
            # logging.info(f"用户: {local_updates11[user_idx]}")
            trust_score_value = trust_scores[user_idx]
            user_confidence_scores[user_idx] = (((0.125 * trust_score_value) + (distance * 0.875)))
            # if(user_confidence_scores[user_idx] <= 0.125):
            #     user_confidence_scores[user_idx] = 0
            trust_scores[user_idx] = user_confidence_scores[user_idx]
            # print('权重分数',trust_scores[user_idx] )

    # Print cluster statistics
    unique_labels, cluster_counts = np.unique(labels, return_counts=True)
    for cluster_id, count in zip(unique_labels, cluster_counts):
        print(f"Cluster {cluster_id} has {count} users")

    print('信任分数： ', trust_scores)

    # Select updates
    selected_updates = [local_updates1[i] for i in random_users_in_trusted_cluster]
    selected_user_scores = user_confidence_scores[random_users_in_trusted_cluster]
    selected_user_weights = selected_user_scores / np.sum(selected_user_scores)
    update_weights.append(selected_user_weights)
    update_weights = np.array(update_weights).flatten()

    return best_k, selected_updates, user_confidence_scores, update_weights



parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedDekrum")
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('-nc', '--num_of_clients', type=int, default=100, help='number of the clients')
parser.add_argument('-cf', '--cfraction', type=float, default=0.5, help='C fraction, 0 means 1 client, 1 means total clients')
parser.add_argument('-E', '--epoch', type=int, default=5, help='local train epoch')
parser.add_argument('-B', '--batchsize', type=int, default=128, help='local train batch size')
parser.add_argument('-mn', '--model_name', type=str, default='mnist_cnn', help='the model to train')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.05, help="learning rate, use value from origin paper as default")
parser.add_argument('-vf', "--val_freq", type=int, default=1, help="model validation frequency(of communications)")
parser.add_argument('-sf', '--save_freq', type=int, default=1000, help='global model save frequency(of communication)')
parser.add_argument('-ncomm', '--num_comm', type=int, default=200, help='number of communications')
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')
parser.add_argument('-atp', '--attack_type', type=int, default=3, help='the types to attack')
parser.add_argument('-att', '--attack_turn', type=int, default=1, help='the turns to attack')
parser.add_argument('-atn', '--attack_num', type=int, default=50, help='the num to attack')
parser.add_argument('-gs', '--group_size', type=int, default=10, help='number of attackers in a group')
parser.add_argument('-ab', '--activate_backdoor', type=int, default=1, help='yes/no backdoor')

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

    # 选出的进行comm的客户端集合
    attack_rounds = list(range(0, args['num_comm'], args['attack_turn']))
    # print(attack_rounds)
    myClients = ClientsGroup('mnist', args['IID'], args['num_of_clients'], dev,attack_rounds=attack_rounds)
    tempGroup = myClients
    testDataLoader = myClients.test_data_loader
    num_in_comm = int(max(args['num_of_clients'] * 1, 1))
    num_in_comm1 = int(max(args['num_of_clients'] * args['cfraction'], 1))
    print(num_in_comm1)

    # 获取上一轮的模型参数
    global_parameters = {}
    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone()

    # 初始化信任分数矩阵
    trust_scores=[]
    trust_scores= np.ones(args['num_of_clients']) * 0

    max_group_id = args['attack_num'] // args['group_size']
    groups= [[0]*args['group_size'] for _ in range(max_group_id)]
    for i in range(args['num_comm']):
        start_time = time.time()
        myClients = tempGroup
        groupFlag=[0]*max_group_id
        print("communication round {}".format(i + 1))
        local_updates = [] 
        local_updatesarr = []
        local_arrays = []
        update_weights = []
        # 随机选取用户
        order = np.random.permutation(args['num_of_clients'])
        clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]
        clients_in_comm1 = ['client{}'.format(i) for i in range(num_in_comm1)]
        # myClients.inject_backdoor(clients_in_comm1, add_pixel_pattern)
        print(clients_in_comm1)

        # global_model_parameters = get_weight(global_model_parameters, global_parameters)
        # ----------------持续/轮次标签反转攻击-----------------
        if(args['attack_type']==3):
            if (i + 1) in myClients.attack_rounds:
                trigger = net.backdoor_trigger
        # 应用后门触发器模式
                # myClients.inject_backdoor(clients_in_comm1, lambda images: add_pixel_pattern(images, trigger))d
                myClients.inject_backdoor(clients_in_comm1, add_backdoor_trigger)
                # myClients.flag_attack(args['attack_num'],clients_in_comm=clients_in_comm1)
        # ----------------分组标签反转攻击-----------------
        if(args['attack_type']==4):
            if (i + 1) in myClients.attack_rounds:
                for client in clients_in_comm:
                    myClients.flag_group_attack(args['attack_num'],  client=client,  groups=groups,   group_size=args['group_size'],   groupFlag=groupFlag,  max_group_id=max_group_id, clients_in_comm=clients_in_comm1)
         # 中央服务器训练
        global_model_parameters = myClients.centralTrain(args['epoch'], args['batchsize'], net, loss_func, opti, global_parameters,args['activate_backdoor'])
        global_updates = model2vector(global_model_parameters)
        global_change = copy.deepcopy(model2vector(get_weight(global_model_parameters, global_parameters)))
        for client in tqdm(clients_in_comm):
            # 本地客户端训练
            local_parameters = myClients.clients_set[client].localUpdate(args['epoch'], args['batchsize'], net, loss_func, opti, global_parameters, args['activate_backdoor'])
            # 获取本地模型与全局模型的变化量 local_update
            local_array = copy.deepcopy(model2vector(local_parameters))
            local_arrays.append(local_array)

            local_update = get_weight(local_parameters, global_parameters)
            # 将字典添加到列表中
            temp_update = copy.deepcopy(local_parameters)
            temp_updates = copy.deepcopy(model2vector(local_update))
            local_updatesarr.append(temp_updates)
            local_updates.append(temp_update)  # 添加副本到列表

        # for idex,client_update in enumerate(local_updates):
        #     print('0 :',model2vector(client_update))\
        best_k, selected_updates, user_confidence_scores, update_weights= detection1(local_updates, local_updatesarr,local_arrays, global_updates,clients_in_comm ,trust_scores,global_change,update_weights )

        # print(f"Best number of clusters: {best_k}, trust_scores: {trust_scores}")
        # 初始化聚合更新字典
        aggregated_update = {key: torch.zeros_like(param).to(dev) for key, param in global_parameters.items()}

        for index, client_update in enumerate(selected_updates):
            # print(index,model2vector(client_update))
            for var in aggregated_update:
                if var in client_update:
                    client_update_tensor = client_update[var].clone().detach().to(dev)
                    aggregated_update[var] += (client_update_tensor * update_weights[index])
        evaluate_backdoor(net, testDataLoader, add_pixel_pattern)
        # 使用SGD更新全局参数
        if any(torch.norm(param) > 0 for param in aggregated_update.values()):
            for var in global_parameters:
                # global_parameters[var] -= args['learning_rate'] * aggregated_update[var]
                global_parameters[var] +=  aggregated_update[var]
        else :
            for var in global_parameters:
                # global_parameters[var] -= args['learning_rate'] * aggregated_update[var]
                global_parameters[var] =  global_parameters[var]

        with torch.no_grad():
            if (i + 1) % args['val_freq'] == 0:
                net.load_state_dict(global_parameters, strict=True)
                sum_accu = 0
                num = 0
                for data, label in testDataLoader:
                    data, label = data.to(dev), label.to(dev)
                    # preds = net(data)
                    preds = net(data, args['activate_backdoor'])
                    preds = torch.argmax(preds, dim=1)
                    sum_accu += (preds == label).float().mean()
                    num += 1
                accuracy = sum_accu / num
                end_time = time.time()
                duration = end_time - start_time
                logging.info(f"K-Fedtrust round {i+1} completed in {duration:.2f} seconds. Validation accuracy: {accuracy:.4f}")
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
    plt.savefig(os.path.join(res_dir, 'OUR11持续.png'))