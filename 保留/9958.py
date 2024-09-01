'''
任务：
1.聚合算法(记得修改 KRUM)
2.kmeans
3.实验结果: 先跑个50轮看看效果
目前代码中存在的问题: .·´¯(>▂<)´¯(>▂<)´¯(>▂<)´¯·.
1.local_updates.append之后每轮的参数梯度变为一个(7.15号前解决),包括krum实现问题[目前不知怎么解决(T_T)] ,正在尝试ing....¯(>▂<)¯(7.14尝试失败)(7.15已解决)
2.kmeans算法聚类问题(7.16号前解决),已解决99.99%
3.kmeans算法计算慢
4.攻击问题 soluting......
'''
import os
import copy
import argparse
import pickle
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
from torch import optim
from sklearn.cluster import KMeans
# from kmeans_pytorch import kmeans
# from collections import defaultdict
from Models import Mnist_2NN, Mnist_CNN 
from clients import ClientsGroup  
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

# def poison_update(update, flip=True):
#     for key, var in update.items():
#         if flip:
#             if 'label' in key:
#                 var = torch.tensor(flip_labels(var.numpy()))
#             else:
#                 noise = torch.randn_like(var) * 10
#                 var += noise
#         update[key] = var
#     return update

def poison_update(update):
    print("attack happen here!")
    '''Simulate a poison update by adding random noise'''
    for key, var in update.items():
        noise = torch.randn_like(var) * 1.5
        update[key] += noise
    return update

def detection1(local_updates1, update_vectors, global_model_parameters, clients_in_comm, trust_scores, update_weights=None):
    if update_weights is None:
        update_weights = []
    # 数据预处理
    global_vector = model2vector(global_model_parameters)
    update_vectors = np.array(update_vectors)
    imputer = SimpleImputer(strategy='mean')
    update_vectors = imputer.fit_transform(update_vectors)
    # PCA降维
    # pca = PCA(n_components= 10)
    # pca.fit(update_vectors)
    # update_vectors = pca.transform(update_vectors)
    # global_vector = pca.transform(global_vector.reshape(1, -1)).flatten() 
    # print(global_vector)
    # print('update_vectors: ', update_vectors)
    # K-means聚合, gap统计
    num_clusters_range = range(1, 4)
    best_gap = 0
    best_k = 1
    nrefs = 10
    gaps = np.zeros(len(num_clusters_range))
    gapDiff = np.zeros(len(num_clusters_range) - 1)
    sdk = np.zeros(len(num_clusters_range))

    for i, k in enumerate(num_clusters_range):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(update_vectors)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        # gap检验
        # Wk = np.sum([np.linalg.norm(u - centroids[l]) for u, l in zip(update_vectors, labels) if l < k])
        Wk = np.sum([np.square(u - centroids[l]) for u, l in zip(update_vectors, labels) if l < k])
        WkRef_sum = 0
        for _ in range(nrefs):
            rand_data = np.random.rand(*update_vectors.shape)
            rand_data = (rand_data - np.min(rand_data)) / (np.max(rand_data) - np.min(rand_data))  # 归一化随机数据集
            ref_kmeans = KMeans(n_clusters=k, random_state=42)
            ref_kmeans.fit(rand_data)
            ref_labels = ref_kmeans.labels_
            ref_centroids = ref_kmeans.cluster_centers_
            # WkRef_sum += np.sum([np.linalg.norm(r - ref_centroids[ref_labels[m]]) for m, r in enumerate(rand_data) if ref_labels[m] < k])
            WkRef_sum += np.sum([np.square(r - ref_centroids[ref_labels[m]]) for m, r in enumerate(rand_data) if ref_labels[m] < k])
        
        WkRef_avg = WkRef_sum / nrefs
        gaps[i] = np.log(WkRef_avg) - np.log(Wk / len(update_vectors)) if Wk else float('inf')
        sdk[i] = np.sqrt((1.0 + nrefs) / nrefs) * np.std(np.log(WkRef_sum / nrefs))

        if i > 0:
            gapDiff[i - 1] = gaps[i - 1] - gaps[i] + sdk[i]
        if i == 0:
            best_gap = gaps[i]
            best_k = k
        elif gaps[i] + sdk[i] > best_gap:
            best_gap = gaps[i] + sdk[i]
            best_k = k
        # WkRef_avg = WkRef_sum / nrefs
        # gap = np.log(WkRef_avg) - np.log(Wk / len(update_vectors)) if Wk else float('inf')

        # if gap > best_gap:
        #     best_gap = gap
        #     best_k = k

    print(f"Best k: {best_k}, best Gap: {best_gap}")

    kmeans_best = KMeans(n_clusters=best_k, init='k-means++', random_state=42)
    kmeans_best.fit(update_vectors)
    labels = kmeans_best.labels_
    centroids = kmeans_best.cluster_centers_
    # 计算簇的 cos分数
    trust_scores_temp = []
    for i in range(best_k):
        update_vectors_cluster = update_vectors[labels == i]
        if update_vectors_cluster.size > 0:
            centroid = update_vectors_cluster.mean(axis=0)
            sim = cos(centroid, global_vector)
            trust_score = 1 + sim
            trust_scores_temp.append((trust_score, i))

    trust_scores_temp.sort(key=lambda x: x[0], reverse=True)    
    print('trust_scores_temp : ', trust_scores_temp)
    
    valid_clusters = [cluster_idx for _, (trust_score, cluster_idx) in enumerate(trust_scores_temp) if trust_score > 1]
    valid_user_indices = np.isin(labels, valid_clusters)
    print('valid_user_indices :', valid_user_indices)
    
    # 计算用户分数
    user_confidence_scores = np.zeros(len(update_vectors))
    for idx, centroid in enumerate(centroids):
        cluster_indices = np.where(labels == idx)[0]
        for user_idx in cluster_indices:
            distance = cos(update_vectors[user_idx], centroid)
            trust_score_value = trust_scores[int(clients_in_comm[user_idx][6:])]
            a = math.exp(-(trust_score_value ** 2))
            # 计算用户置信分数，并乘以用户所在簇的分数
            user_confidence_scores[user_idx] = ((1 / (1 + a)) + distance)* trust_scores_temp[idx][0]
            if user_confidence_scores[user_idx] < 0:
                user_confidence_scores[user_idx] = 0
            trust_scores[int(clients_in_comm[user_idx][6:])] = user_confidence_scores[user_idx]
    
    # 统计每个簇中的用户个数并输出簇编号
    unique_labels, cluster_counts = np.unique(labels, return_counts=True)
    for cluster_id, count in zip(unique_labels, cluster_counts):
        print(f"Cluster {cluster_id} has {count} users")

    # 选择簇分数大于1，用户分数大于1的用户
    learning_indices = np.where(user_confidence_scores[valid_user_indices] >= 1)[0]
    selected_user_scores = user_confidence_scores[valid_user_indices][learning_indices]
    selected_user_indices = np.argsort(-selected_user_scores)[:len(learning_indices)]

    selected_updates = [local_updates1[i] for i in selected_user_indices]
    selected_user_weights = selected_user_scores / np.sum(selected_user_scores)
    update_weights.append(selected_user_weights)
    update_weights = np.array(update_weights).flatten()  
    # update_weights = np.array(update_weights).flatten()

    print(f"Selected user scores: {selected_user_scores}")
    print(f"Selected user weights: {update_weights}")

    return best_k, selected_updates, user_confidence_scores, update_weights

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
parser.add_argument('-ncomm', '--num_comm', type=int, default=200, help='number of communications')
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')
parser.add_argument('-atp', '--attack_type', type=int, default=10, help='the type to attack')
parser.add_argument('-att', '--attack_turn', type=int, default=10, help='the turns to attack')
parser.add_argument('-atn', '--attack_num', type=int, default=40, help='the num to attack')
parser.add_argument('-gs', '--group_size', type=int, default=10, help='number of attackers in a group')

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
    myClients = ClientsGroup('mnist', args['IID'], args['num_of_clients'], dev)
    testDataLoader = myClients.test_data_loader

    num_in_comm = int(max(args['num_of_clients'] * args['cfraction'], 1))

    global_parameters = {}
    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone()
    trust_scores=[]
    trust_scores= np.ones((args['num_of_clients'],)) *1
    max_group_id = args['attack_num'] // args['group_size']
    groups= [[0]*args['group_size'] for _ in range(max_group_id)]
    for i in range(args['num_comm']):
        groupFlag=[0]*max_group_id
        print("communication round {}".format(i + 1))
        local_updates = [] 
        local_arrays = []
        update_weights = []
        order = np.random.permutation(args['num_of_clients'])
        clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]

        global_model_parameters = myClients.centralTrain(args['epoch'], args['batchsize'], net, loss_func, opti, global_parameters)
        global_model_parameters = get_weight(global_model_parameters, global_parameters)
        # ----------------标签反转攻击-----------------
        # if(args['attack_type']==3):
        #     if (i + 1) in myClients.attack_rounds:
        #         print(f"Attack round: {i + 1}")
        #         myClients.flag_attack()
        #         print('I was attacked')
        # ----------------标签反转攻击-----------------
        for client_name in tqdm(clients_in_comm):
            # local_parameters = myClients.clients_set[client_name].localUpdate(args['epoch'], args['batchsize'], net, loss_func, opti, global_parameters)
            # print( '攻击之前的', model2vector(local_parameters))
            # print('1 : ' , model2vector(local_update))
            local_parameters = myClients.clients_set[client_name].localUpdate(args['epoch'], args['batchsize'], net, loss_func, opti, global_parameters)

            if (i + 1) % args['attack_turn'] == 0:   
                #type表示间隔攻击,随机找attack_num个用户投毒    
                if(args['attack_type']==0):
                    if  int(client_name[6:]) <= args['attack_num']:
                        # myClients.flag_attack()
                        local_parameters = poison_update(local_parameters)
                        # print('I was attacked')
                        # print( '攻击之后的', model2vector(local_parameters))
                # type=1表示分组投毒
                elif args['attack_type']==1 and int(client_name[6:])<=args['attack_num']:
                    # print(f"抽取出来的用户id:{int(client_name[6])}")
                    for g_num in range(max_group_id):
                        for index in range(args['group_size']):
                            if(groupFlag[g_num]==0):
                            # print(f"{g_num*args['group_size']+index},{int(client_name[6:])},{int(client_name[6:])//args['group_size']},{int(client_name[6:])%args['group_size']}")
                                if g_num*args['group_size']+index==int(client_name[6:]) and groups[int(client_name[6:])//args['group_size']][int(client_name[6:])%args['group_size']]==0:
                                    groups[g_num][index]=1
                                    groupFlag[g_num]=1
                                    # myClients.flag_attack()
                                    local_parameters = poison_update(local_parameters)
                                    # print('I was attacked')
                                    if groups[g_num][args['group_size']-1]==1:
                                        groups[g_num][:]=[0]*args['group_size']

            # 将 local_update 添加到当前轮的更新列表中
            # print('open :'  , model2vector(local_parameters))
            local_update = get_weight(local_parameters, global_parameters)
            # 将字典添加到列表中
            # temp_update = local_update.copy()  # 使用副本进行测试
            temp_update = copy.deepcopy(local_update)
            local_updates.append(temp_update)  # 添加副本到列表

            local_array = copy.deepcopy(model2vector(local_update))
            local_arrays.append(local_array)

        best_k, selected_updates, user_confidence_scores, update_weights= detection1(local_updates, local_arrays, global_model_parameters,clients_in_comm ,trust_scores,update_weights )

        print(f"Best number of clusters: {best_k}, trust_scores: {trust_scores}")
        
        # 聚合函数
        # 初始化聚合更新字典
        aggregated_update = {key: torch.zeros_like(param).to(dev) for key, param in global_parameters.items()}

        for index, client_update in enumerate(selected_updates):
            # print(index,model2vector(client_update))
            for var in aggregated_update:
                if var in client_update:
                    client_update_tensor = client_update[var].clone().detach().to(dev)
                    # 应用加权更新
                    aggregated_update[var] += (client_update_tensor * update_weights[index])

        # 使用SGD更新全局参数
        for var in global_parameters:
            # global_parameters[var] += args['learning_rate'] * aggregated_update[var]
            global_parameters[var] +=  aggregated_update[var]

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