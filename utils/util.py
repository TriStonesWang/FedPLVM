import numpy as np
import copy
import torch
from sklearn.cluster import KMeans
from utils.finch import FINCH


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Method   : {args.method}\n')
    print(f'    Model     : {args.model}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.rounds}\n')
    print(f'    Clients   : {args.num_clients}\n')

    print('    Federated parameters:')
    if args.label_iid:
        print('   Label IID')
    else:
        print('   Label Non-IID')
    if args.feature_iid:
        print('   Feature IID')
    else:
        print('   Feature Non-IID')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.train_ep}\n')
    return


def average_protos(protos):
    """
    Average the protos for each local user
    """
    agg_protos = {}
    for [label, proto_list] in protos.items():
        proto = np.stack(proto_list)
        agg_protos[label] = np.mean(proto, axis=0)

    return agg_protos


def cluster_protos(protos, num_cluster=3):
    cluster_centers_label = {}
    for [label, proto_list] in protos.items():
        proto = np.stack(proto_list)
        kmeans = KMeans(n_clusters=num_cluster, random_state=0, n_init="auto").fit(proto)
        cluster_centers_label[label] = kmeans.cluster_centers_
    return cluster_centers_label


def cluster_protos_finch(protos_label_dict):
    agg_protos = {}
    num_p = 0
    for [label, proto_list] in protos_label_dict.items():
        proto_list = np.stack(proto_list)
        c, num_clust, req_c = FINCH(proto_list, initial_rank=None, req_clust=None, distance='cosine',
                                    ensure_early_exit=False, verbose=False)
        num_protos, num_partition = c.shape
        class_cluster_list = []
        for idx in range(num_protos):
            class_cluster_list.append(c[idx, -1])
        class_cluster_array = np.array(class_cluster_list)
        uniqure_cluster = np.unique(class_cluster_array).tolist()
        agg_selected_proto = []

        for _, cluster_index in enumerate(uniqure_cluster):
            selected_array = np.where(class_cluster_array == cluster_index)
            selected_proto_list = proto_list[selected_array]
            cluster_proto_center = np.mean(selected_proto_list, axis=0)
            agg_selected_proto.append(cluster_proto_center)

        agg_protos[label] = agg_selected_proto
        num_p += num_clust[-1]
    return agg_protos, num_p / len(protos_label_dict)


def local_cluster_collect(local_cluster_protos):
    global_collected_protos = {}
    for [idx, cluster_protos_label] in local_cluster_protos.items():
        for [label, cluster_protos_list] in cluster_protos_label.items():
            for i in range(len(cluster_protos_list)):
                if label in global_collected_protos.keys():
                    global_collected_protos[label].append(cluster_protos_list[i])
                else:
                    global_collected_protos[label] = [cluster_protos_list[i]]
    return global_collected_protos


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w)
    for key in w[0].keys():
        for i in range(1, len(w)):
            w_avg[0][key] += w[i][key]
        w_avg[0][key] = torch.true_divide(w_avg[0][key], len(w))
        for i in range(1, len(w)):
            w_avg[i][key] = w_avg[0][key]
    return w_avg


def average_weights_noniid(w, num_list):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w)
    num_sum = sum(num_list)
    for key in w[0].keys():
        w_avg[0][key] = num_list[0] * w[0][key]
        for i in range(1, len(w)):
            w_avg[0][key] += num_list[i] * w[i][key]
        w_avg[0][key] = torch.true_divide(w_avg[0][key], num_sum)
        for i in range(1, len(w)):
            w_avg[i][key] = w_avg[0][key]
    return w_avg


def proto_aggregation(local_protos_list):
    agg_protos_label = dict()
    for idx in local_protos_list:
        local_protos = local_protos_list[idx]
        for label in local_protos.keys():

            if label in agg_protos_label:
                agg_protos_label[label].append(local_protos[label])
            else:
                agg_protos_label[label] = [local_protos[label]]

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0]
            for i in proto_list:
                proto += i
            agg_protos_label[label] = proto / len(proto_list)
        else:
            agg_protos_label[label] = proto_list[0]
    return agg_protos_label


def proto_aggregation_cluster(global_protos_list):
    agg_protos_label = dict()
    for label in global_protos_list.keys():
        for i in range(len(global_protos_list[label])):
            if label in agg_protos_label:
                agg_protos_label[label].append(global_protos_list[label][i])
            else:
                agg_protos_label[label] = [global_protos_list[label][i]]

    for [label, proto_list] in agg_protos_label.items():
        # print(len(proto_list))
        if len(proto_list) > 1:
            proto = 0 * proto_list[0]
            for i in proto_list:
                proto += i
            agg_protos_label[label] = proto / len(proto_list)
        else:
            agg_protos_label[label] = proto_list[0]

    return agg_protos_label
