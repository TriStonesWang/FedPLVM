from torchvision import datasets, transforms
import numpy as np
import torch
import seaborn as sns
import pandas as pd

from utils.dataset import DigitsDataset, OfficeDataset, DomainNetDataset

data_root = './data/'

def visualize_data_dist_fnln(args, df, K, num_users):
    for k in range(K - 1, 0, -1):
        for j in range(df.shape[0]):
            df[j, k] = df[j, k] - df[j, k - 1]
    df_1 = np.zeros([num_users * K, 3], dtype=int)

    for j in range(num_users):
        for k in range(K):
            for i in range(3):
                df_1[j * K + k, 0] = int(j)
                df_1[j * K + k, 1] = int(k)
                df_1[j * K + k, 2] = int(df[j, k])

    sns.set_theme(style="darkgrid")

    sns.set(font_scale=1.5)  ####
    corr_mat = pd.DataFrame(data=df_1, columns=['Client Index', 'Class Index', 'Num of Samples'])

    g = sns.relplot(
        data=corr_mat, x='Client Index', y='Class Index', hue='Client Index', size='Num of Samples',
        palette="tab10", edgecolor=".7", sizes=(50, 250),
    )

    # Tweak the figure to finalize
    g.set(xlabel="Client Index", ylabel="Class Index", xticks=[0,1,2,3], yticks=np.arange(0, K, 1))
    # plt.legend(labels=['', '', 'MNIST', 'SVHN', 'USPS', 'Synth', 'MNIST-M'], bbox_to_anchor=(0.9, 0.52))

    for artist in g.legend.legendHandles:
        artist.set_edgecolor(".7")

    g.savefig('./noniid_train_'+ str(args.num_classes) + 'c_' +str(num_users)+ 'u_' +'beta' +str(args.beta)+'_f'+str(args.feature_iid)+'_l'+str(args.label_iid)+'.pdf', format='pdf')

def prepare_data_digit(args):
    # Prepare digit (feature non-iid, label iid)
    if args.model == 'resnet' or 'resnet_fa':
        transform_mnist = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_svhn = transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_usps = transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_synth = transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_mnistm = transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    # MNIST
    mnist_trainset = DigitsDataset(args=args, data_path=data_root + "digit/MNIST", channels=1, train=True,
                                   transform=transform_mnist)
    mnist_testset = DigitsDataset(args=args, data_path=data_root + "digit/MNIST", channels=1, train=False,
                                  transform=transform_mnist)

    # SVHN
    svhn_trainset = DigitsDataset(args=args, data_path=data_root + 'digit/SVHN', channels=3, train=True,
                                  transform=transform_svhn)
    svhn_testset = DigitsDataset(args=args, data_path=data_root + 'digit/SVHN', channels=3, train=False,
                                 transform=transform_svhn)

    # Synth Digits
    synth_trainset = DigitsDataset(args=args, data_path=data_root + 'digit/SynthDigits/', channels=3, train=True,
                                   transform=transform_synth)
    synth_testset = DigitsDataset(args=args, data_path=data_root + 'digit/SynthDigits/', channels=3, train=False,
                                  transform=transform_synth)

    # USPS
    usps_trainset = DigitsDataset(args=args, data_path=data_root + 'digit/USPS', channels=1, train=True,
                                  transform=transform_usps)
    usps_testset = DigitsDataset(args=args, data_path=data_root + 'digit/USPS', channels=1, train=False,
                                 transform=transform_usps)

    # MNIST-M
    mnistm_trainset = DigitsDataset(args=args, data_path=data_root + 'digit/MNIST_M/', channels=3, train=True,
                                    transform=transform_mnistm)
    mnistm_testset = DigitsDataset(args=args, data_path=data_root + 'digit/MNIST_M/', channels=3, train=False,
                                   transform=transform_mnistm)

    train_dataset_list = [mnist_trainset, svhn_trainset, usps_trainset, synth_trainset, mnistm_trainset]
    test_dataset_list = [mnist_testset, svhn_testset, usps_testset, synth_testset, mnistm_testset]

    # train idx
    idx_batch_train = [[] for _ in range(args.num_clients)]
    user_groups = {}
    for i in range(args.num_clients):
        y_train = train_dataset_list[i%5].labels
        for k in range(args.num_classes):
            idx_k = np.where(y_train == k)[0]
            idx_batch_train[i].extend(idx_k[int(i/5):int(i/5+args.n_per_class)].tolist())
        user_groups[i] = idx_batch_train[i]

    # test idx
    idx_batch_test = [[] for _ in range(args.num_clients)]
    user_groups_test = {}
    for i in range(args.num_clients):
        y_test = test_dataset_list[i%5].labels
        for k in range(args.num_classes):
            idx_k = np.where(y_test == k)[0]
            idx_batch_test[i].extend(idx_k[int(i/5):int(i/5+args.n_per_class_test)].tolist())
        user_groups_test[i] = idx_batch_test[i]

    return train_dataset_list, test_dataset_list, user_groups, user_groups_test


def prepare_data_digits_noniid(num_users, args):
    # Prepare digit (feature non-iid, label non-iid)
    if args.model == 'resnet' or 'resnet_fa':
        transform_mnist = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_svhn = transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_usps = transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_synth = transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_mnistm = transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


    # MNIST
    mnist_trainset = DigitsDataset(args=args, data_path=data_root+'digit/MNIST', channels=1, train=True, transform=transform_mnist)
    mnist_testset = DigitsDataset(args=args, data_path=data_root+'digit/MNIST', channels=1, train=False, transform=transform_mnist)

    # SVHN
    svhn_trainset = DigitsDataset(args=args, data_path=data_root+'digit/SVHN', channels=3, train=True, transform=transform_svhn)
    svhn_testset = DigitsDataset(args=args, data_path=data_root+'digit/SVHN', channels=3, train=False, transform=transform_svhn)

    # USPS
    usps_trainset = DigitsDataset(args=args, data_path=data_root+'digit/USPS', channels=1, train=True, transform=transform_usps)
    usps_testset = DigitsDataset(args=args, data_path=data_root+'digit/USPS', channels=1, train=False, transform=transform_usps)

    # Synth Digits
    synth_trainset = DigitsDataset(args=args, data_path=data_root+'digit/SynthDigits/', channels=3, train=True, transform=transform_synth)
    synth_testset = DigitsDataset(args=args, data_path=data_root+'digit/SynthDigits/', channels=3, train=False, transform=transform_synth)

    # MNIST-M
    mnistm_trainset = DigitsDataset(args=args, data_path=data_root+'digit/MNIST_M/', channels=3, train=True, transform=transform_mnistm)
    mnistm_testset = DigitsDataset(args=args, data_path=data_root+'digit/MNIST_M/', channels=3, train=False, transform=transform_mnistm)

    train_dataset_list = [mnist_trainset, svhn_trainset, usps_trainset, synth_trainset, mnistm_trainset]
    test_dataset_list = [mnist_testset, svhn_testset, usps_testset, synth_testset, mnistm_testset]

    # generate train idx
    idx_batch_train = [[] for _ in range(num_users)]
    user_groups = {}
    user_groups_test = {}
    idx_batch_test = [[] for _ in range(num_users)]
    K = args.num_classes
    df = np.zeros([num_users, K])
    for k in range(K):
        proportions = np.random.dirichlet(np.repeat(args.beta, num_users))
        proportions = proportions / proportions.sum()
        proportions_train = np.ceil((proportions) * (num_users*args.n_per_class)).astype(int)
        print(proportions_train)
        proportions_test = np.ceil((proportions) * (num_users * args.n_per_class_test)).astype(int)
        # print(proportions)
        for i in range(num_users):
            y_train = train_dataset_list[i%5].labels
            idx_k = np.where(y_train == k)[0]
            idx_batch_train[i].extend(idx_k[int(i/5):int(i/5)+proportions_train[i]].tolist())

            y_test = test_dataset_list[i % 5].labels
            idx_k = np.where(y_test == k)[0]
            idx_batch_test[i].extend(idx_k[int(i / 5):int(i / 5) + proportions_test[i]].tolist())

        j = 0
        for idx_j in idx_batch_train:
            if k != 0:
                df[j, k] = int(len(idx_j))
            else:
                df[j, k] = int(len(idx_j))
            j += 1
    for i in range(num_users):
        user_groups[i] = idx_batch_train[i]

    # generate test idx
    for i in range(num_users):
        user_groups_test[i] = idx_batch_test[i]

    # # visualize data distribution
    # visualize_data_dist_fnln(args, df, K, num_users)

    return train_dataset_list, test_dataset_list, user_groups, user_groups_test


def prepare_data_office(args):
    # Prepare office (feature noniid, label iid)
    if args.model == 'resnet' or 'resnet_fa':
        transform_office = transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.ToTensor(),
        ])

        transform_test = transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.ToTensor(),
        ])

    data_base_path = "./data/office/"

    # amazon
    amazon_trainset = OfficeDataset(data_base_path, 'amazon', transform=(transform_office))
    amazon_testset = OfficeDataset(data_base_path, 'amazon', transform=transform_test, train=False)

    # caltech
    caltech_trainset = OfficeDataset(data_base_path, 'caltech', transform=(transform_office))
    caltech_testset = OfficeDataset(data_base_path, 'caltech', transform=transform_test, train=False)

    # dslr
    dslr_trainset = OfficeDataset(data_base_path, 'dslr', transform=(transform_office))
    dslr_testset = OfficeDataset(data_base_path, 'dslr', transform=transform_test, train=False)

    # webcam
    webcam_trainset = OfficeDataset(data_base_path, 'webcam', transform=(transform_office))
    webcam_testset = OfficeDataset(data_base_path, 'webcam', transform=transform_test, train=False)

    train_dataset_list = [amazon_trainset, caltech_trainset, dslr_trainset, webcam_trainset]
    test_dataset_list = [amazon_testset, caltech_testset, dslr_testset, webcam_testset]
    num_users = args.num_clients
    if args.label_iid:

        K = args.num_classes
        idx_batch_train = [[] for _ in range(num_users)]
        user_groups = {}
        for i in range(num_users):
            # ds_idx = ds_idx_list[i]
            # print(i)
            y_train = train_dataset_list[i % 4].labels
            y_train = torch.FloatTensor(y_train)
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                idx_batch_train[i].extend(idx_k[int(i/4):int(i/4)+10].tolist())
            user_groups[i] = idx_batch_train[i]
        # test idx
        user_groups_test = {}
        idx_batch_test = [[] for _ in range(num_users)]
        for i in range(num_users):
            user_groups_test[i] = []
            y_test = test_dataset_list[i % 4].labels
            y_test = torch.FloatTensor(y_test)
            idx_batch_test[i] = []
            for k in range(K):
                idx_k = np.where(y_test == k)[0]
                idx_batch_test[i].extend(idx_k[0:100].tolist())
            user_groups_test[i] = idx_batch_test[i]

    return train_dataset_list, test_dataset_list, user_groups, user_groups_test


def prepare_data_office_noniid(num_users, args):
    # Prepare office (feature noniid, label noniid)

    transform_office = transforms.Compose([
        transforms.Resize([64, 64]),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.Resize([64, 64]),
        transforms.ToTensor(),
    ])

    data_base_path = "./data/office/"

    # amazon
    amazon_trainset = OfficeDataset(data_base_path, 'amazon', transform=(transform_office))
    amazon_testset = OfficeDataset(data_base_path, 'amazon', transform=transform_test, train=False)

    # caltech
    caltech_trainset = OfficeDataset(data_base_path, 'caltech', transform=(transform_office))
    caltech_testset = OfficeDataset(data_base_path, 'caltech', transform=transform_test, train=False)

    # dslr
    dslr_trainset = OfficeDataset(data_base_path, 'dslr', transform=(transform_office))
    dslr_testset = OfficeDataset(data_base_path, 'dslr', transform=transform_test, train=False)

    # webcam
    webcam_trainset = OfficeDataset(data_base_path, 'webcam', transform=(transform_office))
    webcam_testset = OfficeDataset(data_base_path, 'webcam', transform=transform_test, train=False)

    train_dataset_list = [amazon_trainset, caltech_trainset, dslr_trainset, webcam_trainset]
    test_dataset_list = [amazon_testset, caltech_testset, dslr_testset, webcam_testset]

    # generate train idx
    idx_batch_train = [[] for _ in range(num_users)]
    user_groups = {}
    user_groups_test = {}
    idx_batch_test = [[] for _ in range(num_users)]
    K = args.num_classes
    df = np.zeros([num_users, K])
    for k in range(K):
        proportions = np.random.dirichlet(np.repeat(args.alpha, num_users))
        proportions = proportions / proportions.sum()
        proportions_train = ((proportions) * (num_users * 10)).astype(int)
        proportions_test = ((proportions) * (num_users * 100)).astype(int)
        for i in range(num_users):
            y_train = train_dataset_list[i].labels
            y_train = torch.FloatTensor(y_train)
            idx_k = np.where(y_train == k)[0]
            idx_batch_train[i].extend(idx_k[int(i/4):int(i/4)+proportions_train[i]].tolist())

            y_test = test_dataset_list[i].labels
            y_test = torch.FloatTensor(y_test)
            idx_k = np.where(y_test == k)[0]
            idx_batch_test[i].extend(idx_k[int(i / 4):int(i / 4) + proportions_test[i]].tolist())

        j = 0
        for idx_j in idx_batch_train:
            if k != 0:
                df[j, k] = int(len(idx_j))
            else:
                df[j, k] = int(len(idx_j))
            j += 1

    for i in range(num_users):
        user_groups[i] = idx_batch_train[i]

    # generate test idx
    for i in range(num_users):
        user_groups_test[i] = idx_batch_test[i]

    visualize_data_dist_fnln(args, df, K, num_users)

    return train_dataset_list, test_dataset_list, user_groups, user_groups_test


def prepare_data_domain(args):
    # Prepare data
    transform_train = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-30, 30)),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
    ])

    data_base_path = "./data/domain/"

    # clipart
    clipart_trainset = DomainNetDataset(data_base_path, 'clipart', transform=transform_train)
    clipart_testset = DomainNetDataset(data_base_path, 'clipart', transform=transform_test, train=False)
    # infograph
    infograph_trainset = DomainNetDataset(data_base_path, 'infograph', transform=transform_train)
    infograph_testset = DomainNetDataset(data_base_path, 'infograph', transform=transform_test, train=False)
    # painting
    painting_trainset = DomainNetDataset(data_base_path, 'painting', transform=transform_train)
    painting_testset = DomainNetDataset(data_base_path, 'painting', transform=transform_test, train=False)
    # quickdraw
    quickdraw_trainset = DomainNetDataset(data_base_path, 'quickdraw', transform=transform_train)
    quickdraw_testset = DomainNetDataset(data_base_path, 'quickdraw', transform=transform_test, train=False)
    # real
    real_trainset = DomainNetDataset(data_base_path, 'real', transform=transform_train)
    real_testset = DomainNetDataset(data_base_path, 'real', transform=transform_test, train=False)
    # sketch
    sketch_trainset = DomainNetDataset(data_base_path, 'sketch', transform=transform_train)
    sketch_testset = DomainNetDataset(data_base_path, 'sketch', transform=transform_test, train=False)

    train_dataset_list = [clipart_trainset, infograph_trainset, painting_trainset, quickdraw_trainset, real_trainset, sketch_trainset]
    test_dataset_list = [clipart_testset, infograph_testset, painting_testset, quickdraw_testset, real_testset, sketch_testset]

    num_users = args.num_clients
    K = args.num_classes
    idx_batch_train = [[] for _ in range(num_users)]
    user_groups = {}
    for i in range(num_users):
        y_train = torch.Tensor(train_dataset_list[i].labels)
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            idx_batch_train[i].extend(idx_k[0:30].tolist())
        user_groups[i] = idx_batch_train[i]
        # user_groups[i] = list(range(105))

    # test idx
    user_groups_test = {}
    idx_batch_test = [[] for _ in range(num_users)]
    for i in range(num_users):
        y_test = test_dataset_list[i].labels
        # for k in range(K):
        #     idx_k = np.where(y_test == k)[0]
        #     idx_batch_test[i].extend(idx_k[0:100].tolist())
        print(len(y_test))
        user_groups_test[i] = list(range(len(y_test)))
        # user_groups_test[i] = idx_batch_test[i]

    return train_dataset_list, test_dataset_list, user_groups, user_groups_test