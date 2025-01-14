import copy

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.nn.functional as F

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        if isinstance(image, torch.Tensor):
            image = image.clone().detach()
        else:
            image = image
        if isinstance(label, torch.Tensor):
            label = label.clone().detach()
        else:
            label = torch.tensor(label)
        return image, label


class train_update(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.trainloader = self.train_val_test(dataset, list(idxs))
        self.device = args.device
        self.criterion_CE = nn.CrossEntropyLoss().to(self.device)
        self.criterion_MSE = nn.MSELoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        idxs_train = idxs[:int(len(idxs))]
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train), batch_size=self.args.local_bs, shuffle=True, drop_last=True)
        return trainloader

    def criterion_correction(self, features, labels, protos_dict):
        bs = features.shape[0]
        target = torch.ones(bs)
        data = torch.zeros(bs).to(self.device)
        for i in range(bs):
            label = labels[i].item()
            data[i] = torch.cosine_similarity(features[i], torch.tensor(protos_dict[label]).to(self.device), dim=0)
        target = torch.tensor(np.array(target)).to(self.device)
        loss = 2 * self.criterion_MSE(data, target)
        return loss


    def criterion_InfoNCE(self, features, labels, protos_dict):
        temperature = self.args.tau
        bs = features.shape[0]
        protos_list = []
        num_protos = 0
        protos_key_list = []
        for label in protos_dict.keys():
            if protos_dict[label].ndim == 1:
                protos_list_label = protos_dict[label].reshape(1,-1)
            else:
                protos_list_label = protos_dict[label]

            for i in range(len(protos_list_label)):
                protos_list.append(protos_list_label[i])
                protos_key_list.append(label)

        num_protos += len(protos_list)
        mask = np.zeros((bs, num_protos))

        protos_list = torch.tensor(np.array(protos_list)).to(self.device)
        protos_key_list = np.array(protos_key_list)
        logits = torch.zeros(bs, num_protos).to(self.device)

        for i in range(bs):
            label = labels[i].item()
            mask[i][np.where(protos_key_list == label)] = 1
            logits[i] = torch.cosine_similarity(features[i].unsqueeze(0), protos_list, dim=1)
        mask = torch.tensor(mask).to(self.device)

        logits = logits / temperature

        exp_logits = torch.exp(logits)
        sum_exp_logits = exp_logits.sum(1, keepdim=True)
        pos_logits = exp_logits * mask
        sum_pos_logits = pos_logits.sum(1, keepdim=True)

        loss = - torch.log(sum_pos_logits/sum_exp_logits)
        return loss.mean()

    def criterion_InfoNCE_alpha(self, features, labels, protos_dict):
        temperature = self.args.tau
        alpha = self.args.alpha
        bs = features.shape[0]
        protos_list = []
        num_protos = 0
        protos_key_list = []
        for label in protos_dict.keys():
            protos_dict[label] = np.array(protos_dict[label])
            if protos_dict[label].ndim == 1:
                protos_list_label = protos_dict[label].reshape(1, -1)
            else:
                protos_list_label = protos_dict[label]
            for i in range(len(protos_list_label)):
                protos_list.append(protos_list_label[i])
                protos_key_list.append(label)
        num_protos += len(protos_list)
        mask = np.zeros((bs, num_protos))
        protos_list = torch.tensor(np.array(protos_list)).to(self.device)
        protos_key_list = np.array(protos_key_list)
        logits = torch.zeros(bs, num_protos).to(self.device)
        for i in range(bs):
            label = labels[i].item()
            mask[i][np.where(protos_key_list == label)] = 1
            logits[i] = torch.cosine_similarity(features[i].unsqueeze(0), protos_list, dim=1)
        mask = torch.tensor(mask).to(self.device)
        logits = logits.pow(alpha)
        logits = logits / temperature
        exp_logits = torch.exp(logits)
        sum_exp_logits = exp_logits.sum(1, keepdim=True)
        pos_logits = exp_logits * mask
        sum_pos_logits = pos_logits.sum(1, keepdim=True)
        loss = - torch.log(sum_pos_logits / sum_exp_logits)
        return loss.mean()


    def update_weights_fedavg(self, idx, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = {'total': [], '1': [], '2': []}
        # Set optimizer for the local updates
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=1e-5)

        for iter in range(self.args.train_ep):
            batch_loss = {'1': [], '2': [], 'total': []}
            for batch_idx, (images, labels) in enumerate(self.trainloader):

                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                probs, features = model(images)
                loss1 = self.criterion_CE(probs, labels)
                loss2 = 0 * loss1
                loss = loss1

                # SGD
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss['1'].append(loss1.item())
                batch_loss['2'].append(loss2.item())
                batch_loss['total'].append(loss.item())
                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss1: {:.3f}\tLoss2: {:.3f}\tLoss: {:.3f}'.format(
                            global_round, idx, iter, batch_idx * len(images),
                                    len(self.trainloader.dataset),
                                    100. * batch_idx / len(self.trainloader),
                                    loss1.item(),
                                    loss2.item(),
                                    loss.item()))

            epoch_loss['1'].append(sum(batch_loss['1']) / len(batch_loss['1']))
            epoch_loss['2'].append(sum(batch_loss['2']) / len(batch_loss['2']))
            epoch_loss['total'].append(sum(batch_loss['total']) / len(batch_loss['total']))

        epoch_loss['1'] = sum(epoch_loss['1']) / len(epoch_loss['1'])
        epoch_loss['2'] = sum(epoch_loss['2']) / len(epoch_loss['2'])
        epoch_loss['total'] = sum(epoch_loss['total']) / len(epoch_loss['total'])

        return model.state_dict(), epoch_loss


    def update_weights_proposed(self, idx, local_cluster_protos, global_cluster_protos, global_protos, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = {'total': [], '1': [], '2': []}
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,  momentum=self.args.momentum, weight_decay=1e-5)

        for iter in range(self.args.train_ep):
            batch_loss = {'1': [], '2': [], 'total': []}
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                probs, features = model(images)
                # Compute CE loss
                loss1 = self.criterion_CE(probs, labels)
                # Compute prototype loss term
                loss2 = 0 * loss1
                loss3 = 0 * loss1
                if len(local_cluster_protos) == self.args.num_clients:
                    for i in range(self.args.num_clients):
                        for label in global_cluster_protos.keys():
                            if label not in local_cluster_protos[i].keys():
                                local_cluster_protos[i][label] = global_protos[label]
                    loss2 += self.criterion_InfoNCE_alpha(features, labels, global_cluster_protos)
                    loss3 += self.criterion_correction(features, labels, global_protos)
                    loss2 += loss3
                loss = loss1 + self.args.lamb * loss2

                # SGD
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss['1'].append(loss1.item())
                batch_loss['2'].append(loss2.item())
                batch_loss['total'].append(loss.item())
                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss1: {:.3f}\tLoss2: {:.3f}\tLoss3: {:.3f}\tLoss: {:.3f}'.format(
                            global_round, idx, iter, batch_idx * len(images),
                                    len(self.trainloader.dataset),
                                    100. * batch_idx / len(self.trainloader),
                                    loss1.item(),
                                    loss2.item(),
                                    loss3.item(),
                                    loss.item()))

            epoch_loss['1'].append(sum(batch_loss['1']) / len(batch_loss['1']))
            epoch_loss['2'].append(sum(batch_loss['2']) / len(batch_loss['2']))
            epoch_loss['total'].append(sum(batch_loss['total']) / len(batch_loss['total']))

        epoch_loss['1'] = sum(epoch_loss['1']) / len(epoch_loss['1'])
        epoch_loss['2'] = sum(epoch_loss['2']) / len(epoch_loss['2'])
        epoch_loss['total'] = sum(epoch_loss['total']) / len(epoch_loss['total'])

        # generate representation
        protos_dict = {}
        model.eval()
        for batch_idx, (images, labels) in enumerate(self.trainloader):
            images, labels = images.to(self.device), labels.to(self.device)

            _, features = model(images)
            for i in range(len(labels)):
                proto = features[i, :].detach().cpu().numpy()
                if labels[i].item() in protos_dict:
                    protos_dict[labels[i].item()].append(proto)
                else:
                    protos_dict[labels[i].item()] = [proto]

        return model.state_dict(), epoch_loss, protos_dict


class test_update(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.testloader = self.test_split(dataset, list(idxs))
        self.device = args.device

    def test_split(self, dataset, idxs):
        idxs_test = idxs[:int(1 * len(idxs))]
        testloader = DataLoader(DatasetSplit(dataset, idxs_test), batch_size=self.args.test_bs, shuffle=False)
        return testloader

    def test_inference(self, idx, local_model):
        model = local_model
        model.to(self.args.device)
        total, correct = 0.0, 0.0
        protos_test = [[] for _ in range(self.args.num_classes)]
        # test
        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.args.device), labels.to(self.args.device)

            probs, features = model(images)
            if self.args.saveprotos:
                for i in range(images.shape[0]):
                    protos_test[labels[i].item()].append(features[i].detach().cpu().numpy())
            # prediction
            _, pred_labels = torch.max(probs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
        protos_test = np.array(protos_test)
        acc = correct / total
        print('| User: {} | Test Acc: {:.5f}'.format(idx, acc))

        return acc, protos_test