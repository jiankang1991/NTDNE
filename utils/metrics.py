
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
import numpy as np
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, classification_report


class MetricTracker(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class KNNClassification(nn.Module):

    def __init__(self, X_train, Y_true, K=10):
        super().__init__()

        self.K = K

        self.KNN = KNeighborsClassifier(n_neighbors=self.K, weights='distance')
        self.KNN.fit(X_train, Y_true)

    def forward(self, X_test, y_true):

        y_pred = self.KNN.predict(X_test)

        acc = accuracy_score(y_true, y_pred)

        return acc


class calssification_report(nn.Module):

    def __init__(self, target_names):
        super().__init__()
        self.target_names = target_names
    def forward(self, predict_labels, true_labels):

        report = classification_report(true_labels, predict_labels, target_names=self.target_names, output_dict=True)

        return report


class NormSoftmaxLoss(nn.Module):
    """
    L2 normalize weights and apply temperature scaling on logits.
    """
    def __init__(self,
                 dim,
                 num_instances,
                 temperature=0.05):
        super(NormSoftmaxLoss, self).__init__()

        self.weight = Parameter(torch.Tensor(num_instances, dim))
        # Initialization from nn.Linear (https://github.com/pytorch/pytorch/blob/v1.0.0/torch/nn/modules/linear.py#L129)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        self.temperature = temperature
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, embeddings, instance_targets):
        norm_weight = nn.functional.normalize(self.weight, dim=1)

        prediction_logits = nn.functional.linear(embeddings, norm_weight)

        if instance_targets is not None:
            loss = self.loss_fn(prediction_logits / self.temperature, instance_targets)
            return prediction_logits, loss
        else:
            return prediction_logits


class NormSoftmaxLoss_Margin(nn.Module):
    """ 
    https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py
    """
    def __init__(self,
                 dim,
                 num_instances,
                 margin=0.5,
                 temperature=0.05):
        super().__init__()

        self.weight = Parameter(torch.Tensor(num_instances, dim))
        # Initialization from nn.Linear (https://github.com/pytorch/pytorch/blob/v1.0.0/torch/nn/modules/linear.py#L129)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        self.temperature = temperature
        self.loss_fn = nn.CrossEntropyLoss()

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
    
    def forward(self, embeddings, label):
        norm_weight = F.normalize(self.weight, dim=1)
        cosine = F.linear(embeddings, norm_weight)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        loss = self.loss_fn(logits / self.temperature, label)
        return logits, loss


class TruncatedNSM(nn.Module):
    def __init__(self, dim, nb_class, nb_train_samples, temperature=0.05, q=0.7, k=0.5):
        super().__init__()

        self.cls_weights = Parameter(torch.Tensor(nb_class, dim))
        stdv = 1. / math.sqrt(self.cls_weights.size(1))
        self.cls_weights.data.uniform_(-stdv, stdv)

        self.temperature = temperature
        self.q = q
        self.k = k
        self.weight = Parameter(data=torch.ones(nb_train_samples, 1), requires_grad=False)

    def forward(self, embeddings, targets, indexes):

        norm_weight = nn.functional.normalize(self.cls_weights, dim=1)
        logits = nn.functional.linear(embeddings, norm_weight)
        logits /= self.temperature

        p = F.softmax(logits, dim=1)

        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))

        loss = ((1-(Yg**self.q))/self.q)*self.weight[indexes] - ((1-(self.k**self.q))/self.q)*self.weight[indexes]
        loss = torch.mean(loss)

        return loss


    def update_weight(self, embeddings, targets, indexes):
        
        norm_weight = nn.functional.normalize(self.cls_weights, dim=1)
        logits = nn.functional.linear(embeddings, norm_weight)
        logits /= self.temperature

        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        Lq = ((1-(Yg**self.q))/self.q)
        Lqk = np.repeat(((1-(self.k**self.q))/self.q), targets.size(0))
        Lqk = torch.from_numpy(Lqk).type(torch.cuda.FloatTensor)
        Lqk = torch.unsqueeze(Lqk, 1)

        condition = torch.gt(Lqk, Lq)
        self.weight[indexes] = condition.type(torch.cuda.FloatTensor)


class TruncatedNSM_M(nn.Module):
    def __init__(self, dim, nb_class, nb_train_samples, temperature=0.05, q=0.7, k=0.5, margin=0.5):
        super().__init__()

        self.cls_weights = Parameter(torch.Tensor(nb_class, dim))
        stdv = 1. / math.sqrt(self.cls_weights.size(1))
        self.cls_weights.data.uniform_(-stdv, stdv)

        self.temperature = temperature
        self.q = q
        self.k = k
        self.weight = Parameter(data=torch.ones(nb_train_samples, 1), requires_grad=False)
        
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
    
    def forward(self, embeddings, targets, indexes):
        norm_weight = nn.functional.normalize(self.cls_weights, dim=1)
        cosine = F.linear(embeddings, norm_weight)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, targets.view(-1, 1).long(), 1)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits /= self.temperature
        
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        loss = ((1-(Yg**self.q))/self.q)*self.weight[indexes] - ((1-(self.k**self.q))/self.q)*self.weight[indexes]
        loss = torch.mean(loss)

        return loss
    
    def update_weight(self, embeddings, targets, indexes):
        norm_weight = nn.functional.normalize(self.cls_weights, dim=1)
        cosine = F.linear(embeddings, norm_weight)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, targets.view(-1, 1).long(), 1)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits /= self.temperature

        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        Lq = ((1-(Yg**self.q))/self.q)
        Lqk = np.repeat(((1-(self.k**self.q))/self.q), targets.size(0))
        Lqk = torch.from_numpy(Lqk).type(torch.cuda.FloatTensor)
        Lqk = torch.unsqueeze(Lqk, 1)

        condition = torch.gt(Lqk, Lq)
        self.weight[indexes] = condition.type(torch.cuda.FloatTensor)


class RNSL(nn.Module):
    def __init__(self, dim, nb_class, temperature=0.05, q=0.7):
        super().__init__()

        self.cls_weights = Parameter(torch.Tensor(nb_class, dim))
        stdv = 1. / math.sqrt(self.cls_weights.size(1))
        self.cls_weights.data.uniform_(-stdv, stdv)

        self.temperature = temperature
        self.q = q

    def forward(self, embeddings, targets):
        norm_weight = nn.functional.normalize(self.cls_weights, dim=1)
        logits = nn.functional.linear(embeddings, norm_weight)
        logits /= self.temperature

        p = F.softmax(logits, dim=1)

        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))

        loss = (1-(Yg**self.q))/self.q
        loss = torch.mean(loss)

        return loss





class HingeLoss(nn.Module):
    """
    Hinge loss based on the paper:
    when deep learning meets metric learning:remote sensing image scene classification
    via learning discriminative CNNs 
    https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/9
    """

    def __init__(self, margin=0.44):
        super().__init__()
        
        self.margin = margin

    def forward(self, oneHotCodes, features):
        
        L_S = oneHotCodes.mm(torch.t(oneHotCodes))
        Dist = torch.norm(features[:,None] - features, dim=2, p=2)**2

        Dist = self.margin - Dist
        
        L_S[L_S==0] = -1

        Dist = 0.05 - L_S * Dist

        loss = torch.triu(Dist, diagonal=1)

        loss[loss < 0] = 0

        return torch.mean(loss)


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class GCE(nn.Module):

    def __init__(self, q=0.7, k=0.5, trainset_size=50000):
        super().__init__()
        self.q = q
        self.k = k
        self.weight = torch.nn.Parameter(data=torch.ones(trainset_size, 1), requires_grad=False)
             
    def forward(self, logits, targets, indexes):
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))

        loss = ((1-(Yg**self.q))/self.q)*self.weight[indexes] - ((1-(self.k**self.q))/self.q)*self.weight[indexes]
        loss = torch.mean(loss)

        return loss

    def update_weight(self, logits, targets, indexes):
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        Lq = ((1-(Yg**self.q))/self.q)
        Lqk = np.repeat(((1-(self.k**self.q))/self.q), targets.size(0))
        Lqk = torch.from_numpy(Lqk).type(torch.cuda.FloatTensor)
        Lqk = torch.unsqueeze(Lqk, 1)
        

        condition = torch.gt(Lqk, Lq)
        self.weight[indexes] = condition.type(torch.cuda.FloatTensor)



class SCELoss(torch.nn.Module):
    def __init__(self, alpha=0.1, beta=1, num_classes=10):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss

