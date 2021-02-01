import torch
from torch import nn
from torch.autograd import Function
import math
import numpy as np

eps = 1e-8



class NCA_Lp(nn.Module):

    def __init__(self, train_size, labels, start_epoch, q=0.7, k=0.5):
        super().__init__()
        # self.T = T
        self.register_buffer('labels', torch.LongTensor(labels.size(0)))
        self.register_buffer('weights', torch.ones(train_size, 1))
        self.labels = labels
        # self.margin = margin
        self.start_epoch = start_epoch
        self.q = q
        self.k = k

    def forward(self, x, indexes):
        
        batchSize = x.size(0)
        n = x.size(1) # size of training images
        exp = torch.exp(x)

        batch_weights = torch.index_select(self.weights, 0, indexes.data)
        y = torch.index_select(self.labels, 0, indexes.data).view(batchSize, 1) 
        same = y.repeat(1, n).eq_(self.labels)

        # self similarities are set as 0
        exp.data.scatter_(1, indexes.data.view(-1,1), 0)
        p = torch.mul(exp, same.float()).sum(dim=1)
        Z = exp.sum(dim=1)

        # Z_exclude = Z - p
        # p = p.div(math.exp(self.margin))
        prob = torch.div(p, Z)

        loss = ((1-(prob**self.q))/self.q)*batch_weights - ((1-(self.k**self.q))/self.q)*batch_weights
        loss = torch.mean(loss)

        return loss

    def update_weight(self, x, indexes):
        
        batchSize = x.size(0)
        n = x.size(1) # size of training images
        exp = torch.exp(x)

        batch_weights = torch.index_select(self.weights, 0, indexes.data)
        y = torch.index_select(self.labels, 0, indexes.data).view(batchSize, 1) 
        same = y.repeat(1, n).eq_(self.labels)

        # self similarities are set as 0
        exp.data.scatter_(1, indexes.data.view(-1,1), 0)
        p = torch.mul(exp, same.float()).sum(dim=1)
        Z = exp.sum(dim=1)

        # Z_exclude = Z - p
        # p = p.div(math.exp(self.margin))
        prob = torch.div(p, Z)
        
        Lq = ((1-(prob**self.q))/self.q)
        Lqk = np.repeat(((1-(self.k**self.q))/self.q), batchSize)
        Lqk = torch.from_numpy(Lqk).type(torch.cuda.FloatTensor)
        # Lqk = torch.unsqueeze(Lqk, 1)

        condition = torch.gt(Lqk, Lq)
        self.weights[indexes] = torch.unsqueeze(condition.type(torch.cuda.FloatTensor), 1)










