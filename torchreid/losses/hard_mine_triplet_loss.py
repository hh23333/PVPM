from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import numpy as np
# import miosqp

class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    
    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.
    
    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """
    
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        n = inputs.size(0)
        
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)

class Part_similarity_constrain(nn.Module):
    #output a binary vector:b which indicate the matching label of each part
    #output the cost function:l for updating the pose subnet
    def __init__(self, momentum=0.5, ss_init=0.9, lambd_rate=0.9, part_num=6):
        super(Part_similarity_constrain, self).__init__()
        self.lambd_rate=lambd_rate
        self.momentum = momentum   #sue for update lambd and ss
        self.matching_criterion = nn.BCELoss()  # visibility verification loss in the paper
        #part-part similarity between gallery and query images
        self.register_buffer('lambd', lambd_rate*torch.ones(part_num))
        ##self_similarity between each part
        self.register_buffer('ss_mean', torch.zeros((part_num, part_num)))

    def forward(self, inputs, targets, matching_inputs, use_matching_loss=False):
        #input_size = [N,c,p,1]
        #matching_inputs [1] N*P which is the output of PVP module
        loss = 0
        inputs = torch.cat(inputs, dim=2)
        batchsize = inputs.size(0)
        num_part = matching_inputs.size(1)
        ss_diff_=inputs.expand((batchsize,inputs.size(1),num_part, num_part))
        ss_diff = ss_diff_-ss_diff_.permute(0,1,3,2)
        # normalized difference vectors between different part: N*c*p*p
        ss_diff_norm=torch.nn.functional.normalize(ss_diff,p=2,dim=1)
        device_id = inputs.get_device()
        mask_s = (torch.ones((num_part,num_part))-torch.eye(num_part)).cuda(device_id)
        matching_targets = []
        matching_logit = []
        cs_batch = 0
        ss_batch = 0
        inputs = torch.nn.functional.normalize(inputs, p=2, dim=1).squeeze()
        ss = torch.matmul(inputs.transpose(1, 2), inputs)    # similarity matrix between different part
        lambd = (ss*mask_s).mean(1)*num_part/(num_part-1)    # lambda used for matching loss
        for i in range(batchsize):
            l_q = targets[i]
            for j in range(i+1,batchsize):
                l_g = targets[j]
                if  i!=j and l_g==l_q:
                    cs_ij_=(inputs[i]*inputs[j]).sum(0)   #cross_similarity P*P
                    cs_batch = cs_batch+cs_ij_.detach()
                    cs_ij=torch.diag(cs_ij_)   #cross_similarity P*P
                    s_constr_= (ss_diff_norm[i]*ss_diff_norm[j]).sum(0)
                    ss_batch=ss_batch+s_constr_.detach()
                    W = cs_ij+(s_constr_-self.ss_mean)
                    if use_matching_loss:
                        x_optim, _= self.IQP_solver(W,self.lambd)
                        x_optim = torch.from_numpy(x_optim).cuda(device_id)
                    else:
                        x_optim = torch.from_numpy(np.ones(W.size(0))).cuda(device_id)
                    matching_targets.append(x_optim)
                    matching_logit.append(matching_inputs[i]*matching_inputs[j])
                    loss += -torch.matmul(torch.matmul(x_optim,W),x_optim) + ((lambd[i]+lambd[j])*x_optim).sum()/2

        loss = loss/len(matching_targets)
        cs_batch=cs_batch/len(matching_targets)
        ss_batch=ss_batch/len(matching_targets)
        self.ss_mean=self.momentum*self.ss_mean+(1-self.momentum)*ss_batch
        self.lambd=self.momentum*self.lambd+(1-self.momentum)*self.lambd_rate*cs_batch
        if use_matching_loss:
            matching_targets = torch.stack(matching_targets)
            matching_logit = torch.stack(matching_logit)
            matching_loss = self.matching_criterion(matching_logit, matching_targets)
        else:
            matching_loss = 0
        return matching_loss, loss

    def IQP_solver(self, W, lambd):
        W = W.data.cpu().numpy()
        lambd = lambd.data.cpu().numpy()
        value = 0
        x_result = np.zeros(W.shape[0], dtype='float32')
        for i in range(2**(W.shape[0])):
            x = np.asarray(list('{:b}'.format(i).zfill(W.shape[0])), dtype='float32')
            value_i = np.matmul(np.matmul(x,W),x)-(lambd*x).sum()
            if value_i>value:
                value = value_i
                x_result = x
        return x_result, value
