from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    r"""Cross entropy loss with label smoothing regularizer.
    
    Reference:
        Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.

    With label smoothing, the label :math:`y` for a class is computed by
    
    .. math::
        \begin{equation}
        (1 - \epsilon) \times y + \frac{\epsilon}{K},
        \end{equation}

    where :math:`K` denotes the number of classes and :math:`\epsilon` is a weight. When
    :math:`\epsilon = 0`, the loss function reduces to the normal cross entropy.
    
    Args:
        num_classes (int): number of classes.
        epsilon (float, optional): weight. Default is 0.1.
        use_gpu (bool, optional): whether to use gpu devices. Default is True.
        label_smooth (bool, optional): whether to apply label smoothing. Default is True.
    """
    
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, label_smooth=True):
        super(CrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon if label_smooth else 0
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets, part_weight=None):
        """
        Args:
            inputs (torch.Tensor): prediction matrix (before softmax) with
                shape (batch_size, num_classes).
            targets (torch.LongTensor): ground truth labels with shape (batch_size).
                Each position contains the label index.
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        if part_weight is None:
            return (- targets * log_probs).mean(0).sum()
        else:
            return ((- targets * log_probs).sum(1) * part_weight / (part_weight.sum()+1e-6)).sum()

class Isolate_loss(nn.Module):
    def forward(self, inputs):
        # att_flatten=nn.functional.softmax(inputs.view(inputs.size(0), inputs.size(1), -1), dim=2)
        att_flatten=nn.functional.normalize(inputs.view(inputs.size(0), inputs.size(1), -1), dim=2)
        att_sim_matrix = att_flatten.matmul(att_flatten.transpose(1,2))
        diag_element_mean = (att_flatten*att_flatten).sum(2).sum(1).mean()
        # att_sim_matrix = torch.triu(att_flatten.matmul(att_flatten.transpose(1,2)), diagonal=1)
        loss = att_sim_matrix.sum(1).sum(1).mean()-diag_element_mean
        return loss
