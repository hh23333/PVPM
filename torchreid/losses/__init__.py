from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .cross_entropy_loss import CrossEntropyLoss, Isolate_loss
from .hard_mine_triplet_loss import TripletLoss, Part_similarity_constrain


def DeepSupervision(criterion, xs, y, part_weights):
    """DeepSupervision

    Applies criterion to each element in a list.

    Args:
        criterion: loss function
        xs: tuple of inputs
        y: ground truth
    """
    loss = 0.
    for i, x in enumerate(xs):
        if part_weights is not None:
            loss += criterion(x, y, part_weights[:,i])
        else:
            loss += criterion(x, y)
    # loss /= len(xs)
    return loss