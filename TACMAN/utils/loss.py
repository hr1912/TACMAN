import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pandas import value_counts
from torch import Tensor


def make_class_weights(labels, astensor=True, foo=np.sqrt, n_add=0):
    if isinstance(labels, Tensor):
        labels = labels.cpu().clone().detach().numpy()

    counts = value_counts(labels).sort_index()  # sort for alignment
    n_cls = len(counts) + n_add
    w = counts.apply(lambda x: 1 / foo(x + 1) if x > 0 else 0)
    w = (w / w.sum() * (1 - n_add / n_cls)).values
    w = np.array(list(w) + [1 / n_cls] * int(n_add))

    return Tensor(w) if astensor else w
