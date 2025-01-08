import torch
from sklearn.metrics import f1_score, adjusted_mutual_info_score

def detach2numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.cpu().clone().detach().numpy()
    return x

def get_acc(label, pred):
    return (label == pred).sum() / len(label)


def get_AMI(y_true, y_pred, **kwds):
    y_true, y_pred = list(map(detach2numpy, (y_true, y_pred)))
    ami = adjusted_mutual_info_score(y_true, y_pred, **kwds)
    return ami


def get_F1_score(y_true, y_pred, average='macro', **kwds):
    y_true, y_pred = list(map(detach2numpy, (y_true, y_pred)))
    f1 = f1_score(y_true, y_pred, average=average, **kwds)
    return f1
