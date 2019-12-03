import torch
import numpy as np


class Metric(object):
    def __init__(self):
        super(Metric, self).__init__()
        pass

    def calculate_metrics(self, out, target, topk=[1,]):
        maxk = max(topk)
        batch_size = out.size(0)
        conf_mat = np.zeros((out.size(1), out.size(1)))

        _, preds = out.topk(maxk, 1, True, True)
        preds = preds.t()
        correct = preds.eq(target.view(1, -1).expand_as(preds))

        for i1, i2 in zip(target.view(-1), preds[0, :].view(-1)):
            conf_mat[i1, i2] += 1

        acc = []
        for k in topk:
            correct_k = correct[:k].view(-1).to(torch.float32).sum(0)
            acc.append(float(correct_k.mul_(100.0 / batch_size)))

        precision, recall = self.get_precision_recall(conf_mat)

        return acc, conf_mat, precision, recall

    @staticmethod
    def get_precision_recall(conf_mat):
        precision = 0
        recall = 0

        # tn = conf_mat[0, 0]
        tp = np.trace(conf_mat[1:, 1:])
        fn = conf_mat[1:, 0].sum()
        fp = conf_mat[1:, 1:].sum() - tp

        if tp + fp > 0:
            precision = 100 * tp / (tp + fp)

        if tp + fn > 0:
            recall = 100 * tp / (tp + fn)

        return precision, recall
