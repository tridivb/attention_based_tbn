import torch


class Metric(object):
    """
    Model evaluation metrics
    """
    def __init__(self):
        super(Metric, self).__init__()
        pass

    def calculate_metrics(self, out, target, device, topk=[1,]):
        """
        Helper function to calculate accuracy and confusion matrix
        
        Args
        ----------
        out: tensor
            Output from forward pass of model
        target: tensor
            Target labels
        device: torch.device
            Torch device to dump tensors on
        topk: list, default = [1,]
            List of top-k accuracies
        """

        maxk = max(topk)
        batch_size = out.size(0)
        conf_mat = torch.zeros((out.size(1), out.size(1)), device=device)

        _, preds = out.topk(maxk, 1, True, True)
        preds = preds.t()
        correct = preds.eq(target.view(1, -1).expand_as(preds))

        for i1, i2 in zip(target.view(-1), preds[0, :].view(-1)):
            conf_mat[i1, i2] += 1

        acc = []
        for k in topk:
            correct_k = correct[:k].view(-1).to(torch.float32).sum(0)
            acc.append(float(correct_k.mul_(100.0 / batch_size)))

        # precision, recall = self.get_precision_recall(conf_mat)

        return acc, conf_mat

    # @staticmethod
    # def get_multi_class_accuracy(out, target):
    #     o = []
    #     t = []
    #     for key in out.keys():
    #         o.extend([out[key]])
    #         t.extend([target[key]])

    #     o = torch.cat(o, dim=1)
    #     t = torch.cat(t, dim=1)

    # @staticmethod
    # def get_precision_recall(conf_mat):
    #     precision = 0
    #     recall = 0

    #     tp = conf_mat[1:, 1:].diag().sum().item()
    #     fn = conf_mat[1:, 0].sum().item()
    #     fp = conf_mat[1:, 1:].sum().item() - tp

    #     if tp + fp > 0:
    #         precision = 100 * tp / (tp + fp)

    #     if tp + fn > 0:
    #         recall = 100 * tp / (tp + fn)

    #     return precision, recall
