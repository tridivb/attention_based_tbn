import torch


class Metric(object):
    """
    Model evaluation metrics
    
    Args
    ----------
    cfg: dict
        Dictionary of config parameters
    no_batches: int
        Number of batches
    device: torch.device
        Torch device to dump tensors on
    """

    def __init__(self, cfg, no_batches, device=torch.device("cuda")):
        super(Metric, self).__init__()
        self.cfg = cfg
        self.topk = self.cfg.VAL.TOPK
        self.device = device
        self.no_batches = no_batches
        self.multi_class = True if len(self.cfg.MODEL.NUM_CLASSES.keys()) > 1 else False

        self.loss = {}
        self.accuracy = {}
        self.conf_mat = {}
        self.loss = {}

        for key, no_cls in self.cfg.MODEL.NUM_CLASSES.items():
            self.accuracy[key] = [0] * (len(cfg.VAL.TOPK))
            self.conf_mat[key] = torch.zeros((no_cls, no_cls), device=device)
            self.loss[key] = 0

        if self.multi_class:
            self.loss["total"] = 0
            self.accuracy["all_class"] = [0] * (len(cfg.VAL.TOPK))
            self.loss["total"] = 0

    def set_metrics(self, out, target, batch_size, batch_loss):
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

        correct = {}
        if self.multi_class:
            correct["all_class"] = []
        for key in out.keys():
            corr, cm = self._get_correct_score(
                out[key], target[key], self.topk, self.device
            )
            self.conf_mat[key] += cm
            correct[key] = corr
            if self.multi_class:
                correct["all_class"].extend([corr])
            self.loss[key] += batch_loss[key].item()

        if self.multi_class:
            self.loss["total"] += batch_loss["total"].item()


        for key in self.accuracy.keys():
            for i, k in enumerate(self.topk):
                if key == "all_class":
                    c = correct[key][0][:k].sum(0)
                    for c2 in correct[key][1:]:
                        # Check where predictions in both classes are correct or 1
                        c = c * c2[:k].sum(0)
                    c = c.to(torch.float32).sum()
                    acc = float(c.mul_(100.0 / batch_size))
                else:
                    correct_k = correct[key][:k].view(-1).to(torch.float32).sum()
                    acc = float(correct_k.mul_(100.0 / batch_size))
                self.accuracy[key][i] += acc

    def get_metrics(self):
        """
        Helper function to calculate average accuracy and loss

        Returns
        ----------
        loss: dict
            Dictionary of losses for each class and sum of all losses
        acc: dict
            Accuracy of each type of class
        confusion_matrix: Tensor
            Tensor of the confusion matrix
        """

        for key in self.accuracy.keys():
            self.accuracy[key] = [
                round(x / self.no_batches, 2) for x in self.accuracy[key]
            ]

        for key in self.loss.keys():
            self.loss[key] = round(self.loss[key] / self.no_batches, 5)

        return self.loss, self.accuracy, self.conf_mat

    @staticmethod
    def _get_correct_score(out, target, topk, device):
        """
        Helper function to calculate confusion matrix and correctness scores
        
        Args
        ----------
        out: tensor
            Output from forward pass of model
        target: tensor
            Target labels
        device: torch.device
            Torch device to dump tensors on
        """

        maxk = max(topk)
        conf_mat = torch.zeros((out.size(1), out.size(1)), device=device)

        _, preds = out.topk(maxk, 1, largest=True, sorted=True)
        preds = preds.t()
        correct = preds.eq(target.view(1, -1).expand_as(preds))

        for i1, i2 in zip(target.view(-1), preds[0, :].view(-1)):
            conf_mat[i1, i2] += 1

        return correct, conf_mat