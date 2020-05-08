from torch.nn import DataParallel


class DataParallel(DataParallel):
    def get_loss(self, criterion, target, preds):
        return self.module.get_loss(criterion, target, preds)
