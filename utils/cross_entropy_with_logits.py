import torch
import torch.nn.functional


class CrossEntropyLossWithLogits(torch.nn.Module):
    def __init__(self, size_average=True):
        super(CrossEntropyLossWithLogits, self).__init__()
        self.size_average = size_average

    def forward(self, logits, labels):
        loss = torch.sum(-labels * torch.nn.functional.log_softmax(logits, -1), -1)
        if self.size_average:
            return torch.mean(loss)
        else:
            return torch.sum(loss)


class MSE(torch.nn.Module):
    def __init__(self, size_average=True):
        super(MSE, self).__init__()
        self.size_average = size_average

    def forward(self, logits, labels):
        loss = torch.sum(torch.nn.functional.mse_loss(logits, labels), -1)

        if self.size_average:
            return torch.mean(loss)
        else:
            return torch.sum(loss)
