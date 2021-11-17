import torch.nn as nn


class ListModule(nn.Module):
    """
    behaves like a list but works with DataParallel
    """
    def __init__(self, *args):
        super().__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)


class BudgetModel(nn.Module):
    def __init__(self, input_channels, num_classes, num_models):
        super().__init__()
        self.input_channesl = input_channels
        self.num_classes = num_classes
        self.num_models = num_models

        models_list = []
        for _ in range(num_models):
            models_list.append(Ensemble(input_channels, num_classes))

        self.models = ListModule(*models_list)

    def get_model_parameters(self, model_index):
        return self.models[model_index].parameters()

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        return outputs


class TwoLayerNet(nn.Module):
    def __init__(self, D_in, l1, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(D_in, l1),
            nn.SELU(),
            nn.Linear(l1, D_out),
            nn.Sigmoid())

    def forward(self, x1):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        probs = (self.linear1(x1))
        return probs


class Ensemble(nn.Module):
    def __init__(self, input_channels, num_classes):
        super().__init__()
        ensemble_models_list = []

        ensemble_models_list.append(TwoLayerNet(256, 128, num_classes))
        self.ensemble_models = ListModule(*ensemble_models_list)

    def forward(self, x):
        outputs = []
        for m in self.ensemble_models:
            outputs.append(m(x))
        return outputs

