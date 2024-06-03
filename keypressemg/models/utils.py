import numpy as np
from torch import nn


def get_n_params(model: nn.Module):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    number_params = sum([np.prod(p.size()) for p in model_parameters])
    return number_params


def initialize_weights(module: nn.Module):
    for m in module.modules():

        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.zero_()
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.zero_()
