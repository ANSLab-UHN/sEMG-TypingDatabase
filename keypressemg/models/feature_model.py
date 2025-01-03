import logging
from torch import nn
import torch.nn.functional as F
from keypressemg.models.nn_blocks import DenseBlock
from keypressemg.models.utils import get_n_params, initialize_weights


# depth_to_num_params = {
#     1:80_138,
#     2:338_954,
#     3:1_372_682,
#     4:5_504_522,
#     5:22_025_738
# }


def config_local_logger(depth_power, num_features, number_of_classes):
    logger = logging.getLogger(
        f'FeatureModel_{num_features}_features_{number_of_classes}_classes_{depth_power}_depth_power')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    level = logging.INFO
    logger.setLevel(level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger

class FeatureModel(nn.Module):
    def __init__(self, num_features=96, number_of_classes=26, depth_power=5,
                 cls_layer=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        logger = config_local_logger(depth_power, num_features, number_of_classes)
        self._output_info_fn = logger.info
        self._output_debug_fn = logger.debug
        self.cls_layer = cls_layer

        blocks = [DenseBlock((2**i) * num_features, (2**(i+1)) * num_features) for i in range(depth_power)]
        blocks.append(DenseBlock((2**depth_power) * num_features, (2**depth_power) * num_features))
        blocks.extend([DenseBlock((2**(i+1)) * num_features, (2**i) * num_features) for i in range(depth_power-1, -1, -1)])
        self._blocks = nn.ModuleList(blocks)
        self._extra_block = DenseBlock(num_features, int((1 / 2) * num_features),
                                        use_dropout=False, activation='elu')
        if self.cls_layer:
            self._output = nn.Linear(int((1 / 2) * num_features), number_of_classes)

        initialize_weights(self)

        self._output_info_fn(str(self))

        self._output_info_fn(f"Number Parameters: {get_n_params(self)}")

    def forward(self, x):
        self._output_debug_fn(f'input {x.shape}')

        for i, block in enumerate(self._blocks):
            x = block(x)
            self._output_debug_fn(f'output block {i} {x.shape}')

        x = self._extra_block(x)
        self._output_debug_fn(f'extra block {x.shape}')

        if self.cls_layer:
            logits = self._output(x)
            self._output_debug_fn(f'logits {logits.shape}')
            probs = F.softmax(logits, dim=1)
            self._output_debug_fn(f'softmax {probs.shape}')
            x = probs

        return x

