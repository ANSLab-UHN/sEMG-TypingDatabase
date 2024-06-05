import logging
from torch import nn
from keypressemg.models.nn_blocks import DenseBlock
from keypressemg.models.utils import get_n_params, initialize_weights


class FeatureModel(nn.Module):
    def __init__(self, num_features=96, number_of_classes=26,
                 cls_layer=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._output_info_fn = logging.info
        self._output_debug_fn = logging.debug
        self.cls_layer = cls_layer

        self._dense_block1 = DenseBlock(num_features, 2 * num_features)
        # use_batchnorm=use_group_norm, use_dropout=use_dropout)

        self._dense_block2 = DenseBlock(2 * num_features, 2 * num_features)
        # use_batchnorm=use_group_norm, use_dropout=use_dropout)

        self._dense_block3 = DenseBlock(2 * num_features, 2 * num_features)
        # use_batchnorm=use_group_norm, use_dropout=use_dropout)
        #
        self._dense_block4 = DenseBlock(2 * num_features, int((1 / 2) * num_features),
                                        use_dropout=False, activation='elu')
        # use_batchnorm=use_group_norm, use_dropout=use_dropout)

        # self._dense_block5 = DenseBlock(4 * num_features, 2 * num_features)
        # # use_batchnorm=use_group_norm, use_dropout=use_dropout)
        if self.cls_layer:
            self._output = nn.Linear(int((1 / 2) * num_features), number_of_classes)

        initialize_weights(self)

        self._output_info_fn(str(self))

        self._output_info_fn(f"Number Parameters: {get_n_params(self)}")

    def forward(self, x):
        self._output_debug_fn(f'input {x.shape}')

        fc1 = self._dense_block1(x)
        self._output_debug_fn(f'fc1 {fc1.shape}')

        fc2 = self._dense_block2(fc1)
        self._output_debug_fn(f'fc2 {fc2.shape}')

        fc3 = self._dense_block3(fc2)
        self._output_debug_fn(f'fc3 {fc3.shape}')

        fc4 = self._dense_block4(fc3)
        self._output_debug_fn(f'fc4 {fc4.shape}')

        if self.cls_layer:
            logits = self._output(fc4)
            self._output_debug_fn(f'logits {logits.shape}')

            # probs = F.softmax(logits, dim=1)
            # self._output_debug_fn(f'softmax {probs.shape}')
            return logits

        return fc4

        #
        # fc5 = self._dense_block5(fc4)
        # self._output_debug_fn(f'fc5 {fc5.shape}')

        # if self.cls_layer:
        #     logits = self._output(fc1)
        #     self._output_debug_fn(f'logits {logits.shape}')
        #
        #     probs = F.softmax(logits, dim=1)
        #     self._output_debug_fn(f'softmax {probs.shape}')
        #
        #     return probs
        # else:
        #     return fc1
