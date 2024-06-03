import logging
from torch import nn
from torch.utils.data import DataLoader
from models.nn_blocks import DenseBlock
from models.utils import get_n_params, initialize_weights
from datasets.split_same_day_dataset import SplitDayDataset
from common.types_defined import Participant, DayT1T2
from torch.nn import functional as F


class FeatureModel(nn.Module):
    def __init__(self, num_features=96, number_of_classes=26, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._output_info_fn = logging.info
        self._output_debug_fn = logging.debug

        self._dense_block1 = DenseBlock(num_features, 2 * num_features)
        # use_batchnorm=use_group_norm, use_dropout=use_dropout)

        # self._dense_block2 = DenseBlock(2 * num_features, 4 * num_features)
        # # use_batchnorm=use_group_norm, use_dropout=use_dropout)
        #
        # self._dense_block3 = DenseBlock(4 * num_features, 8 * num_features)
        # # use_batchnorm=use_group_norm, use_dropout=use_dropout)
        #
        # self._dense_block4 = DenseBlock(8 * num_features, 4 * num_features)
        # use_batchnorm=use_group_norm, use_dropout=use_dropout)

        # self._dense_block5 = DenseBlock(4 * num_features, 2 * num_features)
        # # use_batchnorm=use_group_norm, use_dropout=use_dropout)

        self._output = nn.Linear(2 * num_features, number_of_classes)

        initialize_weights(self)

        self._output_info_fn(str(self))

        self._output_info_fn(f"Number Parameters: {get_n_params(self)}")

    def forward(self, x):
        self._output_debug_fn(f'input {x.shape}')

        fc1 = self._dense_block1(x)
        self._output_debug_fn(f'fc1 {fc1.shape}')

        # fc2 = self._dense_block2(fc1)
        # self._output_debug_fn(f'fc2 {fc2.shape}')
        #
        # fc3 = self._dense_block3(fc2)
        # self._output_debug_fn(f'fc3 {fc3.shape}')
        #
        # fc4 = self._dense_block4(fc3)
        # self._output_debug_fn(f'fc4 {fc4.shape}')
        #
        # fc5 = self._dense_block5(fc4)
        # self._output_debug_fn(f'fc5 {fc5.shape}')

        logits = self._output(fc1)
        self._output_debug_fn(f'logits {logits.shape}')

        probs = F.softmax(logits, dim=1)
        self._output_debug_fn(f'softmax {probs.shape}')

        return probs


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(name)s:%(module)s:%(message)s')

    ds = SplitDayDataset(participant=Participant.P1, test=DayT1T2.T1)
    logging.info(f'dataset contains {ds.__len__()} windows')
    loader = DataLoader(ds, batch_size=8, shuffle=True)
    batch_features, batch_labels = next(iter(loader))
    logging.info(f'batch_windows shape: {batch_features.shape}, labels shape: {len(batch_labels)}')
    logging.info(f'batch_labels: {batch_labels}')
    model = FeatureModel()
    output = model(batch_features)
    logging.info(f'output shape: {output.shape}')
    logging.info(f'output: {output}')
