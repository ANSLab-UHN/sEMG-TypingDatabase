import logging
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.nn_blocks import Conv1DBlock, DenseBlock
from models.utils import initialize_weights, get_n_params
from datasets.split_between_days_dataset import SplitBetweenDaysDataset
from common.types_defined import Participant, DayT1T2
from common.folder_paths import VALID_WINDOWS_ROOT


class WindowModel(nn.Module):
    def __init__(self, number_of_classes=26,
                 window_size=400,
                 num_channels=16,
                 use_group_norm=False,
                 use_dropout=False):
        super(WindowModel, self).__init__()

        self._output_info_fn = logging.info
        self._output_debug_fn = logging.debug

        self._conv_block2 = Conv1DBlock(num_channels, 2 * num_channels,
                                        kernel_size=3,
                                        stride=2,
                                        pool_kernel_size=3,
                                        use_groupnorm=use_group_norm,
                                        use_dropout=use_dropout)

        self._conv_block3 = Conv1DBlock(2 * num_channels,
                                        4 * num_channels,
                                        kernel_size=3,
                                        stride=2,
                                        pool_kernel_size=3,
                                        use_groupnorm=use_group_norm,
                                        use_dropout=use_dropout)

        self._conv_block4 = Conv1DBlock(4 * num_channels, 8 * num_channels,
                                        kernel_size=3,
                                        stride=2,
                                        pool_kernel_size=3,
                                        use_groupnorm=use_group_norm,
                                        use_dropout=use_dropout)

        self.flatten = lambda x: x.view(-1, 8 * num_channels)

        # self._dense_block1 = DenseBlock(8 * num_channels, 8 * num_channels)
        # use_batchnorm=use_group_norm, use_dropout=use_dropout)

        self._dense_block2 = DenseBlock(8 * num_channels, 4 * num_channels)
        # use_batchnorm=use_group_norm, use_dropout=use_dropout)

        self._dense_block3 = DenseBlock(4 * num_channels, 2 * num_channels)
        # use_batchnorm=use_group_norm, use_dropout=use_dropout)

        self._output = nn.Linear(2 * num_channels, number_of_classes)

        initialize_weights(self)

        self._output_info_fn(str(self))

        self._output_info_fn(f"Number Parameters: {get_n_params(self)}")

    def forward(self, x):
        self._output_debug_fn(f'input {x.shape}')
        conv1 = x
        conv2 = self._conv_block2(conv1)
        self._output_debug_fn(f'conv2 {conv2.shape}')

        conv3 = self._conv_block3(conv2)
        self._output_debug_fn(f'conv3 {conv3.shape}')

        conv4 = self._conv_block4(conv3)
        self._output_debug_fn(f'conv4 {conv4.shape}')

        flatten_tensor = self.flatten(conv4)
        self._output_debug_fn(f'flatten_tensor {flatten_tensor.shape}')

        fc1 = flatten_tensor
        # fc1 = self._dense_block1(flatten_tensor)
        self._output_debug_fn(f'fc1 {fc1.shape}')

        fc2 = self._dense_block2(fc1)
        self._output_debug_fn(f'fc2 {fc2.shape}')

        fc3 = self._dense_block3(fc2)
        self._output_debug_fn(f'fc3 {fc3.shape}')

        output = self._output(fc3)
        self._output_debug_fn(f'logits {output.shape}')

        output = F.softmax(output, dim=1)
        self._output_debug_fn(f'softmax {output.shape}')

        return output


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s:%(levelname)s:%(name)s:%(module)s:%(message)s')

    logging.basicConfig(level=logging.INFO)
    ds = SplitBetweenDaysDataset(root=VALID_WINDOWS_ROOT,
                                 participant=Participant.P1,
                                 is_train=True)
    logging.info(f'dataset contains {ds.__len__()} windows')
    loader = DataLoader(ds, batch_size=8, shuffle=True)
    batch_windows, batch_labels = next(iter(loader))
    logging.info(f'batch_windows shape: {batch_windows.shape}, labels shape: {len(batch_labels)}')
    logging.info(f'batch_labels: {batch_labels}')
    model = WindowModel()
    output = model(batch_windows)
    logging.info(f'output shape: {output.shape}')
    logging.info(f'output: {output}')
