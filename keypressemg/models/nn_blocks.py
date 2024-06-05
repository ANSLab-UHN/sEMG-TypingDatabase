from torch import nn
from torch.nn import functional as F


class Conv2DBlock(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size, stride,
                 pool_kernel_size,
                 use_groupnorm=False,
                 num_groups=4,
                 use_dropout=False):
        super(Conv2DBlock, self).__init__()
        self._conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self._pool = nn.AvgPool2d(kernel_size=pool_kernel_size)
        self._group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels) if use_groupnorm \
            else nn.Identity(out_channels)
        self._relu = nn.ReLU(out_channels)
        self._dropout = nn.Dropout2d(.5) if use_dropout \
            else nn.Identity(out_channels)

    def forward(self, x):
        # return self._dropout(self._relu(self._batch_norm(self._conv(x))))
        return F.relu(self._dropout(self._group_norm(self._pool(self._conv(x)))))


class Conv1DBlock(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size, stride,
                 pool_kernel_size,
                 use_groupnorm=False,
                 num_groups=4,
                 use_dropout=False):
        super(Conv1DBlock, self).__init__()
        self._conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self._pool = nn.AvgPool1d(kernel_size=pool_kernel_size)
        self._group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels) if use_groupnorm \
            else nn.Identity(out_channels)
        self._relu = nn.ReLU(out_channels)
        self._dropout = nn.Dropout(.5) if use_dropout \
            else nn.Identity(out_channels)

    def forward(self, x):
        # return self._dropout(self._relu(self._batch_norm(self._conv(x))))
        return F.relu(self._dropout(self._group_norm(self._pool(self._conv(x)))))


class DenseBlock(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 use_batchnorm=False,
                 use_dropout=True,
                 activation='relu'):
        super(DenseBlock, self).__init__()
        self._fc = nn.Linear(in_channels, out_channels)
        self._batch_norm = nn.BatchNorm1d(out_channels) if use_batchnorm \
            else nn.Identity(out_channels)
        self._act = nn.ReLU() if activation == 'relu' else nn.ELU()
        self._dropout = nn.Dropout(.5) if use_dropout else nn.Identity(out_channels)

    def forward(self, x):
        # return self._dropout(self._relu(self._batch_norm(self._fc(x))))
        return self._act(self._dropout(self._batch_norm(self._fc(x))))
