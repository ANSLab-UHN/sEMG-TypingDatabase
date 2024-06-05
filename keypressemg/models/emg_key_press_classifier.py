from enum import Enum
import torch
from torch import nn


class ModuleName(Enum):
    LINEAR = 'Linear'
    CONV1D = 'Conv1d'


def get_module(name: ModuleName, in_f: int, out_f: int) -> nn.Module:
    m = nn.Linear(in_f, out_f) if name == ModuleName.LINEAR else nn.Conv1d(in_f, out_f, kernel_size=3, stride=2)
    nn.init.kaiming_normal_(m.weight)
    m.bias.data.zero_()
    return m


def conv_block(in_f, out_f, activation='relu', pool='avg_pool',
               *args, **kwargs):
    activations = nn.ModuleDict({
        'l_relu': nn.LeakyReLU(),
        'relu': nn.ReLU()
    })
    assert activation in activations.keys(), f'activation should be one of {list(activations.keys())}'
    pools = nn.ModuleDict({
        'max_pool': nn.MaxPool1d(kernel_size=3),
        'avg_pool': nn.AvgPool1d(kernel_size=3),
        'adaptive_avg_pool': nn.AdaptiveAvgPool1d(output_size=1)
    })
    assert pool in pools.keys(), f'pool should be one of {list(pools.keys())}'

    return nn.Sequential(
        get_module(ModuleName.CONV1D, in_f, out_f),
        pools[pool],
        nn.BatchNorm1d(out_f),
        nn.Dropout(p=0.5),
        activations[activation]
    )


def dec_block(in_f, out_f):
    linear = nn.Linear(in_f, out_f)
    nn.init.kaiming_normal_(linear.weight)
    linear.bias.data.zero_()
    return nn.Sequential(
        get_module(ModuleName.LINEAR, in_f, out_f),
        nn.Dropout(p=0.5),
        nn.ReLU()
        # nn.Sigmoid()
    )


class Encoder(nn.Module):
    def __init__(self, enc_sizes, *args, **kwargs):
        super().__init__()
        self._conv_blocks = nn.Sequential(*[conv_block(in_f, out_f, kernel_size=3, padding=1, *args, **kwargs)
                                            for in_f, out_f in zip(enc_sizes, enc_sizes[1:])])

    def forward(self, x):
        return self._conv_blocks(x)


class Decoder(nn.Module):
    def __init__(self, dec_sizes, num_classes):
        super().__init__()
        self.dec_blocks = nn.Sequential(*[dec_block(in_f, out_f)
                                          for in_f, out_f in zip(dec_sizes, dec_sizes[1:])])

        self.last = get_module(ModuleName.LINEAR, dec_sizes[-1], num_classes)

    def forward(self, x):
        # print('before dec blocks', x.shape)
        x = self.dec_blocks(x)

        # print('after temp pool', x.shape)
        return self.last(x)


class EmgKeyPressClassifier(nn.Module):
    def __init__(self, in_c, enc_sizes, dec_sizes, n_classes, activation='relu', pool='avg_pool'):
        super().__init__()
        self.enc_sizes = [in_c, *enc_sizes]
        self.dec_sizes = [enc_sizes[-1], *dec_sizes]

        self.encoder = Encoder(self.enc_sizes, activation=activation, pool=pool)
        self.temp_pool = nn.AdaptiveAvgPool1d(1)

        self.decoder = Decoder(dec_sizes, n_classes)

    def forward(self, x):
        # print('before encoder', x.shape)
        x = self.encoder(x)
        # print('after enc blocks', x.shape)
        x = self.temp_pool(x)
        # print('before flatten', x.shape)
        x = x.flatten(1)  # flat
        # print('before decoder', x.shape)

        x = self.decoder(x)

        x = torch.sigmoid(x)

        return x


if __name__ == '__main__':
    num_channels = 16
    num_classes = 26
    depth = 1
    encoder_sizes = [num_channels * 2 ** i for i in range(1, depth + 1)]
    decoder_sizes = list(reversed(encoder_sizes))
    print(encoder_sizes)
    print(decoder_sizes)
    model = EmgKeyPressClassifier(in_c=16, enc_sizes=encoder_sizes, dec_sizes=decoder_sizes, n_classes=26)
    print('number of parameters', sum([p.numel() for p in model.parameters()]))
    x = torch.randn((64, 16, 400))
    y = model(x)
    print(y.shape)
