import torch
from torch import nn

from models.utils import initialize_weights


def conv_block(in_f, out_f, activation='relu', pool='avg_pool',
               *args, **kwargs):
    activations = nn.ModuleDict([
        ['lrelu', nn.LeakyReLU()],
        ['relu', nn.ReLU()]
    ])

    pools = nn.ModuleDict([
        ['max_pool', nn.MaxPool1d(kernel_size=3)],
        ['avg_pool', nn.AvgPool1d(kernel_size=3)],
        ['adaptiveavgpool', nn.AdaptiveAvgPool1d(output_size=1)]
    ])

    return nn.Sequential(
        nn.Conv1d(in_f, out_f, kernel_size=3, stride=2),
        pools[pool],
        nn.BatchNorm1d(out_f),
        nn.Dropout(p=0.5),
        activations[activation]
    )


def dec_block(in_f, out_f):
    return nn.Sequential(
        nn.Linear(in_f, out_f),
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
    def __init__(self, dec_sizes, n_classes):
        super().__init__()
        self.dec_blocks = nn.Sequential(*[dec_block(in_f, out_f)
                                          for in_f, out_f in zip(dec_sizes, dec_sizes[1:])])

        self.last = nn.Linear(dec_sizes[-1], n_classes)

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

        for m in self.modules():

            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

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
