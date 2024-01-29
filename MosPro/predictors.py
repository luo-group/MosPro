import sys
sys.path.append('..')
import torch
import torch.nn as nn
import torch.nn.functional as F
# import esm_one_hot
from .fitness_dataset import aa2idx, AA_LIST

class LengthMaxPool1D(nn.Module):
    def __init__(self, in_dim, out_dim, linear=False, activation='relu'):
        super().__init__()
        self.linear = linear
        if self.linear:
            self.layer = nn.Linear(in_dim, out_dim)

        if activation == 'swish':
            self.act_fn = lambda x: x * torch.sigmoid(100.0*x)
        elif activation == 'softplus':
            self.act_fn = nn.Softplus()
        elif activation == 'sigmoid':
            self.act_fn = nn.Sigmoid()
        elif activation == 'leakyrelu':
            self.act_fn = nn.LeakyReLU()
        elif activation == 'relu':
            self.act_fn = lambda x: F.relu(x)
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.linear:
            x = self.act_fn(self.layer(x))
        x = torch.max(x, dim=1)[0]
        return x

class BaseCNN(nn.Module):
    def __init__(self, config):
        super(BaseCNN, self).__init__()
        n_tokens = config.n_tokens
        kernel_size = config.kernel_size
        input_size = config.input_size
        dropout = config.dropout
        make_one_hot = config.make_one_hot
        activation = config.activation
        linear = config.linear
        self.encoder = nn.Conv1d(n_tokens, input_size, kernel_size=kernel_size)
        self.dropout = nn.Dropout(dropout) # TODO: actually add this to model
        self.embedding = LengthMaxPool1D(
            linear=linear,
            in_dim=input_size,
            out_dim=input_size*2,
            activation=activation,
        )
        if hasattr(config, 'spectral_norm') and config.spectral_norm:
            self.decoder = nn.utils.parametrizations.spectral_norm(nn.Linear(input_size*2, 1))
        else:
            self.decoder = nn.Linear(input_size*2, 1)
        self.n_tokens = n_tokens
        self.input_size = input_size
        self._make_one_hot = make_one_hot

    def forward(self, x):
        #onehotize
        # print('x:', x, type(x), x.shape)
        if self._make_one_hot:
            x = F.one_hot(x.long(), num_classes=self.n_tokens)
        x = x.permute(0, 2, 1).float()
        # encoder
        x = self.encoder(x).permute(0, 2, 1)
        x = self.dropout(x)
        # embed
        x = self.embedding(x)
        # decoder
        output = self.decoder(x).squeeze(1)
        return output

    def disable_one_hot(self):
        self._make_one_hot = False
    
    def enable_one_hot(self):
        self._make_one_hot = True
    
        
if __name__ == '__main__':
    from easydict import EasyDict
    from utils import common
    config = EasyDict({
        'n_tokens': 20,
        'kernel_size': 5,
        'input_size': 256,
        'dropout': 0.0,
        'make_one_hot': True,
        'activation': 'relu',
        'linear': True,
    })
    model = BaseCNN(config=config)
    print(model)
    x = torch.rand(10, 30)
    y = model(x)
    print(y.shape)

