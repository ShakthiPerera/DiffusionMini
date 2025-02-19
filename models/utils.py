'''Modeling utilities.'''

import torch.nn as nn


def make_dense(in_features,
               out_features,
               bias=True,
               activation=None):
    '''
    Create fully connected layer.

    Parameters
    ----------
    in_features : int
        Number of inputs.
    out_features : int
        Number of outputs.
    bias : bool
        Determines whether a bias is used.
    activation : None or str
        Determines the nonlinearity.

    '''

    linear = nn.Linear(in_features, out_features, bias=bias)
    activation = make_activation(activation)

    layers = [linear, activation]
    dense_block = nn.Sequential(*layers)

    return dense_block


def make_activation(mode):
    '''Create activation.'''
    if mode is None or mode == 'none':
        activation = nn.Identity()
    elif mode == 'sigmoid':
        activation = nn.Sigmoid()
    elif mode == 'tanh':
        activation = nn.Tanh()
    elif mode == 'relu':
        activation = nn.ReLU()
    elif mode == 'leaky_relu':
        activation = nn.LeakyReLU()
    elif mode == 'elu':
        activation = nn.ELU()
    elif mode == 'softplus':
        activation = nn.Softplus()
    elif mode == 'swish':
        activation = nn.SiLU()
    else:
        raise ValueError('Unknown activation function: {}'.format(mode))
    return activation


def make_norm(mode, num_features):
    '''Create normalization.'''
    if mode is None or mode == 'none':
        norm = nn.Identity()
    elif mode == 'batch':
        norm = nn.BatchNorm2d(num_features)
    elif mode == 'instance':
        norm = nn.InstanceNorm2d(num_features)
    else:
        raise ValueError('Unknown normalization type: {}'.format(mode))
    return norm