"""Small neural-network building blocks required by the inference decoder."""

import torch.nn as nn


def build_mlp(in_dim, hidden_dim, fc_num_layers, out_dim):
    mods = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for _ in range(fc_num_layers - 1):
        mods += [nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    mods += [nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*mods)
