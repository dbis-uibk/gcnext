import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import BatchNorm, GCNConv, LayerNorm, SAGEConv, Sequential, GraphConv, GATConv, GATv2Conv


class GCN_Encoder(nn.Module):

    def __init__(self, layer_sizes, dropout=0.0):
        super().__init__()

        assert len(layer_sizes) >= 2
        self.input_size, self.representation_size = layer_sizes[0], layer_sizes[-1]

        layers = []
        for i, (in_dim, out_dim) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            layers.append((GCNConv(in_dim, out_dim, cached=False), 'x, edge_index, edge_weight -> x'))

            if i < len(layer_sizes) - 2:
                layers.append(nn.PReLU(out_dim))
                #layers.append(nn.Dropout(p=dropout, training=self.training))

        self.model = Sequential('x, edge_index, edge_weight', layers)

    def forward(self, x, edge_index, edge_weight):
        return self.model(x, edge_index, edge_weight)

    def reset_parameters(self):
        self.model.reset_parameters()


class SAGE_Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, embedding_size, num_layers=2):
        super().__init__()

        assert num_layers >= 2
        self.num_layers = num_layers

        convs = []
        for n in range(num_layers):
            if n == 0:
                convs.append(SAGEConv(input_size, hidden_size, root_weight=True))
            elif n == num_layers - 1:
                convs.append(SAGEConv(hidden_size, embedding_size, root_weight=True))
            else:
                convs.append(SAGEConv(hidden_size, hidden_size, root_weight=True))

        self.convs = nn.ModuleList(convs)
        self.skip_lins = nn.ModuleList([nn.Linear(input_size, hidden_size, bias=False)] * (num_layers - 1))
        self.activations = nn.ModuleList([nn.PReLU(hidden_size)] * (num_layers - 1) + [nn.PReLU(embedding_size)])

    def forward(self, x, edge_index, edge_weight):
        h = self.convs[0](x, edge_index)
        h = self.activations[0](h)
        h_l = [self.activations[0](h)]

        for l in range(1, self.num_layers):
            x_skip = self.skip_lins[l-1](x)
            h = self.convs[l](sum(h_l) + x_skip, edge_index)
            h = self.activations[l](h)
            h_l.append(h)

        return h

    def reset_parameters(self):
        for m in self.convs:
            m.reset_parameters()
        for m in self.skip_lins:
            m.reset_parameters()
        for m in self.activations:
            m.weight.data.fill_(0.25)


class GraphConv_Encoder(nn.Module):
    """
    https://github.com/pyg-team/pytorch_geometric/issues/1961
    """
    def __init__(self, input_size, hidden_size, embedding_size, num_layers=2, dropout=0.0):
        super().__init__()

        assert num_layers >= 2
        self.num_layers = num_layers
        self.dropout_rate = dropout

        convs = []
        for n in range(num_layers):
            if n == 0:
                convs.append(GraphConv(input_size, hidden_size, aggr='mean'))
            elif n == num_layers - 1:
                convs.append(GraphConv(hidden_size, embedding_size, aggr='mean'))
            else:
                convs.append(GraphConv(hidden_size, hidden_size, aggr='mean'))

        self.convs = nn.ModuleList(convs)
        self.skip_lins = nn.ModuleList([nn.Linear(input_size, hidden_size, bias=False)] * (num_layers - 1))
        self.activations = nn.ModuleList([nn.PReLU(hidden_size)] * (num_layers - 1) + [nn.PReLU(embedding_size)])

    def forward(self, x, edge_index, edge_weight):
        h = self.convs[0](x, edge_index)
        h = self.activations[0](h)
        h_l = [self.activations[0](h)]

        for l in range(1, self.num_layers):
            x_skip = self.skip_lins[l-1](x)
            h = self.convs[l](sum(h_l) + x_skip, edge_index, edge_weight)
            h = self.activations[l](h)
            h_l.append(h)

        return h

    def reset_parameters(self):
        for m in self.convs:
            m.reset_parameters()
        for m in self.skip_lins:
            m.reset_parameters()
        for m in self.activations:
            m.weight.data.fill_(0.25)


class GAT_Encoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size, num_layers=2, heads=1):
        super().__init__()

        assert num_layers >= 1
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATv2Conv(input_size, hidden_size,
                                  heads, edge_dim=1))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATv2Conv(heads * hidden_size, hidden_size, heads, edge_dim=1)) # edge_dim = 1 for edge weights
        self.convs.append(
            GATv2Conv(heads * hidden_size, embedding_size, heads=1, edge_dim=1,
                    concat=False))

        self.skip_lins = nn.ModuleList([nn.Linear(input_size, heads * hidden_size)] * (num_layers - 2) + [nn.Linear(input_size, embedding_size)])
        self.activations = nn.ModuleList([nn.PReLU(heads * hidden_size)] * (num_layers - 1) + [nn.PReLU(embedding_size)])

    def forward(self, x, edge_index, edge_weight):
        h = x
        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index, edge_weight)
            if i != 0:
                h = h + self.skip_lins[i-1](x)
            h = self.activations[i](h)
        return h

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for skip in self.skip_lins:
            skip.reset_parameters()
        for m in self.activations:
            m.weight.data.fill_(0.25)