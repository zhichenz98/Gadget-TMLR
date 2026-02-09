import torch.nn as nn
import torch.nn.functional as F
import torch
from dgl.nn.pytorch import GraphConv, SAGEConv, GINConv, APPNPConv, SGConv


class GCNLayer(nn.Module):
    def __init__(self, in_feats, n_hidden, activation=F.relu, dropout=0.5):
        super(GCNLayer, self).__init__()
        self.layer = GraphConv(in_feats, n_hidden, activation=activation)
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(n_hidden)
        self.activation = activation
        
        
    def reset_parameters(self):
        self.layer.reset_parameters()
        self.bn.reset_parameters()
        
    def forward(self, graph, feat):
        hidden_feat = self.layer(graph, feat, edge_weight=graph.edata.get('w'))
        hidden_feat = self.bn(hidden_feat)
        hidden_feat = self.dropout(hidden_feat)
        return hidden_feat


class APPNPLayer(nn.Module):
    def __init__(self, in_feats, n_hidden, k, alpha, activation=F.relu, dropout=0.5):
        super(APPNPLayer, self).__init__()
        self.activation = activation
        self.lin = torch.nn.Linear(in_feats, n_hidden)
        self.layer = APPNPConv(k, alpha)
        self.bn = nn.BatchNorm1d(n_hidden)
        self.dropout = nn.Dropout(dropout)
    
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.lin.weight)
        self.bn.reset_parameters()

    def forward(self, graph, feat):
        hidden_feat = self.activation(self.lin(feat))
        hidden_feat = self.layer(graph, hidden_feat, edge_weight=graph.edata.get('w'))
        # hidden_feat = self.dropout(hidden_feat)
        hidden_feat = self.bn(hidden_feat)
        return hidden_feat
