"""Modal 2: Seed-anchored PPI network embedding via frozen GCN."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GCNLayer(nn.Module):
    """Graph Convolutional Layer"""
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, X, adj):
        """
        X: (n_nodes, in_features)
        adj: (n_nodes, n_nodes) - 정규화된 인접 행렬
        """
        support = self.linear(X)
        output = torch.matmul(adj, support)
        return output


class Modal2_GCN(nn.Module):
    """
    PPI 네트워크 기반 GCN 모델
    """
    def __init__(self, input_dim, hidden_dims=[64, 32], output_dim=16, dropout=0.5):
        super(Modal2_GCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout
        
        # GCN 레이어 구성
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(GCNLayer(dims[i], dims[i+1]))
        
        self.layers = nn.ModuleList(layers)
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, X, adj):
        """
        X: (n_nodes, input_dim) - 노드 특성
        adj: (n_nodes, n_nodes) - 정규화된 인접 행렬
        """
        h = X
        for i, layer in enumerate(self.layers[:-1]):
            h = layer(h, adj)
            h = F.relu(h)
            h = self.dropout_layer(h)
        
        # 마지막 레이어 (활성화 없음)
        h = self.layers[-1](h, adj)
        
        return h
    
    @staticmethod
    def normalize_adjacency(adj):
        """
        인접 행렬 정규화: D^(-1/2) * A * D^(-1/2)
        """
        adj = adj + torch.eye(adj.size(0))  # Self-loop 추가
        degree = torch.sum(adj, dim=1)
        d_inv_sqrt = torch.pow(degree, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        normalized_adj = torch.matmul(torch.matmul(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        return normalized_adj


# ==================== Cross-Modal Attention ====================