# requires: torch
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DeepFM(nn.Module):
    def __init__(self, dim: int, k: int = 16, hidden_dims: list[int] = [64, 32], dropout: float = 0.2, l2: float = 1e-6, lr: float = 0.001, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.k = k

        # Shared embedding layer for FM and MLP
        self.embedding = nn.Embedding(dim, k)
        # Linear term
        self.linear = nn.Embedding(dim, 1)
        self.bias = nn.Parameter(torch.zeros(1))

        # MLP layers
        # Number of fields = 5 in default config
        num_fields = 5
        input_dim = num_fields * k
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, 1))
        self.mlp = nn.Sequential(*layers)

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=l2)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, X):
        # X: (batch_size, num_fields) tensor of int32/int64
        embed_x = self.embedding(X) # (batch, num_fields, k)

        # 1. Linear part
        lin_out = self.linear(X).squeeze(-1).sum(dim=1) + self.bias

        # 2. FM interaction part
        sum_embed = embed_x.sum(dim=1) # (batch, k)
        sum_square_embed = (embed_x ** 2).sum(dim=1) # (batch, k)
        fm_out = 0.5 * (sum_embed ** 2 - sum_square_embed).sum(dim=1)

        # 3. Deep part (MLP)
        mlp_input = embed_x.view(X.size(0), -1)
        mlp_out = self.mlp(mlp_input).squeeze(-1)

        logits = lin_out + fm_out + mlp_out
        return logits

    def predict_tensor(self, X, bs: int = 200_000):
        self.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X), bs):
                batch_X = torch.from_numpy(X[i:i + bs]).long()
                logits = self.forward(batch_X)
                preds.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(preds) if len(X) else np.zeros(0)

    def step_pointwise(self, X, y, w=None):
        self.train()
        batch_X = torch.from_numpy(X).long()
        batch_y = torch.from_numpy(y).float()

        self.optimizer.zero_grad()
        logits = self.forward(batch_X)
        loss = self.criterion(logits, batch_y)
        if w is not None:
            # If weights are provided, we can weigh the loss
            # For pointwise standard, w is usually None or passed
            pass
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def state(self):
        return {k: v.cpu().clone() for k, v in self.state_dict().items()}

    def load(self, st):
        self.load_state_dict(st)
