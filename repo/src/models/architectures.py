"""STEP 14: The four controlled models (spec section 12) -- nothing more.
Baseline MLP, LSTM, GRU, lightweight TCN. Kept deliberately small/simple
so parameter counts stay comparable and no model wins purely by being
bigger than the others (this is an ablation, not a leaderboard).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BaselineMLP(nn.Module):
    """Simple statistical-representation baseline: mean+std pooling over
    the (masked) time axis, then an MLP. Per spec section 12, this is the
    floor every temporal model must beat to justify using temporal
    modeling at all.
    """

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D), mask: (B, T) bool
        m = mask.unsqueeze(-1).float()
        denom = m.sum(dim=1).clamp(min=1.0)
        mean = (x * m).sum(dim=1) / denom
        var = ((x - mean.unsqueeze(1)) ** 2 * m).sum(dim=1) / denom
        std = torch.sqrt(var.clamp(min=1e-6))
        pooled = torch.cat([mean, std], dim=-1)
        return self.net(pooled)


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128,
                 num_layers: int = 1, dropout: float = 0.3, bidirectional: bool = True):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                             dropout=dropout if num_layers > 1 else 0.0, bidirectional=bidirectional)
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(out_dim, num_classes))

    def forward(self, x, mask):
        lengths = mask.sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        if self.lstm.bidirectional:
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            h_last = h_n[-1]
        return self.head(h_last)


class GRUClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128,
                 num_layers: int = 1, dropout: float = 0.3, bidirectional: bool = True):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0, bidirectional=bidirectional)
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(out_dim, num_classes))

    def forward(self, x, mask):
        lengths = mask.sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        if self.gru.bidirectional:
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            h_last = h_n[-1]
        return self.head(h_last)


class TemporalBlock(nn.Module):
    """One dilated causal-ish conv block for the lightweight TCN."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2  # 'same'-ish, non-causal (offline classification, full sequence available)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.relu(self.bn2(self.conv2(out)))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, channels=(64, 64, 128),
                 kernel_size: int = 3, dropout: float = 0.3):
        super().__init__()
        layers = []
        in_ch = input_dim
        for i, out_ch in enumerate(channels):
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation=2 ** i, dropout=dropout))
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_ch, num_classes))

    def forward(self, x, mask):
        # x: (B, T, D) -> (B, D, T) for conv1d
        h = self.tcn(x.transpose(1, 2))  # (B, C, T)
        m = mask.unsqueeze(1).float()  # (B, 1, T)
        denom = m.sum(dim=2).clamp(min=1.0)
        pooled = (h * m).sum(dim=2) / denom
        return self.head(pooled)


def build_model(name: str, input_dim: int, num_classes: int, **kwargs) -> nn.Module:
    name = name.lower()
    if name == "baseline_mlp":
        return BaselineMLP(input_dim, num_classes, **kwargs)
    if name == "lstm":
        return LSTMClassifier(input_dim, num_classes, **kwargs)
    if name == "gru":
        return GRUClassifier(input_dim, num_classes, **kwargs)
    if name == "tcn":
        return TCNClassifier(input_dim, num_classes, **kwargs)
    raise ValueError(f"Unknown model: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
