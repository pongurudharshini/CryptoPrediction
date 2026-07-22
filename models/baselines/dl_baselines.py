"""
Deep Learning Baseline Architectures (LSTM & GRU).
Standard PyTorch recurrent baseline networks.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Any
from evaluation.metrics import ModelEvaluator


class PyTorchRecurrentBaseline(nn.Module):
    """Configurable PyTorch Recurrent model (LSTM or GRU)."""

    def __init__(
        self, 
        input_size: int, 
        hidden_size: int = 64, 
        num_layers: int = 2, 
        cell_type: str = "LSTM",
        dropout: float = 0.2
    ):
        super().__init__()
        self.cell_type = cell_type.upper()

        if self.cell_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=input_size, 
                hidden_size=hidden_size, 
                num_layers=num_layers, 
                batch_first=True, 
                dropout=dropout if num_layers > 1 else 0
            )
        elif self.cell_type == "GRU":
            self.rnn = nn.GRU(
                input_size=input_size, 
                hidden_size=hidden_size, 
                num_layers=num_layers, 
                batch_first=True, 
                dropout=dropout if num_layers > 1 else 0
            )
        else:
            raise ValueError(f"Unsupported cell_type: {cell_type}. Choose 'LSTM' or 'GRU'.")

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Input shape: (Batch Size, Sequence Length, Features)
        Output shape: (Batch Size, 1)
        """
        out, _ = self.rnn(x)
        # Take the output of the final time step
        last_step_out = out[:, -1, :]
        prediction = self.fc(last_step_out)
        return prediction.squeeze(-1)


def train_recurrent_baseline(
    model: nn.Module, 
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    X_test: np.ndarray, 
    y_test: np.ndarray, 
    epochs: int = 20, 
    batch_size: int = 32, 
    lr: float = 0.001,
    model_name: str = "LSTM"
) -> Dict[str, Any]:
    """
    Trains a PyTorch recurrent baseline and returns performance metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Convert to PyTorch Tensors
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()

    # Inference on test set
    model.eval()
    with torch.no_grad():
        x_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        predictions = model(x_test_tensor).cpu().numpy()

    return ModelEvaluator.evaluate_all(y_test, predictions, model_name=model_name)