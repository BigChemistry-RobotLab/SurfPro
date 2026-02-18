import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torch_geometric.nn.models import AttentiveFP
from torch_geometric.data import Data


class RegressionHead(pl.LightningModule):
    def __init__(self, n_prop, dim):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(normalized_shape=dim)
        self.fc1 = nn.Linear(dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, n_prop)

    def forward(self, x):
        x = self.norm(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class AttentiveFPModel(pl.LightningModule):
    def __init__(
        self,
        props,
        hidden_channels=64,
        out_channels=128,
        num_layers=2,
        num_timesteps=2,
        dropout=0.1,
    ):
        super().__init__()
        self.props = props
        self.n_prop = len(props)
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.dropout = dropout

        self.encoder = AttentiveFP(
            in_channels=39,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            edge_dim=11,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            dropout=dropout,
        )

        self.head = RegressionHead(
            n_prop=self.n_prop,
            dim=out_channels
        )

        self.criterion = nn.HuberLoss()
        self.criterion_mae = nn.L1Loss()
        self.criterion_mse = nn.MSELoss()

        self.learning_rate = 1e-3

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))

    def calculate_errors(self, preds, labels, prefix=""):
        """calculate MAE/RMSE for overall and per-property"""
        # calculate loss on multi-property pred
        loss = self.criterion(preds, labels)

        metrics = {}
        metrics[f"{prefix}/loss"] = loss
        metrics[f"{prefix}/mae"] = self.criterion_mae(preds, labels)
        metrics[f"{
            prefix}/rmse"] = torch.sqrt(self.criterion_mse(preds, labels))

        for i, prop in enumerate(self.props):
            metrics[f"{prefix}/props-mae/{prop}"] = self.criterion_mae(
                labels[:, i], preds[:, i]
            )
            metrics[f"{prefix}/props-rmse/{prop}"] = torch.sqrt(
                self.criterion_mse(labels[:, i], preds[:, i])
            )
        self.log_dict(metrics, batch_size=64)
        return metrics

    def forward(self, feats):
        """encode with attentiveFP, then apply regression head"""
        # encode with attentivefp model
        embs = self.encoder(
            feats.x,
            feats.edge_index,
            feats.edge_attr,
            feats.batch,
        )

        # apply regression head
        preds = self.head(embs)
        return preds

    def training_step(self, batch, batch_idx):
        feats = batch.get("feats")
        masks = batch.get("masks")
        labels = batch.get("labels")

        preds = self.forward(feats) * masks

        labels = torch.nan_to_num(labels, nan=0.0)
        assert labels.shape == preds.shape

        loss = self.criterion(preds, labels)

        metrics = self.calculate_errors(preds, labels, "train")
        metrics["loss"] = loss
        return metrics

    def validation_step(self, batch, batch_idx):
        feats = batch.get("feats")
        masks = batch.get("masks")
        labels = batch.get("labels")

        with torch.set_grad_enabled(False):
            preds = self.forward(feats) * masks

        labels = torch.nan_to_num(labels, nan=0.0)

        metrics = self.calculate_errors(preds, labels, "valid")
        return metrics

    def test_step(self, batch, batch_idx):
        feats = batch.get("feats")
        masks = batch.get("masks")
        labels = batch.get("labels")

        with torch.set_grad_enabled(False):
            preds = self.forward(feats) * masks

        labels = torch.nan_to_num(labels, nan=0.0)

        metrics = self.calculate_errors(preds, labels, "test")
        return metrics

    def predict_step(self, batch, batch_idx):
        """return predictions"""
        smiles = batch.get("smiles")
        feats = batch.get("feats")
        types = batch.get("types")

        with torch.set_grad_enabled(False):
            preds = self.forward(feats)

        return {"smiles": smiles, "types": types, "preds": preds}
