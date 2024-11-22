import pandas as pd
import wandb
import pickle
import json
import numpy as np
import hydra
import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateFinder
from pytorch_lightning.loggers import WandbLogger

# from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import OmegaConf, DictConfig
from src.dataloader import SurfProDB, DataSplit
from src.model import AttentiveFPModel
from torch_geometric.data import Data, Batch

torch.set_float32_matmul_precision("high")
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
wandb.require("core")


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def fill_table(cfg: DictConfig) -> None:
    print("FILL TABLE CONFIG from params.yaml")
    cfg = OmegaConf.load("./params.yaml")
    print(OmegaConf.to_yaml(cfg))

    with open(f"{cfg.host.workdir}/data/{cfg.task.name}/surfpro.pkl", "rb") as f:
        surfpro = pickle.load(f)

    ######################
    # full-dataset prediction
    # uses model ensemble to predict for the entire dataset
    ######################

    train_loader = surfpro.train[0].loader(shuffle=False, num_workers=16)
    valid_loader = surfpro.valid[0].loader(shuffle=False, num_workers=2)
    test_loader = surfpro.test.loader(shuffle=False, num_workers=2)

    kwargs = {
        "props": surfpro.propnames,
        "hidden_channels": cfg.model.hidden_channels,
        "out_channels": cfg.model.out_channels,
        "num_layers": cfg.model.num_layers,
        "num_timesteps": cfg.model.num_timesteps,
        "dropout": cfg.model.dropout,
    }
    model = AttentiveFPModel(**kwargs)

    trainer = pl.Trainer(
        max_epochs=cfg.model.n_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=cfg.host.device if torch.cuda.is_available() else "auto",
        precision=32 if cfg.task.scale else "bf16-mixed",
        default_root_dir=f"{cfg.host.workdir}/out/models/",
    )

    pred_dfs = []
    for fold in range(cfg.task.n_splits):
        model.load_state_dict(
            torch.load(
                f"{cfg.host.workdir}/out/{cfg.task.name}/models/model{fold}.pt")
        )

        train_preds = trainer.predict(model, train_loader)
        valid_preds = trainer.predict(model, valid_loader)
        test_preds = trainer.predict(model, test_loader)

        if cfg.task.scale and surfpro.scaled:
            unscaler = surfpro.unscale
        else:
            unscaler = None

        pred_dfs.append(
            pd.concat(
                [
                    flatten_scale(train_preds, surfpro.propnames,
                                  "tr_va", unscaler),
                    flatten_scale(valid_preds, surfpro.propnames,
                                  "tr_va", unscaler),
                    flatten_scale(test_preds, surfpro.propnames,
                                  "test", unscaler),
                ],
                axis=0,
            )
        )

    df_ensemble_preds = pd.DataFrame()
    df_ensemble_preds["SMILES"] = pred_dfs[0]["SMILES"]
    df_ensemble_preds["types"] = pred_dfs[0]["types"]
    df_ensemble_preds["split"] = pred_dfs[0]["split"]
    for prop in cfg.task.props:
        for fold, preds in enumerate(pred_dfs):
            assert all(df_ensemble_preds["SMILES"] == preds["SMILES"])
        preds = np.array([df[prop] for df in pred_dfs])

        preds_avg = np.mean(preds, axis=0)
        df_ensemble_preds[prop] = preds_avg

        preds_std = np.std(preds, axis=0, dtype=np.float64)
        df_ensemble_preds[f"{prop}_std"] = preds_std

        assert (
            preds_std.shape[0] == preds_avg.shape[0] == len(
                df_ensemble_preds["SMILES"])
        )

    df_ensemble_preds = df_ensemble_preds.sort_values(by="SMILES").reset_index(
        drop=True
    )
    df_ensemble_preds.to_csv(
        f"{cfg.host.workdir}/out/{cfg.task.name}/df_ensemble_preds.csv"
    )

    ######################
    # fill the table: update NaNs in 'raw' dataset with ensemble pred
    ######################
    df_raw = pd.read_csv(f"{cfg.host.workdir}/data/{cfg.task.name}/df_raw.csv")
    df_merged = df_raw.reset_index(drop=True).copy().sort_values(by="SMILES")

    df_merged["types"] = df_ensemble_preds["types"]

    # copy rows for standard deviations to be 'filled', fill with 0 std for not predicted
    for prop in cfg.task.props:
        df_merged[f"{prop}_std"] = df_merged[prop].copy()
        df_merged[f"{prop}_std"] = df_merged[f"{prop}_std"].apply(
            lambda x: 0 if pd.notna(x) else np.nan
        )

    df_merged.update(df_ensemble_preds, overwrite=False)

    df_merged = df_merged.sort_values(
        by=["types", "SMILES"]).reset_index(drop=True)

    order = [pair for prop in cfg.task.props for pair in (prop, f"{prop}_std")]
    df_merged = df_merged[["SMILES", "types", *order]]
    df_merged.to_csv(
        f"{cfg.host.workdir}/out/{cfg.task.name}/filled_table_merged.csv")


def flatten_scale(pred_dict, props, split, unscaler=None):
    """merge batched pred dicts into df"""
    smiles = []
    types = []
    preds = []
    for batch in pred_dict:
        smiles.extend(batch.get("smiles"))
        types.extend(batch.get("types"))
        preds.append(batch.get("preds"))
    preds = torch.cat(preds).float().numpy()

    if unscaler is not None:
        preds = unscaler(preds)

    assert preds.shape[1] == len(props)
    df = pd.DataFrame({"SMILES": smiles, "types": types, "split": split})
    for i, prop in enumerate(props):
        df[prop] = np.array(preds[:, i])

    return df.reset_index(drop=True)


if __name__ == "__main__":
    fill_table()
