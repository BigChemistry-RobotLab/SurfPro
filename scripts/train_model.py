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
from omegaconf import OmegaConf, DictConfig
from surfpro.dataloader import SurfProDB, DataSplit
from surfpro.model import AttentiveFPModel
from torch_geometric.data import Data, Batch

torch.set_float32_matmul_precision("high")
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
wandb.require("core")


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def train(cfg: DictConfig) -> None:
    print("TRAIN CONFIG from params.yaml")
    cfg = OmegaConf.load("./params.yaml")
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.model.seed)

    workdir = f"{cfg.host.workdir}/out/{cfg.task.name}"
    with open(f"{cfg.host.workdir}/data/{cfg.task.name}/surfpro.pkl", "rb") as f:
        surfpro = pickle.load(f)
    test_df = surfpro.test_df

    print("test", surfpro.test)
    for i in range(len(surfpro.train)):
        print("train", i, "\t", len(surfpro.train[i].smiles))
        print("valid", i, "\t", len(surfpro.valid[i].smiles))

    test_loader = surfpro.test.loader(shuffle=False, num_workers=2)

    metrics = {}
    for fold in range(cfg.task.n_splits):
        train_loader = surfpro.train[fold].loader(shuffle=True, num_workers=16)
        valid_loader = surfpro.valid[fold].loader(shuffle=False, num_workers=2)

        wandb_logger = WandbLogger(
            project="SurfPro",
            config=OmegaConf.to_container(cfg, resolve=True),
        )

        kwargs = {
            "props": surfpro.propnames,
            "hidden_channels": cfg.model.hidden_channels,
            "out_channels": cfg.model.out_channels,
            "num_layers": cfg.model.num_layers,
            "num_timesteps": cfg.model.num_timesteps,
            "dropout": cfg.model.dropout,
        }
        model = AttentiveFPModel(**kwargs)

        early_stop = EarlyStopping(
            monitor="valid/mae",
            patience=50,
            mode="min",
            check_on_train_epoch_end=False,
        )
        lr_find = LearningRateFinder()

        trainer = pl.Trainer(
            max_epochs=cfg.model.n_epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=cfg.host.device if torch.cuda.is_available() else "auto",
            logger=wandb_logger if torch.cuda.is_available() else None,
            precision=32 if cfg.task.scale else "bf16-mixed",
            default_root_dir=f"{workdir}/models/",
            callbacks=[lr_find, early_stop],
        )
        print(OmegaConf.to_yaml(cfg))

        trainer.fit(model, train_loader, valid_loader)

        metrics[fold] = trainer.validate(model, valid_loader)[0]
        print("^ VALIDATION: val metrics after fold", fold, metrics)

        torch.save(model.state_dict(), f"{workdir}/models/model{fold}.pt")

        preds = trainer.predict(model, test_loader)
        preds = torch.cat([batch.get("preds")
                          for batch in preds]).float().numpy()

        # unscale preds (in multi-property / all-property setting)
        if cfg.task.scale and surfpro.scaled:
            preds = surfpro.unscale(preds)
            print("preds rescaled to original units")

        # store per-fold predictions in test_df used in evaluate.py()
        for i, prop in enumerate(surfpro.propnames):
            test_df[f"{prop}_fold{fold}"] = preds[:, i]

    # store per-fold predictions in test_df used in evaluate.py()
    test_df.reset_index(drop=True).to_csv(f"{workdir}/test_preds_folds.csv")

    # evaluate best fold test set metrics
    best_fold = np.argmin([v["valid/mae"] for k, v in metrics.items()])
    metrics["best_fold"] = str(best_fold)
    print("best fold was fold", best_fold)

    metrics["test"] = trainer.test(model, test_loader)[0]
    with open(f"{workdir}/metrics.json", "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    train()
