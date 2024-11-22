from copy import deepcopy
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
import pandas as pd
import numpy as np
from typing import List
from sklearn.preprocessing import RobustScaler
from torch_geometric.data import Data, Batch
from src.featurizer import RDKitGraphFeaturizer


class DataSplit(Dataset):
    def __init__(
        self,
        smiles: List[str],
        labels: List[float],
        types: List[str],
        propnames: List[str] = None,
        featurize: str = "graph",  # ['graph', 'rdkit', 'efcp']
    ):
        self.smiles = smiles
        self.labels = torch.tensor(labels, dtype=torch.float32)
        masks = np.where(np.isnan(np.array(labels)), 0, 1)
        self.masks = torch.tensor(masks, dtype=torch.int8)
        self.types = types
        self.propnames = propnames
        self.featurize = featurize

        if self.featurize == "graph":
            featurizer = RDKitGraphFeaturizer(
                bidirectional=True, self_loop=True)
            self.feats = featurizer(smiles)

        elif self.featurize == "rdkit":
            self.feats = [AllChem.RDKFingerprint(
                Chem.MolFromSmiles(s)) for s in smiles]

        elif self.featurize == "ecfp":
            featurizer = rdFingerprintGenerator.GetMorganGenerator(
                fpSize=2048, radius=2
            )
            self.feats = [
                featurizer.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles
            ]

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smiles = self.smiles[idx]
        feats = self.feats[idx]
        # feats = Batch.from_data_list(feats)
        labels = self.labels[idx]
        masks = self.masks[idx]
        types = self.types[idx]
        return {
            "feats": feats,
            "labels": labels,
            "masks": masks,
            "smiles": smiles,
            "types": types,
        }

    def collate(self, batch):
        smiles = [m.get("smiles") for m in batch]
        types = [m.get("types") for m in batch]

        feats = [m.get("feats") for m in batch]
        if self.featurize == "graph":
            feats = Batch.from_data_list(feats)

        masks = torch.stack([m.get("masks") for m in batch], axis=0)
        labels = torch.stack([m.get("labels") for m in batch], axis=0)
        labels = torch.nan_to_num(labels, nan=0.0)

        return {
            "feats": feats,
            "labels": labels,
            "masks": masks,
            "smiles": smiles,
            "types": types,
        }

    def loader(self, batch_size=64, shuffle=False, num_workers=8):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate,
        )


class SurfProDB(Dataset):
    def __init__(
        self,
        task="cmc",  # [all, multi, cmc, awst, gamma, pc20]
        workdir=".",
        featurize="graph",
        n_folds=10,
        scaled=None,  # only scales in multi-property tasks
    ):
        self.task = task
        self.workdir = workdir
        self.featurize = featurize
        self.n_folds = n_folds
        if scaled:
            self.scaled = scaled
        else:
            self.scaled = True if task in ["all", "multi"] else False

        train = pd.read_csv(f"{workdir}/data/surfpro_train.csv")
        test = pd.read_csv(f"{workdir}/data/surfpro_test.csv")

        # Gamma_max scaled by 1e6 in single-task setting
        train.loc[:, "Gamma_max"] = train.loc[:, "Gamma_max"] * 1e6
        test.loc[:, "Gamma_max"] = test.loc[:, "Gamma_max"] * 1e6

        # store pd.DataFrames of entire train/test for given task
        self.train_df = deepcopy(self.make_task(train))
        self.test_df = deepcopy(self.make_task(test))

        if self.scaled and len(self.propnames) > 1:
            print("Scaling ENABLED for multi-property prediction task")
            print("Don't forget to `surfpro.unscale(test_preds)`")
            self.make_scaler(self.train_df)
            train.loc[:, self.propnames] = self.scale(
                train.loc[:, self.propnames])
            test.loc[:, self.propnames] = self.scale(
                test.loc[:, self.propnames])

        # make datasplit for each train/val fold and test set
        self.train, self.valid = [], []
        self.test = self.make_datasplit(self.make_task(test))
        for fold in range(self.n_folds):
            print(f"featurizing fold {fold}")
            self.train.append(
                self.make_datasplit(
                    self.make_task(
                        train.iloc[np.where(train.fold != fold)[0], :])
                )
            )
            self.valid.append(
                self.make_datasplit(
                    self.make_task(
                        train.iloc[np.where(train.fold == fold)[0], :])
                )
            )

    def make_task(self, df):
        """selects properties and SMILES + type for a given 'task'.
        for single-property tasks drops all NaN entrys of that prop"""

        if self.task in ["all", None]:
            propnames = ["pCMC", "AW_ST_CMC",
                         "Gamma_max", "Area_min", "Pi_CMC", "PC20"]

        elif self.task in ["multi"]:
            propnames = ["pCMC", "AW_ST_CMC", "Gamma_max"]

        elif self.task in ["cmc", "pcmc", "CMC", "pCMC"]:
            propnames = ["pCMC"]

        elif self.task in ["awst", "aw_st_cmc", "AW_ST_CMC"]:
            propnames = ["AW_ST_CMC"]

        elif self.task in ["gamma", "gamma_max", "Gamma_max"]:
            propnames = ["Gamma_max"]

        elif self.task in ["pc20", "pC20", "PC20"]:
            propnames = ["PC20"]

        if len(propnames) == 1:
            df = df.loc[pd.notna(df.loc[:, propnames[0]])
                        ].reset_index(drop=True)

        self.propnames = propnames
        cols = ["SMILES", "type"] + propnames

        return df.loc[:, cols].reset_index(drop=True)

    def make_datasplit(self, df):
        return DataSplit(
            smiles=df.SMILES,
            labels=df.loc[:, self.propnames].to_numpy(),
            types=df.type,
            propnames=self.propnames,
            featurize=self.featurize,
        )

    def make_scaler(self, df):
        scaler = RobustScaler()
        scaler = scaler.fit(np.array(df.loc[:, self.propnames]))
        self.scaler = scaler
        self.center_ = scaler.center_
        self.scale_ = scaler.scale_

    def scale(self, y):
        return self.scaler.transform(y)

    def unscale(self, yhat):
        return self.scaler.inverse_transform(yhat)


if __name__ == "__main__":
    from omegaconf import OmegaConf
    import pickle

    cfg = OmegaConf.load("./params.yaml")

    surfpro = SurfProDB(
        task=cfg.data.task,
        workdir=cfg.host.workdir,
        featurize=cfg.model.featurize,
        n_folds=cfg.data.n_splits,
        scaled=cfg.data.scale,
    )

    with open(f"{cfg.host.workdir}/data/{cfg.data.task}/surfpro.pkl", "wb") as f:
        pickle.dump(surfpro, f)

    raw_df = pd.concat([surfpro.train_df, surfpro.test_df])
    df_raw = raw_df.sort_values(by="SMILES").reset_index(drop=True)
    df_raw.to_csv(f"{cfg.host.workdir}/data/{cfg.data.task}/df_raw.csv")

    # with open(f"{cfg.host.workdir}/data/{cfg.data.task}/surfpro.pkl", "rb") as f:
    #     surfpro = pickle.load(f)
    #
    # print(surfpro.train[0].smiles[:12])
    # print(surfpro.test.smiles[:12])

    # for task in ['multi', "pCMC", "gamma", "awst", 'pc20', "all"]:
    #     surfpro = SurfProDB(
    #         task=task,
    #         workdir=cfg.host.workdir,
    #         featurize="graph",
    #         n_folds=1,
    #     )
