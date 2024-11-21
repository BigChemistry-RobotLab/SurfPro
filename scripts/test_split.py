from omegaconf import OmegaConf
import pandas as pd
import numpy as np
from sklearn.model_selection import (
    StratifiedShuffleSplit,
    StratifiedKFold,
)


def map_surfactant_type(stype):
    type_map = {
        "cationic": "cationic",
        "gemini cationic": "gemini cationic",
        "anionic": "anionic",
        "gemini anionic": "anionic",
        "zwitterionic": "zwitterionic",
        "gemini zwitterionic": "zwitterionic",
        "non-ionic": "non-ionic",
        "sugar-based non-ionic": "sugar-based non-ionic",
    }
    return type_map.get(stype)


def test_split():
    cfg = OmegaConf.load("./params.yaml")

    # multi_props = ["pCMC", "AW_ST_CMC", "Gamma_max"]
    all_props = ["pCMC", "AW_ST_CMC",
                 "Gamma_max", "Area_min", "Pi_CMC", "PC20"]
    other = ["SMILES", "type", "temp"]

    df = pd.read_csv(
        f"{cfg.host.workdir}/data/surfpro_literature.csv", sep=",")
    # df["Gamma_max"] = df["Gamma_max"] * 1e6
    df["type"] = df["Surfactant_Type"].apply(map_surfactant_type)
    df["temp"] = df["Temp_Celsius"]

    uni, cnt = np.unique(df.type, return_counts=True)
    print("overall", list(zip(uni, np.round(cnt / len(df), 2))))

    df_all = df.iloc[
        np.all([pd.notna(df.loc[:, p]) for p in all_props], axis=0), :
    ].reset_index(drop=True)
    # print("all", len(df_all))
    # uni, cnt = np.unique(df_all.type, return_counts=True)
    # print(list(zip(uni, cnt, np.round(cnt / len(df_all), 2))))
    # print("all props subset", sum(cnt), len(df_all))
    # df_all.to_csv(
    #     f"{cfg.host.workdir}/data/debug/subset_allprop.csv", index=False)
    # df.to_csv(f"{cfg.host.workdir}/data/debug/subset_fulldf.csv", index=False)

    df_cmc = df.iloc[
        np.all(
            [pd.notna(df["pCMC"]), pd.isna(df["AW_ST_CMC"]),
             pd.isna(df["Gamma_max"])],
            axis=0,
        ),
        :,
    ].reset_index(drop=True)
    # print("cmc", len(df_cmc))
    # uni, cnt = np.unique(df_cmc.type, return_counts=True)
    # print("cmc only subset", sum(cnt), len(df_cmc))
    # print(list(zip(uni, cnt, np.round(cnt / len(df_cmc), 2))))

    test_splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=70, random_state=42)

    # split 70 from `all` subset into test_all
    idx_all_train, idx_all_test = next(
        test_splitter.split(df_all, df_all.type))
    print(len(idx_all_test), "<test, train>", len(idx_all_train))

    # split 70 from `cmc`-only subset into test_all
    idx_cmc_train, idx_cmc_test = next(
        test_splitter.split(df_cmc, df_cmc.type))
    print(len(idx_cmc_test), "<test, train>", len(idx_cmc_train))

    test_all = (
        df_all.loc[idx_all_test, other + all_props]
        .reset_index(drop=True)
        .sort_values(by=["type", "SMILES"])
    )

    # test_all.to_csv(
    #     f"{cfg.host.workdir}/data/surfpro_test_all.csv", index=False)

    test_cmc = (
        df_cmc.loc[idx_cmc_test, other + all_props]
        .reset_index(drop=True)
        .sort_values(by=["type", "SMILES"])
    )
    # test_cmc.to_csv(
    #     f"{cfg.host.workdir}/data/surfpro_test_cmc.csv", index=False)

    # make a test set with both test sets combined, where eval(cmc) is for both
    df_test = pd.concat([test_all, test_cmc], axis=0).reset_index(drop=True)
    df_test.to_csv(f"{cfg.host.workdir}/data/surfpro_test.csv", index=False)

    test_smiles = list(test_all.SMILES) + list(test_cmc.SMILES)
    assert len(set(test_smiles) ^ set(df_test.SMILES)) == 0

    df_train = (
        df.iloc[np.array([smi not in test_smiles for smi in df.SMILES])]
        .reset_index(drop=True)
        .sort_values(by=["type", "Ref_Gamma_max", "Ref_CMC", "SMILES"])
    )
    df_train = df_train.loc[:, other + all_props]
    # df_train.to_csv(f"{cfg.host.workdir}/data/surfpro_train.csv", index=False)

    cv_splitter = StratifiedKFold(n_splits=10, random_state=None)
    folds = cv_splitter.split(df_train, df_train.type)
    for fold_id, (_train_ix, valid_ix) in enumerate(folds):
        df_train.loc[valid_ix, "fold"] = int(fold_id)

    df_train.to_csv(f"{cfg.host.workdir}/data/surfpro_train.csv", index=False)

    # uni, cnt = np.unique(df_test.type, return_counts=True)
    # uni_, cnt_ = np.unique(df.type, return_counts=True)
    # print("test frac", list(zip(uni, cnt, np.round(cnt / cnt_, 3))))


if __name__ == "__main__":
    test_split()
