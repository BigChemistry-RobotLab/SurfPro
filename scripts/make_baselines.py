from omegaconf import OmegaConf
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import numpy as np
import json
from src.dataloader import SurfProDB
import os


def calc_metrics(preds, labels):
    if np.isnan(labels).any():
        labels = labels[:70]
        preds = preds[:70]
    mae = mean_absolute_error(preds, labels)
    rmse = root_mean_squared_error(preds, labels)
    r2 = r2_score(preds, labels)
    return mae, rmse, r2


def train_sklearn(task, feat, modelname):
    # cfg = OmegaConf.load("./params.yaml")

    workdir = "."  # cfg.host.masterdir
    n_splits = 10  # cfg.task.n_splits

    surfpro = SurfProDB(
        task=task, workdir=workdir, featurize=feat, n_folds=n_splits, scaled=False
    )

    # get prop name since single-task only
    prop = surfpro.propnames[0]

    test_df = surfpro.test_df
    print("test", surfpro.test)
    for i in range(n_splits):
        print("train", i, "\t", len(surfpro.train[i].smiles))
        print("valid", i, "\t", len(surfpro.valid[i].smiles))

    if modelname == "rf":
        model = RandomForestRegressor(
            n_estimators=100, min_samples_split=2, min_samples_leaf=1, random_state=42
        )
    elif modelname == "svr":
        model = SVR()
    elif modelname == "ridge":
        model = Ridge()
    elif modelname == "gpr":
        model = GaussianProcessRegressor()
    else:
        raise NotImplementedError

    best_fold = 0
    best_mae = 1e10

    test = surfpro.test
    for fold in range(n_splits):
        train = surfpro.train[fold]
        valid = surfpro.valid[fold]

        model = model.fit(train.feats, train.labels)
        val_preds = model.predict(valid.feats)
        mae, rmse, r2 = calc_metrics(val_preds, valid.labels)
        if mae < best_mae:
            best_mae = mae
            best_fold = fold

        preds = model.predict(test.feats)
        mae, rmse, r2 = calc_metrics(preds, test.labels)
        print("test", mae, rmse, r2)
        test_df[f"{prop}_fold{fold}"] = preds

    results_dir = f"{workdir}/final/{task}/{feat}-{modelname}"
    os.makedirs(results_dir, exist_ok=True)
    test_df.reset_index(drop=True).to_csv(
        f"{results_dir}/test_preds_folds.csv")

    ##########################################
    # EVALUATE - directly adapted from evaluate.py
    df_test = test_df

    results_dict = {}
    results_dict[prop] = {}
    labels = df_test.loc[:, prop]

    # AVERAGE
    metrics = np.array(
        [
            calc_metrics(labels, df_test.loc[:, f"{prop}_fold{fold}"])
            for fold in range(n_splits)
        ]
    )
    print("metrics", metrics.shape)
    assert len(metrics[:, 0]) == n_splits
    results_dict[prop]["avg_mae"] = np.mean(metrics[:, 0])
    results_dict[prop]["std_mae"] = np.std(metrics[:, 0])
    results_dict[prop]["avg_rmse"] = np.mean(metrics[:, 1])
    results_dict[prop]["std_rmse"] = np.std(metrics[:, 1])
    results_dict[prop]["avg_r2"] = np.mean(metrics[:, 2])
    results_dict[prop]["std_r2"] = np.std(metrics[:, 2])
    print("AVERAGE ", results_dict[prop])

    # ENSEMBLE
    ensemble_preds = np.mean(
        [df_test.loc[:, f"{prop}_fold{fold}"] for fold in range(n_splits)],
        axis=0,
    )
    ensemble_std_avg = np.mean(
        np.std(
            [df_test.loc[:, f"{prop}_fold{fold}"] for fold in range(n_splits)],
            axis=0,
        )
    )
    print(len(ensemble_preds), len(labels))
    assert len(ensemble_preds) == len(labels)
    mae, rmse, r2 = calc_metrics(labels, ensemble_preds)
    print("ENSEMBLE ", prop, "mae", mae, "rmse", rmse, "r2", r2)
    results_dict[prop]["ensemble_mae"] = mae
    results_dict[prop]["ensemble_rmse"] = rmse
    results_dict[prop]["ensemble_r2"] = r2
    results_dict[prop]["ensemble_std_avg"] = ensemble_std_avg

    # RAW per-fold
    for fold in range(n_splits):
        mae, rmse, r2 = calc_metrics(
            labels, df_test.loc[:, f"{prop}_fold{fold}"])
        results_dict[prop][f"raw_mae_fold{fold}"] = mae
        results_dict[prop][f"raw_rmse_fold{fold}"] = rmse
        results_dict[prop][f"raw_r2_fold{fold}"] = r2

    results_dict["best_fold"] = best_fold
    print(results_dict)
    with open(f"{results_dir}/results_test_metrics.json", "w") as f:
        json.dump(results_dict, f, indent=4, sort_keys=False)


if __name__ == "__main__":
    for task in ["cmc", "awst", "gamma", "pc20"]:
        for feat in ["ecfp", "rdkit"]:
            for modelname in ["ridge", "svr", "rf", "gpr"]:
                print("training", task, feat, modelname)
                train_sklearn(task=task, feat=feat, modelname=modelname)
