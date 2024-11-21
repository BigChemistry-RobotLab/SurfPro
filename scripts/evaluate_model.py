import pandas as pd
import pickle
import json
import numpy as np
import hydra
from omegaconf import OmegaConf, DictConfig
from src.dataloader import SurfProDB, DataSplit
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


def calc_metrics(labels, preds):
    if np.isnan(labels).any():
        labels = labels[:70]
        preds = preds[:70]
    mae = mean_absolute_error(labels, preds)
    rmse = root_mean_squared_error(labels, preds)
    r2 = r2_score(labels, preds)
    return mae, rmse, r2


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def evaluate(cfg: DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))
    print("EVALUATE CONFIG from params.yaml")
    cfg = OmegaConf.load("./params.yaml")
    print(OmegaConf.to_yaml(cfg))

    dataroot = f"{cfg.host.workdir}/data/{cfg.data.task}"
    root = f"{cfg.host.workdir}/out/{cfg.data.task}"

    with open(f"{cfg.host.workdir}/data/{cfg.data.task}/surfpro.pkl", "rb") as f:
        surfpro = pickle.load(f)

    df_test = pd.read_csv(f"{root}/test_preds_folds.csv")
    for prop in surfpro.propnames:
        for fold in range(cfg.data.n_splits):
            assert f"{prop}_fold{fold}" in list(df_test.columns)

    results_dict = {}
    for prop in surfpro.propnames:
        results_dict[prop] = {}
        labels = df_test.loc[:, prop]

        # AVERAGE
        metrics = np.array(
            [
                calc_metrics(labels, df_test.loc[:, f"{prop}_fold{fold}"])
                for fold in range(cfg.data.n_splits)
            ]
        )
        print("metrics", metrics.shape)
        assert len(metrics) == cfg.data.n_splits
        assert len(metrics[:, 0]) == cfg.data.n_splits
        results_dict[prop]["avg_mae"] = np.mean(metrics[:, 0])
        results_dict[prop]["std_mae"] = np.std(metrics[:, 0])
        results_dict[prop]["avg_rmse"] = np.mean(metrics[:, 1])
        results_dict[prop]["std_rmse"] = np.std(metrics[:, 1])
        results_dict[prop]["avg_r2"] = np.mean(metrics[:, 2])
        results_dict[prop]["std_r2"] = np.std(metrics[:, 2])
        print("AVERAGE ", results_dict[prop])

        # ENSEMBLE
        ensemble_preds = np.mean(
            [df_test.loc[:, f"{prop}_fold{fold}"]
                for fold in range(cfg.data.n_splits)],
            axis=0,
        )
        ensemble_std_avg = np.mean(
            np.std(
                [
                    df_test.loc[:, f"{prop}_fold{fold}"]
                    for fold in range(cfg.data.n_splits)
                ],
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
        for fold in range(cfg.data.n_splits):
            mae, rmse, r2 = calc_metrics(
                labels, df_test.loc[:, f"{prop}_fold{fold}"])
            results_dict[prop][f"raw_mae_fold{fold}"] = mae
            results_dict[prop][f"raw_rmse_fold{fold}"] = rmse
            results_dict[prop][f"raw_r2_fold{fold}"] = r2

    # TODO put them into a DF, export to latex table,
    # pd.DataFrame(eval_data).to_csv(f"{root}/test_metrics.csv")
    results_dict["data"] = OmegaConf.to_container(cfg.data)
    results_dict["model"] = OmegaConf.to_container(cfg.model)
    print(results_dict)
    with open(f"{root}/results_test_metrics.json", "w") as f:
        json.dump(results_dict, f, indent=4, sort_keys=False)

    #########################
    # fill-the-table by filling NaN in raw table with predictions
    #########################

    df_raw = pd.read_csv(f"{dataroot}/df_raw.csv")
    print(df_raw.head(5))
    df_preds = pd.read_csv(f"{root}/df_ensemble_preds.csv")
    print(df_preds.head(5))

    eval_metrics = {}
    for split in ["tr_va", "test"]:
        # all folds are train/val for ensemble
        metrics = {}
        # indices = np.where(df_preds == split)[0]
        indices = np.where(df_preds["split"] == split)[0]

        # CALC ERRORS
        labels = np.array(df_raw.iloc[indices, :][surfpro.propnames])
        preds = np.array(df_preds.iloc[indices, :][surfpro.propnames])
        masks = np.where(np.isnan(np.array(labels)), 0, 1)

        preds = preds * masks
        labels = np.nan_to_num(labels, nan=0.0)

        # overall MAE/RMSE
        metrics["mae"] = mean_absolute_error(labels, preds)
        metrics["rmse"] = root_mean_squared_error(labels, preds)

        # calculate per-surfactant-type
        for types in df_preds["types"].unique():
            indices = np.where(
                (df_preds["split"] == split) & (df_preds["types"] == types)
            )[0]
            if len(indices) > 0:
                labels = np.array(df_raw.iloc[indices, :][surfpro.propnames])
                preds = np.array(df_preds.iloc[indices, :][surfpro.propnames])
                masks = np.where(np.isnan(np.array(labels)), 0, 1)

                preds = preds * masks
                labels = np.nan_to_num(labels, nan=0.0)

                metrics[f"mae_{types}"] = mean_absolute_error(labels, preds)
                metrics[f"rmse_{types}"] = root_mean_squared_error(
                    labels, preds)
            else:
                print(f"No entries in {split} for type {types}")

        # per-property MAE/RMSE
        if len(surfpro.propnames) > 1:
            for i, prop in enumerate(surfpro.propnames):
                metrics[f"mae_{prop}"] = mean_absolute_error(
                    labels[:, i], preds[:, i])
                metrics[f"rmse_{prop}"] = root_mean_squared_error(
                    labels[:, i], preds[:, i]
                )

        eval_metrics[split] = metrics

    with open(f"{root}/eval_metrics.json", "w") as f:
        json.dump(eval_metrics, f, indent=4, sort_keys=True)

    eval_data = {}
    for key in eval_metrics:
        eval_data[f"{key}_mae"] = {
            k[4:]: round(v, 4)
            for k, v in eval_metrics[key].items()
            if k.startswith("mae")
        }
    for key in eval_metrics:
        eval_data[f"{key}_rmse"] = {
            k[5:]: round(v, 4)
            for k, v in eval_metrics[key].items()
            if k.startswith("rmse")
        }
    pd.DataFrame(eval_data).to_csv(f"{root}/eval_metrics.csv")


if __name__ == "__main__":
    evaluate()
