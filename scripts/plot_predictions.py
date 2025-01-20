import pandas as pd
import numpy as np
import hydra
from omegaconf import OmegaConf, DictConfig
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def plot_predictions(cfg: DictConfig) -> None:
    print("PLOT CONFIG from params.yaml")
    cfg = OmegaConf.load("./params.yaml")
    print(OmegaConf.to_yaml(cfg))

    dataroot = f"{cfg.host.workdir}/data/{cfg.task.name}"
    root = f"{cfg.host.workdir}/out/{cfg.task.name}"

    # load data and calculate errors
    df_raw = pd.read_csv(f"{dataroot}/df_raw.csv")
    print(df_raw.sample(5))
    df_preds = pd.read_csv(f"{root}/df_ensemble_preds.csv")

    labels = np.array(df_raw[cfg.task.props])
    masks = np.where(np.isnan(np.array(labels)), 0, 1)
    # labels = np.nan_to_num(labels, nan=0.0)

    preds = np.array(df_preds[cfg.task.props])
    preds = preds * masks

    props = cfg.task.props
    units = cfg.task.units

    types = df_preds.loc[:, "types"]
    # print("Surfacant Type", types)
    type_map = {typ: i for i, typ in enumerate(np.unique(types))}
    inv_map = {v: k for k, v in type_map.items()}
    types = np.array([type_map[typ] for typ in types])

    # assert df_raw['types'] == df_preds['types']
    assert (df_raw["SMILES"] == df_preds["SMILES"]).all()
    assert len(props) == len(units) == labels.shape[1] == preds.shape[1]

    def plot_parity(labels, preds, types, suffix):
        if cfg.task.name == "all":
            fig, axes = plt.subplots(2, 3, figsize=(20, 10))
            axes = axes.flatten()
            fontsize = 10
        elif cfg.task.name in ["multi"]:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            axes = axes.flatten()
            fontsize = 13
        elif cfg.task.name in ["cmc", "awst", "pc20", "gamma"]:
            fig, ax = plt.subplots(figsize=(8, 8))
            fontsize = 13

        for ix, _pname in enumerate(props):
            if cfg.task.name in ["all", "multi"]:
                ax = axes[ix]
            name = props[ix]
            unit = units[ix]

            notnan = np.where(np.isnan(labels[:, ix]), False, True)
            n_samples = np.sum(np.where(np.isnan(labels[:, ix]), 0, 1))

            y = labels[notnan, ix]
            yhat = preds[notnan, ix]

            # print("prop:", _pname, "n:", n_samples,
            #       len(y), len(yhat), "type", types[0])

            # n_samples = len(y[np.where(np.isnan(y) == 1)])
            if len(y) > 1 and len(yhat) > 1:
                mae = mean_absolute_error(y, yhat)
                # rmse = mean_squared_error(y, yhat, squared=False)
                rmse = root_mean_squared_error(y, yhat)
                r2 = r2_score(y, yhat)

                # plot a hexagonal parity plot
                ally = np.concatenate([y, yhat], axis=0)
                lim = [np.floor(np.min(ally)), np.ceil(np.max(ally))]
            else:
                mae, rmse, r2 = 0.0, 0.0, 0.0
                lim = [0.0, 1.0]

            unique_types = np.unique(types)
            for utype in unique_types:
                mask = types[notnan] == utype
                sc = ax.scatter(
                    y[mask], yhat[mask], alpha=0.8, label=f"{inv_map[utype]}"
                )

            ax.set_xlim(lim)
            ax.set_ylim(lim)

            sns.regplot(x=lim, y=lim, ax=ax, color="grey",
                        ci=None, scatter=False)

            if ix == 0:
                handles, labels_legend = ax.get_legend_handles_labels()

            unit = f" [{unit}]" if unit != "" else ""
            ax.set_title(f"{name}{unit} - {suffix} set")
            ax.set_xlabel(f"Experimental {name} [{unit}]", fontsize=fontsize)
            ax.set_ylabel(f"Model {name} [{unit}]", fontsize=fontsize)
            txt = f"RMSE = {rmse: .3f}\nMAE = {
                mae: .3f}\nR2 = {r2: .3f}\nn = {n_samples}"
            ax.text(lim[1], lim[0], txt, ha="right",
                    va="bottom", fontsize=fontsize)

        # fig.legend(loc='upper left')
        fig.legend(handles, labels_legend, loc="upper left")
        fig.subplots_adjust(top=0.95, hspace=0.25, wspace=0.25)
        fig.tight_layout()
        if suffix in ["train", "valid", "test", "all"]:
            fig.savefig(f"{root}/plots/parity_plots_{suffix}.png")
        else:
            fig.savefig(f"{root}/plots/parity/parity_plots_{suffix}.png")
        fig.clf()

    plot_parity(labels, preds, types, suffix="all")

    for split in ["tr_va", "test"]:
        # indices = np.where(df_preds == split)[0]
        indices = np.where(df_preds["split"] == split)[0]
        plot_parity(labels[indices], preds[indices],
                    types[indices], suffix=f"{split}")

        #######################
        # per-type parity plots
        #######################
        # for utypes in df_preds["types"].unique():
        #     indices = np.where(
        #         (df_preds["split"] == split) & (df_preds["types"] == utypes)
        #     )[0]
        #     if len(indices) > 1:
        #         plot_parity(
        #             labels[indices],
        #             preds[indices],
        #             types[indices],
        #             suffix=f"{split}_{utypes}",
        #         )


if __name__ == "__main__":
    plot_predictions()
