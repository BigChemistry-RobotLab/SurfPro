import pandas as pd
import pickle
import numpy as np
import hydra
import json
from omegaconf import OmegaConf, DictConfig
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def plot_model_size(cfg: DictConfig) -> None:
    properties = ["pCMC", "AW_ST_CMC", "Gamma_max", "pC20"]
    properties_tex = [
        "$\mathrm{pCMC}$",
        "$\gamma_{\mathrm{CMC}}$ $(\mathrm{mN/m})$",
        "$\Gamma_{\mathrm{max}}$ $(\mathrm{mol/m^2}$ $\cdot$ $10^6)$",
        "$\mathrm{pC_{20}}$",
    ]

    models = ["AttentiveFP-32d", "AttentiveFP-64d", "AttentiveFP-96d"]
    multitask_models = [
        "AttentiveFP-32d-multi",
        "AttentiveFP-64d-multi",
        "AttentiveFP-96d-multi",
        "AttentiveFP-32d-all",
        "AttentiveFP-64d-all",
        "AttentiveFP-96d-all",
    ]

    # raw_mae = []
    ensemble_mae = {prop: [] for prop in properties}
    avg_mae = {prop: [] for prop in properties}
    std_mae = {prop: [] for prop in properties}

    froot = f"{cfg.host.masterdir}/results"
    abbrev_map = {
        "pCMC": "cmc",
        "AW_ST_CMC": "awst",
        "Gamma_max": "gamma",
        "pC20": "pc20",
        "multi": "multi",
        "all": "all",
    }

    latex_df = pd.DataFrame(
        columns=[
            "pCMC-MAE",
            "AW_ST_CMC-MAE",
            "Gamma_max-MAE",
            "pC20-MAE",
        ]
    )

    # Load single-task AttentiveFP model results
    for model in models:
        for prop in properties:
            abbrev = abbrev_map[prop]
            with open(
                f"{froot}/{abbrev}/{model}-{abbrev}/test_result_final.json",
                "r",
            ) as file:
                metrics = json.load(file)

                ensemble_mae[prop].append(metrics[prop]["ensemble_mae"])
                avg_mae[prop].append(metrics[prop]["avg_mae"])
                std_mae[prop].append(metrics[prop]["std_mae"])

                # logging results for latex table
                for mode in ["avg", "std", "ensemble"]:
                    latex_df.loc[f"{model}-single-{mode}", f"{prop}-MAE"] = metrics[
                        prop
                    ][f"{mode}_mae"]
                    # latex_df.loc[f"{model}-{mode}", f"{prop}-RMSE"] = metrics[prop][
                    #     f"{mode}_rmse"
                    # ]
                    # latex_df.loc[f'{model}-{mode}', f'{prop}-R2'] = \
                    #     metrics[prop][f'{mode}_r2']

    # Load multi-task AttentiveFP model results
    for model in multitask_models:
        abbrev = abbrev_map[model.split("-")[-1]]
        with open(
            f"{froot}/{abbrev}/{model}/test_result_final.json", "r"
        ) as file:  # results_test_metrics.json
            metrics = json.load(file)
            for prop in properties:
                if abbrev in ["multi"] and prop in ["pc20", "pC20"]:
                    ensemble_mae[prop].append(np.nan)
                    avg_mae[prop].append(np.nan)
                    std_mae[prop].append(np.nan)
                    # ensemble_rmse[prop].append(np.nan)
                    # avg_rmse[prop].append(np.nan)
                    continue
                # for i in range(cfg.data.n_splits):
                #     fold_mae = metrics[prop][f"raw_mae_fold{i}"]
                # raw_mae.append([prop, model, fold_mae])
                # fold_rmse = metrics[prop][f"raw_rmse_fold{i}"]
                # raw_rmse.append([prop, model, fold_rmse])
                ensemble_mae[prop].append(metrics[prop]["ensemble_mae"])
                avg_mae[prop].append(metrics[prop]["avg_mae"])
                std_mae[prop].append(metrics[prop]["std_mae"])
                # ensemble_rmse[prop].append(metrics[prop]["ensemble_rmse"])
                # avg_rmse[prop].append(metrics[prop]["avg_rmse"])

                # logging results for latex table
                for mode in ["avg", "std", "ensemble"]:
                    latex_df.loc[f"{model}-{mode}", f"{prop}-MAE"] = metrics[prop][
                        f"{mode}_mae"
                    ]
                    # latex_df.loc[f"{model}-{mode}", f"{prop}-RMSE"] = metrics[prop][
                    #     f"{mode}_rmse"
                    # ]
    print(latex_df.head())
    print(latex_df.index)

    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    for i, (prop, proptex) in enumerate(list(zip(properties, properties_tex))):
        ax = axes[i]

        model_settings = ["single", "multi", "all"]
        colors = ["tab:blue", "tab:orange", "tab:green"]
        model_sizes = [32, 64, 96]
        offset = -1
        for setting, color in list(zip(model_settings, colors)):
            if setting == "multi" and prop in ["pC20"]:
                continue

            avgs, stds, ensembles = [], [], []
            for size in model_sizes:
                model = f"AttentiveFP-{size}d-{setting}"
                avgs.append(latex_df.at[f"{model}-avg", f"{prop}-MAE"])
                stds.append(latex_df.at[f"{model}-std", f"{prop}-MAE"])
                ensembles.append(
                    latex_df.at[f"{model}-ensemble", f"{prop}-MAE"])

            sizes = [int(size) + offset for size in model_sizes]
            offset += 1  # 0.75

            ax.errorbar(
                x=sizes,
                y=avgs,
                yerr=stds,
                fmt="-o",
                color=color,
                capsize=3,
                markersize=8,
            )

            ax.plot(
                model_sizes,
                ensembles,
                linestyle="--",
                marker="o",
                color=color,
                markersize=8,
            )

            ax.set_xticks([32, 64, 96])
            ax.tick_params(axis="x", labelsize=13)
            ax.tick_params(axis="y", labelsize=13)

    axes[0].set_ylabel("Mean Absolute Error (MAE)", fontsize=16)

    off_x = 0.05
    off_y = 0.07
    fig.text(off_x + 0.0, off_y + 0.925, 'a.', fontsize=20, weight='bold')
    fig.text(off_x + 0.245, off_y + 0.925, 'b.', fontsize=20, weight='bold')
    fig.text(off_x + 0.49, off_y + 0.925, 'c.', fontsize=20, weight='bold')
    fig.text(off_x + 0.735, off_y + 0.925, 'd.', fontsize=20, weight='bold')
    fig.supxlabel("AttentiveFP hidden dimension", fontsize=16, y=0.03)

    # set up legend w / ensemble dash line
    handles, labels = axes[0].get_legend_handles_labels()

    for leg, col in list(zip(model_settings, colors)):
        legend_avg = Line2D([0], [0], color=col, linestyle="-", lw=2)
        handles.append(legend_avg)
        labels.append(leg)

    legend_avg = Line2D([0], [0], color="black",
                        linestyle="-", marker="o", lw=3, markersize=9)
    handles.append(legend_avg)
    labels.append("average")

    legend_ens = Line2D([0], [0], color="black",
                        linestyle="--", lw=2, markersize=9)  # marker="o",
    handles.append(legend_ens)
    labels.append("ensemble")
    ax.legend(handles=handles, labels=labels, loc="upper right", fontsize=16)

    plt.tight_layout()
    plt.savefig(
        f"{froot}/plots/model_size_comparison_all.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":
    plot_model_size()
