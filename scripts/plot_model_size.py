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
        "pCMC",
        "$\gamma_{CMC}$",
        "$\Gamma_{max} \cdot 10^6$",
        "$pC_{20}$",
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

    froot = f"{cfg.host.masterdir}/final"  # /results
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
                "r",  # results_test_metrics.json
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
        offset = -0.75
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
            offset += 0.75

            ax.errorbar(
                x=sizes,
                y=avgs,
                yerr=stds,
                fmt="-o",
                color=color,
                # label=f"{setting}\navg + std",
            )

            ax.plot(
                model_sizes,
                ensembles,
                linestyle="--",
                marker="o",
                color=color,
                # label=f"{setting}\nensemble",
            )

            ax.set_xticks([32, 64, 96])
            ax.tick_params(axis="x", labelsize=13)
            ax.tick_params(axis="y", labelsize=13)
            # ax.set_xticklabels([32, 64, 96], fontsize=12)

            # ax.yticks(fontsize=12)

        ax.set_title(
            f"{proptex}",
            fontsize=20,
        )
    axes[0].set_ylabel(
        f"Mean Absolute Error (MAE)",
        fontsize=16,
    )

    fig.supxlabel("AttentiveFP hidden dimension", fontsize=16, y=0.03)
    # ax.set_xlabel(
    #     f"Model size (AttentiveFP hidden dimension)",
    # )

    # fig.suptitle("Model size comparison", fontsize=18)  # , x=0.4)

    # set up legend w / ensemble dash line
    handles, labels = axes[0].get_legend_handles_labels()

    for leg, col in list(zip(model_settings, colors)):
        legend_avg = Line2D([0], [0], color=col, linestyle="-", lw=2)
        handles.append(legend_avg)
        labels.append(leg)

    legend_avg = Line2D([0], [0], color="black",
                        linestyle="-", marker="o", lw=2)
    handles.append(legend_avg)
    labels.append("average")

    legend_ens = Line2D([0], [0], color="black",
                        linestyle="--", lw=2)  # marker="o",
    handles.append(legend_ens)
    labels.append("ensemble")
    ax.legend(handles=handles, labels=labels, loc="upper right", fontsize=16)

    # handles, labels = axes[0].get_legend_handles_labels()
    # handles = handles[3:] + handles[:3]
    # labels = labels[3:] + labels[:3]

    # plt.legend(handles[3:], labels[3:], loc="upper right", fontsize=13)
    # plt.legend(handles[:3], labels[:3], loc="upper right", fontsize=13)
    # plt.legend(handles, labels, loc="center left",
    #            fontsize=13, bbox_to_anchor=(1, 0.5))
    # plt.tight_layout(rect=[0, 0, 0.85, 1])

    plt.tight_layout()

    # save
    plt.savefig(
        f"{froot}/plots/model_size_comparison_all.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # SINGLE PLOT PER PROPERTY
    # # fig, axes = plt.subplots(1, 4, figsize=(12, 6))
    # for i, (prop, proptex) in enumerate(list(zip(properties, properties_tex))):
    #     fig, ax = plt.subplots(figsize=(8, 6))
    #
    #     model_settings = ["single", "multi", "all"]
    #     colors = ["tab:blue", "tab:orange", "tab:green"]
    #     model_sizes = [32, 64, 96]
    #     offset = -0.5
    #     for setting, color in list(zip(model_settings, colors)):
    #         if setting == "multi" and prop in ["PC20"]:
    #             continue
    #
    #         avgs, stds, ensembles = [], [], []
    #         for size in model_sizes:
    #             model = f"AttentiveFP-{size}d-{setting}"
    #             avgs.append(latex_df.at[f"{model}-avg", f"{prop}-MAE"])
    #             stds.append(latex_df.at[f"{model}-std", f"{prop}-MAE"])
    #             ensembles.append(
    #                 latex_df.at[f"{model}-ensemble", f"{prop}-MAE"])
    #
    #         sizes = [int(size) + offset for size in model_sizes]
    #         offset += 0.5
    #         plt.errorbar(
    #             x=sizes,
    #             y=avgs,
    #             yerr=stds,
    #             fmt="-o",
    #             color=color,
    #             label=f"{setting}-property",
    #         )
    #
    #         plt.plot(sizes, ensembles, linestyle="--", color=color)
    #
    #     ax.set_title(
    #         f"Model size comparison: {proptex}",
    #         # f"Model Size Comparison",
    #         # - Test set {error.upper()} (N={
    #         # 140 if prop == 'pCMC' else 70})',
    #         fontsize=15,
    #     )
    #     # axes[0].set_ylabel(
    #     ax.set_ylabel(
    #         f"Mean Absolute Error (MAE)",
    #         fontsize=14,
    #     )
    #     plt.xticks([32, 64, 96])
    #
    #     # set up legend w/ ensemble dash line
    #     handles, labels = ax.get_legend_handles_labels()
    #     # legend_avg = Line2D([0], [0], color="black",
    #     #                     linestyle="-", marker="o", lw=2)
    #     # handles.append(legend_avg)
    #     # labels.append("average + std.dev")
    #
    #     legend_ens = Line2D([0], [0], color="black", linestyle="--", lw=2)
    #     handles.append(legend_ens)
    #     labels.append("ensemble")
    #     ax.legend(handles=handles, labels=labels,
    #               loc="upper right", fontsize=13)
    #
    #     # save
    #     plt.tight_layout()
    #     plt.savefig(
    #         f"{froot}/plots/model_size_comparison_{prop}.png",
    #         dpi=300,
    #         bbox_inches="tight",
    #     )
    #     plt.close(fig)


if __name__ == "__main__":
    plot_model_size()
