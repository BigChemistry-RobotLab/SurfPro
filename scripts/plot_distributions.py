import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from omegaconf import OmegaConf


def map_surfactant_type(stype):
    type_map = {
        "cationic": "cationic",
        "gemini cationic": "gemini cationic",
        "anionic": "anionic",
        "gemini anionic": "anionic",
        "zwitterionic": "zwitterionic",
        "gemini zwitterionic": "zwitterionic",
        "non-ionic": "non-ionic",
        "sugar-based non-ionic": "non-ionic",
    }
    return type_map.get(stype)


# @hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def plot_distributions(arrange):  # cfg: DictConfig) -> None:
    print("PLOT CONFIG from params.yaml")
    cfg = OmegaConf.load("./params.yaml")
    print(OmegaConf.to_yaml(cfg))

    df = pd.read_csv(f'{cfg.host.workdir}/data/surfpro_literature.csv')
    df['type'] = df['Surfactant_Type'].apply(map_surfactant_type)

    # types, types_count = np.unique(df.type, return_counts=True)
    # print('\noverall counts:')
    # [print(k,v) for k,v in list(zip(types, types_count))]
    types = ['gemini cationic', 'cationic', 'non-ionic', 'anionic']

    properties = ['pCMC', 'AW_ST_CMC', 'Gamma_max', 'pC20']
    # units = ['pCMC (-log(M))', 'mN/m', 'mol/m^2 * 1e6', '-log(M)']
    properties_tex = [
        "pCMC",
        "$\gamma_{CMC}$",
        "$\Gamma_{max} \cdot 10^6$",
        "$pC_{20}$",
    ]

    colormap = {
        "full dataset": 'grey',
        "gemini cationic": 'tab:red',
        "cationic": 'tab:orange',
        "anionic": 'tab:green',
        "non-ionic": 'tab:blue',
        # "zwitterionic": 'tab:purple',
    }

    for property, tex in list(zip(properties, properties_tex)):

        if arrange == 'sq':
            fig, axes = plt.subplots(2, 2, figsize=(
                18, 10), sharex=False, sharey=True)
        elif arrange == 'row':
            fig, axes = plt.subplots(1, 4, figsize=(
                24, 6), sharex=True, sharey=True)

        axes = axes.flatten()
        for i, stype in enumerate(types):
            # if stype != 'zwitterionic':
            ax = axes[i]
            all_df = df.loc[:, ['SMILES', 'type', property]]
            all_df = all_df.dropna(axis=0).reset_index(drop=True)
            if property == 'Gamma_max':
                all_df.loc[:, property] = all_df.loc[:, property] * 1e6
            sub_df = all_df.iloc[np.where(all_df['type'] == stype)]

            print('\n', property, stype)
            print(all_df.shape)
            print(sub_df.shape)

            bins = np.linspace(
                all_df.loc[:, property].min(), all_df.loc[:, property].max(), 31)
            ax.hist(all_df.loc[:, property], bins=bins,
                    alpha=0.3, color='grey', edgecolor='black')
            ax.hist(sub_df.loc[:, property], bins=bins,
                    alpha=0.7, color=colormap[stype], edgecolor='black')

            all_median = np.median(all_df.loc[:, property])
            ax.axvline(all_median, color='dimgrey', linestyle='dashed', linewidth=2)
            sub_median = np.median(sub_df.loc[:, property])
            ax.axvline(sub_median, color=colormap[stype], linestyle='dashed', linewidth=2)

            if arrange == 'sq' and i % 2 == 0:
                ax.set_ylabel('Frequency', fontsize=18)
            elif arrange == 'row' and i == 0:
                ax.set_ylabel('Frequency', fontsize=18)

            if i >= 2 or arrange == 'row':
                ax.set_xlabel(tex, fontsize=18)

            blabel = chr(97 + i)  # 'a', 'b', 'c', 'd', ...
            ax.text(-0.01, 1.01, f"{blabel}.", transform=ax.transAxes,
                    fontsize=20, fontweight='bold', va='top', ha='right')

            ax.autoscale(enable=True, axis='both', tight=True)

            ax.grid(axis='y', alpha=0.5)
            ax.tick_params(axis="x", labelsize=13)
            ax.tick_params(axis="y", labelsize=13)

        # legend_patches = [
        #     Patch(facecolor=color, label=label, alpha=0.7)
        #     for label, color in colormap.items()
        # ]
        # axes[1].legend(handles=legend_patches, fontsize=18, loc="upper right")
        # plt.suptitle(f'Comparison of {
        #              property} counts by surfactant type', fontsize=20)
        plt.tight_layout()
        plt.savefig(
            f"{cfg.host.workdir}/results/plots/data_distrib_hist_{property}_{arrange}.png",
            dpi=300,
            bbox_inches="tight",
        )


if __name__ == "__main__":
    plot_distributions(arrange='sq')
    plot_distributions(arrange='row')
