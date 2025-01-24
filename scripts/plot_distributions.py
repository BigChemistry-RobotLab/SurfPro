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
def plot_distributions():  # cfg: DictConfig) -> None:
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
    colormap = {
        "full dataset": 'grey',
        "gemini cationic": 'tab:red',
        "cationic": 'tab:orange',
        "anionic": 'tab:green',
        "non-ionic": 'tab:blue',
        # "zwitterionic": 'tab:purple',
    }

    for property in properties:

        fig, axes = plt.subplots(2, 2, figsize=(
            18, 10), sharex=False, sharey=True)
        axes = axes.flatten()
        for i, stype in enumerate(types):
            # if stype != 'zwitterionic':
            ax = axes[i]
            all_df = df.loc[:, ['SMILES', 'type', property]]
            all_df = all_df.dropna(axis=0).reset_index(drop=True)
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

            if i % 2 == 0:
                ax.set_ylabel('Frequency', fontsize=18)

            ax.autoscale(enable=True, axis='both', tight=True)
            # if property in ['pCMC', 'pC20']:
            #     ax.set_xlim(0, 7)
            #     ax.set_ylim(0, 140)

            ax.set_xlabel(stype, fontsize=18)
            ax.grid(axis='y', alpha=0.5)
            ax.tick_params(axis="x", labelsize=13)
            ax.tick_params(axis="y", labelsize=13)

        # legend_patches = [
        #     Patch(facecolor=color, label=label, alpha=0.7)
        #     for label, color in colormap.items()
        # ]
        # axes[1].legend(handles=legend_patches, fontsize=18, loc="upper right")
        plt.suptitle(f'Comparison of {
                     property} counts by surfactant type', fontsize=20)
        plt.tight_layout()
        plt.savefig(
            f"{cfg.host.workdir}/results/plots/data_distrib_hist_{property}.png",
            dpi=300,
            bbox_inches="tight",
        )


if __name__ == "__main__":
    plot_distributions()
