import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from src.converter import map_smiles_to_class
from omegaconf import OmegaConf
import dvc

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
def plot_distributions():  #cfg: DictConfig) -> None:
    print("PLOT CONFIG from params.yaml")
    cfg = OmegaConf.load("./params.yaml")
    print(OmegaConf.to_yaml(cfg))


    df = pd.read_csv(f'{cfg.host.workdir}/data/surfpro_literature.csv')
    df['type'] = df['Surfactant_Type'].apply(map_surfactant_type)

    types, types_count = np.unique(df.type, return_counts=True)
    print(df.type)
    print('\noverall counts:')
    [print(k,v) for k,v in list(zip(types, types_count))]

    properties = ['pCMC', 'AW_ST_CMC', 'Gamma_max', 'pC20']
    colormap = {
        "gemini cationic": 'tab:red',
        "cationic": 'tab:orange',
        "anionic": 'tab:green',
        "non-ionic": 'tab:blue', 
        # "zwitterionic": 'tab:purple',
    }

    for property in properties: 

        fig, axes = plt.subplots(3, 2)
        assert len(types) == 5
        all_types = df.loc[:, property]
        for i, stype in enumerate(types):
            if stype != 'zwitterionic':
                ax = axes.flatten[i]
                print('\n', property, stype)
                all_df = df.loc[:, ['SMILES', 'type', property]]
                all_df = all_df.dropna(axis=0).reset_index(drop=True)
                # sub_df = all_df.loc[all_df.where(all_df.loc[:, 'type'] == stype), :]
                sub_df = all_df.iloc[np.where(all_df['type'] == stype)]
                print(all_df.shape)
                print(sub_df.shape)
                print(sub_df)

                ax.hist(all_df.loc[:, 'property'], bins=20, label=stype, 
                        alpha = 0.3, color = cmap[stype], edgecolor='black')
                ax.hist(all_df.loc[:, 'property'], bins=20, label=stype, 
                        alpha = 0.7, color = cmap[stype], edgecolor='black')

        # ax.set_title(f'{property}: {stype} ', fontsize=15)
        ax.set_xlabel(prop, fontsize=14)
        ax.set_ylabel('Frequency', fontsize=14)
        ax.grid(axis='y', alpha=0.75)

    plt.tight_layout()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_distributions()












#
#
#
#
#
# properties = ['pCMC', 'T']  # Assuming 'T' is one of the properties (temperature)
# categories = data['category'].unique()
#
# # Create subplots, one for each category
# fig, axes = plt.subplots(len(categories), len(properties), figsize=(15, 5 * len(categories)))
#
# # Ensure axes is iterable in case of a single subplot
# if len(categories) == 1:
#     axes = [axes]
#
# for i, category in enumerate(categories):
#     subset = data[data['category'] == category]
#
#     for j, prop in enumerate(properties):
#         ax = axes[i][j] if len(properties) > 1 else axes[i]
#         # Plot the entire dataset in semi-transparent
#         ax.hist(data[prop], bins=20, alpha=0.3, color='gray', label='All Data', edgecolor='black')
#
#         # Overlay the subset in full color
#         ax.hist(subset[prop], bins=20, alpha=0.7, label=f'{category}', edgecolor='black')
#
#         # Add titles and labels
#         ax.set_title(f'{category} - {prop}', fontsize=14)
#         ax.set_xlabel(prop, fontsize=12)
#         ax.set_ylabel('Frequency', fontsize=12)
#         ax.grid(axis='y', alpha=0.75)
#
# # Adjust layout and show legend
# plt.tight_layout()
# plt.legend()
# plt.show()
#
