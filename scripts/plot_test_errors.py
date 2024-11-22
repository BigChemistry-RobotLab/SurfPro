import json
import pandas as pd
import numpy as np
from omegaconf import OmegaConf, DictConfig
import matplotlib.pyplot as plt
import hydra
plt.rc('text', usetex=True)


def rename_models(models: list[str], prepend_r=False):
    ''' formats model name into latex for plotting '''
    fmodels = []
    for model in models:
        print(model)
        # AttentiveFP model variants
        model = model.replace('-32d', '_{32d}')
        model = model.replace('-64d', '_{64d}')
        model = model.replace('-96d', '_{96d}')
        model = model.replace('AttentiveFP', '\mathrm{AttentiveFP}')

        if 'multi' in model:
            model = model.replace('-multi', '^{multi}')
        elif 'all' in model:
            model = model.replace('-all', '^{all}')
        elif 'AttentiveFP' in model:
            model = model + '^{single}'

        # baselines
        if 'ecfp' in model:
            model = model.replace('ecfp', 'ECFP')
        elif 'rdkit' in model:
            model = model.replace('rdkit', 'RDKit')

        model = model.replace('ridge', 'Ridge')
        model = model.replace('rf', 'RF')

        if 'AttentiveFP' in model:
            if prepend_r:
                model = 'r"$' + model + '$"'
            else:
                model = '$' + model + '$'
        print(model)
        fmodels.append(model)
    return fmodels


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def plot_test_errors(cfg: DictConfig) -> None:
    print("TRAIN CONFIG from params.yaml")
    cfg = OmegaConf.load("./params.yaml")
    print(OmegaConf.to_yaml(cfg))

    # dataroot = f"{cfg.host.workdir}/data/{cfg.task.name}"
    root = f"{cfg.host.workdir}/results/{cfg.task.name}"

    # Load data from JSON files
    properties = ['pCMC', 'AW_ST_CMC', 'Gamma_max', 'PC20']

    properties_tex = [
        'pCMC', '$\gamma_{CMC}$ (AW_ST_CMC)', '$\Gamma_{max} \cdot 10^6$ (Gamma_max) ', '$pC_{20}$']
    models = ['AttentiveFP-32d', 'AttentiveFP-64d', 'AttentiveFP-96d']  #
    multitask_models = ['AttentiveFP-32d-multi',
                        'AttentiveFP-64d-multi',
                        'AttentiveFP-96d-multi',
                        'AttentiveFP-32d-all',
                        'AttentiveFP-64d-all',
                        'AttentiveFP-96d-all']
    baselines = ['rdkit-ridge', 'rdkit-rf',
                 'ecfp-ridge', 'ecfp-rf',]
    raw_mae = []
    ensemble_mae = {prop: [] for prop in properties}
    avg_mae = {prop: [] for prop in properties}

    raw_rmse = []
    ensemble_rmse = {prop: [] for prop in properties}
    avg_rmse = {prop: [] for prop in properties}

    froot = f'{cfg.host.masterdir}/results'
    abbrev_map = {
        'pCMC': 'cmc',
        'AW_ST_CMC': 'awst',
        'Gamma_max': 'gamma',
        'PC20': 'pc20',
        'multi': 'multi',
        'all': 'all'
    }

    latex_df = pd.DataFrame(
        columns=[
            'pCMC-MAE', 'pCMC-RMSE',
            'AW_ST_CMC-MAE', 'AW_ST_CMC-RMSE',
            'Gamma_max-MAE', 'Gamma_max-RMSE',
            'PC20-MAE', 'PC20-RMSE',
        ]
    )

    # Load single-task AttentiveFP model results
    for model in models:
        for prop in properties:
            abbrev = abbrev_map[prop]
            with open(f'{froot}/{abbrev}/{model}-{abbrev}/test_result_final.json', 'r') as file:
                metrics = json.load(file)
                for i in range(cfg.task.n_splits):
                    fold_mae = metrics[prop][f'raw_mae_fold{i}']
                    raw_mae.append([prop, model, fold_mae])

                    fold_rmse = metrics[prop][f'raw_rmse_fold{i}']
                    raw_rmse.append([prop, model, fold_rmse])

                ensemble_mae[prop].append(metrics[prop]['ensemble_mae'])
                avg_mae[prop].append(metrics[prop]['avg_mae'])

                ensemble_rmse[prop].append(metrics[prop]['ensemble_rmse'])
                avg_rmse[prop].append(metrics[prop]['avg_rmse'])

                # logging results for latex table
                for mode in ['avg', 'ensemble']:
                    latex_df.loc[f'{model}-{mode}', f'{prop}-MAE'] = \
                        metrics[prop][f'{mode}_mae']
                    latex_df.loc[f'{model}-{mode}', f'{prop}-RMSE'] = \
                        metrics[prop][f'{mode}_rmse']
                    # latex_df.loc[f'{model}-{mode}', f'{prop}-R2'] = \
                    #     metrics[prop][f'{mode}_r2']

    # Load multi-task AttentiveFP model results
    for model in multitask_models:
        abbrev = abbrev_map[model.split('-')[-1]]
        with open(f'{froot}/{abbrev}/{model}/test_result_final.json', 'r') as file:
            metrics = json.load(file)
            for prop in properties:
                if abbrev in ['multi'] and prop in ['pc20', 'PC20']:
                    # TODO ADD NaN's
                    ensemble_mae[prop].append(np.nan)
                    avg_mae[prop].append(np.nan)
                    ensemble_rmse[prop].append(np.nan)
                    avg_rmse[prop].append(np.nan)
                    continue
                for i in range(cfg.task.n_splits):
                    fold_mae = metrics[prop][f'raw_mae_fold{i}']
                    raw_mae.append([prop, model, fold_mae])
                    fold_rmse = metrics[prop][f'raw_rmse_fold{i}']
                    raw_rmse.append([prop, model, fold_rmse])
                ensemble_mae[prop].append(metrics[prop]['ensemble_mae'])
                avg_mae[prop].append(metrics[prop]['avg_mae'])
                ensemble_rmse[prop].append(metrics[prop]['ensemble_rmse'])
                avg_rmse[prop].append(metrics[prop]['avg_rmse'])

                # logging results for latex table
                for mode in ['avg', 'ensemble']:
                    latex_df.loc[f'{model}-{mode}', f'{prop}-MAE'] = \
                        metrics[prop][f'{mode}_mae']
                    latex_df.loc[f'{model}-{mode}', f'{prop}-RMSE'] = \
                        metrics[prop][f'{mode}_rmse']
                    # latex_df.loc[f'{model}-{mode}', f'{prop}-R2'] = \
                    #     metrics[prop][f'{mode}_r2']

    # Load single-task baseline model results
    for model in baselines:
        for prop in properties:
            abbrev = abbrev_map[prop]
            with open(f'{froot}/{abbrev}/{model}/test_result_final.json', 'r') as file:
                metrics = json.load(file)
                for i in range(cfg.task.n_splits):
                    fold_mae = metrics[prop][f'raw_mae_fold{i}']
                    raw_mae.append([prop, model, fold_mae])
                    fold_rmse = metrics[prop][f'raw_rmse_fold{i}']
                    raw_rmse.append([prop, model, fold_rmse])

                ensemble_mae[prop].append(metrics[prop]['ensemble_mae'])
                avg_mae[prop].append(metrics[prop]['avg_mae'])
                ensemble_rmse[prop].append(metrics[prop]['ensemble_rmse'])
                avg_rmse[prop].append(metrics[prop]['avg_rmse'])

                # logging results for latex table
                for mode in ['avg', 'ensemble']:
                    latex_df.loc[f'{model}-{mode}', f'{prop}-MAE'] = \
                        metrics[prop][f'{mode}_mae']
                    latex_df.loc[f'{model}-{mode}', f'{prop}-RMSE'] = \
                        metrics[prop][f'{mode}_rmse']
                    # latex_df.loc[f'{model}-{mode}', f'{prop}-R2'] = \
                    #     metrics[prop][f'{mode}_r2']

    latex_df.insert(0, 'model', latex_df.index.to_series().apply(
        lambda s: '-'.join(s.split('-')[:-1])))
    latex_df['model'] = rename_models(latex_df['model'])
    latex_df.insert(1, 'mode', latex_df.index.to_series().apply(
        lambda s: s.split('-')[-1]))

    print(latex_df)
    latex_df.to_latex(f"{froot}/plots/results_table.txt", float_format='%.3f',
                      na_rep=' - ', index=False)

    # Convert data to a DataFrame
    df_mae = pd.DataFrame(raw_mae, columns=['Property', 'Model', 'Results'])
    df_mae['Results'] = df_mae['Results'].astype(float)

    df_rmse = pd.DataFrame(raw_rmse, columns=['Property', 'Model', 'Results'])
    df_rmse['Results'] = df_rmse['Results'].astype(float)

    ########################
    # boxplot
    ########################
    for error, df_raw, avg_preds, ensemble_preds in list(
        zip(['mae', 'rmse'],
            [df_mae, df_rmse],
            [avg_mae, avg_rmse],
            [ensemble_mae, ensemble_rmse])):

        for prop, proptex in list(zip(properties, properties_tex)):
            plotted_models = []
            fig, ax = plt.subplots(figsize=(10, 6))
            position = 0

            if prop in ['PC20']:
                colors = ['coral'] * 3 + \
                    ['darkorange'] * 3 + ['dodgerblue'] * 4
            else:
                colors = ['coral'] * 3 + ['darkorange'] * 3 + \
                    ['chocolate'] * 3 + ['dodgerblue'] * 4

            for model_idx, model in enumerate(models + multitask_models + baselines):

                if 'multi' in model and prop in ['PC20']:
                    continue

                plotted_models.append(model)
                position += 1

                subset = df_raw[(df_raw['Property'] == prop) & (
                    df_raw['Model'] == model)]['Results']
                boxplt = ax.boxplot(subset, positions=[position],
                                    widths=0.6, patch_artist=True)

                boxplt['boxes'][0].set_facecolor(colors[position-1])
                boxplt['medians'][0].set_color('black')  # Set to dark blue

                avg_value = avg_preds[prop][model_idx]
                ax.plot(position, avg_value, marker='o',
                        color='navy', markersize=8, linestyle='None',
                        label='Average' if position == 1 else None)

                ensemble_value = ensemble_preds[prop][model_idx]
                ax.plot(position, ensemble_value, marker='*',
                        color='red', markersize=8, linestyle='None',
                        label='Ensemble' if position == 1 else None)

            # Customizing the plot
            ax.set_xticks(range(1, len(plotted_models)+1))
            ax.set_xticklabels(
                # plotted_models,
                rename_models(plotted_models, prepend_r=False),
                rotation=45, ha='right', fontsize=12)

            handles, labels = ax.get_legend_handles_labels()
            handles.append(boxplt['fliers'][0])
            labels.append('Outliers')
            ax.legend(handles=handles, labels=labels,
                      loc='lower right', fontsize=13)

            if error == 'mae':
                ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=14)
            elif error == 'rmse':
                ax.set_ylabel('Root Mean Squared Error (RMSE)', fontsize=14)
            ax.set_title(
                f'{proptex} Model Comparison - Test set {error.upper()} (N={
                    140 if prop == 'pCMC' else 70})',
                fontsize=15)
            plt.tight_layout()
            plt.savefig(f"{froot}/plots/test_box_{prop}_{error}.png",
                        dpi=300, bbox_inches="tight")
            plt.close(fig)


if __name__ == "__main__":
    plot_test_errors()
