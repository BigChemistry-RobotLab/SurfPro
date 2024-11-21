# SurfPro 
### A curated database and predictive model of experimental properties of surfactants
`SurfPro` is a *surf*actant *pro*perty database database of 1624 surfactants and their physical properties, including 1395 experimental measurements of the critical micelle concentration (CMC), 972 air-water surface tension at CMC ($\gamma_{CMC}$) and further properties for over 650 surfactants curated from ~160 literature sources.

In addition to the database, this repository contains all code to reproduce the model training, evaluation and prediction pipeline for the AttentiveFP models and baselines used in the paper.

**Please cite our work if you wish to use any of the data sets**.

## SurfPro Database 
The SurfPro database is provided in three files. `surfpro_literature.csv` is the source file with all 1624 unique surfactant structures (SMILES), their curated experimental properties from literature, as well as the reference / DOI for each property / structure.
We further provide the `surfpro_train.csv` and `surfpro_test.csv` files ready for model training. 
surfpro_test contains a total of 140 surfactant structures, with 140 pCMC measurements, as well as all 6 properties for a subset of 70 structures.
These were obtained through stratified sampling by `surfactant type` from the full database (see Methods in the Paper for more details).

We also provide `surfpro_imputed.csv` file with imputed values from the multi-property ensemble predictions & uncertainties.
```
data/surfpro_literature.csv
data/surfpro_train.csv
data/surfpro_test.csv
data/surfpro_imputed.csv
```

## Properties
We collected 6 properties of surfactants in the database, which are derived from the experimentally determined Langmuir isotherm (see Figure below).
647 structures have all 6 properties reported, while all other 977 structures have partially missing properties.

| Property       | Database Name    | Property Name                        | Unit     | Count |
|----------------|------------------|--------------------------------------|----------|-------|
| CMC \| pCMC    | CMC \| pCMC      | critical micelle concentration       | M        | 1395  |
| $\gamma_{CMC}$ | AW\_ST\_CMC      | (air-water) surface tension at CMC   | $mN/m$   | 972   |
| $\Gamma_{max}$ | Gamma\_max       | maximum surface excess concentration | $mol/m^2$| 672   |
| C20 \| pC20    | C20 \| pC20      | adsorption efficiency                | M        | 657   |
| $A_{min}$      | Area\_CMC        | area at the air-water interface      | $nm^2$   | 678   |
| $\pi_{CMC}$    | Pi\_CMC          | surface pressure at CMC              | $mN/m$   | 744   |

pCMC refers to the negative log10 of the CMC in Molar ([mol/L]): 
$\mathrm{pCMC} = -\log(\mathrm{CMC})$.

Similarly, C20 refers to the inverse of pC20: 
$C_{20} = 10^{-\mathrm{pC_{20}}}$.


## The Langmuir isotherm
<img src="figs/langmuir_isotherm.png" alt="Langmuir Isotherm" width="100%">

Schematic visualization of the Langmuir isotherm based on the Szyszkowski equation and derived properties. 
Surfactant molecules adsorb to the air-water interface and lower the surface tension. 
With increasing surfactant concentration (x-axis, log scale) the surface tension $\gamma$ (y-axis) decreases until the interface is saturated and $\gamma$ stops decreasing further. 
Beyond this critical point, surfactants self-assemble into micelles.
Surfactant properties can be extracted from this experimentally determined isotherm: the critical micelle concentration (CMC) and the surface tension at the CMC ($\gamma_{CMC}$).
C20 is defined as the surfactant concentration required to reduce the surface tension $\gamma_0$ (72mN/m for water at room temperature) by 20 mN/m, which quantifies the surfactant's efficiency.
$\Gamma_{max}$ characterizes the slope at the steepest descent of the isoterm (shown in orange), which is assumed to be at $\gamma_{20}$.
The area of the surfactant at the air-water interface $A_{min}$ and the surface pressure at CMC $\pi_{CMC}$ can also be determined from the isotherm (not visualized).

## Dataloader
We provide a standalone dataloader in [src/dataloader.py](src/dataloader.py), which transforms [surfpro_train.csv](surfpro_train.csv) and [surfpro_test.csv](surfpro_test.csv) into ready-to-use featurized data splits, using the same 10 train/validation folds with featurization for GNNs, ECFP and RDKit (defined in [src.dataloader.SurfProDB](src.dataloader.SurfProDB) and [src.dataloader.DataSplit](src.dataloader.DataSplit)). 

The [scripts/make_baselines.py](scripts/make_baselines.py) script uses the dataloader with established ML models (RandomForest and Ridge) using ECFP or RDKit fingerprints as input features.

The graph neural network (AttentiveFP) training script ([scripts/train_model.py](scripts/train_model.py)) leverages the same dataloader as part of the DVC pipeline.

## Tasks
It also demonstrates the different **tasks** defined: 
- single-property pCMC [[task='cmc']](conf/data/cmc.yaml)
- single-property AW_ST_CMC [[task='awst']](conf/data/awst.yaml)
- single-property Gamma_max [[task='gamma']](conf/data/gamma.yaml)
- single-property pC20 [[task='pc20']](conf/data/pc20.yaml)
- multi-property [[task='multi']](conf/data/multi.yaml)
- all-property [[task='all']](conf/data/all.yaml)

## Imputed database
Using the trained all-property ensemble model, we *imputed* all missing values for all 977 incomplete surfactants, and provide the predictions (mean) and their uncertainties (standard deviation) in `data/surfpro_imputed.csv`. 

# Reproducing results of the paper 
### Conda setup 
```
git clone https://github.com/BigChemistry-RobotLab/SurfPro.git
cd SurfPro
conda create -n surfpro
conda activate surfpro
conda install pip
pip install -r requirements.txt

dvc config hydra.enabled=True

dvc exp run -S 'data=multi' -S 'model=attfp-64d' -S 'host=local' -S 'host.masterdir="/path/to/your/SurfPro"'
```

## DVC + Hydra
This project's pipeline leverages DVC and Hydra to configure the [params.yaml](params.yaml) file on-the-fly for experiments.

The full pipeline being executed is defined in [dvc.yaml](dvc.yaml), and copies the final outputs from ./out/{...} (or the temp run directory when using --queue) into /pathto/SurfPro/results/{task}/{model}/...
For this, the absolute path is necessary, which is defined in [params.yaml](params.yaml) as host.masterdir, in contrast to the data.workdir used within an experiment. You will need to change this config file to successfully run the full 'dvc queue' pipeline.

An individual model run (model variant * task) can be executed via dvc with data={...} and model={...} overrides of config files found in /conf/data/* and /conf/model/*.
You can additionally override individual configurations: '-S 'host.device=[1]'. For `dvc queue`, the setup requires the workdir to be '.' since dvc executes the run in a temporary directory.


### reproduce all AttentiveFP model experiments
Trains and evaluates the AttentiveFP model [conf/model/attfp-32d.yaml](conf/model/attfp-32d.yaml) / [conf/model/attfp-64d.yaml](conf/model/attfp-64d.yaml) / [conf/model/attfp-96d.yaml](conf/model/attfp-96d.yaml).
```
dvc exp run --queue -S 'data=all,multi,cmc,awst,gamma,pc20' -S 'model=attfp-32d,attfp-64d,attfp-96d' -S 'host=queue'

# or directly override: [...] -S host.masterdir="/path/to/your/SurfPro"'
```

### reproduce all baselines experiments
```
python scripts/make_baselines.py
```

### run a single AttentiveFP experiment, possibly overriding specific configuration
```
dvc exp run -S 'data=cmc' -S 'model=attfp-32d' -S 'model.n_epochs=100' -S 'data.n_splits=2'
```

### run multiple experiments using dvc queue
```
dvc queue status
dvc queue logs NAME --follow
dvc queue remove --all

dvc exp run --queue -S 'host.workdir="."' -S 'host.device=[1]' -S 'data=all,multi,cmc,awst,gamma,pc20' -S 'data.n_splits=10' -S 'model.n_epochs=500' -S 'model.hidden_channels=64' -S 'model.out_channels=128'

dvc exp run --queue -S 'data=cmc' -S 'model=attfp-64d' -S 'model.n_epochs=1000' -S 'model.dropout=0.0,0.1,0.2' -S 'model.num_timesteps=2' -S 'model.num_layers=2,3,4' -S 'data.n_splits=10' -S 'host.device=[0]'
```

### process train/test.csv into featurized DataSplits using params.yaml & conf/data/*
```
python src/dataloader.py
# writes surfpro.pickle file to 
# {cfg.host.workdir}/data/{cfg.data.task}/surfpro.pkl
```


## DVC credentials and wandb 
DVC requires git credentials to create local commits tracking experiments. 
You can use arbitrary credentials if you do not want to provide yours.
```
git config user.name "Your Name"
git config user.email "you@example.com"
```
Similarly, this codebase leverages `wandb` to track experiments.
You can set it to run offline / disabled if you do not want to use wandb,
or replace the `wandblogger` with another logger used by the pl.Trainer in train_model.py.
```
wandb offline
# or
wandb disabled
```

