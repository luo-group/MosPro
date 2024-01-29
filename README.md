# Pareto-optimal sampling for multi-objective protein sequence design

## Download data and oracles
Download the data and oracles from Dropbox and unzip them to the main directory
```
wget https://www.dropbox.com/scl/fi/v6rsdcnah88vfp4wkmn9z/data.zip?rlkey=atphgvxj5acwzmya9vp0uemdw&dl=0 -O data.zip
wget https://www.dropbox.com/scl/fi/4btqejohjbj6h75hlvn1w/oracles.zip?rlkey=7k4p3rf8ynv2a8xwa4q70irrj&dl=0 -O oracles.zip
unzip data.zip
unzip oracles.zip
```

## Train sequence-property predictors
```
python scripts/train_predictor.py configs/gfp_ddg/train_predictor_GFP4ddg.yml --logdir logs_predictors
```
On each dataset, one predictor is required for each property of interest.

## Run MosPro sampling
```
python scripts/run_MosPro.py configs/gfp_ddg/MosPro_GFP_stability.yml --logdir logs_MosPro
```
The config files can be found in the `configs` directory and they are organized in subdirectories according to different property objectives. Make sure you change the predictor path in the sampling configuration file to your own.

Each experiment will create a corresponding log directory under the specified directory. All the checkpoints and sample results will be stored in the log directory.

## Evaluate sample results
To evaluate the sample results of MosPro on a certain benchmarking dataset, use the corresponding evaluation script. For example, to evaluate the samples of GFP-stability dataset, run the following:
```
python scripts/evaluate_GFP_stability.py configs/gfp_ddg/evaluate.yml --sample_path path/to/sample_csv
```
A evaluation result file and metric file for 500 randomly selected samples will be generated in the same diretory of `path/to/sample_csv`. 

Note that for evaluating stability you need to download [FoldX](https://foldxsuite.crg.eu/), get a license for activation., and copy the FoldX executable file to the main directory.