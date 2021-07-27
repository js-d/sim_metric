# Grounding Representation Similarity with Statistical Testing

## Description

### `sim_metric`

* `experiments/` contains code to run the experiments in part 4 of the paper:
    - `layer_exp` is the first experiment in part 4, with different random seeds and layer depths
    - `pca_deletion` is the second experiment in part 4, with different numbers of principal components deleted
    - `feather` is the first experiment in part 4.1, with different finetuning seeds
    - `pretrain_finetune` is the second experiment in part 4.2, with different pretraining and finetuning seeds
* `dists/` contains functions to compute dissimilarities between representations.

### `sim_metric_resources`


## Instructions

* clone this repository
* go to TODO: zenodo link and download `sim_metric_resources.tar`
* untar it with `tar -xvf sim_metric_resources sim_metric_resources.tar`
* in `sim_metric/paths.py`, modify the path to `sim_metric_resources`

### Replicating the results

For every experiment (eg `feather`, `pretrain_finetune`, `layer_exp`, or `pca_deletion`):
* the relevant dissimilarities and accuracies differences have already been precomputed and aggregated in a dataframe `full_df`
* to get the results, you can:
    - run the notebook `notebook.ipynb` in the experiment's folder, or 
    - run `script.py` in the experiment's folder, and find the results in `results.txt`, in the same folder

To run the scripts for all four experiments, run `experiments/script.py`.

### Recomputing dissimilarities

For every experiment, you can:
* recompute the dissimilarities between embeddings by running `compute_dists.py` in this experiment's folder
* use these and the accuracy scores to recompute the aggregate dataframe by running `compute_full_df.py` in this experiment's folder
* run the experiments with `script.py` or `notebook.ipynb` as above.

## To add your metric
* add the corresponding function at the end of `dists/scoring.py`
* add a condition in `dists/score_pair.py`, around line 160
* for every experiment in `experiments`, add the name of the metric to the `metrics` list in `compute_dists.py`