## running the experiments:

* TODO: clone this repository
* TODO: download the contents of `sim_metric_resources` (`wget` from somewhere...)
* change `resource_paths` in `paths.py` to the address of `sim_metric_resources` on your device
* TODO: dependencies - have a `requirements.txt` and/or a `requirements.yml`

### the easy way

for every experiment (eg `feather`, `pretrain_finetune`, `layer_exp`, or `pca_deletion`):
* the relevant distances and accuracies differences have already been precomputed and aggregated in a dataframe `full_df`
* all you have to do is run the notebook `notebook.ipynb` in the experiment's folder

### the hard way

you can recompute the distances between embeddings, use these and the accuracy scores to recompute the aggregate dataframe `full_df`


`dists` contains functions to compute distances between representations.
`experiments` contains code to run the experiments in part 4 of the paper.


## to add your metric
- add the corresponding function at the end of `scoring.py`
- add a condition in `score_pair.py`, around line 165 TODO: check that this is still valid
- for every experiment in `experiments`, add the name of the metric to the `metrics` list# sim_metric
