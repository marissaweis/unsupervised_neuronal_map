# An unsupervised map of excitatory neurons’ dendritic morphology in the mouse visual cortex

This repository contains code for the paper [An unsupervised map of excitatory neurons’ dendritic morphology in the mouse visual cortex](https://www.biorxiv.org/content/10.1101/2022.12.22.521541v3).

![Figure 2](Fig2.png?raw=true "Figure 2")

## Preprocessed data and meta data
All meta data and preprocessed data are located in the [data]((https://github.com/marissaweis/unsupervised_neuronal_map/blob/data/) folder. This includes the learned morphological embeddings of the MICrONS neurons as well as morphometrics computed on them.

**Coming soon**


## Downloading the raw data
The raw data can be found [here](https://www.microns-explorer.org/cortical-mm3).

Graphs of the neurons were extracted using [NEURD](https://github.com/reimerlab/NEURD).


## Training **GraphDINO**
[Code](https://eckerlab.org/code/weis2023/) and [publication](https://openreview.net/forum?id=ThhMzfrd6r) of how to train **GraphDINO** can be found here.

The checkpoint for the pre-trained GraphDINO on the [MICrONS data](https://www.microns-explorer.org/) as well as the configuration file can be found here: 
- [ckpt_microns.pt](https://github.com/marissaweis/unsupervised_neuronal_map/blob/data/graphdino/ckpts/)
- [config.json](https://github.com/marissaweis/unsupervised_neuronal_map/blob/data/graphdino/ckpts/config.json)

The learned morphological embeddings of the MICrONS data can be found here:
- [graphdino_morphological_embeddings.pkl](https://github.com/marissaweis/unsupervised_neuronal_map/blob/data/graphdino/embeddings/) - **Coming soon**


## Analyses

Necessary inputs to the evaluation pipeline are explained [here](https://github.com/marissaweis/unsupervised_neuronal_map/blob/data/).

Reproduction of the analysis and figures of the paper:
1. [Preprocessing](https://github.com/marissaweis/unsupervised_neuronal_map/blob/evaluation_pipeline/preprocessing/): Notebooks to perform volume rotation and normalization as well as some quality control of the used neurons.
2. [Classifiers](https://github.com/marissaweis/unsupervised_neuronal_map/blob/evaluation_pipeline/classifiers/): Notebooks and scripts to train supervised classifiers on the labeled subset of the MICrONS data and apply to the whole volume.
3. [Clustering](https://github.com/marissaweis/unsupervised_neuronal_map/blob/evaluation_pipeline/clustering/): Scripts to fit Gaussian mixture models to data.
4. [Cluster versus continuum analysis](https://github.com/marissaweis/cluster_vs_continuum): Scripts and notebooks to generate synthetic data and run our cluster versus contiuum analysis can be found in separate repository.
4. [Quality control](https://github.com/marissaweis/unsupervised_neuronal_map/blob/evaluation_pipeline/quality_control/): Notebooks to excluded fragmented neurons before further analysis.
5. [Visualizations](https://github.com/marissaweis/unsupervised_neuronal_map/blob/evaluation_pipeline/visualizations/): Notebooks to plot figures of the paper.
6. [Basal bias prediction](https://github.com/marissaweis/unsupervised_neuronal_map/blob/evaluation_pipeline/basal_bias_prediction/): Prediction of basal bias from functional embeddings of neurons. Functional embeddings are taken from [Towards a Foundation Model of the Mouse Visual Cortex](https://www.biorxiv.org/content/10.1101/2023.03.21.533548v2).


## Differences to paper
We have simplified the computation of the layer boundary which leads to slightly different layer boundaris. Exclusion of layer 4 and layer 6 neurons was done using earlier version of the layer boundaries. <!--These can be found [here]().-->


## Citation
```
@article{Weis2024,
      title={An unsupervised map of excitatory neurons' dendritic morphology in the mouse visual cortex},
      author = {Weis, Marissa A. and Papadopoulos, Stelios and Hansel, Laura and Lüddecke, Timo and Celii, Brendan and Fahey, Paul G. and Wang, Eric Y. and MICrONS Consortium and Reimer, Jacob and Berens, Philipp and Tolias, Andreas S. and Ecker, Alexander S.}
      journal={bioRxiv},
      doi = {10.1101/2022.12.22.521541},
      year={2024}
}
```
