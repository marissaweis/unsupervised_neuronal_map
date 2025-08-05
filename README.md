# An unsupervised map of excitatory neurons’ dendritic morphology in the mouse visual cortex

This repository contains code for the paper [An unsupervised map of excitatory neuron dendritic morphology in the mouse visual cortex](https://www.nature.com/articles/s41467-025-58763-w).

![Figure 2](Fig2.png?raw=true "Figure 2")

## Preprocessed data and meta data
Data is published under "Source data" with the [paper](https://www.nature.com/articles/s41467-025-58763-w#Sec40). This includes the learned morphological embeddings of the MICrONS neurons as well as morphometrics computed on them.


## Downloading the raw data
The raw data can be found in the [MICrONS Explorer](https://www.microns-explorer.org/cortical-mm3).

Graphs of the neurons were extracted using [NEURD](https://github.com/reimerlab/NEURD).


## Training *GraphDINO*
[Code](https://eckerlab.org/code/weis2023/) and [publication](https://openreview.net/pdf?id=ThhMzfrd6r) on how to train *GraphDINO* can be found here.

The checkpoint for the pre-trained GraphDINO on the [MICrONS data](https://www.microns-explorer.org/) as well as the configuration file can be found here: 
- [ckpt_microns.pt](https://github.com/marissaweis/unsupervised_neuronal_map/blob/main/data/graphdino/ckpts/ckpt_microns.pt)
- [config.json](https://github.com/marissaweis/unsupervised_neuronal_map/blob/main/data/graphdino/ckpts/config.json)

The learned morphological embeddings of the MICrONS data can be found under "Source data" with the [paper](https://www.nature.com/articles/s41467-025-58763-w#Sec40).


## Analyses

<!--Necessary inputs to the evaluation pipeline are explained [here](https://github.com/marissaweis/unsupervised_neuronal_map/blob/main/data/).-->

Reproduction of the analyses and figures of the paper:
1. [Preprocessing](https://github.com/marissaweis/unsupervised_neuronal_map/blob/main/evaluation_pipeline/preprocessing/): Notebooks to perform volume rotation and normalization as well as some quality control of the used neurons.
2. [Classifiers](https://github.com/marissaweis/unsupervised_neuronal_map/blob/main/evaluation_pipeline/classifiers/): Notebooks and scripts to train supervised classifiers on the labeled subset of the MICrONS data and to apply them to the whole volume.
3. [Clustering](https://github.com/marissaweis/unsupervised_neuronal_map/blob/main/evaluation_pipeline/clustering/): Scripts to fit Gaussian mixture models to data.
4. [Cluster versus continuum analysis](https://github.com/marissaweis/cluster_vs_continuum): Scripts and notebooks to generate synthetic data and run our cluster versus contiuum analysis can be found in separate repository.
4. [Quality control](https://github.com/marissaweis/unsupervised_neuronal_map/blob/main/evaluation_pipeline/quality_control/): Notebooks to exclude fragmented neurons before further analysis.
5. [Visualizations](https://github.com/marissaweis/unsupervised_neuronal_map/blob/main/evaluation_pipeline/visualizations/): Notebooks to plot figures of the paper.
6. [Basal bias prediction](https://github.com/marissaweis/unsupervised_neuronal_map/blob/main/evaluation_pipeline/basal_bias_prediction/): Prediction of basal bias from functional embeddings of L4 neurons. Functional embeddings are taken from [Towards a Foundation Model of the Mouse Visual Cortex](https://www.biorxiv.org/content/10.1101/2023.03.21.533548v2).


## Differences to paper
We have simplified the computation of the layer boundaries which leads to slightly different layer boundaris than those used in the paper. Exclusion of layer 4 and layer 6 neurons was done using earlier version of the layer boundaries. <!--These can be found [here]().-->


## Citation
```
@article{Weis2025,
	author = {Weis, Marissa A. and Papadopoulos, Stelios and Hansel, Laura and L{\"u}ddecke, Timo and Celii, Brendan and Fahey, Paul G. and Wang, Eric Y. and Bae, J. Alexander and Bodor, Agnes L. and Brittain, Derrick and Buchanan, JoAnn and Bumbarger, Daniel J. and Castro, Manuel A. and Collman, Forrest and da Costa, Nuno Ma{\c c}arico and Dorkenwald, Sven and Elabbady, Leila and Halageri, Akhilesh and Jia, Zhen and Jordan, Chris and Kapner, Dan and Kemnitz, Nico and Kinn, Sam and Lee, Kisuk and Li, Kai and Lu, Ran and Macrina, Thomas and Mahalingam, Gayathri and Mitchell, Eric and Mondal, Shanka Subhra and Mu, Shang and Nehoran, Barak and Popovych, Sergiy and Reid, R. Clay and Schneider-Mizell, Casey M. and Seung, H. Sebastian and Silversmith, William and Takeno, Marc and Torres, Russel and Turner, Nicholas L. and Wong, William and Wu, Jingpeng and Yin, Wenjing and Yu, Szi-chieh and Reimer, Jacob and Berens, Philipp and Tolias, Andreas S. and Ecker, Alexander S.},
	journal = {Nature Communications},
	number = {1},
	pages = {3361},
	title = {An unsupervised map of excitatory neuron dendritic morphology in the mouse visual cortex},
	volume = {16},
	year = {2025}
}
```
