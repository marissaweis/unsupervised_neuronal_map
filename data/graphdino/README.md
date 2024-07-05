# GraphDINO

## Training
For training GraphDINO, please refer to the [GraphDINO repo](https://github.com/marissaweis/ssl_neuron) and the methods [paper](https://openreview.net/pdf?id=ThhMzfrd6r).
The training configuration as used in the paper is given in the [config file](https://github.com/marissaweis/unsupervised_neuronal_map/blob/main/data/graphdino/ckpts/config.json).

## Inference using pre-trained GraphDINO

The weights of the GraphDINO trained on MICrons are given in the [checkpoint file](https://github.com/marissaweis/unsupervised_neuronal_map/blob/main/data/graphdino/ckpts/ckpt_microns.pt). See the notebook []() for an example, how to perform inference using the pre-trained model.


## Embeddings
[Pickled pandas dataframe](https://github.com/marissaweis/unsupervised_neuronal_map/blob/main/data/graphdino/ckpts/embeddings/xx.pkl) containing learned morphological embeddings for each [MICrONS](https://www.microns-explorer.org/) neuron. Neurons are identified by segment ID and split index, with embeddings being 32-dimensional. GraphDINO was trained using v7 skeletons by [Brendan et al.](https://www.biorxiv.org/content/10.1101/2023.03.14.532674v3) , utilizing only *xyz*-coordinates as node input features. The soma node was centered at (0, 0, 0) and the axon was removed prior to training.
