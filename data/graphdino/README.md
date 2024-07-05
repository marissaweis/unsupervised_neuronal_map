# GraphDINO

## Training
For training GraphDINO, please refer to the [GraphDINO repo](https://github.com/marissaweis/ssl_neuron) and the methods [paper](https://openreview.net/pdf?id=ThhMzfrd6r).
The training configuration as used in the paper is given in the [config file](https://github.com/marissaweis/unsupervised_neuronal_map/blob/main/data/graphdino/ckpts/config.json).

## Inference using pre-trained GraphDINO

The weights of the GraphDINO trained on MICrons are given in the [checkpoint file](https://github.com/marissaweis/unsupervised_neuronal_map/blob/main/data/graphdino/ckpts/ckpt_microns.pt). See the notebook []() for an example, how to perform inference using the pre-trained model.


## Embeddings
[Pickled pandas dataframe](https://github.com/marissaweis/unsupervised_neuronal_map/blob/main/data/graphdino/ckpts/embeddings/xx.pkl) with learned morphological embeddings per [MICrONS](https://www.microns-explorer.org/) neuron. Neurons are identified by segment id and split index and embeddings are 32-dimensional. GraphDINO was trained using the v7 skeletons by [Brendan et al.](https://www.biorxiv.org/content/10.1101/2023.03.14.532674v3) using only xyz-coordinates as node input features and with the soma node being centered to (0, 0, 0) and after axon removal.
