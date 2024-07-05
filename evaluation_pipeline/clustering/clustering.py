import numpy as np
from sklearn.mixture import GaussianMixture
import openTSNE

SEED = 7236


def run_gmm(
    train_latents,
    n_clusters=3,
    covariance_type='diag',
    test_latents=None,
    return_model=False,
):
    if test_latents == None:
        test_latents = train_latents

    gmm = GaussianMixture(
        n_components=n_clusters, random_state=SEED, covariance_type=covariance_type
    ).fit(train_latents)
    prediction = gmm.predict(test_latents)

    if return_model:
        return prediction, gmm
    return prediction


def sort_by_avg_depth(predictions, soma_depth):
    '''Sort cluster order by average soma depth of neurons in clusters.'''
    unique_preds = np.unique(predictions)
    avg_cluster_depth = np.zeros(len(unique_preds))
    for i, cluster_idx in enumerate(unique_preds):
        avg_cluster_depth[i] = soma_depth[predictions == cluster_idx].mean()

    sorted_cluster_ids = list(np.argsort(avg_cluster_depth))

    new_predictions = np.zeros_like(predictions, dtype=int)
    for i, t in enumerate(predictions):
        new_predictions[i] = sorted_cluster_ids.index(list(unique_preds).index(t))

    return new_predictions


def run_open_tsne(latents, verbose=True, perplexity=30):
    '''Run TSNE with 2 output features and perplexity=30 on latent vectors.
    Args:
        latents: (N x D)
    '''
    tsne = openTSNE.TSNE(
        perplexity=perplexity,
        metric='cosine',
        n_jobs=8,
        random_state=402,
        verbose=verbose,
    )
    z = tsne.fit(latents)
    return z
