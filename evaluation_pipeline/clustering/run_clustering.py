import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from clustering import *


COLUMNS = ['segment_id', 'split_index', 'latent_emb', 'soma_y', 'assigned_layer']

parser = argparse.ArgumentParser()
parser.add_argument('data_path')


def main(args):
    data_path = args.data_path

    print('### Run clustering ###')
    df_neuron_emb = pd.read_pickle(Path(data_path, 'df_classifier.pkl'))
    df_morphos = pd.read_pickle(Path(data_path, 'df_morphos.pkl'))
    df_neuron_emb = pd.merge(
        df_neuron_emb,
        df_morphos[['segment_id', 'split_index', 'soma_y', 'exclude']],
        on=['segment_id', 'split_index'],
    )
    df_neuron_emb = df_neuron_emb[~df_neuron_emb.exclude]

    # Run clustering for all excitatory neurons combined as well as layer-wise.
    for layer in ['all', 'L23', 'L4', 'L5', 'L6']:
        n_clusters = 60 if layer == 'all' else 15
        df = df_neuron_emb[COLUMNS].copy()

        if layer != 'all':
            df = df[df.assigned_layer == layer]
        else:
            df = df[~df.assigned_layer.isna()]
        latents = np.stack(df['latent_emb'].values)

        print('Run t-SNE.')
        perplexity = 300 if layer == 'all' else 30
        tsne_emb = run_open_tsne(latents, verbose=False, perplexity=perplexity)
        df[f'tsne_latent_emb_x_{layer}'] = tsne_emb[:, 0].tolist()
        df[f'tsne_latent_emb_y_{layer}'] = tsne_emb[:, 1].tolist()

        print(f'Run clustering for {layer} excitatory neurons (n={len(df)}).')
        predictions = run_gmm(
            latents,
            n_clusters=n_clusters,
            covariance_type='diag',
        )
        predictions = (
            sort_by_avg_depth(predictions, df['soma_y'].values.astype(float)) + 1
        )
        df[f'cluster_{layer}'] = predictions.astype(int)
        df[f'cluster_{layer}'] = df[f'cluster_{layer}'].astype('int')

        df_neuron_emb = pd.merge(
            df_neuron_emb,
            df[
                [
                    'segment_id',
                    'split_index',
                    f'cluster_{layer}',
                    f'tsne_latent_emb_x_{layer}',
                    f'tsne_latent_emb_y_{layer}',
                ]
            ],
            on=['segment_id', 'split_index'],
            how='left',
        )

    df_neuron_emb.to_pickle(Path(data_path, 'df_cluster.pkl'))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
