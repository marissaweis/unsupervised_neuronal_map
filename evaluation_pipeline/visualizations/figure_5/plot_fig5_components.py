import numpy as np
import pandas as pd
import argparse
import pickle
from cmcrameri import cm
from scipy.stats import spearmanr
from pathlib import Path

from figure_utils import *

color_dict = dict(
    {
        'L23': cm.batlow.colors[0],
        'L4': cm.batlow.colors[50],
        'L5': cm.batlow.colors[100],
        'L6': cm.batlow.colors[150],
    }
)


DATA = '../../../data/data_tables/df_morphos.pkl'
OUT = 'figures/'

names = [
    'soma_y',
    'cell_length',
    'apical_total_skeletal_length',
    'apical_width',
    'basal_skeletal_length',
    'depth_vs_basal_mean',
]

label_names = [
    'Depth',
    'Height',
    'Apical length',
    'Apical width',
    'Basal length',
    'Basal bias',
]

parser = argparse.ArgumentParser()
parser.add_argument('path')
parser.add_argument(
    '--layer', type=str, default=None, choices=['L23', 'L4', 'L5', 'L6']
)


def main(args):
    table_path = args.path
    layer = args.layer

    color = color_dict[layer]

    out_path = Path(OUT, layer)
    out_path.mkdir(parents=True, exist_ok=True)

    df1 = pd.read_pickle(table_path)
    df1 = df1[~df1.exclude]
    df1 = df1[df1.assigned_layer == layer]

    df2 = pd.read_pickle(DATA)

    df = pd.merge(
        df1[
            [
                'segment_id',
                'split_index',
                f'tsne_latent_emb_x_{layer}',
                f'tsne_latent_emb_y_{layer}',
                f'cluster_{layer}',
                'latent_emb',
            ]
        ],
        df2,
        on=['segment_id', 'split_index'],
    )
    df = df[
        [
            'segment_id',
            'split_index',
            f'tsne_latent_emb_x_{layer}',
            f'tsne_latent_emb_y_{layer}',
            'latent_emb',
            f'cluster_{layer}',
        ]
        + names
    ]

    df = df.rename(
        columns={
            f'tsne_latent_emb_x_{layer}': 'tsne_latent_emb_x',
            f'tsne_latent_emb_y_{layer}': 'tsne_latent_emb_y',
        }
    )

    if layer == 'L6':
        df['apical_width'] = df['apical_width'].replace(np.nan, value=0.0)

    # Remove rows with NaNs in morphometrics
    print(f'{len(df)} neurons before.')
    df = df.dropna(subset=names)
    df = df.reset_index()
    print(f'{len(df)} neurons after drop of NaNs.')

    # Compute 10th percentiles of morphometrics for plotting.
    percentiles = {}
    for name in names:
        targets, percentile = compute_percentiles(name, df)
        df[f'{name}_perc'] = targets
        percentiles[name] = percentile

    # Save raw data.
    df.to_pickle(Path(out_path, f'{layer}.pkl'))

    # Linear regression + quantify gradients
    df_reg, reg_weights, scores, scores_32d = fit_lin_reg(df, names, out_path)

    # Compute correlations between morphometrics.
    corr = np.zeros((len(names), len(names)))
    for i, data1 in enumerate(names):
        for j, data2 in enumerate(names):
            corr[i, j], _ = spearmanr(df[data1], df[data2])
    data = corr.round(decimals=2)
    df_corr = pd.DataFrame(data=data, columns=names, index=names)
    df_corr.to_pickle(Path(out_path, 'spearm_correlations.pkl'))

    # Plot r2 scores.
    plot_scores(
        scores_32d,
        color,
        names,
        label_names,
        savepath=Path(out_path, f'scores_32d.pdf'),
    )

    # Plot correlation bar plot.
    plot_corr_summary(
        df_corr, label_names, savepath=Path(out_path, f'correlations.pdf')
    )

    # Plot individual scatter correlation plots.
    savepath = Path(out_path, 'corr_plots')
    savepath.mkdir(parents=True, exist_ok=True)
    plot_correlations(df, names, color, savepath, label_names)

    # Rotate according to soma gradient.
    df, soma_angle = rotate_soma_depth(df, reg_weights['soma_y'])

    # Plot t-SNE embeddings colored by morphometrics.
    for name, label_name in zip(names, label_names):
        label_name = label_names[i]
        plot_tsne_rot_soma(df, name, percentiles[name], color, out_path, label_name)
        plot_colorbar(color, df, name, out_path)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
