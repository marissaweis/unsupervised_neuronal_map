import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import sys
from sklearn.linear_model import LinearRegression
from cmcrameri import cm
import pickle
import matplotlib as mpl


def rotate_embedding(features, angle):
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))

    features_rot = np.dot(features, R.T)
    return features_rot


def compute_angle_from_coefs(coefs):
    return 90 - np.rad2deg(np.arctan2(-coefs[1], -coefs[0]))


def compute_percentiles(name, df, t=10):
    '''Compute 10th percentiles and bin data for visualization.'''
    data = df[name].values.astype(float)

    t1 = 100 // t

    perc = []
    for i in range(1, t + 1):
        perc.append(np.nanpercentile(data, t1 * i))

    # assign data to percentile
    data_perc = np.argmin(perc < data[:, None], axis=1)

    perc2 = []
    for i, p in enumerate(perc):
        if p == 0 and i == 0:
            pass
        elif p == 0:
            continue
        perc2.append(p)

    return data_perc, perc2


def rotate_soma_depth(df, coefs):
    features = df[['tsne_latent_emb_x', 'tsne_latent_emb_y']].values

    soma_angle = compute_angle_from_coefs(coefs)
    features_rot = rotate_embedding(features, soma_angle)

    df['rot_soma_tsne_latent_emb_x'] = features_rot[:, 0]
    df['rot_soma_tsne_latent_emb_y'] = features_rot[:, 1]

    return df, soma_angle


def plot_tsne_rot_soma(df, name, percentiles, color, savepath, label_name):

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    sns.scatterplot(
        data=df,
        x='rot_soma_tsne_latent_emb_x',
        y='rot_soma_tsne_latent_emb_y',
        hue=f'{name}_perc',
        alpha=0.5,
        linewidth=0,
        palette=sns.light_palette(color, as_cmap=True),
        legend=False,
        s=20,
        ax=ax,
        rasterized=True,
    )

    x_lims = (
        df['rot_soma_tsne_latent_emb_x'].values.min(),
        df['rot_soma_tsne_latent_emb_x'].values.max(),
    )
    y_lims = (
        df['rot_soma_tsne_latent_emb_y'].values.min(),
        df['rot_soma_tsne_latent_emb_y'].values.max(),
    )

    ax.set_xlim([x_lims[0] - 10, x_lims[1] + 10])
    ax.set_ylim([y_lims[0] - 10, y_lims[1] + 10])
    ax.set_aspect('equal')

    ax.axis('off')

    savepath = Path(savepath, 'tsnes')
    savepath.mkdir(parents=True, exist_ok=True)
    savepath = Path(savepath, f'tsne_{name}.pdf')
    plt.savefig(savepath, bbox_inches='tight', dpi=300)
    plt.close()


def plot_colorbar(color, df, name, savepath):

    label0, label1 = int(np.rint(df[name].min())), int(np.rint(df[name].max()))

    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)

    cmap = sns.light_palette(color, as_cmap=True)

    cb1 = mpl.colorbar.ColorbarBase(
        ax, cmap=cmap, orientation='horizontal', drawedges=False
    )

    cb1.set_ticks([0, 1])
    cb1.set_ticklabels([label0, label1])
    cb1.outline.set_visible(False)
    cb1.ax.tick_params(labelsize=14)

    savepath = Path(savepath, 'tsnes', f'tsne_cbar_{name}.pdf')
    plt.savefig(savepath, bbox_inches='tight')
    plt.close()


def plot_correlations(df, names, color, savepath, label_names):
    '''Plot scatter plots of two morphometrics versus each other.'''
    plt.rcParams.update({'font.size': 16})
    plt.rcParams['axes.linewidth'] = 2

    for i, name1 in enumerate(names):
        for j, name2 in enumerate(names[i + 1 :]):
            x = df[name1].values
            y = df[name2].values

            fig, ax = plt.subplots(1, 1, figsize=(5, 3))
            plt.scatter(x, y, color=color, alpha=0.5, s=5, rasterized=True)
            ax.set_xlabel(label_names[i])
            ax.set_ylabel(label_names[i + 1 + j])
            ax.xaxis.set_tick_params(width=2)
            ax.yaxis.set_tick_params(width=2)
            sns.despine(trim=10)

            plt.savefig(
                Path(savepath, f'{name1}_{name2}.pdf'), bbox_inches='tight', dpi=300
            )
            plt.close()


def plot_corr_summary(df_corr, label_names, savepath=None):
    '''Plot heatmap of correlations between all morphometrics.'''
    plt.rcParams.update({'font.size': 16})
    plt.rcParams['axes.linewidth'] = 2

    df_corr_t = df_corr
    label_names_x = label_names
    label_names_y = label_names

    n = len(df_corr_t)
    mask = np.zeros((n, n))
    mask[np.triu_indices_from(mask, k=1)] = True

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    sns.heatmap(
        data=df_corr_t,
        vmin=-1,
        vmax=1,
        cmap=cm.vik,
        mask=mask,
        annot=True,
        ax=ax,
        cbar=False,
    )

    ax.set_aspect('equal')
    ax.set_xticklabels(label_names_x)
    ax.set_yticklabels([''] * len(label_names_y))
    ax.xaxis.set_tick_params(width=2)
    ax.yaxis.set_tick_params(width=2)

    plt.savefig(savepath, bbox_inches='tight', transparent=True)
    plt.close()


def plot_scores(scores, color, names, label_names, wo_ylabels=False, savepath=None):  #
    plt.rcParams.update({'font.size': 16})
    plt.rcParams['axes.linewidth'] = 2
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    barlist = plt.barh(np.arange(len(scores)), scores, color=color)

    plt.tick_params(axis='y', which='both', left=False, right=False)
    ax.set_yticklabels([])
    ax.set_xlim([0.0, 1.0])
    ax.xaxis.set_tick_params(width=2)

    sns.despine(trim=1, top=True, right=True, left=True, bottom=False)
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()

    plt.savefig(savepath, bbox_inches='tight', transparent=True)
    plt.close()


def fit_lin_reg_single(features, label):
    reg = LinearRegression().fit(features, label)
    score = reg.score(features, label)
    reg_weight = reg.coef_
    return score, reg_weight


def fit_lin_reg(df, names, out_path):
    scores_32d = np.zeros(len(names))
    scores = np.zeros(len(names))
    reg_weights = {}
    for i, name in enumerate(names):
        label = df[f'{name}_perc'].values

        # 32D
        features = np.stack(df['latent_emb'].values)
        score, reg_weight = fit_lin_reg_single(features, label)

        scores_32d[i] = score
        reg_weights[f'{name}_32d'] = reg_weight

        # 2D
        features = df[['tsne_latent_emb_x', 'tsne_latent_emb_y']].values
        score2, reg_weight2 = fit_lin_reg_single(features, label)

        scores[i] = score2
        reg_weights[name] = reg_weight2

    df_reg = pd.DataFrame()
    df_reg['Morphometric'] = list(np.array(names)[np.argsort(scores_32d)[::-1]])
    df_reg['Score'] = scores[np.argsort(scores_32d)[::-1]]
    df_reg['Score_32d'] = scores_32d[np.argsort(scores_32d)[::-1]]
    df_reg.to_pickle(Path(out_path, 'regression_scores.pkl'))

    return df_reg, reg_weights, scores, scores_32d
