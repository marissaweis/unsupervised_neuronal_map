import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from cmcrameri import cm


color_dict = {
    'L23': cm.batlow.colors[0],
    'L4': cm.batlow.colors[50],
    'L5': cm.batlow.colors[100],
    'L6': cm.batlow.colors[150],
}

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


def plot_tsne(df, name, x, y, savepath=None, color_dict=None):
    """Plot t-SNE embedding colored by neuronal property.

    Args:
        df: pd.DataFrame containing t-SNE embedding and property.
        name: column name of property
        savepath: file path to save figure to
        color_dict: dictionary {property: color value}
    """

    if color_dict is None:
        palette = 'husl'
    else:
        palette = color_dict

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    sns.scatterplot(
        data=df,
        x=x,
        y=y,
        hue=name,
        hue_order=list(np.unique(df[name].values)),
        ax=ax,
        palette=palette,
        rasterized=True,
        alpha=0.5,
        linewidth=0,
        s=20,
        legend='full',
    )

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(name)
    plt.legend(bbox_to_anchor=(1, 1))

    if savepath is not None:
        fig.savefig(savepath, bbox_inches='tight', transparent=True)
        plt.close(fig)
    else:
        plt.show()


def compute_percentiles(name, df, t=10):
    '''Compute 10th percentiles and bin data for visualization.'''
    data = df[name].values.astype(float)

    t1 = 100 // t

    perc = []
    for i in range(1, t + 1):
        perc.append(np.nanpercentile(data, t1 * i))

    # Assign data points to percentile bins.
    data_perc = np.argmin(perc < data[:, None], axis=1)

    perc2 = []
    for i, p in enumerate(perc):
        if p == 0 and i == 0:
            pass
        elif p == 0:
            continue
        perc2.append(p)

    return data_perc, perc2


def plot_tsne_perc(df, name, x, y, palette='icefire', savepath=None):

    targets, percentiles = compute_percentiles(name, df)
    df[f'{name}_perc'] = targets

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    sns.scatterplot(
        data=df,
        x=x,
        y=y,
        hue=f'{name}_perc',
        alpha=0.5,
        linewidth=0,
        palette=palette,
        legend=False,
        rasterized=True,
        s=20,
        ax=ax,
    )
    ax.axis('off')
    ax.set_title(name)

    if savepath is not None:
        fig.savefig(savepath, bbox_inches='tight', transparent=True)
        plt.close(fig)
    else:
        plt.show()
