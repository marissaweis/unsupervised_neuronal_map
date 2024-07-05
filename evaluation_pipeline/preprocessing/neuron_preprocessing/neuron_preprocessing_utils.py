from sklearn.decomposition import PCA
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np


PIA_MEAN = 395.9638798743831


def rotate_neuron(features):
    """Rotate neuron around y-axis such that the maximal
    extent in x-direction is aligned with the x-axis.

    Arguments:
        features: np.array (N x 3)

    Output:
        features: np.array (N x 3)
    """
    xz_coord = features[:, [0, 2]]

    pca = PCA(n_components=2)
    rot_xz_coord = pca.fit_transform(xz_coord)

    features[:, [0, 2]] = rot_xz_coord

    return features


def plot_neuron(
    neighbors,
    node_feats,
    ax1=0,
    ax2=1,
    soma_id=0,
    color='lightsteelblue',
    plot_cort_layers=True,
    ax=None,
    bonds=None,
):
    '''Plot volume-normalized neuron with different colors per compartment label.'''

    colors = list(sns.dark_palette(color, n_colors=4))
    ax.set_aspect('equal')

    color_comp_map = {
        0: 3,  # dendrite
        1: 0,  # soma
        2: -1,  # axon
        3: 1,  # basal
        4: 2,  # apical, apical_tuft, apical_shaft, oblique
        5: -1,  # custom
        6: -1,  # undefined
        7: -1,  # glia
    }

    for i, neigh in neighbors.items():
        for j in neigh:
            n1, n2 = node_feats[i], node_feats[j]

            comp_type = color_comp_map[np.argmax(n2[4:9])]

            if comp_type == 0:
                comp_type = color_comp_map[np.argmax(n1[4:9])]

            if comp_type != -1:
                c = colors[comp_type]
                ax.plot(
                    [n1[ax1], n2[ax1]],
                    [n1[ax2], n2[ax2]],
                    color=c,
                    linewidth=1,
                    zorder=2,
                )

    ax.scatter(
        node_feats[soma_id][ax1],
        node_feats[soma_id][ax2],
        color=colors[0],
        s=10,
        zorder=3,
    )

    if plot_cort_layers and ax2 == 1:
        ax.set_ylim([-1200, -350])
        # Plot pia and white matter.
        min_x, max_x = max(node_feats[:, 0]), min(node_feats[:, 0])
        for k, l in enumerate([-PIA_MEAN, -1088.983]):
            ax.plot(
                [min_x, max_x],
                [l, l],
                linestyle='dotted',
                linewidth=1.5,
                color='black',
                zorder=1,
            )

        # Plot layer boundaries.
        if bonds is not None:
            for k, l in enumerate(bonds):
                l2 = -l - PIA_MEAN
                ax.plot(
                    [min_x, max_x],
                    [l2, l2],
                    linestyle='dotted',
                    linewidth=1.5,
                    color='black',
                    zorder=1,
                )

    sns.despine()
