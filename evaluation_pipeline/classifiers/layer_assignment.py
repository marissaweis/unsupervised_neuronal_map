import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from itertools import product
from scipy.interpolate import LinearNDInterpolator
from sklearn.linear_model import LogisticRegression

sys.path.append('../')
from preprocessing.volume_preprocessing.preprocessing_utils import rotate_points


L23 = 'layer_prediction=="L23"'
L4 = 'layer_prediction=="L4"'
L5 = 'layer_prediction=="L5"'
L6 = 'layer_prediction=="L6"'

groups = [[L23, L4], [L4, L5], [L5, L6]]

group_names = ['L23/L4', 'L4/L5', 'L5/L6']

COLUMNS = [
    'segment_id',
    'split_index',
    'soma_x',
    'soma_y',
    'soma_z',
    'soma_x_align',
    'soma_y_align',
    'soma_z_align',
]


parser = argparse.ArgumentParser()
parser.add_argument('output_path')


def compute_layer_thickness(
    boundary1=None, boundary2=None, v1_boundary=1208.162, pia=0.0, wm=693.023
):
    if (boundary1 is not None) and (boundary2 is not None):
        v1 = (
            boundary2[:, 1][boundary2[:, 0] < v1_boundary]
            - boundary1[:, 1][boundary1[:, 0] < v1_boundary]
        )
        hva = (
            boundary2[:, 1][boundary2[:, 0] >= v1_boundary]
            - boundary1[:, 1][boundary1[:, 0] >= v1_boundary]
        )
    elif boundary1 is None:
        v1 = boundary2[:, 1][boundary2[:, 0] < v1_boundary] - pia
        hva = boundary2[:, 1][boundary2[:, 0] >= v1_boundary] - pia
    elif boundary2 is None:
        v1 = wm - boundary1[:, 1][boundary1[:, 0] < v1_boundary]
        hva = wm - boundary1[:, 1][boundary1[:, 0] >= v1_boundary]

    item = {
        'V1_mean': np.nanmean(v1),
        'V1_std': np.nanstd(v1),
        'HVA_mean': np.nanmean(hva),
        'HVA_std': np.nanstd(hva),
    }

    return item


def fit_best_threshold(data1, data2):
    X = np.concatenate([data1, data2])[:, None]
    y = np.concatenate([np.zeros_like(data1), np.ones_like(data2)])

    logreg = LogisticRegression(class_weight='balanced')
    logreg.fit(X, y)

    return -logreg.intercept_[0] / logreg.coef_[0]


def run_assignment(df_pred):

    # Initialize grid.
    # Extent of volume in x- & z- direction.
    xbounds = df_pred.soma_x.min().astype(int) - 1, df_pred.soma_x.max().astype(int) + 1
    zbounds = df_pred.soma_z.min().astype(int) - 1, df_pred.soma_z.max().astype(int) + 1
    xlen = np.diff(xbounds)[0]
    zlen = np.diff(zbounds)[0]

    # Grid for piecewise-linear decision boundary.
    nxbins = 6
    nzbins = 4

    # Corner points of each grid cell.
    grid = np.stack(
        np.meshgrid(
            np.linspace(*xbounds, num=nxbins),
            np.linspace(*zbounds, num=nzbins),
            indexing='ij',
        ),
        -1,
    )
    bin_inds = list(product(np.arange(grid.shape[0] - 1), np.arange(grid.shape[1] - 1)))
    init = np.zeros(np.array(grid.shape[:2]) - 1)

    # Determine best threshold per grid cell and boundary.
    layer_boundary_mgrids = {}
    layer_boundary_interp = {}
    for (grp1, grp2), group in zip(groups, group_names):
        subset_df = df_pred.query(f'{grp1} or {grp2}')
        center_x = init.copy()
        center_z = init.copy()
        boundaries = init.copy()

        for x, z in bin_inds:
            # Retrieve all neurons within a grid cell.
            mins = grid[x, z]
            maxs = grid[x + 1, z + 1]
            center_x[x, z], center_z[x, z] = mins + (maxs - mins) // 2
            bin_df = subset_df.query(
                f'soma_x >= {mins[0]} and soma_x <= {maxs[0]} and soma_z >= {mins[1]} and soma_z <= {maxs[1]}'
            )

            if len(bin_df) > 0:
                # Soma depth of neurons per predicted layer.
                data1 = bin_df.query(grp1).soma_y.values
                data2 = bin_df.query(grp2).soma_y.values
                boundaries[x, z] = fit_best_threshold(data1, data2)

        # At 4 points at corners of the volume to prevent NaN values.
        boundary_points = np.stack(
            [
                [grid[0, 0, 0], boundaries[0, 0], grid[0, 0, 1]],
                [grid[0, -1, 0], boundaries[0, -1], grid[0, -1, 1]],
                [grid[-1, 0, 0], boundaries[-1, 0], grid[-1, 0, 1]],
                [grid[-1, -1, 0], boundaries[-1, -1], grid[-1, 3, 1]],
            ]
        )

        center_points = np.stack([center_x, boundaries, center_z], -1).reshape(-1, 3)
        points = np.concatenate([center_points, boundary_points], axis=0)

        grid_x, grid_z = np.meshgrid(
            np.linspace(*xbounds, 100), np.linspace(*zbounds, 50), indexing='ij'
        )

        interp = LinearNDInterpolator(points[:, [0, 2]], points[:, 1])
        grid_y = interp(grid_x, grid_z)

        assert ~np.any(np.isnan(grid_y))

        layer_boundary_mgrids[group] = np.stack([grid_x, grid_y, grid_z], axis=-1)
        layer_boundary_interp[group] = interp

    # Assign cortical layer per neuron.
    print(f'Assign layer membership based on computed layer boundaries.')
    for i, row in tqdm(df_pred.iterrows()):
        layer_values = [
            np.sign(row.soma_y - layer_boundary_interp[group](row.soma_x, row.soma_z))
            for group in group_names
        ]

        assert ~np.any(np.isnan(layer_values))

        if layer_values == [-1, -1, -1]:
            df_pred.loc[i, 'assigned_layer'] = 'L23'
        elif layer_values == [1, -1, -1]:
            df_pred.loc[i, 'assigned_layer'] = 'L4'
        elif layer_values == [1, 1, -1]:
            df_pred.loc[i, 'assigned_layer'] = 'L5'
        elif layer_values == [1, 1, 1]:
            df_pred.loc[i, 'assigned_layer'] = 'L6'

    # Create boundary table.
    boundaries = pd.DataFrame()
    for group in group_names:
        data = layer_boundary_mgrids[group].reshape(-1, 3)
        data_align = rotate_points(data, axis=1, degree=14)
        df_group = pd.DataFrame(
            data={
                'x': data[:, 0],
                'y': data[:, 1],
                'z': data[:, 2],
                'x_align': data_align[:, 0],
                'y_align': data_align[:, 1],
                'z_align': data_align[:, 2],
                'boundary': group,
            }
        )
        boundaries = pd.concat([boundaries, df_group])

    # Compute average layer thickness.
    l23_l4_boundary_align = boundaries[boundaries.boundary == 'L23/L4'][
        ['x_align', 'y_align', 'z_align']
    ].values
    l4_l5_boundary_align = boundaries[boundaries.boundary == 'L4/L5'][
        ['x_align', 'y_align', 'z_align']
    ].values
    l5_l6_boundary_align = boundaries[boundaries.boundary == 'L5/L6'][
        ['x_align', 'y_align', 'z_align']
    ].values

    l23_thickness = compute_layer_thickness(boundary2=l23_l4_boundary_align)
    l4_thickness = compute_layer_thickness(
        boundary1=l23_l4_boundary_align, boundary2=l4_l5_boundary_align
    )
    l5_thickness = compute_layer_thickness(
        boundary1=l4_l5_boundary_align, boundary2=l5_l6_boundary_align
    )
    l6_thickness = compute_layer_thickness(boundary1=l5_l6_boundary_align)

    data = [l23_thickness, l4_thickness, l5_thickness, l6_thickness]
    layer_thickness = pd.DataFrame(
        data, ['L23', 'L4', 'L5', 'L6'], ['V1_mean', 'V1_std', 'HVA_mean', 'HVA_std']
    )

    # Compute average layer depth.
    avg_layer_depth = {
        'Pia': 0.0,
        'L23/L4': l23_l4_boundary_align[:, 1].mean(),
        'L4/L5': l4_l5_boundary_align[:, 1].mean(),
        'L5/L6': l5_l6_boundary_align[:, 1].mean(),
        'WM': 693.023,
    }

    return df_pred, boundaries, layer_thickness, avg_layer_depth
