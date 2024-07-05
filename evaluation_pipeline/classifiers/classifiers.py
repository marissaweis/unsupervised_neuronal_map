import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

import matplotlib.pylab as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC

import warnings

warnings.filterwarnings('ignore')

ri_label_names = {0: 'neuron', 1: 'reconstruction_issue'}
ritype2int = {
    'exc': 0,
    'inh': 0,
    'ri': 1,
}

ei_label_names = {0: 'exc', 1: 'inh'}
eitype2int = {
    'exc': 0,
    'inh': 1,
}

layer_label_names = {0: 'L23', 1: 'L4', 2: 'L5', 3: 'L6'}
layer2int = {'L23': 0, 'L4': 1, 'L5': 2, 'L6': 3}

celltype_label_names = {
    0: '23P',
    1: '4P',
    2: '5P-IT',
    3: '5P-PT',
    4: '5P-NP',
    5: '6P-IT',
    6: '6P-CT',
}
celltype2int = {
    '23P': 0,
    '4P': 1,
    '5P-IT': 2,
    '5P-PT': 3,
    '5P-NP': 4,
    '6P-IT': 5,
    '6P-CT': 6,
}


def run_ri_classifier(df_classifier, df, path):
    '''Classifier: Whole versus incomplete neuron.'''
    # Prepare data.
    latents = np.stack(df_classifier['latent_emb'].values).astype(float)
    labels = np.stack([ritype2int[l] for l in df_classifier['cell_type_coarse'].values])

    df = run_classifier_pipeline(
        latents, labels, 'accuracy', df, ri_label_names, 'ri', path
    )

    return df


def run_ei_classifier(df_classifier, df, path):
    '''Classifier: Excitatory versus inhibitory neuron.'''
    df_classifier = df_classifier[df_classifier['cell_type_coarse'] != 'ri']

    # Prepare data.
    latents1 = np.stack(df_classifier['latent_emb'].values).astype(float)
    latents2 = np.stack(
        df_classifier[
            ['syn_density_shaft_after_proof', 'spine_density_after_proof']
        ].values
    ).astype(float)
    latents = np.concatenate([latents1, latents2], axis=1)

    labels = np.stack([eitype2int[l] for l in df_classifier['cell_type_coarse'].values])

    df = run_classifier_pipeline(
        latents, labels, 'balanced_accuracy', df, ei_label_names, 'ei', path
    )

    return df


def run_layer_classifier(df_classifier, df, path):
    '''Classifier: Cortical layers.'''
    df_classifier = df_classifier[(df_classifier['cell_type_coarse'] == 'exc')]

    # Prepare data.
    latents = np.stack(df_classifier['latent_emb'].values).astype(float)
    labels = np.stack([layer2int[l] for l in df_classifier['layer'].values])

    df = run_classifier_pipeline(
        latents, labels, 'balanced_accuracy', df, layer_label_names, 'layer', path
    )

    return df


def run_cell_type_classifier(df_classifier, df, path):
    '''Classifier: Excitatory cell type.'''
    df_classifier = df_classifier[(df_classifier['cell_type_coarse'] == 'exc')]

    # Prepare data.
    latents = np.stack(df_classifier['latent_emb'].values).astype(float)
    labels = np.stack([celltype2int[l] for l in df_classifier['cell_type'].values])

    df = run_classifier_pipeline(
        latents,
        labels,
        'balanced_accuracy',
        df,
        celltype_label_names,
        'cell_type',
        path,
    )

    return df


def run_classifier_pipeline(
    latents, labels, metric, df, label_names, classification_type, path
):
    out_path = Path(path, 'classifier')
    out_path.mkdir(parents=True, exist_ok=True)

    print(f'Run cross-validation for {classification_type}-classifier.')
    model_class, acc, params = run_crossvalidation(latents, labels, metric)

    print('Fit best model.')
    model = fit_best_parameters(
        model_class, params, latents, labels, label_names, classification_type, out_path
    )

    print('Run prediction for all neurons.')
    df = predict(model, df, label_names, classification_type)

    # Save model and results.
    file = Path(out_path, f'classifier_{classification_type}.pkl')
    with open(file, 'wb') as f:
        pickle.dump(model, f)

    print(f'Model class: {model_class}\n Accurcay: {acc:.2f}\n Parameters: {params}\n')

    file = Path(out_path, f'classifier_{classification_type}.txt')
    with open(file, 'w') as f:
        f.write(f'Model class: {model_class}\n')
        f.write(f'Accurcay: {acc:.2f}\n')
        f.write(f'Parameters: {params}\n')

    return df


def predict(model, df, label_names, classification_type):
    all_latents = np.stack(df['latent_emb'].values).astype(float)

    if classification_type == 'ei':
        latents2 = np.stack(
            df[['syn_density_shaft_after_proof', 'spine_density_after_proof']].values
        ).astype(float)
        all_latents = np.concatenate([all_latents, latents2], axis=1)

    all_preds = model.predict(all_latents)
    preds = np.stack([label_names[p] for p in all_preds])
    df[f'{classification_type}_prediction'] = preds
    print(f'Prediction stats: {np.unique(preds, return_counts=True)}.')
    return df


def fit_best_parameters(
    model_class, params, latents, labels, label_names, classification_type, out_path
):
    model = model_class(random_state=0, **params)
    model.fit(latents, labels)

    # Plot confusion matrix
    predictions = model.predict(latents)
    plot_confusion(
        predictions, labels, label_names, classification_type, out_path=out_path
    )
    plot_confusion(
        predictions,
        labels,
        label_names,
        classification_type,
        out_path=out_path,
        normalize=True,
    )
    return model


def run_crossvalidation(latents, labels, metric):
    assert metric in ['accuracy', 'balanced_accuracy']

    # Run cross-validation over logisitic regression.
    acc_lg, params_lg = crossvalidate_lg(latents, labels, metric)

    # Run cross-validation over support vector machine.
    acc_svm, params_svm = crossvalidate_svm(latents, labels, metric)

    # Select best model.
    idx = np.argmax([acc_lg, acc_svm])

    if idx == 0:
        return LogisticRegression, acc_lg, params_lg
    else:
        return SVC, acc_svm, params_svm


def crossvalidate_svm(latents, labels, metric):
    '''Perform grid search to find best hyperparameters given validation set.'''

    Cs = [0.5, 1, 3, 5, 10, 20, 30]
    degrees = [2, 3, 5, 7, 10, 20]
    class_weights = [None, 'balanced']
    kernels = ['rbf', 'poly']

    best_val_acc = 0
    for C in tqdm(Cs):
        for class_weight in class_weights:
            for kernel in kernels:
                for degree in degrees:
                    if kernel != 'poly':
                        continue

                    clf = SVC(
                        C=C,
                        kernel=kernel,
                        degree=degree,
                        class_weight=class_weight,
                        random_state=0,
                        max_iter=-1,
                    )

                    scores = cross_validate(
                        clf,
                        latents,
                        labels,
                        cv=10,
                        scoring=metric,
                        return_train_score=True,
                    )

                    acc = scores['test_score'].mean()

                    if best_val_acc < acc:
                        best_C = C
                        best_class_weight = class_weight
                        best_kernel = kernel
                        best_degree = degree
                        best_val_acc = acc

    return best_val_acc, {
        'C': best_C,
        'class_weight': best_class_weight,
        'kernel': best_kernel,
        'degree': best_degree,
    }


def crossvalidate_lg(latents, labels, metric):
    '''Perform grid search to find best hyperparameters given validation set.'''

    Cs = [0.5, 1.0, 3.0, 5.0, 10.0, 20.0, 30.0]
    penaltys = [None, 'l2', 'l1', 'elasticnet']
    class_weights = [None, 'balanced']

    best_val_acc = 0
    for C in tqdm(Cs):
        for class_weight in class_weights:
            for penalty in penaltys:
                clf = LogisticRegression(
                    random_state=0,
                    max_iter=100,
                    C=C,
                    penalty=penalty,
                    class_weight=class_weight,
                    solver='saga',
                    l1_ratio=0.5,
                )

                scores = cross_validate(
                    clf,
                    latents,
                    labels,
                    cv=10,
                    scoring=metric,
                    return_train_score=False,
                )
                acc = scores['test_score'].mean()

                if best_val_acc < acc:
                    best_C = C
                    best_class_weight = class_weight
                    best_penalty = penalty
                    best_val_acc = acc

    return best_val_acc, {
        'C': best_C,
        'class_weight': best_class_weight,
        'penalty': best_penalty,
        # 'solver': 'saga',
        'l1_ratio': 0.5,
    }


def plot_confusion(
    predictions,
    labels,
    label_names,
    classification_type,
    normalize=False,
    out_path=None,
):
    '''Plot confusion matrix.'''
    cm = confusion_matrix(labels, predictions)

    suffix = ''
    if normalize:
        cm = cm / cm.sum(1, keepdims=True)
        suffix = '_normalized'

    fig, ax = plt.subplots(1, 1)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.OrRd, ax=ax)
    ax.set_xticklabels(list(label_names.values()), rotation='vertical')
    ax.set_yticklabels(list(label_names.values()))

    if out_path is not None:
        savepath = Path(out_path, f'{classification_type}_confusion_matrix{suffix}.pdf')
        fig.savefig(savepath, bbox_inches='tight')
        plt.close(fig)
