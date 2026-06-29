"""Train a minimal PyTorch Lightning linear probe on TRIPSO embeddings.

Supports either:

1. HuggingFace datasets saved with ``datasets.save_to_disk`` where each gene
   program (GP) is a feature column, e.g. ``TNFa`` or ``TGFb``.
2. AnnData ``.h5ad`` files where embeddings live in ``.X`` and GP-specific
   dimensions can be selected from ``adata.var.index``.

The probe is exposed as :func:`run_linear_probing` and wired into the package
CLI as the ``probe`` subcommand (see ``tripso/__main__.py``).
"""

import json
import os
import random
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..Utils.utils import wrangle_classification_report

# ===================================
# Reproducibility
# ===================================


def seed_everything(seed):
    """Seed Python, NumPy, PyTorch and Lightning for reproducibility.

    Parameters
    ----------
    seed : int
        Seed value applied across all random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pl.seed_everything(seed, workers=True)


# ===================================
# Data wrangling
# ===================================


def _dense_float32(x):
    """Return a dense float32 array, densifying sparse input if needed.

    Parameters
    ----------
    x : numpy.ndarray or scipy.sparse matrix
        Input feature matrix.

    Returns
    -------
    numpy.ndarray
        Dense float32 array.
    """
    if sp.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def _obs_filter(df, key, value):
    """Build a boolean mask selecting rows whose ``key`` column contains ``value``.

    Parameters
    ----------
    df : pandas.DataFrame
        Metadata frame to filter.
    key : str or None
        Column to filter on. If ``None``, all rows are kept.
    value : str or None
        Substring to match within ``key``. Required when ``key`` is given.

    Returns
    -------
    numpy.ndarray
        Boolean mask of length ``len(df)``.
    """
    if key is None:
        return np.ones(len(df), dtype=bool)
    if key not in df.columns:
        raise KeyError(
            f'filter key {key!r} not found. Available columns: {list(df.columns)}'
        )
    if value is None:
        raise ValueError('filter_obs_key requires filter_obs_value')
    return df[key].astype(str).str.contains(value, regex=False, na=False).to_numpy()


def _select_gp_vars(adata, gp, mode):
    """Subset AnnData vars to the dimensions matching a gene program.

    Parameters
    ----------
    adata : anndata.AnnData
        Input AnnData with embeddings in ``.X``.
    gp : str
        Gene program name or pattern to match against ``var_names``.
    mode : {'cell_token', 'gp_vars_contains', 'gp_vars_prefix', 'gp_vars_exact'}
        How to match ``gp`` against the variable names. ``'cell_token'`` keeps
        the full matrix unchanged.

    Returns
    -------
    anndata.AnnData
        View (copy) restricted to the matched variables.
    """
    if mode == 'cell_token':
        return adata
    names = adata.var_names.astype(str)
    if mode == 'gp_vars_contains':
        mask = names.str.contains(gp, case=False, regex=False)
    elif mode == 'gp_vars_prefix':
        mask = names.str.lower().str.startswith(gp.lower())
    elif mode == 'gp_vars_exact':
        mask = names.str.lower() == gp.lower()
    else:
        raise ValueError(f'Unknown feature mode: {mode}')
    if int(mask.sum()) == 0:
        raise ValueError(f'No AnnData vars matched gp={gp!r} with mode={mode!r}.')
    return adata[:, mask].copy()


def dataset_to_xy(path, gp, label_col, filter_key, filter_value):
    """Load features, labels and metadata from a HuggingFace dataset folder.

    Parameters
    ----------
    path : str
        Path to a dataset saved with ``datasets.save_to_disk``.
    gp : str
        Feature column holding the gene program embeddings.
    label_col : str
        Column holding the target labels.
    filter_key : str or None
        Optional metadata column to filter rows on.
    filter_value : str or None
        Substring to match within ``filter_key``.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray, dict)
        Feature matrix, string labels, and a mapping of column name to the
        filtered metadata series.
    """
    ds = load_from_disk(str(path))
    df = ds.select_columns([c for c in ds.column_names if c != gp]).to_pandas()
    keep = _obs_filter(df, filter_key, filter_value)
    if gp not in ds.column_names:
        raise KeyError(
            f'GP column {gp!r} not found. Available columns: {ds.column_names}'
        )
    if label_col not in df.columns:
        raise KeyError(
            f'label column {label_col!r} not found. '
            f'Available columns: {list(df.columns)}'
        )
    x = np.asarray(ds[gp], dtype=np.float32)[keep]
    y = df.loc[keep, label_col].astype(str).to_numpy()
    groups = df.loc[keep].to_dict('series')
    return x, y, groups


def h5ad_to_xy(
    path,
    gp,
    label_col,
    feature_mode,
    filter_key,
    filter_value,
):
    """Load features, labels and metadata from an AnnData ``.h5ad`` file.

    Parameters
    ----------
    path : str
        Path to a ``.h5ad`` file with embeddings in ``.X``.
    gp : str
        Gene program name or pattern used to select variables.
    label_col : str
        Column in ``adata.obs`` holding the target labels.
    feature_mode : str
        How to select variables, see :func:`_select_gp_vars`.
    filter_key : str or None
        Optional ``adata.obs`` column to filter rows on.
    filter_value : str or None
        Substring to match within ``filter_key``.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray, dict)
        Feature matrix, string labels, and a mapping of ``obs`` column name to
        the corresponding series.
    """
    adata = ad.read_h5ad(path)
    keep = _obs_filter(adata.obs, filter_key, filter_value)
    adata = adata[keep].copy()
    adata = _select_gp_vars(adata, gp, feature_mode)
    if label_col not in adata.obs.columns:
        raise KeyError(
            f'label column {label_col!r} not found. '
            f'Available columns: {list(adata.obs.columns)}'
        )
    x = _dense_float32(adata.X)
    y = adata.obs[label_col].astype(str).to_numpy()
    groups = {c: adata.obs[c].reset_index(drop=True) for c in adata.obs.columns}
    return x, y, groups


def make_indices(
    y,
    groups,
    seed,
    val_size,
    test_size,
    has_external_test,
):
    """Build train/val/test index arrays with optional group-aware splitting.

    Parameters
    ----------
    y : numpy.ndarray
        String labels used for stratification.
    groups : Sequence or None
        Group labels kept wholly within one split (e.g. gene or cell id). If
        ``None``, splits are stratified by ``y`` instead.
    seed : int
        Random seed for the splitters.
    val_size : float
        Fraction of the data used for validation.
    test_size : float
        Fraction of the data used for testing. Ignored when
        ``has_external_test`` is ``True``.
    has_external_test : bool
        Whether a separate test set is supplied externally.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray, numpy.ndarray or None)
        Train, validation and test index arrays. The test array is ``None``
        when ``has_external_test`` is ``True``.
    """
    idx = np.arange(len(y))
    if has_external_test:
        trainval_idx, test_idx = idx, None
    elif groups is not None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        trainval_idx, test_idx = next(splitter.split(idx, y, groups))
    else:
        trainval_idx, test_idx = train_test_split(
            idx, test_size=test_size, stratify=y, random_state=seed
        )

    y_trainval = y[trainval_idx]
    val_fraction = val_size if has_external_test else val_size / (1.0 - test_size)
    if groups is not None:
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=val_fraction, random_state=seed
        )
        tr_rel, va_rel = next(
            splitter.split(trainval_idx, y_trainval, np.asarray(groups)[trainval_idx])
        )
    else:
        tr_rel, va_rel = train_test_split(
            np.arange(len(trainval_idx)),
            test_size=val_fraction,
            stratify=y_trainval,
            random_state=seed,
        )
    return trainval_idx[tr_rel], trainval_idx[va_rel], test_idx


# ===================================
# Datasets
# ===================================


class ArrayDataset(Dataset):
    """In-memory dataset wrapping feature and label arrays.

    Parameters
    ----------
    x : numpy.ndarray
        Feature matrix of shape ``(n_samples, n_features)``.
    y : numpy.ndarray
        Integer-encoded labels of shape ``(n_samples,)``.
    """

    def __init__(self, x, y):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class LinearProbeDataModule(pl.LightningDataModule):
    """Lightning DataModule serving train/val/test array datasets.

    Parameters
    ----------
    x_train, x_val, x_test : numpy.ndarray
        Feature matrices for each split.
    y_train, y_val, y_test : numpy.ndarray
        Integer-encoded labels for each split.
    batch_size : int
        Mini-batch size for all dataloaders.
    num_workers : int
        Number of worker processes for the dataloaders.
    """

    def __init__(
        self,
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        batch_size,
        num_workers,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=['x_train', 'y_train', 'x_val', 'y_val', 'x_test', 'y_test']
        )
        self.train_ds = ArrayDataset(x_train, y_train)
        self.val_ds = ArrayDataset(x_val, y_val)
        self.test_ds = ArrayDataset(x_test, y_test)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )


# ===================================
# Lightning model
# ===================================


class LinearProbe(pl.LightningModule):
    """Single linear layer probe trained with cross-entropy.

    Parameters
    ----------
    in_dim : int
        Number of input features.
    n_classes : int
        Number of target classes.
    lr : float
        Learning rate for the AdamW optimizer.
    weight_decay : float
        Weight decay (L2 regularization) for the optimizer.
    class_weights : Sequence or None, optional
        Per-class weights for the cross-entropy loss.
    """

    def __init__(self, in_dim, n_classes, lr, weight_decay, class_weights=None):
        super().__init__()
        self.save_hyperparameters(ignore=['class_weights'])
        self.linear = nn.Linear(in_dim, n_classes)
        # Non-persistent: only needed for the training loss, and keeping it out
        # of the state_dict lets load_from_checkpoint() rebuild the probe with
        # class_weights=None (the ignored hparam) without key mismatches.
        self.register_buffer(
            'class_weights',
            None
            if class_weights is None
            else torch.as_tensor(class_weights, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, x):
        return self.linear(x)

    def _step(self, batch, stage):
        """Shared logic for a single train/val/test step."""
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y, weight=self.class_weights)
        pred = logits.argmax(dim=1)
        acc = (pred == y).float().mean()
        self.log(
            f'{stage}_loss',
            loss,
            prog_bar=(stage != 'train'),
            on_epoch=True,
            on_step=False,
        )
        self.log(
            f'{stage}_acc',
            acc,
            prog_bar=(stage != 'train'),
            on_epoch=True,
            on_step=False,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        self._step(batch, 'val')

    def test_step(self, batch, batch_idx):
        self._step(batch, 'test')

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )


# ===================================
# Reporting
# ===================================


def save_curves(metrics_csv, out_png):
    """Plot training/validation loss and accuracy curves to a PNG.

    Parameters
    ----------
    metrics_csv : pathlib.Path
        Path to the Lightning ``CSVLogger`` metrics file.
    out_png : pathlib.Path
        Destination path for the rendered figure.
    """
    hist = pd.read_csv(metrics_csv)
    curve = hist.groupby('epoch', as_index=False).last(numeric_only=True)
    metric_cols = [
        c
        for c in ['train_loss', 'val_loss', 'train_acc', 'val_acc']
        if c in curve.columns
    ]
    if not metric_cols:
        return
    ax = curve.plot(x='epoch', y=metric_cols, marker='o')
    ax.set_xlabel('epoch')
    ax.set_title('Linear probe training curves')
    ax.figure.tight_layout()
    ax.figure.savefig(out_png, dpi=200)
    plt.close(ax.figure)


def predict(model, loader):
    """Run a trained probe over a dataloader and return integer predictions.

    Parameters
    ----------
    model : LinearProbe
        Trained probe in evaluation mode.
    loader : torch.utils.data.DataLoader
        Dataloader yielding ``(x, y)`` batches.

    Returns
    -------
    numpy.ndarray
        Concatenated predicted class indices.
    """
    model.eval()
    preds = []
    device = model.device
    with torch.no_grad():
        for x, _ in loader:
            preds.append(model(x.to(device)).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


# ===================================
# Probe entry point
# ===================================


def run_linear_probing(
    data_type,
    gp,
    label_col,
    data_path=None,
    train_path=None,
    test_path=None,
    group_col=None,
    feature_mode='gp_vars_contains',
    filter_obs_key=None,
    filter_obs_value=None,
    output_dir='linear_probe_results',
    val_size=0.15,
    test_size=0.20,
    batch_size=256,
    max_epochs=100,
    patience=10,
    lr=1e-3,
    weight_decay=1e-4,
    num_workers=4,
    seed=0,
    no_class_weights=False,
):
    """Train and evaluate a Lightning linear probe on TRIPSO embeddings.

    Either supply ``data_path`` for an internal train/val/test split, or both
    ``train_path`` and ``test_path`` to use an external test split.

    Parameters
    ----------
    data_type : {'dataset', 'h5ad'}
        Format of the input data.
    gp : str
        GP column name for HuggingFace datasets, or GP var-name pattern for
        AnnData.
    label_col : str
        Column holding the target labels.
    data_path : str or None
        Single input used for an internal train/val/test split.
    train_path, test_path : str or None
        External train and test splits. Provide both or neither.
    group_col : str or None
        Optional group column kept wholly within one split, e.g. gene or idx.
    feature_mode : str
        Variable selection mode for AnnData inputs, see
        :func:`_select_gp_vars`.
    filter_obs_key, filter_obs_value : str or None
        Optional metadata filter applied before splitting.
    output_dir : str
        Directory where results and checkpoints are written.
    val_size, test_size : float
        Validation and test fractions.
    batch_size : int
        Mini-batch size.
    max_epochs : int
        Maximum number of training epochs.
    patience : int
        Early-stopping patience on ``val_loss``.
    lr : float
        Learning rate.
    weight_decay : float
        Weight decay for the optimizer.
    num_workers : int
        Dataloader worker processes.
    seed : int
        Random seed.
    no_class_weights : bool
        If ``True``, disable inverse-frequency class weighting.

    Returns
    -------
    dict
        Summary metrics, also written to ``summary.json`` in ``output_dir``.
    """
    seed_everything(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if bool(train_path) != bool(test_path):
        raise ValueError('Provide both train_path and test_path, or neither.')
    has_external_test = bool(train_path and test_path)
    if not has_external_test and not data_path:
        raise ValueError('Provide data_path, or provide train_path plus test_path.')

    load_fn = dataset_to_xy if data_type == 'dataset' else h5ad_to_xy
    if data_type == 'dataset':
        load_kwargs = dict(
            gp=gp,
            label_col=label_col,
            filter_key=filter_obs_key,
            filter_value=filter_obs_value,
        )
    else:
        load_kwargs = dict(
            gp=gp,
            label_col=label_col,
            feature_mode=feature_mode,
            filter_key=filter_obs_key,
            filter_value=filter_obs_value,
        )

    if has_external_test:
        x_all, y_all, meta = load_fn(train_path, **load_kwargs)
        x_test, y_test_raw, _ = load_fn(test_path, **load_kwargs)
        groups = meta.get(group_col) if group_col else None
        tr_idx, va_idx, _ = make_indices(y_all, groups, seed, val_size, test_size, True)
        x_train_raw, y_train_raw = x_all[tr_idx], y_all[tr_idx]
        x_val_raw, y_val_raw = x_all[va_idx], y_all[va_idx]
    else:
        x_all, y_all, meta = load_fn(data_path, **load_kwargs)
        groups = meta.get(group_col) if group_col else None
        tr_idx, va_idx, te_idx = make_indices(
            y_all, groups, seed, val_size, test_size, False
        )
        x_train_raw, y_train_raw = x_all[tr_idx], y_all[tr_idx]
        x_val_raw, y_val_raw = x_all[va_idx], y_all[va_idx]
        x_test, y_test_raw = x_all[te_idx], y_all[te_idx]

    label_encoder = LabelEncoder().fit(y_train_raw)
    known = np.isin(y_test_raw, label_encoder.classes_)
    if not known.all():
        dropped = sorted(set(y_test_raw[~known]))
        print(
            f'[WARN] Dropping {np.sum(~known)} test rows with labels unseen '
            f'in train: {dropped}'
        )
        x_test, y_test_raw = x_test[known], y_test_raw[known]

    y_train = label_encoder.transform(y_train_raw)
    y_val = label_encoder.transform(y_val_raw)
    y_test = label_encoder.transform(y_test_raw)

    scaler = StandardScaler().fit(x_train_raw)  # fit train only: no leakage
    x_train = scaler.transform(x_train_raw).astype(np.float32)
    x_val = scaler.transform(x_val_raw).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    counts = np.bincount(y_train, minlength=len(label_encoder.classes_))
    class_weights = (
        None
        if no_class_weights
        else (counts.sum() / np.maximum(counts, 1) / len(counts))
    )

    dm = LinearProbeDataModule(
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        batch_size,
        num_workers,
    )
    model = LinearProbe(
        x_train.shape[1],
        len(label_encoder.classes_),
        lr,
        weight_decay,
        class_weights,
    )

    logger = pl.loggers.CSVLogger(save_dir=str(out), name='logs')
    ckpt = ModelCheckpoint(
        monitor='val_loss', mode='min', save_top_k=1, filename='best'
    )
    early = EarlyStopping(monitor='val_loss', mode='min', patience=patience)
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='auto',
        devices='auto',
        logger=logger,
        callbacks=[ckpt, early],
        deterministic=True,
        enable_checkpointing=True,
    )
    trainer.fit(model, dm)

    best_model = (
        LinearProbe.load_from_checkpoint(ckpt.best_model_path)
        if ckpt.best_model_path
        else model
    )
    y_pred = predict(best_model, dm.test_dataloader())
    target_names = label_encoder.inverse_transform(
        np.arange(len(label_encoder.classes_))
    )
    report = classification_report(
        y_test,
        y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    report_df = wrangle_classification_report(report)
    report_df.to_csv(out / 'classification_report.csv', index=False)

    pred_df = pd.DataFrame(
        {
            'y_true': label_encoder.inverse_transform(y_test),
            'y_pred': label_encoder.inverse_transform(y_pred),
        }
    )
    pred_df.to_csv(out / 'predictions.csv', index=False)

    summary = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'macro_f1': float(report['macro avg']['f1-score']),
        'weighted_f1': float(report['weighted avg']['f1-score']),
        'n_train': int(len(y_train)),
        'n_val': int(len(y_val)),
        'n_test': int(len(y_test)),
        'n_features': int(x_train.shape[1]),
        'classes': target_names.tolist(),
        'best_checkpoint': ckpt.best_model_path,
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))

    metrics_csv = Path(logger.log_dir) / 'metrics.csv'
    if os.path.exists(metrics_csv):
        save_curves(metrics_csv, out / 'training_curves.png')

    print(json.dumps(summary, indent=2))
    print(f'Saved outputs to: {out.resolve()}')
    return summary
