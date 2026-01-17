import numpy as np
import pandas as pd
import voxelmorph as vxm
import neurite as ne
import tensorflow as tf
import pathlib
import os

def load_image(f):
    out = vxm.py.utils.load_volfile(f, add_batch_axis=True, add_feat_axis=True)
    return out


def train_generator(label_maps, batch_size=1, same_subj=False):
    """
    Generator that yields batches of label maps for training.
    Args:
        label_maps (np.ndarray): Array of label maps.
        batch_size (int): Number of samples per batch.
        same_subj (bool): If True, second half of batch is a copy of the first.
    Yields:
        tuple: ((moving, fixed), None)
    """
    label_maps = np.expand_dims(label_maps, axis=-1)
    rand = np.random.default_rng()

    while True:
        x = rand.choice(label_maps, size=2 * batch_size)
        if same_subj:
            x[batch_size:] = x[:batch_size]
        yield (x[:batch_size], x[batch_size:]), None


def eval_generator(csv_file, trans_only=False):
    """Load a CSV file into a dictionary using the column names as keys."""
    df = pd.read_csv(csv_file)
    for _, f in df.iterrows():  
        out = {} if trans_only else {k: load_image(v) for k, v in f.items()}

        yield out, f


def get_shape(csv_file):
    """Derive spatial dimensions from dataset CSV file."""
    pair, _ = next(eval_generator(csv_file))
    return next(iter(pair.values())).shape[1:-1]