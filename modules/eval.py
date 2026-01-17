"""Evaluation utilities for LongR² rigid-registration.

This module runs evaluation on CSV-specified image pairs, computes label
metrics and (optionally) saves moved images to disk.
"""


import os
import time
import pickle
import pathlib
import functools
import collections
import numpy as np
import pandas as pd
import scipy.ndimage
import tensorflow as tf

# the third-party imports
import neurite as ne
import voxelmorph as vxm
import surfa as sf

# local imports
from modules.generators import get_shape, eval_generator
import modules.networks as networks
from modules.utils import load_config


def save_moved_image(pair, pair_path, save_dir, dataset, pair_id):
    """Save the moved image as a NIfTI file using the original affine."""
    os.makedirs(save_dir, exist_ok=True)
    image_moved = pair["image_moved"].numpy().squeeze()
    # Load affine from the fixed image file
    _, affine_fix = vxm.py.utils.load_volfile(pair_path["image_fixed"], ret_affine=True)
    out_path = f'{save_dir}/{dataset}_{pair_id}.nii.gz'
    vxm.py.utils.save_volfile(image_moved, out_path, affine_fix)


@tf.function
def _dice(x, y, num_label):
    return ne.metrics.HardDice(nb_labels=num_label).dice(x, y)[0]


@functools.cache
def _rebase_labels(lut_file):
    """Compile output label names and translation function from a LUT file."""
    # Load old-to-new LUT.
    if lut_file.endswith('.pickle'):
        with open(lut_file, mode='rb') as f:
            lut = pickle.load(f)
    elif lut_file.endswith('.npy'):
        lut = np.load(lut_file)

    # Dictionary.
    if not isinstance(lut, dict):
        lut = {i: i for i in lut}

    # Old-to-index lookup for TensorFlow.
    new_labels = list(set(lut.values()))
    new_to_ind = {k: i for i, k in enumerate(new_labels)}
    lut = [new_to_ind.get(lut.get(i, -1), -1) for i in range(max(lut) + 1)]
    lut = tf.cast(lut, tf.int32)

    # Lookup function.
    f = tf.function(lambda x: tf.gather(lut, indices=tf.cast(x, tf.int32)))
    return new_labels, f, lut


def save_df(path, df, append=False):
    """Save a results DataFrame to CSV.
    """
    path = pathlib.Path(path)
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    if append and path.exists():
        df = (pd.read_csv(path), df)
        df = pd.concat(df, ignore_index=True, verify_integrity=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print("\nEvaluation results saved to:", path)


def derive_mask(labels, open=10):
    """Create clean brain mask to evaluate brain-registraton accuracy only."""
    # Remove singleton dimensions.
    labels = np.asarray(labels)
    shape = np.shape(labels)
    mask = np.squeeze(labels > 0)
    full = np.zeros_like(mask)
    assert full.ndim in (2, 3), 'only 2D or 3D arrays supported'

    # Crop for speed: `nonzero` returns (num_dim, num_nonzero). Keep safety
    # margin so non-final dilations do not touch the edge, if possible.
    ind = np.nonzero(mask)
    low = np.clip(a=np.min(ind, axis=-1) - open, a_min=0, a_max=None)
    upp = np.clip(a=np.max(ind, axis=-1) + open, a_min=None, a_max=mask.shape)
    ind = tuple(slice(a, b + 1) for a, b in zip(low, upp))
    mask = mask[ind]

    # Fill holes, do not dilate as want tighter fit than SynthStrip.
    copy = np.copy(mask)
    mask = scipy.ndimage.binary_closing(mask, iterations=open)
    mask = scipy.ndimage.binary_fill_holes(mask)

    # Restore shape. If labels too close to array edge, keep larger mask.
    full[ind] = np.logical_or(mask, copy)
    return np.reshape(full, shape)


def prepare(pair):
    # Copy uint8 instead of float32 to GPU, once. Normalize there. Do not call
    # `minmax_norm` on NumPy arrays, which would copy data to the GPU twice.
    for k in pair:
        if not tf.is_tensor(pair[k]) or not pair[k].dtype.is_floating:
            pair[k] = tf.cast(pair[k], tf.float32)
            if 'image' in k:
                pair[k] = ne.utils.minmax_norm(pair[k])

    # Transform only.
    if not any('labels' in f for f in pair):
        return

    # Label-based brain masks. Needed for brain-specific registration accuracy
    # with image based metrics and transformation metrics.
    if not any('mask' in f for f in pair):
        pair['mask_moving'] = tf.cast(derive_mask(pair['labels_moving']), tf.float32)
        pair['mask_fixed'] = tf.cast(derive_mask(pair['labels_fixed']), tf.float32)

        
def register(model, pair, trans_only=False):
    # Ensure data are on the GPU, FP32, masked.
    prepare(pair)

    # Transform. Deal with bidirectional models.
    pair['trans'] = model((pair['image_moving'], pair['image_fixed']))
    if isinstance(pair['trans'], (list, tuple)):
        pair['trans'], pair['inv'] = pair['trans']

    if trans_only:
        return

    # Image interpolation.
    t = vxm.layers.SpatialTransformer
    prop = dict(shift_center=False, fill_value=0, interp_method='linear')
    pair['image_moved'] = t(**prop)((pair['image_moving'], pair['trans']))

    # Label interpolation.
    prop = dict(shift_center=False, fill_value=0, interp_method='nearest')
    pair['labels_moved'] = t(**prop)((pair['labels_moving'], pair['trans']))


def compute_metrics(pair, lut, out):
    prepare(pair)

    # Stripped images and label maps assumed normalized and on the GPU.
    lab_moved = pair['labels_moved']
    lab_fixed = pair['labels_fixed']

    # Label selection.
    new_labels, rebase, lut = _rebase_labels(lut)
    lab_moved = rebase(lab_moved)
    lab_fixed = rebase(lab_fixed)

    # Label metrics.
    num_label = len(new_labels)
    dice = _dice(lab_moved, lab_fixed, num_label).numpy()

    for n, lab in enumerate(new_labels):
        out[f'Dice-{lab}'].append(dice[n])
    out['Dice-mean'].append(np.mean(dice))


def eval(config, section):
    # Data.
    sets = config[section].pop('data')
    if isinstance(sets, str):
        sets = [sets]

    # Model.
    arg_reg = config.pop('model')
    arg_reg.update(
        in_shape=get_shape(sets[0]),
        bidir=config[section].pop('bidir', True),
        make_dense=config[section].pop('make_dense', True),
    )
    model = getattr(networks, arg_reg.pop('name'))(**arg_reg)

    run_name = config[section].get('run_name', '')
    weights_path = config[section].pop('weights').format(run_name=run_name)
    model.load_weights(weights_path)

    # Eval loop.
    out = collections.defaultdict(list)
    labels_path = config[section].pop('labels')
    out_fig = config[section].pop('out_fig', None)

    for set_idx, csv in enumerate(sets):
        dat = csv.split('/')[-1].replace('.csv', '')
        print(f"\nProcessing dataset {set_idx+1}/{len(sets)}: {dat}")

        gen = eval_generator(csv)
        total_pairs = sum(1 for _ in eval_generator(csv))
        for k, pair in enumerate(gen):
            pair, pair_path = pair
            print(f"  Pair {k+1}/{total_pairs} ...", end=' ', flush=True)
            
            start_time = time.time()
            register(model, pair)
            elapsed = time.time() - start_time
            print(f"done (reg {elapsed:.2f}s)")

            compute_metrics(pair, labels_path, out)

            out['dataset'].append(dat)
            out['pair'].append(f'{k:04d}')

            # Save moved images if out_fig is provided
            if out_fig is not None:
                save_moved_image(pair, pair_path, out_fig, dat, f'{k:04d}')
        
        print(f"Finished dataset {dat}.")

    if out_fig is not None:
        print(f"\nMoved images saved to: {out_fig}")
        
    save_df(config[section].pop('save_name'), out)


if __name__ == "__main__":
    config_path = "configs/config.yaml"
    config = load_config(config_path)
    eval(config, "eval")