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


def save_outputs(pair, pair_path, save_dir, dataset, pair_id):
    """Save the moved image as a NIfTI file, 
       and save transformation matrices as a LTA file.
    """
    os.makedirs(save_dir, exist_ok=True)
    image_moved = pair['image_moved'].numpy().squeeze()
    # Load affine from the fixed image file
    _, affine_fix = vxm.py.utils.load_volfile(pair_path['image_fixed'], ret_affine=True)
    out_path = f'{save_dir}/{dataset}_{pair_id}.nii.gz'
    vxm.py.utils.save_volfile(image_moved, out_path, affine_fix)

    # Get transform from moving to fixed image. FreeSurfer LTAs store the inverse.
    trans = sf.Affine(tf.concat([tf.squeeze(pair['inv']), tf.constant([[0, 0, 0, 1]], dtype=tf.float32)], axis=0).numpy(), 
                                 source=sf.load_volume(pair_path['image_moving']), target=sf.load_volume(pair_path['image_fixed']), space='vox')
    out_name = f'{save_dir}/{dataset}_{pair_id}.lta'
    trans.save(out_name)

    # If instance-optimized outputs exist, save them too.
    if 'image_moved_ins_opt' in pair:
        image_moved_i = pair['image_moved_ins_opt'].numpy().squeeze()
        out_path_i = f'{save_dir}/{dataset}_{pair_id}_insopt.nii.gz'
        vxm.py.utils.save_volfile(image_moved_i, out_path_i, affine_fix)

    if 'trans_ins_opt' in pair:
        # Get transform from fixed to moving image. FreeSurfer LTAs store the inverse.
        trans = sf.Affine(tf.concat([tf.squeeze(pair['trans_ins_opt']), 
                                        tf.constant([[0, 0, 0, 1]], 
                                                    dtype=tf.float32)], axis=0).numpy(), 
                            source=sf.load_volume(pair_path['image_fixed']),  
                            target=sf.load_volume(pair_path['image_moving']), space='vox')
        trans = trans.inv()
        out_name = f'{save_dir}/{dataset}_{pair_id}_insopt.lta'
        trans.save(out_name)


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

    # If instance-optimized labels are available, compute same metrics
    if 'labels_moved_ins_opt' in pair:
        lab_moved_ins = rebase(pair['labels_moved_ins_opt'])
        dice_ins = _dice(lab_moved_ins, lab_fixed, num_label).numpy()
        for n, lab in enumerate(new_labels):
            out[f'Dice-insopt-{lab}'].append(dice_ins[n])
        out['Dice-mean-insopt'].append(np.mean(dice_ins))


def instance_optimization(ins_model, pair, pair_path, ins_opt_args):
    """Instance-specific optimization for a given image pair.
    """
    # Parse args.
    img_loss=ins_opt_args.get('ins_opt_loss')
    lr=ins_opt_args.get('ins_opt_lr')
    epochs=ins_opt_args.get('ins_opt_epochs')

    # Loss.
    if img_loss == 'ncc':
        image_loss_func = vxm.losses.NCC().loss
    elif img_loss == 'mse':
        image_loss_func = vxm.losses.MSE().loss
    elif img_loss == 'mi':
        image_loss_func = vxm.losses.MutualInformation(nb_bins = 64).loss
    else:
        raise ValueError('Image loss should be "mse", "ncc" or "mi", but found "%s"' % img_loss)

    Fake_loss = vxm.losses.MSE().loss
    losses = [image_loss_func, Fake_loss]
    weights = [1, 0]

    # Brain extraction. Make mse more robust for full head
    pair['image_moving'] *= pair['mask_moving']
    pair['image_fixed'] *= pair['mask_fixed']

    # Check for GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Using GPU: {gpus[0]}")
    else:
        print("No GPU found. Using CPU.")
        
    # Initialize zeros
    zeros = np.zeros((1, *(3, 4)), dtype='float32')
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
    # Define the loss function (assuming it is a custom loss function)
    # If you have multiple loss functions, you may need to compute each one separately and apply weights
    def compute_loss(y_true, y_pred, zeros):
        mov_1, aff_1 = y_pred
        loss_fixed = losses[0](y_true, mov_1)
        loss_zeros = losses[1](zeros, aff_1)
        return weights[0] * loss_fixed + weights[1] * loss_zeros
    
    # Custom training loop
    for epoch in range(epochs):
        for step in range(1):  # Assuming steps_per_epoch is 1
            with tf.GradientTape() as tape:
                # Forward pass
                predictions = ins_model(pair['image_moving'], training=True)
                
                # Compute loss
                loss = compute_loss(pair['image_fixed'], predictions, zeros)
            
            # Compute gradients
            gradients = tape.gradient(loss, ins_model.trainable_variables)
            
            # Apply gradients
            optimizer.apply_gradients(zip(gradients, ins_model.trainable_variables))
            
            # Log progress (modify the verbosity as needed)
            # print(f"Epoch {epoch+1}, Step {step+1}, Loss: {loss.numpy():.2e}")
            print(f"Epoch {epoch+1}, Step {step+1}, Loss: {loss.numpy().mean():.2e}")

    # Get warped image and lab.
    _, aff_1 = ins_model(pair['image_moving'])
    pair['trans_ins_opt'] = aff_1.numpy()
    
    # Image interpolation.
    t = vxm.layers.SpatialTransformer
    prop = dict(shift_center=False, fill_value=0, interp_method='linear')
    pair['image_moved_ins_opt'] = t(**prop)((pair['image_moving'], pair['trans_ins_opt']))

    # Label interpolation.
    prop = dict(shift_center=False, fill_value=0, interp_method='nearest')
    pair['labels_moved_ins_opt'] = t(**prop)((pair['labels_moving'], pair['trans_ins_opt']))

    # Clear session to free up memory
    import gc
    # del ins_model
    tf.keras.backend.clear_session()
    gc.collect()


def eval(config, section):
    # Data.
    sets = config[section].pop('data')
    if isinstance(sets, str):
        sets = [sets]

    # Model.
    arg_reg = config.pop('model')
    inshape = get_shape(sets[0])
    arg_reg.update(
        in_shape=inshape,
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
    # Instance-specific optimization flags (read once)
    ins_opt = config[section].pop('ins_opt', False)
    ins_opt_args = config[section].pop('ins_opt_args', {})

    if ins_opt:
        print("Warning: instance-specific optimization is enabled — this may be slow; a GPU is recommended for reasonable speed.")

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

            # If do instance specific optimization
            if ins_opt:
                # Model initialization.
                ins_model = networks.AffineInstanceDense(
                    inshape,
                    aff_shape = (3, 4),
                    nb_feats=1,
                    mult= 1, # Must be 1.
                )
                ins_model.set_flow(pair['trans'])
                instance_optimization(ins_model, pair, pair_path, ins_opt_args)

            compute_metrics(pair, labels_path, out)

            out['dataset'].append(dat)
            out['pair'].append(f'{k:04d}')

            # Save moved images and transformation matrices if out_fig is provided
            if out_fig is not None:
                save_outputs(pair, pair_path, out_fig, dat, f'{k:04d}')
        
        print(f"Finished dataset {dat}.")

    if out_fig is not None:
        print(f"\nMoved images saved to: {out_fig}")
        
    save_df(config[section].pop('save_name'), out)


if __name__ == "__main__":
    config_path = "configs/config.yaml"
    config = load_config(config_path)
    eval(config, "eval")