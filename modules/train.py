# Train symmetric longitudinal rigid registration using YAML config.


import os
import pickle
import numpy as np
import tensorflow as tf

# the third-party imports
import neurite as ne
import voxelmorph as vxm

# local imports
from modules.generators import train_generator
import modules.networks as networks
from modules.utils import load_config


def train(config, section):
    # Data.
    labels_in, label_maps = vxm.py.utils.load_labels(config[section].pop('data'))
    in_shape = label_maps[0].shape
    batch_size = config[section].pop('batch_size', 1)
    same_subj = config[section].pop('same_subj', False)
    gen = train_generator(label_maps, batch_size, same_subj)


    # Output labels.
    labels_out = config[section].pop('out_labels', '')
    if labels_out.endswith('.npy'):
        labels_out = sorted(i for i in np.load(labels_out) if i in labels_in)
    elif labels_out.endswith('.pickle'):
        with open(labels_out, 'rb') as f:
            labels_out = {k: v for k, v in pickle.load(f).items() if k in labels_in}
    else:
        labels_out = [i for i in labels_in if i != 0]


    # Synthesis.
    arg_gen = config.pop('synthesis')
    arg_gen.update(
        in_shape=in_shape,
        labels_in=labels_in,
        labels_out=labels_out,
        half_res=False,
    )
    gen_model_1 = ne.models.labels_to_image_new(**arg_gen, id=0)
    gen_model_2 = ne.models.labels_to_image_new(**arg_gen, id=1)
    ima_1, map_1 = gen_model_1.outputs
    ima_2, map_2 = gen_model_2.outputs


    # Registration.
    inputs = (*gen_model_1.inputs, *gen_model_2.inputs)
    arg_reg = config.pop('model')
    arg_reg.update(
        input_model=tf.keras.Model(inputs, outputs=(ima_1, ima_2)),
        bidir=True,
        make_dense=True,
        half_res=config[section].pop('half_res', True),
        return_trans_to_half_res=config[section].pop('loss_half_res', False),
        return_trans_to_mid_space=config[section].pop('loss_mid_space', False),
    )
    model = getattr(networks, arg_reg.pop('name'))(**arg_reg)
    aff_1, aff_2 = model.outputs


    # Moved labels.
    shape = aff_1.shape[1:-1]
    prop = dict(fill_value=0, shape=shape, shift_center=False)
    mov_1 = vxm.layers.SpatialTransformer(**prop)((map_1, aff_1))
    mov_2 = vxm.layers.SpatialTransformer(**prop)((map_2, aff_2))

    if arg_reg['return_trans_to_half_res']:
        dim = len(shape)
        scale = 2 * tf.eye(dim, dim + 1, batch_shape=[batch_size])
        map_2 = vxm.layers.SpatialTransformer(**prop)((map_2, scale))

    out = (mov_1, mov_2 if arg_reg['return_trans_to_mid_space'] else map_2)


    # Loss.
    loss = config[section].pop('loss')
    loss = getattr(vxm.losses, loss)().loss(*out)

    optim = tf.keras.optimizers.Adam(learning_rate=float(config[section].pop('lr')))
    model.add_loss(loss)
    model.compile(optim)
    model.summary()

    # Callbacks.
    run_name = config[section].pop('run_name', 'rigid_reg')
    log_dir = os.path.join('logs', run_name)
    save_name = config[section].pop('save_name').format(run_name=run_name)
    steps_per_epoch = config[section].pop('steps_per_epoch')
    callbacks = (
        tf.keras.callbacks.TensorBoard(log_dir),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=save_name,
            save_weights_only=True,
            save_freq=steps_per_epoch * config[section].pop('save_freq'),
        ),
    )


    # Training.
    prop = dict(
        initial_epoch=0,
        epochs=config[section].pop('epochs'),
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        verbose=config[section].pop('verbose', 1),
    )
    assert not config[section], f'config not fully consumed: {config[section]}'
    model.fit(gen, **prop)

if __name__ == "__main__":
    config_path = "configs/config.yaml"
    config = load_config(config_path)
    if config.get("setup", {}).get("xla", False):
        tf.config.optimizer.set_jit(True)
    train(config, "train")
