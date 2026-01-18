
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as KL
from voxelmorph import layers, utils
import neurite as ne

# Adapted from voxelmorph.networks.VxmAffineFeatureDetector,
# Support unequal number of features for encoder and decoder (e.g., truncatedunet).
class VxmAffineFeatureDetector(tf.keras.Model):
    """
    SynthMorph network for symmetric affine or rigid registration of two images.

    If you find this work useful, please cite:
        Anatomy-specific acquisition-agnostic affine registration learned from fictitious images
        M Hoffmann, A Hoopes, B Fischl*, AV Dalca* (*equal contribution)
        SPIE Medical Imaging: Image Processing, 12464, p 1246402, 2023
        https://doi.org/10.1117/12.265325

        Learning Accurate Rigid Registration for Longitudinal Brain MRI from Synthetic Data
        J Fu, AV Dalca, B Fischl, R Moreno, and M Hoffmann
        IEEE 22nd International Symposium on Biomedical Imaging (ISBI), pp. 1–5, Houston, TX, USA, 2025
        https://ieeexplore.ieee.org/document/10980859

    """

    def __init__(self,
                 in_shape=None,
                 input_model=None,
                 num_chan=1,
                 num_feat=64,
                 enc_nf=[256] * 4,
                 dec_nf=[256] * 0,
                 add_nf=[256] * 0,
                 per_level=1,
                 dropout=0,
                 half_res=True,
                 weighted=True,
                 rigid=False,
                 make_dense=True,
                 bidir=False,
                 return_trans_to_mid_space=False,
                 return_trans_to_half_res=False,
                 return_moved=False,
                 return_feat=False):
        """
        Internally, the model computes transforms in a centered frame at full resolution. However,
        matrix transforms returned with `make_dense=False` operate on zero-based indices to
        facilitate composition, in particular when changing resolution. Thus, any subsequent
        `SpatialTransformer` or `ComposeTransform` calls require `shift_center=False`.

        While the returned transforms always apply to full-resolution images, you can use the flag
        `return_trans_to_half_res=True` to obtain transforms producing outputs at half resolution,
        for faster training. Careful: this requires setting the adequate output `shape` for
        `SpatialTransformer` when applying transforms.

        Parameters:
            in_shape: Spatial dimensions of the input images, as an iterable.
            input_model: Model whose outputs will be used as data inputs, and whose inputs will be
                used as inputs to the returned model, as an alternative to specifying `in_shape`.
            num_chan: Number of input-image channels.
            num_feat: Number of output feature maps giving rise to centers of mass.
            enc_nf: Number of convolutional encoder filters at each level, as an iterable. The
                model will downsample by a factor of 2 after each convolution.
            dec_nf: Number of convolutional decoder filters at each level, as an iterable. The
                model will upsample by a factor of 2 after each convolution.
            add_nf: Number of additional convolutional filters applied at the end, as an iterable.
                The model will maintain the resolution after these convolutions.
            per_level: Number of encoding and decoding convolution repeats.
            dropout: Spatial dropout rate applied after each convolution.
            half_res: For efficiency, halve the input-image resolution before registration.
            weighted: Fit transforms using weighted instead of ordinary least squares.
            rigid: Discard scaling and shear to return a rigid transform.
            make_dense: Return a dense displacement field instead of a matrix transform.
            bidir: In addition to the transform from image 1 to image 2, also return the inverse.
                The transforms apply to full-resolution images but may end half way and/or at half
                resolution, depending on `return_trans_to_mid_space`, `return_trans_to_half_res`.
                Also return pairs of moved images and feature maps, if requested.
            return_trans_to_mid_space: Return transforms from the input images to the mid-space.
                Careful: your loss inputs must reflect this choice, and training with large
                transforms may lead to NaN loss values. You can change this option after training.
            return_trans_to_half_res: Return transforms from input images at full resolution to
                output images at half resolution. You can change this option after training.
            return_moved: Append the transformed images to the model outputs.
            return_feat: Append the output feature maps to the model outputs.

        """
        # Original inputs.
        if input_model is None:
            inp_1 = tf.keras.Input(shape=(*in_shape, num_chan))
            inp_2 = tf.keras.Input(shape=(*in_shape, num_chan))
            input_model = tf.keras.Model(*[(inp_1, inp_2)] * 2)
        inp_1, inp_2 = input_model.outputs[:2]

        # Dimensions.
        shape_full = np.asarray(inp_1.shape[1:-1])
        shape_half = shape_full // 2
        num_dim = len(shape_full)
        assert num_dim in (2, 3), 'only 2D and 3D supported'
        assert not return_trans_to_half_res or half_res, 'only for `half_res=True`'

        # Layers.
        conv = getattr(KL, f'Conv{num_dim}D')
        pool = getattr(KL, f'MaxPool{num_dim}D')
        drop = getattr(KL, f'SpatialDropout{num_dim}D')
        up = getattr(KL, f'UpSampling{num_dim}D')

        # Static transforms. Function names refer to effect on coordinates.
        dtype = tf.keras.mixed_precision.global_policy().compute_dtype

        def tensor(x):
            x = tf.constant(x[None, :-1, :], dtype)
            return tf.repeat(x, repeats=tf.shape(inp_1)[0], axis=0)

        def cen(shape):
            mat = np.eye(num_dim + 1)
            mat[:-1, -1] = -0.5 * (shape - 1)
            return tensor(mat)

        def un_cen(shape):
            mat = np.eye(num_dim + 1)
            mat[:-1, -1] = +0.5 * (shape - 1)
            return tensor(mat)

        def scale(fact):
            mat = np.diag((*[fact] * num_dim, 1))
            return tensor(mat)

        # Detector inputs.
        if half_res:
            prop = dict(fill_value=0, shape=shape_half, shift_center=False)
            inp_1 = layers.SpatialTransformer(**prop)((inp_1, scale(2)))
            inp_2 = layers.SpatialTransformer(**prop)((inp_2, scale(2)))

        # Feature detector: encoder.
        inp = tf.keras.Input(shape=(*inp_1.shape[1:-1], num_chan))
        out = inp
        prop = dict(kernel_size=3, padding='same')
        enc = []
        for n in enc_nf:
            for _ in range(per_level):
                out = conv(n, **prop)(out)
                out = drop(dropout)(out)
                out = KL.LeakyReLU(0.2)(out)
            enc.append(out)
            out = pool(dtype=tf.float32)(out)

        # Recover last layer.
        if dec_nf:
            out = enc.pop()

        # Decoder.
        for n in dec_nf:
            out = KL.concatenate([up()(out), enc.pop()]) # concate firstly
            for _ in range(per_level):
                out = conv(n, **prop)(out)
                out = drop(dropout)(out)
                out = KL.LeakyReLU(0.2)(out)

        # Additional convolutions.
        for n in add_nf:
            out = conv(n, **prop)(out)
            out = drop(dropout)(out)
            out = KL.LeakyReLU(0.2)(out)

        # Output features.
        out = conv(num_feat, activation='relu', **prop)(out)
        det = tf.keras.Model(inp, out)

        # Always sum and fit affine with single precision.
        feat_1 = det(inp_1)
        feat_2 = det(inp_2)
        if tf.keras.mixed_precision.global_policy().compute_dtype == 'float16':
            feat_1 = tf.cast(feat_1, tf.float32)
            feat_2 = tf.cast(feat_2, tf.float32)

        # Barycenters.
        prop = dict(axes=range(1, num_dim + 1), normalize=True, shift_center=True, dtype=dtype)
        cen_1 = ne.utils.barycenter(feat_1, **prop) * shape_full
        cen_2 = ne.utils.barycenter(feat_2, **prop) * shape_full

        # Channel weights.
        axes = range(1, num_dim + 1)
        pow_1 = tf.reduce_sum(feat_1, axis=axes)
        pow_2 = tf.reduce_sum(feat_2, axis=axes)
        pow_1 /= tf.reduce_sum(pow_1, axis=-1, keepdims=True)
        pow_2 /= tf.reduce_sum(pow_2, axis=-1, keepdims=True)
        weights = pow_1 * pow_2

        # Least-squares fit and average, since the fit is not symmetric.
        aff_1 = utils.fit_affine(cen_1, cen_2, weights=weights if weighted else None)
        aff_2 = utils.fit_affine(cen_2, cen_1, weights=weights if weighted else None)
        aff_1 = 0.5 * (utils.invert_affine(aff_2) + aff_1)

        # Remove scaling and shear.
        if rigid:
            aff_1 = utils.affine_matrix_to_params(aff_1)
            aff_1 = aff_1[:, :num_dim * (num_dim + 1) // 2]
            aff_1 = layers.ParamsToAffineMatrix(ndims=num_dim)(aff_1)

        # Mid-space. Before scaling at either side.
        aff_2 = utils.invert_affine(aff_1)
        if return_trans_to_mid_space:
            aff_1 = utils.make_square_affine(aff_1)
            aff_1 = tf.linalg.sqrtm(aff_1)[:, :-1, :]

            aff_2 = utils.make_square_affine(aff_2)
            aff_2 = tf.linalg.sqrtm(aff_2)[:, :-1, :]

        # Affine transform operating in index space, for full-resolution inputs.
        prop = dict(shift_center=False)
        aff_1 = layers.ComposeTransform(**prop)((un_cen(shape_full), aff_1, cen(shape_full)))
        aff_2 = layers.ComposeTransform(**prop)((un_cen(shape_full), aff_2, cen(shape_full)))
        out = [aff_1, aff_2]

        if return_trans_to_half_res:
            out = [(x, scale(2)) for x in out]
            out = [layers.ComposeTransform(shift_center=False)(x) for x in out]

        if tf.keras.mixed_precision.global_policy().compute_dtype == 'float16':
            out = [tf.cast(x, tf.float16) for x in out]

        shape_out = shape_half if return_trans_to_half_res else shape_full
        if make_dense:
            out = [layers.AffineToDenseShift(shape_out, shift_center=False)(x) for x in out]

        # Additional outputs.
        if return_moved:
            prop = dict(shift_center=False, fill_value=0, shape=shape_out)
            mov_1 = layers.SpatialTransformer(**prop)((input_model.inputs[0], aff_1))
            mov_2 = layers.SpatialTransformer(**prop)((input_model.inputs[1], aff_2))
            out.extend([mov_1, mov_2])

        if return_feat:
            out.extend([feat_1, feat_2])

        if not bidir:
            out = out[::2]

        super().__init__(inputs=input_model.inputs, outputs=out if len(out) > 1 else out[0])
