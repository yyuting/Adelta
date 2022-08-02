tf = None
torch = None

import numpy
import numpy as np
import sys
sys.path += ['util']
import argparse_util
import os
import sys
import skimage.io
import skimage
import importlib
import scipy.stats as st
from scipy import signal
import time
import scipy
import scipy.optimize
import copy
import time
import platform
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import scipy.io.wavfile
import np_util
import Halide_lib
import subprocess

#np.random.seed(0)
np.random.seed()

dtype = None

default_width = 960
default_height = 640
default_depth = 640

camera_width = default_width
camea_height = default_height
camera_depth = default_depth

all_modes = ['visualize_gradient', 'render', 'optimization', 'search_init']
all_metrics = ['1_scale_L2']
all_gradient_methods = ['ours', 'finite_diff', 'finite_diff_pixelwise', 'AD']

invalid_op_types = ['Const', 'Cast', 'ExpandDims', 'Roll', 'StridedSlice']

finite_diff_h = 0.005
learning_rate = 0.002

sps = 44100

random_std = 0.3

finite_step = 2

valid_start_idx = 1
valid_end_idx = -1

top_db = 160

tile_offset = [0, 0, 0]

frame_idx = 0

class AdamOptim():
    """
    Code adapted from
    https://towardsdatascience.com/how-to-implement-an-adam-optimizer-from-scratch-76e7b217f1cc
    and
    https://github.com/halide/Halide/blob/master/src/autoschedulers/adams2019/cost_model_generator.cpp#L48
    """
    def __init__(self, eta=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, opt_subset_idx=None):
        self.m = 0
        self.v = 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.eta = eta
        self.t = 1
        self.opt_subset_idx = opt_subset_idx
        
    def reset(self):
        self.m = 0
        self.v = 0
        self.t = 1
    
    def update(self, val, grad):
        ## momentum 
        self.m = self.beta1*self.m + (1-self.beta1)*grad

        ## rms beta 2
        self.v = self.beta2*self.v + (1-self.beta2)*(grad**2)

        ## bias correction
        m_corr = self.m/(1-self.beta1**self.t)
        v_corr = self.v/(1-self.beta2**self.t)

        ## update weights and biases
        step = self.eta*(m_corr / (np.sqrt(v_corr) + self.epsilon))
        #if self.opt_subset_idx is not None:
        if False:
            val[self.opt_subset_idx] -= step[self.opt_subset_idx]
        else:
            val -= step
        
        self.t += 1
        
        return val
    
def get_do_prune(metric, compiler_module, render_kw, min_loss_par):
        
    # prune optional updates
    if 'sigmas' in render_kw.keys():
        del render_kw['sigmas']

    do_prune = np.zeros(compiler_module.n_updates).astype(bool)

    render_kw['do_prune'] = do_prune
    metric.set_x(compiler_module, func_name=None, render_kw=render_kw)
    
    _, global_min_loss_val, _ = metric.run_wrapper(min_loss_par, stage=-1, get_loss=True, get_dL=False, check_last=False, get_deriv=False, base_loss=True, render_kw=render_kw)

    tolerence = 0.01
    
    while np.sum(do_prune) < do_prune.size:

        success = False

        for idx in range(do_prune.size):
            if not do_prune[idx]:
                # if currently NOT pruned, try prune it
                do_prune[idx] = True

                _, pruned_loss, _ = metric.run_wrapper(min_loss_par, stage=-1, get_loss=True, get_dL=False, check_last=False, get_deriv=False, base_loss=True, render_kw=render_kw)

                if pruned_loss > (1 + tolerence) * global_min_loss_val:
                    # reset if loss becomes higher than original
                    do_prune[idx] = False
                else:
                    success = True

        # If no update is pruned, quit
        if not success:
            break

    return do_prune
    
def generate_interactive_frag(args, params, do_prune):
    
    par_file = os.path.join(args.dir, 'glsl_par.npy')
    np.save(par_file, params)
    
    cmd = f"""cd apps; python render_single.py {args.dir} render_{args.shader} --backend glsl --compiler_modes fw --no_compute_g --par_file %s""" % par_file
    
    if do_prune is not None:
        cmd += ' --do_prune %s' % ','.join(['%d' % val for val in do_prune])
        
    print(cmd)
    
    os.system(cmd)
    
    frag_file = os.path.join(args.dir, 'compiler_problem.frag')
    frag_str = open(frag_file).read()
    
    frag_str = f"""
#define width {default_width}.
#define height {default_height}.

    """ + frag_str
    
    open(frag_file, 'w').write(frag_str)

def imsave(name, img, need_transpose=False, ndims=2):
    
    if ndims == 1:
        plt.plot(img)
        plt.savefig(name)
        plt.close()
    elif ndims == 2:
        if need_transpose:
            if len(img.shape) == 2:
                img = img.transpose
            else:
                assert len(img.shape) == 3
                img = img.transpose((1, 0, 2))
        skimage.io.imsave(name, np.clip(img, 0, 1))
    elif ndims == 3:
        #ax = plt.figure().add_subplot(projection='3d')
        #ax.voxels(img[..., 0].astype(bool))
        
        dense_points = np.where(np.sum(img != 0, -1) > 0)
        ax = plt.figure().add_subplot(projection='3d')
        
        if img.shape[-1] == 1:
            ax.scatter(*dense_points, s=5)
        else:
            assert img.shape[-1] == 3
            ax.scatter(*dense_points, c=np.clip(img[dense_points], 0, 1), s=5)
        
        plt.savefig(name, bbox_inches='tight')
        plt.close()

def prepare_uv(width=default_width, height=default_height, depth=default_depth):
    
    assert valid_start_idx > 0
    assert valid_end_idx < 0
    
    extra_size = valid_start_idx - valid_end_idx
    
    xv, yv, zv = np.meshgrid(np.arange(width + extra_size).astype('f') + tile_offset[0] - valid_start_idx, 
                             np.arange(height + extra_size).astype('f') + tile_offset[1] - valid_start_idx,
                             np.arange(width + extra_size).astype('f') + tile_offset[2] - valid_start_idx,
                             indexing='ij')

    u = np.transpose(xv)
    v = np.transpose(yv)
    w = np.transpose(zv)
        
    return u, v, w

def L2(x, y, extra_args={}):
    # TODO: using reduce_sum for easier manual gradient computation
    # when prototyped to use auto diff, can switch back to the more common reduce_mean
    backend = extra_args.get('backend', 'tf')
    if backend == 'tf':
        return tf.reduce_mean((x - y) ** 2, (1, 2, 3))
    else:
        return torch.mean((x - y) ** 2)
        # the scale is only for debug purpose
        # so that the gradient svalue matches that of tf
        #return 0.5 * torch.sum((x - y) ** 2) / 3
        
def L1(x, y, extra_args={}):
    # TODO: using reduce_sum for easier manual gradient computation
    # when prototyped to use auto diff, can switch back to the more common reduce_mean
    backend = extra_args.get('backend', 'tf')
    if backend == 'tf':
        return tf.reduce_mean(tf.abs(x - y), (1, 2, 3))
    else:
        return torch.mean(torch.abs(x - y))
        # the scale is only for debug purpose
        # so that the gradient svalue matches that of tf
        #return 0.5 * torch.sum((x - y) ** 2) / 3

def naive_sum(x, y, extra_args={}, is_tf=True):
    
    backend = extra_args.get('backend', 'tf')
    
    if is_tf or backend == 'tf':
        return tf.reduce_sum(x, (1, 2, 3))
    elif backend == 'torch':
        return torch.sum(x)
    else:
        return np.sum(x, (1, 2, 3))

def gkern(kernlen, std, is_1d=False):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    gkern2d /= np.sum(gkern2d)
    if is_1d:
        return gkern1d / np.sum(gkern1d)
    else:
        return np.expand_dims(np.expand_dims(gkern2d, -1), -1)
    
def replicate_padding(tensor, pad_size):
    """
    tf does not support replicate padding, so has to implement my own
    assuming pad_size is constant, so as to avoid using tf.while
    also assuming pad_size is the same for both x and y dim, as this is all we need to now
    """
    
    remaining_pad = pad_size
    current_pad = 1
    
    while remaining_pad > 0:
        
        if remaining_pad <= current_pad:
            current_pad = remaining_pad
            
        tensor = tf.pad(tensor,[[0, 0], [current_pad, current_pad], [current_pad, current_pad], [0, 0]], "SYMMETRIC" )
        
        remaining_pad -= current_pad
        
        current_pad *= 2
        
    return tensor

def nscale_metric_functor(nscale, base_metric, smoothing_sigmas=None, axis=None, backend='tf', ndims=2, ignore_last_n_scale=0):
    """
    smoothing_sigmas applied to last scale only
    """
    
    def func(x, y, extra_args={}):
        
        loss_seq = extra_args.get('loss_seq', [])
                
        loss = 0
        
        if backend == 'tf':
            expand_dims = tf.expand_dims
        else:
            expand_dims = torch.unsqueeze
        
        if ndims == 1:
            x = expand_dims(x, -1)
            y = expand_dims(y, -1)
        
        nsamples = int(x.shape[0])
        nchannels = int(x.shape[-1])
        
        downsample_scale = 2
        
        if axis == 'x':
            downsample_scale = [1, 2]
        elif axis == 'y':
            downsample_scale = [2, 1]
            
        if ndims == 3:
            assert nsamples == 1
            if backend == 'torch':
                x = torch.permute(x, (4, 0, 1, 2, 3))
                y = torch.permute(y, (4, 0, 1, 2, 3))
                avg_pool = torch.nn.AvgPool3d(downsample_scale, 2)
            else:
                avg_pool = tf.nn.avg_pool3d
        else:
            if backend == 'torch':
                # use C x 1 x w x h
                # batch should always be 1
                # put color in the N position because all color channels are convolved by the same kernel
                x = torch.permute(x, (3, 0, 1, 2))
                y = torch.permute(y, (3, 0, 1, 2))
                assert nsamples == 1
                avg_pool = torch.nn.AvgPool2d(downsample_scale, 2)
            else:
                avg_pool = tf.nn.avg_pool
        
        
            
        current_scale = nscale - 1
        if smoothing_sigmas is not None:
            current_scale += len(smoothing_sigmas)
            len_sigma = len(smoothing_sigmas)
        else:
            len_sigma = 0
        scale = extra_args.get('scale', current_scale)
        
        for i in range(nscale):
            
            if i > 0:
                if backend == 'tf':
                    x = avg_pool(x, downsample_scale, 2, 'VALID')
                    y = avg_pool(y, downsample_scale, 2, 'VALID')
                elif backend == 'torch':
                    x = avg_pool(x)
                    y = avg_pool(y)
                else:
                    raise
                
            if (backend == 'torch' and scale < current_scale) or (current_scale + ignore_last_n_scale >= nscale + len_sigma):
                pass
            else:
                current_loss = base_metric(x, y, {'backend': backend})
                if backend == 'tf':
                    loss_seq.append((current_loss, x, y))
                loss = loss + current_loss
                    
            current_scale -= 1

            if i == nscale - 1 and smoothing_sigmas is not None:
                
                assert ndims == 2

                if backend == 'tf':
                    nh = int(x.shape[1])
                    nw = int(x.shape[2])

                    # combine n and c dimension so that the identical convolution takes on every channel
                    transposed_x = tf.expand_dims(tf.reshape(tf.transpose(x, (0, 3, 1, 2)), (nsamples * nchannels, nh, nw)), -1)
                    transposed_y = tf.expand_dims(tf.reshape(tf.transpose(y, (0, 3, 1, 2)), (nsamples * nchannels, nh, nw)), -1)
                else:
                    transposed_x = x
                    transposed_y = y

                for current_sigma in smoothing_sigmas:
                    
                    assert current_sigma > 0

                    # use a larger cutoff ratio to save computation, can go back to the normal 3sigma cutoff if needed
                    # can use the fact that convolution of Gaussian is still Gaussian to reduce computation, but for now just using the straightforward way to construct Gaussian kernel for each sigma

                    ksize = int(np.ceil(current_sigma)) * 2 + 1

                    kernel = gkern(ksize, current_sigma, is_1d=True)
                    

                    pad_size = (ksize - 1) // 2
                    #pad_x = tf.pad(transposed_x, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]], "SYMMETRIC")
                    #pad_y = tf.pad(transposed_y, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]], "SYMMETRIC")

                    #pad_x = replicate_padding(transposed_x, pad_size)
                    #pad_y = replicate_padding(transposed_y, pad_size)

                    if backend == 'tf':
                        
                        kernel_horizontal = np.expand_dims(np.expand_dims(kernel, -1), -1)
                        kernel_vertical = np.expand_dims(np.expand_dims(np.transpose(kernel), -1), -1)
                        
                        pad_x = tf.pad(transposed_x, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]], "CONSTANT")
                        pad_y = tf.pad(transposed_y, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]], "CONSTANT")

                        if axis != 'x':
                            smoothed_x = tf.nn.conv2d(pad_x, kernel_horizontal, strides=[1, 1, 1, 1], padding='VALID')
                            smoothed_y = tf.nn.conv2d(pad_y, kernel_horizontal, strides=[1, 1, 1, 1], padding='VALID')
                        else:
                            smoothed_x = pad_x
                            smoothed_y = pad_y

                        if axis != 'y':
                            smoothed_x = tf.nn.conv2d(smoothed_x, kernel_vertical, strides=[1, 1, 1, 1], padding='VALID')
                            smoothed_y = tf.nn.conv2d(smoothed_y, kernel_vertical, strides=[1, 1, 1, 1], padding='VALID')

                        smoothed_x = tf.transpose(tf.reshape(tf.squeeze(smoothed_x), (nsamples, nchannels, nh, nw)), (0, 2, 3, 1))
                        smoothed_y = tf.transpose(tf.reshape(tf.squeeze(smoothed_y), (nsamples, nchannels, nh, nw)), (0, 2, 3, 1))
                    elif backend == 'torch':
                        
                        kernel_horizontal = np.expand_dims(np.expand_dims(kernel, 0), 0)
                        kernel_vertical = np.expand_dims(np.expand_dims(np.transpose(kernel), 0), 0)
                        
                        kernel_horizontal = torch.tensor(kernel_horizontal, dtype=torch.float32, device='cuda')
                        kernel_vertical = torch.tensor(kernel_vertical, dtype=torch.float32, device='cuda')
                        
                        pad_x = torch.nn.functional.pad(transposed_x, [pad_size] * 4)
                        pad_y = torch.nn.functional.pad(transposed_y, [pad_size] * 4)

                        assert axis is None
                        smoothed_x = torch.nn.functional.conv2d(pad_x, kernel_horizontal)
                        smoothed_x = torch.nn.functional.conv2d(smoothed_x, kernel_vertical)
                        smoothed_y = torch.nn.functional.conv2d(pad_y, kernel_horizontal)
                        smoothed_y = torch.nn.functional.conv2d(smoothed_y, kernel_vertical)
                       
                    if (backend == 'torch' and scale < current_scale) or (current_scale + ignore_last_n_scale >= nscale):
                        pass
                    else:
                        current_loss = base_metric(smoothed_x, smoothed_y, {'backend': backend})
                        if backend == 'tf':
                            loss_seq.append((current_loss, smoothed_x, smoothed_y))
                        loss = loss + current_loss

                    current_scale -= 1
        return loss
    return func    


metric = naive_sum

def get_err(x, y):
    """
    Compare 2 errors: relative magnitude and normalized l2
    relative magnitude: abs(norm(x) - norm(y)) / norm(y)
    normalized l2: (normalize(x) - normalize(y)) ** 2
    """
    
    x_norm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y)
    
    relative_magnitude = abs(x_norm - y_norm) / (y_norm + 1e-4)
    
    normalized_l2 = np.mean((x / (x_norm + 1e-4) - y / (y_norm + 1e-4)) ** 2)
    
    return relative_magnitude, normalized_l2
    
def assemble_finite_diff_gt(sess=None, loss=None, set_par=None, init_values=None, old_loss_val=None, output=None, finite_diff_both_sides=False, random_dir=None, spsa_samples=-1, halide_fw=None, render_kw={}, args_range=1, sum_col=True):
    
    # deriv wrt NORMALIZED parameters
    
    is_scalar = True
    

    # to save computation, gt deriv is using (f(x + h) - f(x)) / h
    if old_loss_val is None:

        if halide_fw is None:
            set_par(init_values)
            old_loss_val = sess.run(loss)
        else:
            old_loss_val = halide_fw(init_values * args_range, render_kw=render_kw)
            
    get_slice = None

    if isinstance(old_loss_val, (float, np.float32, np.float64)):
        gt_deriv = np.empty(len(init_values))
        get_slice = lambda i: i
    else:
        is_scalar = False
        if halide_fw is None:
            assert isinstance(old_loss_val, np.ndarray) and len(old_loss_val.shape) == 4
            gt_deriv = np.empty((len(init_values), old_loss_val.shape[1], old_loss_val.shape[2], old_loss_val.shape[3]))
            get_slice = lambda i: i
        else:
            if sum_col or old_loss_val.shape == 2:
                gt_deriv = np.empty((old_loss_val.shape[0], old_loss_val.shape[1], len(init_values)))
                old_loss_val = old_loss_val.sum(-1)
            else:
                gt_deriv = np.empty((old_loss_val.shape[0], old_loss_val.shape[1], old_loss_val.shape[2], len(init_values)))
            get_slice = lambda i: ((Ellipsis,) + (i,))

    if random_dir is not None:
        init_values_pos = init_values.copy()
        init_values_pos += random_dir * finite_diff_h

        if halide_fw is None:
            set_par(init_values_pos)

            pos_val = sess.run(loss)
        else:
            pos_val = halide_fw(init_values_pos * args_range, render_kw)
            if sum_col and (not is_scalar):
                pos_val = pos_val.sum(-1)

        if finite_diff_both_sides:
            init_values_neg = init_values.copy()
            init_values_neg -= random_dir * finite_diff_h

            set_par(init_values_neg)

            neg_val = sess.run(loss)

            gt_deriv = (pos_val - neg_val) / (2 * finite_diff_h)
        else:
            gt_deriv = (pos_val - old_loss_val) / finite_diff_h

    elif spsa_samples > 0:
        # using Simultaneous perturbation stochastic approximation
        
        if halide_fw is None:
            assemble_deriv = lambda x, y, z: (x - y) / (finite_diff_h * z)
        else:
            assemble_deriv = lambda x, y, z: np.expand_dims(x - y, -1) / (finite_diff_h * np.expand_dims(z, (0, 1)))
            
        for _ in range(spsa_samples):
            binomial_dir = np.random.binomial(1, 0.5, init_values.shape) * 2 - 1
            
            init_values_pos = init_values + binomial_dir * finite_diff_h
            
            if halide_fw is None:
                set_par(init_values_pos)

                pos_val = sess.run(loss)
            else:
                pos_val = halide_fw(init_values_pos * args_range, render_kw=render_kw)
                if sum_col and (not is_scalar):
                    pos_val = pos_val.sum(-1)
                
            if finite_diff_both_sides:
                init_values_neg = init_values - binomial_dir * finite_diff_h
                
                if halide_fw is None:
                    set_par(init_values_neg)

                    neg_val = sess.run(loss)
                else:
                    neg_val = halide_fw(init_values_neg * args_range, render_kw=render_kw)
                    if sum_col and (not is_scalar):
                        neg_val = neg_val.sum(-1)
                    
                #gt_deriv += (pos_val - neg_val) / (2 * finite_diff_h * binomial_dir)
                gt_deriv += assemble_deriv(pos_val, neg_val, binomial_dir) / 2
            else:
                #gt_deriv += (pos_val - old_loss_val) / (finite_diff_h * binomial_dir)
                gt_deriv += assemble_deriv(pos_val, old_loss_val, binomial_dir)
                
        gt_deriv /= spsa_samples
    else:

        for i in range(len(init_values)):
            init_values_pos = init_values.copy()
            init_values_pos[i] += finite_diff_h

            if halide_fw is None:
                set_par(init_values_pos)

                pos_val = sess.run(loss)
            else:
                pos_val = halide_fw(init_values_pos * args_range, render_kw=render_kw)
                if sum_col and (not is_scalar):
                    pos_val = pos_val.sum(-1)

            if finite_diff_both_sides:
                init_values_neg = init_values.copy()
                init_values_neg[i] -= finite_diff_h

                if halide_fw is None:
                    set_par(init_values_neg)

                    neg_val = sess.run(loss)
                else:
                    neg_val = halide_fw(init_values_neg * args_range, render_kw=render_kw)
                    if sum_col and (not is_scalar):
                        neg_val = neg_val.sum(-1)

                gt_deriv[get_slice(i)] = (pos_val - neg_val) / (2 * finite_diff_h)
            else:
                gt_deriv[get_slice(i)] = (pos_val - old_loss_val) / finite_diff_h
            
    if halide_fw is None:
        set_par(init_values)

    return gt_deriv

def generate_finite_diff_tensor(loss, h, args_range=None, finite_diff_both_sides=False):
    
    #deriv wrt NOT normalized parameters
    # parameters are normalized, therefore the final gradient needs to divide args_range to recover unnormalized gradient
    if finite_diff_both_sides:
        loss_len = int(loss.shape[0])
        assert loss_len % 2 == 0
        nsamples = loss_len // 2
        diff = loss[:nsamples] - loss[nsamples:]
        denom = 2 * h
    else:
        diff = loss[1:] - loss[0]
        denom = h
        
    gt_deriv = diff / denom
        
    if args_range is not None:
        for _ in range(1, len(diff.shape)):
            args_range = np.expand_dims(args_range, -1)
        gt_deriv /= args_range
    
    return gt_deriv

def visualize_gradient(deriv_img, gt_deriv_img, dir, prefix, is_color=False, same_scale=False, gt_nsamples=None, ndims=2):
    
    current_slice = (slice(None),) + (slice(valid_start_idx, valid_end_idx),) * ndims
    
    deriv_img[np.isnan(deriv_img)] = 0
    
    deriv_img = deriv_img[current_slice]
    if ndims == 1:
        deriv_img = np.expand_dims(deriv_img, -1)
    current_deriv_img = np.empty(deriv_img.shape[1:] + (3,))
    #current_deriv_img = np.empty((deriv_img.shape[1], deriv_img.shape[2], 3))
    
    if gt_deriv_img is not None:
        gt_deriv_img[np.isnan(gt_deriv_img)] = 0
        gt_deriv_img = gt_deriv_img[current_slice]
        if ndims == 1:
            gt_deriv_img = np.expand_dims(gt_deriv_img, -1)
        current_gt_deriv_img = np.empty((deriv_img.shape[1], deriv_img.shape[2], 3))
        
    for k in range(deriv_img.shape[0]):

        img_thre_pct = 90
        nc = 1
        
        current_deriv_img[:] = 0
        current_deriv_img[..., 0] = (deriv_img[k] > 0) * deriv_img[k]
        current_deriv_img[..., 2] = (deriv_img[k] < 0) * (-deriv_img[k])

        if gt_deriv_img is not None:
            current_gt_deriv_img[:] = 0
            current_gt_deriv_img[:, :, 0] = (gt_deriv_img[k, :, :] > 0) * gt_deriv_img[k, :, :]
            current_gt_deriv_img[:, :, 2] = (gt_deriv_img[k, :, :] < 0) * (-gt_deriv_img[k, :, :])

        if current_deriv_img.max() > 0:

            nonzero_deriv_img_vals = current_deriv_img[current_deriv_img > 0]
            try:
                deriv_vals_thre = np.percentile(nonzero_deriv_img_vals, img_thre_pct)
            except:
                deriv_vals_thre = 1

            if gt_deriv_img is not None:
                nonzero_gt_deriv_img_vals = current_gt_deriv_img[current_gt_deriv_img > 0]
                try:
                    gt_deriv_vals_thre = np.percentile(nonzero_gt_deriv_img_vals, img_thre_pct)
                except:
                    gt_deriv_vals_thre = 1

                if same_scale:
                    deriv_vals_thre = gt_deriv_vals_thre

            imsave(os.path.join(dir, '%s_deriv_%d.png' % (prefix, k)), np.clip(current_deriv_img / deriv_vals_thre, 0, 1), ndims=max(ndims, 2))

            if gt_deriv_img is not None:
                imsave(os.path.join(dir, '%s_gt_deriv_%d.png' % (prefix, k)), np.clip(current_gt_deriv_img / gt_deriv_vals_thre, 0, 1), ndims=max(ndims, 2))
            
def generate_tensor(init_values, args_range=None, backend='tf', ndims=2):

    if args_range is None:
        args_range = np.ones(init_values.shape)
        
    
    u, v, w = prepare_uv(width=default_width, height=default_height)
    
    if ndims == 1:
        current_slice = (0, 0)
    elif ndims == 2:
        current_slice = (0,)
    elif ndims == 3:
        current_slice = slice(None)
    else:
        raise
        
    u = u[current_slice]
    v = v[current_slice]
    w = w[current_slice]
    
    u = np.expand_dims(u, 0)
    v = np.expand_dims(v, 0)
    w = np.expand_dims(w, 0)

    uv = [u, v, w][:ndims]

        
    var_initializer = init_values / args_range
    
    ones_like_pl = np.ones_like(u)

    if backend == 'tf':
        tunable_params = tf.Variable(var_initializer, dtype=dtype)
    else:
        tunable_params = []
        for i in range(var_initializer.shape[0]):
            tunable_params.append(torch.tensor(var_initializer[i], dtype=torch.float32, requires_grad=True, device='cuda'))
                                  
        for i in range(len(uv)):
            uv[i] = torch.tensor(uv[i]).to(device='cuda')
        
        #u = torch.tensor(u).to(device='cuda')
        #v = torch.tensor(v).to(device='cuda')
        
        ones_like_pl = torch.tensor(ones_like_pl).cuda()

    X_orig = []
    for n in range(var_initializer.shape[0]):
        param = tunable_params[n]
        X_orig.append(param * ones_like_pl * args_range[n])
        
    
        
    def set_par_functor(sess=None):
        
        if sess is not None:
            assign_init_pl = tf.compat.v1.placeholder(dtype, var_initializer.shape[0])
            assign_op = tunable_params.assign(assign_init_pl)

        nonlocal ones_like_pl, init_values

        def f(par, dummy_last_dim=0):

            X_orig = []
            
            if dummy_last_dim > 0:
                par = par[:-dummy_last_dim]
            
            if sess is not None:
                sess.run(assign_op, feed_dict={assign_init_pl: par})
            else:
                if backend == 'tf':
                    tunable_params.assign(par)
            
            for n in range(init_values.shape[0]):
                if backend == 'torch':
                    tunable_params[n].data = torch.tensor(par[n], dtype=torch.float32, device='cuda')
                
                if sess is None:
                    X_orig.append(tunable_params[n] * args_range[n] * ones_like_pl)
            return X_orig
        return f
    
    
    return uv + [X_orig], set_par_functor, tunable_params
    
def estimate_std(sess, loss, set_random_var_opt, nparams, thre=0.01, feed_dict={}, halide_get_loss=None):
    
    
    current_params = np.ones(nparams)
    tmp_params = np.empty(nparams)
    
    last_step = np.ones(nparams)
    changed_dir = np.zeros(nparams).astype(bool)
    
    if halide_get_loss is None:
        set_random_var_opt(sess, current_params)
    
    last_loss = np.zeros(nparams)
    
    # maximum 10 binary search steps
    for _ in range(10):
        for i in range(nparams):
            tmp_params[:] = 0
            tmp_params[i] = current_params[i]
            
            if halide_get_loss is None:
                set_random_var_opt(sess, tmp_params)
            
                current_loss = sess.run(loss, feed_dict=feed_dict)
            else:
                current_loss = halide_get_loss(tmp_params)

            # assuming initial std is the largest allowed
            if current_loss < thre:
                if changed_dir[i]:
                    # if current_loss is smaller than thre, increase std
                    last_step[i] *= 0.5
                    current_params[i] += last_step[i]
            else:
                changed_dir[i] = True
                last_step[i] *= 0.5
                current_params[i] -= last_step[i]
                
            last_loss[i] = current_loss
                
    return current_params

def get_args(str_args=None):
    parser = argparse_util.ArgumentParser(description='approx gradient')
    parser.add_argument('--dir', dest='dir', default='', help='directory for task')
    parser.add_argument('--shader', dest='shader', default='', help='name of shader')
    parser.add_argument('--backend', dest='backend', default='hl', help='specifies the backend to use, can be hl, tf or jnp')
    parser.add_argument('--halide_so_dir', dest='halide_so_dir', default='', help='specifies the location to the so file for Halide binding')
    parser.add_argument('--init_values_pool', dest='init_values_pool', default='', help='file that saves multiple initial values')
    parser.add_argument('--init_values', dest='init_values', default='', help='specifies initial values, seperated by comma')
    parser.add_argument('--debug_mode', dest='debug_mode', action='store_true', help='using debug features')
    parser.add_argument('--modes', dest='modes', default='all', help='choose the modes to run')
    parser.add_argument('--metrics', dest='metrics', default='1_scale_L2', help='choose the metrics')
    parser.add_argument('--gradient_methods_optimization', dest='gradient_methods_optimization', default='ours', help='choose gradient methods used for optimization')
    parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=learning_rate, help='learning rate for optimization')
    parser.add_argument('--finite_diff_h', dest='finite_diff_h', type=float, default=finite_diff_h, help='h used for finite diff')
    parser.add_argument('--finite_diff_both_sides', dest='finite_diff_both_sides', action='store_true', help='if specified, use both side for finite diff (x + h and x - h)')
    parser.add_argument('--random_purturb_zero_gradient', dest='random_purturb_zero_gradient', action='store_true', help='if specified, random purtub a parameter whenever it has zero gradient.')
    parser.add_argument('--smoothing_sigmas', dest='smoothing_sigmas', default='', help='specifies a series of sigma used to smooth the rendering')
    parser.add_argument('--multi_scale_optimization', dest='multi_scale_optimization', action='store_true', help='if using multi scale loss, then also use a multi scale optimization that first only optimizes the coarsest scale loss, then gradually add detilaed loss')
    parser.add_argument('--ignore_last_n_scale', dest='ignore_last_n_scale', type=int, default=0, help='if positive, will ignore the last n scales for optimization, can be used when we know that opt cannot achieve pixel-wise accuracy')
    parser.add_argument('--base_loss_stage', dest='base_loss_stage', type=int, default=-1, help='if specified, and if in multi_scale_optimization mode, use the nth stage as base_loss, if n = -1, use L2 loss')
    parser.add_argument('--opt_subset_idx', dest='opt_subset_idx', default='', help='if specified, only optimize variables indexed by this argument')
    parser.add_argument('--render_size', dest='render_size', default='', help='if specified, overwrite default_width and default_height')
    parser.add_argument('--camera_size', dest='camera_size', default='', help='if specified, will be used to feed to the shader program if the shader needs information about size of camera receptive field')
    parser.add_argument('--multi_scale_previous_loss_ratio', dest='multi_scale_previous_loss_ratio', type=float, default=1, help='specifies at multi scale optimization, what ratio should the previous stage loss take in the current loss')
    parser.add_argument('--profile_timing', dest='profile_timing', action='store_true', help='if specified, verbosely output timing for each step')
    parser.add_argument('--alternating_times', dest='alternating_times', type=int, default=1, help='specified how many times different metrics would alternate')
    parser.add_argument('--niters', dest='niters', type=int, default=2000, help='specifies number of iterations in total per experiment')
    parser.add_argument('--optimizer', dest='optimizer', default='adam', help='specifies which optimizer to use')
    parser.add_argument('--opt_beta1', dest='opt_beta1', type=float, default=0.9, help='specifies beta1 for adam optimizer')
    parser.add_argument('--opt_beta2', dest='opt_beta2', type=float, default=0.999, help='specifies beta2 for adam optimizer')
    parser.add_argument('--mcmc_use_learning_rate', dest='mcmc_use_learning_rate', action='store_true', help='if specified, use learning rate as step size for mcmc method')
    parser.add_argument('--nrestart', dest='nrestart', type=int, default=1, help='specifies the number of restarts for every experiment')
    parser.add_argument('--constrain_valid_pos', dest='constrain_valid_pos', action='store_true', help='if specified, constrain mcmc search space to a manually defined valid posisiton space')
    parser.add_argument('--save_all_loss', dest='save_all_loss', action='store_true', help='if specified, use some extra computation to compute the last scale loss, and save its value for every iteration')
    parser.add_argument('--save_all_par', dest='save_all_par', action='store_true', help='if specified, save tunable parameter for each iteration')
    parser.add_argument('--save_all_deriv', dest='save_all_deriv', action='store_true', help='if specified, save derivatives for every iteration')
    parser.add_argument('--verbose_save', dest='verbose_save', action='store_true', help='if specified, save every possible for each iteration')
    parser.add_argument('--show_progress', dest='show_progress', action='store_true', help='if specified, output stars that indicate optimization progress')
    parser.add_argument('--save_best_par', dest='save_best_par', action='store_true', help='if specified, save tunable parameter for the best optimization')
    parser.add_argument('--loss_filename', dest='loss_filename', default='', help='specifies a unique filename for saved loss')
    parser.add_argument('--is_color', dest='is_color', action='store_true', help='specifies the rendering is color with 3 channel')
    parser.add_argument('--no_col', dest='is_color', action='store_false', help='specifies the rendering is single channel')
    parser.add_argument('--visualize_same_scale', dest='visualize_same_scale', action='store_true', help='if specified, in visualization use same scale to color ours and gt, otherwise use their indivisual scale')
    parser.add_argument('--finite_diff_random_dir', dest='finite_diff_random_dir', action='store_true', help='if specified, only apply finite diff to one random direction, NOT to all parameter directions')
    parser.add_argument('--finite_diff_spsa_samples', dest='finite_diff_spsa_samples', type=int, default=-1, help='if positive, will use the SPSA algorithm to compute stochastic finite diff, and the value is the number of samples drawn per iter')
    parser.add_argument('--no_reset_opt', dest='reset_opt_each_scale', action='store_false', help='if specified, do not reset optimizer after each scale')
    parser.add_argument('--early_termination_ratio', dest='early_termination_ratio', type=float, default=0.25, help='specifies the ratio used for early termination')
    parser.add_argument('--suffix', dest='suffix', default='', help='specifies a suffix name to output')
    parser.add_argument('--gt_file', dest='gt_file', default='', help='if specified, use gt in file instead of generating gt from paraneters')
    parser.add_argument('--gt_transposed', dest='gt_transposed', action='store_true', help='if specified, treat gt as if it is transposed')
    parser.add_argument('--tunable_param_random_var', dest='tunable_param_random_var', action='store_true', help='if specified, add random offset to the tunable parameters whose indices are specified in the shader')
    parser.add_argument('--tunable_param_random_var_opt', dest='tunable_param_random_var_opt', action='store_true', help='if specified, also optimize the std of the random variable')
    parser.add_argument('--tunable_param_random_var_seperate_opt', dest='tunable_param_random_var_seperate_opt', action='store_true', help='if specified, optimize a seperate variable for each random variable')
    parser.add_argument('--tunable_param_random_var_std', dest='tunable_param_random_var_std', type=float, default=0.1, help='specifies the std of the random noise added to the normalized tunable parameter')
    parser.add_argument('--visualize_idx', dest='visualize_idx', type=int, default=0, help='specifies the index in init_values_pool that is used to visualize gradient')
    parser.add_argument('--tunable_param_random_var_opt_scheduling', dest='tunable_param_random_var_opt_scheduling', default='all', help='how we schedule the optimization for random var, currently only supports all')
    parser.add_argument('--estimate_std_thre', dest='estimate_std_thre', type=float, default=1e-3, help='specifies the threshold used to estimate random variable std')
    parser.add_argument('--update_std_schedule', dest='update_std_schedule', default='periodic', help='specifies the schedule to update std of each parameter, support "periodic" and "once"')
    parser.add_argument('--tile_offset', dest='tile_offset', default='', help='if we want to render only a tile, specifies what offset do we want')
    parser.add_argument('--tile_size', dest='tile_size', default='', help='if we want to render only a tile, specifies what tile size we want')
    parser.add_argument('--disable_gt', dest='disable_gt', action='store_true', help='if specified, disable rendering gt gradient in visualize_gradient mode')
    parser.add_argument('--save_npy', dest='save_npy', action='store_true', help='if specified, save the npy file for generated gradient')
    parser.add_argument('--resample_constrain_violation', dest='resample_constrain_violation', action='store_true', help='if specified, resample variables that violates the constrain')
    parser.add_argument('--deriv_n', dest='deriv_n', type=int, default=4, help='specifies how many discrete steps taken for a circular walk')
    parser.add_argument('--deriv_metric_line', dest='deriv_metric_line', action='store_true', help='if specified, run the line metric for deriv in visualize_gradient mode')
    parser.add_argument('--line_endpoints_idx', dest='line_endpoints_idx', default='0,1', help='specifies the 2 endpoints of a line that is used to approximate deriv error')
    parser.add_argument('--line_endpoints_method', dest='line_endpoints_method', default='idx', help='specifies how to decide the 2 endpoints of a line for error metric, support idx (manually specify 2 endpoints in init_values_pool),  random_heuristic (randomly decide enpoints that are outside kernel regions based on heuristic rule), random_smooth (randomly choose line direction and endpoints, rhs of gradient theorem determined by sampling a smoothed function value), custom_par (debug purpose, will customize a specific parameter setting for each pixel), kernel_smooth (smooth both lhs and rhs of the gradient theorem using the same kernel) and follow_grad (following the gradient directino of current parameter set)')
    parser.add_argument('--kernel_type', dest='kernel_type', default='box', help='specifies kernel type to use for kernel_smooth, supports box and gaussian')
    parser.add_argument('--kernel_sigma', dest='kernel_sigma', type=float, default=1.0, help='specifies the kernel sigma used to smooth the gradient theorem')
    parser.add_argument('--kernel_uv_sigma', dest='kernel_uv_sigma', type=float, default=1.0, help='specifies the kernel sigma on uv dimesion used to smooth the gradient theorem')
    parser.add_argument('--kernel_nsamples', dest='kernel_nsamples', type=int, default=1, help='specifies the number of samples used to sample the smooth kernel')
    parser.add_argument('--metric_save_intermediate', dest='metric_save_intermediate', action='store_true', help='if specified, save intermediate sample values during the line integral')
    parser.add_argument('--no_binary_search_std', dest='binary_search_std', action='store_false', help='if specified, do not apply binary search to check best random var std')
    parser.add_argument('--deriv_metric_visualization_thre', dest='deriv_metric_visualization_thre', type=float, default=0, help='if nonzero, use this threshold to scale the visualization')
    parser.add_argument('--deriv_random_dir_idx', dest='deriv_random_dir_idx', default='', help='if specified, only render a subset of the random direction')
    parser.add_argument('--deriv_metric_combine_data', dest='deriv_metric_combine_data', action='store_true', help='if specified, do not generate new data, instead, read old data from file and combine them')
    parser.add_argument('--deriv_metric_read_from_file', dest='deriv_metric_read_from_file', action='store_true', help='if specified, read lhs from file')
    parser.add_argument('--deriv_metric_skip_lhs', dest='deriv_metric_skip_lhs', action='store_true', help='if specified skip computing lhs')
    parser.add_argument('--deriv_metric_rhs_file', dest='deriv_metric_rhs_file', default='', help='specifies a file to directly read rhs from, skip smoothing for rhs')
    parser.add_argument('--deriv_no_rhs', dest='deriv_compute_rhs', action='store_false', help='if specified, do not compute rhs')
    parser.add_argument('--deriv_metric_endpoint_file', dest='deriv_metric_endpoint_file', default='', help='specifies a file to directly read line integral endpoints, skip the random sampling')
    parser.add_argument('--deriv_metric_finite_diff_schedule', dest='deriv_metric_finite_diff_schedule', default='0.001', help='specifies the what step sizes to use for finite diff when evaluating the derivative metrics')
    parser.add_argument('--deriv_metric_no_ours', dest='deriv_metric_use_ours', action='store_false', help='if specified, do not compute ours integral')
    parser.add_argument('--use_select_rule', dest='use_select_rule', type=int, default=1, help='specifies the select rule used when executing gradient program')
    parser.add_argument('--use_multiplication_rule', dest='use_multiplication_rule', type=int, default=1, help='specifies the multiplication rule used when executing gradient program')
    parser.add_argument('--check_intersect_rule', dest='check_intersect_rule', type=int, default=1, help='when using implicit function theorem, specifies the rule used to check whether the sihoulette is tangent or caused by intersection of surface')
    parser.add_argument('--keep_invalid', dest='keep_invalid', action='store_true', help='instructs the gradient program whether to keep the value of invalid gradients (when more than 1 discontinuity present)')
    parser.add_argument('--delta_rule', dest='delta_rule', type=int, default=0, help='specifies the delta rule used when executing gradient program')
    parser.add_argument('--gradient_variant', dest='gradient_variant', type=int, default=0, help='specifies the variant used for the gradient program')
    parser.add_argument('--deriv_metric_curve_type', dest='deriv_metric_curve_type', default='line', help='specifies the curve type allowed for line integral, supports line (straight line) and piecewise (piecewise linear)')
    parser.add_argument('--deriv_metric_max_halflen', dest='deriv_metric_max_halflen', type=float, default=0.1, help='specifies the maximum half length of the line integral')
    parser.add_argument('--use_cpu', dest='use_cpu', action='store_true', help='if specified, disable gpu and use cpu instead')
    parser.add_argument('--deriv_metric_suffix', dest='deriv_metric_suffix', default='', help='specifies suffix to the name of deriv metric files')
    parser.add_argument('--deriv_metric_record_choose_u', dest='deriv_metric_record_choose_u', action='store_true', help='if specified, record whether to choose u or v for current pixel')
    parser.add_argument('--render_shifted', dest='render_shifted', action='store_true', help='if specified, render rhs with shifted sites')
    parser.add_argument('--choose_u_file', dest='choose_u_file', default='', help='specifies the file used to choose 1d axis')
    parser.add_argument('--deriv_metric_record_kernel_sample', dest='deriv_metric_record_kernel_sample', action='store_true', help='if specified, record the samples when computing kernel_smooth')
    parser.add_argument('--kernel_smooth_exclude_our_kernel', dest='kernel_smooth_exclude_our_kernel', action='store_true', help='if specified, assume using box kernel and uv_sigma = 1, and choose_u_file exists, then for our method we only sample one of the uv axis in a 1D box kernel, based on the counterpart of choose_u')
    parser.add_argument('--kernel_smooth_force_our_direction', dest='kernel_smooth_force_our_direction', action='store_true', help='if specified, force our gradient uses a kernel that is always perpendicular to sampling kernel')
    parser.add_argument('--deriv_metric_use_our_kernel', dest='deriv_metric_use_our_kernel', action='store_true', help='if specified, use our kernel (1D box on either u or v) to smooth the evaluation')
    parser.add_argument('--our_filter_direction', dest='our_filter_direction', default='both', help='specifies the filter direction used for our gradient, support u, v, both, or 2d')
    parser.add_argument('--finite_diff_gt_sample', dest='finite_diff_gt_sample', action='store_true', help='if specified, sample uv location when evaluating finite diff gt')
    parser.add_argument('--deriv_metric_postprocess_choose_u', dest='deriv_metric_postprocess_choose_u', action='store_true', help='if specified, will postprocess choose_u')
    parser.add_argument('--deriv_metric_std', dest='deriv_metric_std', action='store_true', help='if specified, compute std across samples as metric')
    parser.add_argument('--deriv_metric_per_sample', dest='deriv_metric_per_sample', action='store_true', help='if specified, instead of computing deriv_metric on the avg of samples, compute a seperate metric per sample')
    parser.add_argument('--quiet', dest='verbose', action='store_false', help='if specified, minimize printout')
    parser.add_argument('--preload_deriv', dest='preload_deriv', action='store_true', help='if specified, in scipy opt mode, precompute deriv in f() and fetch it later in g()')
    parser.add_argument('--random_uniform_uv_offset', dest='random_uniform_uv_offset', type=float, default=0, help='if nonzero, use this value to randomly offset every pixel center')
    parser.add_argument('--no_reset_sigma', dest='reset_sigma', action='store_false', help='if specified, do not reset sigma at the beginning of each traversal')
    parser.add_argument('--autoscheduler', dest='autoscheduler', action='store_true', help='if specified, use autoscheduler to find the best schedule, otherwise, use whatever default in compiler.py')
    parser.add_argument('--ninit_samples', dest='ninit_samples', type=int, default=10, help='specifies the number of samples tried when sampling initial points')
    parser.add_argument('--ninit_best', dest='ninit_best', type=int, default=1, help='specifies the number of top k best samples to keep after sampling')
    parser.add_argument('--weight_map_file', dest='weight_map_file', default='', help='specifies a file that stores weight map applies to dL/dcol')      
    parser.add_argument('--search_type', dest='search_type', type=int, default=-1, help='if nonnegative, input to search_init functions')
    parser.add_argument('--scalar_loss_scale', dest='scalar_loss_scale', default='1', help='specifies the scale for scalar loss (if any)')
    parser.add_argument('--ignore_glsl', dest='ignore_glsl', action='store_true', help='if specified, skip outputting glsl code')
    parser.add_argument('--target_par_file', dest='target_par_file', default='', help='specifies a file storing target parameters')
    parser.add_argument('--target_weight_file', dest='target_weight_file', default='', help='specifies a file storing weights used to regularize parameters')
    parser.add_argument('--target_regularizer_scale', dest='target_regularizer_scale', type=float, default=1e-8, help='specifies the amount enforced to regularizer')
    parser.add_argument('--ignore_scalar_loss', dest='use_scalar_loss', action='store_false', help='if specified, do not use scalar loss')
    parser.add_argument('--refine_opt', dest='refine_opt', action='store_true', help='if specified, add a last stage to optimization that optimizes using the highest resolution loss and without random noise')
    parser.add_argument('--unnormalized_par', dest='unnormalized_par', action='store_true', help='if specified in render mode, assume input parameters are unnormalized')
    parser.add_argument('--aa_nsamples', dest='aa_nsamples', type=int, default=0, help='if specified, render antialiased image')
    parser.add_argument('--shader_args', dest='shader_args', default='', help='specifies arguments that can be set for shader programs, should be the form of name0:val0#name1:val1... values will be evaluated using eval()')
    parser.add_argument('--ndims', dest='ndims', type=int, default=2, help='specifies the dimensionality of the rendering')
    
    parser.set_defaults(is_color=True)
    
    args = parser.parse_args(str_args)
    
    return args

def main(args):
    
    global finite_diff_h
    
    
    
    global tf, torch
    
    if args.backend == 'tf':
        import tensorflow
        tf = tensorflow
        
        global dtype
        dtype = tf.float32
        
    if args.backend == 'torch':
        import torch
    
    if args.use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        
    if args.verbose_save:
        args.save_all_loss = True
        args.save_all_par = True
        args.save_all_deriv = True
    
    global default_width, default_height, default_depth, camera_width, camera_height, default_depth
    if args.render_size != '':
        render_size = args.render_size.split(',')
        
        assert args.ndims == len(render_size)
        assert args.ndims in [1, 2, 3]
        
        default_width = int(render_size[0])
        
        if args.ndims > 1:
            default_height = int(render_size[1])
        else:
            default_height = -1
            
        if args.ndims > 2:
            default_depth = int(render_size[2])
        else:
            default_depth = -1
        
    if args.camera_size != '':
        camera_size = args.camera_size.split(',')
        assert len(camera_size) == args.ndims
        camera_width = int(camera_size[0])
        
        if args.ndims > 1:
            camera_height = int(camera_size[1])
        else:
            camera_height = -1
            
        if args.ndims > 2:
            camera_depth = int(camera_size[2])
        else:
            camera_depth = -1
    else:
        camera_width = default_width
        camera_height = default_height
        camera_depth = default_depth
        
    global tile_offset, tile_size
    
    tile_size = [default_width, default_height, default_depth]
    
    if args.tile_offset != '':
        offset_str = args.tile_offset.split(',')
        if offset_str[0] == '':
            offset_str = offset_str[1:]
        assert len(offset_str) == args.ndims
        for i in range(args.ndims):
            tile_offset[i] = int(offset_str[i])

    if args.tile_size != '':
        size_str = args.tile_size.split(',')
        assert len(size_str) == args.ndims
        for i in range(args.ndims):
            tile_size[i] = int(size_str[i])
            
    current_slice = (slice(None),) + (slice(valid_start_idx, valid_end_idx),) * args.ndims
    
    if not os.path.isdir(args.dir):
        os.mkdir(args.dir)
    
    if args.backend == 'hl':
        lib_extension = subprocess.check_output('python3-config --extension-suffix', shell=True).decode("utf-8")
        if lib_extension.endswith('\n'):
            lib_extension = lib_extension[:-1]
        compiler_problem_full_name = os.path.abspath(os.path.join(args.dir, 'compiler_problem%s' % lib_extension))
    else:
        if args.backend == 'tf':
            shortname = 'compiler_problem'
        elif args.backend == 'torch':
            shortname = 'compiler_problem_torch'
        compiler_problem_full_name = os.path.abspath(os.path.join(args.dir, shortname + '.py'))
        
    # only useful for Halide, determine what kernels needed, saves time to avoid compiling unnecessary kernels
    per_pixel_offset = False
    compiler_modes = ['fw']
    if args.tunable_param_random_var:
        mode_suffix = '_random_par'
    elif args.deriv_metric_line:
        assert not args.tunable_param_random_var
        assert args.modes == 'visualize_gradient'
        mode_suffix = '_per_pixel_offset'
        per_pixel_offset = True
    else:
        mode_suffix = ''
        
    compiler_modes.append('fw%s' % mode_suffix)
    
    if 'visualize_gradient' in args.modes:
        if args.deriv_metric_line:
            #if args.deriv_metric_use_ours:
            #    compiler_modes.append('bw%s' % mode_suffix)
            
            if args.deriv_metric_finite_diff_schedule != '0':
                if args.finite_diff_spsa_samples > 0:
                    compiler_modes.append('FD%s' % mode_suffix)
                    compiler_modes.append('FD_add_to%s' % mode_suffix)
                else:
                    compiler_modes.append('FD_base_only%s' % mode_suffix)
                    
            if args.line_endpoints_method == 'kernel_smooth_debug':
                compiler_modes.append('bw_denum_only')
            if args.deriv_metric_use_ours:
                if args.kernel_smooth_exclude_our_kernel:
                    compiler_modes.append('bw_choose_u_pl')
                else:
                    compiler_modes.append('bw')
        else:
            compiler_modes.append('bw%s' % mode_suffix)
            compiler_modes.append('FD_base_only%s' % mode_suffix)
    elif 'render' in args.modes and args.backend == 'hl':
        compiler_modes.append('fw_prune_updates')
    elif 'search_init' in args.modes:
        # only need fw, which is already included
        pass
    elif 'optimization' in args.modes:
        #if args.gradient_methods_optimization == 'ours':
        if True:
            compiler_modes.append('bw%s' % mode_suffix)
            compiler_modes.append('producer%s' % mode_suffix)
            
        compiler_modes.append('fw_prune_updates')

    mode_str = ','.join(compiler_modes)
        
    extra_args = ''
    if args.autoscheduler:
        extra_args += ' --autoscheduler '
        
    if args.shader_args != '':
        extra_args += ' --shader_args %s ' % args.shader_args
        
    if args.gradient_methods_optimization == 'AD':
        extra_args += ' --AD_only '
        
    extra_args += '--ndims %d ' % args.ndims
            
    generate_code_cmd = 'cd apps; python render_single.py %s render_%s --backend %s --compiler_modes %s --use_select_rule %d --use_multiplication_rule %d %s; cd ..' % (args.dir, args.shader, args.backend, mode_str, args.use_select_rule, args.use_multiplication_rule, extra_args)
    
    print(generate_code_cmd)
    

    def check_modes_exist(compiler_module):
        for mode in compiler_modes:
            if not compiler_module.check_ok(mode):
                return False
        return True
    
    def load_and_check():
        if args.backend == 'hl':
            
            # we need to check whether needed functions are already compiled into the library
            # the check needs to be done in a subprocess
            # because python will NOT reload C modules after recompilation
            check_subprocess_cmd = f"""python Halide_lib.py {args.dir} --modes {mode_str}"""
            check_result = subprocess.check_output(check_subprocess_cmd, shell=True).decode("utf-8")
            
            if not check_result.startswith('success'):
                ans = os.system(generate_code_cmd)
                print(ans)
             
            Halide_lib.load_so(args.dir)
            
            if not (Halide_lib.so_loaded and Halide_lib.compiler_problem_loaded):
                print("Something's wrong when loading Halide_lib, should debug!")
                raise
            
            compiler_module = Halide_lib.CompilerProblem()
                
            assert check_modes_exist(compiler_module)

            return compiler_module
        else:
            spec = importlib.util.spec_from_file_location("module.name", compiler_problem_full_name)
            compiler_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(compiler_module)
            return compiler_module
    
    if not os.path.exists(compiler_problem_full_name):
        ans = os.system(generate_code_cmd)
        print(ans)
        
    compiler_module = load_and_check()
    nargs = compiler_module.nargs
    
    if args.backend == 'torch':
        shader = compiler_module.CompilerProblem.apply
    else:
        shader = None
    
    if not args.use_scalar_loss:
        compiler_module.scalar_loss = None
    
    compiler_module.scalar_loss_scale = [float(val) for val in args.scalar_loss_scale.split(',')]
    
    
    if hasattr(compiler_module, 'args_range'):
        args_range = compiler_module.args_range
    else:
        args_range = None
            
    if args.backend == 'hl':
        bw_map = compiler_module.bw_map
        set_par = lambda x : 0
        set_uv_f = lambda x: 0
    else:
        bw_map = None
    

    finite_diff_h = args.finite_diff_h
    
    if args.init_values_pool != '':
        init_values_pool = np.load(args.init_values_pool)
    else:
        assert args.init_values != ''
        init_values_pool = np.expand_dims(np.array([float(val) for val in args.init_values.split(',')]), 0)
    
    if args.target_par_file != '':
        target_par = np.load(args.target_par_file)[0]
        
        if args.target_weight_file != '':
            target_weight = np.load(args.target_weight_file)
        else:
            target_weight = np.ones(target_par.shape)
            
        def match_target(par):
            
            dif = (par - target_par) * target_weight / args_range
            
            loss = (dif ** 2).sum() * args.target_regularizer_scale
            
            deriv = args.target_regularizer_scale * 2 * dif * target_weight / args_range
            
            return loss, deriv
    else:
        match_target = None
    
    gt_values = init_values_pool[0]
        
    if args.deriv_metric_line and args.line_endpoints_method == 'kernel_smooth_debug':
        assert args.backend == 'hl'


    xv = None
    yv = None

    if args.metrics == 'all':
        metrics = all_metrics
    else:
        metrics = args.metrics.split(',')

    if args.smoothing_sigmas != '':
            
        smoothing_sigmas = args.smoothing_sigmas.split(',')
        smoothing_sigmas = [float(sigma) for sigma in smoothing_sigmas]
    else:
        smoothing_sigmas = None
        
    if args.opt_subset_idx != '':
        opt_subset_idx = args.opt_subset_idx.split(',')
        opt_subset_idx = [int(idx) for idx in opt_subset_idx]
    else:
        opt_subset_idx = None


    global metric
    
    metric_funcs = []
    metric_has_regular = False
            
    for metric_name in metrics:
        
        if '_scale_' in metric_name:

            identifier_ind = metric_name.index('_scale_')
                        
            try:
                nscale = int(metric_name[:identifier_ind])
            except:
                print('multi scale loss need to be in the format n_scale_metric, where n is a positive integer and metric is the name for base metric')
                raise
            
            base_name = metric_name[identifier_ind+len('_scale_'):]
            axis = None
            
            try:
                base_metric = globals()[base_name]
                assert callable(base_metric)
            except:
                try:
                    base_name, axis = base_name.split('_')
                    base_metric = globals()[base_name]
                    assert callable(base_metric)
                    assert axis in ['x', 'y']
                except:
                    print('base metric %s not supported' % metric_name)
                    raise
                    
            if args.ndims == 1:
                axis = 'y'
                
            #if smoothing_sigmas is not None:
            #    assert len(smoothing_sigmas) == nscale
            
            if args.backend == 'hl':
                assert base_name == 'L2'
                assert axis is None
                
                if smoothing_sigmas is None:
                    current_sigmas = []
                else:
                    current_sigmas = smoothing_sigmas
                
                metric = Halide_lib.nscale_L2(nscale=nscale-1, 
                                              smoothing_sigmas=current_sigmas,
                                              multiple_obj=args.multi_scale_optimization,
                                              ignore_last_n_scale=args.ignore_last_n_scale,
                                              opt_subset_idx=opt_subset_idx,
                                              match_target=match_target)
            else:
                metric = nscale_metric_functor(nscale, base_metric, smoothing_sigmas=smoothing_sigmas, axis=axis, backend=args.backend, ndims=args.ndims, ignore_last_n_scale=args.ignore_last_n_scale)
            
            metric_funcs.append(metric)
        
        else:
            try:
                metric = globals()[metric_name]
                assert callable(metric)
            except:
                print('metric %s not supported' % metric_name)
                raise
                
            metric_funcs.append(metric)

        metric_has_regular = True
        
    metric_name = ','.join(metrics)

    if args.modes == 'all':
        modes = all_modes
    else:
        modes = args.modes.split(',')
        for mode in modes:
            if mode not in all_modes:
                print('mode %s not recognized' % mode)
                raise
                
    if args.weight_map_file != '':
        opt_weight_map = np.load(args.weight_map_file)
    else:
        opt_weight_map = None
    
    def auto_fill_render_kw_functor(extra_kw={}):
        def auto_fill_render_kw(render_kw):
            render_kw['width'] = camera_width
            render_kw['height'] = camera_height
            render_kw['render_size'] = [default_width, default_height]
            
            render_kw['tile_offset'] = tile_offset
            
            if args.random_uniform_uv_offset > 0:
                render_kw['uv_offset'] = (2 * np.random.rand(2) - 1) * args.random_uniform_uv_offset
            
            if 'reset_min' not in render_kw.keys():
                render_kw['reset_min'] = False

            global frame_idx
            render_kw['frame_idx'] = frame_idx
            
            for key in extra_kw.keys():
                render_kw[key] = extra_kw[key]
                
            if opt_weight_map is not None:
                render_kw['weight_map'] = opt_weight_map
            
        return auto_fill_render_kw

    for mode in modes:
        if mode in ['visualize_gradient', 'render']:
            
            if mode == 'visualize_gradient':
                need_grad = True
            else:
                need_grad = False
                
            
            halide_fw = None
            halide_FD = None
            
            if mode == 'render':
                compiler_module.normalized_par = not args.unnormalized_par
            
            if args.tunable_param_random_var:

                if hasattr(compiler_module, 'discont_idx'):
                    random_var_indices = compiler_module.discont_idx
                else:
                    random_var_indices = []
                    
                if args.backend == 'hl':
                    compiler_module.sigmas_scale = args.tunable_param_random_var_std

            if args.backend in ['tf', 'torch']:

                vec_output = [None] * 3
                trace = [None] * compiler_module.f_log_intermediate_len
                
                params_orig, set_par_functor, tunable_params = generate_tensor(gt_values, args_range=args_range, backend=args.backend, ndims=args.ndims)
                                
                u = params_orig[0]
                if args.ndims > 1:
                    v = params_orig[1]
                if args.ndims > 2:
                    w = params_orig[2]
                
                    
                X_orig = params_orig[-1]
                
                X = X_orig


                set_random_var = lambda x0, x1: 0
                set_random_var_opt = lambda x0, x1: 0
                random_noise_lookup = {}

                random_var_opt_len = 0

                random_var_scale_opt = None
                random_noise_lookup = {}

                set_par = set_par_functor()
                trace = compiler_module.f(*params_orig[:-1], X_orig, trace, vec_output, camera_width, camera_height, camera_depth)
                   
                vec_output = trace[:3]
                    
                if args.backend == 'tf':
                    if args.is_color:
                        output = tf.stack(vec_output[:3], -1)
                    else:
                        output = tf.expand_dims(vec_output[0], -1)
                else:
                    if args.is_color:
                        output = torch.stack(vec_output[:3], -1)
                    else:
                        output = torch.unsqueeze(vec_output[0], -1)

                output_valid = output[current_slice]
                output_orig = output[0]                

                feed_dict = {}

                extra_args = {}
            elif args.backend == 'hl':
                
                extra_kw = {}
                
                if args.tunable_param_random_var:
                    assert args.tunable_param_random_var_opt and args.tunable_param_random_var_seperate_opt

                    extra_kw = {'sigmas_idx': random_var_indices}
                
                auto_fill_render_kw = auto_fill_render_kw_functor(extra_kw)
                
                def halide_fw(params, render_kw={}):
                    auto_fill_render_kw(render_kw)
                    if compiler_module.normalized_par:
                        if params.size > nargs:
                            render_kw['sigmas'] = params[nargs:]
                            params = params[:nargs] / args_range
                    return compiler_module.forward(params, render_kw=render_kw)
                
                if need_grad:
                    def halide_FD(params, render_kw={}):
                        auto_fill_render_kw(render_kw)
                        return compiler_module.finite_diff(params, 
                                                              render_kw=render_kw)
            else:
                assert args.backend == 'jnp'
                # TODO: finish logic here
                raise
                
            if args.gt_file != '':
                gt_img = skimage.io.imread(args.gt_file)
                gt_img = skimage.img_as_float(gt_img).astype('f')
                if args.gt_transposed:
                    gt_img = gt_img.transpose((1, 0, 2))
            else:
                if args.backend == 'tf':
                    gt_img = output_valid[0].numpy()
                elif args.backend == 'torch':
                    gt_img = output_valid[0].cpu().detach().numpy()
                elif args.backend == 'hl':
                    gt_img = halide_fw(gt_values)
                else:
                    raise
            
            gt_name = os.path.join(args.dir, 'gt.png')
            if os.path.exists(gt_name):
                os.remove(gt_name)

            if args.ndims <= 3:
                imsave(os.path.join(args.dir, 'gt.png'), gt_img, args.backend == 'hl', ndims=args.ndims)
            else:
                raise
            
            binary_conditions = []
            choose_u_ls = []
            choose_u_pl_ls = []

            print('comparison using metric %s' % metric_name)

            halide_get_derivs = None
            
            if need_grad:
                if args.backend != 'hl':

                    # TODO: lift assertion here
                    assert args.backend in ['tf', 'torch']

                    # It's not feasible to compute per pixel gradient map on TF AD
                    # Because TF can only efficiently generate the Jacobian of a single scalr output
                    # NOT the Jacobian of a w x h x 3 image
                    assert args.gradient_methods_optimization != 'AD'
                    
                    dL_dcol_shape = (1,)
                    for i in range(args.ndims):
                        dL_dcol_shape += (int(output.shape[i + 1]),)
                    dL_dcol = np.ones(dL_dcol_shape)
                    
                    if args.backend == 'torch':
                        dL_dcol = torch.tensor(dL_dcol, dtype=torch.float32, device='cuda')
                    
                    if args.is_color:
                        dL_dcol = [dL_dcol] * 3
                    else:
                        dL_dcol = [dL_dcol]
                        
                    param_gradients = compiler_module.g(*params_orig[:-1], X_orig, dL_dcol, trace, [], camera_width, camera_height, camera_depth)


                    if args.backend == 'tf': 
                        param_gradients = np.concatenate([param_gradients[i].numpy() * args_range[i] for i in range(len(param_gradients))], 0)
                        
                        #param_gradients = tf.concat(param_gradients, 0).numpy()
                        #param_gradients *= np.expand_dims(np.expand_dims(args_range, -1), -1) 
                    else:
                        param_gradients = np.concatenate([param_gradients[i].cpu().detach().numpy() * args_range[i] for i in range(len(param_gradients))], 0)

                else:
                    def halide_get_derivs(params, render_kw={}):
                        auto_fill_render_kw(render_kw)
                        return compiler_module.backward(params, render_kw=render_kw)


            magnitude_errors = np.empty(init_values_pool.shape[0])
            normalized_errors = np.empty(init_values_pool.shape[0])
            
            
                
            def clip_img(img, thre):
                return np.clip(img, 0, thre) / thre

            par_shape = init_values_pool.shape[1]
            
            def get_random_sample(mode):
                                                            
                if mode == 'par':
                    sigma = args.kernel_sigma * (args.deriv_metric_max_halflen ** 2 / nargs) ** 0.5
                    sample_size = par_shape
                elif mode == 'uv':
                    sigma = args.kernel_uv_sigma
                    if isinstance(par_shape, tuple):
                        sample_size = (2, par_shape[1], par_shape[2])
                    else:
                        sample_size = 2
                else:
                    raise


                if args.kernel_type == 'box':
                    # box_half_width = (3 ** 0.5) * sigma
                    # This is incorrect, but easy to use in the command line
                    # to match gaussian kernel, should use the equation above that's commented out
                    box_half_width = sigma
                    par_offset_val = np.random.uniform(low=-box_half_width,
                                                       high=box_half_width,
                                                       size=sample_size)
                elif args.kernel_type == 'gaussian':
                    par_offset_val = np.random.normal(loc=0,
                                                      scale=sigma,
                                                      size=sample_size)
                else:
                    raise

                return par_offset_val

            
            if args.deriv_metric_finite_diff_schedule == '0':
                finite_diff_schedule = []
            else:
                finite_diff_schedule = args.deriv_metric_finite_diff_schedule.split(',')
                finite_diff_schedule = [float(val) for val in finite_diff_schedule]
            
            #finite_diff_schedule = [0.001, 0.01, 0.1]
            #finite_diff_schedule = [0.001]
            
            def accum_line_integral(t_schedule, get_p_func, base_name, min_samples=0, sample_spacing=1, sliding_window_size=1, verbose=True, mode='all', sample_f=None, nsamples=1, endpoint_imgs=[]):
                
                global finite_diff_h
                nonlocal choose_u_ls, choose_u_pl_ls
                
                assert args.backend == 'hl'
                
                
                use_ours = True
                use_finite_diff = True
                if mode == 'ours':
                    use_finite_diff = False
                if mode == 'finite_diff':
                    use_ours = False
                    
                if sample_f is not None:
                    assert nsamples >= 1
                    
                    sample_kernel = True
                else:
                    sample_kernel = False
                
                p_start = get_p_func(t_schedule[0])
                p_end = get_p_func(t_schedule[-1])
                
                def get_img(p_update=None, uv_update=None):
                    
                    assert p_update is not None
                    render_kw = {}
                    if uv_update is not None:
                        render_kw['uv_offset'] = uv_update
                    return halide_fw(p_update * args_range, render_kw={})

                
                img_a = get_img(p_start)
                img_b = get_img(p_end)
                    
                    
                if sample_kernel:
                                        
                    if args.deriv_metric_rhs_file == '':
                    
                        img_a = np.zeros(img_a.shape)
                        img_b = np.zeros(img_b.shape)

                        endpoint_imgs.append([img_a])
                        endpoint_imgs.append([img_b])

                        if args.render_shifted:
                                                        
                            # 5 evaluation sites for ours, (center + 4 additional)
                            # 2 * nargs evaluation sites for each of fintie diff
                            nshifted = 4 + 2 * len(finite_diff_schedule) * p_start.shape[0]

                            for _ in range(nshifted):
                                endpoint_imgs[0].append(np.zeros(img_a.shape))
                                endpoint_imgs[1].append(np.zeros(img_a.shape))
                    
                    derivs_rhs = np.zeros(img_a.shape)
                else:
                    derivs_rhs = img_b - img_a
                        

                # DOES not include color channel, because color channels are always summed up in Halide kernels
                if args.deriv_metric_std:
                    derivs_lhs_shape = derivs_rhs.shape[:2] + (p_start.shape[0],)
                else:
                    derivs_lhs_shape = derivs_rhs.shape[:2]

                if args.deriv_metric_per_sample:
                    derivs_lhs_shape = (nsamples,) + derivs_lhs_shape

                derivs_lhs = np.zeros(derivs_lhs_shape)
                if args.deriv_metric_std:
                    derivs_lhs_2nd = np.zeros(derivs_lhs.shape)
                    derivs_lhs_2nd_finite_diff = []
                    
                derivs_lhs_finite_diff = []
                
                
                
                for _ in range(len(finite_diff_schedule)):
                    derivs_lhs_finite_diff.append(np.zeros(derivs_lhs.shape))
                    if args.deriv_metric_std:
                        derivs_lhs_2nd_finite_diff.append(np.zeros(derivs_lhs.shape))
                    
                current_derivs = []
                current_derivs_finite_diff = []
                for _ in range(len(finite_diff_schedule)):
                    current_derivs_finite_diff.append([])
                endpoints = [p_start, p_end]
                
                if sample_kernel and args.deriv_metric_record_kernel_sample:
                    par_ls = np.empty([(len(t_schedule) - 1), 2 + p_start.shape[0], 1, p_start.shape[2]])
                else:
                    par_ls = []
                    
                choose_u = None
                choose_u_val = None

                current_uv_sample = np.empty([2, default_width, default_height])
                    
                if args.deriv_metric_record_choose_u:
                    
                    func_bw_with_u = None

                    choose_u_val = np.empty([(len(t_schedule) - 1), default_height, default_width, 1]).astype(bool)
                    
                if args.choose_u_file != '':
                    choose_u_val = np.load(args.choose_u_file).astype(int)

                current_deriv = np.empty(derivs_lhs.shape + (p_start.shape[0],))
                    
                if use_finite_diff:
                    finite_diff_derivs = []
                    for _ in range(len(finite_diff_schedule)):
                        finite_diff_derivs.append(np.empty(current_deriv.shape))
                
                def process_sample(p_old, p_new, t_idx=None):
                    
                    nonlocal derivs_lhs, derivs_lhs_finite_diff, derivs_lhs_2nd, derivs_lhs_2nd_finite_diff, img_a, img_b
                    
                    nonlocal current_deriv, finite_diff_derivs
                    
                    nonlocal choose_u_val, current_uv_sample
                    
                    nonlocal args_range
                    
                    nonlocal compiler_module
                    
                    global finite_diff_h
                    
                    step = p_new - p_old
                    p_eval = (p_new + p_old) / 2
                    
                    if sample_kernel:
                        nevals = nsamples
                    else:
                        nevals = 1
                        
                    accum_slice = slice(None)
                    img_slice = slice(None)

                        
                    if not args.deriv_metric_per_sample:
                        if nevals > 1:
                            current_deriv[:] = 0
                            if use_finite_diff:
                                for h_idx in range(len(finite_diff_schedule)):
                                    finite_diff_derivs[h_idx][:] = 0
                            
                    current_p_eval = p_eval
                    
                    uv_sample_fd = None
                    uv_sample = None
                    choose_u_pl_val = None
                    
                    feed_dict = {}
                        
                    for sample_idx in range(nevals):
                                                
                        if sample_kernel:

                            if args.choose_u_file == '':
                                render_kw = {'denum_only': True}
                                current_choose_u = halide_get_derivs(p_eval * args_range, render_kw=render_kw)[1]

                            else:
                                current_choose_u = choose_u_val[t_idx]


                            if args.deriv_metric_use_our_kernel:

                                par_sample = 0
                                uv_sample_raw = np.random.rand(current_uv_sample.shape[1], 
                                                               current_uv_sample.shape[2]) * 2 - 1

                                current_uv_sample[:] = 0
                                if args.our_filter_direction == '2d':
                                    current_uv_sample = np.random.rand(2, default_height, default_width) * 2 - 1
                                elif args.our_filter_direction == 'both':


                                    current_uv_sample[0] = uv_sample_raw * current_choose_u
                                    current_uv_sample[1] = uv_sample_raw * (1 - current_choose_u)
                                elif args.our_filter_direction == 'u':
                                    current_uv_sample[0] = uv_sample_raw
                                else:
                                    current_uv_sample[1] = uv_sample_raw

                                uv_sample = current_uv_sample
                                if args.deriv_metric_record_kernel_sample:
                                    par_ls[t_idx, :2] = uv_sample
                            elif args.kernel_smooth_exclude_our_kernel:

                                choose_u_pl_val = current_choose_u

                                assert args.kernel_type == 'box'
                                assert args.kernel_uv_sigma == 1
                                par_sample = sample_f('par')

                                # sample in the perpendicular direction with choose_u
                                uv_sample_raw = np.random.rand(current_uv_sample.shape[1], 
                                                               current_uv_sample.shape[2]) * 2 - 1
                                if len(finite_diff_schedule) > 0:
                                    uv_sample_fd = np.random.rand(2, 
                                                                  current_uv_sample.shape[1], 
                                                                  current_uv_sample.shape[2]) * 2 - 1

                                current_uv_sample[:] = 0
                                current_uv_sample[0] = uv_sample_raw * (1 - current_choose_u[..., 0])
                                current_uv_sample[1] = uv_sample_raw * current_choose_u[..., 0]
                                uv_sample = current_uv_sample

                                if args.deriv_metric_record_kernel_sample:
                                    par_ls[t_idx] = np.concatenate((uv_sample, par_sample[..., 0]), 0)
                            else:
                                raise
                                
                            current_p_eval = p_eval + par_sample

                            if args.deriv_metric_rhs_file == '':
                                # generate rhs using the same samples
                                if not args.render_shifted:
                                    for pos_idx in range(2):
                                        if pos_idx == 0:
                                            render_pos = p_start
                                        else:
                                            render_pos = p_end

                                        
                                        endpoint_imgs[pos_idx][0] += get_img(render_pos + par_sample)
                                else:
                                    # also accumulate rhs here to reuse same samples

                                    par_sample_offset = np.zeros(par_sample.shape)
                                    uv_sample_offset = np.zeros(uv_sample.shape)

                                    for site_idx in range(len(endpoint_imgs[0])):

                                        par_sample_offset[:] = 0
                                        uv_sample_offset[:] = 0

                                        if site_idx == 0:
                                            # center pos
                                            pass
                                        elif site_idx == 1:
                                            # left pixel
                                            uv_sample_offset[0] = -1
                                        elif site_idx == 2:
                                            # right pixel
                                            uv_sample_offset[0] = 1
                                        elif site_idx == 3:
                                            # top pixel
                                            uv_sample_offset[1] = -1
                                        elif site_idx == 4:
                                            # bottom pixel
                                            uv_sample_offset[1] = 1
                                        else:
                                            h_idx = (site_idx - 5) // (2 * p_start.shape[0])
                                            par_idx = ((site_idx - 5) - h_idx * 2 * p_start.shape[0]) // 2
                                            sign_idx = ((site_idx - 5) - h_idx * 2 * p_start.shape[0]) % 2

                                            if sign_idx == 0:
                                                par_sample_offset[par_idx] = finite_diff_schedule[h_idx]
                                            else:
                                                par_sample_offset[par_idx] = -finite_diff_schedule[h_idx]

                                        for pos_idx in range(2):
                                            if pos_idx == 0:
                                                render_pos = p_start
                                            else:
                                                render_pos = p_end

                                            endpoint_imgs[pos_idx][site_idx] += \
                                            get_img(render_pos + par_sample + par_sample_offset,
                                                    uv_sample + uv_sample_offset)

                            
                        get_choose_u = False
                        if use_ours and args.deriv_metric_record_choose_u:
                            get_choose_u = True
                        if args.deriv_metric_record_choose_u and (not use_ours):
                            # TODO: should finish the logic here, Halide kernels are ready, but python binding not ready
                            raise

                        if sample_kernel:
                            render_kw = {'uv_sample': np.transpose(uv_sample, (1, 2, 0)),
                                         'with_denum': get_choose_u}
                        else:
                            render_kw = {}

                        if choose_u_pl_val is not None:
                            render_kw['choose_u_pl'] = choose_u_pl_val

                        if use_ours:
                            ans = halide_get_derivs(current_p_eval * args_range, render_kw=render_kw)

                            if args.deriv_metric_per_sample:
                                current_deriv[sample_idx] = ans[0]
                            else:
                                if nevals > 1:
                                    current_deriv += ans[0]
                                else:
                                    current_deriv = ans[0]

                            if get_choose_u:
                                choose_u_val[t_idx, ..., 0] = ans[1]

                        if use_finite_diff:
                                                        
                            if uv_sample_fd is not None:
                                pass
                            else:
                                uv_sample_fd = uv_sample
                                
                            if uv_sample_fd is not None:
                                render_kw = {'uv_sample': np.transpose(uv_sample_fd, (1, 2, 0))}
                            else:
                                render_kw = {}
                            
                            for h_idx in range(len(finite_diff_schedule)):
                                h = finite_diff_schedule[h_idx]
                                finite_diff_h = h

                                    
                                render_kw['finite_diff_h'] = finite_diff_h
                                render_kw['SPSA_samples'] = args.finite_diff_spsa_samples

                                ans = halide_FD(current_p_eval * args_range, 
                                                render_kw=render_kw)

                                if args.deriv_metric_per_sample:
                                    finite_diff_derivs[h_idx][sample_idx] = ans
                                else:
                                    assert nevals == 1
                                    # This assertion makes sure we don't have to concatenate full FD
                                    finite_diff_derivs[h_idx] = ans

                                    # NO need to scale step based on args_range
                                    # because we can simply assume the finite diff is computed on normalized space
                                    for idx in range(p_old.shape[0]):
                                        if args.finite_diff_spsa_samples > 0:
                                            derivs_lhs_finite_diff[h_idx] += finite_diff_derivs[h_idx][..., idx] * step[idx]
                                        else:
                                            derivs_lhs_finite_diff[h_idx] += finite_diff_derivs[h_idx][idx][..., 0] * step[idx]

                    if nevals > 1 and not args.deriv_metric_per_sample:
                        current_deriv /= nevals

                        for h_idx in range(len(finite_diff_schedule)):
                            finite_diff_derivs[h_idx] /= nevals

                    if args.metric_save_intermediate:
                        current_derivs.append(current_deriv)
                        for h_idx in range(len(finite_diff_schedule)):
                            current_derivs_finite_diff[h_idx].append(finite_diff_derivs[h_idx])

                    

                    current_lhs_finite_diff = []
                    
                    if args.deriv_metric_std:
                        if use_ours:
                            derivs_lhs += current_deriv
                            derivs_lhs_2nd += (current_deriv) ** 2
                        if use_finite_diff:
                            for h_idx in range(len(finite_diff_schedule)):
                                derivs_lhs_finite_diff[h_idx] += finite_diff_derivs[h_idx]
                                derivs_lhs_2nd_finite_diff[h_idx] += (finite_diff_derivs[h_idx]) ** 2
                    else:

                        for idx in range(p_old.shape[0]):
                            if use_ours:
                                derivs_lhs += current_deriv[..., bw_map[idx]] * step[idx] * args_range[idx]
                            if use_finite_diff:
                                # done inside nsamples loop
                                pass

                    return current_deriv, finite_diff_derivs

                err_finite_diff = []
                
                derivs_a = None
                derivs_b = None
                                

                p_old = p_start

                for t_idx in range(len(t_schedule)-1):
                    if verbose: #and t_idx % 100 == 0:
                        print(t_idx)

                    t = t_schedule[t_idx]
                    t_next = t_schedule[t_idx+1]

                    p_new = get_p_func(t_next)

                    ans = process_sample(p_old, p_new, t_idx=t_idx)

                    if t_idx == 0:
                        derivs_a = ans
                    if t_idx == len(t_schedule) - 2:
                        derivs_b = ans

                    p_old = p_new

                derivs_rhs = 0

                err_ours = np.abs(derivs_lhs - derivs_rhs)

                for h_idx in range(len(finite_diff_schedule)):
                    h = finite_diff_schedule[h_idx]
                    current_lhs = derivs_lhs_finite_diff[h_idx]
                    current_err = np.abs(current_lhs - derivs_rhs)
                    err_finite_diff.append(current_err)

                if args.deriv_metric_std:
                    derivs_lhs /= len(t_schedule) - 1
                    derivs_lhs_2nd /= len(t_schedule) - 1
                    
                    derivs_lhs = np.mean(derivs_lhs_2nd, -2, keepdims=True) - np.mean(derivs_lhs, -2, keepdims=True) ** 2
                    derivs_lhs = np.sum(derivs_lhs, 0)
                    for h_idx in range(len(finite_diff_schedule)):
                        
                        derivs_lhs_finite_diff[h_idx] /= len(t_schedule) - 1
                        derivs_lhs_2nd_finite_diff[h_idx] /= len(t_schedule) - 1
                        
                        derivs_lhs_finite_diff[h_idx] = np.mean(derivs_lhs_2nd_finite_diff[h_idx], -2, keepdims=True) - \
                                                        np.mean(derivs_lhs_finite_diff[h_idx], -2, keepdims=True) ** 2
                        derivs_lhs_finite_diff[h_idx] = np.sum(derivs_lhs_finite_diff[h_idx], 0)
                
                finite_diff_h = args.finite_diff_h
                                
                ans = [endpoints,
                       par_ls,
                       derivs_lhs,
                       derivs_lhs_finite_diff,
                       derivs_rhs,
                       err_ours,
                       err_finite_diff,
                       current_derivs,
                       current_derivs_finite_diff,
                       derivs_a,
                       derivs_b,
                       choose_u_val]

                
                return ans
            
            def save_integral_data(base_name, ans):
                
                endpoints, \
                par_ls, \
                derivs_lhs, \
                derivs_lhs_finite_diff, \
                derivs_rhs, \
                err_ours, \
                err_finite_diff, \
                current_derivs, \
                current_derivs_finite_diff, \
                _, _, \
                choose_u \
                = ans
                
                np.save(os.path.join(args.dir, '%s.npy' % base_name), derivs_lhs)
                
                np.save(os.path.join(args.dir, '%s_par.npy' % base_name), par_ls)
                np.save(os.path.join(args.dir, '%s_endpoints.npy' % base_name), endpoints)
                
                
                np.save(os.path.join(args.dir, '%s_err.npy' % base_name), err_ours)
                np.save(os.path.join(args.dir, '%s_rhs.npy' % base_name), derivs_rhs)
                
                if args.metric_save_intermediate:
                    np.save(os.path.join(args.dir, '%s_debug.npy' % base_name), current_derivs)
                    np.save(os.path.join(args.dir, '%s_debug_finite_diff.npy' % base_name), current_derivs_finite_diff)
                    
                if args.deriv_metric_record_choose_u:
                    np.save(os.path.join(args.dir, '%s_choose_u.npy' % base_name), choose_u)
                    
                for h_idx in range(len(finite_diff_schedule)):
                    h = finite_diff_schedule[h_idx]
                    current_lhs = derivs_lhs_finite_diff[h_idx]

                    current_name = '%s_finite_diff_%f' % (base_name, h)
                    np.save(os.path.join(args.dir, '%s.npy' % current_name), current_lhs)
            
            def save_integral_visualization(base_name, err_ours, err_finite_diff):
                    
                np.save(os.path.join(args.dir, '%s_err.npy' % base_name), err_ours)
                for h_idx in range(len(finite_diff_schedule)):
                    np.save(os.path.join(args.dir, '%s_err_finite_diff_%f.npy' % (base_name, finite_diff_schedule[h_idx])),
                            err_finite_diff[h_idx])
                    
                if len(np.squeeze(err_ours).shape) == 1:
                    def plot_err(name, img, thre_err):
                        plt.clf()
                        plt.plot(np.squeeze(img))
                        plt.ylim(0, thre_err)
                        plt.savefig(name)
                else:
                    def plot_err(name, img, thre_err):
                        skimage.io.imsave(name, clip_img(img, thre_err))
                
                if args.deriv_metric_visualization_thre > 0:
                    thre_err = args.deriv_metric_visualization_thre
                                        
                    #skimage.io.imsave(os.path.join(args.dir, '%s_visualize.png' % base_name), clip_img(err_ours, thre_err))
                    plot_err(os.path.join(args.dir, '%s_visualize.png' % base_name), err_ours, thre_err)
                    for h_idx in range(len(finite_diff_schedule)):
                        current_err = err_finite_diff[h_idx]
                        current_name = '%s_visualize_finite_diff_%f' % (base_name, finite_diff_schedule[h_idx])
                        #skimage.io.imsave(os.path.join(args.dir, '%s.png' % current_name), clip_img(current_err, thre_err))
                        plot_err(os.path.join(args.dir, '%s.png' % current_name), current_err, thre_err)
                    
                    print('current thre: ', thre_err)
                    
                else:
                    
                    thre_err = np.percentile(err_ours[err_ours != 0], 99)

                    modify_thre = True

                    while modify_thre:

                        #skimage.io.imsave(os.path.join(args.dir, '%s_visualize.png' % base_name), clip_img(err_ours, thre_err))
                        plot_err(os.path.join(args.dir, '%s_visualize.png' % base_name), err_ours, thre_err)
                        for h_idx in range(len(finite_diff_schedule)):
                            current_err = err_finite_diff[h_idx]
                            current_name = '%s_visualize_finite_diff_%f' % (base_name, finite_diff_schedule[h_idx])
                            #skimage.io.imsave(os.path.join(args.dir, '%s.png' % current_name), clip_img(current_err, thre_err))
                            plot_err(os.path.join(args.dir, '%s.png' % current_name), current_err, thre_err)

                        print('current thre: ', thre_err)

                        try:
                            ans = input('satisfied with the visualization? if yes, type y, otherwise, type in another threshold: ')

                            if ans == 'y':
                                modify_thre = False
                            else:
                                thre_err = float(ans)
                        except:
                            thre_err = False

            def get_endpoints_and_dir():
                endpoints = np.load(args.deriv_metric_endpoint_file)
                
                p_start = endpoints[0]
                p_end = endpoints[1]
                
                random_dir = p_end - p_start
                                
                assert len(random_dir.shape) == 1
                line_len = np.sum(random_dir ** 2) ** 0.5
                np.allclose(line_len, 2 * args.deriv_metric_max_halflen, 1e-2, 1e-2)

                    
                random_dir /= line_len
                    
                return p_start, p_end, random_dir
                
            if args.deriv_metric_use_ours:
                deriv_mode = 'all'
            else:
                deriv_mode = 'finite_diff'

            render_count = 0
                
            for i in range(init_values_pool.shape[0]):
                init_values = init_values_pool[i]

                normalized_init_values = init_values[:nargs] / args_range

                if any(np.isnan(init_values)):
                    assert not need_grad
                    
                    for _ in range(10):
                        imsave(os.path.join(args.dir, 'init%s%05d.png' % (args.suffix, render_count)), img, args.backend == 'hl', ndims=args.ndims)
                        render_count += 1
                else:
                    
                    if args.backend == 'hl':
                        img = halide_fw(init_values)
                        if args.aa_nsamples > 0:
                            img = np.zeros(img.shape)
                            for _ in range(args.aa_nsamples):
                                img += halide_fw(init_values, render_kw={'uv_offset': np.random.rand(2) - 0.5})
                            img /= args.aa_nsamples
                    else:
                        X_orig = set_par(normalized_init_values)
                        trace = compiler_module.f(*params_orig[:-1], X_orig, trace, vec_output, camera_width, camera_height, camera_depth)
                        if args.backend == 'tf':
                            if args.is_color:
                                img = tf.stack(trace[:3], -1)[0].numpy()
                            else:
                                img = tf.expand_dims(trace[0], -1)[0].numpy()
                        elif args.backend == 'torch':
                            if args.is_color:
                                img = torch.stack(trace[:3], -1)[0].cpu().detach().numpy()
                            else:
                                img = torch.unsqueeze(trace[0], -1)[0].cpu().detach().numpy()
                        else:
                            raise

                    imsave(os.path.join(args.dir, 'init%s%05d.png' % (args.suffix, render_count)), img, args.backend == 'hl', ndims=args.ndims)
                    render_count += 1

                if need_grad and i == args.visualize_idx:

                    imsave(os.path.join(args.dir, 'visualize%s.png' % args.suffix), img, args.backend == 'hl', ndims=args.ndims)
                    
                    if args.deriv_metric_line and args.line_endpoints_method == 'kernel_smooth_debug':
                        
                        max_half_len = args.deriv_metric_max_halflen

                        
                        if args.deriv_metric_endpoint_file != '':
                            endpoint_a, endpoint_b, random_dir = get_endpoints_and_dir()
                        else:
                            endpoint_a = normalized_init_values.copy()
                            endpoint_b = normalized_init_values.copy()

                            random_dir = np.random.rand(normalized_init_values.shape[0]) * 2 - 1
                            random_dir = random_dir / (np.sum(random_dir ** 2) ** 0.5)

                            endpoint_a -= random_dir * max_half_len
                            endpoint_b += random_dir * max_half_len

                        line_dir = endpoint_b - endpoint_a
                        
                        
                        
                        def get_p(t):
                            return endpoint_a + t * line_dir
                            
                        t_schedule = np.linspace(0, 1, args.deriv_n)


                        base_name = 'kernel_smooth_metric_debug_%dX%d_len_%f_kernel_%s_sigma_%f_%f%s' % (args.deriv_n, args.kernel_nsamples, max_half_len, args.kernel_type, args.kernel_uv_sigma, args.kernel_sigma, args.deriv_metric_suffix)
                        
                        endpoint_imgs = []
       
                        nsamples = args.kernel_nsamples
                            
                        
                        ans = accum_line_integral(t_schedule, get_p, base_name, sample_f=get_random_sample, nsamples=nsamples, endpoint_imgs=endpoint_imgs, mode=deriv_mode)
                        
                        np.save(os.path.join(args.dir, '%s_endpoint_imgs.npy' % base_name), endpoint_imgs)
                        save_integral_data(base_name, ans)

                        if False:
                            derivs_lhs = ans[2]

                            assert len(finite_diff_schedule) <= 1
                            if len(finite_diff_schedule):
                                derivs_lhs_finite_diff = ans[3][0]
                            else:
                                derivs_lhs_finite_diff = derivs_lhs

                            derivs_rhs = ans[4]

                            #err_ours = ans[5]
                            #err_finite_diff = ans[6]

                            err_ours = derivs_lhs - derivs_rhs
                            err_finite_diff = derivs_lhs_finite_diff - derivs_rhs

                            ans[5] = err_ours
                            ans[6] = err_finite_diff

                            if args.deriv_metric_rhs_file == '':

                                if len(endpoint_imgs[0]) > 1:

                                    derivs_rhs_ours_shifted = np.zeros(derivs_rhs.shape)
                                    derivs_rhs_finite_diff_shifted = np.zeros(derivs_rhs.shape)

                                    for idx in range(5):
                                        derivs_rhs_ours_shifted += endpoint_imgs[1][idx] - endpoint_imgs[0][idx]
                                    derivs_rhs_ours_shifted /= 5
                                    np.save(os.path.join(args.dir, '%s_rhs_ours_shifted.npy' % base_name), derivs_rhs_ours_shifted)

                                    for idx in range(5, 5 + 2 * endpoint_a.shape[0]):
                                        derivs_rhs_finite_diff_shifted += endpoint_imgs[1][idx] - endpoint_imgs[0][idx]
                                    derivs_rhs_finite_diff_shifted /= (2 * endpoint_a.shape[0])
                                    np.save(os.path.join(args.dir, '%s_rhs_finite_diff_shifted.npy' % base_name), derivs_rhs_finite_diff_shifted)


                                    err_ours_shifted = derivs_lhs - derivs_rhs_ours_shifted                            
                                    err_finite_diff_shifted = derivs_lhs_finite_diff - derivs_rhs_finite_diff_shifted

                                    err_variants = 4
                                else:
                                    err_variants = 2
                            else:
                                err_variants = 0

                            logfilename = os.path.join(args.dir, '%s_stat.txt' % base_name)

                            logfile = open(logfilename, 'a+')

                            def reduce_per_loc(op, arr):
 
                                return op(arr)

                            for err_idx in range(err_variants):
                                if err_idx == 0:
                                    err = err_ours
                                    name = 'ours_non_shifted'
                                elif err_idx == 1:
                                    err = err_finite_diff
                                    name = 'finite_diff_non_shifted'
                                elif err_idx == 2:
                                    err = err_ours_shifted
                                    name = 'ours_shifted'
                                elif err_idx == 3:
                                    err = err_finite_diff_shifted
                                    name = 'finite_diff_shifted'

                                mean_err = reduce_per_loc(np.mean, err)
                                std_err = reduce_per_loc(np.std, err)



                                print(name, 'mean: ', mean_err, 'std: ', std_err)
                                print(name, 'mean: ', mean_err, 'std: ', std_err, file=logfile)



                                np.save(os.path.join(args.dir, '%s_err_%s.npy' % (base_name, name)), err)


                            logfile.close()

                            np.save(os.path.join(args.dir, '%s_endpoint_imgs.npy' % base_name), endpoint_imgs)
                            save_integral_data(base_name, ans)
                            save_integral_visualization(base_name, err_ours, [err_finite_diff])  
                        
                        
                    if args.deriv_metric_line and args.line_endpoints_method == 'random_smooth':
                        
                        assert args.backend == 'hl'

                        
                        # intentially set this small length to test whether sampling smoothed rhs workss
                        max_half_len = args.deriv_metric_max_halflen
                        
                        if args.deriv_metric_endpoint_file != '':
                            endpoint_a, endpoint_b, random_dir = get_endpoints_and_dir()
                        else:
                        
                            random_dir = np.random.rand(init_values.shape[0]) * 2 - 1

                            random_dir = random_dir / (random_dir ** 2).sum() ** 0.5

                            endpoint_a = normalized_init_values - random_dir * max_half_len
                            endpoint_b = normalized_init_values + random_dir * max_half_len
                            
                        line_dir = endpoint_b - endpoint_a
                        
                        def get_p(t):
                            return endpoint_a + t * line_dir

                        t_schedule = np.linspace(0, 1, args.deriv_n)

                        base_name = 'random_smooth_metric_%d_len_%f%s' % (args.deriv_n, max_half_len, args.deriv_metric_suffix)

                        if not args.deriv_metric_read_from_file:
                            ans = accum_line_integral(t_schedule, get_p, base_name, mode=deriv_mode)

                            derivs_lhs = ans[2]
                            if len(finite_diff_schedule):
                                derivs_lhs_finite_diff = ans[3][0]
                            else:
                                derivs_lhs_finite_diff = derivs_lhs
                        else:
                            
                            ours_name = 'random_smooth_metric_%d_len_%f_ours_kernel' % (args.deriv_n, max_half_len)
                            finite_diff_name = 'random_smooth_metric_%d_len_%f_finite_diff_kernel' % (args.deriv_n, max_half_len)
                            
                            derivs_lhs = np.load(os.path.join(args.dir, '%s.npy' % ours_name))
                                  
                            if len(finite_diff_schedule):
                                derivs_lhs_finite_diff = np.load(os.path.join(args.dir, '%s_finite_diff_%f.npy' % (finite_diff_name, finite_diff_schedule[0])))
                            else:
                                derivs_lhs_finite_diff = derivs_lhs
                                
                            current_derivs = 0
                            current_derivs_finite_diff = 0
                            
                            ans = [0, 0,
                                   derivs_lhs, 
                                   [derivs_lhs_finite_diff],
                                   0,
                                   0,
                                   0,
                                   current_derivs, current_derivs_finite_diff, 
                                   0, 0]
                            
                        assert len(finite_diff_schedule) <= 1
                        
                        base_name = 'random_smooth_metric_%dX%d_len_%f%s' % (args.deriv_n, args.kernel_nsamples, max_half_len, args.deriv_metric_suffix)
                        
                            
                        
                        nsamples_1d = 10000

                        extra_padding = 0
                        
                        xv, yv = np.meshgrid(np.arange(derivs_lhs.shape[0] + extra_padding),
                                             np.arange(derivs_lhs.shape[1] + extra_padding),
                                             indexing='ij')
                        
                        samples = np.zeros((2, xv.shape[0], xv.shape[1]))
                        
                        if args.deriv_metric_record_kernel_sample:
                            par_ls = np.zeros((args.kernel_nsamples,) + xv.shape)
                        else:
                            par_ls = 0
                        
                        cheat_start = []
                        cheat_end = []
                        
                        rhs_modes = ['ours']
                        
                        img_avgs = []       
                        
                        generate_sample = False
                        
                        if args.deriv_compute_rhs:
                            for endpoint in [endpoint_a, endpoint_b]:

                                samples[:] = 0

                                raw_params = endpoint * args_range

                                if args.our_filter_direction == 'both':
                                    # True -> 0 (u axis)
                                    # False -> 1 (v axis)

                                    render_kw = {'denum_only': True}
                                    axis_idx = 1 - halide_get_derivs(raw_params, render_kw=render_kw)[1].astype(int)
                                else:
                                    axis_idx = 0

                                img_accum = np.zeros(derivs_lhs.shape)

                                for n in range(args.kernel_nsamples):
                                    
                                    
                                    if n % 1000 == 0:
                                        print(n)
                                    
                                    samples[:] = 0

                                    if not generate_sample:
                                        random_sample = np.random.rand(*xv.shape) * 2 - 1
                                        if args.deriv_metric_record_kernel_sample:
                                            par_ls[n] = random_sample
                                    else:
                                        random_sample = par_ls[n]

                                    if args.our_filter_direction == '2d':
                                        samples = np.random.rand(2, xv.shape[0], xv.shape[1]) * 2 - 1
                                    elif args.our_filter_direction == 'both':
                                        samples[axis_idx, xv, yv] = random_sample
                                    elif args.our_filter_direction == 'u':
                                        samples[0] = random_sample
                                    else:
                                        samples[1] = random_sample


                                    params = (endpoint + get_random_sample('par' )) * args_range

                                    render_kw = {'uv_sample': samples.transpose((1, 2, 0))}
                                    img = halide_fw(params, render_kw=render_kw)
                                    if len(img.shape) > 2:
                                        # NOTE: we're computing the metric based on a scalar funciton that sums up all 3 channels
                                        # Ideally, we could compute the metric wrt each channel individually
                                        # but the gradient program will be less efficient that way
                                        img = img.sum(-1)
                                        
                                    img_accum += img

                                img_avg = img_accum / args.kernel_nsamples

                                img_avgs.append(img_avg)

                                if args.deriv_metric_record_kernel_sample:
                                    # in this case, both endpoints use the same sampels, so we could save 2x storage
                                    generate_sample = True
                        else:
                            img_avgs = [0, 0]

                        derivs_rhs = img_avgs[1] - img_avgs[0]
                        np.save(os.path.join(args.dir, '%s_smoothed_endpoints_f.npy' % base_name), img_avgs)
                        
                        err_ours = np.abs(derivs_lhs - derivs_rhs)
                        err_finite_diff = np.abs(derivs_lhs_finite_diff - derivs_rhs)
                        
                        if isinstance(derivs_rhs, (int, float)):
                            valid_x = 0
                            valid_y = 0
                        else:
                            is_valid = False
                            
                            if len(err_ours.shape) > 2:
                                for c in range(err_ours.shape[-1]):
                                    is_valid = np.logical_or(is_valid, np.logical_or(err_ours[..., c] > 0, derivs_rhs[..., c] != 0))
                            else:
                                is_valid = np.logical_or(err_ours > 0, derivs_rhs != 0)

                            valid_inds = np.where(is_valid)
                            valid_x = valid_inds[0]
                            valid_y = valid_inds[1]

                        
                        if args.deriv_metric_record_kernel_sample:
                            valid_samples = par_ls[:, valid_x, valid_y]
                            valid_samples = np.transpose(valid_samples, (1, 0))
                        else:
                            valid_samples = 0
                  
                        u_val = yv + tile_offset[0]
                        v_val = xv + tile_offset[1]
                        valid_u = u_val[valid_x, valid_y]
                        valid_v = v_val[valid_x, valid_y]
                            
                        
                        np.save(os.path.join(args.dir, '%s_valid_pos.npy' % base_name), np.stack([valid_u, valid_v], 0))
                        
                        if args.deriv_metric_record_choose_u:
                            choose_u_val = ans[-1][..., 0]
                            choose_u_val = choose_u_val[:, valid_x, valid_y]
                            ans[-1] = choose_u_val

                        ans[1] = valid_samples
                        ans[4] = derivs_rhs
                        ans[5] = err_ours
                        ans[6] = err_finite_diff

                        save_integral_data(base_name, ans)
                        save_integral_visualization(base_name, err_ours, [err_finite_diff])
                        
                    gt_nsamples = None
                                            
                    fd_nsamples = 1

                    if args.finite_diff_gt_sample and args.kernel_nsamples > 0:
                        fd_nsamples = args.kernel_nsamples
                        gt_nsamples = fd_nsamples

                    # TODO: finish GT logic using FD kernel
                    # no need to implement GT here

                    if args.backend == 'hl':
                        deriv_img = halide_get_derivs(init_values)[0].transpose((2, 1, 0))
                        render_kw = {'finite_diff_h': finite_diff_h}
                        gt_deriv_img = np.stack(halide_FD(init_values, render_kw), 0)[..., 0].transpose((0, 2, 1))
                    elif args.backend in ['tf', 'torch']:
                        deriv_img = param_gradients 
                        gt_deriv_img = None
                    else:
                        raise


                    numpy.save(os.path.join(args.dir, 'gradient_map.npy'), deriv_img)
                        
                    visualize_gradient(deriv_img, gt_deriv_img, args.dir, metric_name, is_color=args.is_color, same_scale=args.visualize_same_scale, gt_nsamples=gt_nsamples, ndims=args.ndims)

                    if args.save_npy:
                        np.save(os.path.join(args.dir, 'ours_deriv.npy'), deriv_img)

            print('mean magnitude error:', np.mean(magnitude_errors))
            print('mean normalized error:', np.mean(normalized_errors))

            do_prune = None
            if args.backend == 'hl' and not args.ignore_glsl:
                
                if compiler_module.n_updates > 0:
                    metric = metric_funcs[0]
                    metric.set_y(gt_img)
                
                    render_kw = {}
                    auto_fill_render_kw(render_kw)
                    
                    if 'sigmas' in render_kw.keys():
                        del render_kw['sigmas']
                        
                    metric.set_x(compiler_module, func_name=None, render_kw=render_kw)
                    
                    do_prune = get_do_prune(metric, compiler_module, render_kw, init_values_pool[0, :args_range.shape[0]] / args_range)

                generate_interactive_frag(args, init_values_pool[0, :args_range.shape[0]], do_prune)

        elif mode == 'search_init':
            assert args.backend == 'hl'
            
            sys.path += ['apps', 'compiler']
            spec = importlib.util.spec_from_file_location("module.name", os.path.join('apps', 'render_%s.py' % args.shader))
            shader_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(shader_module)
            
            if args.shader_args != '':
                shader_args = args.shader_args.split('#')
                
                for line in shader_args:
                    name, val = line.split(':')
                    val = eval(val)
                    setattr(shader_module, name, val)

                if hasattr(shader_module, 'update_args'):
                    shader_module.update_args()
            
            auto_fill_render_kw = auto_fill_render_kw_functor({})
            
            def halide_fw(params, render_kw={}):
                auto_fill_render_kw(render_kw)
                return compiler_module.forward(params, render_kw=render_kw)
            
            assert args.gt_file != ''
            
            gt_img = skimage.io.imread(args.gt_file)
            gt_img = skimage.img_as_float(gt_img).astype('f')
            
            assert len(gt_img.shape) == 3 and gt_img.shape[2] == 3
            
            if args.gt_transposed:
                gt_img = gt_img.transpose((1, 0, 2))
            
            assert len(metric_funcs) == 1
            
            metric = metric_funcs[0]
            metric.set_y(gt_img)
            
            render_kw = {}
            auto_fill_render_kw(render_kw)
            
            metric.set_x(compiler_module, func_name=None, render_kw=render_kw)
            
            #ns = metric.nsteps - 1
            ns = 0
            
            ref_args = init_values_pool[0]
            
            if args.search_type > 0:
                search_type = args.search_type
            else:
                search_type = None
            
            sample_init = shader_module.SampleInit(search_type=search_type)
            
            best_loss = {}
            last_max = 1e8
            last_idx = None
            
            for idx in range(args.ninit_samples):
                sampled_args, extra_args = sample_init.sample(ref_args, width=default_width, height=default_height, idx=idx)
                _, current_loss, _ = metric.run_wrapper(sampled_args, stage=ns, get_loss=True, get_dL=False, check_last=False, get_deriv=False)

                if len(best_loss) < args.ninit_best or current_loss < last_max:
                    
                    best_loss[idx] = (current_loss, sampled_args, extra_args)
                    
                    if len(best_loss) > args.ninit_best:
                        del best_loss[last_idx]
                        
                    last_max_item = max(best_loss.items(), key=lambda val: val[1][0])
                    last_max = last_max_item[1][0]
                    last_idx = last_max_item[0]
                    
            _, ref_loss, _ = metric.run_wrapper(ref_args, stage=ns, get_loss=True, get_dL=False, check_last=False, get_deriv=False)
            
            if ref_loss <= min([val[0] for val in best_loss.values()]):
                print("Ref parameter gives lowest error, no need to sample")
            else:
                final_sampled_args = np.stack([val[1] for val in best_loss.values()], 0)

                np.save(os.path.join(args.dir, 'sampled%s_init_%d_%d.npy' % (args.suffix, args.ninit_samples, args.ninit_best)), final_sampled_args)

                init_count = 0
                for idx in best_loss.keys():
                    img = halide_fw(best_loss[idx][1])
                    extra_args = best_loss[idx][2]
                    imsave(os.path.join(args.dir, 'sampled%s_init%d.png' % (args.suffix, init_count)), img, True, ndims=args.ndims)
                    print('sampled %d extra args:')
                    print(extra_args)
                    init_count += 1

        elif mode == 'optimization':
            
            if args.backend == 'tf':
                tf.compat.v1.disable_eager_execution()
            
            global frame_idx
            
            base_loss = None

            gradient_method = args.gradient_methods_optimization

            logfilename = os.path.join(args.dir, 'log_%s%s%s%s_%s_%s_%.1e%s.txt' % (gradient_method, '' if args.finite_diff_spsa_samples <= 0 else str(args.finite_diff_spsa_samples), '_both_sides' if args.finite_diff_both_sides else '', '_random_dir' if args.finite_diff_random_dir else '', metric_name, args.optimizer, args.learning_rate, args.suffix))
            logfile = open(logfilename, 'a+')

            halide_fw = None

            set_random_var = lambda x0, x1: 0
            set_random_var_opt = lambda x0, x1: 0
            random_noise_lookup = {}

            random_var_opt_len = 0

            random_var_scale_opt_pl = None
            
            par_len = compiler_module.nargs
            orig_par_len = par_len
            
            use_tf = args.backend == 'tf'

            if args.backend != 'hl':

                assert opt_subset_idx is None
                
                
                
                vec_output = [None] * 3
                trace = [None] * compiler_module.f_log_intermediate_len

                params_orig, set_par_functor, tunable_params = generate_tensor(gt_values, args_range=args_range, backend=args.backend, ndims=args.ndims)

                u = params_orig[0]
                if args.ndims > 1:
                    v = params_orig[1]
                if args.ndims > 2:
                    w = params_orig[2]
                    
                X_orig = params_orig[-1]
                
                X = X_orig

            random_var_scale_opt = None
            apply_random = None

            if args.tunable_param_random_var:
                
                if hasattr(compiler_module, 'discont_idx'):
                    random_var_indices = compiler_module.discont_idx
                else:
                    random_var_indices = []

                if args.tunable_param_random_var_opt:
                    if args.tunable_param_random_var_seperate_opt:
                        random_var_opt_len = len(random_var_indices)
                    else:
                        random_var_opt_len = 1

                base_idx_offset = 0
                assert args.tunable_param_random_var_opt

                if use_tf:

                    random_var_scale = tf.Variable(np.ones(len(random_var_indices)), dtype=tf.float32, trainable=False)
                    random_var_scale_pl = tf.compat.v1.placeholder(tf.float32, len(random_var_indices))
                    random_var_scale_assign = tf.compat.v1.assign(random_var_scale, random_var_scale_pl)

                    random_var_switch = tf.Variable(1.0, dtype=tf.float32, trainable=False)
                    random_var_switch_pl = tf.compat.v1.placeholder(tf.float32, ())
                    random_var_switch_assign = tf.compat.v1.assign(random_var_switch, random_var_switch_pl)

                    def func(sess, state):
                        if isinstance(state, bool):
                            if state:
                                sess.run(random_var_switch_assign, feed_dict={random_var_switch_pl: 1.0})
                            else:
                                sess.run(random_var_switch_assign, feed_dict={random_var_switch_pl: 0.0})          
                        else:
                            sess.run(random_var_scale_assign, feed_dict={random_var_scale_pl: state})

                    set_random_var = func

                    if args.tunable_param_random_var_opt:
                        if args.tunable_param_random_var_seperate_opt:
                            random_var_scale_opt = tf.Variable(np.ones(len(random_var_indices)), dtype=tf.float32, trainable=True)
                        else:
                            random_var_scale_opt = tf.Variable(np.ones(1), dtype=tf.float32, trainable=True)

                        random_var_scale_opt_pl = tf.compat.v1.placeholder(tf.float32, random_var_scale_opt.shape)
                        random_var_scale_opt_assign = tf.compat.v1.assign(random_var_scale_opt, random_var_scale_opt_pl)


                        def func2(sess, val):
                            sess.run(random_var_scale_opt_assign, feed_dict={random_var_scale_opt_pl: val})

                        set_random_var_opt = func2
                    else:
                        random_var_scale_opt = 1.0
                elif args.backend == 'torch':
                    random_var_scale = torch.tensor(np.ones(len(random_var_indices)), dtype=torch.float32, device='cuda')
                    random_var_switch = torch.tensor(1, dtype=torch.float32, device='cuda')
                    
                    def func(sess, state):
                        if isinstance(state, bool):
                            if state:
                                random_var_switch.data = torch.tensor(1., dtype=torch.float32, device='cuda')
                            else:
                                random_var_switch.data = torch.tensor(0., dtype=torch.float32, device='cuda')
                        else:
                            random_var_scale.data = torch.tensor(state, dtype=torch.float32, device='cuda')
                    set_random_var = func
                    
                    if args.tunable_param_random_var_opt:
                        if args.tunable_param_random_var_seperate_opt:
                            random_var_scale_opt = []
                            for _ in range(len(random_var_indices)):
                                random_var_scale_opt.append(torch.tensor(1, dtype=torch.float32, requires_grad=True, device='cuda'))
                        else:
                            random_var_scale_opt = torch.tensor(1, dtype=torch.float32, requires_grad=True, device='cuda')
                        
                        def func2(sess, val):
                            for i in range(len(random_var_scale_opt)):
                                random_var_scale_opt[i].data = torch.tensor(val[i], dtype=torch.float32, device='cuda')
                        set_random_var_opt = func2
                    else:
                        random_var_scale_opt = 1.0

            if args.backend == 'hl':

                extra_kw = {}
                orig_par_len = par_len
                
                if opt_subset_idx is not None:
                    mask = np.zeros(compiler_module.sigmas_range.shape)
                    mask[opt_subset_idx] = 1
                    # set random noiset to 0 for all argumentns NOT optimzied
                    compiler_module.sigmas_range *= mask
                    
                    for idx in random_var_indices:
                        if idx in opt_subset_idx:
                            opt_subset_idx.append(random_var_indices.tolist().index(idx) + compiler_module.nargs)
                
                if args.tunable_param_random_var:
                    assert args.tunable_param_random_var_opt and args.tunable_param_random_var_seperate_opt

                    extra_kw = {'sigmas_idx': random_var_indices}

                auto_fill_render_kw = auto_fill_render_kw_functor(extra_kw)

                def halide_fw(params, render_kw={}):
                    auto_fill_render_kw(render_kw)
                    return compiler_module.forward(params, render_kw=render_kw)

                def halide_FD(params, render_kw={}):
                    auto_fill_render_kw(render_kw)
                    return compiler_module.finite_diff(params, render_kw=render_kw)
            else:

                if args.tunable_param_random_var:

                    if args.debug_mode:

                        extra_size = valid_start_idx - valid_end_idx
                        
                        noise_shape = [len(random_var_indices),
                                       default_height + extra_size,
                                       default_width + extra_size,
                                       default_depth + extra_size]

                        noise_shape = noise_shape[:1 + args.ndims]

                        noise_val = np.random.normal(size=noise_shape).astype('f')

                        #noise_val = np.load('/n/fs/scratch/yutingy/debug_noise.npy')
                        #noise_val = np.transpose(noise_val, (2, 1, 0))
                        #noise_val = noise_val[np.asarray(random_var_indices)]


                    if args.backend == 'tf':
                    
                        for i in range(len(random_var_indices)):
                            idx = random_var_indices[i]

                            if args.debug_mode:
                                current_random_val = tf.expand_dims(noise_val[i], 0)
                            else:
                                current_random_val = tf.random.uniform(X[idx].shape) - 0.5

                            random_noise_lookup[idx] = random_var_switch * random_var_scale[i] * args.tunable_param_random_var_std * current_random_val * compiler_module.sigmas_range[idx]

                            if args.tunable_param_random_var_seperate_opt:
                                X[idx] = X[idx] + random_var_scale_opt[i] * random_noise_lookup[idx]
                            else:
                                X[idx] = X[idx] + random_var_scale_opt * random_noise_lookup[idx]
                    else:
                        def apply_random(X_orig):
                            for i in range(len(random_var_indices)):
                                idx = random_var_indices[i]
                                
                                current_random_val = torch.rand(X_orig[idx].shape, dtype=torch.float32, device='cuda') - 0.5
                                
                                current_random_val *= random_var_switch * random_var_scale[i] * args.tunable_param_random_var_std * compiler_module.sigmas_range[idx]
                                
                                if args.tunable_param_random_var_seperate_opt:
                                    current_random_val *= random_var_scale_opt[i]
                                else:
                                    current_random_val *= random_var_scale_opt
                                    
                                X_orig[idx] = X_orig[idx] + current_random_val
                            return X_orig
                else:
                    random_var_scale_opt = None
                    random_noise_lookup = {}



                if use_tf:
                    config = tf.compat.v1.ConfigProto()  
                    config.gpu_options.allow_growth=True  
                    sess = tf.compat.v1.Session(config=config)  

                    #sess = tf.Session()
                    sess.run(tf.compat.v1.local_variables_initializer())
                    sess.run(tf.compat.v1.global_variables_initializer())
                    

                    with tf.name_scope('forward') as scope:
                        trace = compiler_module.f(*params_orig[:-1], X_orig, trace, vec_output, camera_width, camera_height, camera_depth)

                    if args.is_color:
                        output = tf.stack(trace[:3], -1)
                    else:
                        output = tf.expand_dims(trace[0], -1)
                else:
                    
                    #X_orig = []
                    #for _ in range(nargs):
                    #    X_orig.append(torch.tensor(0., dtype=torch.float32, requires_grad=True, device="cuda"))
                    
                    output = shader(*params_orig[:-1], *X_orig, camera_width, camera_height, camera_depth)
                    sess = None
                    

                

                feed_dict = {}
                raw_spec = []
                
                output_valid = output[current_slice]
                    
                raw_output_valid = output_valid
                    
                output_orig = output[0]
                
              
                
                set_random_var(sess, False)

                

            gt_from_file = True

            if args.gt_file == '':
                gt_from_file = False

            if not gt_from_file:
                
                if args.backend == 'tf':
                    gt_img = sess.run(output_valid[0], feed_dict=feed_dict)
                elif args.backend == 'torch':
                    gt_img = output_valid[0].cpu().detach().numpy()
                elif args.backend == 'hl':
                    gt_img = halide_fw(gt_values)
                else:
                    raise
                    
                if args.ndims <= 3:
                    imsave(os.path.join(args.dir, 'gt.png'), gt_img, args.backend == 'hl', ndims=args.ndims)
                else:
                    raise
            else:
                if args.gt_file.endswith('.png'):
                    gt_img = skimage.io.imread(args.gt_file)
                    gt_img = skimage.img_as_float(gt_img).astype('f')
                    gt_name = os.path.join(args.dir, 'gt.png')
                else:
                    assert args.gt_file.endswith('.npy')
                    gt_img = np.load(args.gt_file).astype(np.float32)
                    gt_name = os.path.join(args.dir, 'gt.npy')
                
                if args.ndims == 2 and args.gt_transposed:
                    if args.is_color:
                        assert gt_img.shape[-1] == 3
                        gt_img = gt_img.transpose((1, 0, 2))
                    else:
                        if len(gt_img.shape) == 3:
                            gt_img = gt_img[..., :1]
                        else:
                            assert len(gt_img.shape) == 2
                            gt_img = np.expand_dims(gt_img, -1)
                        gt_img = gt_img.transpose()
                
                if gt_name != args.gt_file:
                    if os.path.exists(gt_name):
                        os.remove(gt_name)
                    os.symlink(os.path.abspath(args.gt_file), os.path.abspath(gt_name))

            if args.backend != 'hl':
                set_random_var(sess, True)

                set_par = set_par_functor(sess)

                loss_seq = []
                loss_seq_names = []
                deriv_loss_uv = None

                x = output_valid
                y = np.expand_dims(gt_img, 0)

                if args.backend == 'torch':
                    y = torch.tensor(y).cuda()

                both_sides_gradients = None

                # pixelwise derivative can be unchanged no matter we're using multi scale optimization or not
                if gradient_method == 'ours':
                    
                    param_gradients = np.zeros(nargs).tolist()

                    if use_tf:
                                                
                        
                       
                        tiled_deriv_with_pad = None

                        


                elif gradient_method == 'AD':
                    pass
                else:
                    assert gradient_method in ['finite_diff'], 'Unknown gradient method'


            for i in range(len(metric_funcs)):
                metric = metric_funcs[i]

                if args.backend == 'hl':
                    metric.set_y(gt_img)

                    render_kw = {}
                    auto_fill_render_kw(render_kw)

                    if random_var_opt_len > 0:
                        render_kw['sigmas'] = np.ones(random_var_opt_len)

                    #if gradient_method in ['ours', 'AD']:
                    if gradient_method != 'finite_diff':
                        metric.set_x(compiler_module, func_name='backward', render_kw=render_kw)
                    else:
                        assert gradient_method == 'finite_diff'
                        render_kw['finite_diff_h'] = finite_diff_h
                        render_kw['SPSA_samples'] = args.finite_diff_spsa_samples
                        metric.set_x(compiler_module, 
                                     func_name='finite_diff', 
                                     render_kw=render_kw)
                elif args.backend == 'tf':
                    current_loss_seq = []

                    in_x = x
                    in_y = y


                    extra_args = {'sess': sess,
                                  'loss_seq': current_loss_seq,
                                  'tunable_params': tunable_params,
                                  'set_par': set_par,
                                 }


                    loss = metric(in_x, in_y, extra_args)
                    
                    

                    if args.multi_scale_optimization and len(current_loss_seq) > 0:
                        loss_seq = loss_seq + current_loss_seq[::-1]
                        loss_seq_names = loss_seq_names + [metrics[i]] * len(current_loss_seq)
                    else:
                        loss_seq.append((loss, x, y))
                        loss_seq_names.append(metrics[i])

            if args.backend == 'hl':
                nsteps = metric.nsteps

                opt = None
                scipy_optimizer = None
                if args.optimizer.startswith('scipy'):
                    scipy_optimizer = args.optimizer.split('.')[-1]
                elif args.optimizer == 'adam':
                    opt = AdamOptim(eta=args.learning_rate, opt_subset_idx=opt_subset_idx, beta1=args.opt_beta1, beta2=args.opt_beta2)

                if args.optimizer == 'mcmc':
                    compiler_module.normalized_par = False
                else:
                    compiler_module.normalized_par = True

            elif args.backend == 'tf':

                

                print('comparison using metric %s' % metric_name)

                true_loss = tf.reduce_mean(loss)

                
                if base_loss is None:
                    base_loss = tf.reduce_mean((x - y) ** 2)


                if (len(metric_funcs) > 1 or args.multi_scale_optimization) and len(loss_seq) > 0:
                    pass
                else:
                    loss_seq = [(loss,)]



                last_loss = None
                deriv_loss_with_respect_to_param = None
                loss_pixelwise_gradient = None
                last_name = None

                scipy_optimizer = None
                opt = None

                if args.optimizer == 'adam':
                    opt = tf.compat.v1.train.AdamOptimizer(learning_rate=args.learning_rate, beta1=args.opt_beta1, beta2=args.opt_beta2)
                elif args.optimizer == 'gradient_descent':
                    opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=args.learning_rate)
                elif args.optimizer.startswith('scipy'):
                    scipy_optimizer = args.optimizer.split('.')[-1]
                elif args.optimizer == 'mcmc':
                    pass
                else:
                    print('Optimizer not recognized')
                    raise

                steps = []
                loss_to_opt = []
                gvs_seq = []
                finite_diff_pls = []

                gradient_maps = []

                for nl in range(len(loss_seq)):

                    current_term = loss_seq[nl][0]
                    current_name = loss_seq_names[nl]

                    if last_loss is not None and last_name == current_name:
                        follow_last = True
                        current_loss = current_term + last_loss * args.multi_scale_previous_loss_ratio
                    else:
                        follow_last = False
                        current_loss = current_term

                    last_loss = current_loss
                    last_name = current_name

                    true_loss = tf.reduce_mean(current_loss)
                    loss_to_opt.append(true_loss)

                    if opt is not None:

                        if gradient_method == 'ours':

                            deriv_loss_with_respect_to_param = 0
                            gradient_map = 0

                            deriv_denum = raw_output_valid
                                                        
                            if follow_last:
                                old_loss_pixelwise_gradient = loss_pixelwise_gradient * args.multi_scale_previous_loss_ratio
                                loss_pixelwise_gradient = tf.gradients(tf.reduce_mean(current_term), deriv_denum)[0]
                            else:
                                loss_pixelwise_gradient = tf.gradients(true_loss, deriv_denum)[0]

                                
                            if follow_last:
                                loss_pixelwise_gradient = loss_pixelwise_gradient + old_loss_pixelwise_gradient

                            loss_pixelwise_gradient_padded = tf.pad(loss_pixelwise_gradient, 
                                                                    [[0, 0]] + [[1, 1]] * args.ndims + [[0, 0]], "CONSTANT")
                                
                            deriv_output_with_respect_to_param = compiler_module.g(*params_orig[:-1], X_orig, 
                                                                                   [loss_pixelwise_gradient_padded[..., i] for i in range(int(loss_pixelwise_gradient_padded.shape[-1]))], 
                                                             trace, param_gradients, camera_width, camera_height, camera_depth)
                                
                            deriv_output_with_respect_to_param = [deriv_output_with_respect_to_param[i] * args_range[i] for i in range(len(deriv_output_with_respect_to_param))]
                            
                            deriv_output_with_respect_to_param = tf.expand_dims(tf.concat(deriv_output_with_respect_to_param, 0), -1)
                            
                            deriv_output_with_respect_to_param = deriv_output_with_respect_to_param[current_slice]
                            #deriv_output_with_respect_to_param *= np.expand_dims(np.expand_dims(np.expand_dims(args_range, -1), -1), -1)        
                            
                            
                            gradient_map = gradient_map + deriv_output_with_respect_to_param
                            
                            last_ndims = tuple(np.arange(1, int(len(gradient_map.shape))).astype(int))
                            deriv_loss_with_respect_to_param = tf.reduce_sum(gradient_map, last_ndims)

                            gradient_maps.append(gradient_map)
                            
                        elif gradient_method == 'AD':
                            deriv_loss_with_respect_to_param = tf.gradients(true_loss, tunable_params)[0]

                        gvs = []

                        if opt_subset_idx is not None:
                            deriv_loss_with_respect_to_param = deriv_loss_with_respect_to_param[opt_subset_idx]
                            tunable_params = tunable_params[opt_subset_idx]

                        if deriv_loss_with_respect_to_param is not None:
                            # For AD it's possible deriv is None becasue AD cannot capture discont
                            gvs.append((deriv_loss_with_respect_to_param, tunable_params))

                        # also optimize for random_var_scale_opt
                        if random_var_opt_len > 0:
                            # because the derivative relies on random noise values,
                            # finite diff will have inconsitency because it needs multiple session runs
                            # and the random noise value could change
                            assert gradient_method in ['ours', 'AD']
                            
                            if gradient_method == 'ours':

                                base_idx_offset = 0

                                if not args.tunable_param_random_var_seperate_opt:
                                    deriv = np.zeros(1)
                                    for idx in random_noise_lookup.keys():
                                        
                                        current_noise = random_noise_lookup[idx][current_slice]
                                        if args.ndims == 2:
                                            current_noise = tf.expand_dims(current_noise, -1)
                                        
                                        deriv = deriv + tf.reduce_sum(gradient_map[idx-base_idx_offset] * current_noise)
                                else:
                                    deriv = []
                                    for i in range(len(random_var_indices)):
                                        idx = random_var_indices[i]
                                        current_noise = random_noise_lookup[idx][current_slice]
                                        if args.ndims == 2:
                                            current_noise = tf.expand_dims(current_noise, -1)
                                        
                                        # needs to divide by args_range because gradient_map is on NORMALIZED scale, and we want it to be un-normalized
                                        deriv.append(tf.reduce_sum(gradient_map[idx-base_idx_offset] * current_noise) / args_range[idx])
                                    deriv = tf.stack(deriv)
                            else:
                                deriv = tf.gradients(true_loss, random_var_scale_opt)[0]

                            if deriv is not None:
                                # For AD it's possible deriv is None becasue AD cannot capture discont
                                gvs.append((deriv, random_var_scale_opt))

                        gvs_seq.append(gvs)



                        step = opt.apply_gradients(gvs)
                        steps.append(step)
                    else:
                        steps.append(None)
                        gvs_seq.append(None)

                nsteps = len(steps)

                sess.run(tf.compat.v1.local_variables_initializer())
                sess.run(tf.compat.v1.global_variables_initializer())

            elif args.backend == 'torch':
                base_loss = lambda x, y: torch.mean((x - y) ** 2)
                possible_nsteps = nscale
                if smoothing_sigmas is not None:
                    possible_nsteps += len(smoothing_sigmas)
                    
                if args.multi_scale_optimization:
                    nsteps = possible_nsteps
                else:
                    nsteps = 1
                
            if use_tf:
                if random_var_opt_len >= 1:
                    all_params = tf.concat((tf.stack(tunable_params), random_var_scale_opt), 0)
                    par_len = int(all_params.shape[0])
                else:
                    all_params = tunable_params
                    par_len = int(tunable_params.shape[0])

                dummy_last_dim = random_var_opt_len
                if args.backend == 'hl':
                    orig_par_len = int(tunable_params.shape[0])
            else:
                if random_var_opt_len >= 1:
                    par_len = orig_par_len + random_var_opt_len
                    dummy_last_dim = random_var_opt_len
                else:
                    par_len = orig_par_len
                    dummy_last_dim = 0

            if gt_from_file:
                n_opts = init_values_pool.shape[0]
                all_loss_base = 0
            else:
                n_opts = init_values_pool.shape[0] - 1
                all_loss_base = 1
            all_loss = np.zeros((n_opts, args.nrestart * args.niters))
            all_loss_idx = 0

            if args.backend == 'hl' and args.optimizer == 'mcmc':
                # Wrape halide_fw such that it still works on non-normalized parameters
                #def halide_fw_wrapper(func):
                #    return lambda x, y={}: func(x / args_range, y)
                #halide_fw = halide_fw_wrapper(halide_fw)

                def get_loss(x):
                    render_kw = {}
                    auto_fill_render_kw(render_kw)
                    _, loss_val, _ = metric.run_wrapper(x, stage=ns, get_loss=True, get_dL=False, check_last=False, get_deriv=False, render_kw=render_kw)
                    return loss_val

            T_opt_start = time.time()
            
            best_loss = 1e8
            best_par = None

            for i in range(init_values_pool.shape[0]):
                if (not gt_from_file) and i == 0:
                    # the first init value is reference pos, no need to optimize
                    continue
                    
                #if i < 4:
                #    continue

                all_loss_idx = 0

                init_values = init_values_pool[i]
                alternating_times = args.alternating_times

                if args_range is not None:
                    if args.backend == 'hl' and args.optimizer == 'mcmc':
                        # in MCMC case, use unnormalized par
                        pass
                    else:
                        init_values = init_values / args_range

                if dummy_last_dim > 0:

                    if args.backend == 'hl':
                        # because we get rid of all the switching / binary search stuff
                        compiler_module.sigmas_scale = args.tunable_param_random_var_std

                    init_values = np.concatenate((init_values, [1] * dummy_last_dim))

                global_min_loss_val = 1e8
                global_min_loss_par = init_values
                global_last_loss_par = init_values
                global_last_loss_val = None
                
                nstages = nsteps * alternating_times
                if args.refine_opt:
                    nstages += 1

                max_iter = args.niters // nstages
                early_termination_iter = max_iter * args.early_termination_ratio

                all_par = np.zeros([0, par_len])
                all_deriv = np.zeros([0, par_len])

                #if args.save_all_par:
                #    assert args.nrestart == 1

                for nr in range(args.nrestart):



                    min_loss_val = 1e8
                    min_loss_par = init_values.copy()
                    last_loss_par = init_values.copy()
                    last_loss_val = None

                    base_min_loss_val = 1e8
                    base_min_loss_par = init_values

                    not_improving_count = 0
                    min_loss_iter = -1

                    if args.backend != 'hl':
                        current_var_scale = np.ones(dummy_last_dim)

                    if use_tf:
                        sess.run(tf.compat.v1.local_variables_initializer())
                        sess.run(tf.compat.v1.global_variables_initializer())
                    elif args.backend == 'torch':
                        if random_var_scale_opt is None:
                            opt = torch.optim.Adam(tunable_params, lr=args.learning_rate, betas=(args.opt_beta1, args.opt_beta2))
                        else:
                            opt = torch.optim.Adam(tunable_params + random_var_scale_opt, lr=args.learning_rate, betas=(args.opt_beta1, args.opt_beta2))
                    else:
                        if opt is not None:
                            opt.reset()

                    for ns_wide in range(nstages):

                        if ns_wide > 0:
                            if min_loss_iter == -1:
                                not_improving_count += 1
                            else:
                                not_improving_count = 0

                        if not_improving_count >= nsteps:
                            break


                        min_loss_iter = -1
                        
                        if args.show_progress:
                            print('*', end='')

                        if args.refine_opt:
                            if ns_wide < nstages - 1:
                                ns = (ns_wide - 1) % nsteps
                            else:
                                ns = nsteps - 1
                                compiler_module.sigmas_scale = 0
                            nalters = (ns_wide - 1) // nsteps
                        else:
                            ns = ns_wide % nsteps
                            nalters = ns_wide // nsteps

                        if args.save_all_par:
                            current_par = np.zeros([max_iter, int(init_values.shape[0])])
                            if ns == 0 and ns_wide != 0:
                                all_par = np.concatenate((all_par, np.nan * np.ones((1, all_par.shape[1]))), 0)
                        else:
                            current_par = None
                            
                        if args.save_all_deriv:
                            current_deriv = np.zeros([max_iter, int(init_values.shape[0])])
                            if ns == 0 and ns_wide != 0:
                                all_deriv = np.concatenate((all_deriv, np.nan * np.ones((1, all_deriv.shape[1]))), 0)
                        else:
                            current_deriv = None

                        # use a different idx than all loss because we save a seperate npy file for each restart, while when saving all loss, we combine all the restarts
                        current_par_idx = 0

                        feed_dict = {}
                        if args.debug_mode and args.halide_so_dir != '':
                            feed_dict[noise_val] = get_noise(last_loss_par[1:-dummy_last_dim])

                        # reset optimization params
                        if args.reset_opt_each_scale and ns == 0:
                            if use_tf:
                                sess.run(tf.local_variables_initializer())
                                sess.run(tf.global_variables_initializer())
                            elif args.backend == 'torch':
                                if random_var_scale_opt is None:
                                    opt = torch.optim.Adam(tunable_params, lr=args.learning_rate, betas=(args.opt_beta1, args.opt_beta2))
                                else:
                                    opt = torch.optim.Adam(tunable_params + random_var_scale_opt, lr=args.learning_rate, betas=(args.opt_beta1, args.opt_beta2))
                            else:
                                if opt is not None:
                                    opt.reset() 
                        
                        if args.backend == 'hl':
                            if ns_wide == 0:
                                init_img = halide_fw(init_values)
                                imsave(os.path.join(args.dir, 'init%d.png' % i), np.clip(init_img, 0, 1), True, ndims=args.ndims)
                                if compiler_module.sigmas_scale > 0:
                                    old_sigmas_scale = compiler_module.sigmas_scale
                                    compiler_module.sigmas_scale = 0
                                    init_img = halide_fw(init_values)
                                    imsave(os.path.join(args.dir, 'init_no_random%d.png' % i), np.clip(init_img, 0, 1), True, ndims=args.ndims)
                                    compiler_module.sigmas_scale = old_sigmas_scale
                            else:
                                last_loss_par = np.array(min_loss_par)
                                
                            if ns == 0 and dummy_last_dim > 0 and args.reset_sigma:
                                default_sigmas = np.ones(dummy_last_dim)
                                min_loss_par[-dummy_last_dim:] = default_sigmas
                                
                            if args.optimizer == 'mcmc':
                                min_loss_val = get_loss(min_loss_par)
                            else:
                                auto_fill_render_kw(render_kw)
                                next_deriv, min_loss_val, _ = metric.run_wrapper(min_loss_par, stage=ns, get_loss=True, render_kw=render_kw)

                            need_new_deriv = False

                        elif args.backend == 'torch':
                            
                            if ns_wide == 0:
                                X_orig = set_par(init_values)
                                if apply_random is not None:
                                    X_orig = apply_random(X_orig)
                                init_img = shader(*params_orig[:-1], *X_orig, camera_width, camera_height, camera_depth)[0].cpu().detach().numpy()
                                imsave(os.path.join(args.dir, 'init%d.png' % i), init_img, ndims=args.ndims)  
                            else:
                                last_loss_par = np.array(min_loss_par)
                                
                            assert args.optimizer == 'adam'

                            if ns == 0 and dummy_last_dim > 0 and args.reset_sigma:
                                default_sigmas = np.ones(dummy_last_dim)
                                min_loss_par[-dummy_last_dim:] = default_sigmas
                                
                            X_orig = set_par(min_loss_par)
                            set_random_var(sess, current_var_scale)
                            set_random_var_opt(sess, min_loss_par[-dummy_last_dim:])
                            
                            X_orig = set_par(tunable_params, False)
                            if apply_random is not None:
                                X_orig = apply_random(X_orig)
                            x = shader(*params_orig[:-1], *X_orig, camera_width, camera_height, camera_depth)[current_slice]
                            loss = metric_funcs[0](x, y, {'scale': ns})
                            
                            min_loss_val = float(loss)

                        elif args.backend == 'tf':
                            step = steps[ns]
                            true_loss = loss_to_opt[ns]
                            gvs = gvs_seq[ns]

                            set_par(min_loss_par, dummy_last_dim)
                            
                            if ns == 0 and dummy_last_dim > 0 and args.reset_sigma:
                                default_sigmas = np.ones(dummy_last_dim)
                                min_loss_par[-dummy_last_dim:] = default_sigmas

                            set_random_var(sess, current_var_scale)

                            #sess.run(assign_ops, feed_dict={assign_init_pl: min_loss_par})
                            set_random_var_opt(sess, min_loss_par[-dummy_last_dim:])

                            feed_dict = {}

                            if args.debug_mode and args.halide_so_dir != '':
                                feed_dict[noise_val] = get_noise(init_values[1:-dummy_last_dim])
                                
                            if ns_wide == 0:
                                set_par(init_values, dummy_last_dim)
                                init_img = sess.run(output_valid, feed_dict=feed_dict)[0]
                                imsave(os.path.join(args.dir, 'init%d.png' % i), init_img, ndims=args.ndims)  
                            else:
                                last_loss_par = np.array(min_loss_par)


                            min_loss_val = sess.run(true_loss, feed_dict=feed_dict)

                        last_loss_val = min_loss_val

                        if opt is None:

                            if scipy_optimizer is not None:

                                # TODO: finish logic when tunable_param_random_var if True

                                scipy_opt_iter_count = 0
                                iters_left = max_iter

                                opt_order = None
                                if scipy_optimizer in ['Nelder-Mead', 'Powell']:
                                    opt_order = 0
                                elif scipy_optimizer in ['CG', 'BFGS', 'L-BFGS-B']:
                                    opt_order = 1
                                else:
                                    raise 'Unrecognized optimizer method %s' % scipy_optimizer

                                jac_to_opt = None
                                callback = None

                                if args.backend == 'hl':
                                    def func_to_opt(x):
                                        render_kw = {}
                                        auto_fill_render_kw(render_kw)
                                        if opt_order == 0:
                                            _, loss_val, _ = metric.run_wrapper(x, stage=ns, get_loss=True, get_dL=False, check_last=False, get_deriv=False, render_kw=render_kw)
                                        elif opt_order == 1:
                                            _, loss_val, _ = metric.run_wrapper(x, stage=ns, get_loss=True, get_dL=False, check_last=args.preload_deriv, get_deriv=args.preload_deriv, render_kw=render_kw)
                                        else:
                                            raise

                                        return loss_val

                                    def callback(x):

                                        render_kw = {}
                                        auto_fill_render_kw(render_kw)

                                        if args.save_all_loss:
                                            _, ans_base, _ = metric.run_wrapper(x, get_loss=True, get_dL=False, check_last=True, get_deriv=False, base_loss=True, render_kw=render_kw)
                                            nonlocal all_loss_idx
                                            if all_loss_idx < all_loss.shape[1]:
                                                all_loss[i-all_loss_base, all_loss_idx] = ans_base
                                                all_loss_idx += 1
                                        else:
                                            ans_base = None

                                        global frame_idx
                                        frame_idx += 1

                                        nonlocal current_iter
                                        if args.verbose:
                                            print(current_iter, ans_base)
                                        current_iter += 1

                                    if opt_order == 1:
                                        def jac_to_opt(x):

                                            render_kw = {}
                                            auto_fill_render_kw(render_kw)

                                            deriv, _, _ = metric.run_wrapper(x, stage=ns, get_loss=True, get_dL=False, check_last=args.preload_deriv, get_deriv=True, render_kw=render_kw)
                                            if gradient_method == 'ours':
                                                return deriv.astype('float64')
                                            else:
                                                return deriv.astype('float64')


                                else:
                                    def func_to_opt(x):
                                        sess.run(assign_ops, feed_dict={assign_init_pl: x})

                                        if args.save_all_loss:
                                            ans, ans_base = sess.run([true_loss, base_loss])
                                            nonlocal all_loss_idx
                                            if all_loss_idx < all_loss.shape[1]:
                                                all_loss[i-all_loss_base, all_loss_idx] = ans_base
                                                all_loss_idx += 1
                                        else:
                                            ans = sess.run(true_loss)

                                        nonlocal scipy_opt_iter_count
                                        print(i, iters_left, ns_wide, scipy_opt_iter_count, ans)
                                        scipy_opt_iter_count += 1

                                        return ans


                                last_loss_par = min_loss_par

                                while iters_left > 0:
                                    options = {'maxiter': iters_left}

                                    if args.backend == 'hl':
                                        metric.updated = False

                                    current_iter = 0

                                    ans = scipy.optimize.minimize(func_to_opt, last_loss_par, method=scipy_optimizer, options=options, jac=jac_to_opt, callback=callback)

                                    if ans.fun < min_loss_val:
                                        min_loss_val = ans.fun
                                        min_loss_iter = ans.nit
                                        min_loss_par = ans.x

                                    if args.backend == 'hl':
                                        iters_left = 0
                                    else:
                                        iters_left = max_iter - scipy_opt_iter_count
                                        last_loss_par = (numpy.random.rand(int(tunable_params.shape[0])) * (sample_range[:, 1] - sample_range[:, 0]) + sample_range[:, 0]) / args_range



                            elif args.optimizer == 'mcmc':

                                # TODO: finish logic when tunable_param_random_var if True

                                assert sample_range is not None
                                par_val = min_loss_par

                                D = default_width * default_height

                                # sigma2 is normalized using D because our L2 loss is computed over the mean, not sum (which is used in the paper)
                                accept_sigma2 = 0.22 * (D ** 0.5) / D

                                if '_scale_' in metric_name:
                                    assert not args.multi_scale_optimization

                                    if smoothing_sigmas is not None:
                                        accept_sigma2 *= len(smoothing_sigmas) + nscale
                                    else:
                                        accept_sigma2 *= nscale

                                elif metric_name != 'L2':
                                    print('unsupported metric for mcmc')
                                    raise


                                if args.backend == 'hl':
                                    def get_loss(x):

                                        render_kw = {}
                                        auto_fill_render_kw(render_kw)

                                        _, loss_val, _ = metric.run_wrapper(x, stage=ns, get_loss=True, get_dL=False, check_last=False, get_deriv=False, render_kw=render_kw)
                                        return loss_val

                                    old_loss_val = get_loss(min_loss_par)
                                else:
                                    old_loss_val = sess.run(true_loss)

                                #assert valid_pos_func(last_loss_par * args_range, default_width, default_height)

                                for k in range(max_iter):
                                    # randomly select a parameter to resample
                                    perm_idx = np.random.choice(par_len)
                                    par_val = last_loss_par.copy()

                                    if args.backend == 'hl':
                                        par_val[perm_idx] = np.random.rand() * (sample_range[perm_idx, 1] - sample_range[perm_idx, 0]) + sample_range[perm_idx, 0]
                                        loss_val = get_loss(par_val)

                                        if args.save_all_loss:
                                            auto_fill_render_kw(render_kw)
                                            _, ans_base, _ = metric.run_wrapper(par_val, get_loss=True, get_dL=False, check_last=True, get_deriv=False, base_loss=True, render_kw=render_kw)
                                            all_loss[i-all_loss_base, all_loss_idx] = ans_base
                                            all_loss_idx += 1
                                    else:
                                        def update_par_val():

                                            if args.mcmc_use_learning_rate:
                                                par_val[perm_idx] += numpy.random.rand() * args.learning_rate
                                                par_val[perm_idx] = np.clip(par_val[perm_idx], sample_range[perm_idx, 0] / args_range[perm_idx], sample_range[perm_idx, 1] / args_range[perm_idx])
                                            else:
                                                par_val[perm_idx] = (numpy.random.rand() * (sample_range[perm_idx, 1] - sample_range[perm_idx, 0]) + sample_range[perm_idx, 0]) / args_range[perm_idx]

                                            return par_val

                                        if args.constrain_valid_pos:
                                            assert valid_pos_func is not None
                                            while True:
                                                par_val = update_par_val()
                                                if valid_pos_func(par_val * args_range, default_width, default_height):
                                                    break
                                        else:
                                            par_val = update_par_val()                                            

                                        sess.run(assign_ops, feed_dict={assign_init_pl: par_val})

                                        if args.save_all_loss:
                                            loss_val, ans_base = sess.run([true_loss, base_loss])
                                            all_loss[i-all_loss_base, all_loss_idx] = ans_base
                                            all_loss_idx += 1
                                        else:
                                            loss_val = sess.run(true_loss)

                                    accept_prob = min(1, np.exp((-loss_val + old_loss_val) / (2 * accept_sigma2)))

                                    if args.verbose:
                                        print(i, nr, ns_wide, k, accept_prob, loss_val)


                                    if np.random.rand() <= accept_prob:
                                        last_loss_par = par_val
                                        last_loss_val = loss_val
                                        old_loss_val = loss_val

                                        if loss_val < min_loss_val:
                                            min_loss_val = loss_val
                                            min_loss_iter = k
                                            min_loss_par = par_val 

                                    else:
                                        pass

                                    frame_idx += 1

                            else:
                                raise

                        else:    

                            for k in range(max_iter):

                                frame_idx += 1

                                T2 = time.time()

                                if args.backend == 'tf':

                                    if gradient_method in ['finite_diff', 'finite_diff_pixelwise'] and args.finite_diff_random_dir:
                                        # sample a random unit vector u = [w_0, ..., w_n]
                                        # compute finite diff in the direction u
                                        # dL/du = sum(w_i * dL/dx_i)
                                        # for every i, dL/dx_i = w_i * dL/du
                                        # summing them up and becasue u = [w_0, ..., w_n] is a unit vector
                                        # sum(w_i * dL/dx_i) = sum(w_i ** 2 * dL/du) = dL/du
                                        random_dir = np.random.rand(int(tunable_params.shape[0])) - 0.5
                                        random_dir /= np.linalg.norm(random_dir)
                                    else:
                                        random_dir = None

                                    if gradient_method == 'finite_diff':
                                        T0 = time.time()
                                        deriv_val = assemble_finite_diff_gt(sess, true_loss, set_par, last_loss_par, old_loss_val=last_loss_val, output=output_valid, finite_diff_both_sides=args.finite_diff_both_sides, random_dir=random_dir, spsa_samples=args.finite_diff_spsa_samples)
                                        T1 = time.time()
                                        if args.profile_timing:
                                            print('time for finite diff scalar:', T1 - T0)
                                        if args.finite_diff_random_dir:
                                            deriv_val = deriv_val * random_dir
                                        feed_dict = {deriv_loss_with_respect_to_param: deriv_val}
                                    elif gradient_method == 'finite_diff_pixelwise':
                                        T0 = time.time()
                                        deriv_val = assemble_finite_diff_gt(sess, output_valid, set_par, last_loss_par, finite_diff_both_sides=args.finite_diff_both_sides, random_dir=random_dir, spsa_samples=args.finite_diff_spsa_samples)
                                        T1 = time.time()
                                        if args.profile_timing:
                                            print('time for finite diff pixelwise:', T1 - T0)
                                        if args.finite_diff_random_dir:
                                            deriv_val = deriv_val * random_dir[..., None, None, None]
                                        feed_dict = {deriv_output_with_respect_to_param: deriv_val}
                                    else:
                                        feed_dict = {}

                                    # this sequence of sess.run is needed to make sure that execution ordering is exactly as I wanted
                                    # TODO: maybe there's some control_dependency trick that could allow them all to be in the same sess.run with the expected execution ordering

                                    par_changed = False
                                    par_val = last_loss_par.copy()

                                    if args.random_purturb_zero_gradient:
                                        # do not apply random purturbation if any of the constrain is violated
                                        # because it is possible many useful parameters also have 0 gradient at this point
                                        # we don't want to discard all the progress made for those parameters

                                        gradient_val = sess.run(deriv_loss_with_respect_to_param, feed_dict=feed_dict)

                                        # uniformly purturb by -10% - 10% of the entire range
                                        # because parameters are already normalized, there's no need to multiply args_range
                                        if not all(gradient_val):

                                            par_changed = True
                                            zero_par_idx = numpy.where(gradient_val == 0)[0]

                                            if zero_par_idx.size > 1:
                                                print('encounter zero gradient')

                                            par_val[zero_par_idx] += (numpy.random.rand(zero_par_idx.size) - 0.5) / 5

                                    if par_changed:
                                        #sess.run(assign_ops, feed_dict={assign_init_pl: par_val})
                                        set_par(par_val, dummy_last_dim)


                                if args.backend == 'hl':

                                    par_val = last_loss_par

                                    auto_fill_render_kw(render_kw)

                                    if need_new_deriv:
                                        next_deriv, _, _ = metric.run_wrapper(par_val, stage=ns, get_loss=True, get_dL=False, check_last=False, get_deriv=True, render_kw=render_kw)


                                    par_val = opt.update(par_val, next_deriv)

                                    #par_val_orig = par_val * args_range
                                    par_val_orig = par_val


                                    next_deriv, loss_val, _ = metric.run_wrapper(par_val_orig, stage=ns, get_loss=True, get_dL=False, check_last=False, get_deriv=True, render_kw=render_kw)


                                    need_new_deriv = False

                                    if args.save_all_loss:
                                        # save the loss without random noise
                                        old_sigmas_scale = compiler_module.sigmas_scale
                                        compiler_module.sigmas_scale = 0
                                        _, ans_base, _ = metric.run_wrapper(par_val_orig, stage=ns, get_loss=True, get_dL=False, check_last=False, get_deriv=False, base_loss=True, skip_fw=True, render_kw=render_kw)
                                        compiler_module.sigmas_scale = old_sigmas_scale

                                elif args.backend == 'torch':
                                    assert not args.debug_mode
                                    
                                    opt.zero_grad()
                                    
                                    T2 = time.time()
                                    X_orig = set_par(tunable_params, False)
                                    if apply_random is not None:
                                        X_orig = apply_random(X_orig)
                                    x = shader(*params_orig[:-1], *X_orig, camera_width, camera_height, camera_depth)[current_slice]
                                    if args.multi_scale_optimization:
                                        current_step = ns
                                    else:
                                        current_step = possible_nsteps
                                    loss = metric_funcs[0](x, y, {'scale': current_step})
                                    
                                    loss.backward()
                                    opt.step()
                                    T3 = time.time()
                                    
                                    loss_val = loss.cpu()
                                    ans_base = base_loss(x, y)
                                    par_val = [float(val) for val in tunable_params]
                                    if apply_random is not None:
                                        par_val += [float(val) for val in random_var_scale_opt]

                                else:

                                    if args.debug_mode and args.halide_so_dir != '':
                                        feed_dict[noise_val] = get_noise(last_loss_par[1:-dummy_last_dim])

                                    T2 = time.time()
                                    sess.run(step, feed_dict=feed_dict)
                                    T3 = time.time()

                                    if args.profile_timing:
                                        print('time for opt step (including gradient computation for loss)', T3 - T2)

                                    if args.save_all_loss:
                                        loss_val, par_val, ans_base = sess.run([true_loss, all_params, base_loss], feed_dict=feed_dict)
                                    else:
                                        loss_val, par_val = sess.run([true_loss, all_params], feed_dict=feed_dict)

                                if args.save_all_loss:
                                    all_loss[i-all_loss_base, all_loss_idx] = ans_base
                                    all_loss_idx += 1

                                if args.save_all_par:
                                    current_par[current_par_idx, :] = par_val
                                    
                                if args.save_all_deriv:
                                    current_deriv[current_par_idx, :] = sess.run(gvs[0][0])
                                    
                                current_par_idx += 1

                                if args.verbose:
                                    print(i, nr, ns_wide, k, loss_val)

                                has_nan = False
                                if numpy.any(numpy.isnan(par_val)):
                                    has_nan = True



                                if has_nan:
                                    need_new_deriv = True
                                    # to prevent NAN appearing in any of the optimizer variables
                                    sess.run(tf.local_variables_initializer())
                                    sess.run(tf.global_variables_initializer())
                                    #feed_dict = {assign_init_pl: np.array(min_loss_par)}
                                    #sess.run(assign_ops, feed_dict=feed_dict)
                                    set_par(min_loss_par, dummy_last_dim)
                                    last_loss_par = np.array(min_loss_par)
                                    last_loss_val = min_loss_val
                                    continue
                                else:
                                    last_loss_par = np.array(par_val)
                                    last_loss_val = loss_val

                                if loss_val < min_loss_val:
                                    min_loss_val = loss_val
                                    min_loss_iter = k
                                    min_loss_par = par_val
                                elif k > min_loss_iter + early_termination_iter:
                                    break



                                T3 = time.time()
                                if args.profile_timing:
                                    print('time for 1 iter:', T3 - T2)

                            if args.save_all_par:
                                if args.verbose_save:
                                    all_par = np.concatenate((all_par, current_par), 0)
                                else:
                                    all_par = np.concatenate((all_par, current_par[:min_loss_iter+1, :]), 0)
                                
                            if args.save_all_deriv:
                                if args.verbose_save:
                                    all_deriv = np.concatenate((all_deriv, current_deriv), 0)
                                else:
                                    all_deriv = np.concatenate((all_deriv, current_deriv[:min_loss_iter+1, :]), 0)

                    if args.show_progress:
                        print()
                    
                    if args.backend == 'hl':
                        old_sigmas_scale = compiler_module.sigmas_scale
                        compiler_module.sigmas_scale = 0
                        result_img = halide_fw(min_loss_par)
                        if args.optimizer == 'mcmc':
                            min_loss_val = get_loss(min_loss_par)
                        else:
                            # save the loss without random noise
                            if args.base_loss_stage < 0:
                                _, min_loss_val, _ = metric.run_wrapper(min_loss_par, stage=-1, get_loss=True, get_dL=False, check_last=False, get_deriv=False, base_loss=True, render_kw=render_kw)
                            else:
                                _, min_loss_val, _ = metric.run_wrapper(min_loss_par, stage=args.base_loss_stage, get_loss=True, get_dL=False, check_last=False, get_deriv=False, render_kw=render_kw)

                        compiler_module.sigmas_scale = old_sigmas_scale
                    elif args.backend == 'torch':
                        X_orig = set_par(min_loss_par)
                        set_random_var(sess, False)
                        x = shader(*params_orig[:-1], *X_orig, camera_width, camera_height, camera_depth)
                        result_img = x.cpu().detach().numpy()[current_slice][0]
                        min_loss_val = float(base_loss(x[current_slice], y))
                        set_random_var(sess, True)
                    else:
                        set_par(min_loss_par, dummy_last_dim)
                        set_random_var_opt(sess, min_loss_par[-dummy_last_dim:])

                        set_random_var(sess, False)

                        result_img, min_loss_val = sess.run([output_valid, base_loss], feed_dict=feed_dict)
                        result_img = result_img[0]
                        
                        set_random_var(sess, True)

                    name_prefix = '%s%s%s%s_%s_%s_%.1e%s_result%d_%d' % (gradient_method, '' if args.finite_diff_spsa_samples <= 0 else str(args.finite_diff_spsa_samples), '_both_sides' if args.finite_diff_both_sides else '', '_random_dir' if args.finite_diff_random_dir else '', metric_name, args.optimizer, args.learning_rate, args.suffix, i, nr)


                    if args.save_all_par:
                        if dummy_last_dim > 0:
                            all_par[:, :-dummy_last_dim] *= args_range
                        else:
                            all_par *= args_range
                        np.save(os.path.join(args.dir, '%s.npy' % name_prefix), all_par)
                        
                    if args.save_all_deriv:
                        np.save(os.path.join(args.dir, '%s_deriv.npy' % name_prefix), all_deriv)

                    name = '%s.png' % name_prefix

                    if args.backend == 'hl':
                        skimage.io.imsave(os.path.join(args.dir, name), np.clip(result_img.transpose((1, 0, 2)), 0, 1), check_contrast=False)
                    else:
                        imsave(os.path.join(args.dir, name), result_img, args.backend == 'hl', ndims=args.ndims)

                    if random_var_opt_len > 0:
                        if args.backend != 'hl':
                            set_random_var(sess, True)
                            
                            if args.backend == 'tf':
                                result_img = sess.run(output_valid, feed_dict=feed_dict)[0]
                            elif args.backend == 'torch':
                                x = shader(*params_orig[:-1], *X_orig, camera_width, camera_height, camera_depth)
                                result_img = x.cpu().detach().numpy()[current_slice][0]

                            result_img = result_img
                        else:
                            result_img = halide_fw(min_loss_par).transpose((1, 0, 2))

                        imsave(os.path.join(args.dir, '%s_random.png' % name_prefix), result_img, args.backend == 'hl', ndims=args.ndims)
                        
                    if args.verbose:
                        print(i, nr, min_loss_iter, min_loss_val)
                        print(min_loss_par, file=logfile)
                    print(i, nr, min_loss_iter, min_loss_val, file=logfile)

                    if min_loss_val < global_min_loss_val:
                        global_min_loss_val = min_loss_val
                        global_min_loss_par = min_loss_par
                        
                    if global_min_loss_val < best_loss:
                        best_loss = global_min_loss_val
                        best_par = global_min_loss_par

                print(i, 'global', min_loss_iter, global_min_loss_val)
                print(i, 'global', min_loss_iter, global_min_loss_val, file=logfile)
                                
            T_opt_end = time.time()
            T_total = T_opt_start - T_opt_end

            if args.save_all_loss:
                if args.loss_filename == '':
                    filename = '%s%s%s_%s_%s_%.1e%s_all_loss.npy' % (gradient_method, '' if args.finite_diff_spsa_samples <= 0 else str(args.finite_diff_spsa_samples), '_both_sides' if args.finite_diff_both_sides else '', metric_name, args.optimizer, args.learning_rate, args.suffix)
                    filename = os.path.join(args.dir, filename)
                else:
                    filename = os.path.join(args.dir, args.loss_filename + '.npy')
                np.save(filename, all_loss)

            print('running on', platform.node(), file=logfile)
            print('total runtime', T_total, file=logfile)
            if args.save_all_loss:
                total_iters = 0
                for i in range(all_loss.shape[0]):
                    valid_iters = np.where(all_loss[i] == 0)[0]
                    if valid_iters.size > 0:
                        total_iters += np.where(all_loss[i] == 0)[0][0]
                    else:
                        total_iters += all_loss.shape[1]
                print('total iterations', total_iters, file=logfile)
                print('runtime per iter', T_total / total_iters, file=logfile)

            logfile.close()
            
            if args.save_best_par:
                best_par_file = os.path.join(args.dir, 'best_par%s.npy' % args.suffix)
                np.save(best_par_file, np.expand_dims(best_par[:args_range.shape[0]] * args_range, 0))
            
            if not args.ignore_glsl:
                
                if getattr(compiler_module, 'n_updates', 0) > 0:
                    do_prune = get_do_prune(metric, compiler_module, render_kw, best_par[:args_range.shape[0]])
                else:
                    do_prune = None
                    
                if best_par is not None:
                    generate_interactive_frag(args, best_par[:args_range.shape[0]] * args_range, do_prune)

    
if __name__ == '__main__':
    args = get_args(str_args=None)
    main(args)
        
    