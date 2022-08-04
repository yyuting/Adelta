import sys
import time
import importlib
import numpy
import numpy as np
import skimage.io
import os
import warnings
import imageio.core.util
import copy
import pickle
import importlib

np.random.seed(1)

def ignore_warnings(*args, **kwargs):
    pass

imageio.core.util._precision_warn = ignore_warnings

so_lib = None
so_loaded = False

compiler_problem_lib = None
compiler_problem_loaded = False

compiler_problem_scalar_lib = None
compiler_problem_scalar_loaded = False

def load_so(compiler_problem_path=None, verbose=True):
    
    if compiler_problem_path is not None:
        sys.path += [compiler_problem_path]
        
    global compiler_problem_lib
    global compiler_problem_loaded
    
    if not compiler_problem_loaded:
        import compiler_problem
        compiler_problem_lib = compiler_problem
    else:
        importlib.reload(compiler_problem_lib)
        
    if verbose:
        print('loaded compiler_problem from %s' % compiler_problem_path)
    
    global so_lib
    global so_loaded
    
    so_lib = compiler_problem_lib

    compiler_problem_loaded = True
    so_loaded = True
    
    if os.path.exists(os.path.join(compiler_problem_path, 'compiler_problem_scalar.py')):
        global compiler_problem_scalar_lib
        global compiler_problem_scalar_loaded
        import compiler_problem_scalar
        compiler_problem_scalar_lib = compiler_problem_scalar
        compiler_problem_scalar_loaded = True
        
        if verbose:
            print('loaded compiler_problem scalar from %s' % compiler_problem_path)
        
def get_wh(render_kw):
    
    if 'render_size' in render_kw.keys():
        return render_kw['render_size']
    else:
        return 960, 640
    
    #return render_kw.get('width', 960), render_kw.get('height', 640)

def get_buffer(render_kw, info_kw):
        
    width, height = get_wh(render_kw)

    init_buf = True

    if 'buffer' in info_kw.keys():
        if (width, height) in info_kw['buffer'].keys():
            buffer = info_kw['buffer'][(width, height)]
            init_buf = False
        else:
            buffer = []
    else:
        buffer = []

    render_kw['buffer'] = buffer

    return init_buf

def set_buffer(render_kw, info_kw):

    buffer = render_kw['buffer']

    assert len(buffer) > 0

    width, height = get_wh(render_kw)

    if 'buffer' not in info_kw.keys():
        info_kw['buffer'] = {(width, height): buffer}
    else:
        info_kw['buffer'][(width, height)] = buffer
    
class nscale_L2():
    
    def __init__(self, y=None, nscale=0, smoothing_sigmas=[], multiple_obj=True, scipy_mode=False, ignore_last_n_scale=0, opt_subset_idx=None, match_target=None, verbose=True):
        
        self.nscale = nscale
        self.smoothing_sigmas = smoothing_sigmas
        self.multiple_obj = multiple_obj
        self.ignore_last_n_scale = ignore_last_n_scale
        
        self.y_pyramid = None
        self.y_pyramid_arr = None
        
        self.shader = None
        
        self.last_par = 0
        self.last_deriv = None
        
        self.bw_lib = so_lib.nscale_L2

        self.x = None
        self.x_arr = None
        self.dL = None
        self.dL_arr = None
        self.L = so_lib.Buffer((1,))
        self.L_arr = np.asarray(self.L)
        
        self.is_FD = False
        self.is_SPSA = False
        self.finite_diff_h = 0
        
        self.verbose = verbose
        
        self.opt_subset_idx = opt_subset_idx
        
        if y is not None:
            self.set_y(y)
            
        self.steps = []
            
        if not self.check_lib_exists():
            print('Error! some kernels needed are not compiled\n')

        
        self.pipeline_fw = None
        self.pipeline_bw = None
        
        # if is SPSA, need additional arg: SPSA_samples
        self.is_SPSA = False
        
        self.updated = False
        
        self.match_target = match_target
        
        
    def check_lib_exists(self):
        
        pl_bufs = [so_lib.Buffer(())] * 13
        
        success = True
        
        self.steps = []
        
        if self.multiple_obj:
            
            self.nsteps = self.nscale + 1 + len(self.smoothing_sigmas)
            
            for idx in range(self.ignore_last_n_scale, self.nsteps):
                
                def generate_func(current_idx):
                    def func(args=[], kw_args={}):
                        return so_lib.nscale_L2(self.nscale, self.smoothing_sigmas, *args, start_stage=current_idx, **kw_args)
                    return func
                
                self.steps.append(generate_func(idx))
                
                ans = self.steps[-1]([0, 0, *pl_bufs], {'get_loss': False, 'check_ok': True})
                
                if ans != 0:
                    success = False
                    print('Error! kernel nscale = %d, sigmas = %s, start_stage = %d is not compiled!\n' % 
                      (self.nscale, ', '.join([str(val) for val in self.smoothing_sigmas]), idx))
                    
            self.nsteps -= self.ignore_last_n_scale
        else:
            def func(args=[], kw_args={}):
                ans = so_lib.nscale_L2(self.nscale, self.smoothing_sigmas, *args, **kw_args)
                
                if self.ignore_last_n_scale > 0:
                    if self.ignore_last_n_scale <= self.nscale:
                        ans -= so_lib.nscale_L2(self.ignore_last_n_scale, [], *args, **kw_args)
                    else:
                        assert False, 'Not implemented'
                return ans
                
            self.steps.append(func)
            
            ans = func([0, 0, *pl_bufs], {'get_loss': False, 'check_ok': True})
            
            if ans != 0:
                success = False
                print('Error! kernel nscale = %d, sigmas = %s is not compiled!\n' % 
                      (self.nscale, ', '.join([str(val) for val in self.smoothing_sigmas])))
                
            self.nsteps = 1
                
        self.steps = self.steps[::-1]
                    
        return success
        
    def check_wh(self, render_kw):
        assert self.y is not None
        
        if 'render_size' in render_kw.keys():
            assert render_kw['render_size'] == [self.y.shape[0], self.y.shape[1]]
        else:
            render_kw['render_size'] = [self.y.shape[0], self.y.shape[1]]
            
    def run_wrapper(self, params, **kw):
        
        if self.is_FD:
            
            stage = kw.get('stage', 0)
            get_loss = kw.get('get_loss', True)
            get_dL = kw.get('get_dL', False)
            render_kw = kw.get('render_kw', {})
            get_deriv = kw.get('get_deriv', True)
            base_loss = kw.get('base_loss', False)
            
            # get_dL does not make sense here
            assert not get_dL
            
            if get_deriv:
                if self.SPSA_samples > 0:
                    deriv = np.zeros(params.size)
                    for _ in range(self.SPSA_samples):
                        offset = (np.random.binomial(1, 0.5, params.size) * 2 - 1).astype('f')
                        offset *= self.finite_diff_h

                        _, loss_pos, _ = self.run(params + offset, stage=stage, get_loss=True, get_dL=False, render_kw=render_kw, check_last=False, get_deriv=False, base_loss=base_loss, skip_fw=False)
                        
                        _, loss_neg, _ = self.run(params - offset, stage=stage, get_loss=True, get_dL=False, render_kw=render_kw, check_last=False, get_deriv=False, base_loss=base_loss, skip_fw=False)
                        
                        # multiply by offset, as if the actual SPSA is applied on the normalized parameters
                        deriv += (loss_pos - loss_neg) / (2 * offset)
                    deriv /= self.SPSA_samples
                else:
                    deriv = np.empty(params.size)
                    
                    offset = np.zeros(params.size)
                    for idx in range(params.size):
                        offset[:] = 0
                        offset[idx] = self.finite_diff_h
                        
                        _, loss_pos, _ = self.run(params + offset, stage=stage, get_loss=True, get_dL=False, render_kw=render_kw, check_last=False, get_deriv=False, base_loss=base_loss, skip_fw=False)
                        
                        _, loss_neg, _ = self.run(params - offset, stage=stage, get_loss=True, get_dL=False, render_kw=render_kw, check_last=False, get_deriv=False, base_loss=base_loss, skip_fw=False)
                        
                        deriv[idx] = (loss_pos - loss_neg) / (2 * self.finite_diff_h)
                        
                if self.shader.normalized_par:
                    deriv[:self.shader.nargs] *= self.shader.args_range
                    if deriv.size > self.shader.nargs:
                        deriv[self.shader.nargs:] *= self.shader.sigmas_scale * self.shader.sigmas_range[render_kw['sigmas_idx']]
            else:
                deriv = None
                
            if get_loss:
                _, loss_val, _ = self.run(params, stage=stage, get_loss=True, get_dL=False, render_kw=render_kw, check_last=False, get_deriv=False, base_loss=base_loss, skip_fw=False)
            else:
                loss_val = None
                
            return deriv, loss_val, None
        else:
            return self.run(params, **kw)
            
        
    def run(self, params, stage=0, get_loss=True, get_dL=False, render_kw={}, check_last=False, get_deriv=True, base_loss=False, skip_fw=False):
        
        if check_last:
            if self.last_par == (params * self.random_dir).sum():
                if base_loss:
                    skip_fw = True
                else:
                    return self.last_deriv, self.L_arr[0], self.dL_arr
                                
        if params.size > self.shader.nargs:
            assert params.size == self.shader.nargs + render_kw['sigmas_idx'].size
            render_kw['sigmas'] = params[self.shader.nargs:]
            params = params[:self.shader.nargs]
        
        self.updated = True
        
        assert self.x is not None and self.dL is not None and self.y is not None

        self.check_wh(render_kw)
        
        if not skip_fw:
            render_kw['copy_output'] = False
            self.pipeline_fw(params, render_kw)
        
        if not base_loss:
            self.steps[stage]([self.y.shape[0], self.y.shape[1], self.x, *self.y_pyramid, self.dL, self.L],
                              {'get_loss': get_loss,
                               'get_deriv': get_deriv and (not self.is_FD)})
        else:
            so_lib.nscale_L2(0, [], 
                             self.y.shape[0], self.y.shape[1],
                             self.x, *self.y_pyramid, self.dL, self.L,
                             get_loss=get_loss,
                             get_deriv=False)
        
        if render_kw.get('weight_map', None) is not None:
            if get_loss or get_deriv:
                # TODO: slow implementation for experiment purpose
                # if it helps, should implement fast version in Halide
                self.dL.copy_to_host()
                self.dL_arr *= render_kw['weight_map']
                
                if get_deriv:
                    self.dL.set_host_dirty()
                    
                if get_loss:
                    if get_deriv:
                        dim = -1
                    else:
                        dim = 0
                    loss_val = self.dL_arr[..., dim].sum()
        
        if get_deriv:
            if self.pipeline_bw is not None:
                # should copy out gradient per parameter
                render_kw['copy_output'] = False
                render_kw['reset_min'] = False
                render_kw['copy_reduce'] = True
                
                deriv, deriv_sigma = self.pipeline_bw(params, render_kw)
                
                if self.opt_mask_out_idx is not None:
                    deriv[self.opt_mask_out_idx] = 0
                    if deriv_sigma is not None:
                        deriv_sigma[self.opt_sigmas_mask_out_idx] = 0

                if deriv_sigma is not None:
                    deriv = np.concatenate((deriv, deriv_sigma))
        else:
            deriv = None
            deriv_sigma = None
            
        if get_loss:
            if render_kw.get('weight_map', None) is None:
                self.L.copy_to_host()
                loss_val = self.L_arr[0]
        else:
            loss_val = None
            
        def update_extra_loss(func, *args):
            if self.shader.normalized_par:
                input_to_extra = params * self.shader.args_range
            else:
                input_to_extra = params
                
            extra_loss, extra_deriv = func(input_to_extra, *args)
            
            nonlocal loss_val, deriv
            
            #if get_loss:
            #    loss_val = loss_val + extra_loss
                
            if get_deriv:
                if self.shader.normalized_par:
                    extra_deriv *= self.shader.args_range
                deriv[:params.size] += extra_deriv
                
        if get_loss or get_deriv:
            if self.shader.scalar_loss is not None:
                update_extra_loss(self.shader.scalar_loss.f, self.shader.scalar_loss_scale)
            if self.match_target is not None:
                update_extra_loss(self.match_target)
        
        if get_dL:
            self.dL.copy_to_host()
            dL_val = self.dL_arr
        else:
            dL_val = None
            
        self.last_par = (params * self.random_dir).sum()
        self.last_deriv = deriv
            
        return deriv, loss_val, dL_val

    def set_y(self, new_y, base_only=False):
        self.y = new_y
        self.build_y_pyramid(base_only=base_only)
        
        generate_buf = True
        if self.dL is not None and self.dL_arr is not None:
            if self.dL_arr.shape == (self.y.shape[0], self.y.shape[1], 4):
                generate_buf = False
        
        if generate_buf:
            self.dL = so_lib.Buffer((self.y.shape[0], self.y.shape[1], 4))
            self.dL_arr = np.asarray(self.dL)
        
    def set_x(self, shader, func_name=None, render_kw=None):
        
        assert self.y is not None
        
        self.shader = shader
        
        self.random_dir = np.random.rand(shader.nargs)
        
        if self.opt_subset_idx is not None:
            self.opt_mask_out_idx = list(set(np.arange(self.shader.nargs)).difference(set(self.opt_subset_idx)))
            
            self.opt_sigmas_mask_out_idx = []
            for idx in self.opt_mask_out_idx:
                if idx in render_kw.get('sigmas_idx', []):
                    self.opt_sigmas_mask_out_idx.append(render_kw['sigmas_idx'].tolist().index(idx))
        else:
            self.opt_mask_out_idx = None
            self.opt_sigmas_mask_out_idx = None
        
        if render_kw is None:
            render_kw = {}
            
        self.check_wh(render_kw)
        
        allow_producer = False
        if func_name == 'backward':
            bw_lib_type = shader.infer_lib_type(func_name, render_kw=render_kw)
            init_buf = get_buffer(render_kw, shader.bw_info[bw_lib_type])

            bw_buffer_info = shader.bw_info[bw_lib_type]['buffer_info']
            for entry in bw_buffer_info:
                if entry.get('tag', '') == 'producer':
                    allow_producer = True
                    break
                    
        use_producer = False
        rm_sigma = False
        
        fw_info_name = 'fw_info'
        if len(render_kw.get('do_prune', [])) > 0:
            fw_name = 'forward'
            fw_lib_type = shader.infer_lib_type(fw_name, render_kw=render_kw)
            rm_sigma = True
        elif len(render_kw.get('sigmas', [])) > 0:
            # there's no need to run producer for finite diff, fw will be enough
            fw_name = 'forward'
            if allow_producer and func_name != 'finite_diff':
                use_producer = True
                fw_name = 'producer'
            fw_lib_type = shader.infer_lib_type(fw_name, render_kw=render_kw)
        elif func_name is None:
            fw_lib_type = 'regular'
        elif func_name == 'finite_diff':
            fw_lib_type = 'regular'
        else:
            fw_lib_type = 'regular'
            if allow_producer:
                use_producer = True
                
        if use_producer:
            render_kw['seperate_producer'] = True
            fw_info_name = 'producer_info'
                
        def func(param, kw):
            if use_producer:
                kw['seperate_producer'] = True
            if rm_sigma:
                kw['sigmas'] = []
            shader.forward(param, render_kw=kw)
            
        self.pipeline_fw = func
                
        init_fw_buf = get_buffer(render_kw, getattr(shader, fw_info_name)[fw_lib_type])

        if init_fw_buf:
            self.pipeline_fw(np.zeros(shader.nargs), render_kw)

        fw_buffer_info = getattr(shader, fw_info_name)[fw_lib_type]['buffer_info']

        found_fw_col = False
        for idx in range(len(fw_buffer_info)):
            info = fw_buffer_info[idx]
            if 'tag' in info.keys():
                if info['tag'] == 'col':
                    assert not found_fw_col
                    self.x = render_kw['buffer'][idx]
                    self.x_arr = np.asarray(self.x)
                    found_fw_col = True
                    
        assert found_fw_col
        
        if func_name == 'backward':
            def func(param, kw):
                if use_producer:
                    kw['compute_producer'] = False

                deriv, deriv_sigmas = shader.backward(param, render_kw=kw)
                
                if self.shader.normalized_par:
                    deriv *= self.shader.args_range
                    if deriv_sigmas is not None:
                        deriv_sigmas *= self.shader.sigmas_scale * self.shader.sigmas_range[kw['sigmas_idx']]
                
                return deriv, deriv_sigmas

            self.pipeline_bw = func
        elif func_name == 'finite_diff':
            self.is_FD = True
            self.SPSA_samples = render_kw.get('SPSA_samples', -1)
            self.finite_diff_h = render_kw['finite_diff_h']
            self.pipeline_bw = None
        else:
            assert func_name is None
            self.pipeline_bw = None
            
        if self.pipeline_bw is None:
            return
        
        if init_buf:
            self.pipeline_bw(np.zeros(shader.nargs), render_kw)
                
        assert len(render_kw['buffer'])
        
        buffer_info = shader.bw_info[bw_lib_type]['buffer_info']
        
        found_dL_dcol = False
        found_producer = False
        
        for idx in range(len(buffer_info)):
            
            info = buffer_info[idx]
            
            if 'tag' in info.keys():
                if info['tag'] == 'dL_dcol':
                    assert not found_dL_dcol
                    render_kw['buffer'][idx] = self.dL
                    found_dL_dcol = True
                elif info['tag'] == 'producer':
                    assert not found_producer
                    render_kw['buffer'][idx] = self.x
                    found_producer = True

        assert found_dL_dcol
        if use_producer:
            assert found_producer

    def build_y_pyramid(self, base_only=False):
        
        """
        NOTE:
        base_only is a hacky way to only set the highest resolution pyramid to the new image, 
        leaving lower res pyramid untouched.
        This is used to so that the new y is only used for base L2 loss
        User needs to make sure the code behaves a expected
        """
        
        ninputs = 10
        
        generate_buf = True
        if self.y_pyramid is not None:
            assert self.y_pyramid_arr is not None
            
            if self.y_pyramid_arr[0].shape == self.y.shape:
                generate_buf = False
                bufs = self.y_pyramid
                arrs = self.y_pyramid_arr
            
        if generate_buf:
            bufs = []
            arrs = []
        
        for idx in range(self.nscale + 1):
            
            
            
            if generate_buf:
                scale = int(2 ** idx)
                bufs.append(so_lib.Buffer((self.y.shape[0] // scale, self.y.shape[1] // scale, self.y.shape[2])))
                arrs.append(np.asarray(bufs[idx]))
            
            if idx == 0:
                arrs[idx][:] = self.y[:]
                bufs[idx].set_host_dirty()
                #print('initializing y pyramid original res')
                
                if base_only:
                    break
                
            else:
                so_lib.downsample(2, bufs[idx-1], bufs[idx])
                #print('initializing y pyramid, downsample 2x')
            
        if self.smoothing_sigmas is not None and not base_only:
            
            assert isinstance(self.smoothing_sigmas, (list, np.ndarray))
            
            scale = int(2 ** self.nscale)
            
            for idx in range(len(self.smoothing_sigmas)):
                
                if generate_buf:
                    bufs.append(so_lib.Buffer((self.y.shape[0] // scale, self.y.shape[1] // scale, self.y.shape[2])))
                    arrs.append(np.asarray(bufs[-1]))
                
                so_lib.gaussian_conv(self.smoothing_sigmas[idx], bufs[self.nscale], bufs[self.nscale+1+idx])
                
                if self.verbose:
                    print('initialize y pyramic, conv with sigma ', self.smoothing_sigmas[idx])
                
        if generate_buf:
            assert len(bufs) <= ninputs

            for _ in range(ninputs - len(bufs)):
                bufs.append(so_lib.Buffer(()))
                
            self.y_pyramid = bufs
            self.y_pyramid_arr = arrs
    
class GenericShader():
    
    def __init__(self):
        
        self.fw_info = {}
        self.bw_info = {}
        
        self.FD_lib = None
        self.FD_info = None
        
        self.nargs = 0
        self.params_reorder = np.zeros(0)
        
        self.fw_uv_sample_buffer_idx = None
        self.bw_uv_sample_buffer_idx = None
        self.bw_choose_u_pl_buffer_idx = None
        
        self.normalized_par = False
        self.sigmas_scale = 1
        
        # gradient computed from Halide may have a different ordering than its input parameters
        # reordering the gradient map will be expensive
        # instead, we take note of the ordering and will make it correct whenever needed
        # for each instance, this bw_map should be a 1D numpy array
        # the ith input parameter corresponds to the bw_map[i]th gradient map
        self.bw_map = None
        
    def update_FD_info(self):
        self.FD_info = {'FD': 
                        {'buffer_info':
                         [{'ndim': 3, 'nfeats': 3, 'type': 'input', 'default_val': 1, 'tag': 'dL_dcol'},
                          {'ndim': 3, 'nfeats': 2, 'type': 'input', 'default_val': 0}] +  
                         [{'ndim': 3, 'nfeats': 1, 'type': 'output', 'reduce': (0, 1)}] * self.nargs},
                        'SPSA': 
                        {'buffer_info':
                         [{'ndim': 3, 'nfeats': 3, 'type': 'input', 'default_val': 1, 'tag': 'dL_dcol'},
                          {'ndim': 3, 'nfeats': 2, 'type': 'input', 'default_val': 0},
                          {'ndim': 3, 'nfeats': self.nargs, 'type': 'intermediate', 'reduce': (0, self.nargs)},
                          {'ndim': 3, 'nfeats': self.nargs, 'type': 'intermediate', 'reduce': (0, self.nargs)}]}
                       }
        
    def infer_lib_type(self, func_name, render_kw={}):
        
        per_pixel_offset = False
        if 'uv_sample' in render_kw.keys():
            uv_sample = render_kw['uv_sample']
            
            if isinstance(uv_sample, np.ndarray) and uv_sample.size > 2:
                per_pixel_offset = True

        init_buf = False

        if func_name == 'finite_diff':
            if render_kw.get('SPSA_samples', -1) > 0:
                lib_type = 'SPSA'
            else:
                lib_type = 'FD'
            info_ls = self.FD_info
        elif func_name == 'forward':
            lib_type = 'regular'
            if per_pixel_offset:
                lib_type = 'per_pixel_offset'
            elif len(render_kw.get('do_prune', [])) > 0:
                lib_type = 'prune_updates'
            info_ls = self.fw_info
        elif func_name == 'producer':
            lib_type = 'regular'
            info_ls = self.producer_info
        elif func_name == 'backward':
            lib_type = 'regular'
            if per_pixel_offset:
                lib_type = 'per_pixel_offset'
                if 'choose_u_pl' in render_kw.keys():
                    lib_type = 'offset_choose_u_pl'

            with_denum = render_kw.get('with_denum', False)
            denum_only = render_kw.get('denum_only', False)

            if denum_only:
                lib_type = 'denum_only'
            elif with_denum:
                lib_type = 'with_denum'
                    
            info_ls = self.bw_info
        else:
            raise 'Unsuported func_name'
                
        if 'sigmas' in render_kw.keys():
            if len(render_kw['sigmas']) > 0:
                
                if func_name == 'finite_diff':
                    assert lib_type == 'FD'
                    base_idx = 2
                
                orig_lib_type = lib_type
                
                assert 'sigmas_idx' in render_kw.keys() and len(render_kw['sigmas']) == len(render_kw['sigmas_idx'])

                lib_type = 'random_noise_' + '_'.join([str(val) for val in render_kw['sigmas_idx']])
                
                if lib_type not in info_ls.keys():
                    info_ls[lib_type] = {'lib': info_ls[orig_lib_type]['lib']}
                    
                if 'buffer_info' not in info_ls[lib_type].keys():
                    info_ls[lib_type]['buffer_info'] = copy.deepcopy(info_ls[orig_lib_type]['buffer_info'])
                    init_buf = True
                
                    current_buffer = info_ls[lib_type]['buffer_info']
                    sigmas_par_idx = []
                    for idx in range(len(current_buffer)):
                        
                        if func_name == 'finite_diff':
                            if idx - base_idx in render_kw['sigmas_idx']:
                                sigmas_par_idx.append(idx - base_idx)
                                
                                new_buf_info = copy.deepcopy(current_buffer[idx])
                                current_buffer.append(new_buf_info)
                        else:
                        
                            if 'par_idx' in current_buffer[idx].keys():

                                nparams = 0

                                for par_idx in current_buffer[idx]['par_idx']:
                                    if par_idx in render_kw['sigmas_idx']:
                                        sigmas_par_idx.append(par_idx)
                                        nparams += 1

                                current_buffer[idx]['orig_nfeats'] = current_buffer[idx]['nfeats']
                                current_buffer[idx]['nfeats'] += nparams

                                if nparams > 0:
                                    if 'reduce' in current_buffer[idx].keys():
                                        current_buffer[idx]['orig_reduce'] = current_buffer[idx]['reduce']
                                        current_buffer[idx]['reduce'] = (current_buffer[idx]['reduce'][0],
                                                                         current_buffer[idx]['reduce'][1] + nparams)

                    info_ls[lib_type]['sigmas_idx_order'] = np.argsort(sigmas_par_idx)
                     
        
            
        return lib_type
            

    def finite_diff(self, params,
                    path='', name='', visualize=False, transpose=False, reuse_buffer=True, normalize=True, render_kw={}):
        
        assert normalize
        
        finite_diff_h = render_kw.get('finite_diff_h', 0.01)
        SPSA_samples = render_kw.get('SPSA_samples', -1)
        
        if self.FD_info is None:
            self.update_FD_info()
            
        if params.size > self.nargs:
            assert params.size == self.nargs + render_kw['sigmas_idx'].size
            sigmas = params[self.nargs:]
            render_kw['sigmas'] = sigmas
            params = params[:self.nargs]
        else:
            sigmas = render_kw.get('sigmas', [])

        if self.normalized_par:
            params = params * self.args_range
            
        if len(sigmas) > 0:
            if self.normalized_par:
                sigmas = sigmas * self.sigmas_scale * self.sigmas_range[render_kw['sigmas_idx']]
            param_and_sigma = [params, sigmas]
        else:
            param_and_sigma = [params]
            
        if normalize:
            assert hasattr(self, 'args_range')
            
        per_pixel_offset = False
        if 'uv_sample' in render_kw.keys():
            uv_sample = render_kw['uv_sample']
            
            if isinstance(uv_sample, np.ndarray) and uv_sample.size > 2:
                per_pixel_offset = True
                render_kw['uv_sample_buffer_idx'] = 1

        lib_type = self.infer_lib_type('finite_diff', render_kw)
            
        if 'lib' in self.FD_info[lib_type].keys():
            current_lib = self.FD_info[lib_type]['lib']
        elif per_pixel_offset:
            current_lib = self.FD_per_pixel_offset_lib
        else:
            current_lib = self.FD_lib
            
        buffer_info = self.FD_info[lib_type]['buffer_info']
        
        assert current_lib is not None, "Error! Shader's FD kernel NOT specified!"
        
        assert reuse_buffer
        
        width, height = get_wh(render_kw)
        
        init_buf = get_buffer(render_kw, self.FD_info[lib_type])
                
        extra_args = [render_kw.get('uv_offset', [0, 0])[0], render_kw.get('uv_offset', [0, 0])[1],
                      width, height]
        
        if 'tile_offset' in render_kw.keys():
            extra_args[0] += render_kw['tile_offset'][0]
            extra_args[1] += render_kw['tile_offset'][1]
        
        frame_idx = render_kw.get('frame_idx', 0)
        
        if 'is_profile' in render_kw.keys():
            is_profile = render_kw['is_profile']
        else:
            is_profile = False
        
        ans = None
            
        offset_idx = 1
        base_idx = 2
                
        if SPSA_samples > 0:
            
            # TODO: remove this assertion and finish logic with random parameters
            assert len(sigmas) == 0
            
            def lib_wrapper(buf):
                
                nonlocal extra_args, finite_diff_h, SPSA_samples, frame_idx
                
                for idx in range(SPSA_samples):
                    
                    offset = np.random.binomial(1, 0.5, self.nargs) * 2 - 1
                    
                    if normalize:
                        offset = offset.astype('f')
                        offset *= self.args_range
                    
                    if idx < SPSA_samples - 1:
                        divide_by = 1
                    else:
                        divide_by = SPSA_samples
                        
                    old_idx = base_idx + idx % 2
                    current_idx = base_idx +(idx + 1) % 2
                    
                    if per_pixel_offset:
                        buf_args = [buf[0], buf[offset_idx], buf[old_idx], buf[current_idx]]
                    else:
                        buf_args = [buf[0], buf[old_idx], buf[current_idx]]
                    
                    current_lib(*buf_args, 
                                params, offset, *extra_args,
                                finite_diff_h, divide_by,
                                frame_idx = frame_idx,
                                output_base_only = False,
                                add_to_old = idx != 0)
                    
                return
            
            ans = self.run(lib_wrapper, buffer_info, params, use_lib_wrapper=True, **render_kw)
            
            if init_buf:
                set_buffer(render_kw, self.FD_info[lib_type])
            
            if not is_profile:
                
                if render_kw.get('copy_output', True):
                    assert len(ans) == 0
                    out_idx = base_idx + (SPSA_samples) % 2
                    render_kw['buffer'][out_idx].copy_to_host()
                    ans = np.asarray(render_kw['buffer'][out_idx])

                    if normalize:
                        # gradient wrt NORMALIZED parameter
                        # because offset is scaled by args_range, it's no longer 0 or 1
                        # but SPSA wants to be multiplying by the sign of the offset, NOT the actual value of offset
                        
                        # because we use matrix division in np, each step is slower than it should be
                        # but it's ok because we only use it for metric evaluation, and NEVER use it for actual timing
                        # when in optimization this is only a 1D vector division, so the extra overhead should be minimal
                        ans /= np.expand_dims(self.args_range, (0, 1))
                    
                if render_kw.get('copy_reduce', False):

                    assert not render_kw.get('copy_output', True)

                    width, height = get_wh(render_kw)

                    ans = []
                    
                    out_idx = base_idx + (SPSA_samples) % 2
                    
                    if 'reduce_buffer' not in self.FD_info[lib_type].keys():
                        self.FD_info[lib_type]['reduce_buffer'] = {}
                        
                        assert 'reduce' in buffer_info[out_idx].keys()
                        
                        current_start = buffer_info[out_idx]['reduce'][0]
                        current_buf = so_lib.Buffer((buffer_info[out_idx]['reduce'][1],))
                        
                        self.FD_info[lib_type]['reduce_buffer'] = {'start': current_start,
                                                                   'buf': current_buf,
                                                                   'arr': np.asarray(current_buf)}
                        
                    so_lib.reduce_sum(self.FD_info[lib_type]['reduce_buffer']['start'],
                                      width, height,
                                      render_kw['buffer'][out_idx],
                                      self.FD_info[lib_type]['reduce_buffer']['buf'])
                    
                    self.FD_info[lib_type]['reduce_buffer']['buf'].copy_to_host()
                    
                    ans = self.FD_info[lib_type]['reduce_buffer']['arr'] / self.args_range
                    
                    return ans, None
                    
        else:
            
            collected_ans = None
            
            offset = np.zeros(self.nargs)
            offset_sigmas = np.zeros(len(sigmas))
            
            if len(sigmas) > 0:
                params_and_offsets = [params, offset, sigmas, offset_sigmas]
            else:
                params_and_offsets = [params, offset]

            
            def lib_wrapper(buf):
                
                nonlocal extra_args, finite_diff_h, frame_idx, offset, offset_sigmas, params_and_offsets
                
                offset_sigmas[:] = 0
         
                for idx in range(self.nargs):
                    
                    offset[:] = 0
                    offset[idx] = 1
                    
                    if normalize:
                        # gradient wrt NORMALIZED parameter
                        offset[idx] *= self.args_range[idx]
                        
                    if per_pixel_offset:
                        buf_args = [buf[0], buf[offset_idx], buf[base_idx + idx], buf[base_idx + idx]]
                    else:
                        buf_args = [buf[0], buf[base_idx + idx], buf[base_idx + idx]]
                    
                    current_lib(*buf_args, 
                                *params_and_offsets,
                                *extra_args,
                                finite_diff_h, 1.,
                                frame_idx = frame_idx,
                                output_base_only = True,
                                add_to_old = False)
                    
                if len(sigmas) > 0:
                    offset[:] = 0
                    
                    for idx in range(len(sigmas)):
                        offset_sigmas[:] = 0
                        if self.normalized_par:
                            offset_sigmas[idx] = self.sigmas_scale * self.sigmas_range[render_kw['sigmas_idx'][idx]]
                        else:
                            offset_sigmas[idx] = 1
                        
                        current_lib(buf[0], buf[offset_idx], 
                                    buf[base_idx + self.nargs + idx], buf[base_idx + self.nargs + idx], 
                                    *params_and_offsets,
                                    *extra_args,
                                    finite_diff_h, 1.,
                                    frame_idx = frame_idx,
                                    output_base_only = True,
                                    add_to_old = False)
                return
            
            ans = self.run(lib_wrapper, buffer_info, params, use_lib_wrapper=True, **render_kw)
            
            if init_buf:
                set_buffer(render_kw, self.FD_info[lib_type])
            
            collected_ans = []
            
            if not is_profile:
                
                if render_kw.get('copy_output', True):
                    
                    assert len(ans) == self.nargs + len(sigmas)

                    for val in ans:
                        collected_ans.append(np.asarray(val))

                    ans = collected_ans
                
                if render_kw.get('copy_reduce', False):

                    assert not render_kw.get('copy_output', True)

                    width, height = get_wh(render_kw)

                    ans = []

                    if 'reduce_buffer' not in self.FD_info[lib_type].keys():

                        self.FD_info[lib_type]['reduce_buffer'] = []

                        for idx in range(len(buffer_info)):
                            if 'reduce' in buffer_info[idx].keys():
                                current_start = buffer_info[idx]['reduce'][0]
                                current_buf = so_lib.Buffer((buffer_info[idx]['reduce'][1],))
                                new_info = {'start': current_start,
                                            'orig_idx': idx,
                                            'buf': current_buf,
                                            'arr': np.asarray(current_buf)}
                                self.FD_info[lib_type]['reduce_buffer'].append(new_info)

                    for idx in range(len(self.FD_info[lib_type]['reduce_buffer'])):

                        info = self.FD_info[lib_type]['reduce_buffer'][idx]

                        so_lib.reduce_sum(info['start'],
                                          width, height,
                                          render_kw['buffer'][info['orig_idx']],
                                          info['buf'])

                        info['buf'].copy_to_host()

                        ans.append(info['arr'])
                        
                    if len(sigmas) > 0:
                        deriv = np.concatenate(ans[:self.nargs])
                        deriv_sigmas = np.concatenate(ans[self.nargs:])
                    else:
                        deriv = np.concatenate(ans)
                        deriv_sigmas = None
                    
                    return deriv, deriv_sigmas


        if not is_profile:
            if visualize:
                visualize_buffer(ans, False, path=path, name=name)
        
        
            
        return ans
    
    

    def forward(self, params, path='', name='', visualize=False, transpose=False, reuse_buffer=True, render_kw={}, slice_col=False):
        
        assert reuse_buffer
        
        if params.size > self.nargs:
            assert params.size == self.nargs + render_kw['sigmas_idx'].size
            render_kw['sigmas'] = params[self.nargs:]
            params = params[:self.nargs]
            
        if self.normalized_par:
            params = params * self.args_range
        
        per_pixel_offset = False
        if 'uv_sample' in render_kw.keys():
            uv_sample = render_kw['uv_sample']
            
            if isinstance(uv_sample, np.ndarray) and uv_sample.size > 2:
                per_pixel_offset = True
                render_kw['uv_sample_buffer_idx'] = self.fw_uv_sample_buffer_idx
                
        if render_kw.get('seperate_producer', False):
            func_name = 'producer'
            info_name = 'producer_info'
        else:
            func_name = 'forward'
            info_name = 'fw_info'
                
        lib_type = self.infer_lib_type(func_name, render_kw)
                
        current_lib = getattr(self, info_name)[lib_type]['lib']
        buffer_info = getattr(self, info_name)[lib_type]['buffer_info']
        
        assert current_lib is not None, "Error! Shader's forward kernel NOT specified!"
        
        width, height = get_wh(render_kw)
        
        buffer = None
        init_buf = False
        
        if reuse_buffer:
            init_buf = get_buffer(render_kw, getattr(self, info_name)[lib_type])
                
        ans = self.run(current_lib, buffer_info, params, **render_kw)
        
        if init_buf:
            set_buffer(render_kw, getattr(self, info_name)[lib_type])

        if 'is_profile' in render_kw.keys():
            is_profile = render_kw['is_profile']
        else:
            is_profile = False
        
        if not is_profile:
            if render_kw.get('copy_output', True):
                assert len(ans) == 1
                
                output = np.asarray(ans[0])
                
                if len(output.shape) > 3:
                    output = output[..., -1]
                    
                if slice_col and len(output.shape) == 3 and output.shape[2] > 3:
                    output = output[..., :3]

                if visualize:
                    visualize_buffer(output, True, path=path, name=name)

                if transpose:
                    if len(output.shape) == 2:
                        output = output.transpose()
                    else:
                        output = output.transpose((1, 0, 2))

                return output
            else:
                return
        else:
            return ans

    def backward(self, params,
                 reuse_buffer=True, denum_seperate_buffer=True, render_kw={}, config_kw={}):
        
        assert reuse_buffer
        
        if params.size > self.nargs:
            assert params.size == self.nargs + render_kw['sigmas_idx'].size
            render_kw['sigmas'] = params[self.nargs:]
            params = params[:self.nargs]
            
        if self.normalized_par:
            params = params * self.args_range
        
        per_pixel_offset = False
        if 'uv_sample' in render_kw.keys():
            uv_sample = render_kw['uv_sample']
            
            if isinstance(uv_sample, np.ndarray) and uv_sample.size > 2:
                per_pixel_offset = True
                render_kw['uv_sample_buffer_idx'] = self.bw_uv_sample_buffer_idx
                
                if 'choose_u_pl' in render_kw.keys():
                    choose_u_pl = render_kw['choose_u_pl']
                    assert choose_u_pl.shape[:2] == uv_sample.shape[:2] and len(choose_u_pl.shape) == 3 and choose_u_pl.shape[2] == 1

        lib_type = self.infer_lib_type('backward', render_kw)
        
        with_denum = render_kw.get('with_denum', False)
        denum_only = render_kw.get('denum_only', False)
        
        config_kw['with_denum'] = with_denum
        config_kw['denum_only'] = denum_only
        config_kw['per_pixel_offset'] = per_pixel_offset
        config_kw['lib_type'] = lib_type
            
        current_lib = self.bw_info[lib_type]['lib']
        buffer_info = self.bw_info[lib_type]['buffer_info']
        
        assert current_lib is not None, "Error! Shader's forward kernel NOT specified!"
        
        buffer = None
        init_buf = False
        if reuse_buffer:
            init_buf = get_buffer(render_kw, self.bw_info[lib_type])
            
        ans = self.run(current_lib, buffer_info, params, **render_kw)
        
        if init_buf:
            set_buffer(render_kw, self.bw_info[lib_type])
            
        if render_kw.get('copy_reduce', False):
            
            assert not render_kw.get('copy_output', True)
            
            width, height = get_wh(render_kw)
            
            ans = []
            
            if 'reduce_buffer' not in self.bw_info[lib_type].keys():
                
                self.bw_info[lib_type]['reduce_buffer'] = []
                
                for idx in range(len(buffer_info)):
                    if 'reduce' in buffer_info[idx].keys():
                        current_start = buffer_info[idx]['reduce'][0]
                        current_buf = so_lib.Buffer((buffer_info[idx]['reduce'][1],))
                        new_info = {'start': current_start,
                                    'orig_idx': idx,
                                    'buf': current_buf,
                                    'arr': np.asarray(current_buf)}
                        self.bw_info[lib_type]['reduce_buffer'].append(new_info)

            for idx in range(len(self.bw_info[lib_type]['reduce_buffer'])):

                info = self.bw_info[lib_type]['reduce_buffer'][idx]

                so_lib.reduce_sum(info['start'],
                                  width, height,
                                  render_kw['buffer'][info['orig_idx']],
                                  info['buf'])
                
                info['buf'].copy_to_host()
                
                ans.append(info['arr'])

        return ans
        
    def run(self, lib, buffer_info, params,
            sigmas=[],
            do_prune=[],
            use_lib_wrapper=False,
            width=960, height=640, 
            buffer=None, 
            is_profile=False, profile_n=1000, profile_with_copy=False,
            copy_output=True,
            tile_offset=[0, 0], render_size=None,
            uv_offset=[0, 0],
            with_denum=False,
            uv_sample=None, uv_sample_buffer_idx=None,
            choose_u_pl=None,
            reset_min=True,
            compute_producer=True,
            frame_idx=0,
            sigmas_idx=[],
            # Dummy arguments
            seperate_producer=False,
            copy_reduce=False,
            denum_only=False,
            finite_diff_h=0.01, 
            SPSA_samples=-1,
            weight_map=None,
            verbose=True):
        
        assert so_loaded, 'Error! so file not loaded! Please call load_so(path) where path is the directory storing the so file!'
        
        if len(sigmas) > 0:
            if self.normalized_par:
                sigmas = sigmas * self.sigmas_scale * self.sigmas_range[sigmas_idx]
            param_and_sigma = [params, sigmas]
        else:
            param_and_sigma = [params]
            
        if len(do_prune) > 0:
            if len(param_and_sigma) == 2:
                # overwrite sigmas
                param_and_sigma[-1] = do_prune
            else:
                param_and_sigma.append(do_prune)
            
        if buffer is not None and len(buffer) > 0:
            assert len(buffer) == len(buffer_info), 'Error! Number of buffer provided is incorrect!'
            
            if False:
                # NOTE: HalideBuffer doesn't have attribute for shape information
                for i in range(len(buffer)):

                    assert len(buffer[i].shape) == buffer_info[i]['ndim'], \
                    'Error! buffer[%d] has incorrect dimension!' % i

                    assert buffer[i].shape[-1] == buffer_info[i]['nfeats'], \
                    'Error! buffer[%d] has incorrect number of features!' % i

                    if buffer_info[i]['ndim'] == 4:
                        assert buffer[i].shape[-2] == buffer_info[i]['ncols'], \
                        'Error! buffer [%d] has incorrect number of colors!' % i
                    
        else:
            if verbose:
                print("reset_min set to True for newly created buffers")
            
            reset_min = True
            
            if buffer is None:
                buffer = []
            
            if render_size is None:
                render_width = width
                render_height = height
            else:
                assert len(render_size) == 2
                render_width = render_size[0]
                render_height = render_size[1]
            
            for i in range(len(buffer_info)):
                
                assert buffer_info[i]['ndim'] in [2, 3, 4], \
                'Error! only support buffer[%d] with dimension 2, 3 or 4!' % i
                                
                if 'pad' in buffer_info[i].keys():
                    buf_dim = (render_width + buffer_info[i]['pad'] * 2, render_height + buffer_info[i]['pad'] * 2)
                else:
                    buf_dim = (render_width, render_height)
                    
                if buffer_info[i]['ndim'] > 2:
                    if buffer_info[i]['ndim'] == 4:
                        buf_dim += (buffer_info[i]['ncols'],)
                    buf_dim += (buffer_info[i]['nfeats'],)    
                    
                buf_func = None
                dtype = float
                if 'dtype' in buffer_info[i].keys():
                    
                    dtype = buffer_info[i]['dtype']
                    
                    assert dtype in [int, float, bool]
                    
                    if dtype == int:
                        buf_func = so_lib.Buffer_i
                    elif dtype == bool:
                        buf_func = so_lib.Buffer_b
                    else:
                        buf_func = so_lib.Buffer
                else:
                    buf_func = so_lib.Buffer
                
                buf = buf_func(buf_dim)
                
                if buffer_info[i]['type'] == 'input' and 'default_val' in buffer_info[i].keys():
                    arr = np.asarray(buf)
                    arr[:] = buffer_info[i]['default_val']
                    buf.set_host_dirty()
                
                if verbose:
                    print("Instantiated buffer %d with shape " % i, buf_dim)
                
                buffer.append(buf)
            
        if reset_min:
            for i in range(len(buffer_info)):

                additional_pad = 0

                if 'pad' in buffer_info[i].keys():
                    additional_pad = buffer_info[i]['pad']

                buffer[i].device_free()
                buffer[i].set_min((-additional_pad, -additional_pad));

                if buffer_info[i]['type'] == 'input' and 'default_val' in buffer_info[i].keys():
                    buffer[i].set_host_dirty()
                
        per_pixel_offset = False
                
        if uv_sample is not None:
            if isinstance(uv_sample, list):
                assert len(uv_sample) == 2
                uv_offset = uv_sample
            else:
                assert isinstance(uv_sample, np.ndarray)
                
                if uv_sample.size == 2:
                    uv_offset = uv_sample.flat
                else:
                    assert uv_sample_buffer_idx is not None
                    arr_buf = np.asarray(buffer[uv_sample_buffer_idx])
                    assert uv_sample.shape == arr_buf.shape
                    arr_buf[:] = uv_sample[:]
                    buffer[uv_sample_buffer_idx].set_host_dirty()
                    
                    per_pixel_offset = True
                    
                    if choose_u_pl is not None:
                        assert self.bw_choose_u_pl_buffer_idx is not None
                        arr_buf = np.asarray(buffer[self.bw_choose_u_pl_buffer_idx])
                        assert choose_u_pl.shape == arr_buf.shape
                        arr_buf[:] = choose_u_pl[:]
                        buffer[self.bw_choose_u_pl_buffer_idx].set_host_dirty()

        if is_profile:

            # NOTE: we will report profile runtime directly collected in c++, using Halide_profile_generator.py
            # to avoid any possible overhead in the python binding
            # The method here simply serves as a sanity check 
            # to make sure the runtime between python and C++ do not differ too much
            # We disable copying data from device to host and from host to device when profiling
            
            best = 1e8
            for _ in range(10):
                T0 = time.time()
                for _ in range(profile_n):
                    if use_lib_wrapper:
                        lib(buffer)
                    else:
                        lib(*buffer, *param_and_sigma, uv_offset[0], uv_offset[1], width, height, 
                            frame_idx=frame_idx,
                            compute_producer=compute_producer,
                            with_denum=with_denum)
                T1 = time.time()
                best = min(best, T1 - T0)
                
            ns_per_pix = 1e9 * best / (profile_n * width * height)
            
            print('Profiling for %d iterations: %f ns per pixel' % (profile_n, ns_per_pix))
            
            return ns_per_pix
        
        else:
            if use_lib_wrapper:
                lib(buffer)
            else:
                # DOGE: debug only
                lib(*buffer, *param_and_sigma, uv_offset[0] + tile_offset[0], uv_offset[1] + tile_offset[1], width, height, 
                    frame_idx=frame_idx,
                    compute_producer=compute_producer,
                    with_denum=with_denum)

            if copy_output:
                out_buffer = []
                for i in range(len(buffer)):
                    if buffer_info[i]['type'] == 'output':
                        buffer[i].copy_to_host()
                        
                        current_arr = np.asarray(buffer[i])
                        
                        if 'pad' in buffer_info[i].keys():
                            pad = buffer_info[i]['pad']
                            current_arr = current_arr[pad:-pad, pad:-pad]
                        
                        out_buffer.append(current_arr)

                return out_buffer
            else:
                return
            
class CompilerProblem(GenericShader):
    def __init__(self):
        super().__init__()
        
        assert so_loaded, 'Error! so file not loaded! Please call load_so(path) where path is the directory storing the so file!'
        assert compiler_problem_loaded, 'Error! compiler problem not loaded! Please call load_so(compiler_problem_path=path) where path is the directory where compiler problem is compiled!'
        
        if compiler_problem_scalar_loaded:
            self.scalar_loss = compiler_problem_scalar_lib
        else:
            self.scalar_loss = None
        self.scalar_loss_scale = 1
        
        with open(compiler_problem_lib.get_dict_pickle_file(), 'rb') as f:
            buffer_info = pickle.load(f)
            
        self.fw_info = buffer_info[0]
        self.bw_info = buffer_info[1]
        self.producer_info = buffer_info[2]
        
        self.fw_info['regular']['lib'] = compiler_problem_lib.fw
        self.fw_info['per_pixel_offset']['lib'] = compiler_problem_lib.fw_per_pixel_offset
        self.fw_info['prune_updates'] = {'lib': compiler_problem_lib.fw_prune_updates,
                                         'buffer_info': self.fw_info['regular']['buffer_info']}
        self.producer_info['regular']['lib'] = compiler_problem_lib.producer
        self.bw_info['regular']['lib'] = compiler_problem_lib.bw
        self.bw_info['per_pixel_offset']['lib'] = compiler_problem_lib.bw_per_pixel_offset
        self.bw_info['denum_only']['lib'] = compiler_problem_lib.bw_denum_only
        self.bw_info['offset_choose_u_pl']['lib'] = compiler_problem_lib.bw_choose_u_pl
        
        
        
        self.fw_uv_sample_buffer_idx = 0
        self.bw_uv_sample_buffer_idx = 1
        self.bw_choose_u_pl_buffer_idx = 2
        
        self.discont_idx = np.asarray(compiler_problem_lib.get_discont_idx())

        self.nargs = compiler_problem_lib.get_nargs()
        self.args_range = np.asarray(compiler_problem_lib.get_args_range())
        self.sigmas_range = np.asarray(compiler_problem_lib.get_sigmas_range())
        
        self.n_updates = compiler_problem_lib.get_n_optional_updates()
        
        self.bw_map = np.arange(self.nargs)
        
        self.FD_lib = compiler_problem_lib.FD
        self.FD_per_pixel_offset_lib = compiler_problem_lib.FD_per_pixel_offset
        self.update_FD_info()
        
        self.fw_info['random_noise_' + '_'.join(['%d' % idx for idx in self.discont_idx])] = \
        {'lib': compiler_problem_lib.fw_random_par}
        
        self.producer_info['random_noise_' + '_'.join(['%d' % idx for idx in self.discont_idx])] = \
        {'lib': compiler_problem_lib.producer_random_par}
        
        self.bw_info['random_noise_' + '_'.join(['%d' % idx for idx in self.discont_idx])] = \
        {'lib': compiler_problem_lib.bw_random_par}
        
        self.FD_info['random_noise_' + '_'.join(['%d' % idx for idx in self.discont_idx])] = \
        {'lib': compiler_problem_lib.FD_random_par}
        
        self.forward = lambda *args, **kwargs: GenericShader.forward(self, *args, **kwargs, slice_col=True)
        
    def check_ok(self, mode):
        
        is_FD = False
        
        if mode.startswith('fw'):
            info = self.fw_info
        elif mode.startswith('producer'):
            info = self.producer_info
        elif mode.startswith('bw'):
            info = self.bw_info
        else:
            info = self.FD_info
            is_FD = True
            
        if mode.endswith('random_par'):
            orig_lib_type = 'regular'
            lib_type = 'random_noise_' + '_'.join(['%d' % idx for idx in self.discont_idx])
        elif mode.endswith('per_pixel_offset'):
            orig_lib_type = 'per_pixel_offset'
            lib_type = orig_lib_type
        elif mode.endswith('denum_only'):
            orig_lib_type = 'denum_only'
            lib_type = orig_lib_type
        elif mode.endswith('choose_u_pl'):
            orig_lib_type = 'offset_choose_u_pl'
            lib_type = orig_lib_type
        elif mode.endswith('prune_updates'):
            lib_type = 'prune_updates'
            orig_lib_type = 'regular'
        else:
            orig_lib_type = 'regular'
            lib_type = orig_lib_type
            
        param_and_sigma = [np.zeros(self.nargs)]
        if is_FD:
            param_and_sigma.append(param_and_sigma[-1])
        if lib_type.startswith('random_noise'):
            param_and_sigma.append(np.zeros(self.discont_idx.shape[0]))
            if is_FD:
                param_and_sigma.append(param_and_sigma[-1])
        if lib_type.endswith('prune_updates'):
            param_and_sigma.append(np.zeros(self.n_updates).astype(bool))
            
        if not hasattr(self, 'dummy_buf'):
            self.dummy_buf = so_lib.Buffer((1,))
            
        if not hasattr(self, 'dummy_buf_bool'):
            self.dummy_buf_bool = so_lib.Buffer_b((1,))
            
        if not is_FD:
            args_buf = [self.dummy_buf] * len(info[orig_lib_type]['buffer_info'])
            if mode.endswith('choose_u_pl'):
                args_buf[self.bw_choose_u_pl_buffer_idx] = self.dummy_buf_bool
            
            ans = info[lib_type]['lib'](
                *args_buf,
                *param_and_sigma,
                0, 0, 960, 640,
                check_ok=True)
        else:
            if 'base_only' in mode:
                output_base_only = True
            else:
                output_base_only = False
                
            if 'add_to' in mode:
                add_to = True
            else:
                add_to = False
                
            if 'per_pixel_offset' in mode:
                nbufs = 4
                current_lib = self.FD_per_pixel_offset_lib
            else:
                nbufs = 3
                current_lib = info.get(lib_type, {}).get('lib', self.FD_lib)
                
            ans = current_lib(
                *([self.dummy_buf] * nbufs),
                *param_and_sigma,
                0, 0, 960, 640,
                0.01, 1,
                output_base_only=output_base_only,
                add_to_old=add_to,
                check_ok=True)
            
        return ans
    
    def backward(self, params, path='', name='', visualize=False, reuse_buffer=True, render_kw={}):
        
        assert reuse_buffer
        
        config_kw = {}

        ans = super().backward(params, reuse_buffer=reuse_buffer, denum_seperate_buffer=False, 
                               render_kw=render_kw, config_kw=config_kw)
        
        with_denum = config_kw['with_denum']
        denum_only = config_kw['denum_only']
        lib_type = config_kw['lib_type']
        
        is_profile = render_kw.get('is_profile', False)
        
        sigmas = render_kw.get('sigmas', [])
        
        if not is_profile:
            out_idx = 0
            deriv_params = None
            if denum_only:
                assert render_kw.get('copy_output', True) and not render_kw.get('copy_reduce', False)
                assert len(ans) == 1
                return None, ans[0]
            elif render_kw.get('copy_output', True):
                render_width = render_kw.get('render_size', [render_kw['width'], render_kw['height']])[0]
                render_height = render_kw.get('render_size', [render_kw['width'], render_kw['height']])[1]
                deriv = np.empty((render_width, render_height, self.nargs))
            elif render_kw.get('copy_reduce', False):
                assert not denum_only
                deriv = np.empty(self.nargs)
                if lib_type.startswith('random_noise'):
                    deriv_params = np.empty(self.discont_idx.shape[0])
            else:
                return
            
            for info in self.bw_info[lib_type]['buffer_info']:
                if info['type'] == 'output':
                    if 'orig_reduce' in info.keys():
                        orig_reduce = info['orig_reduce']
                    else:
                        orig_reduce = info['reduce']
                        
                    if render_kw.get('copy_reduce', False):
                        # if copy_reduce, ans is already trimmed and contains derivative only
                        deriv_slice = slice(0, orig_reduce[1])
                        deriv_params_slice = slice(orig_reduce[1], None)
                    else:
                        deriv_slice = slice(orig_reduce[0], orig_reduce[0] + orig_reduce[1])
                        deriv_params_slice = slice(orig_reduce[0] + orig_reduce[1], None)
                                                              
                    deriv[..., info['par_idx']] = ans[out_idx][..., deriv_slice]
                    
                    if deriv_params is not None:
                        
                        if 'discont_map' not in info.keys():
                            discont_map = []
                            for par_idx in info['par_idx']:
                                if par_idx in self.discont_idx:
                                    discont_map.append(self.discont_idx.tolist().index(par_idx))
                            info['discont_map'] = discont_map
                            
                        deriv_params[info['discont_map']] = ans[out_idx][..., deriv_params_slice]
                    out_idx += 1
                    
            return deriv, deriv_params
        else:
            return ans
        
class Olympic80(GenericShader):
    def __init__(self):
        super().__init__()
        
        assert so_loaded, 'Error! so file not loaded! Please call load_so(path) where path is the directory storing the so file!'
        
        self.fw_info = {'regular': 
                        {'lib': so_lib.olympic80_fw,
                         'buffer_info': [{'ndim': 3, 'nfeats': 2, 'type': 'input', 'default_val': 0},
                                         {'ndim': 3, 'nfeats': 3, 'type': 'output', 'tag': 'col'}]},
                        'random_noise_40_41_42_43_44_45_46_47_48_49':
                        {'lib': so_lib.olympic80_fw_random_tilt},
                        'random_noise_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_25_26_27_28_29_30_31_32_33_34_35_36_37_38_39_40_41_42_43_44_45_46_47_48_49':
                        {'lib': so_lib.olympic80_fw_random_exclude_col}
                       }
        
        self.bw_info = {'regular': 
                        {'lib': so_lib.olympic80,
                         'buffer_info': [{'ndim': 3, 'nfeats': 3, 'type': 'input', 'default_val': 1, 'tag': 'dL_dcol'},
                                         {'ndim': 3, 'nfeats': 32, 'type': 'output', 'reduce': (2, 30),
                                          'par_idx': np.arange(50, 80)},
                                         {'ndim': 3, 'nfeats': 2, 'type': 'intermediate'},
                                         {'ndim': 3, 'nfeats': 30, 'type': 'output', 'reduce': (0, 25),
                                          'par_idx': np.concatenate([10 * i + np.arange(9, 4, -1) for i in range(5)])},
                                         {'ndim': 3, 'nfeats': 25, 'type': 'output', 'reduce': (0, 25),
                                          'par_idx': np.concatenate([10 * i + np.arange(4, -1, -1) for i in range(5)])},
                                         {'ndim': 2, 'type': 'intermediate', 'dtype': int}]},
                        'random_noise_40_41_42_43_44_45_46_47_48_49': 
                        {'lib': so_lib.olympic80_random_tilt},
                        'random_noise_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_25_26_27_28_29_30_31_32_33_34_35_36_37_38_39_40_41_42_43_44_45_46_47_48_49':
                        {'lib': so_lib.olympic80_random_exclude_col},
                        'with_denum': 
                        {'lib': so_lib.olympic80,
                         'buffer_info': [{'ndim': 3, 'nfeats': 3, 'type': 'input', 'default_val': 1, 'tag': 'dL_dcol'},
                                         {'ndim': 3, 'nfeats': 32, 'type': 'output', 'reduce': (2, 30)},
                                         {'ndim': 3, 'nfeats': 2, 'type': 'intermediate'},
                                         {'ndim': 3, 'nfeats': 30, 'type': 'output', 'reduce': (0, 25)},
                                         {'ndim': 3, 'nfeats': 25, 'type': 'output', 'reduce': (0, 25)},
                                         {'ndim': 2, 'type': 'output', 'dtype': int}]},
                        'denum_only': 
                        {'lib': so_lib.olympic80_denum_only,
                         'buffer_info': [{'ndim': 3, 'nfeats': 3, 'type': 'input', 'default_val': 1, 'tag': 'dL_dcol'},
                                         {'ndim': 3, 'nfeats': 1, 'type': 'intermediate'},
                                         {'ndim': 3, 'nfeats': 1, 'type': 'intermediate'},
                                         {'ndim': 3, 'nfeats': 1, 'type': 'output'}]},
                        'per_pixel_offset': 
                        {'lib': so_lib.olympic80_per_pixel_offset,
                         'buffer_info': [{'ndim': 3, 'nfeats': 3, 'type': 'input', 'default_val': 1, 'tag': 'dL_dcol'},
                                         {'ndim': 3, 'nfeats': 2, 'type': 'input', 'default_val': 0},
                                         {'ndim': 3, 'nfeats': 32, 'type': 'output', 'reduce': (2, 30)},
                                         {'ndim': 3, 'nfeats': 2, 'type': 'intermediate'},
                                         {'ndim': 3, 'nfeats': 30, 'type': 'output', 'reduce': (0, 25)},
                                         {'ndim': 3, 'nfeats': 25, 'type': 'output', 'reduce': (0, 25)},
                                         {'ndim': 2, 'type': 'intermediate', 'dtype': int}]},
                        'offset_choose_u_pl': 
                        {'lib': so_lib.olympic80_per_pixel_offset_choose_u_pl,
                         'buffer_info': [{'ndim': 3, 'nfeats': 3, 'type': 'input', 'default_val': 1, 'tag': 'dL_dcol'},
                                         {'ndim': 3, 'nfeats': 2, 'type': 'input', 'default_val': 0},
                                         {'ndim': 2, 'type': 'input', 'dtype': bool, 'default_val': False},
                                         {'ndim': 3, 'nfeats': 31, 'type': 'output', 'reduce': (1, 30)},
                                         {'ndim': 3, 'nfeats': 30, 'type': 'output', 'reduce': (0, 25)},
                                         {'ndim': 3, 'nfeats': 25, 'type': 'output', 'reduce': (0, 25)}]}
                       }
        
        
                        
        self.FD_lib = so_lib.olympic80_FD

        self.fw_uv_sample_buffer_idx = 0
        self.bw_uv_sample_buffer_idx = 1
        self.bw_choose_u_pl_buffer_idx = 2
                                                 
        self.nargs = 80
        self.update_FD_info()
        
        self.FD_info['random_noise_40_41_42_43_44_45_46_47_48_49'] = {'lib': so_lib.olympic80_FD_random_tilt}
        self.FD_info['random_noise_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_25_26_27_28_29_30_31_32_33_34_35_36_37_38_39_40_41_42_43_44_45_46_47_48_49'] = {'lib': so_lib.olympic80_FD_random_exclude_col}
        
        self.params_reorder = np.concatenate((np.array([4, 3, 2, 1, 0,
                                                        14, 13, 12, 11, 10,
                                                        24, 23, 22, 21, 20,
                                                        34, 33, 32, 31, 30,
                                                        44, 43, 42, 41, 40,
                                                        9, 8, 7, 6, 5,
                                                        19, 18, 17, 16, 15,
                                                        29, 28, 27, 26, 25,
                                                        39, 38, 37, 36, 35,
                                                        49, 48, 47, 46, 45,
                                                       ]),
                                              np.arange(50, 80)))
        self.args_range = np.ones(80)
        # pos_x
        self.args_range[:10] = 960
        #self.args_range[:10] = 100
        # pos_y
        self.args_range[10:20] = 480
        #self.args_range[10:20] = 100
        # radius
        self.args_range[20:30] = 200
        # radius_scale
        self.args_range[30:40] = 1
        # tilt
        self.args_range[40:50] = 1
        # col
        self.args_range[50:] = 1
        
        
        self.sigmas_range = np.ones(80)
        # pos_x
        self.sigmas_range[:10] = 20
        # pos_y
        self.sigmas_range[10:20] = 20
        # radius
        self.sigmas_range[20:30] = 10
        # radius_scale
        self.sigmas_range[30:40] = 0.1
        # tilt
        self.sigmas_range[40:50] = 1
        # col
        self.sigmas_range[50:] = 1
        
        #self.sigmas_range = self.args_range
        
        self.sample_range = np.ones((80, 2))
        # pos_x
        self.sample_range[:10, 0] = -960 * 0.2
        self.sample_range[:10, 1] = 960 + 960 * 0.2
        # pos_y
        self.sample_range[10:20, 0] = -480 * 0.2
        self.sample_range[10:20, 1] = 480 + 480 * 0.2
        # radius
        self.sample_range[20:30, 0] = -200 * 0.2
        self.sample_range[20:30, 1] = 480
        # radius_scale
        self.sample_range[30:40, 0] = 0
        self.sample_range[30:40, 1] = 1
        # tilt
        self.sample_range[40:50, 0] = -1
        self.sample_range[40:50, 1] = 1
        # col
        self.sample_range[50:, 0] = 0
        self.sample_range[50:, 1] = 1
        
        self.bw_map = np.concatenate((np.array([4, 3, 2, 1, 0, 29, 28, 27, 26, 25,
                                                9, 8, 7, 6, 5, 34, 33, 32, 31, 30,
                                                14, 13, 12, 11, 10, 39, 38, 37, 36, 35,
                                                19, 18, 17, 16, 15, 44, 43, 42, 41, 40,
                                                24, 23, 22, 21, 20, 49, 48, 47, 46, 45]),
                                      50 + np.arange(30)))
        
        #self.random_var_indices = np.arange(40, 50)
        self.random_var_indices = np.arange(50)

    def backward(self, params, path='', name='', visualize=False, reuse_buffer=True, render_kw={}):
        
        assert reuse_buffer
        
        config_kw = {}

        ans = super().backward(params, reuse_buffer=reuse_buffer, denum_seperate_buffer=False, 
                               render_kw=render_kw, config_kw=config_kw)
        
        with_denum = config_kw['with_denum']
        denum_only = config_kw['denum_only']
        lib_type = config_kw['lib_type']
        
        is_profile = render_kw.get('is_profile', False)
        
        sigmas = render_kw.get('sigmas', [])
        
        if not is_profile:
            if render_kw.get('copy_output', True):
                # piecing up gradients to be compatible with tf shaders
                
                gradient_tilt = None
                
                if with_denum:
                    assert len(ans) == 4
                    choose_u = np.asarray(ans[3]) >= 16
                elif denum_only:
                    assert len(ans) == 1
                    choose_u = np.asarray(ans[0]).astype(bool)[..., 0]
                    gradient = None
                else:
                    assert len(ans) == 3
                    choose_u = None
                    
                if not denum_only:
                    gradient_AD = np.asarray(ans[0])[..., 2:32]
                    gradient_last_5 = np.asarray(ans[1])[..., :25]
                    gradient_first_5 = np.asarray(ans[2])[..., :25]

                    gradient = np.concatenate((gradient_first_5, gradient_last_5, gradient_AD), -1)
                    #gradient = gradient[:, :, 0, :]
                    
                    if lib_type.startswith('random_noise'):
                        gradient_sigmas_AD = np.asarray(ans[0])[..., 32:]
                        gradient_sigmas_last_5 = np.asarray(ans[1])[..., 25:-5]
                        gradient_sigmas_first_5 = np.asarray(ans[2])[..., 25:]
                        
                        gradient_tilt = np.concatenate((gradient_sigmas_AD, gradient_sigmas_last_5, gradient_sigmas_first_5), -1)

                    if visualize:
                        visualize_buffer(gradient, path=path, name=name, map_ordering=self.bw_map)
                        
                        if lib_type.startswith('random_noise'):
                            visualize_buffer(gradient_tilt, path=path, name=name + '_sigma', 
                                             map_ordering=self.bw_info[lib_type]['sigmas_idx_order'])

                if visualize and choose_u is not None:
                    visualize_buffer(choose_u.astype('f'), single_img=True, path=path, name='%s_denum' % name)

                return gradient, choose_u
            elif render_kw.get('copy_reduce', False):
                assert not denum_only
                assert len(ans) == 3
                
                deriv_params = np.concatenate((ans[2][:25], ans[1][:25], ans[0][:30]))[self.bw_map]
                
                if lib_type.startswith('random_noise'):
                    sigma_params = np.concatenate((ans[0][30:], ans[1][25:], ans[2][25:]))[self.bw_info[lib_type]['sigmas_idx_order']]
                else:
                    sigma_params = None
                
                return deriv_params, sigma_params
            else:
                return
        else:
            return ans
            
class SiggraphLighting(GenericShader):
    def __init__(self):
        super().__init__()
        
        assert so_loaded, 'Error! so file not loaded! Please call load_so(path) where path is the directory storing the so file!'
        
        self.fw_info = {'regular': 
                        {'lib': so_lib.siggraph_lighting_fw,
                         'buffer_info': [{'ndim': 3, 'nfeats': 2, 'type': 'input', 'default_val': 0},
                                         {'ndim': 3, 'nfeats': 3, 'type': 'output', 'tag': 'col'}]},
                        'producer': 
                        {'lib': so_lib.siggraph_lighting_fw_producer,
                         'buffer_info': [{'ndim': 3, 'nfeats': 13, 'type': 'output', 'tag': 'col', 'pad': 1}]},
                        'random_noise_16_17_18_19_20_21_22_23_24_25':
                        {'lib': so_lib.siggraph_lighting_fw_random_light},
                        'producer_random_noise_16_17_18_19_20_21_22_23_24_25':
                        {'lib': so_lib.siggraph_lighting_fw_producer_random_light}
                       }
        
        self.bw_info = {'regular': 
                        {'lib': so_lib.siggraph_lighting,
                         'buffer_info': [{'ndim': 3, 'nfeats': 3, 'type': 'input', 'default_val': 1, 'tag': 'dL_dcol'},
                                         {'ndim': 3, 'nfeats': 13, 'type': 'intermediate', 'pad': 1, 'tag': 'producer'},
                                         {'ndim': 3, 'nfeats': 27, 'type': 'intermediate'},
                                         {'ndim': 3, 'nfeats': 27, 'type': 'intermediate'},
                                         {'ndim': 3, 'nfeats': 27, 'type': 'intermediate'},
                                         {'ndim': 3, 'nfeats': 27, 'type': 'intermediate'},
                                         {'ndim': 3, 'nfeats': 26, 'type': 'output', 'reduce': (0, 26),
                                          'par_idx': np.arange(26)},
                                         {'ndim': 3, 'nfeats': 18, 'type': 'output', 'reduce': (0, 18),
                                          'par_idx': np.arange(26, 44)}]},
                        'random_noise_16_17_18_19_20_21_22_23_24_25':
                        {'lib': so_lib.siggraph_lighting_random_light},
                        'with_denum': 
                        {'lib': so_lib.siggraph_lighting,
                         'buffer_info': [{'ndim': 3, 'nfeats': 3, 'type': 'input', 'default_val': 1, 'tag': 'dL_dcol'},
                                         {'ndim': 3, 'nfeats': 13, 'type': 'intermediate', 'pad': 1, 'tag': 'producer'},
                                         {'ndim': 3, 'nfeats': 27, 'type': 'intermediate'},
                                         {'ndim': 3, 'nfeats': 27, 'type': 'intermediate'},
                                         {'ndim': 3, 'nfeats': 27, 'type': 'intermediate'},
                                         {'ndim': 3, 'nfeats': 27, 'type': 'intermediate'},
                                         {'ndim': 3, 'nfeats': 27, 'type': 'output', 'reduce': (0, 26)},
                                         {'ndim': 3, 'nfeats': 18, 'type': 'output', 'reduce': (0, 18)}]},
                        'denum_only': 
                        {'lib': so_lib.siggraph_lighting_denum_only,
                         'buffer_info': [{'ndim': 3, 'nfeats': 3, 'type': 'input', 'default_val': 1, 'pad': 1, 'tag': 'dL_dcol'},
                                         {'ndim': 3, 'nfeats': 1, 'type': 'intermediate'},
                                         {'ndim': 3, 'nfeats': 1, 'type': 'intermediate'},
                                         {'ndim': 3, 'nfeats': 1, 'type': 'intermediate'},
                                         {'ndim': 3, 'nfeats': 1, 'type': 'intermediate'},
                                         {'ndim': 3, 'nfeats': 1, 'type': 'output'},
                                         {'ndim': 3, 'nfeats': 13, 'type': 'intermediate', 'pad': 1, 'tag': 'producer'}]},
                        'per_pixel_offset': 
                        {'lib': so_lib.siggraph_lighting_per_pixel_offset,
                         'buffer_info': [{'ndim': 3, 'nfeats': 3, 'type': 'input', 'default_val': 1, 'tag': 'dL_dcol'},
                                         {'ndim': 3, 'nfeats': 2, 'type': 'input', 'default_val': 0},
                                         {'ndim': 3, 'nfeats': 13, 'type': 'intermediate'},
                                         {'ndim': 3, 'nfeats': 10, 'type': 'intermediate'},
                                         {'ndim': 3, 'nfeats': 10, 'type': 'intermediate'},
                                         {'ndim': 3, 'nfeats': 10, 'type': 'intermediate'},
                                         {'ndim': 3, 'nfeats': 10, 'type': 'intermediate'},
                                         {'ndim': 3, 'nfeats': 27, 'type': 'intermediate'},
                                         {'ndim': 3, 'nfeats': 27, 'type': 'intermediate'},
                                         {'ndim': 3, 'nfeats': 27, 'type': 'intermediate'},
                                         {'ndim': 3, 'nfeats': 27, 'type': 'intermediate'},
                                         {'ndim': 3, 'nfeats': 26, 'type': 'output', 'reduce': (0, 26)},
                                         {'ndim': 3, 'nfeats': 18, 'type': 'output', 'reduce': (0, 18)}]},
                        'offset_choose_u_pl': 
                        {'lib': so_lib.siggraph_lighting_per_pixel_offset_choose_u_pl,
                         'buffer_info': [{'ndim': 3, 'nfeats': 3, 'type': 'input', 'default_val': 1, 'tag': 'dL_dcol'},
                                         {'ndim': 3, 'nfeats': 2, 'type': 'input', 'default_val': 0},
                                         {'ndim': 2, 'type': 'input', 'dtype': bool, 'default_val': False},
                                         {'ndim': 3, 'nfeats': 13, 'type': 'intermediate'},
                                         {'ndim': 3, 'nfeats': 10, 'type': 'intermediate'},
                                         {'ndim': 3, 'nfeats': 10, 'type': 'intermediate'},
                                         {'ndim': 3, 'nfeats': 10, 'type': 'intermediate'},
                                         {'ndim': 3, 'nfeats': 10, 'type': 'intermediate'},
                                         {'ndim': 3, 'nfeats': 26, 'type': 'intermediate'},
                                         {'ndim': 3, 'nfeats': 26, 'type': 'intermediate'},
                                         {'ndim': 3, 'nfeats': 26, 'type': 'intermediate'},
                                         {'ndim': 3, 'nfeats': 26, 'type': 'intermediate'},
                                         {'ndim': 3, 'nfeats': 26, 'type': 'output', 'reduce': (0, 26)},
                                         {'ndim': 3, 'nfeats': 18, 'type': 'output', 'reduce': (0, 18)}]}
                       }
        
        
        
        self.FD_lib = so_lib.siggraph_lighting_FD

      
        self.fw_uv_sample_buffer_idx = 0
        self.bw_uv_sample_buffer_idx = 1
        self.bw_choose_u_pl_buffer_idx = 2
        
        self.nargs = 44
        self.update_FD_info()
        
        self.FD_info['random_noise_16_17_18_19_20_21_22_23_24_25'] = {'lib': so_lib.siggraph_lighting_FD_random_light}
        
        self.params_reorder = np.arange(self.nargs)
        self.args_range = np.array([0.5] * 8 + [6.28] + [0.1] * 7 + [1] * 28)
        self.sigmas_range = self.args_range
        self.sample_range = np.ones((44, 2))
        # pos
        self.sample_range[:3, 0] = -5
        self.sample_range[:3, 1] = 5
        # ang
        self.sample_range[3:9, 0] = 0
        self.sample_range[3:9, 1] = 2 * np.pi
        # ellipse_ratio
        self.sample_range[9] = [0, 10]
        # ce and thre
        self.sample_range[10:16, 0] = -5
        self.sample_range[10:16, 1] = 5
        # lig ang
        self.sample_range[16:20, 0] = 0
        self.sample_range[16:20, 1] = 2 * np.pi
        # lig pos
        self.sample_range[20:26, 0] = -5
        self.sample_range[20:26, 1] = 5
        # col
        self.sample_range[26:, 0] = 0
        self.sample_range[26:, 1] = 1
        
        self.random_var_indices = np.arange(10) + 16
        
        
        self.bw_map = np.arange(self.nargs)
        
    def backward(self, params, path='', name='', visualize=False, reuse_buffer=True, render_kw={}):
        
        assert reuse_buffer
        
        config_kw = {}
                                  
        ans = super().backward(params, reuse_buffer=reuse_buffer, denum_seperate_buffer=True, 
                               render_kw=render_kw, config_kw=config_kw)
        
        with_denum = config_kw['with_denum']
        denum_only = config_kw['denum_only']
        lib_type = config_kw['lib_type']
        per_pixel_offset = config_kw['per_pixel_offset']
        
        is_profile = render_kw.get('is_profile', False)
        
        if not is_profile:
            if render_kw.get('copy_output', True):
                # piecing up gradients to be compatible with tf shaders
                if denum_only:
                    assert len(ans) == 1
                    choose_u = np.asarray(ans[0]).astype(bool)[..., 0]
                    gradient = None
                else:
                    assert len(ans) == 2

                if not denum_only:
                    gradient_delta = np.asarray(ans[0])[..., :26]

                    if with_denum:
                        choose_u = gradient_delta[..., -1].astype(bool)
                        gradient_delta = gradient_delta[..., :-1]
                    else:
                        choose_u = None

                    gradient_AD = np.asarray(ans[1])
                    gradient = np.concatenate((gradient_delta, gradient_AD), -1)
                    
                    if lib_type.startswith('random_noise'):
                        gradient_sigmas = np.asarray(ans[0])[..., 26:]

                    if visualize:
                        visualize_buffer(gradient, path=path, name=name, map_ordering=self.bw_map)
                        
                        if lib_type.startswith('random_noise'):
                            visualize_buffer(gradient_sigmas, path=path, name=name + '_sigma', 
                                             map_ordering=self.bw_info[lib_type]['sigmas_idx_order'])

                if choose_u is not None and visualize:
                    visualize_buffer(choose_u.astype('f'), single_img=True, path=path, name='%s_denum' % name)

                return gradient, choose_u
            elif render_kw.get('copy_reduce', False):
                assert not denum_only
                assert len(ans) == 2
                
                deriv_params = np.concatenate((ans[0][:26], ans[1]))[self.bw_map]
                
                if lib_type.startswith('random_noise'):
                    sigma_params = ans[0][26:][self.bw_info[lib_type]['sigmas_idx_order']]
                else:
                    sigma_params = None
                    
                return deriv_params, sigma_params
                
                #if not per_pixel_offset:
                #    return np.concatenate((ans[0], ans[1]))[self.bw_map], None
                #else:
                #    return np.concatenate((ans[1], ans[0]))[self.bw_map], None
            else:
                return
        else:
            return ans
        
def visualize_buffer(buffer, single_img=False, name='visualize', path='', map_ordering=None, sparse_idx=None):
    
    if isinstance(buffer, list):
        buffer = np.concatenate(buffer, -1)
    
    if single_img:
        buffer = np.expand_dims(buffer, -1)
        
    if map_ordering is None:
        map_ordering = np.arange(buffer.shape[-1])
        
    for i in range(buffer.shape[-1]):
        
        buffer_idx = map_ordering[i]
        
        val = buffer[..., buffer_idx]
        
        if len(val.shape) == 3:
            val = val.transpose((1, 0, 2))
        else:
            val = val.transpose()
            
        if val.max() != 0 and val.max() > 0:
            val /= np.abs(val.max())
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            if sparse_idx is None:
                output_idx = i
            else:
                output_idx = sparse_idx[i]
            
            skimage.io.imsave(os.path.join(path, '%s%d.png' % (name, output_idx)), val, check_contrast=False)
    
if __name__ == '__main__':
    lib_path = None
    if len(sys.argv) > 1:
        lib_path = sys.argv[1]
        
    load_so(lib_path, verbose=False)
    compiler_module = CompilerProblem()
    
    success = True
    
    if '--modes' in sys.argv:
        modes_idx = sys.argv.index('--modes')
        modes = sys.argv[modes_idx+1].split(',')
        
        for mode in modes:
            if not compiler_module.check_ok(mode):
                success = False
                break
            
    if success:
        print('success')
    else:
        print('fail')