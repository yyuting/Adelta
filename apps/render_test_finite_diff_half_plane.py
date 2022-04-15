"""
------------------------------------------------------------------------------------------------------------------------------
# Visualize gradient

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_half_plane --shader test_finite_diff_half_plane --init_values_pool apps/example_init_values/test_finite_diff_half_plane_init_values_pool.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_col
"""

from render_util import *
from render_single import render_single

compiler.log_prefix_only = False
compiler.log_intermediates_less = True

def cmd_template():
    cmd = f"""python approx_gradient.py --shader test_finite_diff_half_plane --init_values_pool apps/example_init_values/test_finite_diff_half_plane_init_values_pool.npy --metrics 5_scale_L2 --is_col"""
    
    return cmd

nargs = 1
args_range = np.array([600])
args_constrain = [None, None]

approx_mode = '1D_2samples'

use_select_rule = 1

def test_finite_diff_half_plane(u, v, X, width=960, height=640):
    """
    X has shape nargs + 3
    first 3 entries are u, v, time
    the other entries are scalar tunable parameters
    """

    thre = X[0]
    
    cond_diff = Var('cond_diff', u - thre)
    
    cond = Var('cond', cond_diff > 0)
    
    col = Var('col', select(cond, 1, 0))
    
    return output_color([col, col * 0.5, col * 0.2])

shaders = [test_finite_diff_half_plane]
is_color = True

class Gradient:
    def __init__(self, X, trace, extra_args={}):
        self.partial_trace_coord = {'left': {}, 'right': {}}
        self.partial_AB = {}

        self.partial_trace_coord_idx = []
        
        self.partial_trace_coord_idx = list(set(self.partial_trace_coord_idx))
        
        self.register_idx('col_idx', 0)


        self.register_idx('cond_idx', 1)


        self.register_idx('cond_diff_idx', 2)
        
        
        self.X = X
        self.trace = trace
        self.width = extra_args.get('width', 960)
        self.height = extra_args.get('height', 640)
        self.base = None
        self.sess = extra_args.get('sess', None)
        self.deriv_two_sides = extra_args.get('deriv_two_sides', False)
        
        self.gradient_func = {
            (): self.generate_I_var
        }
        
        self.u, self.v, self.time, \
        self.thre = X

        self.calc_partial_trace_coord(sess=self.sess, precompute=False)
        
        self.zeros_pl = tf.zeros_like(self.u)
        self.ones_pl = tf.ones_like(self.u)
        
        self.step_mode = None

        
    def register_idx(self, name, idx):
        if isinstance(idx, list):
            for val in idx:
                if val is not None:
                    self.partial_trace_coord_idx.append(val)
        else:
            self.partial_trace_coord_idx.append(idx)

        setattr(self, name, idx)

    def safe_division(self, a, b, safe_value=None):
        if isinstance(b, (float, int)):
            if b == 0:
                return self.zeros_pl
            else:
                return a / b
        
        if safe_value is None:
            return tf.where(tf.equal(b, 0), b, a / b)
        else:
            return tf.where(tf.equal(b, 0), safe_value, a / b)
        
    def register_partial_AB_by_name(self, name, func):
        actual_name = name
        if self.base is not None:
            actual_name = actual_name + '_' + self.base
        if not actual_name in self.partial_AB.keys():
            self.partial_AB[actual_name] = func()
        return self.partial_AB[actual_name]
    
    def get_neighbor(self, node):
        
        if isinstance(node, (int, float, bool)):
            return node
        
        if self.step_mode == 'left' and self.base == 'u':
            return tf.roll(node, -1, axis=2)
        elif self.step_mode == 'left' and self.base == 'v':
            return tf.roll(node, -1, axis=1)
        elif self.step_mode == 'right' and self.base == 'u':
            return tf.roll(node, 1, axis=2)
        elif self.step_mode == 'right' and self.base == 'v':
            return tf.roll(node, 1, axis=1)
        raise
    
    def get_partial_trace_coord(self, entry):
        sess = self.sess
        
        assert self.step_mode is not None
        
        if entry in self.partial_trace_coord[self.step_mode].keys():
            return self.partial_trace_coord[self.step_mode][entry]
        else:
            idx = entry[0]
            base = entry[1]
            
            if isinstance(self.trace[idx], (int, float, bool)):
                collected_trace = tf.cast(self.trace[idx], tf.float32) * tf.ones_like(self.u)
            elif self.trace[idx].dtype == tf.bool:
                collected_trace = tf.cast(self.trace[idx], tf.float32)
            else:
                collected_trace = self.trace[idx]
            
            collected_trace = tf.expand_dims(collected_trace, -1)
            
            if self.step_mode == 'left':
                pI_pu = tf.roll(collected_trace, -1, axis=2) - collected_trace
                pI_pv = tf.roll(collected_trace, -1, axis=1) - collected_trace
            else:
                pI_pu = collected_trace - tf.roll(collected_trace, 1, axis=2)
                pI_pv = collected_trace - tf.roll(collected_trace, 1, axis=1)

                
            pI_pu /= self.pI_pu_denum[self.step_mode]
            pI_pv /= self.pI_pv_denum[self.step_mode]
            
            if isinstance(self.trace[idx], bool) or self.trace[idx].dtype == tf.bool:
                if True:
                    self.partial_trace_coord[self.step_mode][(idx, 'u')] = pI_pu[..., 0] * self.pI_pu_denum[self.step_mode][..., 0] / 2
                    self.partial_trace_coord[self.step_mode][(idx, 'v')] = pI_pv[..., 0] * self.pI_pv_denum[self.step_mode][..., 0] / 2
                else:
                    self.partial_trace_coord[self.step_mode][(idx, 'u')] = tf.abs(pI_pu[..., 0]) > 0.01
                    self.partial_trace_coord[self.step_mode][(idx, 'v')] = tf.abs(pI_pv[..., 0]) > 0.01
            else:
                self.partial_trace_coord[self.step_mode][(idx, 'u')] = pI_pu[..., 0]
                self.partial_trace_coord[self.step_mode][(idx, 'v')] = pI_pv[..., 0]

            return self.partial_trace_coord[self.step_mode][(idx, base)]
            

    def calc_partial_trace_coord(self, sess=None, precompute=True):
        
        collected_trace = [self.u, self.v]
        
        if precompute:
            
            for idx in self.partial_trace_coord_idx:

                if isinstance(self.trace[idx], (int, float, bool)):
                    collected_trace.append(tf.cast(self.trace[idx], tf.float32) * tf.ones_like(self.u))
                elif self.trace[idx].dtype == tf.bool:
                    collected_trace.append(tf.cast(self.trace[idx], tf.float32))
                else:
                    collected_trace.append(self.trace[idx])
        
        collected_trace = tf.stack(collected_trace, -1)
        
        self.pI_pu_denum = {}
        self.pI_pv_denum = {}
        
        for mode in ['left', 'right']:

            if mode == 'left':
                pI_pu = tf.roll(collected_trace, -1, axis=2) - collected_trace
                pI_pv = tf.roll(collected_trace, -1, axis=1) - collected_trace
            else:
                pI_pu = collected_trace - tf.roll(collected_trace, 1, axis=2)
                pI_pv = collected_trace - tf.roll(collected_trace, 1, axis=1)

            
            self.pI_pu_denum[mode] = pI_pu[..., 0:1]
            self.pI_pv_denum[mode] = pI_pv[..., 1:2]
        
            if precompute:

                pI_pu /= pI_pu[..., 0:1]
                pI_pv /= pI_pv[..., 1:2]

                for i in range(len(self.partial_trace_coord_idx)):

                    idx = self.partial_trace_coord_idx[i]


                    if isinstance(self.trace[idx], bool) or self.trace[idx].dtype == tf.bool:
                        # special handling for Boolean trace (we want true if they flipped state, false if the state is the same)
                        # how to distinguish the case of true -> false and false -> true?
                        # doesn't matter as long as it's consistent for all conds involved
                        if True:
                            self.partial_trace_coord[mode][(idx, 'u')] = pI_pu[..., i + 2] * self.pI_pu_denum[self.step_mode][..., 0] / 2
                            self.partial_trace_coord[mode][(idx, 'v')] = pI_pv[..., i + 2] * self.pI_pv_denum[self.step_mode][..., 0] / 2
                        else:
                            self.partial_trace_coord[mode][(idx, 'u')] = tf.abs(pI_pu[..., i + 2]) > 0.01
                            self.partial_trace_coord[mode][(idx, 'v')] = tf.abs(pI_pv[..., i + 2]) > 0.01
                    else:
                        self.partial_trace_coord[mode][(idx, 'u')] = pI_pu[..., i + 2]
                        self.partial_trace_coord[mode][(idx, 'v')] = pI_pv[..., i + 2]

    def combine_uv_ans(self, ans):
        assert len(ans) == 2
        return tf.where(tf.math.logical_or(tf.abs(ans[0]) <= 0, tf.math.logical_and(tf.abs(ans[0]) >= tf.abs(ans[1]), tf.abs(ans[1]) > 0)), ans[1], ans[0])
    
    def generate_I_var(self, sess=None, tunable_params=None, assign_ops=None, assign_init_pl=None, uv_offset=None):
        
        ans = []
        
        deriv_components = {}
                
        for base in ['u', 'v']:
            self.base = base
            
            final_derivs_l = None
            final_derivs_r = None
            
            for mode in ['left', 'right']:
                self.step_mode = mode
                
                col_scale = 1
                
                cond_scale = tf.sign(self.get_partial_trace_coord((self.cond_idx, base)))
                
                if use_select_rule == 1:
                    col_scale = self.get_neighbor(col_scale)

                col_scale *= cond_scale
                
                if use_select_rule == 0:
                    cond_xor = tf.math.logical_xor(self.trace[self.cond_idx],
                                                   tf.not_equal(cond_scale, 0))
                
                partial_cond_diff_coord = self.get_partial_trace_coord((self.cond_diff_idx, base))
                partial_col_cond_diff = self.safe_division(col_scale, partial_cond_diff_coord)
                
                partial_col_thre = -partial_col_cond_diff

                final_derivs = [partial_col_thre]
                
                final_derivs = tf.expand_dims(tf.concat(final_derivs, 0), -1)
                
                if mode == 'left':
                    final_derivs_l = final_derivs
                else:
                    final_derivs_r = final_derivs

            final_derivs_avg = 0.5 * (final_derivs_l + final_derivs_r)
                
            if self.deriv_two_sides:
                deriv_components[self.base] = {'left': final_derivs_l,
                                               'right': final_derivs_r,
                                               'avg': final_derivs_avg}

            ans.append(final_derivs_avg)

        if self.deriv_two_sides:
            combined_ans = {}
            for base, other_base in [('u', 'v'), ('v', 'u')]:
                combined_ans[base] = {}
                for step_mode in ['left', 'right']:
                    combined_ans[base][step_mode] = self.combine_uv_ans([deriv_components[base][step_mode],
                                                                         deriv_components[other_base]['avg']])
        else:
            combined_ans = self.combine_uv_ans(ans)

        return combined_ans
                    
                
                
              