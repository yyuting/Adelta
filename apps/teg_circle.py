# python teg_circle.py --init_values_pool ../test_finite_diff_circle_metric_init.npy --dir /n/fs/scratch/yutingy/test_finite_diff_circle_bw --deriv_metric_endpoint_file /n/fs/scratch/yutingy/test_finite_diff_circle_bw/random_smooth_metric_2X1_len_0.001000_2D_kernel_endpoints.npy

from teg import (
    Const,
    Var,
    TegVar,
    IfElse,
    Teg,
    Tup,
)
from teg.derivs.reverse_deriv import reverse_deriv
from teg.passes.reduce import reduce_to_base
from teg.math import Cos, Sin
from teg.maps.transform import teg_abs
import sys
from plot import render_image, save_image
from teg.eval import evaluate
from teg.maps.polar import polar_2d_map

import numpy as np
import os

args_range = np.array([256., 256., 256.])

is_sparse = False

def render_circle():
    
    x = TegVar('x')
    y = TegVar('y')
    r = TegVar('r')
    
    x_lb = Var('x_lb')
    x_ub = Var('x_ub')
    y_lb = Var('y_lb')
    y_ub = Var('y_ub')

    radius, ox, oy = Var('r'), Var('ox'), Var('oy')
    shader_args = [radius, ox, oy]
    
    radius = radius * args_range[0]
    ox = ox * args_range[1]
    oy = oy * args_range[2]
    
    integral = Teg(
                y_lb - oy, y_ub - oy,
                Teg(x_lb - ox, x_ub - ox,
                    polar_2d_map(IfElse(r < radius, 1, 0.0), x=x, y=y, r=r), x
                    ), y
                )
    
    return integral, shader_args, ((x_lb, x_ub), (y_lb, y_ub))

def main():
    
    if '--res_x' in sys.argv:
        res_x_idx = sys.argv.index('--res_x')
        res_x = int(sys.argv[res_x_idx + 1])
    else:
        res_x = 256
        
    if '--res_y' in sys.argv:
        res_y_idx = sys.argv.index('--res_y')
        res_y = int(sys.argv[res_y_idx + 1])
    else:
        res_y = 256
        
    if '--nsamples' in sys.argv:
        nsamples_idx = sys.argv.indes('--nsamples')
        nsamples = int(sys.argv[nsamples_idx + 1])
    else:
        nsamples = 10
        
    assert '--init_values_pool' in sys.argv
    init_values_pool_idx = sys.argv.index('--init_values_pool')
    init_values = np.load(sys.argv[init_values_pool_idx + 1])[0] / args_range
    
    assert '--deriv_metric_endpoint_file' in sys.argv
    endpoint_file_idx = sys.argv.index('--deriv_metric_endpoint_file')
    endpoint = np.load(sys.argv[endpoint_file_idx+1])
    random_dir = endpoint[1] - endpoint[0]
    random_dir /= np.sum(random_dir ** 2) ** 0.5
    
    assert '--dir' in sys.argv
    dir_idx = sys.argv.index('--dir')
    outdir = sys.argv[dir_idx + 1]
    
    integral, shader_args, teg_vars = render_circle()
    
    d_vars, dt_exprs = reverse_deriv(integral, Tup(Const(1)), output_list=shader_args)
    integral = reduce_to_base(integral)
    
    bindings = {shader_args[idx]: init_values[idx] for idx in range(len(shader_args))}
    
    # forward pass, use 1 sample for sanity check, should be pixelwise identical to the output of our compiler
    image = render_image(integral,
                         variables=teg_vars,
                         bindings=bindings,
                         res=(res_x, res_y),
                         kernel_size=[1, 1],
                         nsamples=1
                         )
    
    save_image(image, filename=os.path.join(outdir, 'visualize_teg.png'))
    
    t_schedule = np.linspace(0, 1, 10000)
    step_size = (endpoint[1] - endpoint[0]) * t_schedule[1]
    
    accum = 0
    count = 0
    for dt_expr,  in zip(dt_exprs):
        
        accum = accum + reduce_to_base(dt_expr) * step_size[count]
        count += 1
            
    if is_sparse:
        rhs = np.load('/n/fs/scratch/yutingy/test_finite_diff_circle_bw/random_smooth_metric_2X100000_len_0.001000_2D_kernel_rhs.npy')
        sparse_x, sparse_y = np.where(rhs != 0)
        
        lhs = np.zeros(rhs.shape)
        
        debug_lhs = np.zeros((sparse_y.shape[0], 10000))
    else:
        lhs = None
        debug_lhs = None
    
    halflen = 1e-3
    par_sigma = 0.1 * (halflen ** 2 / args_range.shape[0]) ** 0.5
    
    for t_idx in range(t_schedule.shape[0] - 1):
        t = t_schedule[t_idx]
        p_old = endpoint[0] + t_idx * step_size
        p_new = p_old + step_size
        
        p_eval = (p_old + p_new) / 2
        p_eval += np.random.uniform(low=-par_sigma, high=par_sigma, size=args_range.shape)
        
        bindings = {shader_args[idx]: p_eval[idx] for idx in range(len(shader_args))}
        
        if is_sparse:
            for idx in range(sparse_y.shape[0]):
                y_val = sparse_y[idx]
                x_val = sparse_x[idx]
                
                x_lb = x_val - 1
                x_ub = x_val + 1
                y_lb = y_val - 1
                y_ub = y_val + 1
                
                value = evaluate(accum, bindings={**bindings,
                                             teg_vars[0][0]: x_lb,
                                             teg_vars[0][1]: x_ub,
                                             teg_vars[1][0]: y_lb,
                                             teg_vars[1][1]: y_ub},
                                 num_samples=nsamples, backend='C_PyBind') / 4
                
                lhs[x_val, y_val] += value
                
                debug_lhs[idx, t_idx] = value
        else:
            current_lhs = render_image(accum,
                         variables=teg_vars,
                         bindings=bindings,
                         res=(res_x, res_y),
                         kernel_size=[1, 1],
                         nsamples=nsamples
                         ) / 4
            if lhs is None:
                lhs = current_lhs
            else:
                lhs += current_lhs
        
        print(t_idx)
        
    # because our rhs assumes output is a 3 channel image and sums across each channel
    np.save(os.path.join(outdir, 'teg_lhs_%d.npy' % nsamples), lhs * 3)     
    if debug_lhs is not None:
        np.save(os.path.join(outdir, 'debug_lhs.npy'), debug_lhs)
    
if __name__ == '__main__':
    main()
            
