# python teg_quadrant.py --init_values_pool ../test_finite_diff_quadrant_init_values.npy --dir /n/fs/scratch/yutingy/test_finite_diff_quadrant --deriv_metric_endpoint_file /n/fs/scratch/yutingy/test_finite_diff_quadrant/random_smooth_metric_2X100000_len_0.001000_2D_kernel_endpoints.npy --res_x 160 --res_y 160


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

import numpy as np
import os

args_range = np.array([320, 320])

def dot(x, y):
    assert len(x) == len(y)
    return sum([x[idx] * y[idx] for idx in range(len(x))])

def render_quadrant():
    
    x = TegVar('x')
    y = TegVar('y')
    
    x_lb = Var('x_lb')
    x_ub = Var('x_ub')
    y_lb = Var('y_lb')
    y_ub = Var('y_ub')
    
    ox, oy = Var('ox'), Var('oy')
    shader_args = [ox, oy]
    
    ox = ox * args_range[0]
    oy = oy * args_range[1]
    
    integral = Teg(
                y_lb, y_ub,
                Teg(x_lb, x_ub,
                    Sin(IfElse(x > ox, 1, 0) + IfElse(y > oy, 1, 0)), x), y)
    
    return integral, shader_args, ((x_lb, x_ub), (y_lb, y_ub))

def main():
    if '--res_x' in sys.argv:
        res_x_idx = sys.argv.index('--res_x')
        res_x = int(sys.argv[res_x_idx + 1])
    else:
        res_x = 320
        
    if '--res_y' in sys.argv:
        res_y_idx = sys.argv.index('--res_y')
        res_y = int(sys.argv[res_y_idx + 1])
    else:
        res_y = 320
        
    if '--nsamples' in sys.argv:
        nsamples_idx = sys.argv.indes('--nsamples')
        nsamples = int(sys.argv[nsamples_idx + 1])
    else:
        nsamples = 10
        
    tile_offset = [390, 100]
        
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
        
    integral, shader_args, teg_vars = render_quadrant()
    
    d_vars, dt_exprs = reverse_deriv(integral, Tup(Const(1)), output_list=shader_args)
    integral = reduce_to_base(integral)
    
    bindings = {shader_args[idx]: init_values[idx] for idx in range(len(shader_args))}
    
    # forward pass, use 1 sample for sanity check, should be pixelwise identical to the output of our compiler
    image = render_image(integral,
                         variables=teg_vars,
                         bindings=bindings,
                         tile_offsets=tile_offset,
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

        current_lhs = render_image(accum,
                     variables=teg_vars,
                     bindings=bindings,
                     tile_offsets=tile_offset,
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
    
if __name__ == '__main__':
    main()