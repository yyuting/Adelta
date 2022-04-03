# python teg_rectangle.py --init_values_pool ../test_finite_diff_rectangle_init_values.npy --dir /n/fs/scratch/yutingy/test_finite_diff_rectangle_2step --deriv_metric_endpoint_file /n/fs/scratch/yutingy/test_finite_diff_rectangle_2step/random_smooth_metric_2X1_len_0.001000_2D_kernel_endpoints.npy

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
sys.path += ['..']
from plot import render_image, save_image
from teg.eval import evaluate

import numpy as np
import os

args_range = np.array([600, 600, 600, 600, 6.29])

def dot(x, y):
    assert len(x) == len(y)
    return sum([x[idx] * y[idx] for idx in range(len(x))])

def render_rectangle():
    
    x = TegVar('x')
    y = TegVar('y')
    
    x_lb = Var('x_lb')
    x_ub = Var('x_ub')
    y_lb = Var('y_lb')
    y_ub = Var('y_ub')
    
    bottom_center_x, bottom_center_y, rec_width, rec_height, rec_theta = Var('cx'), Var('cy'), Var('w'), Var('h'), Var('theta')
    shader_args = [bottom_center_x, bottom_center_y, rec_width, rec_height, rec_theta]
    
    bottom_center_x = bottom_center_x * args_range[0]
    bottom_center_y = bottom_center_y * args_range[1]
    rec_width = rec_width * args_range[2] 
    rec_height = rec_height * args_range[3]
    rec_theta = rec_theta * args_range[4]
    
    sin_theta = Sin(rec_theta)
    cos_theta = Cos(rec_theta)
    
    vertical_axis = [cos_theta, sin_theta]
    horizontal_axis = [sin_theta, -cos_theta]
    
    dist_to_vertical_axis = dot(vertical_axis, [x - bottom_center_x, y - bottom_center_y])
    dist_to_horizontal_axis = dot(horizontal_axis, [x - bottom_center_x, y - bottom_center_y])
    
    col0 = IfElse(dist_to_vertical_axis - rec_width / 2 > 0, 0, 1)
    col1 = IfElse(-dist_to_vertical_axis - rec_width / 2 > 0, 0, 1) * col0
    col2 = IfElse(-dist_to_horizontal_axis > 0, 0, 1) * col1
    col3 = IfElse(dist_to_horizontal_axis - rec_height > 0, 0, 1) * col2
        
    integral = Teg(
                y_lb, y_ub,
                Teg(x_lb, x_ub,
                    col3, x), y)
    
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
        
    integral, shader_args, teg_vars = render_rectangle()
    
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
    if debug_lhs is not None:
        np.save(os.path.join(outdir, 'debug_lhs.npy'), debug_lhs)
    
if __name__ == '__main__':
    main()
    