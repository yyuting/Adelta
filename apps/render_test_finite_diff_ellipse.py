"""
------------------------------------------------------------------------------------------------------------------------------
# visualize gradient and generate gt

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ellipse --shader test_finite_diff_ellipse --init_values_pool apps/example_init_values/test_finite_diff_ellipse.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_col --render_size 256,256 --aa_nsamples 1000
"""

from render_util import *
from render_single import render_single

compiler.log_prefix_only = False
compiler.log_intermediates_less = True

def cmd_template():
    
    cmd = f"""python approx_gradient.py --shader test_finite_diff_ellipse --init_values_pool apps/example_init_values/test_finite_diff_ellipse.npy --metrics 1_scale_L2 --is_col --render_size 256,256"""
    
    return cmd

nargs = 9
args_range = np.array([256., 256., 256., 1., 1., 1., 1., 1, 1])

def test_finite_diff_ellipse(u, v, X, width=960, height=640):
    
    radius = X[0]
    origin_x = X[1]
    origin_y = X[2]
    
    fill_col = np.array([X[3], X[4], X[5]])
    alpha = X[6]
    fill_col = fill_col * alpha
    
    ratio = X[7]
    theta = X[8]
    
    bg_col = np.array([0., 0., 0.])
    
    u_diff = Var('u_diff', u - origin_x)
    v_diff = Var('v_diff', v - origin_y)
    
    x_diff = u_diff * sin(theta) + v_diff * cos(theta)
    y_diff = u_diff * cos(theta) - v_diff * sin(theta)
    y_diff = y_diff * ratio
    
    dist2 = (x_diff) ** 2 + (y_diff) ** 2
    
    radius2 = radius ** 2
    
    cond_diff = Var('cond_diff', dist2 - radius2)
    
    col = Var('col', select(cond_diff < 0, fill_col, bg_col))
    
    return col

shaders = [test_finite_diff_ellipse]
is_color = True