from render_util import *
from render_single import render_single

compiler.log_prefix_only = False
compiler.log_intermediates_less = True

def cmd_template():
    
    cmd = f"""python approx_gradient.py --shader test_finite_diff_circles_Z --init_values_pool apps/example_init_values/test_finite_diff_circles_Z_init_values.npy --metrics 1_scale_L2 --render_size 256,256 --is_color --niters 100"""
    
    return cmd

init_name = 'test_finite_diff_ellipse'
default_size = [256,256]
default_opt_arg = '1_scale_L2'

nargs = 16
args_range = np.array([256.] * 6 + [1.] * 10)

default_Z = -1e4

def test_finite_diff_circles_Z(u, v, X, width=960, height=640):
    """
    X has shape nargs + 3
    first 3 entries are u, v, time
    the other entries are scalar tunable parameters
    """
    
    current_Z = default_Z
    current_col = np.array([0., 0., 0.])
    
    for i in range(2):
        radius = X[i]
        origin_x = X[2 + i]
        origin_y = X[4 + i]
        
        fill_col = np.array([X[6 + i], X[8 + i], X[10 + i]])
        alpha = X[12 + i]
        
        Z = X[14 + i]
        
        fill_col = fill_col * alpha
        
        dist2 = (u - origin_x) ** 2 + (v - origin_y) ** 2
        radius2 = radius ** 2
        
        cond0 = dist2 < radius2
        cond1 = Z > current_Z
        cond = cond0 & cond1
        
        current_col = select(cond, fill_col, current_col)
        current_Z = select(cond, Z, current_Z)
    
    return current_col

shaders = [test_finite_diff_circles_Z]
is_color = True