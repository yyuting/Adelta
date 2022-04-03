"""
------------------------------------------------------------------------------------------------------------------------------
# visualize gradient

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ellipse_ring --shader test_finite_diff_ellipse_ring --init_values_pool test_finite_diff_ellipse_ring.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_col --render_size 256,256 --aa_nsamples 1000
"""

from render_util import *
from render_single import render_single

compiler.log_prefix_only = False
compiler.log_intermediates_less = True

nargs = 9
args_range = np.array([256., 256., 256., 100., 1., 1., 1., 1., 1])

def test_finite_diff_ellipse_ring(u, v, X, width=960, height=640):
    
    radius = X[0]
    origin_x = X[1]
    origin_y = X[2]
    stroke_width = X[3]
    
    stroke_col = np.array([X[4], X[5], X[6]])
    alpha = X[7]
    stroke_col = stroke_col * alpha
    
    alpha = X[8]
    
    bg_col = np.array([0., 0., 0.])
    
    u_diff = Var('u_diff', u - origin_x)
    v_diff = Var('v_diff', v - origin_y)
    
    x_diff = u_diff * np.sin(np.pi / 4) + v_diff * np.cos(np.pi / 4)
    y_diff = u_diff * np.cos(np.pi / 4) - v_diff * np.sin(np.pi / 4)
    y_diff = y_diff * alpha
    
    dist2 = (x_diff) ** 2 + (y_diff) ** 2
    
    dist = dist2 ** 0.5
    
    cond0 = Var('cond0', dist - radius - stroke_width < 0)
    cond1 = Var('cond1', dist - radius + stroke_width > 0)
    
    col = Var('col', select(cond0 & cond1, stroke_col, bg_col))
    
    return col

shaders = [test_finite_diff_ellipse_ring]
is_color = True