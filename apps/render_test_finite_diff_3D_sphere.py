"""
------------------------------------------------------------------------------------------------------------------------------
# command for visualization
python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_3D_sphere --shader test_finite_diff_3D_sphere --init_values_pool apps/example_init_values/test_finite_diff_3D_sphere_init_values.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 50,50,50 --backend torch --ndims 3 --ignore_glsl

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_3D_sphere --shader test_finite_diff_3D_sphere --init_values_pool apps/example_init_values/test_finite_diff_3D_sphere_init_values.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 50,50,50 --backend torch --ndims 3 --ignore_glsl

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_3D_sphere --shader test_finite_diff_3D_sphere --init_values_pool apps/example_init_values/test_finite_diff_3D_sphere_init_values.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 50,50,50 --backend torch --ndims 3 --ignore_glsl --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --no_reset_opt --no_binary_search_std --no_reset_sigma
"""

from render_util import *
from render_single import render_single

compiler.log_prefix_only = False
compiler.log_intermediates_less = True

nargs = 7
args_range = np.ones(nargs)
args_range[:4] = 10
sigmas_range = args_range

width = ArgumentScalar('width')
height = ArgumentScalar('height')
depth = ArgumentScalar('depth')

def test_finite_diff_3D_sphere(u, v, w, X, scalar_loss_scale):
    
    pos = [X[0], X[1], X[2]]
    radius = X[3]
    scale = [X[4], X[5], X[6]]
    
    col = select(((u - pos[0]) * scale[0]) ** 2 + 
                 ((v - pos[1]) * scale[1]) ** 2 + 
                 ((w - pos[2]) * scale[2]) ** 2 <= radius ** 2,
                 1.,
                 0.)
    
    return Compound([col])

shaders = [test_finite_diff_3D_sphere]
is_color = False