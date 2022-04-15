"""
------------------------------------------------------------------------------------------------------------------------------
# command for visualization
python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_1D_pulse --shader test_finite_diff_1D_pulse --init_values_pool apps/example_init_values/test_finite_diff_1D_pulse_init_values.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 600 --backend torch --ndims 1 --ignore_glsl

------------------------------------------------------------------------------------------------------------------------------
# command for optimization
python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_1D_pulse --shader test_finite_diff_1D_pulse --init_values_pool apps/example_init_values/test_finite_diff_1D_pulse_init_values.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 600 --backend torch --ndims 1 --ignore_glsl

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_1D_pulse --shader test_finite_diff_1D_pulse --init_values_pool apps/example_init_values/test_finite_diff_1D_pulse_init_values.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 600 --backend torch --ndims 1 --ignore_glsl --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --no_reset_opt --no_binary_search_std --no_reset_sigma
"""

from render_util import *
from render_single import render_single

compiler.log_prefix_only = False
compiler.log_intermediates_less = True

nargs = 3
args_range = np.array([300, 200, 1])
sigmas_range = args_range

def cmd_check(backend):
    if backend in ['hl', 'tf']:
        print('Error! this shader cannot be compiled into %s backend' % backend)
        raise

def cmd_template():
    
    cmd = f"""python approx_gradient.py --shader test_finite_diff_1D_pulse --init_values_pool apps/example_init_values/test_finite_diff_1D_pulse_init_values.npy --metrics 5_scale_L2 --render_size 600 --ndims 1 --ignore_glsl"""
        
    return cmd


width = ArgumentScalar('width')

def test_finite_diff_1D_pulse(u, X, scalar_loss_scale):
    
    pulse_center = X[0]
    pulse_half_width = X[1]
    pulse_col = X[2]
    
    col = select((u >= pulse_center - pulse_half_width) & (u <= pulse_center + pulse_half_width),
                 pulse_col,
                 0.)
    
    return Compound([col])

shaders = [test_finite_diff_1D_pulse]
is_color = False