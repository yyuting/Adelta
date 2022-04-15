"""
------------------------------------------------------------------------------------------------------------------------------
# command for visualization
python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_quadrant --shader test_finite_diff_quadrant --init_values_pool apps/example_init_values/test_finite_diff_quadrant_init_values.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_col --render_size 160,160

------------------------------------------------------------------------------------------------------------------------------
# Quantitative Metric using 2D box kernels

# get endpoints
python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_quadrant --shader test_finite_diff_quadrant --init_values_pool apps/example_init_values/test_finite_diff_quadrant_init_values.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_col --render_size 160,160 --line_endpoints_method random_smooth --deriv_metric_visualization_thre 0.1 --deriv_metric_max_halflen 1e-3 --deriv_metric_line --deriv_metric_finite_diff_schedule 0 --kernel_nsamples 1 --deriv_n 2 --deriv_metric_suffix _2D_kernel --our_filter_direction 2d

# rhs
python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_quadrant --shader test_finite_diff_quadrant --init_values_pool apps/example_init_values/test_finite_diff_quadrant_init_values.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_col --render_size 160,160 --line_endpoints_method random_smooth --deriv_metric_visualization_thre 0.1 --deriv_metric_max_halflen 1e-3 --deriv_metric_line --deriv_metric_finite_diff_schedule 0 --kernel_nsamples 100000 --deriv_n 2 --deriv_metric_suffix _2D_kernel --our_filter_direction 2d --deriv_metric_endpoint_file /n/fs/scratch/yutingy/test_finite_diff_quadrant/random_smooth_metric_2X1_len_0.001000_2D_kernel_endpoints.npy

# ours
python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_quadrant --shader test_finite_diff_quadrant --init_values_pool apps/example_init_values/test_finite_diff_quadrant_init_values.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_col --render_size 160,160 --line_endpoints_method kernel_smooth_debug --deriv_metric_visualization_thre 0.1 --deriv_metric_max_halflen 1e-3 --deriv_metric_line --deriv_metric_finite_diff_schedule 0 --kernel_nsamples 1 --deriv_n 10000 --deriv_metric_suffix _2D_kernel --kernel_sigma 0.1 --kernel_uv_sigma 1 --kernel_smooth_exclude_our_kernel --deriv_metric_endpoint_file /n/fs/scratch/yutingy/test_finite_diff_quadrant/random_smooth_metric_2X100000_len_0.001000_2D_kernel_endpoints.npy --deriv_metric_rhs_file /n/fs/scratch/yutingy/test_finite_diff_quadrant/random_smooth_metric_2X100000_len_0.001000_2D_kernel_rhs.npy

# FD
python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_quadrant --shader test_finite_diff_quadrant --init_values_pool apps/example_init_values/test_finite_diff_quadrant_init_values.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_col --render_size 160,160 --line_endpoints_method kernel_smooth_debug --deriv_metric_visualization_thre 0.1 --deriv_metric_max_halflen 1e-3 --deriv_metric_line --deriv_metric_finite_diff_schedule 1e-1,1e-2,1e-3,1e-4,1e-5 --kernel_nsamples 1 --deriv_n 10000 --deriv_metric_suffix _2D_kernel_FD --kernel_sigma 0.1 --kernel_uv_sigma 1 --kernel_smooth_exclude_our_kernel --deriv_metric_no_ours --deriv_metric_endpoint_file /n/fs/scratch/yutingy/test_finite_diff_quadrant/random_smooth_metric_2X100000_len_0.001000_2D_kernel_endpoints.npy --deriv_metric_rhs_file /n/fs/scratch/yutingy/test_finite_diff_quadrant/random_smooth_metric_2X100000_len_0.001000_2D_kernel_rhs.npy

# TEG

python apps/teg_quadrant.py --init_values_pool apps/example_init_values/test_finite_diff_quadrant_init_values.npy --dir /n/fs/scratch/yutingy/test_finite_diff_quadrant --deriv_metric_endpoint_file /n/fs/scratch/yutingy/test_finite_diff_quadrant/random_smooth_metric_2X100000_len_0.001000_2D_kernel_endpoints.npy --res_x 160 --res_y 160

cp /n/fs/scratch/yutingy/test_finite_diff_quadrant/teg_lhs_10.npy /n/fs/scratch/yutingy/test_finite_diff_quadrant/kernel_smooth_metric_debug_10000X1_len_0.001000_kernel_box_sigma_1.000000_0.100000_teg_2D_kernel.npy

# plot

python metric_compare_line_integral.py --baseline_dir /n/fs/scratch/yutingy/test_finite_diff_quadrant --deriv_metric_suffix _2D_kernel_FD_finite_diff_0.100000,_2D_kernel_FD_finite_diff_0.010000,_2D_kernel_FD_finite_diff_0.001000,_2D_kernel_FD_finite_diff_0.000100,_2D_kernel_FD_finite_diff_0.000010,_2D_kernel,_teg_2D_kernel --eval_labels FD1e-1,FD1e-2,FD1e-3,FD1e-4,FD1e-5,Ours,TEG --max_half_len 1e-3 --rhs_file /n/fs/scratch/yutingy/test_finite_diff_quadrant/random_smooth_metric_2X100000_len_0.001000_2D_kernel_rhs.npy --visualization_thre 0.01 --ncols 8 --visualize_img_name /n/fs/scratch/yutingy/test_finite_diff_quadrant/visualize.png

"""


from render_util import *
from render_single import render_single

compiler.log_prefix_only = False
compiler.log_intermediates_less = True

def cmd_template():
    
    cmd = f"""python approx_gradient.py --shader test_finite_diff_quadrant --init_values_pool apps/example_init_values/test_finite_diff_quadrant_init_values.npy --metrics 1_scale_L2 --is_col --render_size 160,160"""
    
    return cmd


nargs = 2
args_range = np.array([320, 320])

approx_mode = '1D_2samples'

use_select_rule = 1

def test_finite_diff_quadrant(u, v, X, width=960, height=640):
    """
    X has shape nargs + 3
    first 3 entries are u, v, time
    the other entries are scalar tunable parameters
    """
    
    ox = X[0]
    oy = X[1]
    
    col = sin(select(u > ox, 1., 0.) + select(v > oy, 1., 0.))
    
    return output_color([col, col, col])

shaders = [test_finite_diff_quadrant]
is_color = True