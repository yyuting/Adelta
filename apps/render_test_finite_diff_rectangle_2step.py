"""
------------------------------------------------------------------------------------------------------------------------------
# command for visualization
python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_rectangle_2step --shader test_finite_diff_rectangle_2step --init_values_pool apps/example_init_values/test_finite_diff_rectangle_init_values.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_col --camera_size 960,640 --render_size 320,320 --tile_offset 390,100

------------------------------------------------------------------------------------------------------------------------------
# Quantitative Metric using 2D box kernels

# get endpoints
python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_rectangle_2step --shader test_finite_diff_rectangle_2step --init_values_pool apps/example_init_values/test_finite_diff_rectangle_init_values.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_col --backend hl --line_endpoints_method random_smooth --deriv_metric_visualization_thre 0.1 --deriv_metric_max_halflen 1e-3 --deriv_metric_line --deriv_metric_finite_diff_schedule 0 --kernel_nsamples 1 --deriv_n 2 --deriv_metric_suffix _2D_kernel --our_filter_direction 2d --camera_size 960,640 --render_size 320,320 --tile_offset 390,100

# rhs
python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_rectangle_2step --shader test_finite_diff_rectangle_2step --init_values_pool apps/example_init_values/test_finite_diff_rectangle_init_values.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_col --backend hl --line_endpoints_method random_smooth --deriv_metric_visualization_thre 0.1 --deriv_metric_max_halflen 1e-3 --deriv_metric_line --deriv_metric_finite_diff_schedule 0 --kernel_nsamples 100000 --deriv_n 2 --deriv_metric_suffix _2D_kernel --our_filter_direction 2d --camera_size 960,640 --render_size 320,320 --tile_offset 390,100 --deriv_metric_endpoint_file /n/fs/scratch/yutingy/test_finite_diff_rectangle_2step/random_smooth_metric_2X1_len_0.001000_2D_kernel_endpoints.npy --kernel_sigma 0.1

# ours
python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_rectangle_2step --shader test_finite_diff_rectangle_2step --init_values_pool apps/example_init_values/test_finite_diff_rectangle_init_values.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_col --backend hl --line_endpoints_method kernel_smooth_debug --deriv_metric_visualization_thre 0.1 --deriv_metric_max_halflen 1e-3 --deriv_metric_line --deriv_metric_finite_diff_schedule 0 --kernel_nsamples 1 --deriv_n 10000 --deriv_metric_suffix _2D_kernel --kernel_sigma 0.1 --kernel_uv_sigma 1 --kernel_smooth_exclude_our_kernel --camera_size 960,640 --render_size 320,320 --tile_offset 390,100 --deriv_metric_endpoint_file /n/fs/scratch/yutingy/test_finite_diff_rectangle_2step/random_smooth_metric_2X100000_len_0.001000_2D_kernel_endpoints.npy --deriv_metric_rhs_file /n/fs/scratch/yutingy/test_finite_diff_rectangle_2step/random_smooth_metric_2X100000_len_0.001000_2D_kernel_rhs.npy

# FD
python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_rectangle_2step --shader test_finite_diff_rectangle_2step --init_values_pool apps/example_init_values/test_finite_diff_rectangle_init_values.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_col --backend hl --line_endpoints_method kernel_smooth_debug --deriv_metric_visualization_thre 0.1 --deriv_metric_max_halflen 1e-3 --deriv_metric_line --deriv_metric_finite_diff_schedule 1e-1,1e-2,1e-3,1e-4,1e-5 --kernel_nsamples 1 --deriv_n 10000 --deriv_metric_suffix _2D_kernel_FD --kernel_sigma 0.1 --kernel_uv_sigma 1 --kernel_smooth_exclude_our_kernel --camera_size 960,640 --render_size 320,320 --tile_offset 390,100 --deriv_metric_no_ours --deriv_metric_endpoint_file /n/fs/scratch/yutingy/test_finite_diff_rectangle_2step/random_smooth_metric_2X100000_len_0.001000_2D_kernel_endpoints.npy --deriv_metric_rhs_file /n/fs/scratch/yutingy/test_finite_diff_rectangle_2step/random_smooth_metric_2X100000_len_0.001000_2D_kernel_rhs.npy

# teg

python apps/teg_rectangle.py --init_values_pool apps/example_init_values/test_finite_diff_rectangle_init_values.npy --dir /n/fs/scratch/yutingy/test_finite_diff_rectangle_2step --deriv_metric_endpoint_file /n/fs/scratch/yutingy/test_finite_diff_rectangle_2step/random_smooth_metric_2X1_len_0.001000_2D_kernel_endpoints.npy

cp /n/fs/scratch/yutingy/test_finite_diff_rectangle_2step/teg_lhs_10.npy /n/fs/scratch/yutingy/test_finite_diff_rectangle_2step/kernel_smooth_metric_debug_10000X1_len_0.001000_kernel_box_sigma_1.000000_0.100000_teg_2D_kernel.npy

# plot

python metric_compare_line_integral.py --baseline_dir /n/fs/scratch/yutingy/test_finite_diff_rectangle_2step --deriv_metric_suffix _2D_kernel_FD_finite_diff_0.100000,_2D_kernel_FD_finite_diff_0.010000,_2D_kernel_FD_finite_diff_0.001000,_2D_kernel_FD_finite_diff_0.000100,_2D_kernel_FD_finite_diff_0.000010,_2D_kernel,_teg_2D_kernel --eval_labels FD1e-1,FD1e-2,FD1e-3,FD1e-4,FD1e-5,Ours,TEG --max_half_len 1e-3 --rhs_file /n/fs/scratch/yutingy/test_finite_diff_rectangle_2step/random_smooth_metric_2X100000_len_0.001000_2D_kernel_rhs.npy --visualization_thre 0.01 --ncols 8 --visualize_img_name /n/fs/scratch/yutingy/test_finite_diff_rectangle_2step/visualize.png
"""

from render_util import *
from render_single import render_single

compiler.log_prefix_only = False
compiler.log_intermediates_less = True

def cmd_template():
    cmd = f"""python approx_gradient.py --shader test_finite_diff_rectangle_2step --init_values_pool apps/example_init_values/test_finite_diff_rectangle_init_values.npy --metrics naive_sum --is_col --camera_size 960,640 --render_size 320,320 --tile_offset 390,100"""
    
    return cmd

nargs = 5
args_range = np.array([600, 600, 600, 600, 6.29])

def test_finite_diff_rectangle_2step(u, v, X, width=960, height=640):
    """
    X has shape nargs + 3
    first 3 entries are u, v, time
    the other entries are scalar tunable parameters
    """

    bottom_center_x = X[0]
    bottom_center_y = X[1]
    rec_width = X[2]
    rec_height = X[3]
    rec_theta = X[4]
    
    sin_theta = Var('sin_theta', sin(rec_theta))
    cos_theta = Var('cos_theta', cos(rec_theta))
    
    vertical_axis = [cos_theta, sin_theta]
    horizontal_axis = [sin_theta, -cos_theta]
    
    dist_to_vertical_axis = Var('dist2v', dot(vertical_axis, [u - bottom_center_x, v - bottom_center_y]))
    dist_to_horizontal_axis = Var('dist2h', dot(horizontal_axis, [u - bottom_center_x, v - bottom_center_y]))
    
    cond0_diff = Var('cond0_diff', abs(dist_to_vertical_axis) - rec_width / 2)
    cond1_diff = Var('cond1_diff', -dist_to_horizontal_axis)
    cond2_diff = Var('cond2_diff', dist_to_horizontal_axis - rec_height)
    
    cond0 = Var('cond0', cond0_diff > 0)
    cond1 = Var('cond1', cond1_diff > 0)
    cond2 = Var('cond2', cond2_diff > 0)

    col0 = Var('col0', select(cond0, 0, 1))
    col1 = Var('col1', select(cond1, 0, col0))
    col2 = Var('col2', select(cond2, 0, col1))
    
    return output_color([col2, col2, col2])

shaders = [test_finite_diff_rectangle_2step]
is_color = True