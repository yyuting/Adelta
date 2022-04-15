"""
# black and white version of circle for easy metric comparison with TEG

------------------------------------------------------------------------------------------------------------------------------
# Quantitative Metric using 2D box kernels

# get endpoints
python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_circle_bw --shader test_finite_diff_circle_bw --init_values_pool apps/example_init_values/test_finite_diff_circle_metric_init.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_col --backend hl --line_endpoints_method random_smooth --deriv_metric_visualization_thre 0.1 --deriv_metric_max_halflen 1e-3 --deriv_metric_line --deriv_metric_finite_diff_schedule 0 --kernel_nsamples 1 --deriv_n 2 --deriv_metric_suffix _2D_kernel --our_filter_direction 2d --render_size 256,256

# rhs
python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_circle_bw --shader test_finite_diff_circle_bw --init_values_pool apps/example_init_values/test_finite_diff_circle_metric_init.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_col --backend hl --line_endpoints_method random_smooth --deriv_metric_visualization_thre 0.1 --deriv_metric_max_halflen 1e-3 --deriv_metric_line --deriv_metric_finite_diff_schedule 0 --kernel_nsamples 100000 --deriv_n 2 --deriv_metric_suffix _2D_kernel --our_filter_direction 2d --render_size 256,256 --deriv_metric_endpoint_file /n/fs/scratch/yutingy/test_finite_diff_circle_bw/random_smooth_metric_2X1_len_0.001000_2D_kernel_endpoints.npy --kernel_sigma 0.1

# ours
python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_circle_bw --shader test_finite_diff_circle_bw --init_values_pool apps/example_init_values/test_finite_diff_circle_metric_init.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_col --backend hl --line_endpoints_method kernel_smooth_debug --deriv_metric_visualization_thre 0.1 --deriv_metric_max_halflen 1e-3 --deriv_metric_line --deriv_metric_finite_diff_schedule 0 --kernel_nsamples 1 --deriv_n 10000 --deriv_metric_suffix _2D_kernel --kernel_sigma 0.1 --kernel_uv_sigma 1 --kernel_smooth_exclude_our_kernel --render_size 256,256 --deriv_metric_endpoint_file /n/fs/scratch/yutingy/test_finite_diff_circle_bw/random_smooth_metric_2X1_len_0.001000_2D_kernel_endpoints.npy --deriv_metric_rhs_file /n/fs/scratch/yutingy/test_finite_diff_circle_bw/random_smooth_metric_2X100000_len_0.001000_2D_kernel_rhs.npy

# FD
python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_circle_bw --shader test_finite_diff_circle_bw --init_values_pool apps/example_init_values/test_finite_diff_circle_metric_init.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_col --backend hl --line_endpoints_method kernel_smooth_debug --deriv_metric_visualization_thre 0.1 --deriv_metric_max_halflen 1e-3 --deriv_metric_line --deriv_metric_finite_diff_schedule 1e-1,1e-2,1e-3,1e-4,1e-5 --kernel_nsamples 1 --deriv_n 10000 --deriv_metric_suffix _2D_kernel_FD --kernel_sigma 0.1 --kernel_uv_sigma 1 --kernel_smooth_exclude_our_kernel --render_size 256,256 --deriv_metric_no_ours --deriv_metric_endpoint_file /n/fs/scratch/yutingy/test_finite_diff_circle_bw/random_smooth_metric_2X1_len_0.001000_2D_kernel_endpoints.npy --deriv_metric_rhs_file /n/fs/scratch/yutingy/test_finite_diff_circle_bw/random_smooth_metric_2X100000_len_0.001000_2D_kernel_rhs.npy

# teg

python apps/teg_circle.py --init_values_pool apps/example_init_values/test_finite_diff_circle_metric_init.npy --dir /n/fs/scratch/yutingy/test_finite_diff_circle_bw --deriv_metric_endpoint_file /n/fs/scratch/yutingy/test_finite_diff_circle_bw/random_smooth_metric_2X1_len_0.001000_2D_kernel_endpoints.npy

cp /n/fs/scratch/yutingy/test_finite_diff_circle_bw/teg_lhs_10.npy /n/fs/scratch/yutingy/test_finite_diff_circle_bw/kernel_smooth_metric_debug_10000X1_len_0.001000_kernel_box_sigma_1.000000_0.100000_teg_2D_kernel.npy

# plot

python metric_compare_line_integral.py --baseline_dir /n/fs/scratch/yutingy/test_finite_diff_circle_bw --deriv_metric_suffix _2D_kernel_FD_finite_diff_0.100000,_2D_kernel_FD_finite_diff_0.010000,_2D_kernel_FD_finite_diff_0.001000,_2D_kernel_FD_finite_diff_0.000100,_2D_kernel_FD_finite_diff_0.000010,_2D_kernel,_teg_2D_kernel --eval_labels FD1e-1,FD1e-2,FD1e-3,FD1e-4,FD1e-5,Ours,TEG --max_half_len 1e-3 --rhs_file /n/fs/scratch/yutingy/test_finite_diff_circle_bw/random_smooth_metric_2X100000_len_0.001000_2D_kernel_rhs.npy --visualization_thre 0.01 --ncols 8 --visualize_img_name /n/fs/scratch/yutingy/test_finite_diff_circle_bw/visualize.png

"""

from render_util import *
from render_single import render_single

compiler.log_prefix_only = False
compiler.log_intermediates_less = True

def cmd_template():
    
    cmd = f"""python approx_gradient.py --shader test_finite_diff_circle_bw --init_values_pool apps/example_init_values/test_finite_diff_circle_metric_init.npy --metrics 1_scale_L2 --is_col --render_size 256,256"""
    
    return cmd

nargs = 3
args_range = np.array([256., 256., 256.])

def test_finite_diff_circle_bw(u, v, X, width=960, height=640):
    """
    X has shape nargs + 3
    first 3 entries are u, v, time
    the other entries are scalar tunable parameters
    """
    radius = X[0]
    origin_x = X[1]
    origin_y = X[2]

    fill_col = np.array([1., 1., 1.])
    
    bg_col = np.array([0., 0., 0.])
    
    u_diff = Var('u_diff', u - origin_x)
    v_diff = Var('v_diff', v - origin_y)
    
    dist2 = (u_diff) ** 2 + (v_diff) ** 2
    
    radius2 = radius ** 2
    
    cond_diff = Var('cond_diff', dist2 - radius2)
    
    col = Var('col', select(cond_diff < 0, fill_col, bg_col))
    
    return col

shaders = [test_finite_diff_circle_bw]
is_color = True