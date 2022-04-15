"""
------------------------------------------------------------------------------------------------------------------------------
# command for visualization
python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool apps/example_init_values/test_finite_diff_siggraph_cone_init_values_pool.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_col --camera_size 960,960 --render_size 600,600 --tile_offset 180,180 --autoscheduler --gt_file siggraph_gradient.png --gt_transposed

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone_AD --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool apps/example_init_values/test_finite_diff_siggraph_cone_init_values_pool.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization AD --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_col --camera_size 960,960 --render_size 600,600 --tile_offset 180,180 --autoscheduler --gt_file siggraph_gradient.png --gt_transposed

------------------------------------------------------------------------------------------------------------------------------
# optimization with random var

----------------------------------------------------------------------------------------
# ours

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool apps/example_init_values/test_finite_diff_siggraph_cone_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 960,960 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file siggraph_gradient.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_random --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --no_binary_search_std --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --save_best_par --no_reset_sigma

# ours without random variables

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool apps/example_init_values/test_finite_diff_siggraph_cone_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 960,960 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file siggraph_gradient.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_no_random --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --no_reset_sigma

----------------------------------------------------------------------------------------
# AD

---------------------------------------------
# same opt as ours and FD

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone_AD --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool apps/example_init_values/test_finite_diff_siggraph_cone_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization AD --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 960,960 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file siggraph_gradient.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_random --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --no_binary_search_std --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --save_best_par --no_reset_sigma

---------------------------------------------
# vanilla opt

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone_AD --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool apps/example_init_values/test_finite_diff_siggraph_cone_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization AD --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 960,960 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file siggraph_gradient.png --gt_transposed --suffix _from_real_vanilla --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --no_reset_sigma

----------------------------------------------------------------------------------------
# Zeroth Order

---------------------------------------------
# vanilla

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool apps/example_init_values/test_finite_diff_siggraph_cone_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 960,960 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file siggraph_gradient.png --gt_transposed --suffix _from_real_vanilla --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --optimizer scipy.Nelder-Mead --niters 10000

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool apps/example_init_values/test_finite_diff_siggraph_cone_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 960,960 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file siggraph_gradient.png --gt_transposed --suffix _from_real_vanilla --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --optimizer scipy.Powell --niters 10000

----------------------------------------------------------------------------------------
# FD

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool apps/example_init_values/test_finite_diff_siggraph_cone_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --render_size 960,960 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file siggraph_gradient.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_fd_h_01 --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --no_binary_search_std --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool apps/example_init_values/test_finite_diff_siggraph_cone_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 960,960 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file siggraph_gradient.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_fd_h_001 --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --no_binary_search_std --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool apps/example_init_values/test_finite_diff_siggraph_cone_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.001 --finite_diff_both_sides --render_size 960,960 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file siggraph_gradient.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_fd_h_0001 --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --no_binary_search_std --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool apps/example_init_values/test_finite_diff_siggraph_cone_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.0001 --finite_diff_both_sides --render_size 960,960 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file siggraph_gradient.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_fd_h_00001 --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --no_binary_search_std --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool apps/example_init_values/test_finite_diff_siggraph_cone_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.00001 --finite_diff_both_sides --render_size 960,960 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file siggraph_gradient.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_fd_h_000001 --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --no_binary_search_std --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --no_reset_sigma

---------------------------------------------
# No random opt

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool apps/example_init_values/test_finite_diff_siggraph_cone_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --render_size 960,960 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file siggraph_gradient.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_no_random_fd_h_01 --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool apps/example_init_values/test_finite_diff_siggraph_cone_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 960,960 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file siggraph_gradient.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_no_random_fd_h_001 --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool apps/example_init_values/test_finite_diff_siggraph_cone_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.001 --finite_diff_both_sides --render_size 960,960 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file siggraph_gradient.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_no_random_fd_h_0001 --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool apps/example_init_values/test_finite_diff_siggraph_cone_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.0001 --finite_diff_both_sides --render_size 960,960 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file siggraph_gradient.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_no_random_fd_h_00001 --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool apps/example_init_values/test_finite_diff_siggraph_cone_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.00001 --finite_diff_both_sides --render_size 960,960 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file siggraph_gradient.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_no_random_fd_h_000001 --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --no_reset_sigma

----------------------------------------------------------------------------------------
# SPSA

---------------------------------------------
# same opt as ours and FD
# Runtime
# Ours: 0.0072
# SPSA: 0.0048
# SPSA iter match ours runtime: 3000

-------------------------
# 1 sample (only runned with one choice of h to get a runtime estimate)

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool apps/example_init_values/test_finite_diff_siggraph_cone_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --render_size 960,960 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file siggraph_gradient.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_fd_h_01 --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --no_binary_search_std --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --finite_diff_spsa_samples 1 --no_reset_sigma

-------------------------
# 1 sample iteration scaled by runtime

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool apps/example_init_values/test_finite_diff_siggraph_cone_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --render_size 960,960 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file siggraph_gradient.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_fd_h_01_scaled_niter --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --no_binary_search_std --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --finite_diff_spsa_samples 1 --niters 3000 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool apps/example_init_values/test_finite_diff_siggraph_cone_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 960,960 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file siggraph_gradient.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_fd_h_001_scaled_niter --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --no_binary_search_std --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --finite_diff_spsa_samples 1 --niters 3000 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool apps/example_init_values/test_finite_diff_siggraph_cone_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.001 --finite_diff_both_sides --render_size 960,960 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file siggraph_gradient.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_fd_h_0001_scaled_niter --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --no_binary_search_std --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --finite_diff_spsa_samples 1 --niters 3000 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool apps/example_init_values/test_finite_diff_siggraph_cone_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.0001 --finite_diff_both_sides --render_size 960,960 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file siggraph_gradient.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_fd_h_00001_scaled_niter --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --no_binary_search_std --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --finite_diff_spsa_samples 1 --niters 3000 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool apps/example_init_values/test_finite_diff_siggraph_cone_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.00001 --finite_diff_both_sides --render_size 960,960 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file siggraph_gradient.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_fd_h_000001_scaled_niter --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --no_binary_search_std --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --finite_diff_spsa_samples 1 --niters 3000 --no_reset_sigma

-------------------------
# half sample as FD: 27 samples

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool apps/example_init_values/test_finite_diff_siggraph_cone_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --render_size 960,960 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file siggraph_gradient.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_fd_h_01 --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --no_binary_search_std --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --finite_diff_spsa_samples 27 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool apps/example_init_values/test_finite_diff_siggraph_cone_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 960,960 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file siggraph_gradient.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_fd_h_001 --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --no_binary_search_std --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --finite_diff_spsa_samples 27 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool apps/example_init_values/test_finite_diff_siggraph_cone_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.001 --finite_diff_both_sides --render_size 960,960 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file siggraph_gradient.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_fd_h_0001 --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --no_binary_search_std --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --finite_diff_spsa_samples 27 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool apps/example_init_values/test_finite_diff_siggraph_cone_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.0001 --finite_diff_both_sides --render_size 960,960 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file siggraph_gradient.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_fd_h_00001 --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --no_binary_search_std --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --finite_diff_spsa_samples 27 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool apps/example_init_values/test_finite_diff_siggraph_cone_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.00001 --finite_diff_both_sides --render_size 960,960 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file siggraph_gradient.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_fd_h_000001 --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --no_binary_search_std --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --finite_diff_spsa_samples 27 --no_reset_sigma

---------------------------------------------
# vanilla opt
# no random noise, no alternation, no multi stage
# Runtime
# Ours: 0.0072
# SPSA: 0.0041
# SPSA iter match ours runtime: 3512

-------------------------
# 1 sample (only runned with one choice of h to get a runtime estimate)

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool apps/example_init_values/test_finite_diff_siggraph_cone_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --render_size 960,960 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file siggraph_gradient.png --gt_transposed --suffix _from_real_vanilla_fd_h_01 --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --finite_diff_spsa_samples 1 --no_reset_sigma

-------------------------
# 1 sample iteration scaled by runtime

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool apps/example_init_values/test_finite_diff_siggraph_cone_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --render_size 960,960 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file siggraph_gradient.png --gt_transposed --suffix _from_real_vanilla_fd_h_01_scaled_niter --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --finite_diff_spsa_samples 1 --niters 3512 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool apps/example_init_values/test_finite_diff_siggraph_cone_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 960,960 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file siggraph_gradient.png --gt_transposed --suffix _from_real_vanilla_fd_h_001_scaled_niter --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --finite_diff_spsa_samples 1 --niters 3512 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool apps/example_init_values/test_finite_diff_siggraph_cone_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.001 --finite_diff_both_sides --render_size 960,960 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file siggraph_gradient.png --gt_transposed --suffix _from_real_vanilla_fd_h_0001_scaled_niter --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --finite_diff_spsa_samples 1 --niters 3512 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool apps/example_init_values/test_finite_diff_siggraph_cone_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.0001 --finite_diff_both_sides --render_size 960,960 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file siggraph_gradient.png --gt_transposed --suffix _from_real_vanilla_fd_h_00001_scaled_niter --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --finite_diff_spsa_samples 1 --niters 3512 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool apps/example_init_values/test_finite_diff_siggraph_cone_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.00001 --finite_diff_both_sides --render_size 960,960 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file siggraph_gradient.png --gt_transposed --suffix _from_real_vanilla_fd_h_000001_scaled_niter --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --finite_diff_spsa_samples 1 --niters 3512 --no_reset_sigma

-------------------------
# half sample as FD: 22 samples

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool apps/example_init_values/test_finite_diff_siggraph_cone_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --render_size 960,960 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file siggraph_gradient.png --gt_transposed --suffix _from_real_vanilla_fd_h_01 --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --finite_diff_spsa_samples 22 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool apps/example_init_values/test_finite_diff_siggraph_cone_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 960,960 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file siggraph_gradient.png --gt_transposed --suffix _from_real_vanilla_fd_h_001 --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --finite_diff_spsa_samples 22 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool apps/example_init_values/test_finite_diff_siggraph_cone_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.001 --finite_diff_both_sides --render_size 960,960 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file siggraph_gradient.png --gt_transposed --suffix _from_real_vanilla_fd_h_0001 --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --finite_diff_spsa_samples 22 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool apps/example_init_values/test_finite_diff_siggraph_cone_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.0001 --finite_diff_both_sides --render_size 960,960 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file siggraph_gradient.png --gt_transposed --suffix _from_real_vanilla_fd_h_00001 --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --finite_diff_spsa_samples 22 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool apps/example_init_values/test_finite_diff_siggraph_cone_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.00001 --finite_diff_both_sides --render_size 960,960 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file siggraph_gradient.png --gt_transposed --suffix _from_real_vanilla_fd_h_000001 --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --finite_diff_spsa_samples 22 --no_reset_sigma

---------------------------------------------
# no random, same opt process as ours and FD
# Runtime
# Ours: 0.0072
# SPSA: 0.0042
# SPSA iter match ours runtime: 3429

-------------------------
# 1 sample (only runned with one choice of h to get a runtime estimate)

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool apps/example_init_values/test_finite_diff_siggraph_cone_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --render_size 960,960 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file siggraph_gradient.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_no_random_fd_h_01 --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --finite_diff_spsa_samples 1 --no_reset_sigma

-------------------------
# 1 sample iteration scaled by runtime

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool apps/example_init_values/test_finite_diff_siggraph_cone_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --render_size 960,960 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file siggraph_gradient.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_no_random_fd_h_01_scaled_niter --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --finite_diff_spsa_samples 1 --no_reset_sigma --niter 3429

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool apps/example_init_values/test_finite_diff_siggraph_cone_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 960,960 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file siggraph_gradient.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_no_random_fd_h_001_scaled_niter --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --finite_diff_spsa_samples 1 --no_reset_sigma --niter 3429

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool apps/example_init_values/test_finite_diff_siggraph_cone_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.001 --finite_diff_both_sides --render_size 960,960 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file siggraph_gradient.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_no_random_fd_h_0001_scaled_niter --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --finite_diff_spsa_samples 1 --no_reset_sigma --niter 3429

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool apps/example_init_values/test_finite_diff_siggraph_cone_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.0001 --finite_diff_both_sides --render_size 960,960 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file siggraph_gradient.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_no_random_fd_h_00001_scaled_niter --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --finite_diff_spsa_samples 1 --no_reset_sigma --niter 3429

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool apps/example_init_values/test_finite_diff_siggraph_cone_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.00001 --finite_diff_both_sides --render_size 960,960 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file siggraph_gradient.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_no_random_fd_h_000001_scaled_niter --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --finite_diff_spsa_samples 1 --no_reset_sigma --niter 3429

-------------------------
# half sample as FD: 22 samples

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool apps/example_init_values/test_finite_diff_siggraph_cone_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --render_size 960,960 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file siggraph_gradient.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_no_random_fd_h_01 --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --finite_diff_spsa_samples 22 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool apps/example_init_values/test_finite_diff_siggraph_cone_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 960,960 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file siggraph_gradient.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_no_random_fd_h_001 --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --finite_diff_spsa_samples 22 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool apps/example_init_values/test_finite_diff_siggraph_cone_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.001 --finite_diff_both_sides --render_size 960,960 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file siggraph_gradient.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_no_random_fd_h_0001 --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --finite_diff_spsa_samples 22 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool apps/example_init_values/test_finite_diff_siggraph_cone_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.0001 --finite_diff_both_sides --render_size 960,960 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file siggraph_gradient.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_no_random_fd_h_00001 --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --finite_diff_spsa_samples 22 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool apps/example_init_values/test_finite_diff_siggraph_cone_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.00001 --finite_diff_both_sides --render_size 960,960 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file siggraph_gradient.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_no_random_fd_h_000001 --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --finite_diff_spsa_samples 22 --no_reset_sigma

------------------------------------------------------------------------------------------------------------------------------
# command for generating glsl in a seperate process

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone/best_par_from_real_random.npy --modes render --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 960,960 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file siggraph_gradient.png --gt_transposed --backend hl --quiet --autoscheduler

------------------------------------------------------------------------------------------------------------------------------
# render optimization process

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool test_finite_diff_siggraph_cone_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 960,960 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file siggraph_gradient.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_random --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --no_binary_search_std --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --save_all_par

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone/ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_result3_0.npy --modes render --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 960,960 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file siggraph_gradient.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _opt --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --no_binary_search_std --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler

cd /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone; ffmpeg -i init_opt%d.png -r 30 -c:v libx264 -preset slow -crf 0 -r 30 animation.mp4; cd /n/fs/shaderml/differentiable_compiler/

------------------------------------------------------------------------------------------------------------------------------
# Quantitative Metric using 2D box kernels
# Optimization runtime
# Ours: 0.0072
# SPSA 1 sample: 0.0042
# FD 1 sample: 0.1060

# Ours need 15 samples for similar runtime with FD in opt
# SPSA need 2 samples for similar runtime with ours in opt

# endpoint

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone/best_par_from_real_random.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_col --render_size 960,960 --line_endpoints_method random_smooth --deriv_metric_visualization_thre 0.1 --deriv_metric_max_halflen 2e-3 --deriv_metric_line --deriv_metric_finite_diff_schedule 0 --kernel_nsamples 1 --deriv_n 2 --deriv_metric_no_ours --our_filter_direction 2d --deriv_metric_suffix _2D_kernel

# rhs

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone/best_par_from_real_random.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_col --render_size 960,960 --line_endpoints_method random_smooth --deriv_metric_visualization_thre 0.1 --deriv_metric_max_halflen 2e-3 --deriv_metric_line --deriv_metric_finite_diff_schedule 0 --kernel_nsamples 100000 --deriv_n 2 --deriv_metric_no_ours --our_filter_direction 2d --deriv_metric_suffix _2D_kernel --deriv_metric_endpoint_file /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone/random_smooth_metric_2X1_len_0.002000_2D_kernel_endpoints.npy --kernel_sigma 0.1

----------------------------------------------------------------------------------------
# ours

# 1 sample

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone/best_par_from_real_random.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_color --render_size 960,960 --line_endpoints_method kernel_smooth_debug --deriv_metric_visualization_thre 0.1 --deriv_metric_max_halflen 2e-3 --deriv_metric_line --deriv_metric_endpoint_file /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone/random_smooth_metric_2X1_len_0.002000_2D_kernel_endpoints.npy --kernel_nsamples 1 --deriv_metric_rhs_file /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone/random_smooth_metric_2X100000_len_0.002000_2D_kernel_rhs.npy --deriv_metric_suffix _2D_kernel --kernel_sigma 0 --kernel_uv_sigma 1 --deriv_n 10000 --deriv_metric_finite_diff_schedule 0 --kernel_smooth_exclude_our_kernel --autoscheduler --kernel_sigma 0.1

----------------------------------------------------------------------------------------
# FD

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone/best_par_from_real_random.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_color --render_size 960,960 --line_endpoints_method kernel_smooth_debug --deriv_metric_visualization_thre 0.1 --deriv_metric_max_halflen 2e-3 --deriv_metric_line --deriv_metric_endpoint_file /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone/random_smooth_metric_2X1_len_0.002000_2D_kernel_endpoints.npy --kernel_nsamples 1 --deriv_metric_rhs_file /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone/random_smooth_metric_2X100000_len_0.002000_2D_kernel_rhs.npy --deriv_metric_suffix _2D_kernel_FD --kernel_sigma 0 --kernel_uv_sigma 1 --deriv_n 10000 --deriv_metric_finite_diff_schedule 1e-1,1e-2,1e-3,1e-4,1e-5 --kernel_smooth_exclude_our_kernel --deriv_metric_no_ours --kernel_sigma 0.1

----------------------------------------------------------------------------------------
# SPSA
# 2 samples
# number of samples corresponds to runtime in our optimization pipeline (with random, multi stage loss)

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone/best_par_from_real_random.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_color --render_size 960,960 --line_endpoints_method kernel_smooth_debug --deriv_metric_visualization_thre 0.1 --deriv_metric_max_halflen 2e-3 --deriv_metric_line --deriv_metric_endpoint_file /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone/random_smooth_metric_2X1_len_0.002000_2D_kernel_endpoints.npy --kernel_nsamples 1 --deriv_metric_rhs_file /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone/random_smooth_metric_2X100000_len_0.002000_2D_kernel_rhs.npy --deriv_metric_suffix _2D_kernel_SPSA_2 --kernel_sigma 0 --kernel_uv_sigma 1 --deriv_n 10000 --deriv_metric_finite_diff_schedule 1e-1,1e-2,1e-3,1e-4,1e-5 --kernel_smooth_exclude_our_kernel --deriv_metric_no_ours --finite_diff_spsa_samples 2 --kernel_sigma 0.1

------------------------------------------------------------------------------------------------------------------------------
# plot

----------------------------------------------------------------------------------------
# plot

---------------------------------------------
# preprocess

python preprocess_raw_loss_data.py /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_000001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_000001_scaled_niter_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_00001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_00001_scaled_niter_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_0001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_0001_scaled_niter_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_001_scaled_niter_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_01_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_01_scaled_niter_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_000001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_000001_scaled_niter_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_00001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_00001_scaled_niter_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_0001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_0001_scaled_niter_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_001_scaled_niter_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_01_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_01_scaled_niter_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone/finite_diff22_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_000001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone/finite_diff22_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_00001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone/finite_diff22_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_0001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone/finite_diff22_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone/finite_diff22_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_01_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone/finite_diff27_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_000001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone/finite_diff27_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_00001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone/finite_diff27_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_0001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone/finite_diff27_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone/finite_diff27_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_01_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone/finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_000001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone/finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_00001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone/finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_0001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone/finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone/finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_01_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone/ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_all_loss.npy

---------------------------------------------
# plot median 

# ours vs FD
python plot_opt_baseline.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --filenames ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_all_loss.npy,finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_01_all_loss.npy,finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_001_all_loss.npy,finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_0001_all_loss.npy,finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_00001_all_loss.npy,finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_000001_all_loss.npy --labels ours,FD_1e-1,FD_1e-2,FD_1e-3,FD_1e-4,FD_1e-5 --scales 0.00780,0.106,0.117,0.115,0.147,0.106 --suffix _ours_FD_median --median

# ours vs SPSA
python plot_opt_baseline.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --filenames ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_01_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_001_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_0001_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_00001_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_000001_scaled_niter_all_loss.npy --labels ours,SPSA_1e-1,SPSA_1e-2,SPSA_1e-3,SPSA_1e-4,SPSA_1e-5 --scales 0.00780,0.00494,0.00492,0.00499,0.00500,0.00509 --suffix _ours_SPSA_median --median

# ours vs SPSA multiple samples (27, half samples as FD)
python plot_opt_baseline.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --filenames ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_all_loss.npy,finite_diff27_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_01_all_loss.npy,finite_diff27_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_001_all_loss.npy,finite_diff27_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_0001_all_loss.npy,finite_diff27_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_00001_all_loss.npy,finite_diff27_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_000001_all_loss.npy --labels ours,multi_SPSA_1e-1,multi_SPSA_1e-2,multi_SPSA_1e-3,multi_SPSA_1e-4,multi_SPSA_1e-5 --scales 0.00780,0.0613,0.0602,0.0600,0.0600,0.0593 --suffix _ours_multi_SPSA_median --median

# ours vs SPSA vanilla
python plot_opt_baseline.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --filenames ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_01_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_001_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_0001_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_00001_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_000001_scaled_niter_all_loss.npy --labels ours,SPSA_vanilla_1e-1,SPSA_vanilla_1e-2,SPSA_vanilla_1e-3,SPSA_vanilla_1e-4,SPSA_vanilla_1e-5 --scales 0.00780,0.00440,0.00407,0.00406,0.00406,0.00406 --suffix _ours_SPSA_vanilla_median --median

# ours vs SPSA vanilla multiple samples (22, half samples as FD)
python plot_opt_baseline.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --filenames ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_all_loss.npy,finite_diff22_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_01_all_loss.npy,finite_diff22_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_001_all_loss.npy,finite_diff22_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_0001_all_loss.npy,finite_diff22_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_00001_all_loss.npy,finite_diff22_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_000001_all_loss.npy --labels ours,multi_SPSA_vanilla_1e-1,multi_SPSA_vanilla_1e-2,multi_SPSA_vanilla_1e-3,multi_SPSA_vanilla_1e-4,multi_SPSA_vanilla_1e-5 --scales 0.00780,0.0486,0.0501,0.0505,0.0500,0.0501 --suffix _ours_multi_SPSA_vanilla_median --median

# ours vs FD ans SPSA on best performing h
python plot_opt_baseline.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --filenames ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_all_loss.npy,finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_001_all_loss.npy,finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_0001_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_001_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_0001_scaled_niter_all_loss.npy,finite_diff27_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_001_all_loss.npy,finite_diff27_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_0001_all_loss.npy --labels ours,FD_1e-2,FD_1e-3,SPSA_1e-2,SPSA_1e-3,multi_SPSA_1e-2,multi_SPSA_1e-3 --scales 0.00780,0.117,0.115,0.00492,0.00499,0.0602,0.0600 --suffix _ours_FD_SPSA_best_h_median --median

# ours vs FD ans vanilla SPSA on best performing h
python plot_opt_baseline.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --filenames ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_001_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_0001_scaled_niter_all_loss.npy,finite_diff22_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_001_all_loss.npy,finite_diff22_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_0001_all_loss.npy --labels ours,SPSA_vanilla_1e-2,SPSA_vanilla_1e-3,multi_SPSA_vanilla_1e-2,multi_SPSA_vanilla_1e-3 --scales 0.00780,0.00407,0.00406,0.0501,0.0505 --suffix _ours_FD_SPSA_vanilla_best_h_median --median

---------------------------------------------
# plot transparent

# ours vs FD ans SPSA on best performing h
python plot_opt_baseline.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --filenames ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_all_loss.npy,finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_001_all_loss.npy,finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_0001_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_001_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_0001_scaled_niter_all_loss.npy,finite_diff27_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_001_all_loss.npy,finite_diff27_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_0001_all_loss.npy --labels ours,FD_1e-2,FD_1e-3,SPSA_1e-2,SPSA_1e-3,multi_SPSA_1e-2,multi_SPSA_1e-3 --scales 0.00780,0.117,0.115,0.00492,0.00499,0.0602,0.0600 --suffix _ours_FD_SPSA_best_h_transparent --transparent

# ours vs FD ans vanilla SPSA on best performing h
python plot_opt_baseline.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --filenames ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_001_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_0001_scaled_niter_all_loss.npy,finite_diff22_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_001_all_loss.npy,finite_diff22_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_0001_all_loss.npy --labels ours,SPSA_vanilla_1e-2,SPSA_vanilla_1e-3,multi_SPSA_vanilla_1e-2,multi_SPSA_vanilla_1e-3 --scales 0.00780,0.00407,0.00406,0.0501,0.0505 --suffix _ours_FD_SPSA_vanilla_best_h_transparent --transparent

---------------------------------------------
# plot metric

python metric_compare_line_integral.py --baseline_dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone --deriv_metric_suffix _2D_kernel_FD_finite_diff_0.100000,_2D_kernel_FD_finite_diff_0.010000,_2D_kernel_FD_finite_diff_0.001000,_2D_kernel_FD_finite_diff_0.000100,_2D_kernel_FD_finite_diff_0.000010,_2D_kernel_SPSA_2_finite_diff_0.100000,_2D_kernel_SPSA_2_finite_diff_0.010000,_2D_kernel_SPSA_2_finite_diff_0.001000,_2D_kernel_SPSA_2_finite_diff_0.000100,_2D_kernel_SPSA_2_finite_diff_0.000010,_2D_kernel --eval_labels FD1e-1,FD1e-2,FD1e-3,FD1e-4,FD1e-5,2SPSA1e-1,2SPSA1e-2,2SPSA1e-3,2SPSA1e-4,2SPSA1e-5,ours --max_half_len 2e-3 --rhs_file /n/fs/scratch/yutingy/test_finite_diff_raymarching_siggraph_cone/random_smooth_metric_2X100000_len_0.002000_2D_kernel_rhs.npy --visualization_thre 0.01 --ncols 5

"""

from render_util import *
from render_single import render_single

compiler.log_prefix_only = False
compiler.log_intermediates_less = True

def cmd_template():
    cmd = f"""python approx_gradient.py --shader test_finite_diff_raymarching_siggraph_cone --init_values_pool apps/example_init_values/test_finite_diff_siggraph_cone_extra_init_values_pool.npy --metrics 5_scale_L2 --render_size 960,960 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file siggraph_gradient.png --multi_scale_optimization --alternating_times 5"""
    return cmd

check_intersect_thre = 0.01

nargs = 43
# args_range currently tuned for best finite_diff result, not necessariliy best normalization for optimziation
args_range = np.array([0.5] * 8 + [0.1] * 4 + [0.02] + [0.1] * 2 + [1] * 28)

#args_range = np.array([0.5] * 16 + [1] * 28)

width = ArgumentScalar('width')
height = ArgumentScalar('height')

raymarching_loop = 64
use_select_rule = 1

form_ellipse_ratio = 0

label_x = 0.0
label_y = 1.0

col0 = np.array([0.7, 0.1, 0.1])
col1 = np.array([0, 0.2, 1])

def test_finite_diff_raymarching_siggraph_cone(u, v, X, scalar_loss_scale):
    """
    X has shape nargs + 3
    first 3 entries are u, v, time
    the other entries are scalar tunable parameters
    """
    
    origin_x, origin_y, origin_z = vec('origin', [X[0], X[1], X[2]])
    ang1, ang2, ang3 = vec('ang', [X[3], X[4], X[5]])
    
    ax_theta, ax_phi = vec('ax_ang', [X[6], X[7]])
    
    cone_theta, cone_phi, cone_alpha, cone_ang = vec('cone_ang', [X[8], X[9], X[10], X[12]])
    ellipse_ratio = Var('ellipse_ratio', X[11])
    
    d1_thre, d2_thre = vec('d_thre', [X[13], X[14]])
    
    angs_lig0_x = vec('angs_lig0_x', [X[15], X[16]])
    angs_lig0_y = vec('angs_lig0_y', [X[17], X[18]])
    
    pos_lig1_x = vec('pos_lig1_x', [X[19], X[20], X[21]])
    pos_lig1_y = vec('pos_lig1_y', [X[22], X[23], X[24]])
    
    amb_x = vec('amb_x', [X[25], X[26], X[27]])
    amb_y = vec('amb_y', [X[28], X[29], X[30]])
    
    kd0_x = vec('kd0_x', [X[31], X[32], X[33]])
    kd0_y = vec('kd0_y', [X[34], X[35], X[36]])
    
    kd1_x = vec('kd1_x', [X[37], X[38], X[39]])
    kd1_y = vec('kd1_y', [X[40], X[41], X[42]])
    
    sin_theta = Var('sin_theta', sin(ax_theta))
    cos_theta = Var('cos_theta', cos(ax_theta))
    sin_phi = Var('sin_phi', sin(ax_phi))
    cos_phi = Var('cos_phi', cos(ax_phi))
    
    sin_cone_theta = Var('sin_cone_theta', sin(cone_theta))
    cos_cone_theta = Var('cos_cone_theta', cos(cone_theta))
    sin_cone_phi = Var('sin_cone_phi', sin(cone_phi))
    cos_cone_phi = Var('cos_cone_phi', cos(cone_phi))
    sin_cone_alpha = Var('sin_cone_alpha', sin(cone_alpha))
    cos_cone_alpha = Var('cos_cone_alpha', cos(cone_alpha))
    
    sin_cone_ang = Var('sin_cone_ang', sin(cone_ang))
    cos_cone_ang = Var('cos_cone_ang', cos(cone_ang))
    
    ax = vec('', [sin_theta * cos_phi,
                  cos_theta * cos_phi,
                  sin_phi],
             style='ax%d')
    
    cone_v0 = vec('', [sin_cone_theta * sin_cone_phi,
                       cos_cone_theta * sin_cone_phi,
                       cos_cone_phi],
                  style='cone_v0_%d')
    cone_v1 = vec('', [sin_cone_theta * cos_cone_phi,
                       cos_cone_theta * cos_cone_phi,
                       -sin_cone_phi],
                  style='cone_v1_%d')
    cone_v2 = vec('', [-cos_cone_theta,
                       sin_cone_theta,
                       ConstExpr(0.)],
                  style='cone_v2_%d')
    

    ro = np.array([origin_x, origin_y, origin_z])


    ray_dir = [u - width / 2, v - height / 2, width / 2]
    rd_norm2 = Var('rd_norm2', ray_dir[0] ** 2 + ray_dir[1] ** 2 + ray_dir[2] ** 2)
    ray_dir_norm = Var('rd_norm',  rd_norm2 ** 0.5)

    ray_dir = [Var('raw_rd0', ray_dir[0] / ray_dir_norm),
               Var('raw_rd1', ray_dir[1] / ray_dir_norm),
               Var('raw_rd2', ray_dir[2] / ray_dir_norm)]

    sin1 = Var('sin1', sin(ang1))
    cos1 = Var('cos1', cos(ang1))
    sin2 = Var('sin2', sin(ang2))
    cos2 = Var('cos2', cos(ang2))
    sin3 = Var('sin3', sin(ang3))
    cos3 = Var('cos3', cos(ang3))

    ray_dir_p = [cos2 * cos3 * ray_dir[0] + (-cos1 * sin3 + sin1 * sin2 * cos3) * ray_dir[1] + (sin1 * sin3 + cos1 * sin2 * cos3) * ray_dir[2],
                 cos2 * sin3 * ray_dir[0] + (cos1 * cos3 + sin1 * sin2 * sin3) * ray_dir[1] + (-sin1 * cos3 + cos1 * sin2 * sin3) * ray_dir[2],
                 -sin2 * ray_dir[0] + sin1 * cos2 * ray_dir[1] + cos1 * cos2 * ray_dir[2]]

    rd = np.array(ray_dir_p)

    rd = [Var('rd0', rd[0]),
          Var('rd1', rd[1]),
          Var('rd2', rd[2])]
    rd = np.array(rd)
    
    def deriv_obj(pos, label, scale, tag):
        
        q = vec('', scale * (pos - ro), style='q%d_' + tag)
        
        pos = pos * scale
        
        d1 = Var('d1_%s' % tag, dot(pos, ax) - d1_thre)
        
        pos_squared = [pos[0] ** 2,
                       pos[1] ** 2,
                       pos[2] ** 2]
        pos_squared = vec('', pos_squared, style='pos%s_squared_' + tag)
        
        dist2 = Var('dist2_%s' % tag, pos_squared[0] + pos_squared[1] + pos_squared[2])
        
        dist = Var('dist_%s' % tag, dist2 ** 0.5)
                
        d2 = Var('d2_%s' % tag, dist - d2_thre)
        
        deriv_d1_pos = ax
        deriv_d2_pos = vec('', pos, style='deriv_d2_pos%d_' + tag)

        cond0 = Var('cond0_%s' % tag, d1 > d2)
        
        t_shell = maximum(d1, d2)
        
        q0 = Var('q0_%s' % tag, dot(q, cone_v0))
        q1 = Var('q1_%s' % tag, dot(q, cone_v1))
        q2 = Var('q2_%s' % tag, dot(q, cone_v2))
        
        r1 = Var('r1_%s' % tag, q1 * cos_cone_alpha + q2 * sin_cone_alpha) * ellipse_ratio
        r2 = Var('r2_%s' % tag, q1 * sin_cone_alpha + q2 * cos_cone_alpha)
        
        scaled_dist = Var('scaled_dist_%s' % tag, (r1 ** 2 + r2 ** 2) ** 0.5)
        
        d3 = Var('d3_%s' % tag, cos_cone_ang * scaled_dist + sin_cone_ang * q0)
                
        res_x = maximum(t_shell, -d3)
        
        deriv_t_shell_pos0 = select(cond0, deriv_d1_pos[0], deriv_d2_pos[0])
        deriv_t_shell_pos1 = select(cond0, deriv_d1_pos[1], deriv_d2_pos[1])
        deriv_t_shell_pos2 = select(cond0, deriv_d1_pos[2], deriv_d2_pos[2])
        
        deriv_t_shell_pos = vec('', [deriv_t_shell_pos0,
                                     deriv_t_shell_pos1,
                                     deriv_t_shell_pos2],
                                style='deriv_t_shell_pos%d_' + tag)
        
        return [res_x, label], deriv_t_shell_pos
    
    def deriv_map_obj(pos):
        
        tag_x = 'x'
        
        resx, derivx = deriv_obj(pos, label_x, np.ones(3), tag_x)
        
        tag_y = 'y'
        
        resy, derivy = deriv_obj(pos, label_y, np.array([-1, -1, 1]), tag_y)
        
        cond_xy = Var('cond_xy', resy[0] - resx[0] > 0)
        
        res0 = minimum(resx[0], resy[0])
        res1 = Var('combined_res1', select(cond_xy, resx[1], resy[1]))
        
        deriv0 = select(cond_xy, derivx[0], -derivy[0])
        deriv1 = select(cond_xy, derivx[1], -derivy[1])
        deriv2 = select(cond_xy, derivx[2], derivy[2])
        
        return res0, res1, deriv0, deriv1, deriv2

    t = 0
    tmax = 10
    t_closest = 0
    res0_closest = 10
    
    def raymarching_body(x, y, z):
        return deriv_map_obj([x, y, z])
    
    raymarching_ans = RaymarchingWrapper(raymarching_body, ro, rd, 0, raymarching_loop, include_derivs=True)
        
    t_closest = raymarching_ans[1]
    res0 = raymarching_ans[4]
    res1 = raymarching_ans[5]
    
    deriv_sdf = [raymarching_ans[6],
                 raymarching_ans[7],
                 raymarching_ans[8]]
    
    pos = vec('pos', ro + rd * t_closest)
    
    obj_label = res1
    
    cond_converge = raymarching_ans[0]
    
    is_valid = Var('is_valid', cond_converge)
    
    deriv_sdf = vec('', deriv_sdf, style='deriv_sdf%d')
    
    nor = normalize(deriv_sdf, prefix='surface_normal')
    
    animate = Animate('animate_raymarching', inout_ls=[nor], in_ls=[pos])
    
    nor, = animate.update()
        
    sin_theta_lig0_x = Var('sin_theta_lig0_x', sin(angs_lig0_x[0]))
    cos_theta_lig0_x = Var('cos_theta_lig0_x', cos(angs_lig0_x[0]))
    sin_phi_lig0_x = Var('sin_phi_lig0_x', sin(angs_lig0_x[1]))
    cos_phi_lig0_x = Var('cos_phi_lig0_x', cos(angs_lig0_x[1]))
    
    sin_theta_lig0_y = Var('sin_theta_lig0_y', sin(angs_lig0_y[0]))
    cos_theta_lig0_y = Var('cos_theta_lig0_y', cos(angs_lig0_y[0]))
    sin_phi_lig0_y = Var('sin_phi_lig0_y', sin(angs_lig0_y[1]))
    cos_phi_lig0_y = Var('cos_phi_lig0_y', cos(angs_lig0_y[1]))
    
    dir_lig0_x = [sin_theta_lig0_x * cos_phi_lig0_x,
                  cos_theta_lig0_x * cos_phi_lig0_x,
                  sin_phi_lig0_x]
    
    dir_lig0_x = vec('', dir_lig0_x, style='dir_lig0_x%d')
    
    dir_lig0_y = [sin_theta_lig0_y * cos_phi_lig0_y,
                  cos_theta_lig0_y * cos_phi_lig0_y,
                  sin_phi_lig0_y]
    
    dir_lig0_y = vec('', dir_lig0_y, style='dir_lig0_y%d')
    
    dot_lig0_x = Var('dot_lig0_x', dot(nor, dir_lig0_x))
    dot_lig0_y = Var('dot_lig0_y', dot(nor, dir_lig0_y))
    
    
    
    dir_lig1_x_diff = vec('', pos_lig1_x - pos, style='dir_lig1_x_diff%d')
    
    dir_lig1_x = normalize(dir_lig1_x_diff, prefix='dir_lig1_x_diff')
    dir_lig1_x = vec('', dir_lig1_x, style='dir_lig1_x%d')
    
    dir_lig1_y_diff = vec('', pos_lig1_y - pos, style='dir_lig1_y_diff%d')
    
    dir_lig1_y = normalize(dir_lig1_y_diff, prefix='dir_lig1_y_diff')
    dir_lig1_y = vec('', dir_lig1_y, style='dir_lig1_y%d')
    
    dot_lig1_x = Var('dot_lig1_x', dot(nor, dir_lig1_x))
    dot_lig1_y = Var('dot_lig1_y', dot(nor, dir_lig1_y))
    
    cond_dif0_x = Var('cond_dif0_x', dot_lig0_x > 0)
    cond_dif0_y = Var('cond_dif0_y', dot_lig0_y > 0)
    cond_dif1_x = Var('cond_dif1_x', dot_lig1_x > 0)
    cond_dif1_y = Var('cond_dif1_y', dot_lig1_y > 0)
    
    # scalar
    dif0_x_sc = Var('dif0_x_sc', select(cond_dif0_x, dot_lig0_x, 0))
    dif0_y_sc = Var('dif0_y_sc', select(cond_dif0_y, dot_lig0_y, 0))
    dif1_x_sc = Var('dif1_x_sc', select(cond_dif1_x, dot_lig1_x, 0))
    dif1_y_sc = Var('dif1_y_sc', select(cond_dif1_y, dot_lig1_y, 0))
    
    # vec3
    dif0_x = dif0_x_sc * kd0_x
    dif0_y = dif0_y_sc * kd0_y
    dif1_x = dif1_x_sc * kd1_x
    dif1_y = dif1_y_sc * kd1_y
    
    col_x = vec('', amb_x + dif0_x + dif1_x, style='col_x%d')
    col_y = vec('', amb_y + dif0_y + dif1_y, style='col_y%d')
    
    col_obj = mix(col_x, col_y, obj_label)
    col_obj = vec('', col_obj, style='col_obj%d')
    
    col_R = Var('col_R', select(is_valid, col_obj[0], 1.0))
    col_G = Var('col_G', select(is_valid, col_obj[1], 1.0))
    col_B = Var('col_B', select(is_valid, col_obj[2], 1.0))
    
    return output_color([col_R, col_G, col_B])

shaders = [test_finite_diff_raymarching_siggraph_cone]
is_color = True

def sample_init(nsamples, target_width=960, target_height=480):
    
    init = np.zeros([nsamples, nargs])
    
    # random sample x, y, z
    init[:, :3] = np.random.rand(nsamples, 3) * 0.4 - 0.2
    init[:, 2] += 2.0
    
    # rejection sampling for camera angle
    for idx in range(nsamples):
        
        obj_dir = -init[idx, :3] / np.linalg.norm(init[idx, :3])
        
        while True:
            ang1, ang2, ang3 = np.random.rand(3) * np.pi * 2
            
            sin1 = np.sin(ang1)
            cos1 = np.cos(ang1)
            sin2 = np.sin(ang2)
            cos2 = np.cos(ang2)
            sin3 = np.sin(ang3)
            cos3 = np.cos(ang3)
            
            # center pixel is [0, 0, 1] in camera frame
            # compute center pixel direction in world frame
            center_dir = np.array([sin1 * sin3 + cos1 * sin2 * cos3,
                                   -sin1 * cos3 + cos1 * sin2 * sin3,
                                   cos1 * cos2])
            
            cos_dir = (obj_dir * center_dir).sum()
            
            if cos_dir > 0.95:
                break
        
        init[idx, 3:6] = [ang1, ang2, ang3]
        
    # random purturbation around a good initial guess (form shadertoy shader)
    init[:, 6:8] = np.random.rand(nsamples, 2) * 0.2
    init[:, 6] -= 0.7
    init[:, 7] += 0.5
    
    init[:, 8:12] = np.random.rand(nsamples, 4) * 0.1
    init[:, 8] += 0.8
    init[:, 9] += 0.15
    init[:, 10] -= 0.05
    init[:, 11] += 0.3
    
    init[:, 12] = np.random.rand(nsamples) * 0.05 + 0.17
    
    init[:, 13:15] = (np.random.rand(nsamples, 3) - 0.5) * 0.1
    init[:, 13] += 0.1
    init[:, 14] += 1
    
    init[:, 15:19] = np.random.rand(nsamples, 4) * 2 * np.pi
    
    init[:, 19:25] = np.random.rand(nsamples, 6) * 6 - 3
    
    init[:, 25:] = np.random.rand(nsamples, 18)
    
    return init

if __name__ == '__main__':
    init = sample_init(100)
    
    np.save('../apps/example_init_values/test_finite_diff_siggraph_cone_extra_init_values_pool.npy', init)
            
        