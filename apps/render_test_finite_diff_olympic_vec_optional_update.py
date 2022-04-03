"""
------------------------------------------------------------------------------------------------------------------------------
# command for visualization

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool apps/example_init_values/test_finite_diff_olympic_10_rings_init_values_pool.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_col --render_size 960,480 --autoscheduler --ignore_glsl --gt_file olympic_rgb.png --gt_transposed

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update_AD --shader test_finite_diff_olympic_vec_optional_update --init_values_pool apps/example_init_values/test_finite_diff_olympic_10_rings_init_values_pool.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization AD --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_col --render_size 960,480 --autoscheduler --ignore_glsl --gt_file olympic_rgb.png --gt_transposed

------------------------------------------------------------------------------------------------------------------------------
# optimization with random var

----------------------------------------------------------------------------------------
# ours
# 44 min for autoscheduler if compile_time_limit = 600

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool apps/example_init_values/test_finite_diff_olympic_tilt_80_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --multi_scale_optimization --alternating_times 5 --suffix _from_real_random --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --no_reset_opt --no_binary_search_std --save_all_loss --autoscheduler --save_best_par --backend hl --gt_transposed --quiet --no_reset_sigma

# ours no random variable

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool apps/example_init_values/test_finite_diff_olympic_tilt_80_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --multi_scale_optimization --alternating_times 5 --suffix _from_real_no_random --no_reset_opt --save_all_loss --autoscheduler --backend hl --gt_transposed --quiet --no_reset_sigma

# ours 5 rings

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_5_rings --shader test_finite_diff_olympic_vec_optional_update --init_values_pool test_finite_diff_olympic_tilt_40_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --multi_scale_optimization --alternating_times 5 --suffix _from_real_random --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --no_reset_opt --no_binary_search_std --save_all_loss --autoscheduler --save_best_par --backend hl --gt_transposed --quiet --no_reset_sigma --shader_args nrings:5 --ignore_glsl

cp /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_5_rings/ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update/ours_both_sides_5_scale_L2_adam_1.0e-02_from_real5_rings_all_loss.npy

----------------------------------------------------------------------------------------
# AD

---------------------------------------------
# same opt as ours and FD

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update_AD --shader test_finite_diff_olympic_vec_optional_update --init_values_pool apps/example_init_values/test_finite_diff_olympic_tilt_80_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization AD --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_random --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --autoscheduler --save_best_par --no_reset_sigma

---------------------------------------------
# vanilla opt
# no random noise, no alternation, no multi stage

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update_AD --shader test_finite_diff_olympic_vec_optional_update --init_values_pool apps/example_init_values/test_finite_diff_olympic_tilt_80_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization AD --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --suffix _from_real_vanilla --backend hl --quiet --no_reset_opt --save_all_loss --autoscheduler --no_reset_sigma

----------------------------------------------------------------------------------------
# Zeroth-order

# vanilla

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool apps/example_init_values/test_finite_diff_olympic_tilt_80_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --suffix _from_real_random --backend hl --quiet --no_reset_opt --save_all_loss --autoscheduler --niters 10000 --optimizer scipy.Nelder-Mead

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool apps/example_init_values/test_finite_diff_olympic_tilt_80_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --suffix _from_real_random --backend hl --quiet --no_reset_opt --save_all_loss --autoscheduler --niters 10000 --optimizer scipy.Powell

----------------------------------------------------------------------------------------
# FD

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool apps/example_init_values/test_finite_diff_olympic_tilt_80_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_fd_h_01 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool apps/example_init_values/test_finite_diff_olympic_tilt_80_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_fd_h_001 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool apps/example_init_values/test_finite_diff_olympic_tilt_80_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.001 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_fd_h_0001 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool apps/example_init_values/test_finite_diff_olympic_tilt_80_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.0001 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_fd_h_00001 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool apps/example_init_values/test_finite_diff_olympic_tilt_80_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.00001 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_fd_h_000001 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --autoscheduler --no_reset_sigma

# 5 rings

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_5_rings --shader test_finite_diff_olympic_vec_optional_update --init_values_pool test_finite_diff_olympic_tilt_40_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_5_rings_fd_h_01 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --autoscheduler --no_reset_sigma --shader_args nrings:5 --ignore_glsl

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_5_rings --shader test_finite_diff_olympic_vec_optional_update --init_values_pool test_finite_diff_olympic_tilt_40_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_5_rings_fd_h_001 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --autoscheduler --no_reset_sigma --shader_args nrings:5 --ignore_glsl

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_5_rings --shader test_finite_diff_olympic_vec_optional_update --init_values_pool test_finite_diff_olympic_tilt_40_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.001 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_5_rings_fd_h_0001 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --autoscheduler --no_reset_sigma --shader_args nrings:5 --ignore_glsl

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_5_rings --shader test_finite_diff_olympic_vec_optional_update --init_values_pool test_finite_diff_olympic_tilt_40_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.0001 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_5_rings_fd_h_00001 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --autoscheduler --no_reset_sigma --shader_args nrings:5 --ignore_glsl

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_5_rings --shader test_finite_diff_olympic_vec_optional_update --init_values_pool test_finite_diff_olympic_tilt_40_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.00001 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_5_rings_fd_h_000001 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --autoscheduler --no_reset_sigma --shader_args nrings:5 --ignore_glsl

cp /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_5_rings/ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update/ours_both_sides_5_scale_L2_adam_1.0e-02_from_real5_rings_all_loss.npy

---------------------------------------------
# No random opt

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool apps/example_init_values/test_finite_diff_olympic_tilt_80_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_no_random_fd_h_01 --backend hl --quiet --no_reset_opt --save_all_loss --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool apps/example_init_values/test_finite_diff_olympic_tilt_80_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_no_random_fd_h_001 --backend hl --quiet --no_reset_opt --save_all_loss --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool apps/example_init_values/test_finite_diff_olympic_tilt_80_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.001 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_no_random_fd_h_0001 --backend hl --quiet --no_reset_opt --save_all_loss --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool apps/example_init_values/test_finite_diff_olympic_tilt_80_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.0001 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_no_random_fd_h_00001 --backend hl --quiet --no_reset_opt --save_all_loss --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool apps/example_init_values/test_finite_diff_olympic_tilt_80_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.00001 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_no_random_fd_h_000001 --backend hl --quiet --no_reset_opt --save_all_loss --autoscheduler --no_reset_sigma

----------------------------------------------------------------------------------------
# SPSA

---------------------------------------------
# same opt as ours and FD
# Runtime
# Ours: 0.0052
# SPSA: 0.0023
# SPSA iter match ours runtime: 4522

-------------------------
# 1 sample (only runned with one choice of h to get a runtime estimate)

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool apps/example_init_values/test_finite_diff_olympic_tilt_80_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_fd_h_01 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --autoscheduler --finite_diff_spsa_samples 1 --no_reset_sigma

-------------------------
# 1 sample iteration scaled by runtime

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool apps/example_init_values/test_finite_diff_olympic_tilt_80_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_fd_h_01_scaled_niter --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --autoscheduler --finite_diff_spsa_samples 1 --niters 4522 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool apps/example_init_values/test_finite_diff_olympic_tilt_80_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_fd_h_001_scaled_niter --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --autoscheduler --finite_diff_spsa_samples 1 --niters 4522 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool apps/example_init_values/test_finite_diff_olympic_tilt_80_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.001 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_fd_h_0001_scaled_niter --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --autoscheduler --finite_diff_spsa_samples 1 --niters 4522 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool apps/example_init_values/test_finite_diff_olympic_tilt_80_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.0001 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_fd_h_00001_scaled_niter --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --autoscheduler --finite_diff_spsa_samples 1 --niters 4522 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool apps/example_init_values/test_finite_diff_olympic_tilt_80_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.00001 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_fd_h_000001_scaled_niter --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --autoscheduler --finite_diff_spsa_samples 1 --niters 4522 --no_reset_sigma

-------------------------
# half sample as FD: 65 samples

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool apps/example_init_values/test_finite_diff_olympic_tilt_80_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_fd_h_01 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --autoscheduler --finite_diff_spsa_samples 65 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool apps/example_init_values/test_finite_diff_olympic_tilt_80_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_fd_h_001 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --autoscheduler --finite_diff_spsa_samples 65 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool apps/example_init_values/test_finite_diff_olympic_tilt_80_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.001 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_fd_h_0001 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --autoscheduler --finite_diff_spsa_samples 65 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool apps/example_init_values/test_finite_diff_olympic_tilt_80_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.0001 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_fd_h_00001 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --autoscheduler --finite_diff_spsa_samples 65 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool apps/example_init_values/test_finite_diff_olympic_tilt_80_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.00001 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_fd_h_000001 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --autoscheduler --finite_diff_spsa_samples 65 --no_reset_sigma

---------------------------------------------
# vanilla opt
# no random noise, no alternation, no multi stage
# Runtime
# Ours: 0.0052
# SPSA: 0.0010
# SPSA iter match ours runtime: 10400

-------------------------
# 1 sample (only runned with one choice of h to get a runtime estimate)

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool apps/example_init_values/test_finite_diff_olympic_tilt_80_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --suffix _from_real_vanilla_fd_h_01 --backend hl --quiet --no_reset_opt --save_all_loss --autoscheduler --finite_diff_spsa_samples 1 --no_reset_sigma

-------------------------
# 1 sample iteration scaled by runtime

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool apps/example_init_values/test_finite_diff_olympic_tilt_80_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --suffix _from_real_vanilla_fd_h_01_scaled_niter --backend hl --quiet --no_reset_opt --save_all_loss --autoscheduler --finite_diff_spsa_samples 1 --niters 10400 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool apps/example_init_values/test_finite_diff_olympic_tilt_80_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --suffix _from_real_vanilla_fd_h_001_scaled_niter --backend hl --quiet --no_reset_opt --save_all_loss --autoscheduler --finite_diff_spsa_samples 1 --niters 10400 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool apps/example_init_values/test_finite_diff_olympic_tilt_80_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.001 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --suffix _from_real_vanilla_fd_h_0001_scaled_niter --backend hl --quiet --no_reset_opt --save_all_loss --autoscheduler --finite_diff_spsa_samples 1 --niters 10400 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool apps/example_init_values/test_finite_diff_olympic_tilt_80_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.0001 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --suffix _from_real_vanilla_fd_h_00001_scaled_niter --backend hl --quiet --no_reset_opt --save_all_loss --autoscheduler --finite_diff_spsa_samples 1 --niters 10400 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool apps/example_init_values/test_finite_diff_olympic_tilt_80_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.00001 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --suffix _from_real_vanilla_fd_h_000001_scaled_niter --backend hl --quiet --no_reset_opt --save_all_loss --autoscheduler --finite_diff_spsa_samples 1 --niters 10400 --no_reset_sigma

-------------------------
# half sample as FD: 40 samples

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool apps/example_init_values/test_finite_diff_olympic_tilt_80_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --suffix _from_real_vanilla_fd_h_01 --backend hl --quiet --no_reset_opt --save_all_loss --autoscheduler --finite_diff_spsa_samples 40 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool apps/example_init_values/test_finite_diff_olympic_tilt_80_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --suffix _from_real_vanilla_fd_h_001 --backend hl --quiet --no_reset_opt --save_all_loss --autoscheduler --finite_diff_spsa_samples 40 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool apps/example_init_values/test_finite_diff_olympic_tilt_80_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.001 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --suffix _from_real_vanilla_fd_h_0001 --backend hl --quiet --no_reset_opt --save_all_loss --autoscheduler --finite_diff_spsa_samples 40 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool apps/example_init_values/test_finite_diff_olympic_tilt_80_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.0001 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --suffix _from_real_vanilla_fd_h_00001 --backend hl --quiet --no_reset_opt --save_all_loss --autoscheduler --finite_diff_spsa_samples 40 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool apps/example_init_values/test_finite_diff_olympic_tilt_80_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.00001 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --suffix _from_real_vanilla_fd_h_000001 --backend hl --quiet --no_reset_opt --save_all_loss --autoscheduler --finite_diff_spsa_samples 40 --no_reset_sigma

---------------------------------------------
# No random but same opt process as ours and FD
# Runtime
# Ours: 0.0052
# SPSA: 0.0011
# SPSA iter match ours runtime: 9455

-------------------------
# 1 sample (only runned with one choice of h to get a runtime estimate)

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool apps/example_init_values/test_finite_diff_olympic_tilt_80_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_no_random_fd_h_01 --backend hl --quiet --no_reset_opt --no_binary_search_std --save_all_loss --autoscheduler --finite_diff_spsa_samples 1 --no_reset_sigma

-------------------------
# 1 sample iteration scaled by runtime

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool apps/example_init_values/test_finite_diff_olympic_tilt_80_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_no_random_fd_h_01_scaled_niter --backend hl --quiet --no_reset_opt --no_binary_search_std --save_all_loss --autoscheduler --finite_diff_spsa_samples 1 --no_reset_sigma --niter 9455

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool apps/example_init_values/test_finite_diff_olympic_tilt_80_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_no_random_fd_h_001_scaled_niter --backend hl --quiet --no_reset_opt --no_binary_search_std --save_all_loss --autoscheduler --finite_diff_spsa_samples 1 --no_reset_sigma --niter 9455

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool apps/example_init_values/test_finite_diff_olympic_tilt_80_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.001 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_no_random_fd_h_0001_scaled_niter --backend hl --quiet --no_reset_opt --no_binary_search_std --save_all_loss --autoscheduler --finite_diff_spsa_samples 1 --no_reset_sigma --niter 9455

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool apps/example_init_values/test_finite_diff_olympic_tilt_80_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.0001 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_no_random_fd_h_00001_scaled_niter --backend hl --quiet --no_reset_opt --no_binary_search_std --save_all_loss --autoscheduler --finite_diff_spsa_samples 1 --no_reset_sigma --niter 9455

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool apps/example_init_values/test_finite_diff_olympic_tilt_80_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.00001 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_no_random_fd_h_000001_scaled_niter --backend hl --quiet --no_reset_opt --no_binary_search_std --save_all_loss --autoscheduler --finite_diff_spsa_samples 1 --no_reset_sigma --niter 9455

-------------------------
# half sample as FD: 40 samples

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool apps/example_init_values/test_finite_diff_olympic_tilt_80_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_no_random_fd_h_01 --backend hl --quiet --no_reset_opt --no_binary_search_std --save_all_loss --autoscheduler --finite_diff_spsa_samples 40 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool apps/example_init_values/test_finite_diff_olympic_tilt_80_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_no_random_fd_h_001 --backend hl --quiet --no_reset_opt --no_binary_search_std --save_all_loss --autoscheduler --finite_diff_spsa_samples 40 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool apps/example_init_values/test_finite_diff_olympic_tilt_80_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.001 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_no_random_fd_h_0001 --backend hl --quiet --no_reset_opt --no_binary_search_std --save_all_loss --autoscheduler --finite_diff_spsa_samples 40 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool apps/example_init_values/test_finite_diff_olympic_tilt_80_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.0001 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_no_random_fd_h_00001 --backend hl --quiet --no_reset_opt --no_binary_search_std --save_all_loss --autoscheduler --finite_diff_spsa_samples 40 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool apps/example_init_values/test_finite_diff_olympic_tilt_80_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.00001 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_no_random_fd_h_000001 --backend hl --quiet --no_reset_opt --no_binary_search_std --save_all_loss --autoscheduler --finite_diff_spsa_samples 40 --no_reset_sigma

----------------------------------------------------------------------------------------
# SPSA 5 rings

---------------------------------------------
# same opt as ours and FD
# Runtime
# Ours: 0.0022
# SPSA: 0.0022
# SPSA iter match ours runtime: 2000

-------------------------
# 1 sample (only runned with one choice of h to get a runtime estimate)

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_5_rings --shader test_finite_diff_olympic_vec_optional_update --init_values_pool test_finite_diff_olympic_tilt_40_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_5_rings_fd_h_01 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --autoscheduler --finite_diff_spsa_samples 1 --no_reset_sigma --shader_args nrings:5 --ignore_glsl

-------------------------
# 1 sample iteration scaled by runtime

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_5_rings --shader test_finite_diff_olympic_vec_optional_update --init_values_pool test_finite_diff_olympic_tilt_40_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_5_rings_fd_h_01_scaled_niter --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --autoscheduler --finite_diff_spsa_samples 1 --no_reset_sigma --shader_args nrings:5 --niters 2000 --ignore_glsl

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_5_rings --shader test_finite_diff_olympic_vec_optional_update --init_values_pool test_finite_diff_olympic_tilt_40_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_5_rings_fd_h_001_scaled_niter --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --autoscheduler --finite_diff_spsa_samples 1 --no_reset_sigma --shader_args nrings:5 --niters 2000 --ignore_glsl

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_5_rings --shader test_finite_diff_olympic_vec_optional_update --init_values_pool test_finite_diff_olympic_tilt_40_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.001 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_5_rings_fd_h_0001_scaled_niter --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --autoscheduler --finite_diff_spsa_samples 1 --no_reset_sigma --shader_args nrings:5 --niters 2000 --ignore_glsl

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_5_rings --shader test_finite_diff_olympic_vec_optional_update --init_values_pool test_finite_diff_olympic_tilt_40_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.0001 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_5_rings_fd_h_00001_scaled_niter --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --autoscheduler --finite_diff_spsa_samples 1 --no_reset_sigma --shader_args nrings:5 --niters 2000 --ignore_glsl

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_5_rings --shader test_finite_diff_olympic_vec_optional_update --init_values_pool test_finite_diff_olympic_tilt_40_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.00001 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_5_rings_fd_h_000001_scaled_niter --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --autoscheduler --finite_diff_spsa_samples 1 --no_reset_sigma --shader_args nrings:5 --niters 2000 --ignore_glsl

-------------------------
# half sample as FD: 33 samples

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_5_rings --shader test_finite_diff_olympic_vec_optional_update --init_values_pool test_finite_diff_olympic_tilt_40_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_5_rings_fd_h_01 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --autoscheduler --finite_diff_spsa_samples 33 --no_reset_sigma --shader_args nrings:5 --ignore_glsl

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_5_rings --shader test_finite_diff_olympic_vec_optional_update --init_values_pool test_finite_diff_olympic_tilt_40_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_5_rings_fd_h_001 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --autoscheduler --finite_diff_spsa_samples 33 --no_reset_sigma --shader_args nrings:5 --ignore_glsl

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_5_rings --shader test_finite_diff_olympic_vec_optional_update --init_values_pool test_finite_diff_olympic_tilt_40_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.001 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_5_rings_fd_h_0001 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --autoscheduler --finite_diff_spsa_samples 33 --no_reset_sigma --shader_args nrings:5 --ignore_glsl

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_5_rings --shader test_finite_diff_olympic_vec_optional_update --init_values_pool test_finite_diff_olympic_tilt_40_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.0001 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_5_rings_fd_h_00001 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --autoscheduler --finite_diff_spsa_samples 33 --no_reset_sigma --shader_args nrings:5 --ignore_glsl

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_5_rings --shader test_finite_diff_olympic_vec_optional_update --init_values_pool test_finite_diff_olympic_tilt_40_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.00001 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_5_rings_fd_h_000001 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --autoscheduler --finite_diff_spsa_samples 33 --no_reset_sigma --shader_args nrings:5 --ignore_glsl

---------------------------------------------
# vanilla opt
# no random noise, no alternation, no multi stage
# Runtime
# Ours: 0.0022
# SPSA: 0.0010
# SPSA iter match ours runtime: 4400

-------------------------
# 1 sample (only runned with one choice of h to get a runtime estimate)

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_5_rings --shader test_finite_diff_olympic_vec_optional_update --init_values_pool test_finite_diff_olympic_tilt_40_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --suffix _from_real_vanilla_5_rings_fd_h_01 --backend hl --quiet --no_reset_opt --save_all_loss --autoscheduler --finite_diff_spsa_samples 1 --no_reset_sigma --shader_args nrings:5 --ignore_glsl

-------------------------
# 1 sample iteration scaled by runtime

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_5_rings --shader test_finite_diff_olympic_vec_optional_update --init_values_pool test_finite_diff_olympic_tilt_40_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --suffix _from_real_vanilla_5_rings_fd_h_01_scaled_niter --backend hl --quiet --no_reset_opt --save_all_loss --autoscheduler --finite_diff_spsa_samples 1 --no_reset_sigma --shader_args nrings:5 --niters 4400 --ignore_glsl

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_5_rings --shader test_finite_diff_olympic_vec_optional_update --init_values_pool test_finite_diff_olympic_tilt_40_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --suffix _from_real_vanilla_5_rings_fd_h_001_scaled_niter --backend hl --quiet --no_reset_opt --save_all_loss --autoscheduler --finite_diff_spsa_samples 1 --no_reset_sigma --shader_args nrings:5 --niters 4400 --ignore_glsl

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_5_rings --shader test_finite_diff_olympic_vec_optional_update --init_values_pool test_finite_diff_olympic_tilt_40_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.001 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --suffix _from_real_vanilla_5_rings_fd_h_0001_scaled_niter --backend hl --quiet --no_reset_opt --save_all_loss --autoscheduler --finite_diff_spsa_samples 1 --no_reset_sigma --shader_args nrings:5 --niters 4400 --ignore_glsl

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_5_rings --shader test_finite_diff_olympic_vec_optional_update --init_values_pool test_finite_diff_olympic_tilt_40_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.0001 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --suffix _from_real_vanilla_5_rings_fd_h_00001_scaled_niter --backend hl --quiet --no_reset_opt --save_all_loss --autoscheduler --finite_diff_spsa_samples 1 --no_reset_sigma --shader_args nrings:5 --niters 4400 --ignore_glsl

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_5_rings --shader test_finite_diff_olympic_vec_optional_update --init_values_pool test_finite_diff_olympic_tilt_40_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.00001 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --suffix _from_real_vanilla_5_rings_fd_h_000001_scaled_niter --backend hl --quiet --no_reset_opt --save_all_loss --autoscheduler --finite_diff_spsa_samples 1 --no_reset_sigma --shader_args nrings:5 --niters 4400 --ignore_glsl

-------------------------
# half sample as FD: 20 samples

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_5_rings --shader test_finite_diff_olympic_vec_optional_update --init_values_pool test_finite_diff_olympic_tilt_40_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --suffix _from_real_vanilla_5_rings_fd_h_01 --backend hl --quiet --no_reset_opt --save_all_loss --autoscheduler --finite_diff_spsa_samples 20 --no_reset_sigma --shader_args nrings:5 --ignore_glsl

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_5_rings --shader test_finite_diff_olympic_vec_optional_update --init_values_pool test_finite_diff_olympic_tilt_40_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --suffix _from_real_vanilla_5_rings_fd_h_001 --backend hl --quiet --no_reset_opt --save_all_loss --autoscheduler --finite_diff_spsa_samples 20 --no_reset_sigma --shader_args nrings:5 --ignore_glsl

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_5_rings --shader test_finite_diff_olympic_vec_optional_update --init_values_pool test_finite_diff_olympic_tilt_40_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.001 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --suffix _from_real_vanilla_5_rings_fd_h_0001 --backend hl --quiet --no_reset_opt --save_all_loss --autoscheduler --finite_diff_spsa_samples 20 --no_reset_sigma --shader_args nrings:5 --ignore_glsl

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_5_rings --shader test_finite_diff_olympic_vec_optional_update --init_values_pool test_finite_diff_olympic_tilt_40_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.0001 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --suffix _from_real_vanilla_5_rings_fd_h_00001 --backend hl --quiet --no_reset_opt --save_all_loss --autoscheduler --finite_diff_spsa_samples 20 --no_reset_sigma --shader_args nrings:5 --ignore_glsl

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_5_rings --shader test_finite_diff_olympic_vec_optional_update --init_values_pool test_finite_diff_olympic_tilt_40_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.00001 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --suffix _from_real_vanilla_5_rings_fd_h_000001 --backend hl --quiet --no_reset_opt --save_all_loss --autoscheduler --finite_diff_spsa_samples 20 --no_reset_sigma --shader_args nrings:5 --ignore_glsl

------------------------------------------------------------------------------------------------------------------------------
# command for generating noisy init for the paper

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update/best_par_from_real_random.npy --modes optimization --metrics 1_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _paper_fig_0 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 5 --quiet --no_reset_opt --no_binary_search_std --autoscheduler --opt_subset_idx 40,41,42,43,44,45,46,47,48,49 --niters 0

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update/best_par_from_real_random.npy --modes optimization --metrics 1_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _paper_fig_1 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 0.2 --quiet --no_reset_opt --no_binary_search_std --autoscheduler --niters 0

------------------------------------------------------------------------------------------------------------------------------
# command for generating glsl in a seperate process

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update/best_par_from_real_random.npy --modes render --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --backend hl --quiet --autoscheduler

------------------------------------------------------------------------------------------------------------------------------
# render optimization process

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool apps/example_init_values/test_finite_diff_olympic_tilt_80_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real_random --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --autoscheduler --save_all_par

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update/ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_result5_0.npy --modes render --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _opt --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --autoscheduler

cd /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update; ffmpeg -i init_opt%05d.png -r 30 -c:v libx264 -preset slow -crf 0 -r 30 animation.mp4; cd /n/fs/shaderml/differentiable_compiler/

# the following command is quicktime compatible
ffmpeg -i init_opt%05d.png -r 30 -c:v libx264 -pix_fmt yuv420p -r 30 animation.mp4

# without random var

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool apps/example_init_values/test_finite_diff_olympic_tilt_80_param_init_values_pool_extra.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _from_real --backend hl --quiet --no_reset_opt --save_all_loss --autoscheduler --save_all_par

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update/ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_result5_0.npy --modes render --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 960,480 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file olympic_rgb.png --gt_transposed --multi_scale_optimization --alternating_times 5 --suffix _opt_no_random --backend hl --quiet --no_reset_opt --save_all_loss --autoscheduler --save_all_par --unnormalized_par

cd /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update; ffmpeg -i init_opt_no_random%05d.png -r 30 -c:v libx264 -preset slow -crf 0 -r 30 animation_no_random.mp4; cd /n/fs/shaderml/differentiable_compiler/

# the following command is quicktime compatible
ffmpeg -i init_opt_no_random%05d.png -r 30 -c:v libx264 -pix_fmt yuv420p -r 30 animation_no_random.mp4

------------------------------------------------------------------------------------------------------------------------------
# Quantitative Metric using 2D box kernels
# Optimization runtime
# Ours: 0.0052
# SPSA 1 sample: 0.0023
# FD 1 sample: 0.0784

# SPSA need 2 samples for similar runtime with ours in opt

# get endpoints
python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update/best_par_from_real_random.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_col --backend hl --line_endpoints_method random_smooth --deriv_metric_visualization_thre 0.1 --deriv_metric_max_halflen 3e-3 --deriv_metric_line --deriv_metric_finite_diff_schedule 0 --kernel_nsamples 1 --deriv_n 2 --deriv_metric_suffix _2D_kernel --our_filter_direction 2d --render_size 960,480 --autoscheduler --ignore_glsl --kernel_sigma 0.1

# rhs

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update/best_par_from_real_random.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_col --backend hl --line_endpoints_method random_smooth --deriv_metric_visualization_thre 0.1 --deriv_metric_max_halflen 3e-3 --deriv_metric_line --deriv_metric_finite_diff_schedule 0 --kernel_nsamples 100000 --deriv_n 2 --deriv_metric_suffix _2D_kernel --our_filter_direction 2d --render_size 960,480 --autoscheduler --ignore_glsl --kernel_sigma 0.1 --deriv_metric_endpoint_file /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update/random_smooth_metric_2X1_len_0.003000_2D_kernel_endpoints.npy

----------------------------------------------------------------------------------------
# ours

# 1 sample

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update/best_par_from_real_random.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_col --backend hl --line_endpoints_method kernel_smooth_debug --deriv_metric_visualization_thre 0.1 --deriv_metric_max_halflen 3e-3 --deriv_metric_line --deriv_metric_finite_diff_schedule 0 --kernel_nsamples 1 --deriv_n 10000 --deriv_metric_suffix _2D_kernel --kernel_sigma 0.1 --kernel_uv_sigma 1 --kernel_smooth_exclude_our_kernel --render_size 960,480 --deriv_metric_endpoint_file /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update/random_smooth_metric_2X1_len_0.003000_2D_kernel_endpoints.npy --deriv_metric_rhs_file /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update/random_smooth_metric_2X100000_len_0.003000_2D_kernel_rhs.npy --ignore_glsl --autoscheduler

----------------------------------------------------------------------------------------
# FD

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update/best_par_from_real_random.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_col --backend hl --line_endpoints_method kernel_smooth_debug --deriv_metric_visualization_thre 0.1 --deriv_metric_max_halflen 3e-3 --deriv_metric_line --deriv_metric_finite_diff_schedule 1e-1,1e-2,1e-3,1e-4,1e-5 --kernel_nsamples 1 --deriv_n 10000 --deriv_metric_suffix _2D_kernel_FD --kernel_sigma 0.1 --kernel_uv_sigma 1 --kernel_smooth_exclude_our_kernel --render_size 960,480 --deriv_metric_no_ours --deriv_metric_endpoint_file /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update/random_smooth_metric_2X1_len_0.003000_2D_kernel_endpoints.npy --deriv_metric_rhs_file /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update/random_smooth_metric_2X100000_len_0.003000_2D_kernel_rhs.npy --ignore_glsl --autoscheduler
 
----------------------------------------------------------------------------------------
# SPSA
# 2 samples
# number of samples corresponds to runtime in our optimization pipeline (with random, multi stage loss)

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --shader test_finite_diff_olympic_vec_optional_update --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update/best_par_from_real_random.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_col --backend hl --line_endpoints_method kernel_smooth_debug --deriv_metric_visualization_thre 0.1 --deriv_metric_max_halflen 3e-3 --deriv_metric_line --deriv_metric_finite_diff_schedule 1e-1,1e-2,1e-3,1e-4,1e-5 --kernel_nsamples 1 --deriv_n 10000 --deriv_metric_suffix _2D_kernel_SPSA_2 --kernel_sigma 0.1 --kernel_uv_sigma 1 --kernel_smooth_exclude_our_kernel --render_size 960,480 --deriv_metric_no_ours --deriv_metric_endpoint_file /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update/random_smooth_metric_2X1_len_0.003000_2D_kernel_endpoints.npy --deriv_metric_rhs_file /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update/random_smooth_metric_2X100000_len_0.003000_2D_kernel_rhs.npy --finite_diff_spsa_samples 2 --ignore_glsl --autoscheduler

----------------------------------------------------------------------------------------
# plot

---------------------------------------------
# preprocess

python preprocess_raw_loss_data.py /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update  /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_000001_scaled_niter_all_loss.npy  /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_00001_scaled_niter_all_loss.npy  /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_0001_scaled_niter_all_loss.npy  /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_001_scaled_niter_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_01_scaled_niter_all_loss.npy  /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_000001_scaled_niter_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_00001_scaled_niter_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_0001_scaled_niter_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_001_scaled_niter_all_loss.npy  /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_01_scaled_niter_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update/finite_diff40_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_000001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update/finite_diff40_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_00001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update/finite_diff40_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_0001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update/finite_diff40_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update/finite_diff40_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_01_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update/finite_diff65_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_000001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update/finite_diff65_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_00001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update/finite_diff65_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_0001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update/finite_diff65_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update/finite_diff65_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_01_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update/finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_000001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update/finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_00001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update/finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_0001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update/finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update/finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_01_all_loss.npy  /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update/ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_all_loss.npy

---------------------------------------------
# plot median 

# ours vs FD
python plot_opt_baseline.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --filenames ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_all_loss.npy,finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_01_all_loss.npy,finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_001_all_loss.npy,finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_0001_all_loss.npy,finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_00001_all_loss.npy,finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_00001_all_loss.npy --labels ours,FD_1e-1,FD_1e-2,FD_1e-3,FD_1e-4,FD_1e-5  --scales 0.0052,0.0812,0.0811,0.0759,0.0778,0.0739 --suffix _ours_FD_median --median

# ours vs SPSA
python plot_opt_baseline.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --filenames ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_01_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_001_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_0001_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_00001_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_000001_scaled_niter_all_loss.npy --labels ours,SPSA_1e-1,SPSA_1e-2,SPSA_1e-3,SPSA_1e-4,SPSA_1e-5 --scales 0.0052,0.0018,0.0018,0.0018,0.0021,0.0033 --suffix _ours_SPSA_median --median

# ours vs SPSA multiple samples (65, half samples as FD)
python plot_opt_baseline.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --filenames ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_all_loss.npy,finite_diff65_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_01_all_loss.npy,finite_diff65_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_001_all_loss.npy,finite_diff65_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_0001_all_loss.npy,finite_diff65_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_00001_all_loss.npy,finite_diff65_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_000001_all_loss.npy --labels ours,multi_SPSA_1e-1,multi_SPSA_1e-2,multi_SPSA_1e-3,multi_SPSA_1e-4,multi_SPSA_1e-5 --scales 0.0052,0.0434,0.0431,0.0423,0.0408,0.0405 --suffix _ours_multi_SPSA_median --median

# ours vs SPSA vanilla
python plot_opt_baseline.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --filenames ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_01_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_001_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_0001_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_00001_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_000001_scaled_niter_all_loss.npy --labels ours,SPSA_vanilla_1e-1,SPSA_vanilla_1e-2,SPSA_vanilla_1e-3,SPSA_vanilla_1e-4,SPSA_vanilla_1e-5 --scales 0.0052,0.00096,0.00099,0.00098,0.00097,0.00097 --suffix _ours_SPSA_vanilla_median --median

# ours vs SPSA vanilla multiple samples (40, half samples as FD)
python plot_opt_baseline.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --filenames ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_all_loss.npy,finite_diff40_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_01_all_loss.npy,finite_diff40_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_001_all_loss.npy,finite_diff40_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_0001_all_loss.npy,finite_diff40_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_00001_all_loss.npy,finite_diff40_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_000001_all_loss.npy --labels ours,multi_SPSA_vanilla_1e-1,multi_SPSA_vanilla_1e-2,multi_SPSA_vanilla_1e-3,multi_SPSA_vanilla_1e-4,multi_SPSA_vanilla_1e-5 --scales 0.0052,0.0196,0.0193,0.0194,0.0194,0.0251 --suffix _ours_multi_SPSA_vanilla_median --median

# ours vs FD ans SPSA on best performing h
python plot_opt_baseline.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --filenames ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_all_loss.npy,finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_001_all_loss.npy,finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_0001_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_001_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_0001_scaled_niter_all_loss.npy,finite_diff65_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_001_all_loss.npy,finite_diff65_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_0001_all_loss.npy --labels ours,FD_1e-2,FD_1e-3,SPSA_1e-2,SPSA_1e-3,multi_SPSA_1e-2,multi_SPSA_1e-3 --scales 0.0052,0.0811,0.0759,0.0018,0.0018,0.0431,0.0423 --suffix _ours_FD_SPSA_best_h_median --median

# ours vs FD ans SPSA vanilla on best performing h
python plot_opt_baseline.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --filenames ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_001_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_0001_scaled_niter_all_loss.npy,finite_diff40_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_001_all_loss.npy,finite_diff40_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_0001_all_loss.npy --labels ours,SPSA_vanilla_1e-2,SPSA_vanilla_1e-3,multi_SPSA_vanilla_1e-2,multi_SPSA_vanilla_1e-3 --scales 0.0052,0.00099,0.00098,0.0193,0.0194 --suffix _ours_FD_SPSA_vanilla_best_h_median --median

---------------------------------------------
# plot transparent

# ours vs FD ans SPSA on best performing h
python plot_opt_baseline.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --filenames ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_all_loss.npy,finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_001_all_loss.npy,finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_0001_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_001_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_0001_scaled_niter_all_loss.npy,finite_diff65_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_001_all_loss.npy,finite_diff65_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_h_0001_all_loss.npy --labels ours,FD_1e-2,FD_1e-3,SPSA_1e-2,SPSA_1e-3,multi_SPSA_1e-2,multi_SPSA_1e-3 --scales 0.0052,0.0811,0.0759,0.0018,0.0018,0.0431,0.0423 --suffix _ours_FD_SPSA_best_h_transparent --transparent

# ours vs FD ans SPSA vanilla on best performing h
python plot_opt_baseline.py --dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --filenames ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_001_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_0001_scaled_niter_all_loss.npy,finite_diff40_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_001_all_loss.npy,finite_diff40_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_0001_all_loss.npy --labels ours,SPSA_vanilla_1e-2,SPSA_vanilla_1e-3,multi_SPSA_vanilla_1e-2,multi_SPSA_vanilla_1e-4 --scales 0.0052,0.00099,0.00098,0.0193,0.0194 --suffix _ours_FD_SPSA_vanilla_best_h_transparent --transparent

---------------------------------------------
# plot metric

python metric_compare_line_integral.py --baseline_dir /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update --deriv_metric_suffix _2D_kernel_FD_finite_diff_0.100000,_2D_kernel_FD_finite_diff_0.010000,_2D_kernel_FD_finite_diff_0.001000,_2D_kernel_FD_finite_diff_0.000100,_2D_kernel_FD_finite_diff_0.000010,_2D_kernel_SPSA_2_finite_diff_0.100000,_2D_kernel_SPSA_2_finite_diff_0.010000,_2D_kernel_SPSA_2_finite_diff_0.001000,_2D_kernel_SPSA_2_finite_diff_0.000100,_2D_kernel_SPSA_2_finite_diff_0.000010,_2D_kernel --eval_labels FD1e-1,FD1e-2,FD1e-3,FD1e-4,FD1e-5,2SPSA1e-1,2SPSA1e-2,2SPSA1e-3,2SPSA1e-4,2SPSA1e-5,ours --max_half_len 3e-3 --rhs_file /n/fs/scratch/yutingy/test_finite_diff_olympic_vec_optional_update/random_smooth_metric_2X100000_len_0.003000_2D_kernel_rhs.npy --visualization_thre 0.01 --ncols 5

"""

from render_util import *
from render_single import render_single

compiler.log_prefix_only = False
compiler.log_intermediates_less = True

nrings = 10
nargs = 8 * nrings
args_range = np.ones(nargs)
sigmas_range = np.ones(nargs)
max_iter = nrings - 1

def update_args():
    
    global nrings, nargs, args_range, sigmas_range, max_iter

    nargs = 8 * nrings
    args_range = np.ones(nargs)

    # pos_x, pos_y
    args_range[:2*nrings] = 200
    # radius
    args_range[2*nrings:3*nrings] = 100
    # radius_scale
    args_range[3*nrings:4*nrings] = 1
    # tilt
    args_range[4*nrings:5*nrings] = 10
    # col
    args_range[5*nrings:] = 1

    sigmas_range = args_range
    
    max_iter = nrings - 1
    
update_args()


width = ArgumentScalar('width')
height = ArgumentScalar('height')



#random_var_indices = np.arange(10).astype(int) + 43
random_var_indices = np.arange(50).astype(int) + 3
default_phase = -1e4

use_select_rule = 1

def test_finite_diff_olympic_vec_optional_update(u, v, X, scalar_loss_scale):
    
    rings = []
    for i in range(nrings):
        
        ring_params = [X[i + k * nrings] for k in range(8)]
        ring = Object('ring', 
                      pos = ring_params[:2],
                      radius = ring_params[2],
                      radius_scale = ring_params[3],
                      tilt = ring_params[4],
                      col = ring_params[5:])
        rings.append(ring)
    
    def update_ring(old_vals, ring):
        # Update function should be side-effect free
        
        old_col, old_phase = old_vals[0], old_vals[1]
        
        rel_pos = np.array([u, v]) - ring.pos
        
        dist2 = Var('dist2_%s' % ring.name, rel_pos[0] ** 2 + rel_pos[1] ** 2)
        dist = Var('dist_%s' % ring.name, dist2 ** 0.5)
        
        phase = Var('phase_raw_%s' % ring.name, rel_pos[0] * ring.tilt)
        
        phase_diff = Var('phase_diff_%s' % ring.name, phase - old_phase)
        cond0_diff = Var('cond0_diff_%s' % ring.name, dist - ring.radius)
        cond1_diff = Var('cond1_diff_%s' % ring.name, dist - ring.radius * ring.radius_scale)
        
        cond0 = cond0_diff < 0
        cond1 = cond1_diff > 0
        cond2 = phase_diff > 0
        
        cond0 = Var('cond0_%s' % ring.name, cond0)
        cond1 = Var('cond1_%s' % ring.name, cond1)
        cond2 = Var('cond2_%s' % ring.name, cond2)
        
        cond_all = Var('cond_all_%s' % ring.name, cond0 & cond1 & cond2)
        
        col = Var('col_%s' % ring.name, select(cond_all, ring.col, old_col))
        out_phase = Var('phase_%s' % ring.name, select(cond_all, phase, old_phase))
        
        return [col, out_phase]

    global default_phase
    # BG
    col = Compound([1, 1, 1])
    
    vals = [col, default_phase]
    
    for i in range(nrings):
        possible_update(vals, update_ring, rings[i])
        
    return vals[0]

shaders = [test_finite_diff_olympic_vec_optional_update]
is_color = True

def sample_init(nsamples, target_width=960, target_height=480):
    
    init = np.zeros([nsamples, nargs])
    
    init[:, :10] = np.random.rand(nsamples, 10) * 0.6 * taget_width + 0.2 * target_width
    init[:, 10:20] = np.random.rand(nsamples, 10) * 0.6 * target_height + 0.2 * target_height
    init[:, 20:30] = np.random.rand(nsamples, 10) * 100 + 100
    init[:, 30:40] = np.random.rand(nsamples, 10) * 0.5 + 0.4
    init[:, 40:50] = np.random.rand(nsamples, 10) * 2 - 1
    init[:, 50:] = np.random.rand(nsamples, 30)
    
    return init