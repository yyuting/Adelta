"""
------------------------------------------------------------------------------------------------------------------------------
# command for visualization

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 640,640 --is_color --backend hl --quiet --gt_file tf_logo.png --gt_transposed --autoscheduler --save_best_par

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo_AD --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization AD --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 640,640 --is_color --backend hl --quiet --gt_file tf_logo.png --gt_transposed --autoscheduler --save_best_par

------------------------------------------------------------------------------------------------------------------------------
# optimization with random par

----------------------------------------------------------------------------------------
# ours

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --suffix _from_real_random --gt_file tf_logo.png --gt_transposed --autoscheduler --save_best_par --no_reset_sigma

# ours without random variables

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --backend hl --quiet --no_reset_opt --save_all_loss --suffix _from_real_no_random --gt_file tf_logo.png --gt_transposed --autoscheduler --no_reset_sigma

----------------------------------------------------------------------------------------
# AD

---------------------------------------------
# same opt as ours and FD

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo_AD --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization AD --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --suffix _from_real_random --gt_file tf_logo.png --gt_transposed --autoscheduler --save_best_par --no_reset_sigma

---------------------------------------------
# vanilla opt
# no random noise, no alternation, no multi stage

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo_AD --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization AD --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --backend hl --quiet --no_reset_opt --save_all_loss --suffix _from_real_vanilla --gt_file tf_logo.png --gt_transposed --autoscheduler --no_reset_sigma

----------------------------------------------------------------------------------------
# Zeroth-Order

---------------------------------------------
# vanilla

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --backend hl --quiet --no_reset_opt --save_all_loss --suffix _from_real_vanilla --gt_file tf_logo.png --gt_transposed --autoscheduler --optimizer scipy.Nelder-Mead --niters 25000

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --backend hl --quiet --no_reset_opt --save_all_loss --suffix _from_real_vanilla --gt_file tf_logo.png --gt_transposed --autoscheduler --optimizer scipy.Powell --niters 25000

----------------------------------------------------------------------------------------
# FD

---------------------------------------------
# same opt as ours and FD

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --suffix _from_real_random_fd_01 --gt_file tf_logo.png --gt_transposed --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --suffix _from_real_random_fd_001 --gt_file tf_logo.png --gt_transposed --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.001 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --suffix _from_real_random_fd_0001 --gt_file tf_logo.png --gt_transposed --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.0001 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --suffix _from_real_random_fd_00001 --gt_file tf_logo.png --gt_transposed --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.00001 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --suffix _from_real_random_fd_000001 --gt_file tf_logo.png --gt_transposed --autoscheduler --no_reset_sigma

---------------------------------------------
# No random opt

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --backend hl --quiet --no_reset_opt --save_all_loss --suffix _from_real_no_random_fd_01 --gt_file tf_logo.png --gt_transposed --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --backend hl --quiet --no_reset_opt --save_all_loss --suffix _from_real_no_random_fd_001 --gt_file tf_logo.png --gt_transposed --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.001 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --backend hl --quiet --no_reset_opt --save_all_loss --suffix _from_real_no_random_fd_0001 --gt_file tf_logo.png --gt_transposed --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.0001 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --backend hl --quiet --no_reset_opt --save_all_loss --suffix _from_real_no_random_fd_00001 --gt_file tf_logo.png --gt_transposed --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.00001 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --backend hl --quiet --no_reset_opt --save_all_loss --suffix _from_real_no_random_fd_000001 --gt_file tf_logo.png --gt_transposed --autoscheduler --no_reset_sigma

---------------------------------------------
# vanilla opt
# no random noise, no alternation, no multi stage

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --backend hl --quiet --no_reset_opt --save_all_loss --suffix _from_real_vanilla_fd_h_01 --gt_file tf_logo.png --gt_transposed --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --backend hl --quiet --no_reset_opt --save_all_loss --suffix _from_real_vanilla_fd_h_001 --gt_file tf_logo.png --gt_transposed --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.001 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --backend hl --quiet --no_reset_opt --save_all_loss --suffix _from_real_vanilla_fd_h_0001 --gt_file tf_logo.png --gt_transposed --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.0001 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --backend hl --quiet --no_reset_opt --save_all_loss --suffix _from_real_vanilla_fd_h_00001 --gt_file tf_logo.png --gt_transposed --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.00001 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --backend hl --quiet --no_reset_opt --save_all_loss --suffix _from_real_vanilla_fd_h_000001 --gt_file tf_logo.png --gt_transposed --autoscheduler --no_reset_sigma

----------------------------------------------------------------------------------------
# SPSA

---------------------------------------------
# same opt as ours
# Runtime
# Ours: 0.0068
# SPSA: 0.0016
# SPSA iter match ours runtime: 8500

-------------------------
# 1 sample (only runned with one choice of h to get a runtime estimate)

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --suffix _from_real_random_fd_01 --gt_file tf_logo.png --gt_transposed --finite_diff_spsa_samples 1 --autoscheduler --no_reset_sigma

-------------------------
# 1 sample iteration scaled by runtime

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --suffix _from_real_random_fd_01_scaled_niter --gt_file tf_logo.png --gt_transposed --finite_diff_spsa_samples 1 --niters 8500 --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --suffix _from_real_random_fd_001_scaled_niter --gt_file tf_logo.png --gt_transposed --finite_diff_spsa_samples 1 --niters 8500 --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.001 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --suffix _from_real_random_fd_0001_scaled_niter --gt_file tf_logo.png --gt_transposed --finite_diff_spsa_samples 1 --niters 8500 --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.0001 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --suffix _from_real_random_fd_00001_scaled_niter --gt_file tf_logo.png --gt_transposed --finite_diff_spsa_samples 1 --niters 8500 --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.00001 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --suffix _from_real_random_fd_000001_scaled_niter --gt_file tf_logo.png --gt_transposed --finite_diff_spsa_samples 1 --niters 8500 --autoscheduler --no_reset_sigma

-------------------------
# half sample as FD: 27 samples

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --suffix _from_real_random_fd_01 --gt_file tf_logo.png --gt_transposed --finite_diff_spsa_samples 27 --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --suffix _from_real_random_fd_001 --gt_file tf_logo.png --gt_transposed --finite_diff_spsa_samples 27 --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.001 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --suffix _from_real_random_fd_0001 --gt_file tf_logo.png --gt_transposed --finite_diff_spsa_samples 27 --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.0001 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --suffix _from_real_random_fd_00001 --gt_file tf_logo.png --gt_transposed --finite_diff_spsa_samples 27 --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.00001 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --suffix _from_real_random_fd_000001 --gt_file tf_logo.png --gt_transposed --finite_diff_spsa_samples 27 --autoscheduler --no_reset_sigma

---------------------------------------------
# vanilla opt
# no random noise, no alternation, no multi stage
# Runtime
# Ours: 0.0068
# SPSA: 0.0009
# SPSA iter match ours runtime: 15111

-------------------------
# 1 sample (only runned with one choice of h to get a runtime estimate)

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --backend hl --quiet --no_reset_opt --save_all_loss --suffix _from_real_vanilla_fd_h_01 --gt_file tf_logo.png --gt_transposed --finite_diff_spsa_samples 1 --autoscheduler --no_reset_sigma

-------------------------
# 1 sample iteration scaled by runtime

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --backend hl --quiet --no_reset_opt --save_all_loss --suffix _from_real_vanilla_fd_h_01_scaled_niter --gt_file tf_logo.png --gt_transposed --finite_diff_spsa_samples 1 --autoscheduler --niters 15111 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --backend hl --quiet --no_reset_opt --save_all_loss --suffix _from_real_vanilla_fd_h_001_scaled_niter --gt_file tf_logo.png --gt_transposed --finite_diff_spsa_samples 1 --autoscheduler --niters 15111 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.001 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --backend hl --quiet --no_reset_opt --save_all_loss --suffix _from_real_vanilla_fd_h_0001_scaled_niter --gt_file tf_logo.png --gt_transposed --finite_diff_spsa_samples 1 --autoscheduler --niters 15111 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.0001 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --backend hl --quiet --no_reset_opt --save_all_loss --suffix _from_real_vanilla_fd_h_00001_scaled_niter --gt_file tf_logo.png --gt_transposed --finite_diff_spsa_samples 1 --autoscheduler --niters 15111 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.00001 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --backend hl --quiet --no_reset_opt --save_all_loss --suffix _from_real_vanilla_fd_h_000001_scaled_niter --gt_file tf_logo.png --gt_transposed --finite_diff_spsa_samples 1 --autoscheduler --niters 15111 --no_reset_sigma

-------------------------
# half sample as FD: 16 samples

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --backend hl --quiet --no_reset_opt --save_all_loss --suffix _from_real_vanilla_fd_h_01 --gt_file tf_logo.png --gt_transposed --finite_diff_spsa_samples 16 --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --backend hl --quiet --no_reset_opt --save_all_loss --suffix _from_real_vanilla_fd_h_001 --gt_file tf_logo.png --gt_transposed --finite_diff_spsa_samples 16 --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.001 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --backend hl --quiet --no_reset_opt --save_all_loss --suffix _from_real_vanilla_fd_h_0001 --gt_file tf_logo.png --gt_transposed --finite_diff_spsa_samples 16 --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.0001 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --backend hl --quiet --no_reset_opt --save_all_loss --suffix _from_real_vanilla_fd_h_00001 --gt_file tf_logo.png --gt_transposed --finite_diff_spsa_samples 16 --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.00001 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --backend hl --quiet --no_reset_opt --save_all_loss --suffix _from_real_vanilla_fd_h_000001 --gt_file tf_logo.png --gt_transposed --finite_diff_spsa_samples 16 --autoscheduler --no_reset_sigma

---------------------------------------------
# no random but same opt as ours and FD
# Runtime
# Ours: 0.0068
# SPSA: 0.0008
# SPSA iter match ours runtime: 17000

-------------------------
# 1 sample (only runned with one choice of h to get a runtime estimate)

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --backend hl --quiet --no_reset_opt --no_binary_search_std --save_all_loss --suffix _from_real_no_random_fd_01 --gt_file tf_logo.png --gt_transposed --finite_diff_spsa_samples 1 --autoscheduler --no_reset_sigma

-------------------------
# 1 sample iteration scaled by runtime

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --backend hl --quiet --no_reset_opt --no_binary_search_std --save_all_loss --suffix _from_real_no_random_fd_01_scaled_niter --gt_file tf_logo.png --gt_transposed --finite_diff_spsa_samples 1 --autoscheduler --no_reset_sigma --niter 17000

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --backend hl --quiet --no_reset_opt --no_binary_search_std --save_all_loss --suffix _from_real_no_random_fd_001_scaled_niter --gt_file tf_logo.png --gt_transposed --finite_diff_spsa_samples 1 --autoscheduler --no_reset_sigma --niter 17000

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.001 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --backend hl --quiet --no_reset_opt --no_binary_search_std --save_all_loss --suffix _from_real_no_random_fd_0001_scaled_niter --gt_file tf_logo.png --gt_transposed --finite_diff_spsa_samples 1 --autoscheduler --no_reset_sigma --niter 17000

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.0001 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --backend hl --quiet --no_reset_opt --no_binary_search_std --save_all_loss --suffix _from_real_no_random_fd_00001_scaled_niter --gt_file tf_logo.png --gt_transposed --finite_diff_spsa_samples 1 --autoscheduler --no_reset_sigma --niter 17000

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.00001 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --backend hl --quiet --no_reset_opt --no_binary_search_std --save_all_loss --suffix _from_real_no_random_fd_000001_scaled_niter --gt_file tf_logo.png --gt_transposed --finite_diff_spsa_samples 1 --autoscheduler --no_reset_sigma --niter 17000

-------------------------
# half sample as FD: 16 samples

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --backend hl --quiet --no_reset_opt --no_binary_search_std --save_all_loss --suffix _from_real_no_random_fd_01 --gt_file tf_logo.png --gt_transposed --finite_diff_spsa_samples 16 --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --backend hl --quiet --no_reset_opt --no_binary_search_std --save_all_loss --suffix _from_real_no_random_fd_001 --gt_file tf_logo.png --gt_transposed --finite_diff_spsa_samples 16 --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.001 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --backend hl --quiet --no_reset_opt --no_binary_search_std --save_all_loss --suffix _from_real_no_random_fd_0001 --gt_file tf_logo.png --gt_transposed --finite_diff_spsa_samples 16 --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.0001 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --backend hl --quiet --no_reset_opt --no_binary_search_std --save_all_loss --suffix _from_real_no_random_fd_00001 --gt_file tf_logo.png --gt_transposed --finite_diff_spsa_samples 16 --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.00001 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --backend hl --quiet --no_reset_opt --no_binary_search_std --save_all_loss --suffix _from_real_no_random_fd_000001 --gt_file tf_logo.png --gt_transposed --finite_diff_spsa_samples 16 --autoscheduler --no_reset_sigma

------------------------------------------------------------------------------------------------------------------------------
# animation

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_structured_tf_logo_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --gt_file tf_logo.png --gt_transposed --save_all_par --autoscheduler

# GLSL code

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo/best_par_from_real_random.npy --modes render --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --backend hl --quiet --gt_file tf_logo.png --gt_transposed --unnormalized_par --autoscheduler

# opt animation

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo/ours_both_sides_5_scale_L2_adam_1.0e-02_result0_0.npy --modes render --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --gt_file tf_logo.png --gt_transposed --save_best_par --save_all_par --suffix _opt --autoscheduler

ffmpeg -i /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo/init_opt%05d.png -r 30 -c:v libx264 -preset slow -crf 0 -r 30 /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo/animation.mp4

------------------------------------------------------------------------------------------------------------------------------
# Quantitative Metric using 2D box kernels

# Ours Runtime: 0.0068
# SPSA Runtime: 0.0016

# get endpoints

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo/best_par_from_real_random.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_col --render_size 640,640 --line_endpoints_method random_smooth --deriv_metric_visualization_thre 0.1 --deriv_metric_max_halflen 2e-3 --deriv_metric_line --deriv_metric_finite_diff_schedule 0 --kernel_nsamples 1 --deriv_n 2 --deriv_metric_no_ours --our_filter_direction 2d --deriv_metric_suffix _2D_kernel --gt_file tf_logo.png --gt_transposed --autoscheduler

# rhs

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo/best_par_from_real_random.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_col --render_size 640,640 --line_endpoints_method random_smooth --deriv_metric_visualization_thre 0.1 --deriv_metric_max_halflen 2e-3 --deriv_metric_line --deriv_metric_finite_diff_schedule 0 --kernel_nsamples 100000 --deriv_n 2 --deriv_metric_no_ours --our_filter_direction 2d --deriv_metric_suffix _2D_kernel --deriv_metric_endpoint_file /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo/random_smooth_metric_2X1_len_0.002000_2D_kernel_endpoints.npy --gt_file tf_logo.png --gt_transposed --autoscheduler --ignore_glsl --kernel_sigma 0.1

----------------------------------------------------------------------------------------
# ours

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo/best_par_from_real_random.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_color --render_size 640,640 --line_endpoints_method kernel_smooth_debug --deriv_metric_visualization_thre 0.1 --deriv_metric_max_halflen 2e-3 --deriv_metric_line --deriv_metric_endpoint_file /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo/random_smooth_metric_2X1_len_0.002000_2D_kernel_endpoints.npy --deriv_metric_rhs_file /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo/random_smooth_metric_2X1_len_0.002000_2D_kernel_rhs.npy --deriv_metric_suffix _2D_kernel --kernel_sigma 0 --kernel_uv_sigma 1 --deriv_n 10000 --deriv_metric_finite_diff_schedule 0 --kernel_smooth_exclude_our_kernel --kernel_nsamples 1 --gt_file tf_logo.png --gt_transposed --autoscheduler --ignore_glsl --kernel_sigma 0.1

----------------------------------------------------------------------------------------
# FD

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo/best_par_from_real_random.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_color --render_size 640,640 --line_endpoints_method kernel_smooth_debug --deriv_metric_visualization_thre 0.1 --deriv_metric_max_halflen 2e-3 --deriv_metric_line --deriv_metric_endpoint_file /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo/random_smooth_metric_2X1_len_0.002000_2D_kernel_endpoints.npy --deriv_metric_rhs_file /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo/random_smooth_metric_2X1_len_0.002000_2D_kernel_rhs.npy --deriv_metric_suffix _2D_kernel_FD --kernel_sigma 0 --kernel_uv_sigma 1 --deriv_n 10000 --deriv_metric_finite_diff_schedule 1e-1,1e-2,1e-3,1e-4,1e-5 --kernel_smooth_exclude_our_kernel --deriv_metric_no_ours --kernel_nsamples 1 --gt_file tf_logo.png --gt_transposed --autoscheduler --ignore_glsl --kernel_sigma 0.1

----------------------------------------------------------------------------------------
# SPSA
# 4 samples

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo/best_par_from_real_random.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_color --render_size 640,640 --line_endpoints_method kernel_smooth_debug --deriv_metric_visualization_thre 0.1 --deriv_metric_max_halflen 2e-3 --deriv_metric_line --deriv_metric_endpoint_file /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo/random_smooth_metric_2X1_len_0.002000_2D_kernel_endpoints.npy --deriv_metric_rhs_file /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo/random_smooth_metric_2X1_len_0.002000_2D_kernel_rhs.npy --deriv_metric_suffix _2D_kernel_SPSA_4 --kernel_sigma 0 --kernel_uv_sigma 1 --deriv_n 10000 --deriv_metric_finite_diff_schedule 1e-1,1e-2,1e-3,1e-4,1e-5 --kernel_smooth_exclude_our_kernel --deriv_metric_no_ours --kernel_nsamples 1 --gt_file tf_logo.png --gt_transposed --finite_diff_spsa_samples 4 --autoscheduler --ignore_glsl --kernel_sigma 0.1

------------------------------------------------------------------------------------------------------------------------------
# plot

python preprocess_raw_loss_data.py /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo/ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo/finite_diff16_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_000001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo/finite_diff16_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_00001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo/finite_diff16_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_0001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo/finite_diff16_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo/finite_diff16_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_01_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_000001_scaled_niter_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_00001_scaled_niter_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_0001_scaled_niter_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_001_scaled_niter_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_01_scaled_niter_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_000001_scaled_niter_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_00001_scaled_niter_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_0001_scaled_niter_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_001_scaled_niter_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_01_scaled_niter_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo/finite_diff27_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_000001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo/finite_diff27_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_00001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo/finite_diff27_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_0001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo/finite_diff27_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo/finite_diff27_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_01_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo/finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_000001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo/finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_00001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo/finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_0001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo/finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo/finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_01_all_loss.npy

---------------------------------------------
# plot median 

# ours vs FD

python plot_opt_baseline.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --filenames ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_all_loss.npy,finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_01_all_loss.npy,finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_001_all_loss.npy,finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_0001_all_loss.npy,finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_00001_all_loss.npy,finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_000001_all_loss.npy --labels ours,FD1e-1,FD1e-2,FD1e-3,FD1e-4,FD1e-5 --scales 0.0068,0.0263,0.0258,0.0302,0.0258,0.0260 --suffix _FD_median --median

# ours vs SPSA

python plot_opt_baseline.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --filenames ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_01_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_001_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_0001_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_00001_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_000001_scaled_niter_all_loss.npy --labels ours,SPSA1e-1,SPSA1e-2,SPSA1e-3,SPSA1e-4,SPSA1e-5 --scales 0.0068,0.0011,0.0011,0.0012,0.0012,0.0012 --suffix _SPSA_median --median

# ours vs SPSA multiple samples (27, half samples as FD)

python plot_opt_baseline.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --filenames ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_all_loss.npy,finite_diff27_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_01_all_loss.npy,finite_diff27_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_001_all_loss.npy,finite_diff27_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_0001_all_loss.npy,finite_diff27_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_00001_all_loss.npy,finite_diff27_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_000001_all_loss.npy --labels ours,multi_SPSA_1e-1,multi_SPSA_1e-2,multi_SPSA_1e-3,multi_SPSA_1e-4,multi_SPSA_1e-5 --scales 0.0068,0.0146,0.0144,0.0137,0.0138,0.0139 --suffix _ours_multi_SPSA_median --median

# ours vs SPSA vanilla
python plot_opt_baseline.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --filenames ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_01_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_001_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_0001_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_00001_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_000001_scaled_niter_all_loss.npy --labels ours,SPSA_vanilla_1e-1,SPSA_vanilla_1e-2,SPSA_vanilla_1e-3,SPSA_vanilla_1e-4,SPSA_vanilla_1e-5 --scales 0.0068,0.00084,0.00081,0.00086,0.00080,0.00081 --suffix _ours_SPSA_vanilla_median --median

# ours vs SPSA vanilla multiple samples (16, half samples as FD)
python plot_opt_baseline.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --filenames ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_all_loss.npy,finite_diff16_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_01_all_loss.npy,finite_diff16_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_001_all_loss.npy,finite_diff16_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_0001_all_loss.npy,finite_diff16_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_00001_all_loss.npy,finite_diff16_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_000001_all_loss.npy --labels ours,multi_SPSA_vanilla_1e-1,multi_SPSA_vanilla_1e-2,multi_SPSA_vanilla_1e-3,multi_SPSA_vanilla_1e-4,multi_SPSA_vanilla_1e-5 --scales 0.0068,0.0074,0.0070,0.0070,0.0071,0.0069 --suffix _ours_multi_SPSA_vanilla_median --median

# ours vs FD ans SPSA on best performing h
python plot_opt_baseline.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --filenames ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_all_loss.npy,finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_01_all_loss.npy,finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_001_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_01_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_001_scaled_niter_all_loss.npy,finite_diff27_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_01_all_loss.npy,finite_diff27_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_001_all_loss.npy --labels ours,FD_1e-1,FD_1e-2,SPSA_1e-1,SPSA_1e-2,multi_SPSA_1e-1,multi_SPSA_1e-2 --scales 0.0068,0.0263,0.0258,0.0011,0.0011,0.0146,0.0144 --suffix _best_h_1e-2_median --median

# ours vs FD ans SPSA vanilla on best performing h
python plot_opt_baseline.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --filenames ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_01_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_001_scaled_niter_all_loss.npy,finite_diff16_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_01_all_loss.npy,finite_diff16_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_001_all_loss.npy --labels ours,SPSA_vanilla_1e-1,SPSA_vanilla_1e-2,multi_SPSA_vanilla_1e-1,multi_SPSA_vanilla_1e-2 --scales 0.0068,0.00084,0.00081,0.0074,0.0070 --suffix _ours_FD_SPSA_vanilla_best_h_median --median

---------------------------------------------
# plot transparent

# ours vs FD ans SPSA on best performing h
python plot_opt_baseline.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --filenames ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_all_loss.npy,finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_01_all_loss.npy,finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_001_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_01_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_001_scaled_niter_all_loss.npy,finite_diff27_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_01_all_loss.npy,finite_diff27_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_001_all_loss.npy --labels ours,FD_1e-1,FD_1e-2,SPSA_1e-1,SPSA_1e-2,multi_SPSA_1e-1,multi_SPSA_1e-2 --scales 0.0068,0.0263,0.0258,0.0011,0.0011,0.0146,0.0144 --suffix _best_h_1e-2_transparent --transparent

# ours vs FD ans SPSA vanilla on best performing h
python plot_opt_baseline.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --filenames ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_01_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_001_scaled_niter_all_loss.npy,finite_diff16_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_01_all_loss.npy,finite_diff16_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_001_all_loss.npy --labels ours,SPSA_vanilla_1e-1,SPSA_vanilla_1e-2,multi_SPSA_vanilla_1e-1,multi_SPSA_vanilla_1e-2 --scales 0.0068,0.00084,0.00081,0.0074,0.0070 --suffix _ours_FD_SPSA_vanilla_best_h_transparent --transparent

----------------------------------------------------------------------------------------
# plot metric

python metric_compare_line_integral.py --baseline_dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo --deriv_metric_suffix _2D_kernel_FD_finite_diff_0.100000,_2D_kernel_FD_finite_diff_0.010000,_2D_kernel_FD_finite_diff_0.001000,_2D_kernel_FD_finite_diff_0.000100,_2D_kernel_FD_finite_diff_0.000010,_2D_kernel_SPSA_4_finite_diff_0.100000,_2D_kernel_SPSA_4_finite_diff_0.010000,_2D_kernel_SPSA_4_finite_diff_0.001000,_2D_kernel_SPSA_4_finite_diff_0.000100,_2D_kernel_SPSA_4_finite_diff_0.000010,_2D_kernel --eval_labels FD1e-1,FD1e-2,FD1e-3,FD1e-4,FD1e-5,4SPSA1e-1,4SPSA1e-2,4SPSA1e-3,4SPSA1e-4,4SPSA1e-5,ours --max_half_len 2e-3 --rhs_file /n/fs/scratch/yutingy/test_finite_diff_raytracing_structured_tf_logo/random_smooth_metric_2X100000_len_0.002000_2D_kernel_rhs.npy --visualization_thre 0.01 --ncols 5

"""

from render_util import *
from render_single import render_single

nargs = 31

args_range = 0.5 * np.ones(nargs)
sigmas_range = args_range

width = ArgumentScalar('width')
height = ArgumentScalar('height')

compiler.log_prefix_only = False
compiler.log_intermediates_less = True

def cmd_template():
    cmd = f"""python approx_gradient.py --shader test_finite_diff_raytracing_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --metrics 5_scale_L2 --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --gt_file tf_logo.png"""
    
    return cmd

use_select_rule = 1

def test_finite_diff_raytracing_structured_tf_logo(u, v, X, scalar_loss_scale):
    
    x_scale = X[0] / width
    y_scale = X[1] / height

    origin_x = X[2]
    origin_y = X[3]
    origin_z = X[4]
    
    ang1 = X[5]
    ang2 = X[6]
    ang3 = X[7]
    
    amb = np.array([X[8], X[9], X[10]])
    kd = np.array([X[11], X[12], X[13]])
    
    lig_ang0 = X[14]
    lig_ang1 = X[15]
        
    boxes = []
    for idx in range(4):
        
        if idx == 0:
            # horizontal stroke of T, both pos and dimension are free
            current_pos = [X[16], X[17], X[18]]
            current_dim = [X[19], X[20], X[21]]
        elif idx == 1:
            # vertical stroke of T
            # dim: x fixed, y, z can extend
            # pos: x, y constrained, z can slide
            current_dim = [boxes[0].dim[0], X[22], X[23]]
            current_pos = [boxes[0].pos[0],
                           boxes[0].pos[1] - boxes[0].dim[1] - current_dim[1],
                           X[24]]
        elif idx == 2:
            # top horizontal stroke of F
            # dim: free
            # pos: constrained
            current_dim = [X[25], X[26], X[27]]
            current_pos = [boxes[0].pos[0] - boxes[0].dim[0] + current_dim[0],
                           boxes[0].pos[1],
                           boxes[0].pos[2] - boxes[0].dim[2] - current_dim[2]]
        else:
            # bottom horizontal stroke of F
            # dim: z fixed, x, y can extend
            # pos: x, z constrained, y can slide
            current_dim = [X[28], X[29], boxes[1].dim[2]]
            current_pos = [boxes[1].pos[0] + boxes[1].dim[0] + current_dim[0],
                           X[30],
                           boxes[1].pos[2]]

        boxes.append(Object('box',
                            pos = current_pos,
                            dim = current_dim))
    
    ro = vec('ro', [origin_x, origin_y, origin_z])
    ang1, ang2, ang3 = vec('ang', [ang1, ang2, ang3])
    
    animate_pos = Animate('animate_camera_pos', inout_ls=[ro, ang1, ang2, ang3])
    ro, ang1, ang2, ang3 = animate_pos.update()

    sin1 = Var('sin1', sin(ang1))
    cos1 = Var('cos1', cos(ang1))
    sin2 = Var('sin2', sin(ang2))
    cos2 = Var('cos2', cos(ang2))
    sin3 = Var('sin3', sin(ang3))
    cos3 = Var('cos3', cos(ang3))
    
    # use parallel projection
    rd = np.array([sin1 * sin3 + cos1 * sin2 * cos3,
                   -sin1 * cos3 + cos1 * sin2 * sin3,
                   cos1 * cos2])
    
    offset_raw = [x_scale * (u - width / 2),
                  y_scale * (v - height / 2)]
    
    offset = [cos2 * cos3 * offset_raw[0] + (-cos1 * sin3 + sin1 * sin2 * cos3) * offset_raw[1],
              cos2 * sin3 * offset_raw[0] + (cos1 * cos3 + sin1 * sin2 * sin3) * offset_raw[1],
              -sin2 * offset_raw[0] + sin1 * cos2 * offset_raw[1]]
    
    ro = ro + np.array(offset)
    
    is_valid = None
    final_t = None
    final_deriv = None
        
    for box in boxes:
        
        offset = box.pos - ro
        t0 = vec('t0_%s' % box.name, (offset + box.dim) / rd)
        t1 = vec('t1_%s' % box.name, (offset - box.dim) / rd)
        
        tx_min = minimum(t0[0], t1[0])
        tx_max = maximum(t0[0], t1[0])

        ty_min = minimum(t0[1], t1[1])
        ty_max = maximum(t0[1], t1[1])

        tz_min = minimum(t0[2], t1[2])
        tz_max = maximum(t0[2], t1[2])

        max_txy = maximum(tx_min, ty_min)

        tmin = Var('tmin_%s' % box.name, maximum(max_txy, tz_min))
        tmax = Var('tmax_%s' % box.name, minimum(minimum(tx_max, ty_max), tz_max))
        
        current_t = tmin
        current_deriv = [select(max_txy >= tz_min, select(tx_min >= ty_min, sign(offset[0]), 0.), 0.),
                         select(max_txy >= tz_min, select(tx_min >= ty_min, 0., sign(offset[1])), 0.),
                         select(max_txy >= tz_min, 0., sign(offset[2]))]
        
        current_is_valid = Var('is_valid_%s' % box.name, (tmax > 0) & (tmin <= tmax))
        
        if final_t is None:
            is_valid = current_is_valid
            final_t = current_t
            final_deriv = current_deriv
        else:
            
            do_update = current_is_valid & ((~is_valid) | (current_t <= final_t))
            
            final_deriv = [select(do_update, current_deriv[i], final_deriv[i]) for i in range(3)]
            final_t = select(do_update, current_t, final_t)
            is_valid = is_valid | current_is_valid
            
    lig = np.array([cos(lig_ang0),
                    sin(lig_ang0) * cos(lig_ang1),
                    sin(lig_ang0) * sin(lig_ang1)])
    
    dif = dot(lig, final_deriv)
    
    col = select(is_valid, amb + dif * kd, 1.)
    
    return col

shaders = [test_finite_diff_raytracing_structured_tf_logo]
is_color = True