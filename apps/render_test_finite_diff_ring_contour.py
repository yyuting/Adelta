"""
------------------------------------------------------------------------------------------------------------------------------
# command for visualization

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_init_values_pool.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_col --autoscheduler --ignore_glsl --gt_file celtic_knot.png --gt_transposed

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour_AD --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_init_values_pool.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization AD --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_col --autoscheduler --ignore_glsl --gt_file celtic_knot.png --gt_transposed

------------------------------------------------------------------------------------------------------------------------------
# optimization with random var

----------------------------------------------------------------------------------------
# ours

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --gt_file celtic_knot.png --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --suffix _from_real_random --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --no_reset_opt --no_binary_search_std --save_all_loss --save_best_par --backend hl --quiet --gt_transposed --no_reset_sigma --autoscheduler 

# ours no random variable

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --gt_file celtic_knot.png --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --suffix _from_real_no_random --no_reset_opt --save_all_loss --autoscheduler --backend hl --quiet --gt_transposed --no_reset_sigma

# test convergence with random noise

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_ring_contour/best_par_from_real_random.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --suffix _test_random_noise_convergence --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --autoscheduler --nrestart 100 --no_reset_sigma

----------------------------------------------------------------------------------------
# AD

---------------------------------------------
# same opt as ours and FD

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour_AD --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization AD --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --suffix _from_real_random --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --save_best_par --autoscheduler --no_reset_sigma

---------------------------------------------
# vanilla opt
# no random noise, no alternation, no multi stage

-------------------------
# 1 sample (only runned with one choice of h to get a runtime estimate)

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour_AD --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization AD --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --suffix _from_real_vanilla --backend hl --quiet --no_reset_opt --save_all_loss --autoscheduler --no_reset_opt

----------------------------------------------------------------------------------------
# Zeroth-order

---------------------------------------------
# vanilla

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --suffix _from_real_vanilla --backend hl --quiet --no_reset_opt --save_all_loss --autoscheduler --optimizer scipy.Nelder-Mead --niters 10000

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --suffix _from_real_vanilla --backend hl --quiet --no_reset_opt --save_all_loss --autoscheduler --optimizer scipy.Powell --niters 10000

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --suffix _from_real_vanilla --backend hl --quiet --no_reset_opt --save_all_loss --autoscheduler --optimizer mcmc --niters 10000

----------------------------------------------------------------------------------------
# FD

---------------------------------------------
# same opt as ours and FD

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_fd_01 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_fd_001 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.001 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_fd_0001 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.0001 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_fd_00001 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.00001 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_fd_000001 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --autoscheduler --no_reset_sigma

---------------------------------------------
# test convergence with random noise

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_ring_contour/best_par_from_real_random.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --suffix _test_random_noise_convergence_fd_001 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --autoscheduler --nrestart 100 --no_reset_sigma

---------------------------------------------
# No random opt

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --suffix _from_real_no_random_fd_01 --backend hl --quiet --no_reset_opt --save_all_loss --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --suffix _from_real_no_random_fd_001 --backend hl --quiet --no_reset_opt --save_all_loss --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.001 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --suffix _from_real_no_random_fd_0001 --backend hl --quiet --no_reset_opt --save_all_loss --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.0001 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --suffix _from_real_no_random_fd_00001 --backend hl --quiet --no_reset_opt --save_all_loss --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.00001 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --suffix _from_real_no_random_fd_000001 --backend hl --quiet --no_reset_opt --save_all_loss --autoscheduler --no_reset_sigma

---------------------------------------------
# vanilla opt
# no random noise, no alternation, no multi stage

-------------------------
# 1 sample (only runned with one choice of h to get a runtime estimate)

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --suffix _from_real_vanilla_fd_h_01 --backend hl --quiet --no_reset_opt --save_all_loss --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --suffix _from_real_vanilla_fd_h_001 --backend hl --quiet --no_reset_opt --save_all_loss --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.001 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --suffix _from_real_vanilla_fd_h_0001 --backend hl --quiet --no_reset_opt --save_all_loss --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.0001 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --suffix _from_real_vanilla_fd_h_00001 --backend hl --quiet --no_reset_opt --save_all_loss --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.00001 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --suffix _from_real_vanilla_fd_h_000001 --backend hl --quiet --no_reset_opt --save_all_loss --autoscheduler --no_reset_sigma

----------------------------------------------------------------------------------------
# SPSA

---------------------------------------------
# same opt as ours
# Runtime
# Ours: 0.0039
# SPSA: 0.0019
# SPSA iter match ours runtime: 4105

-------------------------
# 1 sample (only runned with one choice of h to get a runtime estimate)

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_fd_01 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --finite_diff_spsa_samples 1 --autoscheduler --no_reset_sigma

-------------------------
# 1 sample iteration scaled by runtime

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_fd_01_scaled_niter --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --finite_diff_spsa_samples 1 --autoscheduler --niters 4105 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_fd_001_scaled_niter --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --finite_diff_spsa_samples 1 --autoscheduler --niters 4105 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.001 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_fd_0001_scaled_niter --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --finite_diff_spsa_samples 1 --autoscheduler --niters 4105 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.0001 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_fd_00001_scaled_niter --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --finite_diff_spsa_samples 1 --autoscheduler --niters 4105 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.00001 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_fd_000001_scaled_niter --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --finite_diff_spsa_samples 1 --autoscheduler --niters 4105 --no_reset_sigma

-------------------------
# half sample as FD: 42 samples

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_fd_01 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --finite_diff_spsa_samples 42 --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_fd_001 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --finite_diff_spsa_samples 42 --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.001 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_fd_0001 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --finite_diff_spsa_samples 42 --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.0001 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_fd_00001 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --finite_diff_spsa_samples 42 --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.00001 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --suffix _from_real_random_fd_000001 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --finite_diff_spsa_samples 42 --autoscheduler --no_reset_sigma

---------------------------------------------
# vanilla opt
# no random noise, no alternation, no multi stage
# Runtime
# Ours: 0.0039
# SPSA: 0.0010
# SPSA iter match ours runtime: 7800

-------------------------
# 1 sample (only runned with one choice of h to get a runtime estimate)

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --suffix _from_real_vanilla_fd_h_01 --backend hl --quiet --no_reset_opt --save_all_loss --finite_diff_spsa_samples 1 --autoscheduler --no_reset_sigma

-------------------------
# 1 sample iteration scaled by runtime

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --suffix _from_real_vanilla_fd_h_01_scaled_niter --backend hl --quiet --no_reset_opt --save_all_loss --finite_diff_spsa_samples 1 --autoscheduler --niters 7800 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --suffix _from_real_vanilla_fd_h_001_scaled_niter --backend hl --quiet --no_reset_opt --save_all_loss --finite_diff_spsa_samples 1 --autoscheduler --niters 7800 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.001 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --suffix _from_real_vanilla_fd_h_0001_scaled_niter --backend hl --quiet --no_reset_opt --save_all_loss --finite_diff_spsa_samples 1 --autoscheduler --niters 7800 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.0001 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --suffix _from_real_vanilla_fd_h_00001_scaled_niter --backend hl --quiet --no_reset_opt --save_all_loss --finite_diff_spsa_samples 1 --autoscheduler --niters 7800 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.00001 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --suffix _from_real_vanilla_fd_h_000001_scaled_niter --backend hl --quiet --no_reset_opt --save_all_loss --finite_diff_spsa_samples 1 --autoscheduler --niters 7800 --no_reset_sigma

-------------------------
# half sample as FD: 21 samples

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --suffix _from_real_vanilla_fd_h_01 --backend hl --quiet --no_reset_opt --save_all_loss --finite_diff_spsa_samples 21 --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --suffix _from_real_vanilla_fd_h_001 --backend hl --quiet --no_reset_opt --save_all_loss --finite_diff_spsa_samples 21 --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.001 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --suffix _from_real_vanilla_fd_h_0001 --backend hl --quiet --no_reset_opt --save_all_loss --finite_diff_spsa_samples 21 --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.0001 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --suffix _from_real_vanilla_fd_h_00001 --backend hl --quiet --no_reset_opt --save_all_loss --finite_diff_spsa_samples 21 --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.00001 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --suffix _from_real_vanilla_fd_h_000001 --backend hl --quiet --no_reset_opt --save_all_loss --finite_diff_spsa_samples 21 --autoscheduler --no_reset_sigma

---------------------------------------------
# no random but same opt process as ours and FD
# Runtime
# Ours: 0.0039
# SPSA: 0.0010
# SPSA iter match ours runtime: 7800

-------------------------
# 1 sample (only runned with one choice of h to get a runtime estimate)

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --suffix _from_real_no_random_fd_01 --backend hl --quiet --no_reset_opt --no_binary_search_std --save_all_loss --finite_diff_spsa_samples 1 --autoscheduler --no_reset_sigma

-------------------------
# 1 sample iteration scaled by runtime

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --suffix _from_real_no_random_fd_01_scaled_niter --backend hl --quiet --no_reset_opt --no_binary_search_std --save_all_loss --finite_diff_spsa_samples 1 --autoscheduler --no_reset_sigma --niter 7800

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --suffix _from_real_no_random_fd_001_scaled_niter --backend hl --quiet --no_reset_opt --no_binary_search_std --save_all_loss --finite_diff_spsa_samples 1 --autoscheduler --no_reset_sigma --niter 7800

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.001 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --suffix _from_real_no_random_fd_0001_scaled_niter --backend hl --quiet --no_reset_opt --no_binary_search_std --save_all_loss --finite_diff_spsa_samples 1 --autoscheduler --no_reset_sigma --niter 7800

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.0001 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --suffix _from_real_no_random_fd_00001_scaled_niter --backend hl --quiet --no_reset_opt --no_binary_search_std --save_all_loss --finite_diff_spsa_samples 1 --autoscheduler --no_reset_sigma --niter 7800

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.00001 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --suffix _from_real_no_random_fd_000001_scaled_niter --backend hl --quiet --no_reset_opt --no_binary_search_std --save_all_loss --finite_diff_spsa_samples 1 --autoscheduler --no_reset_sigma --niter 7800

-------------------------
# half sample as FD: 21 samples

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --suffix _from_real_no_random_fd_01 --backend hl --quiet --no_reset_opt --no_binary_search_std --save_all_loss --finite_diff_spsa_samples 21 --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --suffix _from_real_no_random_fd_001 --backend hl --quiet --no_reset_opt --no_binary_search_std --save_all_loss --finite_diff_spsa_samples 21 --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.001 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --suffix _from_real_no_random_fd_0001 --backend hl --quiet --no_reset_opt --no_binary_search_std --save_all_loss --finite_diff_spsa_samples 21 --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.0001 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --suffix _from_real_no_random_fd_00001 --backend hl --quiet --no_reset_opt --no_binary_search_std --save_all_loss --finite_diff_spsa_samples 21 --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.00001 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --suffix _from_real_no_random_fd_000001 --backend hl --quiet --no_reset_opt --no_binary_search_std --save_all_loss --finite_diff_spsa_samples 21 --autoscheduler --no_reset_sigma

------------------------------------------------------------------------------------------------------------------------------
# command for generating glsl in a seperate process

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_ring_contour/best_par_from_real_random.npy --modes render --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --backend hl --quiet

------------------------------------------------------------------------------------------------------------------------------
# render optimization process

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --suffix _from_real_random --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_par --autoscheduler

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_ring_contour/ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_result0_0.npy --modes render --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --gt_file celtic_knot.png --gt_transposed --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --suffix _from_real_random --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --suffix _opt

cd /n/fs/scratch/yutingy/test_finite_diff_ring_contour; ffmpeg -i init_opt%d.png -r 30 -c:v libx264 -preset slow -crf 0 -r 30 animation.mp4; cd /n/fs/shaderml/differentiable_compiler/

# quicktime compatible
ffmpeg -i init_opt%d.png -r 30 -c:v libx264 -pix_fmt yuv420p -r 30 animation.mp4

------------------------------------------------------------------------------------------------------------------------------
# Quantitative Metric using 2D box kernels
# Optimization runtime
# Ours: 0.0039
# SPSA 1 sample: 0.0019
# FD 1 sample: 0.

# SPSA need 2 samples for similar runtime with ours in opt

# get endpoints
python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_ring_contour/best_par_from_real_random.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_col --backend hl --line_endpoints_method random_smooth --deriv_metric_visualization_thre 0.1 --deriv_metric_max_halflen 2e-3 --deriv_metric_line --deriv_metric_finite_diff_schedule 0 --kernel_nsamples 1 --deriv_n 2 --deriv_metric_suffix _2D_kernel --our_filter_direction 2d --render_size 640,640 --autoscheduler --ignore_glsl

# rhs

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_ring_contour/best_par_from_real_random.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_col --backend hl --line_endpoints_method random_smooth --deriv_metric_visualization_thre 0.1 --deriv_metric_max_halflen 2e-3 --deriv_metric_line --deriv_metric_finite_diff_schedule 0 --kernel_nsamples 100000 --deriv_n 2 --deriv_metric_suffix _2D_kernel --our_filter_direction 2d --render_size 640,640 --autoscheduler --ignore_glsl --deriv_metric_endpoint_file /n/fs/scratch/yutingy/test_finite_diff_ring_contour/random_smooth_metric_2X1_len_0.002000_2D_kernel_endpoints.npy --kernel_sigma 0.1

----------------------------------------------------------------------------------------
# ours

# 1 sample

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_ring_contour/best_par_from_real_random.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_col --backend hl --line_endpoints_method kernel_smooth_debug --deriv_metric_visualization_thre 0.1 --deriv_metric_max_halflen 2e-3 --deriv_metric_line --deriv_metric_finite_diff_schedule 0 --kernel_nsamples 1 --deriv_n 10000 --deriv_metric_suffix _2D_kernel --kernel_sigma 0 --kernel_uv_sigma 1 --kernel_smooth_exclude_our_kernel --render_size 640,640 --deriv_metric_endpoint_file /n/fs/scratch/yutingy/test_finite_diff_ring_contour/random_smooth_metric_2X1_len_0.002000_2D_kernel_endpoints.npy --deriv_metric_rhs_file /n/fs/scratch/yutingy/test_finite_diff_ring_contour/random_smooth_metric_2X100000_len_0.002000_2D_kernel_rhs.npy --ignore_glsl --autoscheduler --kernel_sigma 0.1

----------------------------------------------------------------------------------------
# FD

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_ring_contour/best_par_from_real_random.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_col --backend hl --line_endpoints_method kernel_smooth_debug --deriv_metric_visualization_thre 0.1 --deriv_metric_max_halflen 2e-3 --deriv_metric_line --deriv_metric_finite_diff_schedule 1e-1,1e-2,1e-3,1e-4,1e-5 --kernel_nsamples 1 --deriv_n 10000 --deriv_metric_suffix _2D_kernel_FD --kernel_sigma 0 --kernel_uv_sigma 1 --kernel_smooth_exclude_our_kernel --render_size 640,640 --deriv_metric_no_ours --deriv_metric_endpoint_file /n/fs/scratch/yutingy/test_finite_diff_ring_contour/random_smooth_metric_2X1_len_0.002000_2D_kernel_endpoints.npy --deriv_metric_rhs_file /n/fs/scratch/yutingy/test_finite_diff_ring_contour/random_smooth_metric_2X100000_len_0.002000_2D_kernel_rhs.npy --ignore_glsl --autoscheduler --kernel_sigma 0.1

----------------------------------------------------------------------------------------
# SPSA
# 2 samples
# number of samples corresponds to runtime in our optimization pipeline (with random, multi stage loss)

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --shader test_finite_diff_ring_contour --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_ring_contour/best_par_from_real_random.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_col --backend hl --line_endpoints_method kernel_smooth_debug --deriv_metric_visualization_thre 0.1 --deriv_metric_max_halflen 2e-3 --deriv_metric_line --deriv_metric_finite_diff_schedule 1e-1,1e-2,1e-3,1e-4,1e-5 --kernel_nsamples 1 --deriv_n 10000 --deriv_metric_suffix _2D_kernel_SPSA_2 --kernel_sigma 0 --kernel_uv_sigma 1 --kernel_smooth_exclude_our_kernel --render_size 640,640 --deriv_metric_no_ours --deriv_metric_endpoint_file /n/fs/scratch/yutingy/test_finite_diff_ring_contour/random_smooth_metric_2X1_len_0.002000_2D_kernel_endpoints.npy --deriv_metric_rhs_file /n/fs/scratch/yutingy/test_finite_diff_ring_contour/random_smooth_metric_2X100000_len_0.002000_2D_kernel_rhs.npy --finite_diff_spsa_samples 2 --ignore_glsl --autoscheduler --kernel_sigma 0.1

----------------------------------------------------------------------------------------
# plot

---------------------------------------------

# preprocess

python preprocess_raw_loss_data.py /n/fs/scratch/yutingy/test_finite_diff_ring_contour /n/fs/scratch/yutingy/test_finite_diff_ring_contour/ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_ring_contour/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_000001_scaled_niter_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_ring_contour/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_00001_scaled_niter_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_ring_contour/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_0001_scaled_niter_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_ring_contour/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_001_scaled_niter_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_ring_contour/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_01_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_ring_contour/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_01_scaled_niter_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_ring_contour/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_000001_scaled_niter_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_ring_contour/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_00001_scaled_niter_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_ring_contour/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_0001_scaled_niter_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_ring_contour/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_001_scaled_niter_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_ring_contour/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_01_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_ring_contour/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_01_scaled_niter_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_ring_contour/finite_diff21_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_000001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_ring_contour/finite_diff21_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_00001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_ring_contour/finite_diff21_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_0001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_ring_contour/finite_diff21_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_ring_contour/finite_diff21_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_01_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_ring_contour/finite_diff42_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_000001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_ring_contour/finite_diff42_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_00001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_ring_contour/finite_diff42_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_0001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_ring_contour/finite_diff42_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_ring_contour/finite_diff42_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_01_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_ring_contour/finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_000001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_ring_contour/finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_00001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_ring_contour/finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_0001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_ring_contour/finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_ring_contour/finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_01_all_loss.npy

---------------------------------------------
# plot median 

# ours vs FD

python plot_opt_baseline.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --filenames  ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_all_loss.npy,finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_01_all_loss.npy,finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_001_all_loss.npy,finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_0001_all_loss.npy,finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_00001_all_loss.npy,finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_000001_all_loss.npy --labels ours,FD_1e-1,FD_1e-2,FD_1e-3,FD_1e-4,FD_1e-5 --scales 0.0039,0.0456,0.0453,0.0468,0.0524,0.0437 --suffix _ours_FD_median --median

# ours vs SPSA
python plot_opt_baseline.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --filenames ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_01_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_001_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_0001_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_00001_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_000001_scaled_niter_all_loss.npy --labels ours,SPSA_1e-1,SPSA_1e-2,SPSA_1e-3,SPSA_1e-4,SPSA_1e-5 --scales 0.0039,0.0015,0.0015,0.0017,0.0018,0.0022 --suffix _ours_SPSA_median --median

# ours vs SPSA multiple samples (42, half samples as FD)
python plot_opt_baseline.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --filenames ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_all_loss.npy,finite_diff42_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_01_all_loss.npy,finite_diff42_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_001_all_loss.npy,finite_diff42_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_0001_all_loss.npy,finite_diff42_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_00001_all_loss.npy,finite_diff42_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_000001_all_loss.npy --labels ours,multi_SPSA_1e-1,multi_SPSA_1e-2,multi_SPSA_1e-3,multi_SPSA_1e-4,multi_SPSA_1e-5 --scales 0.0039,0.0242,0.0244,0.0250,0.0239,0.0243 --suffix _ours_multi_SPSA_median --median

# ours vs SPSA vanilla
python plot_opt_baseline.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --filenames ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_01_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_001_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_0001_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_00001_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_000001_scaled_niter_all_loss.npy --labels ours,SPSA_vanilla_1e-1,SPSA_vanilla_1e-2,SPSA_vanilla_1e-3,SPSA_vanilla_1e-4,SPSA_vanilla_1e-5 --scales 0.0039,0.00089,0.00088,0.00090,0.00092,0.00093 --suffix _ours_SPSA_vanilla_median --median

# ours vs SPSA vanilla multiple samples (40, half samples as FD)
python plot_opt_baseline.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --filenames ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_all_loss.npy,finite_diff21_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_01_all_loss.npy,finite_diff21_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_001_all_loss.npy,finite_diff21_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_0001_all_loss.npy,finite_diff21_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_00001_all_loss.npy,finite_diff21_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_000001_all_loss.npy --labels ours,multi_SPSA_vanilla_1e-1,multi_SPSA_vanilla_1e-2,multi_SPSA_vanilla_1e-3,multi_SPSA_vanilla_1e-4,multi_SPSA_vanilla_1e-5 --scales 0.0039,0.0096,0.0095,0.0095,0.0096,0.0098 --suffix _ours_multi_SPSA_vanilla_median --median

# ours vs FD ans SPSA on best performing h
python plot_opt_baseline.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --filenames ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_all_loss.npy,finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_001_all_loss.npy,finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_0001_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_001_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_0001_scaled_niter_all_loss.npy,finite_diff42_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_001_all_loss.npy,finite_diff42_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_0001_all_loss.npy --labels ours,FD_1e-2,FD_1e-3,SPSA_1e-2,SPSA_1e-3,multi_SPSA_1e-2,multi_SPSA_1e-3 --scales 0.0039,0.0453,0.0468,0.0015,0.0017,0.0244,0.0250 --suffix _ours_FD_SPSA_best_h_median --median

# ours vs FD ans SPSA vanilla on best performing h
python plot_opt_baseline.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --filenames ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_001_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_0001_scaled_niter_all_loss.npy,finite_diff21_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_001_all_loss.npy,finite_diff21_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_0001_all_loss.npy --labels ours,SPSA_vanilla_1e-2,SPSA_vanilla_1e-3,multi_SPSA_vanilla_1e-2,multi_SPSA_vanilla_1e-4 --scales 0.0039,0.00088,0.00090,0.0095,0.0095 --suffix _ours_FD_SPSA_vanilla_best_h_median --median

---------------------------------------------
# plot transparent 

# ours vs FD

# ours vs FD ans SPSA on best performing h
python plot_opt_baseline.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --filenames ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_all_loss.npy,finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_001_all_loss.npy,finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_0001_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_001_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_0001_scaled_niter_all_loss.npy,finite_diff42_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_001_all_loss.npy,finite_diff42_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_fd_0001_all_loss.npy --labels ours,FD_1e-2,FD_1e-3,SPSA_1e-2,SPSA_1e-3,multi_SPSA_1e-2,multi_SPSA_1e-3 --scales 0.0039,0.0453,0.0468,0.0015,0.0017,0.0244,0.0250 --suffix _ours_FD_SPSA_best_h_transparent --transparent

# ours vs FD ans SPSA vanilla on best performing h
python plot_opt_baseline.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --filenames ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_random_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_001_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_0001_scaled_niter_all_loss.npy,finite_diff21_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_001_all_loss.npy,finite_diff21_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_0001_all_loss.npy --labels ours,SPSA_vanilla_1e-2,SPSA_vanilla_1e-3,multi_SPSA_vanilla_1e-2,multi_SPSA_vanilla_1e-4 --scales 0.0039,0.00088,0.00090,0.0095,0.0095 --suffix _ours_FD_SPSA_vanilla_best_h_transparent --transparent

---------------------------------------------
# plot metric

python metric_compare_line_integral.py --baseline_dir /n/fs/scratch/yutingy/test_finite_diff_ring_contour --deriv_metric_suffix _2D_kernel_FD_finite_diff_0.100000,_2D_kernel_FD_finite_diff_0.010000,_2D_kernel_FD_finite_diff_0.001000,_2D_kernel_FD_finite_diff_0.000100,_2D_kernel_FD_finite_diff_0.000010,_2D_kernel_SPSA_2_finite_diff_0.100000,_2D_kernel_SPSA_2_finite_diff_0.010000,_2D_kernel_SPSA_2_finite_diff_0.001000,_2D_kernel_SPSA_2_finite_diff_0.000100,_2D_kernel_SPSA_2_finite_diff_0.000010,_2D_kernel --eval_labels FD1e-1,FD1e-2,FD1e-3,FD1e-4,FD1e-5,2SPSA1e-1,2SPSA1e-2,2SPSA1e-3,2SPSA1e-4,2SPSA1e-5,ours --max_half_len 2e-3 --rhs_file /n/fs/scratch/yutingy/test_finite_diff_ring_contour/random_smooth_metric_2X100000_len_0.002000_2D_kernel_rhs.npy --visualization_thre 0.01 --ncols 5

"""

from render_util import *
from render_single import render_single

compiler.log_prefix_only = False
compiler.log_intermediates_less = True

def cmd_template():
    cmd = f"""python approx_gradient.py --shader test_finite_diff_ring_contour --init_values_pool apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy --metrics 5_scale_L2 --gt_file celtic_knot.png --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5"""
    
    return cmd

nrings = 10

nargs = 4 * nrings + 2

args_range = np.ones(nargs)

args_range[0] = 1
args_range[1] = 0.5

# pos_x, pos_y
args_range[2:2+2*nrings] = 200
# radius
args_range[2+2*nrings:2+3*nrings] = 100
# tilt
args_range[2+3*nrings:2+4*nrings] = 10

sigmas_range = args_range

width = ArgumentScalar('width')
height = ArgumentScalar('height')

max_iter = nrings - 1

default_phase = -1e4

def test_finite_diff_ring_contour(u, v, X, scalar_loss_scale):
    
    # make sure it's non-negative
    curve_width = X[0] ** 2
    curve_edge = X[1] ** 2
    
    fill_col = Var('fill_col', Compound([1., 1., 1.]))
    edge_col = Compound([0., 0., 0.])
    
    animate_coord = Animate('animate_coord', inout_ls=[Var('u', u), Var('v', v)])
    u, v = animate_coord.update()
    
    rings = []
    for i in range(nrings):
        
        ring_params = [X[i + k * nrings + 2] for k in range(4)]
        ring = Object('ring', 
                      pos = ring_params[:2],
                      radius = ring_params[2],
                      tilt = ring_params[3])
        rings.append(ring)
        
    def update_ring(old_vals, ring, idx):
        # Update function should be side-effect free
        
        old_col, old_phase = old_vals[0], old_vals[1]
        
        rel_pos = vec('rel_pos_%d' % idx, np.array([u, v]) - ring.pos)
        
        animate_fill_col = Animate('animate_ring_col_%d' % idx, inout_ls=[fill_col], in_ls=[rel_pos])
        updated_fill_col, = animate_fill_col.update()
        ring.fill_col = updated_fill_col
        
        dist2 = Var('dist2_%s' % ring.name, rel_pos[0] ** 2 + rel_pos[1] ** 2)
        dist = Var('dist_%s' % ring.name, dist2 ** 0.5)
        
        phase = Var('phase_raw_%s' % ring.name, rel_pos[0] * ring.tilt)
        
        dist2circle = abs(dist - ring.radius)
        
        cond0_diff = Var('cond0_diff_%s' % ring.name, dist2circle - curve_width / 2)
        cond1_diff = Var('cond1_diff_%s' % ring.name, cond0_diff + curve_edge)
        phase_diff = Var('phase_diff_%s' % ring.name, phase - old_phase)
        
        cond0 = Var('cond0_%s' % ring.name, cond0_diff < 0)
        cond1 = Var('cond1_%s' % ring.name, cond1_diff > 0)
        cond2 = Var('cond2_%s' % ring.name, phase_diff > 0)
        
        cond_valid = Var('cond_valid_%s' % ring.name, cond0 & cond2)
        
        col_current = Var('col_current_%s' % ring.name, select(cond1, edge_col, ring.fill_col))
        
        col = Var('col_%s' % ring.name, select(cond_valid, col_current, old_col))
        
        out_phase = Var('phase_%s' % ring.name, select(cond_valid, phase, old_phase))
        
        return [col, out_phase]
    
    global default_phase
    # BG
    col = Compound([1, 1, 1])
    
    vals = [col, default_phase]
    
    for i in range(nrings):
        possible_update(vals, update_ring, rings[i], i)
        
    return vals[0]

shaders = [test_finite_diff_ring_contour]
is_color = True

def sample_init(nsamples, target_width=640, target_height=640):
    
    init = np.zeros([nsamples, nargs])
    
    init[:, 0] = np.random.rand(nsamples) * 5 + 5
    init[:, 1] = np.random.rand(nsamples) + 1
    
    init[:, 2 : 2 + nrings] = np.random.rand(nsamples, nrings) * 0.6 * target_width + 0.2 * target_width
    init[:, 2 + nrings : 2 + 2 * nrings] = np.random.rand(nsamples, nrings) * 0.6 * target_height + 0.2 * target_height
    init[:, 2 + 2 * nrings : 2 + 3 * nrings] = np.random.rand(nsamples, nrings) * 100 + 100
    init[:, 2 + 3 * nrings:] = np.random.rand(nsamples, nrings) * 2 - 1
    
    return init

if __name__ == '__main__':
    init = sample_init(100)
    np.save('../apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy', init)