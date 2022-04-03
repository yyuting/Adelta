"""
------------------------------------------------------------------------------------------------------------------------------
# command for visualization
python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated --shader test_finite_diff_raymarching_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_init_values_pool.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 640,640 --is_color --autoscheduler --gt_file tf_logo.png --gt_transposed 

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated_AD --shader test_finite_diff_raymarching_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_init_values_pool.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization AD --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 640,640 --is_color --autoscheduler --gt_file tf_logo.png --gt_transposed 

------------------------------------------------------------------------------------------------------------------------------
# optimization

----------------------------------------------------------------------------------------
# ours

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated --shader test_finite_diff_raymarching_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file tf_logo.png --gt_transposed --suffix _from_real --save_all_loss --no_reset_opt --backend hl --quiet --multi_scale_optimization --autoscheduler --alternating_times 5 --save_best_par --no_reset_sigma

----------------------------------------------------------------------------------------
# AD

---------------------------------------------
# same as Ours and FD

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated_AD --shader test_finite_diff_raymarching_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization AD --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file tf_logo.png --gt_transposed --suffix _from_real --save_all_loss --no_reset_opt --backend hl --quiet --multi_scale_optimization --autoscheduler --alternating_times 5 --save_best_par --no_reset_sigma

---------------------------------------------
# vanilla opt

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated_AD --shader test_finite_diff_raymarching_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization AD --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file tf_logo.png --gt_transposed --suffix _from_real_vanilla --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --save_best_par --no_reset_sigma

----------------------------------------------------------------------------------------
# Zeroth-Order

---------------------------------------------
# varnilla

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated --shader test_finite_diff_raymarching_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file tf_logo.png --gt_transposed --suffix _from_real_vanilla --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --save_best_par --optimizer scipy.Nelder-Mead --niters 10000

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated --shader test_finite_diff_raymarching_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file tf_logo.png --gt_transposed --suffix _from_real_vanilla --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --save_best_par --optimizer scipy.Powell --niters 10000

----------------------------------------------------------------------------------------
# FD

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated --shader test_finite_diff_raymarching_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file tf_logo.png --gt_transposed --suffix _from_real_fd_h_01 --save_all_loss --no_reset_opt --backend hl --quiet --multi_scale_optimization --autoscheduler --alternating_times 5 --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated --shader test_finite_diff_raymarching_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file tf_logo.png --gt_transposed --suffix _from_real_fd_h_001 --save_all_loss --no_reset_opt --backend hl --quiet --multi_scale_optimization --autoscheduler --alternating_times 5 --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated --shader test_finite_diff_raymarching_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.001 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file tf_logo.png --gt_transposed --suffix _from_real_fd_h_0001 --save_all_loss --no_reset_opt --backend hl --quiet --multi_scale_optimization --autoscheduler --alternating_times 5 --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated --shader test_finite_diff_raymarching_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.0001 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file tf_logo.png --gt_transposed --suffix _from_real_fd_h_00001 --save_all_loss --no_reset_opt --backend hl --quiet --multi_scale_optimization --autoscheduler --alternating_times 5 --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated --shader test_finite_diff_raymarching_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.00001 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file tf_logo.png --gt_transposed --suffix _from_real_fd_h_000001 --save_all_loss --no_reset_opt --backend hl --quiet --multi_scale_optimization --autoscheduler --alternating_times 5 --autoscheduler --no_reset_sigma

----------------------------------------------------------------------------------------
# SPSA

---------------------------------------------
# same opt as ours and FD
# Runtime
# Ours: 0.0015
# SPSA: 0.0018
# SPSA iter match ours runtime: 2000 (should be at least as much as ours)

-------------------------
# 1 sample (only runned with one choice of h to get a runtime estimate)

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated --shader test_finite_diff_raymarching_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.001 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file tf_logo.png --gt_transposed --suffix _from_real_fd_h_0001 --save_all_loss --no_reset_opt --backend hl --quiet --multi_scale_optimization --autoscheduler --finite_diff_spsa_samples 1 --alternating_times 5 --autoscheduler --no_reset_sigma

-------------------------
# 1 sample iteration scaled by runtime

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated --shader test_finite_diff_raymarching_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file tf_logo.png --gt_transposed --suffix _from_real_fd_h_01_scaled_niter --save_all_loss --no_reset_opt --backend hl --quiet --multi_scale_optimization --autoscheduler --finite_diff_spsa_samples 1 --niters 2000 --alternating_times 5 --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated --shader test_finite_diff_raymarching_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file tf_logo.png --gt_transposed --suffix _from_real_fd_h_001_scaled_niter --save_all_loss --no_reset_opt --backend hl --quiet --multi_scale_optimization --autoscheduler --finite_diff_spsa_samples 1 --niters 2000 --alternating_times 5 --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated --shader test_finite_diff_raymarching_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.001 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file tf_logo.png --gt_transposed --suffix _from_real_fd_h_0001_scaled_niter --save_all_loss --no_reset_opt --backend hl --quiet --multi_scale_optimization --autoscheduler --finite_diff_spsa_samples 1 --niters 2000 --alternating_times 5 --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated --shader test_finite_diff_raymarching_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.0001 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file tf_logo.png --gt_transposed --suffix _from_real_fd_h_00001_scaled_niter --save_all_loss --no_reset_opt --backend hl --quiet --multi_scale_optimization --autoscheduler --finite_diff_spsa_samples 1 --niters 2000 --alternating_times 5 --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated --shader test_finite_diff_raymarching_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.00001 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file tf_logo.png --gt_transposed --suffix _from_real_fd_h_000001_scaled_niter --save_all_loss --no_reset_opt --backend hl --quiet --multi_scale_optimization --autoscheduler --finite_diff_spsa_samples 1 --niters 2000 --alternating_times 5 --autoscheduler --no_reset_sigma

-------------------------
# half sample as FD: 16 samples

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated --shader test_finite_diff_raymarching_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file tf_logo.png --gt_transposed --suffix _from_real_fd_h_01 --save_all_loss --no_reset_opt --backend hl --quiet --multi_scale_optimization --autoscheduler --finite_diff_spsa_samples 16 --alternating_times 5 --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated --shader test_finite_diff_raymarching_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file tf_logo.png --gt_transposed --suffix _from_real_fd_h_001 --save_all_loss --no_reset_opt --backend hl --quiet --multi_scale_optimization --autoscheduler --finite_diff_spsa_samples 16 --alternating_times 5 --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated --shader test_finite_diff_raymarching_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.001 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file tf_logo.png --gt_transposed --suffix _from_real_fd_h_0001 --save_all_loss --no_reset_opt --backend hl --quiet --multi_scale_optimization --autoscheduler --finite_diff_spsa_samples 16 --alternating_times 5 --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated --shader test_finite_diff_raymarching_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.0001 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file tf_logo.png --gt_transposed --suffix _from_real_fd_h_00001 --save_all_loss --no_reset_opt --backend hl --quiet --multi_scale_optimization --autoscheduler --finite_diff_spsa_samples 16 --alternating_times 5 --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated --shader test_finite_diff_raymarching_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.00001 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file tf_logo.png --gt_transposed --suffix _from_real_fd_h_000001 --save_all_loss --no_reset_opt --backend hl --quiet --multi_scale_optimization --autoscheduler --finite_diff_spsa_samples 16 --alternating_times 5 --autoscheduler --no_reset_sigma

---------------------------------------------
# vanilla opt
# no random noise, no alternation, no multi stage
# Runtime
# Ours: 0.0015
# SPSA: 0.0017
# SPSA iter match ours runtime: 2000 (should be at least as much as ours)

-------------------------
# 1 sample (only runned with one choice of h to get a runtime estimate)

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated --shader test_finite_diff_raymarching_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file tf_logo.png --gt_transposed --suffix _from_real_vanilla_fd_h_01 --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --finite_diff_spsa_samples 1 --autoscheduler --no_reset_sigma

-------------------------
# 1 sample iteration scaled by runtime

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated --shader test_finite_diff_raymarching_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file tf_logo.png --gt_transposed --suffix _from_real_vanilla_fd_h_01_scaled_niter --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --finite_diff_spsa_samples 1 --autoscheduler --niters 2000 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated --shader test_finite_diff_raymarching_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file tf_logo.png --gt_transposed --suffix _from_real_vanilla_fd_h_001_scaled_niter --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --finite_diff_spsa_samples 1 --autoscheduler --niters 2000 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated --shader test_finite_diff_raymarching_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.001 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file tf_logo.png --gt_transposed --suffix _from_real_vanilla_fd_h_0001_scaled_niter --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --finite_diff_spsa_samples 1 --autoscheduler --niters 2000 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated --shader test_finite_diff_raymarching_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.0001 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file tf_logo.png --gt_transposed --suffix _from_real_vanilla_fd_h_00001_scaled_niter --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --finite_diff_spsa_samples 1 --autoscheduler --niters 2000 --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated --shader test_finite_diff_raymarching_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.00001 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file tf_logo.png --gt_transposed --suffix _from_real_vanilla_fd_h_000001_scaled_niter --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --finite_diff_spsa_samples 1 --autoscheduler --niters 2000 --no_reset_sigma

-------------------------
# half sample as FD: 16 samples

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated --shader test_finite_diff_raymarching_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.1 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file tf_logo.png --gt_transposed --suffix _from_real_vanilla_fd_h_01 --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --finite_diff_spsa_samples 16 --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated --shader test_finite_diff_raymarching_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file tf_logo.png --gt_transposed --suffix _from_real_vanilla_fd_h_001 --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --finite_diff_spsa_samples 16 --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated --shader test_finite_diff_raymarching_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.100 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file tf_logo.png --gt_transposed --suffix _from_real_vanilla_fd_h_0001 --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --finite_diff_spsa_samples 16 --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated --shader test_finite_diff_raymarching_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.0001 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file tf_logo.png --gt_transposed --suffix _from_real_vanilla_fd_h_00001 --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --finite_diff_spsa_samples 16 --autoscheduler --no_reset_sigma

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated --shader test_finite_diff_raymarching_structured_tf_logo --init_values_pool apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.00001 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file tf_logo.png --gt_transposed --suffix _from_real_vanilla_fd_h_000001 --save_all_loss --no_reset_opt --backend hl --quiet --autoscheduler --finite_diff_spsa_samples 16 --autoscheduler --no_reset_sigma

------------------------------------------------------------------------------------------------------------------------------
# command for generating glsl in a seperate process

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated --shader test_finite_diff_raymarching_structured_tf_logo --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated/best_par_from_real.npy --modes render --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file tf_logo.png --gt_transposed --backend hl --quiet --multi_scale_optimization --autoscheduler

# animation

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated --shader test_finite_diff_raymarching_structured_tf_logo --init_values_pool test_finite_diff_structured_tf_logo_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file tf_logo.png --gt_transposed --suffix _animation --save_all_par --no_reset_opt --backend hl --quiet --multi_scale_optimization --autoscheduler --alternating_times 5

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated --shader test_finite_diff_raymarching_structured_tf_logo --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated/ours_both_sides_5_scale_L2_adam_1.0e-02_animation_result2_0.npy --modes render --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --gt_file tf_logo.png --gt_transposed --suffix _animation --save_all_par --no_reset_opt --backend hl --quiet --multi_scale_optimization --autoscheduler --alternating_times 5 --suffix _opt

cd /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated; ffmpeg -i init_opt%d.png -r 30 -c:v libx264 -preset slow -crf 0 -r 30 animation.mp4

------------------------------------------------------------------------------------------------------------------------------
# Quantitative Metric using 2D box kernels

# Runtime
# Ours: 0.0028
# SPSA: 0.0017

# SPSA needs 2 samples for similar runtime with ours in opt

# get endpoints

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated --shader test_finite_diff_raymarching_structured_tf_logo --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated/best_par_from_real.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_col --render_size 640,640 --line_endpoints_method random_smooth --deriv_metric_visualization_thre 0.1 --deriv_metric_max_halflen 2e-3 --deriv_metric_line --deriv_metric_finite_diff_schedule 0 --kernel_nsamples 1 --deriv_n 2 --deriv_metric_no_ours --our_filter_direction 2d --deriv_metric_suffix _2D_kernel --autoscheduler

# rhs
python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated --shader test_finite_diff_raymarching_structured_tf_logo --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated/best_par_from_real.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_col --render_size 640,640 --line_endpoints_method random_smooth --deriv_metric_visualization_thre 0.1 --deriv_metric_max_halflen 2e-3 --deriv_metric_line --deriv_metric_finite_diff_schedule 0 --kernel_nsamples 100000 --deriv_n 2 --deriv_metric_no_ours --our_filter_direction 2d --deriv_metric_suffix _2D_kernel --deriv_metric_endpoint_file /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated/random_smooth_metric_2X1_len_0.002000_2D_kernel_endpoints.npy --autoscheduler --kernel_sigma 0.1

----------------------------------------------------------------------------------------
# ours

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated --shader test_finite_diff_raymarching_structured_tf_logo --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated/best_par_from_real.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_color --render_size 640,640 --line_endpoints_method kernel_smooth_debug --deriv_metric_visualization_thre 0.1 --deriv_metric_max_halflen 2e-3 --deriv_metric_line --deriv_metric_endpoint_file /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated/random_smooth_metric_2X1_len_0.002000_2D_kernel_endpoints.npy --deriv_metric_rhs_file /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated/random_smooth_metric_2X1_len_0.002000_2D_kernel_rhs.npy --deriv_metric_suffix _2D_kernel --kernel_sigma 0 --kernel_uv_sigma 1 --deriv_n 10000 --deriv_metric_finite_diff_schedule 0 --kernel_smooth_exclude_our_kernel --autoscheduler --kernel_nsamples 1 --kernel_sigma 0.1

----------------------------------------------------------------------------------------
# FD

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated --shader test_finite_diff_raymarching_structured_tf_logo --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated/best_par_from_real.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_color --render_size 640,640 --line_endpoints_method kernel_smooth_debug --deriv_metric_visualization_thre 0.1 --deriv_metric_max_halflen 2e-3 --deriv_metric_line --deriv_metric_endpoint_file /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated/random_smooth_metric_2X1_len_0.002000_2D_kernel_endpoints.npy --deriv_metric_rhs_file /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated/random_smooth_metric_2X1_len_0.002000_2D_kernel_rhs.npy --deriv_metric_suffix _2D_kernel_FD --kernel_sigma 0 --kernel_uv_sigma 1 --deriv_n 10000 --deriv_metric_finite_diff_schedule 1e-1,1e-2,1e-3,1e-4,1e-5 --kernel_smooth_exclude_our_kernel --deriv_metric_no_ours --kernel_nsamples 1 --autoscheduler --kernel_sigma 0.1

----------------------------------------------------------------------------------------
# SPSA
# 1 samples
# number of samples corresponds to runtime in our optimization pipeline (multi stage loss)

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated --shader test_finite_diff_raymarching_structured_tf_logo --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated/best_par_from_real.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_color --render_size 640,640 --line_endpoints_method kernel_smooth_debug --deriv_metric_visualization_thre 0.1 --deriv_metric_max_halflen 2e-3 --deriv_metric_line --deriv_metric_endpoint_file /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated/random_smooth_metric_2X1_len_0.002000_2D_kernel_endpoints.npy --deriv_metric_rhs_file /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated/random_smooth_metric_2X1_len_0.002000_2D_kernel_rhs.npy --deriv_metric_suffix _2D_kernel_SPSA_1 --kernel_sigma 0 --kernel_uv_sigma 1 --deriv_n 10000 --deriv_metric_finite_diff_schedule 1e-1,1e-2,1e-3,1e-4,1e-5 --kernel_smooth_exclude_our_kernel --deriv_metric_no_ours --finite_diff_spsa_samples 1 --kernel_nsamples 1 --autoscheduler --kernel_sigma 0.1

------------------------------------------------------------------------------------------------------------------------------
# plot

---------------------------------------------
# preprocess

python preprocess_raw_loss_data.py /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated/ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated/finite_diff16_both_sides_5_scale_L2_adam_1.0e-02_from_real_fd_h_000001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated/finite_diff16_both_sides_5_scale_L2_adam_1.0e-02_from_real_fd_h_00001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated/finite_diff16_both_sides_5_scale_L2_adam_1.0e-02_from_real_fd_h_0001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated/finite_diff16_both_sides_5_scale_L2_adam_1.0e-02_from_real_fd_h_001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated/finite_diff16_both_sides_5_scale_L2_adam_1.0e-02_from_real_fd_h_01_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated/finite_diff16_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_000001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated/finite_diff16_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_00001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated/finite_diff16_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_0001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated/finite_diff16_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated/finite_diff16_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_01_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_fd_h_000001_scaled_niter_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_fd_h_00001_scaled_niter_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_fd_h_0001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_fd_h_0001_scaled_niter_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_fd_h_001_scaled_niter_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_fd_h_01_scaled_niter_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_000001_scaled_niter_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_00001_scaled_niter_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_0001_scaled_niter_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_001_scaled_niter_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_01_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated/finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_01_scaled_niter_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated/finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_fd_h_000001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated/finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_fd_h_00001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated/finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_fd_h_0001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated/finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_fd_h_001_all_loss.npy /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated/finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_fd_h_01_all_loss.npy

---------------------------------------------
# plot median 

# ours vs FD

python plot_opt_baseline.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated --filenames ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_all_loss.npy,finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_fd_h_01_all_loss.npy,finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_fd_h_001_all_loss.npy,finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_fd_h_0001_all_loss.npy,finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_fd_h_00001_all_loss.npy,finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_fd_h_000001_all_loss.npy --labels ours,FD_1e-1,FD_1e-2,FD_1e-3,FD_1e-4,FD_1e-5 --scales 0.0028,0.0250,0.0247,0.0245,0.0248,0.0249 --suffix _ours_FD_median --median

# ours vs SPSA
python plot_opt_baseline.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated --filenames ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_fd_h_01_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_fd_h_001_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_fd_h_0001_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_fd_h_00001_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_fd_h_000001_scaled_niter_all_loss.npy --labels ours,SPSA_1e-1,SPSA_1e-2,SPSA_1e-3,SPSA_1e-4,SPSA_1e-5 --scales 0.0028,0.0017,0.0017,0.0017,0.0017,0.0017 --suffix _ours_SPSA_median --median

# ours vs SPSA multiple samples (16, half samples as FD)
python plot_opt_baseline.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated --filenames ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_all_loss.npy,finite_diff16_both_sides_5_scale_L2_adam_1.0e-02_from_real_fd_h_01_all_loss.npy,finite_diff16_both_sides_5_scale_L2_adam_1.0e-02_from_real_fd_h_001_all_loss.npy,finite_diff16_both_sides_5_scale_L2_adam_1.0e-02_from_real_fd_h_0001_all_loss.npy,finite_diff16_both_sides_5_scale_L2_adam_1.0e-02_from_real_fd_h_00001_all_loss.npy,finite_diff16_both_sides_5_scale_L2_adam_1.0e-02_from_real_fd_h_000001_all_loss.npy --labels ours,multi_SPSA_1e-1,multi_SPSA_1e-2,multi_SPSA_1e-3,multi_SPSA_1e-4,multi_SPSA_1e-5 --scales 0.0028,0.0144,0.0144,0.0144,0.0143,0.0144 --suffix _ours_multi_SPSA_median --median

# ours vs SPSA vanilla
python plot_opt_baseline.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated --filenames ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_01_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_001_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_0001_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_00001_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_000001_scaled_niter_all_loss.npy --labels ours,SPSA_vanilla_1e-1,SPSA_vanilla_1e-2,SPSA_vanilla_1e-3,SPSA_vanilla_1e-4,SPSA_vanilla_1e-5 --scales 0.0028,0.0017,0.0017,0.0017,0.0017,0.0017 --suffix _ours_SPSA_vanilla_median --median

# ours vs SPSA vanilla multiple samples (16, half samples as FD)
python plot_opt_baseline.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated --filenames ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_all_loss.npy,finite_diff16_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_01_all_loss.npy,finite_diff16_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_001_all_loss.npy,finite_diff16_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_0001_all_loss.npy,finite_diff16_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_00001_all_loss.npy,finite_diff16_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_000001_all_loss.npy --labels ours,multi_SPSA_vanilla_1e-1,multi_SPSA_vanilla_1e-2,multi_SPSA_vanilla_1e-3,multi_SPSA_vanilla_1e-4,multi_SPSA_vanilla_1e-5 --scales 0.0028,0.0145,0.0145,0.0145,0.0143,0.0142 --suffix _ours_multi_SPSA_vanilla_median --median

# ours vs FD ans SPSA on best performing h
python plot_opt_baseline.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated --filenames ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_all_loss.npy,finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_fd_h_001_all_loss.npy,finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_fd_h_0001_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_fd_h_001_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_fd_h_0001_scaled_niter_all_loss.npy,finite_diff16_both_sides_5_scale_L2_adam_1.0e-02_from_real_fd_h_001_all_loss.npy,finite_diff16_both_sides_5_scale_L2_adam_1.0e-02_from_real_fd_h_0001_all_loss.npy --labels ours,FD_1e-2,FD_1e-3,SPSA_1e-2,SPSA_1e-3,multi_SPSA_1e-2,multi_SPSA_1e-3 --scales 0.0028,0.0247,0.0245,0.0017,0.0017,0.0144,0.0144 --suffix _ours_FD_SPSA_best_h_median --median

# ours vs FD ans SPSA vanilla on best performing h
python plot_opt_baseline.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated --filenames ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_001_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_0001_scaled_niter_all_loss.npy,finite_diff16_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_001_all_loss.npy,finite_diff16_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_0001_all_loss.npy --labels ours,SPSA_vanilla_1e-2,SPSA_vanilla_1e-3,multi_SPSA_vanilla_1e-2,multi_SPSA_vanilla_1e-4 --scales 0.0028,0.0017,0.0017,0.0145,0.0145 --suffix _ours_FD_SPSA_vanilla_best_h_median --median

---------------------------------------------
# plot transparent

# ours vs FD ans SPSA on best performing h
python plot_opt_baseline.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated --filenames ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_all_loss.npy,finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_fd_h_001_all_loss.npy,finite_diff_both_sides_5_scale_L2_adam_1.0e-02_from_real_fd_h_0001_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_fd_h_001_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_fd_h_0001_scaled_niter_all_loss.npy,finite_diff16_both_sides_5_scale_L2_adam_1.0e-02_from_real_fd_h_001_all_loss.npy,finite_diff16_both_sides_5_scale_L2_adam_1.0e-02_from_real_fd_h_0001_all_loss.npy --labels ours,FD_1e-2,FD_1e-3,SPSA_1e-2,SPSA_1e-3,multi_SPSA_1e-2,multi_SPSA_1e-3 --scales 0.0028,0.0247,0.0245,0.0017,0.0017,0.0144,0.0144 --suffix _ours_FD_SPSA_best_h_transparent --transparent

# ours vs FD ans SPSA vanilla on best performing h
python plot_opt_baseline.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated --filenames ours_both_sides_5_scale_L2_adam_1.0e-02_from_real_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_001_scaled_niter_all_loss.npy,finite_diff1_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_0001_scaled_niter_all_loss.npy,finite_diff16_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_001_all_loss.npy,finite_diff16_both_sides_5_scale_L2_adam_1.0e-02_from_real_vanilla_fd_h_0001_all_loss.npy --labels ours,SPSA_vanilla_1e-2,SPSA_vanilla_1e-3,multi_SPSA_vanilla_1e-2,multi_SPSA_vanilla_1e-4 --scales 0.0028,0.0017,0.0017,0.0145,0.0145 --suffix _ours_FD_SPSA_vanilla_best_h_transparent --transparent

---------------------------------------------
# plot metric

python metric_compare_line_integral.py --baseline_dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated --deriv_metric_suffix _2D_kernel_FD_finite_diff_0.100000,_2D_kernel_FD_finite_diff_0.010000,_2D_kernel_FD_finite_diff_0.001000,_2D_kernel_FD_finite_diff_0.000100,_2D_kernel_FD_finite_diff_0.000010,_2D_kernel_SPSA_1_finite_diff_0.100000,_2D_kernel_SPSA_1_finite_diff_0.010000,_2D_kernel_SPSA_1_finite_diff_0.001000,_2D_kernel_SPSA_1_finite_diff_0.000100,_2D_kernel_SPSA_1_finite_diff_0.000010,_2D_kernel --eval_labels FD1e-1,FD1e-2,FD1e-3,FD1e-4,FD1e-5,2SPSA1e-1,2SPSA1e-2,2SPSA1e-3,2SPSA1e-4,2SPSA1e-5,ours --max_half_len 2e-3 --rhs_file /n/fs/scratch/yutingy/test_finite_diff_raymarching_structured_tf_logo_updated/random_smooth_metric_2X100000_len_0.002000_2D_kernel_rhs.npy --visualization_thre 0.01 --ncols 5

"""

from render_util import *
from render_single import render_single

nargs = 31

args_range = np.ones(nargs)

width = ArgumentScalar('width')
height = ArgumentScalar('height')

raymarching_loop = 64

compiler.log_prefix_only = False
compiler.log_intermediates_less = True

use_select_rule = 1


def sdBox_deriv(pos, boxes):
    
    dist = None
    deriv = None
    
    for box in boxes:
        
        offset = np.array(pos) - box.pos 
        
        # avoid dimension going to negative
        f0 = abs(offset[0]) - box.dim[0]
        f1 = abs(offset[1]) - box.dim[1]
        f2 = abs(offset[2]) - box.dim[2]
    
        q0 = maximum(f0, 0)
        q1 = maximum(f1, 0)
        q2 = maximum(f2, 0)
    
        max_f1f2 = maximum(f1, f2)

        current_dist = maximum(f0, max_f1f2)
    
        choose_f0 = f0 >= max_f1f2
        choose_f1 = f1 >= f2
    
        current_deriv = [select(choose_f0, sign(offset[0]), 0.),
                         select(choose_f0, 0., select(choose_f1, sign(offset[1]), 0.)),
                         select(choose_f0, 0., select(choose_f1, 0., sign(offset[2])))]
        
        if dist is not None:
            choose_old = dist <= current_dist
            dist = minimum(dist, current_dist)
            for idx in range(3):
                deriv[idx] = select(choose_old, deriv[idx], current_deriv[idx])
        else:
            dist = current_dist
            deriv = current_deriv
    
    return dist, 1, deriv[0], deriv[1], deriv[2]

def test_finite_diff_raymarching_structured_tf_logo(u, v, X, scalar_loss=None):
    """
    X has shape nargs + 3
    first 3 entries are u, v, time
    the other entries are scalar tunable parameters
    """
    
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
    
    def raymarching_body(x, y, z):
        return sdBox_deriv([x, y, z], boxes)
    
    raymarching_ans = RaymarchingWrapper(raymarching_body, ro, rd, 0, raymarching_loop, include_derivs=True)
    
    cond_converge = raymarching_ans[0]
    t_closest = raymarching_ans[1]
    
    deriv_sdf = [raymarching_ans[6],
                 raymarching_ans[7],
                 raymarching_ans[8]]
    
    lig = np.array([cos(lig_ang0),
                    sin(lig_ang0) * cos(lig_ang1),
                    sin(lig_ang0) * sin(lig_ang1)])
    
    dif = dot(lig, deriv_sdf)
    
    #dif = select(dif > 0, dif, 0)
    
    col = select(cond_converge, amb + dif * kd, 1.)
    
    return col
    
    #col = select(cond_converge, 0., 1.)

    return output_color([col, col, col])
        
shaders = [test_finite_diff_raymarching_structured_tf_logo]
is_color = True

def sample_init(nsamples, target_width=640, target_height=640):
    
    init = np.zeros([nsamples, nargs])
    
    # random sample light col and direction
    init[:, 8:14] = np.random.rand(nsamples, 6) * 0.5
    init[:, 14:16] = np.random.rand(nsamples, 2) * 2 * np.pi
    
    # random sample box dimension and position
    # dimension base is always sampled between 0.2 and 0.4
    # longer edge is sampled 2x of the base
    # position is always sampled between -0.1 to 0.1 offset to the ref position
    
    # pos of box 0
    init[:, 16:19] = np.random.rand(nsamples, 3) * 0.2 - 0.1
    # dim of box 0
    init[:, 19:22] = np.random.rand(nsamples, 3) * 0.2 + 0.2
    init[:, 21] *= 2
    # dim of box 1
    init[:, 22:24] = np.random.rand(nsamples, 2) * 0.2 + 0.2
    init[:, 22] *= 2
    # pos of box 1
    init[:, 24] = np.random.rand(nsamples) * 0.2 - 0.1 + init[:, 18]
    
    # Use center of box 1 as proxy center of the structure
    proxy_x = init[:, 16]
    proxy_y = init[:, 17] - init[:, 20] - init[:, 22]
    proxy_z = init[:, 24]
    
    # dim of box 2
    init[:, 25:28] = np.random.rand(nsamples, 3) * 0.2 + 0.2
    init[:, 25] *= 2
    # dim of box 3
    init[:, 28:30] = np.random.rand(nsamples, 2) * 0.2 + 0.2
    # pos of box 3
    init[:, 30] = np.random.rand(nsamples) * 0.2 - 0.1 + proxy_y
    
    # rejection sampling for camera parameters
    for idx in range(nsamples):
                
        while True:
            scale_x, scale_y = np.random.rand(2) * 2 + 5
            x, y, z = np.random.rand(3) * 2 + 1
            z += 1
            ang1, ang2, ang3 = np.random.rand(3) * np.pi * 2
            
            sin1 = np.sin(ang1)
            cos1 = np.cos(ang1)
            sin2 = np.sin(ang2)
            cos2 = np.cos(ang2)
            sin3 = np.sin(ang3)
            cos3 = np.cos(ang3)
            
            ray_dir = np.array([sin1 * sin3 + cos1 * sin2 * cos3,
                                -sin1 * cos3 + cos1 * sin2 * sin3,
                                cos1 * cos2])
            
            # Rejection criterion 1:
            # the ray direction should be within the quadrature that all faces needed are visible
            # TODO: is this really necessary? Or do we only need to make sure all 4 boxes are visible
            # TODO: first try the easier setup, then see if we can extend to the more difficult one
            if np.any(ray_dir >= -0.2):
                continue
                
            # Rejection criterion 2:
            # These are the assumptions made when modeling the structure
            # image space horizontal should be roughly aligned with world coord x
            # image space vertical should be roughly aligned with world coord y
            horizontal_offset_dir = cos2 * cos3
            vertical_offset_dir = cos1 * cos3 + sin1 * sin2 * sin3
            ang_thre = 0.75
            if horizontal_offset_dir <= ang_thre or vertical_offset_dir >= -ang_thre:
                continue
                
            break
                
        # camera x, y, z should be constrained to target proxy center
        t = np.random.rand() * 2 + 3
        x = proxy_x[idx] - t * ray_dir[0]
        y = proxy_y[idx] - t * ray_dir[1]
        z = proxy_z[idx] - t * ray_dir[2]
                
        init[idx, :8] = [scale_x, scale_y,
                         x, y, z,
                         ang1, ang2, ang3]
        
    return init

if __name__ == '__main__':
    init = sample_init(100)
    np.save('../apps/example_init_values/test_finite_diff_raymarching_structured_tf_logo_extra_init_values_pool.npy', init)
    