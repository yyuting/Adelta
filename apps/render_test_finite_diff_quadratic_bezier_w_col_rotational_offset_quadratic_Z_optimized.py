"""

echo "NOTE: The spline representation is not pixel-wise perfectly reconstructing the target animation frames, the overlapping regions may not correspond exactly. This occasionally lead to the problem that the optimal parameters in the L2 sense do not correspond to human intuition. Therefore, human effort may be involved to reject these optimization results at various stages."

------------------------------------------------------------------------------------------------------------------------------
# optimization with random var

----------------------------------------------------------------------------------------

# thief knot

python check_new_rope.py knots_imgs/thief/1.png knots_imgs/thief/2.png knots_imgs/thief/3.png knots_imgs/thief/4.png knots_imgs/thief/5.png knots_imgs/thief/6.png knots_imgs/thief/7.png knots_imgs/thief/8.png

mkdir /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_1_rotational_offset_quadratic_Z_optimized_knot_1

python bezier_util.py initialize 5 101,295 214,273 1 /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_1_rotational_offset_quadratic_Z_optimized_knot_1/sampled_guess.npy

# nsplines = 1

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_1_rotational_offset_quadratic_Z_optimized_knot_1 --shader test_finite_diff_quadratic_bezier_w_col_rotational_offset_quadratic_Z_optimized --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_1_rotational_offset_quadratic_Z_optimized_knot_1/sampled_guess.npy --gt_file knots_imgs/thief/1.png --gt_transposed --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 480,480 --is_color --backend hl --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 0.05 --no_reset_opt --no_binary_search_std --save_all_loss --quiet --suffix _sampled_guess_1 --save_best_par --multi_scale_optimization --ignore_last_n_scale 2 --base_loss_stage 2 --shader_args nropes:1#all_nsplines:[1] --ignore_glsl

# nsplines = 2

mkdir /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_2_rotational_offset_quadratic_Z_optimized_knot_1

python bezier_util.py expand 2 1 1 /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_1_rotational_offset_quadratic_Z_optimized_knot_1/best_par_sampled_guess_1.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_2_rotational_offset_quadratic_Z_optimized_knot_1/sampled_guess

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_2_rotational_offset_quadratic_Z_optimized_knot_1 --shader test_finite_diff_quadratic_bezier_w_col_rotational_offset_quadratic_Z_optimized --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_2_rotational_offset_quadratic_Z_optimized_knot_1/sampled_guess.npy --gt_file knots_imgs/thief/2.png --gt_transposed --modes search_init --metrics 5_scale_L2 --render_size 480,480 --is_color --backend hl --ninit_samples 200 --ninit_best 5 --suffix _searched_guess --search_type 2 --shader_args nropes:1#all_nsplines:[2] --ignore_glsl

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_2_rotational_offset_quadratic_Z_optimized_knot_1 --shader test_finite_diff_quadratic_bezier_w_col_rotational_offset_quadratic_Z_optimized --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_2_rotational_offset_quadratic_Z_optimized_knot_1/sampled_searched_guess_init_200_5.npy --gt_file knots_imgs/thief/2.png --gt_transposed --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 480,480 --is_color --backend hl --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 0.05 --no_reset_opt --no_binary_search_std --save_all_loss --quiet --suffix _sampled_guess_2 --save_best_par --multi_scale_optimization --ignore_last_n_scale 2 --base_loss_stage 2 --target_par_file /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_2_rotational_offset_quadratic_Z_optimized_knot_1/sampled_guess.npy --target_weight_file /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_2_rotational_offset_quadratic_Z_optimized_knot_1/sampled_guess_weight.npy --target_regularizer_scale 1e-10 --shader_args nropes:1#all_nsplines:[2] --ignore_glsl

# nsplines = 3

mkdir /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_3_rotational_offset_quadratic_Z_optimized_knot_1

python bezier_util.py expand 3 1 1 /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_2_rotational_offset_quadratic_Z_optimized_knot_1/best_par_sampled_guess_2.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_3_rotational_offset_quadratic_Z_optimized_knot_1/sampled_guess

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_3_rotational_offset_quadratic_Z_optimized_knot_1 --shader test_finite_diff_quadratic_bezier_w_col_rotational_offset_quadratic_Z_optimized --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_3_rotational_offset_quadratic_Z_optimized_knot_1/sampled_guess.npy --gt_file knots_imgs/thief/3.png --gt_transposed --modes search_init --metrics 5_scale_L2 --render_size 480,480 --is_color --backend hl --ninit_samples 200 --ninit_best 5 --suffix _searched_guess --search_type 2 --shader_args nropes:1#all_nsplines:[3] --ignore_glsl

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_3_rotational_offset_quadratic_Z_optimized_knot_1 --shader test_finite_diff_quadratic_bezier_w_col_rotational_offset_quadratic_Z_optimized --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_3_rotational_offset_quadratic_Z_optimized_knot_1/sampled_searched_guess_init_200_5.npy --gt_file knots_imgs/thief/3.png --gt_transposed --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 480,480 --is_color --backend hl --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 0.05 --no_reset_opt --no_binary_search_std --save_all_loss --quiet --suffix _sampled_guess_3 --save_best_par --multi_scale_optimization --ignore_last_n_scale 2 --base_loss_stage 2 --target_par_file /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_3_rotational_offset_quadratic_Z_optimized_knot_1/sampled_guess.npy --target_weight_file /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_3_rotational_offset_quadratic_Z_optimized_knot_1/sampled_guess_weight.npy --target_regularizer_scale 1e-10 --shader_args nropes:1#all_nsplines:[3] --ignore_glsl

# nsplines = 4

mkdir /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_rotational_offset_quadratic_Z_optimized_knot_1

python bezier_util.py expand 4 1 1 /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_3_rotational_offset_quadratic_Z_optimized_knot_1/best_par_sampled_guess_3.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_rotational_offset_quadratic_Z_optimized_knot_1/sampled_guess

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_rotational_offset_quadratic_Z_optimized_knot_1 --shader test_finite_diff_quadratic_bezier_w_col_rotational_offset_quadratic_Z_optimized --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_rotational_offset_quadratic_Z_optimized_knot_1/sampled_guess.npy --gt_file knots_imgs/thief/4.png --gt_transposed --modes search_init --metrics 5_scale_L2 --render_size 480,480 --is_color --backend hl --ninit_samples 200 --ninit_best 5 --suffix _searched_guess --search_type 2 --shader_args nropes:1#all_nsplines:[4] --ignore_glsl

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_rotational_offset_quadratic_Z_optimized_knot_1 --shader test_finite_diff_quadratic_bezier_w_col_rotational_offset_quadratic_Z_optimized --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_rotational_offset_quadratic_Z_optimized_knot_1/sampled_searched_guess_init_200_5.npy --gt_file knots_imgs/thief/4.png --gt_transposed --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 480,480 --is_color --backend hl --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 0.05 --no_reset_opt --no_binary_search_std --save_all_loss --quiet --suffix _sampled_guess_4 --save_best_par --multi_scale_optimization --ignore_last_n_scale 2 --base_loss_stage 2 --target_par_file /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_rotational_offset_quadratic_Z_optimized_knot_1/sampled_guess.npy --target_weight_file /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_rotational_offset_quadratic_Z_optimized_knot_1/sampled_guess_weight.npy --target_regularizer_scale 1e-10 --shader_args nropes:1#all_nsplines:[4] --ignore_glsl

# nropes = 2, nsplines = [4, 1]

mkdir /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_1_rotational_offset_quadratic_Z_optimized_knot_1

python bezier_util.py initialize 5 358,197 236,216 1 /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_1_rotational_offset_quadratic_Z_optimized_knot_1/sampled_guess.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_rotational_offset_quadratic_Z_optimized_knot_1/best_par_sampled_guess_4.npy 4

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_1_rotational_offset_quadratic_Z_optimized_knot_1 --shader test_finite_diff_quadratic_bezier_w_col_rotational_offset_quadratic_Z_optimized --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_1_rotational_offset_quadratic_Z_optimized_knot_1/sampled_guess.npy --gt_file knots_imgs/thief/5.png --gt_transposed --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 480,480 --is_color --backend hl --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 0.05 --no_reset_opt --no_binary_search_std --save_all_loss --quiet --suffix _sampled_guess_5 --save_best_par --multi_scale_optimization --ignore_last_n_scale 2 --base_loss_stage 2 --target_par_file /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_1_rotational_offset_quadratic_Z_optimized_knot_1/sampled_guess.npy --target_weight_file /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_1_rotational_offset_quadratic_Z_optimized_knot_1/sampled_guess_weight.npy --target_regularizer_scale 1e-10 --shader_args nropes:2#all_nsplines:[4,1] --ignore_glsl

# nropes = 2, nsplines = [4, 2]

mkdir /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_2_rotational_offset_quadratic_Z_optimized_knot_1

python bezier_util.py expand 2 1 1 /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_1_rotational_offset_quadratic_Z_optimized_knot_1/best_par_sampled_guess_5.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_2_rotational_offset_quadratic_Z_optimized_knot_1/sampled_guess 4

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_2_rotational_offset_quadratic_Z_optimized_knot_1 --shader test_finite_diff_quadratic_bezier_w_col_rotational_offset_quadratic_Z_optimized --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_2_rotational_offset_quadratic_Z_optimized_knot_1/sampled_guess.npy --gt_file knots_imgs/thief/6.png --gt_transposed --modes search_init --metrics 5_scale_L2 --render_size 480,480 --is_color --backend hl --ninit_samples 200 --ninit_best 5 --suffix _searched_guess --search_type 2 --shader_args nropes:2#all_nsplines:[4,2] --ignore_glsl

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_2_rotational_offset_quadratic_Z_optimized_knot_1 --shader test_finite_diff_quadratic_bezier_w_col_rotational_offset_quadratic_Z_optimized --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_2_rotational_offset_quadratic_Z_optimized_knot_1/sampled_searched_guess_init_200_5.npy --gt_file knots_imgs/thief/6.png --gt_transposed --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 480,480 --is_color --backend hl --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 0.05 --no_reset_opt --no_binary_search_std --save_all_loss --quiet --suffix _sampled_guess_6 --save_best_par --multi_scale_optimization --ignore_last_n_scale 2 --base_loss_stage 2 --target_par_file /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_2_rotational_offset_quadratic_Z_optimized_knot_1/sampled_guess.npy --target_weight_file /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_2_rotational_offset_quadratic_Z_optimized_knot_1/sampled_guess_weight.npy --target_regularizer_scale 1e-10 --shader_args nropes:2#all_nsplines:[4,2] --ignore_glsl

# nropes = 2, nsplines = [4, 3]

mkdir /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_3_rotational_offset_quadratic_Z_optimized_knot_1

python bezier_util.py expand 3 1 1 /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_2_rotational_offset_quadratic_Z_optimized_knot_1/best_par_sampled_guess_6.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_3_rotational_offset_quadratic_Z_optimized_knot_1/sampled_guess 4

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_3_rotational_offset_quadratic_Z_optimized_knot_1 --shader test_finite_diff_quadratic_bezier_w_col_rotational_offset_quadratic_Z_optimized --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_3_rotational_offset_quadratic_Z_optimized_knot_1/sampled_guess.npy --gt_file knots_imgs/thief/7.png --gt_transposed --modes search_init --metrics 5_scale_L2 --render_size 480,480 --is_color --backend hl --ninit_samples 200 --ninit_best 5 --suffix _searched_guess --search_type 2 --shader_args nropes:2#all_nsplines:[4,3] --ignore_glsl

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_3_rotational_offset_quadratic_Z_optimized_knot_1 --shader test_finite_diff_quadratic_bezier_w_col_rotational_offset_quadratic_Z_optimized --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_3_rotational_offset_quadratic_Z_optimized_knot_1/sampled_searched_guess_init_200_5.npy --gt_file knots_imgs/thief/7.png --gt_transposed --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 480,480 --is_color --backend hl --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 0.05 --no_reset_opt --no_binary_search_std --save_all_loss --quiet --suffix _sampled_guess_7 --save_best_par --multi_scale_optimization --ignore_last_n_scale 2 --base_loss_stage 2 --target_par_file /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_3_rotational_offset_quadratic_Z_optimized_knot_1/sampled_guess.npy --target_weight_file /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_3_rotational_offset_quadratic_Z_optimized_knot_1/sampled_guess_weight.npy --target_regularizer_scale 1e-10 --shader_args nropes:2#all_nsplines:[4,3] --ignore_glsl

# nropes = 2, nsplines = [4, 4]

mkdir /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1

python bezier_util.py expand 4 5 1 /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_3_rotational_offset_quadratic_Z_optimized_knot_1/best_par_sampled_guess_7.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1/sampled_guess 4

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1 --shader test_finite_diff_quadratic_bezier_w_col_rotational_offset_quadratic_Z_optimized --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1/sampled_guess.npy --gt_file knots_imgs/thief/8.png --gt_transposed --modes search_init --metrics 5_scale_L2 --render_size 480,480 --is_color --backend hl --ninit_samples 200 --ninit_best 5 --suffix _searched_guess --search_type 2 --shader_args nropes:2#all_nsplines:[4,4] --ignore_glsl

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1 --shader test_finite_diff_quadratic_bezier_w_col_rotational_offset_quadratic_Z_optimized --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1/sampled_searched_guess_init_200_5.npy --gt_file knots_imgs/thief/8.png --gt_transposed --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 480,480 --is_color --backend hl --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 0.05 --no_reset_opt --no_binary_search_std --save_all_loss --quiet --suffix _sampled_guess_8 --save_best_par --multi_scale_optimization --ignore_last_n_scale 2 --base_loss_stage 2 --target_par_file /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1/sampled_guess.npy --target_weight_file /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1/sampled_guess_weight.npy --target_regularizer_scale 1e-10 --refine_opt --shader_args nropes:2#all_nsplines:[4,4] --ignore_glsl

python bezier_util.py reset_phase 4,4 5 1 /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1/best_par_sampled_guess_8.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1/reset_phase.npy

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1 --shader test_finite_diff_quadratic_bezier_w_col_rotational_offset_quadratic_Z_optimized --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1/reset_phase.npy --gt_file knots_imgs/thief/8.png --gt_transposed --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 480,480 --is_color --backend hl --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 2 --no_reset_opt --no_binary_search_std --save_all_loss --quiet --suffix _finalize_phase --save_best_par --multi_scale_optimization --ignore_last_n_scale 2 --base_loss_stage 2 --opt_subset_idx 32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49 --shader_args nropes:2#all_nsplines:[4,4]

# pass back phase

python bezier_util.py transfer expand 4,4 4,3 1 0,0,0,0,0,0 /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1/best_par_finalize_phase.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_3_rotational_offset_quadratic_Z_optimized_knot_1/best_par_sampled_guess_7.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1/best_par_from_4_3_auto.npy 

python bezier_util.py transfer expand 4,4 4,2 1 0,0,0,0,0,0 /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1/best_par_finalize_phase.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_2_rotational_offset_quadratic_Z_optimized_knot_1/best_par_sampled_guess_6.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1/best_par_from_4_2_auto.npy 

python bezier_util.py transfer expand 4,4 4,1 1 0,0,0,0,0,0 /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1/best_par_finalize_phase.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_1_rotational_offset_quadratic_Z_optimized_knot_1/best_par_sampled_guess_5.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1/best_par_from_4_1_auto.npy 

python bezier_util.py transfer expand 4,4 4 1 0,0,0,0,0,0 /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1/best_par_finalize_phase.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_rotational_offset_quadratic_Z_optimized_knot_1/best_par_sampled_guess_4.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1/best_par_from_4_0_auto.npy 

python bezier_util.py transfer expand 4,4 3 1 0,0,0,0,0,0 /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1/best_par_finalize_phase.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_3_rotational_offset_quadratic_Z_optimized_knot_1/best_par_sampled_guess_3.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1/best_par_from_3_0_auto.npy

python bezier_util.py transfer expand 4,4 2 1 0,0,0,0,0,0 /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1/best_par_finalize_phase.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_2_rotational_offset_quadratic_Z_optimized_knot_1/best_par_sampled_guess_2.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1/best_par_from_2_0_auto.npy

python bezier_util.py transfer expand 4,4 1 1 0,0,0,0,0,0 /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1/best_par_finalize_phase.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_1_rotational_offset_quadratic_Z_optimized_knot_1/best_par_sampled_guess_1.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1/best_par_from_1_0_auto.npy

python bezier_util.py interp_multiple 4,4 /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1/best_par_from_1_0_auto_safe.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1/best_par_from_2_0_auto_safe.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1/render_1_to_2_auto.npy 1

python bezier_util.py interp_multiple 4,4 /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1/best_par_from_2_0_auto_safe.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1/best_par_from_3_0_auto_safe.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1/render_2_to_3_auto.npy 2

python bezier_util.py interp_multiple 4,4 /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1/best_par_from_3_0_auto_safe.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1/best_par_from_4_0_auto_safe.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1/render_3_to_4_auto.npy 3

python bezier_util.py interp_multiple 4,4 /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1/best_par_from_4_0_auto.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1/best_par_from_4_1_auto.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1/render_4_0_to_4_1_auto.npy 4,0

python bezier_util.py interp_multiple 4,4 /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1/best_par_from_4_1_auto.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1/best_par_from_4_2_auto.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1/render_4_1_to_4_2_auto.npy 4,1

python bezier_util.py interp_multiple 4,4 /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1/best_par_from_4_2_auto.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1/best_par_from_4_3_auto.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1/render_4_2_to_4_3_auto.npy 4,2

python bezier_util.py interp_multiple 4,4 /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1/best_par_from_4_3_auto.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1/best_par_finalize_phase.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1/render_4_3_to_4_4_auto.npy 4,3

python bezier_util.py combine /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1/render_1_to_2_auto.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1/render_2_to_3_auto.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1/render_3_to_4_auto.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1/render_4_0_to_4_1_auto.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1/render_4_1_to_4_2_auto.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1/render_4_2_to_4_3_auto.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1/render_4_3_to_4_4_auto.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1/render_all.npy

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_rotational_offset_better_rendering_quadratic_Z_knot_1 --shader test_finite_diff_quadratic_bezier_w_col_rotational_offset_better_rendering_quadratic_Z --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1/render_all.npy --gt_file knots_imgs/thief/8.png --gt_transposed --modes render --metrics 3_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_color --unnormalized_par --shader_args nropes:2#all_nsplines:[4,4] --backend tf --camera_size 1500,1500 --render_size 960,480 --tile_offset 232,510

ffmpeg -i /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_rotational_offset_better_rendering_quadratic_Z_knot_1/init%05d.png /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_rotational_offset_better_rendering_quadratic_Z_knot_1/animation.mp4

----------------------------------------------------------------------------------------

# overhand knot

python check_new_rope.py knots_imgs/overhand/1c.png knots_imgs/overhand/2c.png knots_imgs/overhand/4c.png knots_imgs/overhand/6c.png knots_imgs/overhand/7c.png knots_imgs/overhand/9c.png

# initialize using user clicked control points

mkdir /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_1_rotational_offset_quadratic_Z_optimized_knot_0

python bezier_util.py initialize 5 116,265 351,322 1 /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_1_rotational_offset_quadratic_Z_optimized_knot_0/sampled_guess.npy

# nsplines = 1

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_1_rotational_offset_quadratic_Z_optimized_knot_0 --shader test_finite_diff_quadratic_bezier_w_col_rotational_offset_quadratic_Z_optimized --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_1_rotational_offset_quadratic_Z_optimized_knot_0/sampled_guess.npy --gt_file knots_imgs/overhand/1c.png --gt_transposed --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 480,480 --is_color --backend hl --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 0.05 --no_reset_opt --no_binary_search_std --save_all_loss --quiet --suffix _sampled_guess_1_ratio --save_best_par --multi_scale_optimization --ignore_last_n_scale 2 --base_loss_stage 2 --shader_args nropes:1#all_nsplines:[1] --ignore_glsl

# nsplines = 2

mkdir /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_2_rotational_offset_quadratic_Z_optimized_knot_0

python bezier_util.py expand 2 5 1 /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_1_rotational_offset_quadratic_Z_optimized_knot_0/best_par_sampled_guess_1_ratio.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_2_rotational_offset_quadratic_Z_optimized_knot_0/sampled_guess

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_2_rotational_offset_quadratic_Z_optimized_knot_0 --shader test_finite_diff_quadratic_bezier_w_col_rotational_offset_quadratic_Z_optimized --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_2_rotational_offset_quadratic_Z_optimized_knot_0/sampled_guess.npy --gt_file knots_imgs/overhand/2c.png --gt_transposed --modes search_init --metrics 5_scale_L2 --render_size 480,480 --is_color --backend hl --ninit_samples 200 --ninit_best 5 --suffix _searched_guess --search_type 2 --shader_args nropes:1#all_nsplines:[2] --ignore_glsl

python bezier_util.py subdivide 1 2 5 1 /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_1_rotational_offset_quadratic_Z_optimized_knot_0/best_par_sampled_guess_1_ratio.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_2_rotational_offset_quadratic_Z_optimized_knot_0/subdivided_guess

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_2_rotational_offset_quadratic_Z_optimized_knot_0 --shader test_finite_diff_quadratic_bezier_w_col_rotational_offset_quadratic_Z_optimized --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_2_rotational_offset_quadratic_Z_optimized_knot_0/subdivided_guess.npy --gt_file knots_imgs/overhand/2c.png --gt_transposed --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 480,480 --is_color --backend hl --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 0.05 --no_reset_opt --no_binary_search_std --save_all_loss --quiet --suffix _sampled_guess_2_subdivide --save_best_par --multi_scale_optimization --ignore_last_n_scale 2 --base_loss_stage 2 --target_par_file /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_2_rotational_offset_quadratic_Z_optimized_knot_0/subdivided_guess.npy --target_weight_file /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_2_rotational_offset_quadratic_Z_optimized_knot_0/subdivided_guess_weight.npy --target_regularizer_scale 1e-10 --shader_args nropes:1#all_nsplines:[2] --ignore_glsl

# nsplines = 3

mkdir /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_3_rotational_offset_quadratic_Z_optimized_knot_0

python bezier_util.py expand 3 5 1 /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_2_rotational_offset_quadratic_Z_optimized_knot_0/best_par_sampled_guess_2_subdivide.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_3_rotational_offset_quadratic_Z_optimized_knot_0/sampled_guess

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_3_rotational_offset_quadratic_Z_optimized_knot_0 --shader test_finite_diff_quadratic_bezier_w_col_rotational_offset_quadratic_Z_optimized --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_3_rotational_offset_quadratic_Z_optimized_knot_0/sampled_guess.npy --gt_file knots_imgs/overhand/4c.png --gt_transposed --modes search_init --metrics 5_scale_L2 --render_size 480,480 --is_color --backend hl --ninit_samples 200 --ninit_best 5 --suffix _searched_guess --search_type 2 --shader_args nropes:1#all_nsplines:[3] --ignore_glsl

python bezier_util.py subdivide 2 3 5  1 /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_2_rotational_offset_quadratic_Z_optimized_knot_0/best_par_sampled_guess_2_subdivide.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_3_rotational_offset_quadratic_Z_optimized_knot_0/subdivided_guess

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_3_rotational_offset_quadratic_Z_optimized_knot_0 --shader test_finite_diff_quadratic_bezier_w_col_rotational_offset_quadratic_Z_optimized --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_3_rotational_offset_quadratic_Z_optimized_knot_0/subdivided_guess.npy --gt_file knots_imgs/overhand/4c.png --gt_transposed --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 480,480 --is_color --backend hl --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 0.05 --no_reset_opt --no_binary_search_std --save_all_loss --quiet --suffix _sampled_guess_4_ratio --save_best_par --multi_scale_optimization --ignore_last_n_scale 2 --base_loss_stage 2 --target_par_file /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_3_rotational_offset_quadratic_Z_optimized_knot_0/subdivided_guess.npy --target_weight_file /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_3_rotational_offset_quadratic_Z_optimized_knot_0/subdivided_guess_weight.npy --target_regularizer_scale 1e-10 --shader_args nropes:1#all_nsplines:[3] --ignore_glsl

# nsplines = 4

mkdir /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_rotational_offset_quadratic_Z_optimized_knot_0

python bezier_util.py expand 4 5 1 /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_3_rotational_offset_quadratic_Z_optimized_knot_0/best_par_sampled_guess_4_ratio.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_rotational_offset_quadratic_Z_optimized_knot_0/sampled_guess

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_rotational_offset_quadratic_Z_optimized_knot_0 --shader test_finite_diff_quadratic_bezier_w_col_rotational_offset_quadratic_Z_optimized --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_rotational_offset_quadratic_Z_optimized_knot_0/sampled_guess.npy --gt_file knots_imgs/overhand/6c.png --gt_transposed --modes search_init --metrics 5_scale_L2 --render_size 480,480 --is_color --backend hl --ninit_samples 200 --ninit_best 5 --suffix _searched_guess --search_type 2 --shader_args nropes:1#all_nsplines:[4] --ignore_glsl

python bezier_util.py subdivide 3 4 5 1 /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_3_rotational_offset_quadratic_Z_optimized_knot_0/best_par_sampled_guess_4_ratio.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_rotational_offset_quadratic_Z_optimized_knot_0/subdivided_guess

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_rotational_offset_quadratic_Z_optimized_knot_0 --shader test_finite_diff_quadratic_bezier_w_col_rotational_offset_quadratic_Z_optimized --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_rotational_offset_quadratic_Z_optimized_knot_0/subdivided_guess.npy --gt_file knots_imgs/overhand/6c.png --gt_transposed --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 480,480 --is_color --backend hl --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 0.05 --no_reset_opt --no_binary_search_std --save_all_loss --quiet --suffix _sampled_guess_6_ratio --save_best_par --multi_scale_optimization --ignore_last_n_scale 2 --base_loss_stage 2 --target_par_file /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_rotational_offset_quadratic_Z_optimized_knot_0/subdivided_guess.npy --target_weight_file /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_rotational_offset_quadratic_Z_optimized_knot_0/subdivided_guess_weight.npy --target_regularizer_scale 1e-10 --shader_args nropes:1#all_nsplines:[4] --ignore_glsl

# nsplines = 5

mkdir /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_5_rotational_offset_quadratic_Z_optimized_knot_0

python bezier_util.py expand 5 10 1 /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_rotational_offset_quadratic_Z_optimized_knot_0/best_par_sampled_guess_6_ratio.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_5_rotational_offset_quadratic_Z_optimized_knot_0/sampled_guess

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_5_rotational_offset_quadratic_Z_optimized_knot_0 --shader test_finite_diff_quadratic_bezier_w_col_rotational_offset_quadratic_Z_optimized --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_5_rotational_offset_quadratic_Z_optimized_knot_0/sampled_guess.npy --gt_file knots_imgs/overhand/7c.png --gt_transposed --modes search_init --metrics 5_scale_L2 --render_size 480,480 --is_color --backend hl --ninit_samples 200 --ninit_best 5 --suffix _searched_guess --search_type 2 --shader_args nropes:1#all_nsplines:[5] --ignore_glsl

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_5_rotational_offset_quadratic_Z_optimized_knot_0 --shader test_finite_diff_quadratic_bezier_w_col_rotational_offset_quadratic_Z_optimized --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_5_rotational_offset_quadratic_Z_optimized_knot_0/sampled_searched_guess_init_200_5.npy --gt_file knots_imgs/overhand/7c.png --gt_transposed --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 480,480 --is_color --backend hl --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 0.05 --no_reset_opt --no_binary_search_std --save_all_loss --quiet --suffix _sampled_guess_7_ratio --save_best_par --multi_scale_optimization --ignore_last_n_scale 2 --base_loss_stage 2 --target_par_file /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_5_rotational_offset_quadratic_Z_optimized_knot_0/sampled_guess.npy --target_weight_file /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_5_rotational_offset_quadratic_Z_optimized_knot_0/sampled_guess_weight.npy --target_regularizer_scale 1e-10 --shader_args nropes:1#all_nsplines:[5] --ignore_glsl

# nsplines = 6

mkdir /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_6_rotational_offset_quadratic_Z_optimized_knot_0

python bezier_util.py expand 6 1 1 /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_5_rotational_offset_quadratic_Z_optimized_knot_0/best_par_sampled_guess_7_ratio.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_6_rotational_offset_quadratic_Z_optimized_knot_0/sampled_guess

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_6_rotational_offset_quadratic_Z_optimized_knot_0 --shader test_finite_diff_quadratic_bezier_w_col_rotational_offset_quadratic_Z_optimized --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_6_rotational_offset_quadratic_Z_optimized_knot_0/sampled_guess.npy --gt_file knots_imgs/overhand/9c.png --gt_transposed --modes search_init --metrics 5_scale_L2 --render_size 480,480 --is_color --backend hl --ninit_samples 200 --ninit_best 5 --suffix _searched_guess --search_type 2 --shader_args nropes:1#all_nsplines:[6] --ignore_glsl

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_6_rotational_offset_quadratic_Z_optimized_knot_0 --shader test_finite_diff_quadratic_bezier_w_col_rotational_offset_quadratic_Z_optimized --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_6_rotational_offset_quadratic_Z_optimized_knot_0/sampled_searched_guess_init_200_5.npy --gt_file knots_imgs/overhand/9c.png --gt_transposed --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 480,480 --is_color --backend hl --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 0.05 --no_reset_opt --no_binary_search_std --save_all_loss --quiet --suffix _sampled_guess_9_ratio --save_best_par --multi_scale_optimization --ignore_last_n_scale 2 --base_loss_stage 2 --target_par_file /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_6_rotational_offset_quadratic_Z_optimized_knot_0/sampled_guess.npy --target_weight_file /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_6_rotational_offset_quadratic_Z_optimized_knot_0/sampled_guess_weight.npy --target_regularizer_scale 1e-10 --refine_opt --shader_args nropes:1#all_nsplines:[6] --ignore_glsl

python bezier_util.py reset_phase 6 5 1 /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_6_rotational_offset_quadratic_Z_optimized_knot_0/best_par_sampled_guess_9_ratio.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_6_rotational_offset_quadratic_Z_optimized_knot_0/reset_phase.npy

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_6_rotational_offset_quadratic_Z_optimized_knot_0 --shader test_finite_diff_quadratic_bezier_w_col_rotational_offset_quadratic_Z_optimized --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_6_rotational_offset_quadratic_Z_optimized_knot_0/reset_phase.npy --gt_file knots_imgs/overhand/9c.png --gt_transposed --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 480,480 --is_color --backend hl --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 2 --no_reset_opt --no_binary_search_std --save_all_loss --quiet --suffix _finalize_phase --save_best_par --multi_scale_optimization --ignore_last_n_scale 2 --base_loss_stage 2 --opt_subset_idx 23,24,25,26,27,28,29,30,31,32,33,34,35 --shader_args nropes:1#all_nsplines:[6] --ignore_glsl

# pass back phase

python bezier_util.py transfer expand 6 5 1 1,1,1,0,0 /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_6_rotational_offset_quadratic_Z_optimized_knot_0/best_par_finalize_phase.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_5_rotational_offset_quadratic_Z_optimized_knot_0/best_par_sampled_guess_7_ratio.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_6_rotational_offset_quadratic_Z_optimized_knot_0/best_par_from_5_auto.npy 

python bezier_util.py transfer expand 6 4 1 1,1,1,0,0 /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_6_rotational_offset_quadratic_Z_optimized_knot_0/best_par_finalize_phase.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_rotational_offset_quadratic_Z_optimized_knot_0/best_par_sampled_guess_6_ratio.npy  /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_6_rotational_offset_quadratic_Z_optimized_knot_0/best_par_from_4_auto.npy 

python bezier_util.py transfer expand 6 3 1 1,1,1,0,0 /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_6_rotational_offset_quadratic_Z_optimized_knot_0/best_par_finalize_phase.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_3_rotational_offset_quadratic_Z_optimized_knot_0/best_par_sampled_guess_4_ratio.npy  /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_6_rotational_offset_quadratic_Z_optimized_knot_0/best_par_from_3_auto.npy 

python bezier_util.py transfer expand 6 2 1 1,1,1,0,0 /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_6_rotational_offset_quadratic_Z_optimized_knot_0/best_par_finalize_phase.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_2_rotational_offset_quadratic_Z_optimized_knot_0/best_par_sampled_guess_2_subdivide.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_6_rotational_offset_quadratic_Z_optimized_knot_0/best_par_from_2_auto.npy 

python bezier_util.py transfer expand 6 1 1 1,1,1,0,0 /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_6_rotational_offset_quadratic_Z_optimized_knot_0/best_par_finalize_phase.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_1_rotational_offset_quadratic_Z_optimized_knot_0/best_par_sampled_guess_1_ratio.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_6_rotational_offset_quadratic_Z_optimized_knot_0/best_par_from_1_auto.npy

python bezier_util.py interp_multiple 6 /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_6_rotational_offset_quadratic_Z_optimized_knot_0/best_par_from_1_auto.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_6_rotational_offset_quadratic_Z_optimized_knot_0/best_par_from_2_auto.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_6_rotational_offset_quadratic_Z_optimized_knot_0/render_1_to_2_auto.npy

python bezier_util.py interp_multiple 6 /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_6_rotational_offset_quadratic_Z_optimized_knot_0/best_par_from_2_auto.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_6_rotational_offset_quadratic_Z_optimized_knot_0/best_par_from_3_auto.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_6_rotational_offset_quadratic_Z_optimized_knot_0/render_2_to_3_auto.npy

python bezier_util.py interp_multiple 6 /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_6_rotational_offset_quadratic_Z_optimized_knot_0/best_par_from_3_auto.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_6_rotational_offset_quadratic_Z_optimized_knot_0/best_par_from_4_auto.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_6_rotational_offset_quadratic_Z_optimized_knot_0/render_3_to_4_auto.npy

python bezier_util.py interp_multiple 6 /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_6_rotational_offset_quadratic_Z_optimized_knot_0/best_par_from_4_auto.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_6_rotational_offset_quadratic_Z_optimized_knot_0/best_par_from_5_auto.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_6_rotational_offset_quadratic_Z_optimized_knot_0/render_4_to_5_auto.npy

python bezier_util.py interp_multiple 6 /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_6_rotational_offset_quadratic_Z_optimized_knot_0/best_par_from_5_auto.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_6_rotational_offset_quadratic_Z_optimized_knot_0/best_par_finalize_phase.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_6_rotational_offset_quadratic_Z_optimized_knot_0/render_5_to_6_auto.npy

python bezier_util.py combine /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_6_rotational_offset_quadratic_Z_optimized_knot_0/render_1_to_2_auto.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_6_rotational_offset_quadratic_Z_optimized_knot_0/render_2_to_3_auto.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_6_rotational_offset_quadratic_Z_optimized_knot_0/render_3_to_4_auto.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_6_rotational_offset_quadratic_Z_optimized_knot_0/render_4_to_5_auto.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_6_rotational_offset_quadratic_Z_optimized_knot_0/render_5_to_6_auto.npy /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_6_rotational_offset_quadratic_Z_optimized_knot_0/render_all.npy

# somehow Halide is not happy with the complexity

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_rotational_offset_better_rendering_quadratic_Z_knot_0 --shader test_finite_diff_quadratic_bezier_w_col_rotational_offset_better_rendering_quadratic_Z --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_6_rotational_offset_quadratic_Z_optimized_knot_0/render_all.npy --gt_file knots_imgs/overhand/7c.png --gt_transposed --modes render --metrics 3_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_color --backend tf --unnormalized_par --shader_args nropes:1#all_nsplines:[6] --ignore_glsl --camera_size 1600,1600 --render_size 960,480 --tile_offset 280,640

ffmpeg -i /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_rotational_offset_better_rendering_quadratic_Z_knot_0/init%05d.png /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_rotational_offset_better_rendering_quadratic_Z_knot_0/animation.mp4

"""

from render_util import *
from render_single import render_single

compiler.log_prefix_only = False
compiler.log_intermediates_less = True

nropes = 1
all_nsplines = [6]

def update_args():

    assert len(all_nsplines) == nropes

    global total_n_pos 
    global total_nrotations 
    global total_ntilts 
    
    total_n_pos = 0
    total_nrotations = 0
    total_ntilts = 0

    global all_nlengths 
    global all_nrotations 
    global all_ntilts 
    
    global nlengths
    global nrotations
    global ntilts
    
    all_nlengths = []
    all_nrotations = []
    all_ntilts = []

    for n in range(nropes):
        nsplines = all_nsplines[n]

        nrotations = nsplines + 1
        nlengths = 2 * nsplines
        ntilts = 2 * nsplines + 1

        total_n_pos += nlengths + 2
        total_nrotations += nrotations
        total_ntilts += ntilts

        all_nlengths.append(nlengths)
        all_nrotations.append(nrotations)
        all_ntilts.append(ntilts)

    global nargs
    nargs = 2 + total_n_pos + total_nrotations + total_ntilts
    
    global args_range 
    args_range = np.ones(nargs)

    args_range[0] = 1
    args_range[1] = 0.5
    args_range[2 : 2 + total_n_pos] = 20
    args_range[2 + total_n_pos : 2 + total_n_pos + total_nrotations] = np.pi / 2
    args_range[2 + total_n_pos + total_nrotations :] = 1

    global sigmas_range 
    sigmas_range = args_range.copy()
    sigmas_range[2 + total_n_pos : 2 + total_n_pos + total_nrotations] = np.pi / 10
    
update_args()

width = ArgumentScalar('width')
height = ArgumentScalar('height')

default_phase = -1e4

# color for overhand knot
#fill_cols = [Compound([0.93333333, 0.57254902, 0.3372549 ])]

# color for thief knot
fill_cols = [Compound([0.9215686274509803, 0.5686274509803921, 0.3333333333333333]),
             Compound([0.796078431372549, 0.7450980392156863, 0.5372549019607843])]

assert len(fill_cols) >= nropes

def det(a, b):
    return a[0] * b[1] - a[1] * b[0]

def get_distance(pos, spline, last_det=None):
    
    b0 = spline.A - pos
    b1 = spline.B - pos
    b2 = spline.C - pos
    
    a = det(b0, b2)
        
    if last_det is not None:
        # last_det = 2 * det(last_b2, last_b1)
        b = last_det
    else:
        b = 2 * det(b1, b0)
    
    d = 2 * det(b2, b1)
    
    f = b * d - a * a
    
    d21 = spline.C - spline.B
    d10 = spline.B - spline.A
    d20 = spline.C - spline.A
    
    gf = 2 * (b * d21 + d * d10 + a * d20)
    
    gf = np.array([gf[1], -gf[0]])
    
    pp = -f * gf / dot(gf,gf)
    
    d0p = b0 - pp
    
    ap = det(d0p, d20)
    bp = 2 * det(d10, d0p)
    
    t_raw = (ap + bp) / (2 * a + b + d)
    
    t = maximum(minimum(t_raw, 1), 0)

    dist = length(mix(mix(b0, b1, t), mix(b1, b2, t), t), 2)
    
    phase = mix(mix(spline.tilt_0, spline.tilt_1, t), mix(spline.tilt_1, spline.tilt_2, t), t)
        
    phase = phase - 10000 * ((maximum(t_raw, 1) - 1) ** 2 + minimum(t_raw, 0) ** 2)
    
    return dist, phase, d

def test_finite_diff_quadratic_bezier_w_col_rotational_offset_quadratic_Z_optimized(u, v, X, scalar_loss_scale):
    
    # make sure it's non-negative
    curve_width = X[0] ** 2
    curve_edge = X[1] ** 2
    
    edge_col = Compound([0., 0., 0.])
    
    old_col = Compound([1, 1, 1])
    
    pos_base = 0
    rotation_base = 0
    tilt_base = 0
    
    dists = []
    phases = []
    
    for n in range(nropes):
        
        nsplines = all_nsplines[n]
        
        current_rotation = 0
        old_dist = None
        old_phase = default_phase

        last_spline = None
        last_det = None
        last_BC = None

        for idx in range(nsplines):
            if idx == 0:

                current_A = np.array([X[2 + pos_base], X[2 + pos_base + 1]])

                current_rotation = X[2 + total_n_pos + rotation_base]
                current_AB = X[2 + pos_base + 2] ** 2
                current_B = current_A + current_AB * np.array([cos(current_rotation), sin(current_rotation)])

                current_rotation = current_rotation + X[2 + total_n_pos + rotation_base + 1]
                current_BC = X[2 + pos_base + 3] ** 2
                current_C = current_B + current_BC * np.array([cos(current_rotation), sin(current_rotation)])

                spline = Object('spline',
                                A = current_A,
                                B = current_B,
                                C = current_C,
                                tilt_0 = X[2 + total_n_pos + total_nrotations + tilt_base],
                                tilt_1 = X[2 + total_n_pos + total_nrotations + tilt_base + 1],
                                tilt_2 = X[2 + total_n_pos + total_nrotations + tilt_base + 2])
            else:
                assert last_spline is not None

                current_A = last_spline.C

                current_AB = X[2 + pos_base + 2 + idx * 2] ** 2
                current_B = current_A + current_AB * np.array([cos(current_rotation), sin(current_rotation)])

                current_rotation = current_rotation + X[2 + total_n_pos + rotation_base + idx + 1]
                current_BC = X[2 + pos_base + 2 + idx * 2 + 1] ** 2
                current_C = current_B + current_BC * np.array([cos(current_rotation), sin(current_rotation)]) 

                spline = Object('spline',
                                A = current_A,
                                B = current_B,
                                C = current_C,
                                tilt_0 = last_spline.tilt_2,
                                tilt_1 = X[2 + total_n_pos + total_nrotations + tilt_base + 2 * idx + 1],
                                tilt_2 = X[2 + total_n_pos + total_nrotations + tilt_base + 2 * idx + 2])

            if last_det is not None:
                last_det = last_det * current_AB / last_BC

            dist, phase, last_det = get_distance(np.array([u, v]), spline, last_det)

            dist = Var('dist_%s' % spline.name, dist)
            phase = Var('phase_raw_%s' % spline.name, phase)

            cond0_diff = Var('cond0_diff_%s' % spline.name, dist - curve_width / 2)
            cond2_diff = Var('cond2_diff_%s' % spline.name, phase - old_phase)

            cond0 = Var('cond0_%s' % spline.name, cond0_diff < 0)
            cond2 = Var('cond2_%s' % spline.name, cond2_diff > 0)

            cond_valid = Var('cond_valid_%s' % spline.name, cond0 & cond2)

            out_phase = Var('phase_%s' % spline.name, select(cond_valid, phase, old_phase))
            if old_dist is None:
                old_dist = dist
            else:
                old_dist = select(cond_valid, dist, old_dist)
            old_phase = out_phase

            last_spline = spline
            last_BC = current_BC
            
        pos_base += 2 + all_nlengths[n]
        rotation_base += all_nrotations[n]
        tilt_base += all_ntilts[n]
        
        dists.append(old_dist)
        phases.append(old_phase)
        
    for n in range(nropes):
        if n == 0:
            col = select(dists[n] - curve_width / 2 < 0,
                         select(dists[n] - curve_width / 2 + curve_edge > 0, edge_col, fill_cols[n]),
                         old_col)
            phase = phases[0]
        else:
            # no neeck to check whether dist is close enough to curve
            # because when dist far away from curve, phase is still kept as default value
            
            col = select((dists[n] - curve_width / 2 < 0) & (phases[n] > phase),
                         select(dists[n] - curve_width / 2 + curve_edge > 0, edge_col, fill_cols[n]),
                         col)
            
            phase = maximum(phases[n], phase)
        
    return col
    

shaders = [test_finite_diff_quadratic_bezier_w_col_rotational_offset_quadratic_Z_optimized]
is_color = True

class SampleInit:
    def __init__(self, search_type=None):
        if search_type is None:
            self.search_type = 1
        else:
            self.search_type = search_type

        import bezier_util
        self.bezier_util = bezier_util
    
    def sample(self, par, width=480, height=480, idx=0):
        
        if self.search_type == 1:
            # search to shrink last n segments
            if idx == 0:
                return par, ''
            elif idx < 0:
                raise
            else:
                last_shrink_idx = nsplines - idx
                new_par, extra_args = self.bezier_util.shrink_par(par, nsplines, last_shrink_idx)
                extra_args = '--opt_subset_idx ' + ','.join([str(val) for val in extra_args])
                return new_par, extra_args
        elif self.search_type == 2:
            # search to extend the last segment
            new_par = par.copy()
            
            total_n_pos = 0
            total_nrotations = 0
            
            for n in range(nropes - 1):
                total_n_pos += 2 + 2 * all_nsplines[n]
                total_nrotations += all_nsplines[n] + 1
            
            new_par[2 + total_n_pos + 2 + (nlengths - 2) : 2 + total_n_pos + 2 + nlengths] = [5 ** 0.5, 10 ** 0.5]
            #new_par[2 + total_n_pos + 2 + (nlengths - 2) : 2 + total_n_pos + 2 + nlengths] = np.random.rand(2) * 5
            
            # this makes overhand 2 successfully subdivide
            new_par[2 + total_n_pos + 2 + nlengths + total_nrotations + nrotations - 1] = \
            np.random.rand() * 0.7 * np.pi + 0.05 * np.pi
            
            #new_par[2 + total_n_pos + 2 + nlengths + total_nrotations + nrotations - 1] = \
            #np.random.rand() * 0.9 * np.pi + 0.05 * np.pi
            
            if np.random.rand() < 0.5:
                new_par[2 + total_n_pos + 2 + nlengths + total_nrotations + nrotations - 1] = \
                2 * np.pi - new_par[2 + total_n_pos + 2 + nlengths + total_nrotations + nrotations - 1]
                
            new_par[-ntilts:-2] = 0
            new_par[-2:] = np.random.rand(2) - 1
            
            new_par[-ntilts:] = 0
                        
            return new_par, ''
        
        
        
        