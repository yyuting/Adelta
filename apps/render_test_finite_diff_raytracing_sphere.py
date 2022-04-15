"""
------------------------------------------------------------------------------------------------------------------------------
# command for visualization

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_sphere --shader test_finite_diff_raytracing_sphere --init_values_pool apps/example_init_values/test_finite_diff_raymarching_sphere_init_values_pool.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_col --render_size 640,640

------------------------------------------------------------------------------------------------------------------------------
# optimization with random var

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_sphere --shader test_finite_diff_raytracing_sphere --init_values_pool apps/example_init_values/test_finite_diff_raymarching_sphere_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --save_all_par --suffix _random

# animate optimization

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_sphere --shader test_finite_diff_raytracing_sphere --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_raytracing_sphere/ours_both_sides_5_scale_L2_adam_1.0e-02_random_result1_0.npy --modes render --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --suffix _animation --save_all_par --no_reset_opt --backend hl --quiet --multi_scale_optimization --alternating_times 5 --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --save_all_par --suffix _random

ffmpeg -i /n/fs/scratch/yutingy/test_finite_diff_raytracing_sphere/init_random%05d.png -r 30 -c:v libx264 -preset slow -crf 0 -r 30 /n/fs/scratch/yutingy/test_finite_diff_raytracing_sphere/animation.mp4
"""

from render_util import *
from render_single import render_single

compiler.log_prefix_only = False
compiler.log_intermediates_less = True

def cmd_template():
    cmd = f"""python approx_gradient.py --shader test_finite_diff_raytracing_sphere --init_values_pool apps/example_init_values/test_finite_diff_raymarching_sphere_init_values_pool.npy --metrics 5_scale_L2 --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5"""
    
    return cmd


nargs = 10

args_range = np.ones(nargs)
sigmas_range = args_range

width = ArgumentScalar('width')
height = ArgumentScalar('height')

def test_finite_diff_raytracing_sphere(u, v, X, scalar_loss_scale):
    
    origin_x = X[0]
    origin_y = X[1]
    origin_z = X[2]
    
    ro = np.array([origin_x, origin_y, origin_z])
    
    ang1 = X[3]
    ang2 = X[4]
    ang3 = X[5]
    
    pos_x = X[6]
    pos_y = X[7]
    pos_z = X[8]
    radius = X[9]
    
    ray_dir = [u - width / 2, v - height / 2, 1.73 * width / 2]
    rd_norm2 = Var('rd_norm2', ray_dir[0] ** 2 + ray_dir[1] ** 2 + ray_dir[2] ** 2)
    ray_dir_norm = Var('rd_norm',  rd_norm2 ** 0.5)
    
    ray_dir = np.array([Var('raw_rd0', ray_dir[0] / ray_dir_norm),
                        Var('raw_rd1', ray_dir[1] / ray_dir_norm),
                        Var('raw_rd2', ray_dir[2] / ray_dir_norm)])

    sin1 = Var('sin1', sin(ang1))
    cos1 = Var('cos1', cos(ang1))
    sin2 = Var('sin2', sin(ang2))
    cos2 = Var('cos2', cos(ang2))
    sin3 = Var('sin3', sin(ang3))
    cos3 = Var('cos3', cos(ang3))
    
    ray_dir_p = [cos2 * cos3 * ray_dir[0] + (-cos1 * sin3 + sin1 * sin2 * cos3) * ray_dir[1] + (sin1 * sin3 + cos1 * sin2 * cos3) * ray_dir[2],
                 cos2 * sin3 * ray_dir[0] + (cos1 * cos3 + sin1 * sin2 * sin3) * ray_dir[1] + (-sin1 * cos3 + cos1 * sin2 * sin3) * ray_dir[2],
                 -sin2 * ray_dir[0] + sin1 * cos2 * ray_dir[1] + cos1 * cos2 * ray_dir[2]]
    
    rd = np.array([Var('rd0', ray_dir_p[0]),
                   Var('rd1', ray_dir_p[1]),
                   Var('rd2', ray_dir_p[2])])
    
    sphere_o = np.array([pos_x, pos_y, pos_z]) - ro
    
    dist_A = dot(rd, rd)
    dist_B = 2. * dot(sphere_o, rd)
    dist_C = dot(sphere_o, sphere_o) - radius ** 2
    
    dist_discrim = dist_B * dist_B - 4. * dist_A * dist_C
    dist_valid = dist_discrim >= 0
    
    col = select(dist_valid, 1, 0)
    
    return output_color([col, col, col])
    
shaders = [test_finite_diff_raytracing_sphere]
is_color = True