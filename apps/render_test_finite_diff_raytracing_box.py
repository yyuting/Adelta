"""
------------------------------------------------------------------------------------------------------------------------------
# command for visualization

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_box --shader test_finite_diff_raytracing_box --init_values_pool apps/example_init_values/test_finite_diff_raymarching_box_init_values_pool.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_col --render_size 640,640

------------------------------------------------------------------------------------------------------------------------------
# optimization with random var

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_box --shader test_finite_diff_raytracing_box --init_values_pool apps/example_init_values/test_finite_diff_raymarching_box_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 0.5 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --save_all_par --suffix _random

# profile using FD

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raytracing_box --shader test_finite_diff_raytracing_box --init_values_pool apps/example_init_values/test_finite_diff_raymarching_box_init_values_pool.npy --modes optimization --metrics 5_scale_L2 --gradient_methods_optimization finite_diff --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 640,640 --is_color --smoothing_sigmas 0.5,1,2,5 --multi_scale_optimization --alternating_times 5 --backend hl --quiet --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 0.5 --quiet --no_reset_opt --no_binary_search_std --save_all_loss --save_all_par --suffix _random

"""

from render_util import *
from render_single import render_single

compiler.log_prefix_only = False
compiler.log_intermediates_less = True

nargs = 20

args_range = np.ones(nargs)
sigmas_range = args_range

width = ArgumentScalar('width')
height = ArgumentScalar('height')

def test_finite_diff_raytracing_box(u, v, X, scalar_loss_scale):
    
    origin_x = X[0]
    origin_y = X[1]
    origin_z = X[2]
    
    ro = np.array([origin_x, origin_y, origin_z])
    
    ang1 = X[3]
    ang2 = X[4]
    ang3 = X[5]
    
    amb = np.array([X[6], X[7], X[8]])
    kd = np.array([X[9], X[10], X[11]])
    
    lig_ang0 = X[12]
    lig_ang1 = X[13]
    
    pos_x = X[14]
    pos_y = X[15]
    pos_z = X[16]
    
    dim_x = X[17]
    dim_y = X[18]
    dim_z = X[19]
    
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
    
    box_o = np.array([pos_x, pos_y, pos_z])
    box_dim = np.array([dim_x, dim_y, dim_z])
    
    offset = box_o - ro
        
    t0 = vec('t0', (offset + box_dim) / rd)
    t1 = vec('t1', (offset - box_dim) / rd)
    
    tx_min = minimum(t0[0], t1[0])
    tx_max = maximum(t0[0], t1[0])
    
    ty_min = minimum(t0[1], t1[1])
    ty_max = maximum(t0[1], t1[1])
    
    tz_min = minimum(t0[2], t1[2])
    tz_max = maximum(t0[2], t1[2])
    
    max_txy = maximum(tx_min, ty_min)
    
    tmin = Var('tmin', maximum(max_txy, tz_min))
    tmax = Var('tmax', minimum(minimum(tx_max, ty_max), tz_max))
    
    is_valid = (tmax > 0) & (tmin <= tmax)
    
    t = tmin
    
    lig = np.array([cos(lig_ang0),
                    sin(lig_ang0) * cos(lig_ang1),
                    sin(lig_ang0) * sin(lig_ang1)])
    
    deriv_sdf = [select(max_txy >= tz_min, select(tx_min >= ty_min, sign(offset[0]), 0.), 0.),
                 select(max_txy >= tz_min, select(tx_min >= ty_min, 0., sign(offset[1])), 0.),
                 select(max_txy >= tz_min, 0., sign(offset[2]))]
    
    dif = dot(lig, deriv_sdf)
    
    col = select(is_valid, amb + dif * kd, 1.)
    
    return col

shaders = [test_finite_diff_raytracing_box]
is_color = True
                   
              