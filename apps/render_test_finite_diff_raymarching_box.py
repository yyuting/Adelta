"""
# command for visualization
python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_raymarching_box --shader test_finite_diff_raymarching_box --init_values_pool apps/example_init_values/test_finite_diff_raymarching_half_cube_init_values_pool.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_col
"""

from render_util import *
from render_single import render_single

def cmd_template():
    
    cmd = f"""python approx_gradient.py --shader test_finite_diff_raymarching_box --init_values_pool apps/example_init_values/test_finite_diff_raymarching_half_cube_init_values_pool.npy --metrics 5_scale_L2 --is_col"""
    
    return cmd

nargs = 9
args_range = np.array([10, 10, 10, 6.28, 6.28, 6.28, 1, 1, 1])


width=960
height=640

raymarching_loop = 32

compiler.log_prefix_only = False
compiler.log_intermediates_less = True

check_intersect_rule = 1
check_intersect_thre = 0.01

use_select_rule = 1

def sdBox_deriv(pos, cube_dims, tag=None):
    f0 = abs(pos[0]) - cube_dims[0]
    f1 = abs(pos[1]) - cube_dims[1]
    f2 = abs(pos[2]) - cube_dims[2]
    
    q0 = maximum(f0, 0)
    q1 = maximum(f1, 0)
    q2 = maximum(f2, 0)
    
    max_f1f2 = maximum(f1, f2)
    
    dist = (q0 ** 2 + q1 ** 2 + q2 ** 2) ** 0.5 + minimum(maximum(f0, max_f1f2), 0)
    
    choose_f0 = f0 > max_f1f2
    choose_f1 = f1 > f2
    
    deriv = [select(choose_f0, sign(pos[0]), 0.),
             select(choose_f0, 0., select(choose_f1, sign(pos[1]), 0.)),
             select(choose_f0, 0., select(choose_f1, 0., sign(pos[2])))]
    
    return dist, deriv

def test_finite_diff_raymarching_box(u, v, X, scalar_loss=None):
    """
    X has shape nargs + 3
    first 3 entries are u, v, time
    the other entries are scalar tunable parameters
    """

    origin_x = X[0]
    origin_y = X[1]
    origin_z = X[2]
    
    ang1 = X[3]
    ang2 = X[4]
    ang3 = X[5]
    
    dimx = X[6]
    dimy = X[7]
    dimz = X[8]
    
    def deriv_map_obj(pos):

        cube_diff = pos
        
        res, deriv = sdBox_deriv(cube_diff, [dimx, dimy, dimz])
        
        return res, 1, deriv[0], deriv[1], deriv[2]
    
    ro = np.array([origin_x, origin_y, origin_z])
    
    
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
    
    def raymarching_body(x, y, z):
        return deriv_map_obj([x, y, z])
    
    raymarching_ans = RaymarchingWrapper(raymarching_body, ro, rd, 0, raymarching_loop, include_derivs=True)
    
    cond_converge = raymarching_ans[0]
    t_closest = raymarching_ans[1]
    
    deriv_sdf = [raymarching_ans[6],
                 raymarching_ans[7],
                 raymarching_ans[8]]
    
    lig = np.array([1., 2., 3.])
    lig /= np.linalg.norm(lig)
    
    dif = dot(lig, deriv_sdf)
    amb = np.array([0.2, 0.1, 0.1])
    kd = np.array([0.8, 0.3, 0.3])
    
    #col = select(cond_converge, amb + dif * kd, 1.)
    col = select(cond_converge, 0., 1.)

    return output_color([col, col, col])
        
shaders = [test_finite_diff_raymarching_box]
is_color = True