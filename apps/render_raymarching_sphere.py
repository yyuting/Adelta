"""
python approx_gradient.py --dir /n/fs/scratch/yutingy/raymarching_sphere --shader raymarching_sphere --init_values_pool apps/example_init_values/test_finite_diff_raymarching_sphere_init_values_pool.npy --modes visualize_gradient
"""

from render_util import *
from render_single import render_single

def cmd_template():
    cmd = f"""python approx_gradient.py --shader raymarching_sphere --init_values_pool apps/example_init_values/test_finite_diff_raymarching_sphere_init_values_pool.npy --metrics 5_scale_L2 --is_col"""
    
    return cmd

nargs = 13
args_range = np.array([10, 10, 10, 6.28, 6.28, 6.28, 5, 5, 5, 2, 1, 1, 1])

sigmas_range = np.ones(10)

width=960
height=640

raymarching_loop = 32

compiler.log_prefix_only = False
compiler.log_intermediates_less = True

def raymarching_sphere(u, v, X, scalar_loss=None):
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
    
    sphere_px = X[6]
    sphere_py = X[7]
    sphere_pz = X[8]
    
    sphere_r = X[9]
    
    col = np.array([X[10], X[11], X[12]])
    
    def sdSphere(x, y, z):
        
        pos = np.array([x, y, z])
        
        sphere_diff = pos - np.array([sphere_px, sphere_py, sphere_pz])
        
        dist = length(sphere_diff, 2) - sphere_r
        
        label = 1
        
        return dist, label
    
    ro = np.array([origin_x, origin_y, origin_z])
    
    
    ray_dir = [u - width / 2, v - height / 2, 1.73 * width / 2]
    ray_dir_norm = (ray_dir[0] ** 2 + ray_dir[1] ** 2 + ray_dir[2] ** 2) ** 0.5
    #ray_dir = np.array(ray_dir) / ray_dir_norm
    
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
    
    raymarching_ans = RaymarchingWrapper(sdSphere, ro, rd, 0, raymarching_loop, include_derivs=False)
    cond_converge = raymarching_ans.is_converge
    
    col = select(cond_converge, col, np.zeros(3))
    return col
        
shaders = [raymarching_sphere]
