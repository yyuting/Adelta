from render_util import *

nargs = 43

width = ArgumentScalar('width')
height = ArgumentScalar('height')

raymarching_loop = 64

label_x = 0.0
label_y = 1.0

def siggraph(u, v, X, scalar_loss_scale):
    
    # Access input parameters
    
    # camera position and angles
    origin_x, origin_y, origin_z = vec('origin', [X[0], X[1], X[2]])
    ang1, ang2, ang3 = vec('ang', [X[3], X[4], X[5]])
    
    # direction for the halfspace
    ax_theta, ax_phi = vec('ax_ang', [X[6], X[7]])
    
    # direction and shape of the cone
    cone_theta, cone_phi, cone_alpha, cone_ang = vec('cone_ang', [X[8], X[9], X[10], X[12]])
    ellipse_ratio = Var('ellipse_ratio', X[11])
    
    # threshold for sphere and halfspace
    d1_thre, d2_thre = vec('d_thre', [X[13], X[14]])
    
    # directional light angles for both halves
    angs_lig0_x = vec('angs_lig0_x', [X[15], X[16]])
    angs_lig0_y = vec('angs_lig0_y', [X[17], X[18]])
    
    # point light location for both halves
    pos_lig1_x = vec('pos_lig1_x', [X[19], X[20], X[21]])
    pos_lig1_y = vec('pos_lig1_y', [X[22], X[23], X[24]])
    
    # ambient and diffuse coefficients for both halves.
    amb_x = vec('amb_x', [X[25], X[26], X[27]])
    amb_y = vec('amb_y', [X[28], X[29], X[30]])
    
    kd0_x = vec('kd0_x', [X[31], X[32], X[33]])
    kd0_y = vec('kd0_y', [X[34], X[35], X[36]])
    
    kd1_x = vec('kd1_x', [X[37], X[38], X[39]])
    kd1_y = vec('kd1_y', [X[40], X[41], X[42]])
    
    # computing sin and cos of the angles
    
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
    
    # computing halfspace and cone directons
    
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
    
    # camera setup: using 
    
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
    
    # SDF representing the red or blue half of the geometry
    def single_obj(pos, scale, tag):
        
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
        
        return res_x, deriv_t_shell_pos
    
    # SDF representing the entire geometry
    def combined_obj(pos):
        
        tag_x = 'x'
        
        resx, derivx = single_obj(pos, np.ones(3), tag_x)
        
        tag_y = 'y'
        
        resy, derivy = single_obj(pos, np.array([-1, -1, 1]), tag_y)
        
        cond_xy = Var('cond_xy', resy - resx > 0)
        
        res0 = minimum(resx, resy)
        res1 = Var('combined_res1', select(cond_xy, label_x, label_y))
        
        deriv0 = select(cond_xy, derivx[0], -derivy[0])
        deriv1 = select(cond_xy, derivx[1], -derivy[1])
        deriv2 = select(cond_xy, derivx[2], derivy[2])
        
        return res0, res1, deriv0, deriv1, deriv2
    
    # calling raymarching primitive
    t = 0
    tmax = 10
    t_closest = 0
    res0_closest = 10
    
    def raymarching_body(x, y, z):
        return combined_obj([x, y, z])
    
    raymarching_ans = RaymarchingWrapper(raymarching_body, ro, rd, 0, raymarching_loop, include_derivs=True)
    
    # access raymarching outputs
    
    t_closest = raymarching_ans.t
    res1 = raymarching_ans.label
    deriv_sdf = raymarching_ans.derivs
    cond_converge = raymarching_ans.is_converge
    
    # shading the geometry
    
    pos = vec('pos', ro + rd * t_closest)
    
    obj_label = res1    
    
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

shaders = [siggraph]
is_color = True

args_range = np.array([0.5] * 8 + [0.1] * 4 + [0.02] + [0.1] * 2 + [1] * 28)
