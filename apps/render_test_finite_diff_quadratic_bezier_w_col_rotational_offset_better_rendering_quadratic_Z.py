"""
------------------------------------------------------------------------------------------------------------------------------
# render

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_rotational_offset_better_rendering_quadratic_Z --shader test_finite_diff_quadratic_bezier_w_col_rotational_offset_better_rendering_quadratic_Z --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_6_rotational_offset_quadratic_Z/render_all.npy --gt_file knots_imgs/overhand/7c.png --gt_transposed --modes render --metrics 3_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 480,480 --is_color --backend hl

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_rotational_offset_better_rendering_quadratic_Z_no_control --shader test_finite_diff_quadratic_bezier_w_col_rotational_offset_better_rendering_quadratic_Z --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_6_rotational_offset_quadratic_Z/render_all.npy --gt_file knots_imgs/overhand/7c.png --gt_transposed --modes render --metrics 3_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 480,480 --is_color --backend hl

# somehow Halide is not happy with the complexity

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_rotational_offset_better_rendering_quadratic_Z_knot_0 --shader test_finite_diff_quadratic_bezier_w_col_rotational_offset_better_rendering_quadratic_Z --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_6_rotational_offset_quadratic_Z_optimized_knot_0/render_all.npy --gt_file knots_imgs/overhand/7c.png --gt_transposed --modes render --metrics 3_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_color --backend tf --unnormalized_par --shader_args nropes:1#all_nsplines:[6] --ignore_glsl --camera_size 1600,1600 --render_size 960,480 --tile_offset 280,640

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_rotational_offset_better_rendering_quadratic_Z_knot_0 --shader test_finite_diff_quadratic_bezier_w_col_rotational_offset_better_rendering_quadratic_Z --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_6_rotational_offset_quadratic_Z_optimized_knot_0/render_all.npy --gt_file knots_imgs/overhand/7c.png --gt_transposed --modes render --metrics 3_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_color --backend tf --unnormalized_par --shader_args nropes:1#all_nsplines:[6] --ignore_glsl --render_size 480,480 --suffix _small

cd /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_rotational_offset_better_rendering_quadratic_Z_knot_0; ffmpeg -i init%05d.png -r 30 -c:v libx264 -preset slow -crf 0 -r 30 animation.mp4; cd /n/fs/shaderml/differentiable_compiler

# quicktime compatible
ffmpeg -i init%05d.png -r 30 -c:v libx264 -pix_fmt yuv420p -filter:v "setpts=0.5*PTS" -r 30 animation.mp4


python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_rotational_offset_better_rendering_quadratic_Z_knot_1 --shader test_finite_diff_quadratic_bezier_w_col_rotational_offset_better_rendering_quadratic_Z --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1/render_all.npy --gt_file knots_imgs/thief/8.png --gt_transposed --modes render --metrics 3_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_color --unnormalized_par --shader_args nropes:2#all_nsplines:[4,4] --backend tf --camera_size 1500,1500 --render_size 960,480 --tile_offset 232,510

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_rotational_offset_better_rendering_quadratic_Z_knot_1 --shader test_finite_diff_quadratic_bezier_w_col_rotational_offset_better_rendering_quadratic_Z --init_values_pool /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_4_4_rotational_offset_quadratic_Z_optimized_knot_1/render_all.npy --gt_file knots_imgs/thief/8.png --gt_transposed --modes render --metrics 3_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_color --unnormalized_par --shader_args nropes:2#all_nsplines:[4,4] --backend tf --render_size 480,480 --suffix _small

cd /n/fs/scratch/yutingy/test_finite_diff_quadratic_bezier_w_col_rotational_offset_better_rendering_quadratic_Z_knot_1; ffmpeg -i init%05d.png init%05d.png -r 30 -c:v libx264 -preset slow -crf 0 -r 30 animation.mp4; cd /n/fs/shaderml/differentiable_compiler
"""

from render_util import *
from render_single import render_single

compiler.log_prefix_only = False
compiler.log_intermediates_less = True

nropes = 2
all_nsplines = [4, 4]

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
    global nsplines
    
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
    
    global fill_cols
    
    if nropes == 2:
        # color for thief knot
        fill_cols = [np.array([0.9215686274509803, 0.5686274509803921, 0.3333333333333333]),
                     np.array([0.796078431372549, 0.7450980392156863, 0.5372549019607843])]
    else:
        # color for overhand knot
        fill_cols = [np.array([0.93333333, 0.57254902, 0.3372549 ])]
    
update_args()

width = ArgumentScalar('width')
height = ArgumentScalar('height')

default_phase = -1e4
default_dist = 10





def det(a, b):
    return a[0] * b[1] - a[1] * b[0]

def get_distance_line(pos, spline, idx):
    
    A = spline.A
    B = 0.25 * spline.A + 0.5 * spline.B + 0.25 * spline.C
    C = spline.C
    
    b0 = pos - A
    b1 = B - A
    
    t0_raw = safe_division(dot(b0, b1), dot(b1, b1), -1.)
    t0 = maximum(minimum(t0_raw, 1), 0)
    
    dist0 = length(mix(A, B, t0) - pos, 2)
    
    b2 = pos - B
    b3 = C - B
    
    t1_raw = safe_division(dot(b2, b3), dot(b3, b3), -1.)
    t1 = maximum(minimum(t1_raw, 1), 0)
    
    dist1 = length(mix(B, C, t1) - pos, 2)
    
    t_raw = select(dist0 < dist1, t0_raw / 2., t1_raw / 2. + 0.5)
    t = maximum(minimum(t_raw, 1), 0)
    dist = minimum(dist0, dist1)
    
    phase = mix(mix(spline.tilt_0, spline.tilt_1, t), mix(spline.tilt_1, spline.tilt_2, t), t)
    
    if False:
        # Should check if there's any artifact
        # the idea is, becasue we enforce tangent continuity between segments, there's no need to add edge to t=0 and t=1 at all
        if idx != 0:
            #phase = phase - 10000 * minimum(t_raw, 0) ** 2
            phase = select(t_raw < 0., default_phase / 2., phase)
        if idx != nsplines - 1:
            #phase = phase - 10000 * (maximum(t_raw, 1) - 1) ** 2
            phase = select(t_raw > 1., default_phase / 2., phase)
        
    return dist, phase, t_raw

def get_distance_spline(pos, spline, idx, last_det=None):
    
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
    
    if False:
        # Should check if there's any artifact
        # the idea is, becasue we enforce tangent continuity between segments, there's no need to add edge to t=0 and t=1 at all
        if idx != 0:
            phase = phase - 10000 * minimum(t_raw, 0) ** 2
            #phase = select(t_raw < 0., default_phase / 2., phase)
        if idx != nsplines - 1:
            phase = phase - 10000 * (maximum(t_raw, 1) - 1) ** 2
            #phase = select(t_raw > 1., default_phase / 2., phase)
        
    return dist, phase, d, t_raw

def test_finite_diff_quadratic_bezier_w_col_rotational_offset_better_rendering_quadratic_Z(u, v, X, scalar_loss_scale):
    
    # make sure it's non-negative
    curve_width = X[0] ** 2
    curve_edge = X[1] ** 2
    
    edge_col = np.array([0., 0., 0.])
    
    antialias_thre = 1
    
    bg_col = np.array([1., 1., 1.])
    
    current_rotation = 0
    
    is_control = ConstExpr(False)
    control_r = 2
    
    pos = np.array([u * 480 / width, v * 480 / height])
    
    pos_base = 0
    rotation_base = 0
    tilt_base = 0
    
    dists = []
    phases = []
    
    count = 0
    
    for n in range(nropes):
        nsplines = all_nsplines[n]
        
        current_rotation = 0
        old_dist = curve_width / 2 + antialias_thre
        old_phase = default_phase

        last_spline = None
        last_det = None
        last_BC = None
        
        last_dist = None
        last_t = None
        
        current_dists = []
        current_phases = []
        conds = []
    
        for idx in range(nsplines):
                        
            if idx == 0:

                current_A = np.array([X[2 + pos_base], X[2 + pos_base + 1]])

                current_rotation = X[2 + total_n_pos + rotation_base]
                current_AB = X[2 + pos_base + 2] ** 2
                current_B = current_A + current_AB * np.array([cos(current_rotation), sin(current_rotation)])

                current_rotation = current_rotation + X[2 + total_n_pos + rotation_base + 1]
                current_BC = X[2 + pos_base + 3] ** 2
                current_C = current_B + current_BC * np.array([cos(current_rotation), sin(current_rotation)])

                is_control = is_control | (length(pos - current_A, 2) < control_r)

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

            is_control = is_control | (length(pos - current_B, 2) < control_r)
            is_control = is_control | (length(pos - current_C, 2) < control_r)

            if last_det is not None:
                last_det = last_det * current_AB / last_BC

            line_AC = spline.C - spline.A
            squared_AC = (line_AC[0] ** 2 + line_AC[1] ** 2)
            norm_AC = squared_AC ** 0.5

            cos_thre = 0.1
            len_thre = 3

            allow_spline = ((1 - cos(X[2 + total_n_pos + rotation_base + idx + 1])) > cos_thre) & (current_AB > len_thre) & (current_BC > len_thre)

            dist_sp, phase_sp, last_det, t_sp = get_distance_spline(pos, spline, idx, last_det)
            dist_line, phase_line, t_line = get_distance_line(pos, spline, idx)

            dist = select(allow_spline, dist_sp, dist_line)
            phase = select(allow_spline, phase_sp, phase_line)
            
            t = select(allow_spline, t_sp, t_line)

            dist = Var('dist_%s' % spline.name, dist)
            phase = Var('phase_raw_%s' % spline.name, phase)

            cond0_diff = Var('cond0_diff_%s' % spline.name, dist - curve_width / 2)

            cond0 = smoothstep(antialias_thre, -antialias_thre, cond0_diff) * cast2f(current_AB > 0.1) * cast2f(current_BC > 0.1)

            last_dist = dist
            
            current_dists.append(dist)
            current_phases.append(phase)
            conds.append(cond0)

            #dist = Var('accum_dist_%s' % spline.name,  cond_valid * dist + (1. - cond_valid) * old_dist)
            #out_phase = Var('phase_%s' % spline.name, cond_valid * phase + (1. - cond_valid) * old_phase)

            #old_dist = dist
            #old_phase = out_phase

            last_spline = spline
            last_BC = current_BC
            
        old_dist = curve_width / 2 + antialias_thre
        old_phase = default_phase
        for idx in range(nsplines):
            current_dist = current_dists[idx]
            if idx > 0:
                current_dist = select(conds[idx-1], minimum(current_dist, current_dists[idx-1]), current_dist)
            if idx < nsplines - 1:
                current_dist = select(conds[idx+1], minimum(current_dist, current_dists[idx+1]), current_dist)
                
            cond2_diff = Var('cond2_diff_%d' % count, current_phases[idx] - old_phase)
            cond2 = Var('cond2_%d' % count, cond2_diff > 0)
            cond_valid = Var('cond_valid_%d' % count, cast2f(conds[idx]) * cast2f(cond2))
                
            old_dist = Var('accum_dist_%d' % count,  cond_valid * current_dist + (1. - cond_valid) * old_dist)
            old_phase = Var('phase_%d' % count, cond_valid * current_phases[idx] + (1. - cond_valid) * old_phase)
            
            count += 1
                                       
            
        pos_base += 2 + all_nlengths[n]
        rotation_base += all_nrotations[n]
        tilt_base += all_ntilts[n]
        
        dists.append(old_dist)
        phases.append(old_phase)
        
    col = bg_col
        
    for n in range(nropes):
        cond0 = smoothstep(antialias_thre, -antialias_thre, dists[n] - curve_width / 2)
        cond1 = smoothstep(-antialias_thre, antialias_thre, dists[n] - curve_width / 2 + curve_edge)
        
        if n == 0:
            phase = phases[0]
        else:
            cond0 = cond0 * cast2f(phases[n] > phase)
            phase = maximum(phases[n], phase)
    
        col = cond0 * (cond1 * edge_col + (1 - cond1) * fill_cols[n]) + (1 - cond0) * col
    
    return Compound(col.tolist())
    
    col = select(is_control, np.array([0., 1., 1.]), col)
    
    return col
    
shaders = [test_finite_diff_quadratic_bezier_w_col_rotational_offset_better_rendering_quadratic_Z]
is_color = True