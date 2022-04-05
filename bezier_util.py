import numpy as np
import skimage
import skimage.io
import os
import sys
import scipy.interpolate

def calc_weight_map(dest_dir, name0, name1, scale=10):
    
    imgs = []
    for name in [name0, name1]:
        imgs.append(skimage.img_as_float(skimage.io.imread(os.path.join('knots_imgs/overhand', name + '.png'))))
    
    # assume BG is white, diff finds pixels that are non BG in either img0 or img1, but not both
    diff = (imgs[0].sum(-1) != 3) != (imgs[1].sum(-1) != 3)
    
    weight = np.ones(imgs[0].shape[:2])
    weight[diff] = scale
    
    np.save(os.path.join(dest_dir, 'weight_%s_to_%s.npy' % (name0, name1)), np.expand_dims(weight, -1))
                    
def invert_params(par, nsplines, keep_phase=False):
    
    ans = np.zeros(par.shape)
    ans[:2] = par[:2]
    
    nlengths = 2 * nsplines
    nrotations = nsplines + 1
    
    angs = par[4+nlengths:4+nlengths+nrotations]
    current_ang = angs[0]
    
    current_A = par[2:4]
    
    for n in range(nsplines):
            
        len_AB = par[4 + n * 2] ** 2
        current_B = current_A + len_AB * np.array([np.cos(current_ang), np.sin(current_ang)])

        current_ang += angs[n+1]
        len_BC = par[5 + n * 2] ** 2
        current_C = current_B + len_BC * np.array([np.cos(current_ang), np.sin(current_ang)])

        current_A = current_C
        
    ans[2:4] = current_C
    ans[4:4+nlengths] = par[4:4+nlengths][::-1]
    ans[4+nlengths] = current_ang + np.pi
    ans[5+nlengths:4+nlengths+nrotations] = -par[5+nlengths:4+nlengths+nrotations][::-1]
    
    if keep_phase:
        ans[4+nlengths+nrotations:] = par[4+nlengths+nrotations:][::-1]
    
    return ans

def get_par(points, par_ref):
    
    nsplines = (points.shape[0] - 1) // 2
    
    ans = par_ref.copy()
    ans[2:4] = points[0]
    
    nlengths = 2 * nsplines
    nrotations = nsplines + 1
    
    old_ang = 0
    
    for n in range(nsplines):
        len_AB = np.linalg.norm(points[2 * n] - points[2 * n + 1])
        len_BC = np.linalg.norm(points[2 * n + 1] - points[2 * n + 2])
        
        ans[4 + n * 2] = max(len_AB ** 0.5, 0.1)
        ans[5 + n * 2] = max(len_BC ** 0.5, 0.1)
        
        if n == 0:
            old_ang = np.arctan2(*((points[1] - points[0])[::-1]))
            ans[4 + nlengths] = old_ang
            
        if n < nsplines - 1:
            current_ang = np.arctan2(*((points[2 * n + 3] - points[2 * n + 1])[::-1]))
        else:
            current_ang = np.arctan2(*((points[2 * n + 2] - points[2 * n + 1])[::-1]))
            
        ans[4 + nlengths + n + 1] = current_ang - old_ang
        
        old_ang = current_ang
        
    return ans

def get_points(ans, nsplines, offset=True, rotational=True):
    ncontrols = nsplines + 2
    
    if offset:
        start_idx = 2
    else:
        start_idx = 8
    
    if not rotational:
        explicit_points = ans[start_idx:start_idx+2*ncontrols].reshape(ncontrols, 2)

        if nsplines == 1:
            return explicit_points
        else:
            points = np.zeros((2 * nsplines + 1, 2))
            points[:3] = explicit_points[:3]

            last_B = explicit_points[1]
            last_C = explicit_points[2]

            point_count = 3
            for n in range(1, nsplines):
                this_scale = ans[start_idx + 4 + 3 * nsplines + n] ** 2
                this_B = (last_C - (1 - this_scale) * last_B) / this_scale

                if offset:
                    last_C = last_C + explicit_points[2 + n]
                else:
                    last_C = explicit_points[2 + n]

                last_B = this_B

                points[point_count] = this_B
                points[point_count + 1] = last_C

                point_count += 2

            return points, last_B
    else:
        
        nlengths = 2 * nsplines
        nrotations = nsplines + 1
        
        points = np.zeros((2 * nsplines + 1, 2))
        points[0] = ans[2:4]
        
        angs = ans[4+nlengths:4+nlengths+nrotations]
        
        current_ang = angs[0]
        current_A = points[0]
        
        for n in range(nsplines):
            
            len_AB = ans[4 + n * 2] ** 2
            current_B = current_A + len_AB * np.array([np.cos(current_ang), np.sin(current_ang)])
            
            current_ang += angs[n+1]
            len_BC = ans[5 + n * 2] ** 2
            current_C = current_B + len_BC * np.array([np.cos(current_ang), np.sin(current_ang)])
            
            points[1 + 2 * n] = current_B
            points[2 + 2 * n] = current_C
            
            current_A = current_C
            
        return points
    
def shrink_par(par, nsplines, last_shrink_idx=None):
    
    par = par.copy()
    
    nlengths = 2 * nsplines
    nrotations = nsplines + 1

    points = get_points(par, nsplines, True, True)

    if last_shrink_idx is None:
        for idx in range(nsplines - 1, -1, -1):
            if np.allclose(points[2 * idx], points[2 * idx + 1], atol=2) and \
               np.allclose(points[2 * idx + 1], points[2 * idx + 2], atol=2):
                last_shrink_idx = idx

    if last_shrink_idx is not None:
        if True:
            # keep extra splines at the tail
            par[4 + last_shrink_idx * 2 : 4 + nlengths] = 1
            par[5 + nlengths + last_shrink_idx : 4 + nlengths + nrotations] = 0.1

            freeze_idx = np.arange(4 + last_shrink_idx * 2, 4 + nlengths).tolist() + \
                         np.arange(5 + nlengths + last_shrink_idx, 4 + nlengths + nrotations).tolist()
        else:
            # hide extra splines at the beginning
            n_shrink = nsplines - last_shrink_idx

            par[4 + 2 * n_shrink : 4 + nlengths] = par[4 : 4 + last_shrink_idx * 2]
            par[4 + nlengths + n_shrink : 4 + nlengths + nrotations] = par[4 + nlengths : 5 + nlengths + last_shrink_idx]

            par[4 : 4 + 2 * n_shrink] = 0
            par[4 + nlengths : 4 + nlengths + n_shrink] = 0

            freeze_idx = np.arange(4, 4 + 2 * n_shrink).tolist() + \
                         np.arange(4 + nlengths, 4 + nlengths + n_shrink).tolist()

        remaining_idx = list(set(np.arange(par.shape[0])).difference(set(freeze_idx)))
    else:
        remaining_idx = np.arange(par.shape[0]).tolist()

    print('remaining idx:', ','.join([str(val) for val in remaining_idx]))
    
    return par, remaining_idx

def expand_spline(old_par, nsplines, nsamples, free_last_n=1, par_format=0, old_nlengths=0, old_nrotations=0):
    
    nrotations = nsplines + 1
    nlengths = 2 * nsplines
    
    if par_format == 0:
        extra_tilt = 1
    elif par_format == 1:
        extra_tilt = 2
    else:
        raise
    
    nargs = old_par.shape[0] + (3 + extra_tilt)
    
    total_nlengths = nlengths + old_nlengths
    total_nrotations = nrotations + old_nrotations
    
    old_nrotations += nsplines
    old_nlengths += 2 * (nsplines - 1)

    par = np.zeros([nsamples, nargs])
    par[:, :4 + old_nlengths] = old_par[:4 + old_nlengths]
    
    par[:, 4 + total_nlengths : 4 + total_nlengths + old_nrotations] = \
    old_par[4 + old_nlengths : 4 + old_nlengths + old_nrotations]
    
    par[:, 4 + total_nlengths + total_nrotations : -extra_tilt] = old_par[4 + old_nlengths + old_nrotations:]

    if False:
        par[:, 4 + old_nlengths] = 5 ** 0.5
        par[:, 4 + old_nlengths + 1] = 10 ** 0.5
        par[:, 4 + nlengths + old_nrotations : 4 + nlengths + nrotations] = np.random.rand(nsamples, 1) * np.pi - np.pi / 2
    else:
        par[:, 4 + old_nlengths] = 0.1
        par[:, 4 + old_nlengths + 1] = 0.1
        
        par[:, 4 + total_nlengths + old_nrotations : 4 + total_nlengths + total_nrotations] = \
        np.random.rand(nsamples, 1) * 0.1 - 0.05

    par_match_weight = np.ones(nargs)
    par_match_weight[4 + total_nlengths - 2 * free_last_n : 4 + total_nlengths] = 0
    #par_match_weight[4 + nlengths + old_nrotations : 4 + nlengths + nrotations] = 0
    par_match_weight[4 + total_nlengths : 4 + total_nlengths + total_nrotations] = 0
    par_match_weight[-free_last_n * extra_tilt:] = 0
    
    return par, par_match_weight

def subdivide(orig_nsplines, new_nsplines, old_par, par_format=0):
    nextra = new_nsplines - orig_nsplines
        
    old_points = get_points(old_par, orig_nsplines, True, True)

    new_points = np.zeros((old_points.shape[0] + nextra * 2, 2))
    new_points[:old_points.shape[0] - 2] = old_points[:-2]

    A = old_points[-3]
    B = old_points[-2]
    C = old_points[-1]

    for idx in range(nextra + 1):
        if idx < nextra:
            dt = 1 / (nextra + 1 - idx)

            new_A = A
            new_B = (1 - dt) * A + dt * B
            new_C = (1 - dt) ** 2 * A + 2 * (1 - dt) * dt * B + dt ** 2 * C

            A = new_C
            B = (1 - dt) * B + dt * C
        else:
            new_A = A
            new_B = B
            new_C = C

        new_points[old_points.shape[0] - 2 + 2 * idx] = new_B
        new_points[old_points.shape[0] - 2 + 2 * idx + 1] = new_C

    new_par_ref, par_match_weight = expand_spline(old_par, new_nsplines, 1, free_last_n=2, par_format=par_format)

    new_par = get_par(new_points, new_par_ref[0])
    
    return new_par, par_match_weight

def transfer(transfer_mode, src_nsplines, tar_nsplines, par_format, src_par, tar_par, subdivide_pattern):
        
    assert src_nsplines >= tar_nsplines

    

    allow_subdivide = True
    last_phase_idx = src_nsplines - tar_nsplines
    #last_phase_idx = -(src_nsplines - tar_nsplines) - 1

    # expand target to be same size as source
    for level in range(tar_nsplines - 1, src_nsplines - 1):

        is_subdivide = allow_subdivide and subdivide_pattern[level]

        if is_subdivide:
            last_phase_idx -= 1
            if transfer_mode == 'expand':
                tar_par, _ = subdivide(level + 1, level + 2, tar_par, par_format=par_format)
        else:
            allow_subdivide = False
            if transfer_mode == 'expand':
                tar_par, _ = expand_spline(tar_par, level + 2, 1, par_format=par_format)
                tar_par = tar_par[0]

    if transfer_mode == 'expand':
        if par_format == 0:
            tar_par[-src_nsplines - 1 : -(src_nsplines - tar_nsplines) - 1] = src_par[-src_nsplines - 1 : -(src_nsplines - tar_nsplines) - 1]
            tar_par[-(src_nsplines - tar_nsplines) - 1:] = src_par[-last_phase_idx - 1]
        elif par_format == 1:
            tar_par[-2 * src_nsplines - 1 : -2 * (src_nsplines - tar_nsplines) - 2] = src_par[-2 * src_nsplines - 1 : -2 * (src_nsplines - tar_nsplines) - 2]
            if allow_subdivide:
                tar_par[-2 * (src_nsplines - tar_nsplines) - 2] = src_par[-2 * (last_phase_idx + 1) - 1]
            else:
                tar_par[-2 * (src_nsplines - tar_nsplines) - 2] = src_par[-2 * (src_nsplines - tar_nsplines) - 2]
            tar_par[-2 * (src_nsplines - tar_nsplines) - 1:] = src_par[-2 * last_phase_idx - 1]
        else:
            raise
    elif transfer_mode == 'orig':
        assert par_format == 0
        tar_par[-tar_nsplines - 1:-1] = src_par[-src_nsplines - 1 : -(src_nsplines - tar_nsplines) - 1]
        tar_par[-1] = src_par[last_phase_idx]
    else:
        raise
        
    return tar_par

def disassemble_par(nsplines, par):
    
    total_n_pos = 0
    total_nrotations = 0
    for n in nsplines:
        total_n_pos += 2 + 2 * n
        total_nrotations += n + 1
        
    n_pos_stop_idx = total_n_pos + 2
    nrotations_stop_idx = n_pos_stop_idx + total_nrotations
    
    pars = []
    
    last_n_pos_idx = 2
    last_nrotations_idx = n_pos_stop_idx
    last_ntilts_idx = nrotations_stop_idx
    
    for n in nsplines:
        current_n_pos = 2 + 2 * n
        current_nrotations = n + 1
        current_ntilts = 2 * n + 1
        
        current_par = np.concatenate((par[:2],
                                      par[last_n_pos_idx : last_n_pos_idx + current_n_pos],
                                      par[last_nrotations_idx : last_nrotations_idx + current_nrotations],
                                      par[last_ntilts_idx : last_ntilts_idx + current_ntilts]))
        
        pars.append(current_par)

        last_n_pos_idx += current_n_pos
        last_nrotations_idx += current_nrotations
        last_ntilts_idx += current_ntilts
        
    return pars

def assemble_par(nsplines, pars):
    
    total_n_pos = 0
    total_nrotations = 0
    total_ntilts = 0
    for n in nsplines:
        total_n_pos += 2 + 2 * n
        total_nrotations += n + 1
        total_ntilts += 2 * n + 1
        
    n_pos_stop_idx = total_n_pos + 2
    nrotations_stop_idx = n_pos_stop_idx + total_nrotations
    
    total_nargs = 2 + total_n_pos + total_nrotations + total_ntilts
    
    par = np.zeros((pars[0].shape[0], total_nargs))
    par[:, :2] = pars[0][:, :2]
    
    last_n_pos_idx = 2
    last_nrotations_idx = n_pos_stop_idx
    last_ntilts_idx = nrotations_stop_idx
    
    for idx in range(len(nsplines)):
        n = nsplines[idx]
        
        current_n_pos = 2 + 2 * n
        current_nrotations = n + 1
        current_ntilts = 2 * n + 1
        
        par[:, last_n_pos_idx : last_n_pos_idx + current_n_pos] = pars[idx][:, 2 : 2 + current_n_pos]
        par[:, last_nrotations_idx : last_nrotations_idx + current_nrotations] = pars[idx][:, 2 + current_n_pos : 2 + current_n_pos + current_nrotations]
        par[:, last_ntilts_idx : last_ntilts_idx + current_ntilts] = pars[idx][:, -current_ntilts:]
        
        last_n_pos_idx += current_n_pos
        last_nrotations_idx += current_nrotations
        last_ntilts_idx += current_ntilts
    
    return par

def interp(nsplines, nsamples, pars0, pars1, grow_from=None):
        
    xs = np.linspace(0, 1, nsamples)

    points0 = get_points(pars0, nsplines, True, True)
    points1 = get_points(pars1, nsplines, True, True)

    if grow_from is None:
        cs_points = scipy.interpolate.CubicSpline(np.arange(2), np.stack([points0, points1], 0), bc_type='clamped')
        points_interp = cs_points(xs)
    else:
        #points_interp = np.tile(np.expand_dims(points0, 0), [nsamples, 1, 1])
        
        cs_points = scipy.interpolate.CubicSpline(np.arange(2), np.stack([points0, points1], 0), bc_type='clamped')
        points_interp = cs_points(xs)
        
        points_mid_idx = 2 * grow_from + 1
        points_last_idx = points_mid_idx + 1
        
        points0 = np.expand_dims(points0, 0)
        points1 = np.expand_dims(points1, 0)
        xs_expanded = np.expand_dims(xs, 1)
        
        points_last = points0[:, points_last_idx - 2] * (1 - xs_expanded) ** 2 + \
                      points1[:, points_last_idx - 1] * 2 * xs_expanded * (1 - xs_expanded) + \
                      points1[:, points_last_idx] * xs_expanded ** 2
        
        points_mid = points0[:, points_last_idx - 2] * (1 - xs_expanded / 2) ** 2 + \
                     points1[:, points_last_idx - 1] * 2 * xs_expanded / 2 * (1 - xs_expanded / 2) + \
                     points1[:, points_last_idx] * (xs_expanded / 2) ** 2
        
        points_second_last = 2 * (points_mid - 0.25 * points_last - 0.25 * points0[:, points_last_idx - 2])
        
        points_interp[:, points_last_idx - 1] = points_second_last
        points_interp[:, points_last_idx:] = np.expand_dims(points_last, 1)
        
    cs_par = scipy.interpolate.CubicSpline(np.arange(2), np.stack([pars0, pars1], 0), bc_type='clamped')
    par_interp = cs_par(xs)
    #par_interp = np.linspace(pars[0], pars[1], nsamples)

    # assuming at most one intersection when growing one spline
    # par always follows that of the later spline
    # in 3D, this is as if always stretch the rope to far behind or far back before making the rope crossing
    #par_interp = np.tile(np.expand_dims(pars1, 0), [nsamples, 1])

    # phase interpolation should be that assuming t is uniformly sampled in [0, 1] for every sample
    # the endpoint value should be the quadratic interpolation at t of the original bezier
    # current midpoint should be the quadratic interpolation at t / 2 of the original bezier
    
    if grow_from is not None:
        
        par_last_idx = 2 + 2 + 2 * nsplines + nsplines + 1 + 2 * (grow_from + 1)
        
        #par_last_idx = np.where(pars1 == pars1[-1])[0][0]
        par_last = pars1[par_last_idx - 2] * (1 - xs) ** 2 + \
                   pars1[par_last_idx - 1] * 2 * xs * (1 - xs) + \
                   pars1[par_last_idx] * xs ** 2

        par_mid = pars1[par_last_idx - 2] * (1 - xs / 2) ** 2 + \
                  pars1[par_last_idx - 1] * 2 * xs / 2 * (1 - xs / 2) + \
                  pars1[par_last_idx] * (xs / 2) ** 2

        par_second_last = 2 * (par_mid - 0.25 * par_last - 0.25 * pars1[par_last_idx - 2])

        par_interp[:, par_last_idx - 1] = par_second_last
        par_interp[:, par_last_idx:] = np.expand_dims(par_last, -1)

    final_par = np.zeros(par_interp.shape)

    for idx in range(nsamples):
        current_par = get_par(points_interp[idx], par_interp[idx])
        
        final_par[idx] = current_par
        
    return final_par
        
if __name__ == '__main__':
    
    mode = sys.argv[1]
    
    if mode == 'revert':
        par = np.load(sys.argv[2])[0]
        nsplines = int(sys.argv[3])

        if len(sys.argv) >= 6:
            keep_phase = bool(int(sys.argv[5]))
        else:
            keep_phase = False

        ans = invert_params(par, nsplines, keep_phase)

        np.save(sys.argv[4], np.expand_dims(ans, 0))
    elif mode == 'interp':
        par0 = np.load(sys.argv[2])[0]
        par1 = np.load(sys.argv[3])[0]
        
        nsplines = int(sys.argv[4])
        
        points0 = get_points(par0, nsplines, True, True)
        points1 = get_points(par1, nsplines, True, True)
        
        nsamples = 100
        
        points_interp = np.linspace(points0, points1, nsamples)
        par_interp = np.linspace(par0, par1, nsamples)
        
        final_par = np.zeros(par_interp.shape)
        
        for idx in range(nsamples):
            final_par[idx] = get_par(points_interp[idx], par_interp[idx])
            
        np.save(sys.argv[5], final_par)
    elif mode == 'interp_multiple':
        nsplines = [int(val) for val in sys.argv[2].split(',')]
        output_file = sys.argv[5]
        
        input_files = sys.argv[3:5]
        
        if len(sys.argv) >= 7:
            par0_grow_from = [int(val) for val in sys.argv[6].split(',')]
        else:
            par0_grow_from = None
        
        nsamples = (len(input_files) - 1) * 100
        
        pars = []
        for file in input_files:
            pars.append(np.load(file)[0])
            
        if len(nsplines) == 1:
            final_par = interp(nsplines[0], nsamples, pars[0], pars[1])
        else:
            pars0 = disassemble_par(nsplines, pars[0])
            pars1 = disassemble_par(nsplines, pars[1])
            
            final_pars = []
            for idx in range(len(nsplines)):
                
                if par0_grow_from is None:
                    grow_from = None
                elif len(par0_grow_from) <= idx:
                    grow_from = None
                elif par0_grow_from[idx] == nsplines[idx]:
                    grow_from = None
                else:
                    grow_from = par0_grow_from[idx]
                
                final_pars.append(interp(nsplines[idx], nsamples, pars0[idx], pars1[idx], grow_from=grow_from))
            final_par = assemble_par(nsplines, final_pars)
            
        np.save(output_file, final_par)
            
    elif mode == 'shrink':
        
        orig_file = sys.argv[2]
        
        par = np.load(orig_file)[0]
        nsplines = int(sys.argv[3])
        
        if len(sys.argv) >= 5:
            last_shrink_idx = int(sys.argv[4])
        else:
            last_shrink_idx = None
        
        par, _ = shrink_par(par, nsplines, last_shrink_idx)
            
        filename, _ = os.path.splitext(orig_file)
        np.save(filename + '_shrinked.npy', np.expand_dims(par, 0))           
        
    elif mode == 'combine':
        combined = None
        for file in sys.argv[2:-1]:
            if combined is not None:
                combined = np.concatenate((combined, np.load(file)), 0)
            else:
                combined = np.load(file)
                
        np.save(sys.argv[-1], combined)
    elif mode == 'initialize':
        nsplines = 1
        nsamples = int(sys.argv[2])
       
        start = np.array([int(val) for val in sys.argv[3].split(',')])
        end = np.array([int(val) for val in sys.argv[4].split(',')])
        
        par_format = int(sys.argv[5])
        
        nrotations = nsplines + 1
        nlengths = 2 * nsplines
        
        if par_format == 0:
            ntilts = nsplines + 1
        elif par_format == 1:
            ntilts = 2 * nsplines + 1
        else:
            pass

        nargs = 2 + 2 + nlengths + nrotations + ntilts
        
        if len(sys.argv) > 7:
            prev_rope_par = np.load(sys.argv[7])
            prev_nsplines = [int(val) for val in sys.argv[8].split(',')]
            
            old_n_pos = 0
            old_nrotations = 0
            old_ntilts = 0
            
            for n in prev_nsplines:
                old_n_pos += 2 + 2 * n
                old_nrotations += n + 1
                if par_format == 0:
                    old_ntilts += n + 1
                else:
                    old_ntilts += 2 * n + 1
                    
            nargs += old_n_pos + old_nrotations + old_ntilts
            
            par = np.zeros([nsamples, nargs])
            
            par[:, :2 + old_n_pos] = prev_rope_par[0, :2 + old_n_pos]
            
            par[:, 2 + old_n_pos + 2 + nlengths : 2 + old_n_pos + 2 + nlengths + old_nrotations] = \
            prev_rope_par[0, 2 + old_n_pos : 2 + old_n_pos + old_nrotations]
            
            par[:, 2 + old_n_pos + 2 + nlengths + old_nrotations + nrotations : -ntilts] = \
            prev_rope_par[0, -old_ntilts:]
            
            par_match_weight = np.ones(nargs)
            par_match_weight[2 + old_n_pos : 2 + old_n_pos + 2 + nlengths] = 0
            par_match_weight[2 + old_n_pos + 2 + nlengths + old_nrotations : 2 + old_n_pos + 2 + nlengths + old_nrotations + nrotations] = 0
            par_match_weight[-ntilts:] = 0
            
            np.save(sys.argv[6][:-4] + '_weight.npy', par_match_weight)
        else:
            old_n_pos = 0
            old_nrotations = 0
            old_ntilts = 0
        
            par = np.zeros([nsamples, nargs])
            
        for idx in range(nsamples):
            if old_n_pos == 0:
                par[idx, 0] = np.random.rand() + 3
                par[idx, 1] = np.random.rand() + 1
            
            current_start = start + np.random.rand(2) * 10 - 5
            current_end = end + np.random.rand(2) * 10 - 5
            
            current_len = np.linalg.norm(current_start - current_end) / 2
            
            current_ang = np.arctan2(*((current_end - current_start)[::-1]))
            
            par[idx, 2 + old_n_pos :2 + old_n_pos + 2] = current_start
            par[idx, 2 + old_n_pos + 2 : 2 + old_n_pos + 4] = current_len ** 0.5
            par[idx, 2 + old_n_pos + 2 + nlengths + old_nrotations] = current_ang
            par[idx, 2 + old_n_pos + 2 + nlengths + old_nrotations + 1] = np.random.rand() * 0.1 - 0.05
        
        np.save(sys.argv[6], par)
    elif mode == 'expand':
        nsplines = int(sys.argv[2])
        nsamples = int(sys.argv[3])
        par_format = int(sys.argv[4])
        
        old_par = np.load(sys.argv[5])[0]
        
        old_n_pos = 0
        old_nrotations = 0
        
        if len(sys.argv) > 7:
            prev_nsplines = [int(val) for val in sys.argv[7].split(',')]
            for n in prev_nsplines:
                old_n_pos += 2 + 2 * n
                old_nrotations += n + 1
        
        par, par_match_weight = expand_spline(old_par, nsplines, nsamples, par_format=par_format, old_nlengths=old_n_pos, old_nrotations=old_nrotations)
        
        np.save(sys.argv[6] + '.npy', par)
        np.save(sys.argv[6] + '_weight.npy', par_match_weight)
        
    elif mode == 'subdivide':
        orig_nsplines = int(sys.argv[2])
        new_nsplines = int(sys.argv[3])
        nsamples = int(sys.argv[4])
        par_format = int(sys.argv[5])
        
        old_par = np.load(sys.argv[6])[0]
        
        new_par, par_match_weight = subdivide(orig_nsplines, new_nsplines, old_par, par_format=par_format)

        np.save(sys.argv[7] + '.npy', np.tile(np.expand_dims(new_par, 0), (nsamples, 1)))
        np.save(sys.argv[7] + '_weight.npy', par_match_weight)
        
    elif mode == 'reset_phase':
        all_splines = [int(val) for val in sys.argv[2].split(',')]
        nrepeat = int(sys.argv[3])
        par_format = int(sys.argv[4])
        par = np.load(sys.argv[5])
        
        if par_format == 0:
            ntilt = sum([nsplines + 1 for nsplines in all_splines])
        elif par_format == 1:
            ntilt = sum([2 * nsplines + 1 for nsplines in all_splines])
        else:
            raise
        
        par[:, -ntilt:] = 0
        par = np.tile(par, (nrepeat, 1))
        np.save(sys.argv[6], par)
    elif mode == 'transfer':
        
        transfer_mode = sys.argv[2]
        
        src_nsplines = [int(val) for val in sys.argv[3].split(',')]
        tar_nsplines = [int(val) for val in sys.argv[4].split(',')]
        
        par_format = int(sys.argv[5])

        src_par = np.load(sys.argv[7])[0]
        tar_par = np.load(sys.argv[8])[0]
        
        subdivide_pattern = [int(val) > 0 for val in sys.argv[6].split(',')]
        
        accum_n = 0
        tar_par_safe = None
        
        if len(src_nsplines) == 1 and len(tar_nsplines) == 1:
            tar_par = transfer(transfer_mode, src_nsplines[0], tar_nsplines[0], par_format, src_par, tar_par, subdivide_pattern)
        else:
            
            src_pars = disassemble_par(src_nsplines, src_par)
            tar_pars = disassemble_par(tar_nsplines, tar_par)
            
            pars = []
            pars_safe = []
            
            for idx in range(len(src_nsplines)):
                if len(tar_nsplines) <= idx:
                    current_tar = np.zeros(src_pars[idx].shape)
                    current_tar[:4] = src_pars[idx][:4]
                    current_tar[-2 * src_nsplines[idx] - 1:] = src_pars[idx][-2 * src_nsplines[idx] - 1]
                    
                    current_tar_safe = current_tar.copy()
                    #current_tar_safe[-2 * src_nsplines[idx] - 1:] = -1e4-1
                    
                    if len(pars) != len(pars_safe):
                        assert len(pars) > len(pars_safe)
                        pars_safe = pars_safe + pars[len(pars_safe):]
                        pars_safe.append(np.expand_dims(current_tar_safe, 0))
                    
                    pars.append(np.expand_dims(current_tar, 0))
                    
                    continue
                    
                    
                    current_tar[:2] = tar_pars[0][:2]
                    current_tar[2:] = src_pars[idx][2:4]
                    current_tar_n = 0
                else:
                    current_tar = tar_pars[idx]
                    current_tar_n = tar_nsplines[idx]
                
                current_src = src_pars[idx]
                
                #if current_tar_n == src_nsplines[idx]:
                #    pars.append(current_tar)
                #    continue
                
                current_par = transfer(transfer_mode, src_nsplines[idx], current_tar_n, par_format, 
                                       current_src, current_tar,
                                       subdivide_pattern[accum_n : accum_n + src_nsplines[idx] - 1])
                pars.append(np.expand_dims(current_par, 0))
                
                accum_n += src_nsplines[idx] - 1
            
            tar_par = assemble_par(src_nsplines, pars)[0]
            if len(pars_safe) > 0:
                pars_safe = pars_safe + pars[len(pars_safe):]
                tar_par_safe = assemble_par(src_nsplines, pars_safe)[0]
                                                
        np.save(sys.argv[9], np.expand_dims(tar_par, 0))
        if tar_par_safe is not None:
            np.save(sys.argv[9][:-4] + '_safe.npy', np.expand_dims(tar_par_safe, 0))
        
        
        
    