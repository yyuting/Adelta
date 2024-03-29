from render_util import *

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

# placeholder for render size
width = ArgumentScalar('width')
height = ArgumentScalar('height')

max_iter = nrings - 1

default_phase = -1e4

def celtic_knot(u, v, X, scalar_loss_scale):
    
    # inout_ls: correspond to GLSL keywork inout, variables can be modified
    u,v = Animate("animate_uv", inout_ls=[Var("u", u), Var("v", v)]).update()
    
    # make sure it's non-negative
    curve_width = X[0] ** 2
    curve_edge = X[1] ** 2
    
    # Var: give variables a semantically meaningful name
    # Compound: a primitive that works similar to array
    fill_col = Var('fill_col', Compound([1., 1., 1.]))
    edge_col = Compound([0., 0., 0.])
    
    # Group each ring-related parameter into a seperate object
    rings = []
    for i in range(nrings):
        
        ring_params = [X[i + k * nrings + 2] for k in range(4)]
        ring = Object('ring', 
                      pos = ring_params[:2],
                      radius = ring_params[2],
                      tilt = ring_params[3])
        rings.append(ring)
        
    # render each ring
    def update_ring(old_vals, ring, idx):
        # Update function should be side-effect free
        
        old_col, old_phase = old_vals[0], old_vals[1]
        
        rel_pos = vec('rel_pos_%d' % idx, np.array([u, v]) - ring.pos)
        
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
        
        # in_ls: corresponds to GLSL keyword in, variables are read only
        ring.fill_col, = Animate("ring_col%s"%idx, inout_ls=[fill_col], in_ls=[rel_pos]).update()
        
        # select: used to represent if/else branches
        col_current = Var('col_current_%s' % ring.name, select(cond1, edge_col, ring.fill_col))
        
        col = Var('col_%s' % ring.name, select(cond_valid, col_current, old_col))
        
        out_phase = Var('phase_%s' % ring.name, select(cond_valid, phase, old_phase))
        
        return [col, out_phase]
    
    global default_phase
    # BG
    col = Compound([1, 1, 1])
    
    vals = [col, default_phase]
    
    for i in range(nrings):
        optional_update(update_ring, vals, rings[i], i)
        
    return vals[0]

shaders = [celtic_knot]
is_color = True