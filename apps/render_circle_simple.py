from render_util import *
nargs = 6
args_range = np.array([256., 256., 256., 1., 1., 1.])

def circle_simple(u, v, X):
    ox = X[0]
    oy = X[1]
    r = X[2]
    col = np.array([X[3], X[4], X[5]])
    bg = np.zeros(3)
    cond = (ox - u) ** 2 + (oy - v) ** 2 <= r ** 2
    out = select(cond, col, bg)
    return out

shaders = [circle_simple]