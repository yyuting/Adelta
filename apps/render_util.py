
"""
Utility functions for rendering.
"""

import math
import pprint
import sys
sys.path += ['../compiler']
import time
import hashlib
import os, os.path
import shutil
import copy
import multiprocessing
#from compiler import *
#import compiler
from compiler import *
import compiler
import numpy.random
import numpy
import numpy as np
import numpy.linalg
import skimage
import skimage.io
import skimage.feature

from tempfile import NamedTemporaryFile

import subprocess

default_is_color = True

log_prefix = '_log_'

def unique_id():
    return hashlib.md5(str(time.time()).encode('utf-8')).hexdigest()

def vec(prefix, point, style=None):
    """
    Declare several Var() instances and return them as a numpy array.

     - prefix: string prefix for the variable names.
     - point:  array-like object storing the point (e.g. containing floats or Exprs), up to 4D
    """
    ans = numpy.zeros(len(point), dtype='object')
    if style is not None:
        for i in range(len(point)):
            ans[i] = Var(style % i, point[i])
    else:
        for i in range(len(point)):
            ans[i] = Var(prefix + '_' + 'xyzw'[i], point[i])
    return ans

def set_channels(v):
    assert len(v) in [3, 4]
    v[0].channel = 'r'
    v[1].channel = 'g'
    v[2].channel = 'b'
    if len(v) == 4:
        v[3].channel = 'w'

def vec_color(prefix, point):
    assert len(point) == 3
    ans = vec(prefix, point)
    set_channels(ans)
    return ans


def vec_long(prefix, point):
    """
    Declare several Var() instances and return them as a numpy array.

     - prefix: string prefix for the variable names.
     - point:  array-like object storing the point (e.g. containing floats or Exprs), up to 4D
    """
    ans = numpy.zeros(len(point), dtype='object')
    for i in range(len(point)):
        ans[i] = Var(prefix + '_' + str(i), point[i])
    return ans

def equal(a, b):
    return (a >= b) * (a <= b)

def nequal(a, b):
    return (a > b) + (a < b)


def normalize(point, prefix=None):
    """
    Normalize a given numpy array of constants or Exprs, returning a new array with the resulting normalized Exprs.
    """
    
    if prefix is None:
        prefix = unique_id()
    
    point_squared = []
    for i in range(len(point)):
        current_squared = Var('%s_in_squared%d_%s' % (NORMALIZE_PREFIX, i, prefix), point[i] ** 2)
        point_squared.append(current_squared)
    
    var_norm2 = Var('%s_norm2_%s' % (NORMALIZE_PREFIX, prefix), sum(x for x in point_squared))
    
    var_inv_norm = Var('%s_inv_norm_%s' % (NORMALIZE_PREFIX, prefix), var_norm2 ** (-0.5))
    
    ans = vec('', np.array([x * var_inv_norm for x in point]), style=NORMALIZE_PREFIX + '_final%d_' + prefix)
    return ans


def normalize_const(v):
    """
    Normalize a numpy array of floats or doubles.
    """
    return v / numpy.linalg.norm(v)

def smoothstep(edge0, edge1, x):
    """
    re-implementation of opengl smoothstep
    https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/smoothstep.xhtml
    genType t;  /* Or genDType t; */
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
    """
    t0 = (x - edge0) / (edge1 - edge0)
    t = maximum(minimum(t0, 1.0), 0.0)
    return t * t * (3.0 - 2.0 * t)

def mix(x, y, a):
    """
    re-implementation of opengl mix
    https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/mix.xhtml
    The return value is computed as x×(1−a)+y×a
    """
    return x * (1.0 - a) + y * a
            

def clip_0_1(val):
    return max(min(val, 1.0), 0.0)

def length(vec, norm):
    
    if isinstance(vec, Expr):
        L = vec.ndims
    else:
        assert isinstance(vec, (list, tuple, np.ndarray))
        L = len(vec)
    
    return sum(vec[i] ** norm for i in range(L)) ** (1 / norm)

def det2x2(A):
    """
    Calculate determinant of a 2x2 matrix
    """
    return A[0,0]*A[1,1] - A[0,1]*A[1,0]

def det3x3(A):
    """
    Calculate determinant of a 3x3 matrix
    """
    return A[0, 0] * A[1, 1] * A[2, 2] + A[0, 2] * A[1, 0] * A[2, 1] + \
           A[0, 1] * A[1, 2] * A[2, 0] - A[0, 2] * A[1, 1] * A[2, 0] - \
           A[0, 1] * A[1, 0] * A[2, 2] - A[0, 0] * A[1, 2] * A[2, 1]

def transpose_3x3(A):
    """
    transpose a 3x3 matrix
    """
    B = A[:]


def inv3x3(A):
    """
    Inverse of a 3x3 matrix
    """
    a00 = A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]
    a01 = A[0, 2] * A[2, 1] - A[0, 1] * A[2, 2]
    a02 = A[0, 1] * A[1, 2] - A[0, 2] * A[1, 1]
    a10 = A[1, 2] * A[2, 0] - A[1, 0] * A[2, 2]
    a11 = A[0, 0] * A[2, 2] - A[0, 2] * A[2, 0]
    a12 = A[0, 2] * A[1, 0] - A[0, 0] * A[1, 2]
    a20 = A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0]
    a21 = A[0, 1] * A[2, 0] - A[0, 0] * A[2, 1]
    a22 = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    det = det3x3(A)
    return numpy.array([[a00, a01, a02],
                        [a10, a11, a12],
                        [a20, a21, a22]]) / det

def matrix_vec_mul3(A, x):
    """
    Matrix multiplication with vector x
    Matrix is 3x3, x is 1x3
    """
    y0 = A[0, 0] * x[0] + A[0, 1] * x[1] + A[0, 2] * x[2]
    y1 = A[1, 0] * x[0] + A[1, 1] * x[1] + A[1, 2] * x[2]
    y2 = A[2, 0] * x[0] + A[2, 1] * x[1] + A[2, 2] * x[2]
    return numpy.array([y0, y1, y2])

def cross(x, y):
    """
    cross product of 2 length 3 vectors
    """
    z0 = x[1] * y[2] - x[2] * y[1]
    z1 = x[2] * y[0] - x[0] * y[2]
    z2 = x[0] * y[1] - x[1] * y[0]
    return numpy.array([z0, z1, z2])

def list_or_scalar_to_str(render_t):
    if hasattr(render_t, '__len__'):
        return ','.join(str(x) for x in render_t)
    else:
        return str(render_t)

def render_shader(objective_functor, is_color=default_is_color, base_dir='out', extra_suffix='', input_nargs=0, backend='tf', args_range=None, sigmas_range=None, debug_ast=False, compiler_modes=None, select_rule=1, compute_g=True, autoscheduler=False, do_prune=None, par_vals=None, multiplication_rule=3, AD_only=False, allow_raymarching_random=False, ndims=2):
    """
    Low-level routine for rendering a shader.

    """
    
    if backend == 'glsl':
        compiler.need_animate = True

    T0 = time.time()
    
    X = ArgumentArray(ndims=input_nargs)
    scalar_loss_scale = ArgumentArray(ndims=input_nargs, name='scalar_loss_scale')
    
    u = ArgumentScalar(DEFAULT_ARGUMENT_SCALAR_U_NAME)
    tup = (u,)
    
    if ndims > 1:
        v = ArgumentScalar(DEFAULT_ARGUMENT_SCALAR_V_NAME)
        tup += (v,)
        
    if ndims > 2:
        w = ArgumentScalar(DEFAULT_ARGUMENT_SCALAR_W_NAME)
        tup += (w,)
    
    f = objective_functor(*tup, X, scalar_loss_scale)
    if isinstance(f, (list, tuple)):
        scalar_loss = f[1]
        f = f[0]
    else:
        scalar_loss = None
        
    f.root = True

    c = CompilerParams(input_nargs=input_nargs, backend=backend, args_range=args_range, sigmas_range=sigmas_range, debug_ast=debug_ast, select_rule=select_rule, multiplication_rule=multiplication_rule, compute_g=compute_g, autoscheduler=autoscheduler, do_prune=do_prune, par_vals=par_vals, gradient_mode='AD' if AD_only else 'ours', allow_raymarching_random=allow_raymarching_random, ndims=ndims, is_color=is_color)
    if is_color:
        c.constructor_code = [OUTPUT_ARRAY + '.resize(3);']

    check(f, c, ndims=3, outdir=base_dir, compiler_modes=compiler_modes, scalar_loss=scalar_loss)
    
    return

def multiple_shaders():
    def f(objective_functor_L, *args, **kw):
        ans = []
        for objective_functor in objective_functor_L:
            ans.append(render_shader(objective_functor, *args, **kw))
        return ans
    return f

def output_color(c):
    """
    Given a array-like object with 3 Exprs, output an RGB color for a shader. Has the side effect of setting channels of c.
    """
    set_channels(c)
    return Compound(c)