import tensorflow as tf
import numpy as np
import numpy
import math

dtype = tf.float32

tag_discontinuous = False

def get_scale(pix_idx):
    if pix_idx in [1, 3]:
        return -1
    else:
        return 1

def new_mul(x, y):
    try:
        if x == 0.0 or y == 0.0:
            return 0.0
    except:
        pass
    
    try:
        if x == 1:
            return y
        if y == 1:
            return x
    except:
        pass
    
    try:
        if (x.dtype == bool) and (y.dtype == bool):
            return tf.logical_and(x, y)
        
        if (x.dtype == bool):
            x = tf.cast(x, y.dtype)
        
        if (y.dtype == bool):
            y = tf.cast(y, x.dtype)
        
    except:
        if x is True:
            return y
        if y is True:
            return x
        if x is False or y is False:
            return False
    return tf.multiply(x, y)

def new_add(x, y):
    try:
        if (x.dtype == bool) and (y.dtype == bool) :
            return tf.logical_or(x, y)
        
        if isinstance(x, tf.Tensor) and x.dtype == bool:
            x = tf.cast(x, dtype)
            
        if isinstance(y, tf.Tensor) and y.dtype == bool:
            y = tf.cast(y, dtype)
    except:
        if x == False:
            return y
        if y == False:
            return x
        if x == True or y == True:
            return True

    return tf.add(x, y)

def new_neg(x):
    try:
        if (x.dtype == bool):
            return tf.logical_not(x)
    except:
        if x == False:
            return True
        if x == True:
            return False
        
    return tf.negative(x)
    

tf.Tensor.__mul__ = new_mul
tf.Tensor.__rmul__ = new_mul
tf.Tensor.__add__ = new_add
tf.Tensor.__radd__ = new_add
tf.Tensor.__neg__ = new_neg

def select_smooth(a, b, c):
    return a * b + (1 - a) * c

def select_nosmooth(a, b, c):

    all_numerical = True
    base_tensor = None
    count = 0
    for tensor in [a, b, c]:
        if not isinstance(tensor, (int, float, bool, np.bool_, np.ndarray)):
            all_numerical = False
            base_tensor = tensor
            count += 1
    
    if all_numerical:
        if isinstance(a, np.ndarray):
            return np.where(a, b, c)
        else:
            return b if a else c

    if base_tensor is a and count == 1:
        if isinstance(b, (bool, np.bool_)) and isinstance(c, (bool, np.bool_)):
            actual_dtype = tf.bool
        else:
            actual_dtype = dtype
    elif base_tensor is not a:
        actual_dtype = base_tensor.dtype
    else:
        if isinstance(b, tf.Tensor):
            actual_dtype = b.dtype
        else:
            actual_dtype = c.dtype

    if isinstance(b, (int, float, bool)):
        if b == 0.0:
            b = tf.zeros_like(base_tensor, dtype=actual_dtype)
        else:
            if actual_dtype == tf.bool:
                b = tf.ones_like(base_tensor, dtype=actual_dtype)
            else:
                b = b * tf.ones_like(base_tensor, dtype=actual_dtype)
    if isinstance(c, (int, float, bool)):
        if c == 0.0:
            c = tf.zeros_like(base_tensor, dtype=actual_dtype)
        else:
            if actual_dtype == tf.bool:
                c = tf.ones_like(base_tensor, dtype=actual_dtype)
            else:
                c = c * tf.ones_like(base_tensor, dtype=actual_dtype)
        
    return tf.where(tf.cast(a, bool), b, c)

select = select_nosmooth

def tf_fract(x):
    return tf.floormod(x, 1.0)

def tf_np_wrapper(func):
    def f(x, y=None):
                
        if func == 'sign_up':
            if isinstance(x, tf.Tensor):
                return 2.0 * tf.cast(x >= 0.0, x.dtype) - 1.0
            else:
                return 2.0 * float(x >= 0.0) - 1.0
        elif func == 'sign_down':
            if isinstance(x, tf.Tensor):
                return 2.0 * tf.cast(x > 0.0, x.dtype) - 1.0
            else:
                return 2.0 * float(x > 0.0) - 1.0
        elif func == 'random_normal':
            return tf.random_normal(tf.shape(x), dtype=x.dtype)
        elif func == 'equal':
            return tf.equal(x, y)
        elif func == 'nequal':
            return tf.math.logical_not(tf.equal(x, y))
        elif func == 'expand_1D':
            return tf.expand_dims(x, 1)

        if isinstance(x, (tf.Tensor, tf.Variable)) or isinstance(y, (tf.Tensor, tf.Variable)):
            if func == 'fmod':
                actual_func = tf.floormod
            else:
                actual_func = getattr(tf, func)
        else:
            try:
                actual_func = getattr(np, func)
            except:
                actual_func = getattr(math, func)
        if y is None:
            return actual_func(x)
        else:
            return actual_func(x, y)

    if tag_discontinuous:
        def f_tag(x, y=None):
            ans = f(x, y)
            if func in ['sign', 'sign_up', 'sign_down', 'floor', 'ceil']:
                ans.discontinuous = True
            return ans
        return f_tag
    else:
        return f
    
wrapper = tf_np_wrapper
    
def get_neighbor(*args):
    
    is_batch = False
        
    if len(args) == 2:
        node = args[0]
        pix_idx = args[1]
    elif len(args) == 3:
        node = args[0]
        pix_idx = args[1]
        is_batch = args[2]
    elif len(args) >= 5:
        buffer = args[0]
        pix_idx = args[1]
        read_idx = args[2]
        node = buffer[read_idx]
    else:
        raise 'Unknown signature to get_neighbor'

    if pix_idx == 0:
        return node
    elif pix_idx == 1:
        return tf.roll(node, -1, axis=-1)
    elif pix_idx == 2:
        return tf.roll(node, 1, axis=-1)
    elif pix_idx == 3:
        return tf.roll(node, -1, axis=-2)
    elif pix_idx == 4:
        return tf.roll(node, 1, axis=-2)
    elif pix_idx == 5:
        return tf.roll(node, -1, axis=-3)
    elif pix_idx == 6:
        return tf.roll(node, 1, axis=-3)
    else:
        raise
        
def get_partial_trace_coord(*args):
    
    is_batch = False
    
    if len(args) == 2:
        node = args[0]
        pix_idx = args[1]
    elif len(args) == 3:
        node = args[0]
        pix_idx = args[1]
        is_batch = args[2]
    elif len(args) >= 5:
        buffer = args[0]
        pix_idx = args[1]
        read_idx = args[2]
        node = buffer[read_idx]
    else:
        raise 'Unknown signature to get_neighbor'
    
    ans = tf.cast(node, tf.float32) - tf.cast(get_neighbor(node, pix_idx, is_batch), tf.float32)
    
    if pix_idx in [1, 3, 5]:
        ans = -ans
    
    return ans
        
def cast2f(node):
    return tf.cast(node, tf.float32)

def cast2b(node):
    if isinstance(node, (bool, np.bool_)):
        return node
    return tf.cast(node, tf.bool)