import torch
import numpy as np
import numpy
import math

dtype = np.float32

tag_discontinuous = False

def select_nosmooth(a, b, c):
    
    if isinstance(a, bool):
        if a:
            return b
        else:
            return c
    
    if isinstance(b, (int, float)) and not isinstance(b, bool):
        b = torch.tensor(b).cuda().float()
    if isinstance(c, (int, float)) and not isinstance(c, bool):
        c = torch.tensor(c).cuda().float()
    
    return torch.where(a, b, c)

select = select_nosmooth

def wrapper(func):
    def f(x, y=None):
        
        actual_func = getattr(torch, func)
                
        if y is None:
            return actual_func(x)
        else:
            return actual_func(x, y)

    return f
    
def get_neighbor(*args):
    
    if len(args) == 2:
        node = args[0]
        pix_idx = args[1]
        
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
        return torch.roll(node, -1, dims=2)
    elif pix_idx == 2:
        return torch.roll(node, 1, dims=2)
    elif pix_idx == 3:
        return torch.roll(node, -1, dims=1)
    elif pix_idx == 4:
        return torch.roll(node, 1, dims=1)
    else:
        raise
        
def get_partial_trace_coord(*args):
    
    if len(args) == 2:
        node = args[0]
        pix_idx = args[1]
        
    elif len(args) >= 5:
        buffer = args[0]
        pix_idx = args[1]
        read_idx = args[2]
        node = buffer[read_idx]
    else:
        raise 'Unknown signature to get_neighbor'
    
    ans = node.float() - get_neighbor(node, pix_idx).float()
    
    if pix_idx in [1, 3]:
        ans = -ans
    
    return ans
        
def cast2f(node):
    return node.float()

def cast2b(node):
    if isinstance(node, (bool, np.bool_)):
        return node
    return node.bool()