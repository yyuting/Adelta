"""
Modified from
https://github.com/ChezJrk/Teg/blob/master/tests/plot.py
"""

from teg.eval import evaluate
import numpy as np
import png
import time


def render_image(expr, variables=(), res=(64, 64), tile_offsets=[0, 0], kernel_size=[1, 1], bindings={}, silent=False, nsamples=1):
    assert len(variables) == 2, 'Exactly two variable-pairs required'

    image = np.zeros(res)
    
    for nx in range(res[0]):
        for ny in range(res[1]):
            x_lb = nx + tile_offsets[0] - kernel_size[0]
            x_ub = nx + tile_offsets[0] + kernel_size[0]
            y_lb = ny + tile_offsets[1] - kernel_size[1]
            y_ub = ny + tile_offsets[1] + kernel_size[1]
            
            value = evaluate(expr, bindings={**bindings,
                                             variables[0][0]: x_lb,
                                             variables[0][1]: x_ub,
                                             variables[1][0]: y_lb,
                                             variables[1][1]: y_ub},
                             num_samples=nsamples, backend='C_PyBind')
            
            image[nx, ny] = value
                
            if nx == 0 and ny == 0:
                # ignore first pixel because compilation can take longer
                T0 = time.time()

    T1 = time.time()
    
    print('total runtime: ', T1 - T0)
    
    return image


def save_image(image, filename):
    image = image.T.copy()
    image = ((image / np.max(image)) * 255.0).astype(np.uint8)
    png.from_array(image, mode="L").save(filename)
