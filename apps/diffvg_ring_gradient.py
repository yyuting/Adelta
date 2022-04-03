import pydiffvg
import diffvg
import torch
import skimage
import numpy as np
import os
import time
import platform
import sys

# Use GPU if available
pydiffvg.set_use_gpu(torch.cuda.is_available())
render = pydiffvg.RenderFunction.apply

def ring():
    circle = pydiffvg.Circle(radius = torch.tensor(80.0),
                             center = torch.tensor([128.0, 128.0]),
                             stroke_width = torch.tensor(16.0))
    
    shapes = [circle]
    circle_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([0]),
        fill_color = None,
        stroke_color = torch.tensor([1.0, 0.3, 0.6, 1.0]))
    shape_groups = [circle_group]
    
    return shapes, shape_groups

def get_img(shapes, shape_groups, width, height, radius_val, nsamples=2, seed=None):

    shapes[0].radius = radius_val
    
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        width, height, shapes, shape_groups)
    
    img = render(width, # width
                 height, # height
                 nsamples,   # num_samples_x
                 nsamples,   # num_samples_y
                 seed if seed is not None else 0,   # seed
                 None,
                 *scene_args)
    
    return img

def main():
    
    if '--res_x' in sys.argv:
        res_x_idx = sys.argv.index('--res_x')
        res_x = int(sys.argv[res_x_idx + 1])
    else:
        res_x = 256
        
    if '--res_y' in sys.argv:
        res_y_idx = sys.argv.index('--res_y')
        res_y = int(sys.argv[res_y_idx + 1])
    else:
        res_y = 256
        
    if '--nsamples' in sys.argv:
        nsamples_idx = sys.argv.index('--nsamples')
        nsamples = int(sys.argv[nsamples_idx + 1])
    else:
        nsamples = 2
        
    canvas_width = res_x
    canvas_height = res_y
    
    outdir = ''
        
    save_dir = outdir

    canvas_width, canvas_height = 256, 256
    
    shapes, shape_groups = ring()
    
    radius_val = torch.tensor(80., requires_grad=True)
    
    img = get_img(shapes, shape_groups, res_x, res_y, radius_val, nsamples=nsamples).cpu().detach().numpy().copy()[..., 0]
    skimage.io.imsave(os.path.join(outdir, 'img.png'), img)
    
    radius_val.data += 1.
    img_pos = get_img(shapes, shape_groups, res_x, res_y, radius_val, nsamples=nsamples).cpu().detach().numpy().copy()[..., 0]
    radius_val.data -= 2.
    img_neg = get_img(shapes, shape_groups, res_x, res_y, radius_val, nsamples=nsamples).cpu().detach().numpy().copy()[..., 0]

    fd_wrt_radius = (img_pos - img_neg) / 2.
    
    skimage.io.imsave(os.path.join(outdir, 'fd_gradient_wrt_radius.png'), fd_wrt_radius)
    
    sparse_x, sparse_y = np.where(fd_wrt_radius != 0)
    
    # only compute gradient of R channel wrt radius
    diffvg_wrt_radius = np.zeros((res_x, res_y))

    radius_val.data += 1.
    for i in range(sparse_x.shape[0]):
        if radius_val.grad is not None:
            radius_val.grad.data.zero_()
        get_img(shapes, shape_groups, res_x, res_y, radius_val, nsamples=nsamples)[sparse_x[i], sparse_y[i], 0].backward()
        diffvg_wrt_radius[sparse_x[i], sparse_y[i]] = radius_val.grad
        
        if i % 100 == 0:
            print(i)
        
    # use skimage to avoid negative value being clipped
    
    skimage.io.imsave(os.path.join(outdir, 'diffvg_gradient_wrt_radius.png'), diffvg_wrt_radius)

if __name__ == '__main__':
    main()