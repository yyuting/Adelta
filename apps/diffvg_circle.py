# python diffvg_circle.py --dir /n/fs/scratch/yutingy/diffvg/results/single_circle --init_values_pool ../test_finite_diff_circle_extra_init_values.npy

# python diffvg_circle.py --dir /n/fs/scratch/yutingy/diffvg/results/single_circle_nsamples_2 --init_values_pool ../test_finite_diff_circle_extra_init_values.npy --nsamples 2

import pydiffvg
import torch
import skimage
import numpy as np
import os
import time
import platform
render = pydiffvg.RenderFunction.apply
import sys

# Use GPU if available
pydiffvg.set_use_gpu(torch.cuda.is_available())

def circle():
    circle = pydiffvg.Circle(radius = torch.tensor(40.0),
                         center = torch.tensor([128.0, 128.0]))
    shapes = [circle]
    circle_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([0]),
        fill_color = torch.tensor([1.0, 0.8, 0.8, 1.0]))
    shape_groups = [circle_group]
    
    return shapes, shape_groups

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
                
    assert '--dir' in sys.argv
    dir_idx = sys.argv.index('--dir')
    outdir = sys.argv[dir_idx + 1]
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    
    assert '--init_values_pool' in sys.argv
    init_values_pool_idx = sys.argv.index('--init_values_pool')
    init_values_pool = np.load(sys.argv[init_values_pool_idx + 1])
    
    assert '--gt_file' in sys.argv
    gt_file_idx = sys.argv.index('--gt_file')
    gt_file = sys.argv[gt_file_idx + 1]
    
    shapes, shape_groups = circle()
    
    target = torch.tensor(skimage.img_as_float(skimage.io.imread(gt_file))).to(device='cuda')
    
    # Move the circle to produce initial guess
    # normalize radius & center for easier learning rate
    radius_n = torch.tensor(20.0 / 256.0, requires_grad=True)
    center_n = torch.tensor([108.0 / 256.0, 138.0 / 256.0], requires_grad=True)
    color = torch.tensor([0.3, 0.2, 0.8, 1.0], requires_grad=True)

    init_values_pool[:, :3] /= 256.
    
    niters = 100

    all_loss = np.zeros((init_values_pool.shape[0], niters))
    logfile = open(os.path.join(outdir, 'log.txt'), 'a+')

    T0 = time.time()
    
    for idx in range(init_values_pool.shape[0]):
    
        radius_n.data = torch.tensor(init_values_pool[idx, 0])
        center_n.data = torch.tensor(init_values_pool[idx, 1:3])
        color.data = torch.tensor(init_values_pool[idx, 3:])

        shapes[0].radius = radius_n * 256
        shapes[0].center = center_n * 256
        shape_groups[0].fill_color = color
        scene_args = pydiffvg.RenderFunction.serialize_scene(\
            res_x, res_y, shapes, shape_groups)
        img = render(res_x, # width
                     res_y, # height
                     nsamples,   # num_samples_x
                     nsamples,   # num_samples_y
                     1,   # seed
                     None,
                     *scene_args)

        img = img[..., :-1] * img[..., -1:]
        #pydiffvg.imwrite(img.cpu(), os.path.join(outdir, 'init%d.png' % idx))
        skimage.io.imsave(os.path.join(outdir, 'init%d.png' % idx), img.cpu().detach().numpy())

        # Optimize for radius & center & color
        optimizer = torch.optim.Adam([radius_n, center_n, color], lr=1e-2)
        
        min_loss_val = 1e8
        min_loss_par = None

        # Run 100 Adam iterations.
        for t in range(niters):        
            optimizer.zero_grad()
            # Forward pass: render the image.
            shapes[0].radius = radius_n * 256
            shapes[0].center = center_n * 256
            shape_groups[0].fill_color = color
            scene_args = pydiffvg.RenderFunction.serialize_scene(\
                res_x, res_y, shapes, shape_groups)
            img = render(res_x,   # width
                         res_y,   # height
                         nsamples,     # num_samples_x
                         nsamples,     # num_samples_y
                         t+1,   # seed
                         None,
                         *scene_args)
            img = img[..., :-1] * img[..., -1:]

            # Compute the loss function. Here it is L2.
            # need x 0.5 to match that in our framework
            loss = (img - target).pow(2).mean() * 0.5

            # Backpropagate the gradients.
            loss.backward()

            # Take a gradient descent step.
            optimizer.step()

            # in our framework, we always save loss AFTER optimizer.step
            # we emulate the behavior here, but also gives an EXTRA benefit to diffvg to save the time of running another forward pass
            if t > 0:
                all_loss[idx, t - 1] = loss.cpu()
                
            if loss < min_loss_val:
                min_loss_par = [float(radius_n), center_n.detach().numpy().copy(), color.detach().numpy().copy()]
                min_loss_val = float(loss)

        # Render the final result.
        shapes[0].radius.data = torch.tensor(min_loss_par[0] * 256)
        shapes[0].center.data = torch.tensor(min_loss_par[1] * 256)
        shape_groups[0].fill_color = torch.tensor(min_loss_par[2])
        scene_args = pydiffvg.RenderFunction.serialize_scene(\
            res_x, res_y, shapes, shape_groups)

        img = render(res_x,   # width
                     res_y,   # height
                     nsamples,     # num_samples_x
                     nsamples,     # num_samples_y
                     niters + 1,    # seed
                     None,
                     *scene_args)

        img = img[..., :-1] * img[..., -1:]
        loss = (img - target).pow(2).mean() * 0.5

        all_loss[idx, -1] = loss.cpu()

        #pydiffvg.imwrite(img.cpu(), os.path.join(outdir, 'final%d.png' % idx))
        skimage.io.imsave(os.path.join(outdir, 'final%d.png' % idx), img.cpu().detach().numpy())
        print(idx, loss)
        print(idx, float(loss), file=logfile)

    T1 = time.time()
    
    np.save(os.path.join(outdir, 'all_loss.npy'), all_loss)

    logfile = open(os.path.join(outdir, 'log.txt'), 'a+')

    print('total runtime', T1 - T0, file=logfile)
    print('total iterations', niters * init_values_pool.shape[0], file=logfile)
    print('runtime per iter', (T1 - T0) / (niters * init_values_pool.shape[0]), file=logfile)
    logfile.close()
    
if __name__ == '__main__':
    main()
    
    
    





