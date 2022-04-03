# python diffvg_ring.py --dir /n/fs/scratch/yutingy/diffvg/results/single_circle_outline_nsamples_2 --init_values_pool ../test_finite_diff_ring_extra_init_values.npy --nsamples 2

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
    
    save_dir = outdir

    canvas_width, canvas_height = 256, 256
    shapes, shape_groups = ring()

    circle = shapes[0]
    circle_group = shape_groups[0]

    target = torch.tensor(skimage.img_as_float(skimage.io.imread(gt_file))).to(device='cuda')

    # Move the circle to produce initial guess
    # normalize radius & center for easier learning rate
    radius_n = torch.tensor(50.0 / 256.0, requires_grad=True)
    center_n = torch.tensor([128.0 / 256.0, 128.0 / 256.0], requires_grad=True)
    stroke_color = torch.tensor([0.6, 0.3, 0.6, 0.8], requires_grad=True)
    stroke_width_n = torch.tensor(5.0 / 100.0, requires_grad=True)

    init_values_pool[:, :3] /= 256.
    init_values_pool[:, 3] /= 100

    niters = 100

    all_loss = np.zeros((init_values_pool.shape[0], niters))
    
    logfile = open(os.path.join(save_dir, 'log.txt'), 'a+')

    T0 = time.time()

    for idx in range(init_values_pool.shape[0]):

        radius_n.data = torch.tensor(init_values_pool[idx, 0])
        center_n.data = torch.tensor(init_values_pool[idx, 1:3])
        stroke_width_n.data = torch.tensor(init_values_pool[idx, 3])
        stroke_color.data = torch.tensor(init_values_pool[idx, 4:])

        circle.radius = radius_n * 256
        circle.center = center_n * 256
        circle.stroke_width = stroke_width_n * 100
        circle_group.stroke_color = stroke_color

        scene_args = pydiffvg.RenderFunction.serialize_scene(\
            canvas_width, canvas_height, shapes, shape_groups)
        img = render(256, # width
                     256, # height
                     nsamples,   # num_samples_x
                     nsamples,   # num_samples_y
                     1,   # seed
                     None,
                     *scene_args)
        img = img[..., :3] * img[..., -1:]
        skimage.io.imsave(os.path.join(outdir, 'init%d.png' % idx), img.cpu().detach().numpy())

        min_loss_val = 1e8
        min_loss_par = None

        # Optimize for radius & center
        optimizer = torch.optim.Adam([radius_n, center_n, stroke_color, stroke_width_n], lr=1e-2)
        # Run 200 Adam iterations.
        for t in range(niters):
            optimizer.zero_grad()

            # Forward pass: render the image.
            if radius_n < 0:
                # avoids rendering error
                radius_n.data = torch.tensor(np.float64(0.1))

            circle.radius = radius_n * 256
            circle.center = center_n * 256
            circle.stroke_width = stroke_width_n * 100
            circle_group.stroke_color = stroke_color

            scene_args = pydiffvg.RenderFunction.serialize_scene(
                canvas_width, canvas_height, shapes, shape_groups)
            img = render(256,   # width
                         256,   # height
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

            optimizer.step()

            # in our framework, we always save loss AFTER optimizer.step
            # we emulate the behavior here, but also gives EXTRA benefit to diffvg to save another forward pass
            if t > 0:
                all_loss[idx, t - 1] = loss.cpu()

            if loss < min_loss_val:
                min_loss_par = [float(radius_n), center_n.detach().numpy().copy(), float(stroke_width_n), stroke_color.detach().numpy().copy()]
                min_loss_val = float(loss)

        # Render the final result.
        # extra safeguard on radius
        circle.radius.data = torch.tensor(max(min_loss_par[0], 0.1) * 256)
        circle.center.data = torch.tensor(min_loss_par[1] * 256)
        circle.stroke_width.data = torch.tensor(min_loss_par[2] * 100)
        circle_group.stroke_color.data = torch.tensor(min_loss_par[3])

        scene_args = pydiffvg.RenderFunction.serialize_scene(\
            canvas_width, canvas_height, shapes, shape_groups)
        img = render(256,   # width
                     256,   # height
                     nsamples,     # num_samples_x
                     nsamples,     # num_samples_y
                     niters + 1,    # seed
                     None,
                     *scene_args)

        img = img[..., :-1] * img[..., -1:]
        loss = (img - target).pow(2).mean() * 0.5

        all_loss[idx, -1] = loss.cpu()

        skimage.io.imsave(os.path.join(outdir, 'final%d.png' % idx), img.cpu().detach().numpy())
        print(idx, loss)
        print(idx, loss, file=logfile)

    T1 = time.time()

    np.save(os.path.join(save_dir, 'all_loss.npy'), all_loss)

    

    print('running on', platform.node(), file=logfile)
    print('total runtime', T1 - T0, file=logfile)
    print('total iterations', niters * init_values_pool.shape[0], file=logfile)
    print('runtime per iter', (T1 - T0) / (niters * init_values_pool.shape[0]), file=logfile)
    logfile.close()
    
    # get a clear graph for computing the gradient map
    shapes, shape_groups = ring()
    
    radius_val = torch.tensor(80., requires_grad=True)
    
    img = get_img(shapes, shape_groups, res_x, res_y, radius_val, nsamples=nsamples).cpu().detach().numpy().copy()[..., 0]
    skimage.io.imsave(os.path.join(outdir, 'img.png'), img)
    
    radius_val.data += 1.
    img_pos = get_img(shapes, shape_groups, res_x, res_y, radius_val, nsamples=nsamples).cpu().detach().numpy().copy()[..., 0]
    radius_val.data -= 2.
    img_neg = get_img(shapes, shape_groups, res_x, res_y, radius_val, nsamples=nsamples).cpu().detach().numpy().copy()[..., 0]

    fd_wrt_radius = (img_pos - img_neg) / 4.
    
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
   
    np.save(os.path.join(outdir, 'diffvg_gradient_wrt_radius.npy'), diffvg_wrt_radius)
    np.save(os.path.join(outdir, 'fd_gradient_wrt_radius.npy'), fd_wrt_radius)

if __name__ == '__main__':
    main()