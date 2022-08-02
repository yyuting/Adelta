import os

def set_argument(args, dest, shader, backend):
    args.dir = dest
    args.shader = shader
    args.backend = backend
    args.show_progress = True
    if backend == 'hl':
        args.gt_transposed = True
    else:
        args.gt_transposed = False
        
    if args.shader == 'siggraph':
        args.init_values_pool = 'apps/example_init_values/test_finite_diff_siggraph_cone_init_values_pool.npy'
        args.gt_file = 'siggraph_gradient.png'
        args.render_size = '960,960'
        opt = True
    elif args.shader.startswith('celtic_knot'):
        args.render_size = '640,640'
        args.gt_file = 'celtic_knot.png'
        if args.shader == 'celtic_knot':
            opt = True
            if getattr(args, 'extra_runs', False):
                args.init_values_pool = 'apps/example_init_values/test_finite_diffring_contour_extra_init_values_pool.npy'
            else:
                args.init_values_pool = 'apps/example_init_values/test_finite_diffring_contour_init_values_pool.npy'
        elif args.shader == 'celtic_knot2':
            opt = False
            args.init_values_pool = os.path.join(os.path.abspath('../celtic_knot'), 'best_par.npy')
            
    if opt:
        args.modes = 'optimization'
        args.metrics = '5_scale_L2'
        args.smoothing_sigmas = '0.5,1,2,5'
        args.learning_rate = 0.01
        args.multi_scale_optimization = True
        args.alternating_times = 5
        args.tunable_param_random_var = True
        args.tunable_param_random_var_opt = True
        args.tunable_param_random_var_seperate_opt = True
        args.tunable_param_random_var_std = 1.
        args.save_all_loss = True
        args.reset_opt_each_scale = False
        args.reset_sigma = False
        if args.backend in ['tf', 'torch']:
            args.verbose = True
        else:
            args.verbose = False
        args.save_best_par = True
    else:
        args.modes = 'render'