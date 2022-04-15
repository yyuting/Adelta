import os
import sys
sys.path += ['../util']
import argparse_util
import importlib

default_path = '/n/fs/scratch/yutingy'

def exec_parent_dir(cmd):
    cwd = os.getcwd()
    os.chdir('../')
    os.system(cmd)
    os.chdir(cwd)
        
def main():
    
    parser = argparse_util.ArgumentParser(description='run shader')
    parser.add_argument('shader_file', help='shader file to run')
    parser.add_argument('--dir', dest='dir', default=default_path, help='directory to save result')
    parser.add_argument('--mode', dest='mode', default='visualize_gradient', choices=['visualize_gradient', 'optimization'], help='what mode to run the shader, empty means execute all the command written at the beginning of the shader file')
    parser.add_argument('--backend', dest='backend', default='hl', choices=['hl', 'tf', 'torch'], help='specifies backend')
    parser.add_argument('--gradient_method', dest='gradient_method', default='ours', choices=['ours', 'fd', 'spsa'], help='specifies what gradient approximation to use')
    parser.add_argument('--finite_diff_h', dest='finite_diff_h', type=float, default=0.01, help='step size for finite diff')
    parser.add_argument('--spsa_samples', dest='spsa_samples', type=int, default=1, help='number of samples for spsa')
    parser.add_argument('--use_random', dest='use_random', action='store_true', help='if running optimization cmd, apply random variables to parameters')
    parser.add_argument('--use_autoscheduler', dest='use_autoscheduler', action='store_true', help='if in hl backend, use autoscheduler instead of naive schedule')
    
    args = parser.parse_args()
        
    if not args.shader_file.startswith('render_'):
        print('Error: shader file has to be saved in apps directory and needs to start with render_')
        raise

    if not args.shader_file.endswith('.py'):
        print('Error: shader file has to be .py')
        raise

    if args.backend in ['tf', 'torch'] and args.gradient_method != 'ours':
        print('TF or Torch with FD or SPSA not finished')
        raise

    spec = importlib.util.spec_from_file_location("module.name", args.shader_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, 'cmd_check'):
        module.cmd_check(args.backend)

    if args.gradient_method in ['fd', 'spsa']:
        gradient_str = f"""finite_diff --finite_diff_h {args.finite_diff_h}"""
        if args.gradient_method == 'spsa':
            gradient_str += f""" --finite_diff_spsa_samples {args.spsa_samples}"""
    else:
        gradient_str = 'ours'

    cmd = module.cmd_template()

    if args.use_random:
        cmd += ' --tunable_param_random_var --tunable_param_random_var_opt --tunable_param_random_var_seperate_opt --tunable_param_random_var_opt_scheduling all --tunable_param_random_var_std 1 --no_binary_search_std'

    if args.use_autoscheduler:
        cmd += ' --autoscheduler'

    if args.backend == 'hl':
        cmd += ' --gt_transposed'

    cmd += f""" --backend {args.backend} --dir {args.dir} --modes {args.mode} --gradient_methods_optimization {gradient_str} --learning_rate 0.01 --finite_diff_both_sides --no_reset_sigma --no_reset_opt"""

    print(cmd)
    exec_parent_dir(cmd)
        
if __name__ == '__main__':
    main()
    