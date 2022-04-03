
from render_util import *
import importlib
import time
import sys
import json

def trace(frame, event, arg):
    #print("%s, %s:%d" % (event, frame.f_code.co_filename, frame.f_lineno))
    if frame.f_code.co_filename.endswith('compiler.py'):
        print("%s:%d" % (event, frame.f_lineno))
    return trace

#sys.settrace(trace)

def render_single(base_dir, shader_name, args, outdir=None, render_size=None):
    """
    Generate AST and code for a single shader
    """

    T0 = time.time()

    kw = {}
        
    if '--backend' in args:
        backend_idx = args.index('--backend')
        backend = args[backend_idx+1]
        kw['backend'] = backend
        
    if '--debug_ast' in args:
        kw['debug_ast'] = True
        
    if '--compiler_modes' in args:
        compiler_modes_idx = args.index('--compiler_modes')
        compiler_modes = args[compiler_modes_idx+1].split(',')
        kw['compiler_modes'] = compiler_modes
        if 'bw' not in args[compiler_modes_idx+1]:
            kw['compute_g'] = False
        
    if '--use_select_rule' in args:
        select_rule_idx = args.index('--use_select_rule')
        select_rule = int(args[select_rule_idx+1])
        kw['select_rule'] = select_rule
        
    if '--use_multiplication_rule' in args:
        multiplication_rule_idx = args.index('--use_multiplication_rule')
        multiplication_rule = int(args[multiplication_rule_idx+1])
        kw['multiplication_rule'] = multiplication_rule
        
    if '--no_compute_g' in args:
        kw['compute_g'] = False
        
    if '--autoscheduler' in args:
        kw['autoscheduler'] = True
        
    if '--do_prune' in args:
        do_prune_idx = args.index('--do_prune')
        kw['do_prune'] = [bool(int(val)) for val in args[do_prune_idx+1].split(',')]
        
    if '--par_file' in args:
        par_file_idx = args.index('--par_file')
        kw['par_vals'] = np.load(args[par_file_idx+1])
        
    if '--AD_only' in args:
        kw['AD_only'] = True
        
    if '--allow_raymarching_random' in args:
        kw['allow_raymarching_random'] = True
        
    m = importlib.import_module(shader_name)
    
    shaders = m.shaders
    
    if '--shader_args' in args:
        shader_args_idx = args.index('--shader_args')
        shader_args = args[shader_args_idx + 1].split('#')
        
        for line in shader_args:
            name, val = line.split(':')
            val = eval(val)
            setattr(m, name, val)
            
        if hasattr(m, 'update_args'):
            m.update_args()

    if hasattr(m, 'nargs'):
        kw['input_nargs'] = m.nargs
        
    if hasattr(m, 'args_range'):
        kw['args_range'] = m.args_range
        
    if hasattr(m, 'sigmas_range'):
        kw['sigmas_range'] = m.sigmas_range
        
    
        
    ans = multiple_shaders()(shaders, base_dir=base_dir, is_color=m.is_color, **kw)
    T1 = time.time()
    
    return ans

def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print('python render_single.py base_dir shadername [--no-ground-truth] [--novel-camera-view] [--time-error] [--camera-path i]')
        print('  Renders a single shader by name, with specified geometry and normal map.')
        sys.exit(1)

    (base_dir, shader_name) = args[:2]

    render_single(base_dir, shader_name, args)

if __name__ == '__main__':
    main()
