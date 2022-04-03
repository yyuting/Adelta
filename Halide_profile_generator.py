import sys
import os
import numpy as np
import argparse
import sys
import operator

class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kw):
        kw['formatter_class'] = SortingHelpFormatter
        argparse.ArgumentParser.__init__(self, *args, **kw)
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(1)

class SortingHelpFormatter(argparse.HelpFormatter):
    def add_arguments(self, actions):
        actions = sorted(actions, key=operator.attrgetter('option_strings'))
        super(SortingHelpFormatter, self).add_arguments(actions)

def get_exec_cpp_name(name, target_dir, height, width, niters):
    
    exec_name_short = '%s_%dX%dX%drun' % (name.split(',')[0], height, width, niters)
    
    cpp_name_short = '%s.cpp' % exec_name_short
    
    exec_name = os.path.join(target_dir, exec_name_short)
    cpp_name = os.path.join(target_dir, cpp_name_short)
    return exec_name_short, exec_name, cpp_name_short, cpp_name

def main():
    
    
    parser = ArgumentParser(description='approx gradient')
    parser.add_argument('--lib_name', dest='lib_name', default='', help='specifies the name for .a file')
    parser.add_argument('--p_bound', dest='p_bound', type=int, default=0, help='expected output channel')
    parser.add_argument('--width', dest='width', type=int, default=960, help='specifies the width of output')
    parser.add_argument('--height', dest='height', type=int, default=640, help='specifies the height of output')
    parser.add_argument('--niters', dest='niters', type=int, default=10, help='specifies number of iterations for benchmarking')
    parser.add_argument('--init_values_file', dest='init_values_file', default='', help='specifies the npy file that stores initial values')
    parser.add_argument('--nparams', dest='nparams', type=int, default=0, help='specifies number of input paramters')
    parser.add_argument('--target_dir', dest='target_dir', default='./', help='specifies the directory to that lib file stores and all the tmp files are written to')
    parser.add_argument('--input_dims', dest='input_dims', default='', help='specifies the number of input dimension needed')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='if true, verbosely printout lots of output for debug')
    parser.add_argument('--producer_pad', dest='producer_pad', type=int, default=0, help='specifies the padding on each side the producer needs to make')
    parser.add_argument('--input_vals', dest='input_vals', default='0', help='if producer not available, use this val to feed input buffer')
    parser.add_argument('--uv_offset', dest='uv_offset', default='0,0', help='specifies uv offset, seperated by comma')
    parser.add_argument('--additional_sigma', dest='additional_sigma', default='', help='inputs as additional sigma value')
    parser.add_argument('--use_frame_idx', dest='use_frame_idx', action='store_true', help='if specified, add additional input after height, which specifies frame index that controls random noise')
    parser.add_argument('--frame_idx', dest='frame_idx', type=int, default=0, help='specifies the frame idx used to control random value')
    parser.add_argument('--render_output', dest='render_output', action='store_true', help='if specified, write output to disk for sanity check')
    
    args = parser.parse_args()

    if args.init_values_file != '':
        init_values = np.load(args.init_values_file)[0]
    else:
        init_values = np.random.rand(args.nparams)
    cpp_code = run_code_generator(args, init_values)
    exec_name_short, exec_name, cpp_name_short, cpp_name = get_exec_cpp_name(args.lib_name, 
                                                                             args.target_dir, 
                                                                             args.width, 
                                                                             args.height, 
                                                                             args.niters)
    
    open(cpp_name, 'w').write(cpp_code)
    
    cmd = cmd_generator(args, exec_name_short, cpp_name_short)
        
    os.system(cmd)
    
def cmd_generator(args, exec_name_short, cpp_name):
    
    lib_name = args.lib_name
    target_dir = args.target_dir

    if ',' in lib_name:
        lib_name = lib_name.split(',')
    else:
        lib_name = [lib_name]
        
    lib_a = ""
    for name in lib_name:
        lib_a += f""" {name}.a """
        
    if args.render_output:
        render_flags = '-ljpeg -L /n/fs/shaderml/local_build/build_libpng/lib `libpng-config --cflags --ldflags`'
    else: 
        render_flags = ''
         
    cwd = os.getcwd()
    
    cmd = f"""
cd {target_dir}; g++ {cpp_name} -I $HALIDE_INCLUDE_PATH -I $HALIDE_TOOL_PATH {lib_a} runtime.a -ldl -lpthread -o {exec_name_short} {render_flags}; ./{exec_name_short}; cd {cwd}
"""
    
    if args.verbose:
        print(cmd.replace(';', '\n'))
    
    return cmd
    
def run_code_generator(args, init_values):
    
    lib_name = args.lib_name
    p_bound = args.p_bound
    width = args.width
    height = args.height
    niters = args.niters
    input_dims = args.input_dims
    
    if ',' in lib_name:
        lib_name = lib_name.split(',')
    else:
        lib_name = [lib_name]
        
    include_lib_str = ""
    for name in lib_name:
        include_lib_str += f"""
#include "{name}.h"
        """
        
    param_arguments = ','.join(['%ff' % val for val in init_values])
            
    uv_strs = args.uv_offset.split(',')[:2]

    for uv_str in uv_strs:
        param_arguments += ', '
        param_arguments += uv_str
        if '.' not in uv_str:
            param_arguments += '.f'
        else:
            param_arguments += 'f'
                
    if len(param_arguments):
        param_arguments += ','
            
    if args.use_frame_idx:
        frame_idx_str = ', %d' % args.frame_idx
    else:
        frame_idx_str = ''
        
    device_sync_str = ""
    device_free_str = ""
        
    if input_dims != '':
        
        producer_w = width + 2 * args.producer_pad
        produce_h = height + 2 * args.producer_pad
        
        input_dims_vals = [int(val) for val in input_dims.split(',')]
        input_feed_vals = [float(val) for val in args.input_vals.split(',')]
        
        assert len(input_feed_vals) in [1, len(input_dims_vals)]

        input_types = ['float'] * len(input_dims_vals)
        
        
        
        input_def = ""
        input_arg = ""
        input_boundary = ""
        
        for idx in range(len(input_dims_vals)):
            
            current_dim = input_dims_vals[idx]
            current_type = input_types[idx]
            
            if current_dim > 0:
                last_dim_arg = ', %s' % current_dim
            else:
                last_dim_arg = ''
            
            input_def += f"""Halide::Runtime::Buffer<{current_type}> input{idx}({producer_w}, {produce_h}{last_dim_arg});"""
            
            if len(input_feed_vals) == 1:
                current_feed_val = input_feed_vals[0]
            else:
                current_feed_val = input_feed_vals[idx]
        
                
            if current_dim > 0:
                input_def += f"""
    for (int x = 0; x < {producer_w}; x++) {{
        for (int y = 0; y < {produce_h}; y++) {{
            for (int p = 0; p < {current_dim}; p++) {{
                input{idx}(x, y, p) = ({current_type}) ({current_feed_val});
            }}
        }}
    }}
                """

        
            input_arg += ", input%d" % idx
        
            device_sync_str += f"""
    input{idx}.device_sync();
            """
        
        
            if args.producer_pad > 0:
                input_boundary += f"""
    input{idx}.set_min(-{args.producer_pad}, -{args.producer_pad});
    """
    else:
        input_def = ""
        input_arg = ""
        input_boundary = ''
        
    gradients_declare_str = ""
    lib_call_str = ""
    
    spatial_args = f"""{width}, {height}{frame_idx_str}, """

    input_arg = input_arg[1:]

    if input_arg != '':
        input_arg += ', '
        
    for i in range(len(lib_name)):
        name = lib_name[i]

        gradients_declare_str += f"""
    Halide::Runtime::Buffer<float> gradients{i}(width, height, {p_bound});
            """

        lib_call_str += f"""
    {name}({param_arguments} {spatial_args}{args.additional_sigma}{input_arg} gradients{i});
            """

        device_sync_str += f"""
    gradients{i}.device_sync();
            """
        
    if args.render_output:
        name = 'gradients0'
        
        render_header = f"""
#include "halide_image_io.h"
        """
        
        if args.verbose:
            print_gmax_min = f"""printf("gmin, gmax for idx %d: %f, %f\\n", idx, g_min, g_max);"""
        else:
            print_gmax_min = ''
            
        render_code = f"""
    Halide::Runtime::Buffer<uint8_t> uint_vec_output({width}, {height});
        
    {name}.copy_to_host();
    
    float g_min = 0.f;
    float g_max = 1.f;
    
    for (int idx = 0; idx < {p_bound}; idx++) {{

        
        g_min = 100000000.f;
        g_max = -100000000.f;

        for (int y = 0; y < height; y++) {{
            for (int x = 0; x < width; x++) {{
                if ({name}(x, y, idx) < g_min) {{
                    g_min = {name}(x, y, idx);
                }}
                if ({name}(x, y, idx) > g_max) {{
                    g_max = {name}(x, y, idx);
                }}
            }}
        }}
        
        {print_gmax_min}
        
        for (int y = 0; y < height; y++) {{
            for (int x = 0; x < width; x++) {{
                uint_vec_output(x, y) = (uint8_t) (clamp(({name}(x, y, idx) - g_min) / (g_max - g_min) * 255.f, 0.f, 255.f));
            }}
        }}
        
        save_image(uint_vec_output, "visulize" + std::to_string(idx) + ".png");
    }}
"""
    else:
        render_header = ''
        render_code = ''
    
    cpp_code = f"""
{include_lib_str}

#include "HalideBuffer.h"
#include "halide_benchmark.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

{render_header}
using namespace Halide::Tools;

float clamp(float val, float lo, float hi) {{
    if (val < lo) return lo;
    else if (val > hi) return hi;
    else return val;
}}

bool f_close(float a, float b) {{
    // Emulates np.allclose

    float rtol = 0.01f;
    float atol = 0.01f;
    
    if (std::abs(a - b) <= atol + rtol * std::abs(b)) return true;
    else return false;
}}

int main(int argc, char **argv) {{
    int width = {width};
    int height = {height};
    
    {gradients_declare_str}
    
    {input_def}
    {input_boundary}
    
    double t = Halide::Tools::benchmark(10, {niters}, [&]() {{
    {device_free_str}
    {lib_call_str}
    {device_sync_str}
                                             }});
                                             
    printf("%g\\n", (t / (width * height) * 1e9));
    
    {render_code}
    
    return 0;
}}
    """
    
    return cpp_code;

if __name__ == '__main__':
    main()