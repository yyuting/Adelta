import os
import sys
import shutil

aot_files = ['Halide_multi_scale_L2_loss_global_read_generate',
             'Halide_downsample_generate',
             'Halide_gaussian_conv_generate',
             'Halide_reduce_single_tensor_2d_generate',
             'Halide_reduce_single_tensor_generate']

compile_cmds = [
    # runtime
    './Halide_multi_scale_L2_loss_global_read_generate -o . -r runtime target=host-cuda auto_schedule=false',
    # 2D reduce sum
    './Halide_reduce_single_tensor_2d_generate -o . -g reduce -f Halide_reduce_single_tensor_2d -e static_library,h,schedule,assembly target=host-cuda-no_runtime auto_schedule=false',
    # 3D reduce sum
    './Halide_reduce_single_tensor_generate -o . -g reduce -f Halide_reduce_single_tensor_start_0 -e static_library,h,schedule,assembly target=host-cuda-no_runtime auto_schedule=false; ./Halide_reduce_single_tensor_generate -o . -g reduce -f Halide_reduce_single_tensor_start_2 -e static_library,h,schedule,assembly target=host-cuda-no_runtime auto_schedule=false start_idx=2; ./Halide_reduce_single_tensor_generate -o . -g reduce -f Halide_reduce_single_tensor_start_5 -e static_library,h,schedule,assembly target=host-cuda-no_runtime auto_schedule=false start_idx=5; ./Halide_reduce_single_tensor_generate -o . -g reduce -f Halide_reduce_single_tensor_start_10 -e static_library,h,schedule,assembly target=host-cuda-no_runtime auto_schedule=false start_idx=10',
    # downsample 2x
    './Halide_downsample_generate -o . -g layer -f Halide_downsample_2 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false scale=2',
    # conv with a Gaussian kernel at various sigma
    './Halide_gaussian_conv_generate -o . -g layer -f Halide_gaussian_conv_05 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false sigma=0.5; ./Halide_gaussian_conv_generate -o . -g layer -f Halide_gaussian_conv_1 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false sigma=1; ./Halide_gaussian_conv_generate -o . -g layer -f Halide_gaussian_conv_2 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false sigma=2; ./Halide_gaussian_conv_generate -o . -g layer -f Halide_gaussian_conv_5 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false sigma=5',
]

lib_cmd_template_start = 'g++ Halide_lib.cpp -std=c++11 -I $HALIDE_INCLUDE_PATH -I $HALIDE_TOOL_PATH -L $HALIDE_LIB_PATH Halide_downsample_2.a Halide_gaussian_conv_05.a Halide_gaussian_conv_1.a Halide_gaussian_conv_2.a Halide_gaussian_conv_5.a Halide_reduce_single_tensor_2d.a Halide_reduce_single_tensor_start_0.a Halide_reduce_single_tensor_start_2.a Halide_multi_scale_L2_1_start_0_loss_only.a Halide_multi_scale_L2_1_start_1_loss_only.a Halide_multi_scale_L2_1_start_0.a Halide_multi_scale_L2_1_start_1.a Halide_reduce_single_tensor_start_5.a Halide_reduce_single_tensor_start_10.a runtime.a '

lib_cmd_template_end = '`libpng-config --cflags --ldflags` -ljpeg -ldl -lpthread -fPIC -Wall -shared -lpthread -lHalide -I ./ -o Halide_lib.o'

def augment_compile_cmds(lib_template_str):
    global compile_cmds
    
    lib_cmd = lib_cmd_template_start
    lib_body = ''
    
    for nscale in range(5):
        
        base_name = 'Halide_multi_scale_L2_%d' % nscale
        

        for include_sigmas in [True, False]:
            
            if lib_body == '':
                if_str = 'if'
            else:
                if_str = 'else if'
                        
            if include_sigmas:
                sigmas_str = '_sigma_05_1_2_5'
                sigmas_cmd = ' smoothing_sigmas=0.5,1,2,5 '
                nstages = nscale + 5
                    
                lib_body += f"""
    {if_str} (nscale == {nscale} && sigmas.size() == 4 && sigmas[0] == 0.5f && sigmas[1] == 1.f && sigmas[2] == 2.f && sigmas[3] == 5.f && start_stage < {nstages}) {{
        if (!check_ok) {{
        """
            else:
                sigmas_str = ''
                sigmas_cmd = ''
                nstages = nscale + 1
                lib_body += f"""
    {if_str} (nscale == {nscale} && sigmas.size() == 0 && start_stage < {nstages}) {{
        if (!check_ok) {{
        """
                
            for start_stage in range(nstages):
                
                current_base = '%s%s_start_%d' % (base_name, sigmas_str, start_stage)
                
                if start_stage == 0:
                    if_str = 'if'
                else:
                    if_str = 'else if'
                
                lib_body += f"""
            {if_str} (start_stage == {start_stage}) {{
                if (get_deriv) {{
                    {current_base}(
                    width, height,
                    *input0.get(), *input1.get(), *input2.get(), *input3.get(), *input4.get(), *input5.get(),
                    *input6.get(), *input7.get(), *input8.get(), *input9.get(), *input10.get(), *gradients.get());
                }} else {{
                    {current_base}_loss_only(
                    width, height,
                    *input0.get(), *input1.get(), *input2.get(), *input3.get(), *input4.get(), *input5.get(),
                    *input6.get(), *input7.get(), *input8.get(), *input9.get(), *input10.get(), *gradients.get());
                }}
            }}
                """
                
                for loss_only in [True, False]:
                    kernel_name = '%s%s' % (current_base, '_loss_only' if loss_only else '')
                    cmd = './Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f %s -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=%d %s start_stage=%d loss_only=%s' % (kernel_name, nscale, sigmas_cmd, start_stage, 'true' if loss_only else 'false')
                    compile_cmds.append(cmd)
                    lib_cmd += ' %s.a ' % kernel_name
                    lib_template_str = '#include "%s.h"\n' % kernel_name + lib_template_str
                    
            lib_body += f"""
        }}
    }}
            """
        
    lib_cmd += lib_cmd_template_end
    
    lib_template_str = lib_template_str.replace('__nscale_L2_main_body__', lib_body)
    
    return lib_cmd, lib_template_str
        

def main():
    if len(sys.argv) != 3:
        print('Usage: python script_compile_halide.py <halide_path> <build_path>')
        print('eg: python script_compile_halide.py /n/fs/shaderml/Halide_binary/Halide-10.0.0-x86-64-linux /n/fs/scratch/yutingy/Halide')
        return

    hl_path = sys.argv[1]
    path = sys.argv[2]
    
    if not os.path.isdir(path):
        os.makedirs(path)
        
    lib_template_str = open('Halide_lib_template.cpp').read()
        
    lib_cmd, lib_template_str = augment_compile_cmds(lib_template_str)
    
    open(os.path.join(path, 'Halide_lib.cpp'), 'w').write(lib_template_str)
        
    shutil.copyfile('Halide_lib.h', os.path.join(path, 'Halide_lib.h'))

    cwd = os.getcwd()

    os.chdir(path)
    
    cmd = ''
    
    cmd += 'export HALIDE_TOOL_PATH="%s/share/Halide/tools";\n' % hl_path
    cmd += 'export HALIDE_INCLUDE_PATH="%s/include";\n' % hl_path
    cmd += 'export HALIDE_LIB_PATH="%s/lib";\n' % hl_path
    
    for file in aot_files:
        cmd += 'g++ %s/%s.cpp $HALIDE_TOOL_PATH/GenGen.cpp -g -std=c++11 -fno-rtti -I $HALIDE_INCLUDE_PATH -L $HALIDE_LIB_PATH -lHalide -lpthread -ldl -Wl,-rpath,${HALIDE_LIB_PATH} -o %s; \n' % (cwd, file, file)
        
    for val in compile_cmds:
        cmd += '%s;\n' % val
        
    cmd += lib_cmd + ';\n'
    cmd += 'cp Halide_lib.o %s;\n' % cwd
        
    for single_cmd in cmd.split(';'):
        print(single_cmd)
        os.system(single_cmd)
        
if __name__ == '__main__':
    main()