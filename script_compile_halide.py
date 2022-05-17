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
    # multi scale L2 without deriv
    # 1 scale (with and without deriv)
    './Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_1_start_0_loss_only -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=1 start_stage=0 loss_only=true; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_1_start_0 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=1 start_stage=0; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_1_start_1_loss_only -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=1 start_stage=1 loss_only=true; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_1_start_1 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=1 start_stage=1',
    # 2 scale (with and without deriv)
    './Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_2_start_0_loss_only -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=2 start_stage=0 loss_only=true; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_2_start_0 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=2 start_stage=0; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_2_start_1_loss_only -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=2 start_stage=1 loss_only=true; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_2_start_1 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=2 start_stage=1; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_2_start_2_loss_only -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=2 start_stage=2 loss_only=true; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_2_start_2 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=2 start_stage=2',
    # 3 scale (with and without deriv)
    './Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_3_start_0_loss_only -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=3 start_stage=0 loss_only=true; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_3_start_0 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=3 start_stage=0; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_3_start_1_loss_only -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=3 start_stage=1 loss_only=true; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_3_start_1 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=3 start_stage=1; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_3_start_2_loss_only -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=3 start_stage=2 loss_only=true; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_3_start_2 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=3 start_stage=2; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_3_start_3_loss_only -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=3 start_stage=3 loss_only=true; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_3_start_3 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=3 start_stage=3',
    # 4 scale (with and without deriv)
    './Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_4_start_0_loss_only -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 start_stage=0 loss_only=true; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_4_start_0 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 start_stage=0; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_4_start_1_loss_only -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 start_stage=1 loss_only=true; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_4_start_1 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 start_stage=1; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_4_start_2_loss_only -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 start_stage=2 loss_only=true; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_4_start_2 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 start_stage=2; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_4_start_3_loss_only -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 start_stage=3 loss_only=true; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_4_start_3 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 start_stage=3; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_4_start_4_loss_only -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 start_stage=4 loss_only=true; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_4_start_4 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 start_stage=4',
    # 4 scale with sigma 0.5 (with and without deriv)
    './Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_4_sigma_05_start_0_loss_only -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5 start_stage=0 loss_only=true; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_4_sigma_05_start_0 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5 start_stage=0; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_4_sigma_05_start_1_loss_only -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5 start_stage=1 loss_only=true; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_4_sigma_05_start_1 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5 start_stage=1; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_4_sigma_05_start_2_loss_only -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5 start_stage=2 loss_only=true; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_4_sigma_05_start_2 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5 start_stage=2; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_4_sigma_05_start_3_loss_only -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5 start_stage=3 loss_only=true; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_4_sigma_05_start_3 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5 start_stage=3; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_4_sigma_05_start_4_loss_only -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5 start_stage=4 loss_only=true; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_4_sigma_05_start_4 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5 start_stage=4; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_4_sigma_05_start_5_loss_only -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5 start_stage=5 loss_only=true; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_4_sigma_05_start_5 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5 start_stage=5',
    # 4 scale with sigmas 0.5, 1 (with and without deriv)
    './Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_4_sigma_05_1_start_0_loss_only -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5,1 start_stage=0 loss_only=true; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_4_sigma_05_1_start_0 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5,1 start_stage=0; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_4_sigma_05_1_start_1_loss_only -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5,1 start_stage=1 loss_only=true; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_4_sigma_05_1_start_1 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5,1 start_stage=1; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_4_sigma_05_1_start_2_loss_only -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5,1 start_stage=2 loss_only=true; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_4_sigma_05_1_start_2 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5,1 start_stage=2; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_4_sigma_05_1_start_3_loss_only -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5,1 start_stage=3 loss_only=true; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_4_sigma_05_1_start_3 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5,1 start_stage=3; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_4_sigma_05_1_start_4_loss_only -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5,1 start_stage=4 loss_only=true; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_4_sigma_05_1_start_4 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5,1 start_stage=4; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_4_sigma_05_1_start_5_loss_only -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5,1 start_stage=5 loss_only=true; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_4_sigma_05_1_start_5 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5,1 start_stage=5; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_4_sigma_05_1_start_6_loss_only -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5,1 start_stage=6 loss_only=true; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_4_sigma_05_1_start_6 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5,1 start_stage=6',
    # 4 scale with sigmas 0.5, 1, 2
    './Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_4_sigma_05_1_2_start_0_loss_only -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5,1,2 start_stage=0 loss_only=true; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_4_sigma_05_1_2_start_1_loss_only -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5,1,2 start_stage=1 loss_only=true; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_4_sigma_05_1_2_start_2_loss_only -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5,1,2 start_stage=2 loss_only=true; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_4_sigma_05_1_2_start_3_loss_only -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5,1,2 start_stage=3 loss_only=true; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_4_sigma_05_1_2_start_4_loss_only -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5,1,2 start_stage=4 loss_only=true; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_4_sigma_05_1_2_start_5_loss_only -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5,1,2 start_stage=5 loss_only=true; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_4_sigma_05_1_2_start_6_loss_only -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5,1,2 start_stage=6 loss_only=true; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_4_sigma_05_1_2_start_7_loss_only -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5,1,2 start_stage=7 loss_only=true',
     # 4 scale with sigmas 0.5, 1, 2, 5
    './Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_4_sigma_05_1_2_5_start_0_loss_only -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5,1,2,5 start_stage=0 loss_only=true; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_4_sigma_05_1_2_5_start_1_loss_only -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5,1,2,5 start_stage=1 loss_only=true; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_4_sigma_05_1_2_5_start_2_loss_only -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5,1,2,5 start_stage=2 loss_only=true; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_4_sigma_05_1_2_5_start_3_loss_only -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5,1,2,5 start_stage=3 loss_only=true; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_4_sigma_05_1_2_5_start_4_loss_only -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5,1,2,5 start_stage=4 loss_only=true; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_4_sigma_05_1_2_5_start_5_loss_only -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5,1,2,5 start_stage=5 loss_only=true; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_4_sigma_05_1_2_5_start_6_loss_only -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5,1,2,5 start_stage=6 loss_only=true; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_4_sigma_05_1_2_5_start_7_loss_only -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5,1,2,5 start_stage=7 loss_only=true; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_4_sigma_05_1_2_5_start_8_loss_only -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5,1,2,5 start_stage=8 loss_only=true; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_0_loss_only -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=0 loss_only=true',
    # multi scale L2
    # 4 scale with sigmas 0.5, 1, 2
    './Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_start_0 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5,1,2 start_stage=0; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_start_1 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5,1,2 start_stage=1; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_start_2 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5,1,2 start_stage=2; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_start_3 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5,1,2 start_stage=3; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_start_4 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5,1,2 start_stage=4; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_start_5 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5,1,2 start_stage=5; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_start_6 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5,1,2 start_stage=6; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_start_7 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5,1,2 start_stage=7',
    # 4 scale with sigmas 0.5, 1, 2, 5
    './Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_5_start_0 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5,1,2,5 start_stage=0; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_5_start_1 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5,1,2,5 start_stage=1; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_5_start_2 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5,1,2,5 start_stage=2; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_5_start_3 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5,1,2,5 start_stage=3; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_5_start_4 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5,1,2,5 start_stage=4; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_5_start_5 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5,1,2,5 start_stage=5; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_5_start_6 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5,1,2,5 start_stage=6; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_5_start_7 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5,1,2,5 start_stage=7; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_5_start_8 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5,1,2,5 start_stage=8; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_0 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=0; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_1 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=1; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_2 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=2; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_3 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=3; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_5 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=5; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5,1; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5,1,2; ./Halide_multi_scale_L2_loss_global_read_generate -o . -g loss -f Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_5 -e static_library,h,schedule,stmt_html target=host-cuda-no_runtime auto_schedule=false nscale=4 smoothing_sigmas=0.5,1,2,5',
    # compile everything to a .o file
    'g++ Halide_lib.cpp -std=c++11 -I $HALIDE_INCLUDE_PATH -I $HALIDE_TOOL_PATH -L $HALIDE_LIB_PATH Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05.a Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1.a Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2.a Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_5.a Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_0.a Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_1.a Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_2.a Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_3.a Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4.a Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_5.a Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_5_start_0.a Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_5_start_1.a Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_5_start_2.a Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_5_start_3.a Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_5_start_4.a Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_5_start_5.a Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_5_start_6.a Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_5_start_7.a Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_5_start_8.a Halide_multi_scale_L2_4_sigma_05_1_2_5_start_0_loss_only.a Halide_multi_scale_L2_4_sigma_05_1_2_5_start_1_loss_only.a Halide_multi_scale_L2_4_sigma_05_1_2_5_start_2_loss_only.a Halide_multi_scale_L2_4_sigma_05_1_2_5_start_3_loss_only.a Halide_multi_scale_L2_4_sigma_05_1_2_5_start_4_loss_only.a Halide_multi_scale_L2_4_sigma_05_1_2_5_start_5_loss_only.a Halide_multi_scale_L2_4_sigma_05_1_2_5_start_6_loss_only.a Halide_multi_scale_L2_4_sigma_05_1_2_5_start_7_loss_only.a Halide_multi_scale_L2_4_sigma_05_1_2_5_start_8_loss_only.a Halide_multi_scale_L2_0_loss_only.a Halide_multi_scale_L2_4_start_0_loss_only.a Halide_multi_scale_L2_4_start_1_loss_only.a Halide_multi_scale_L2_4_start_2_loss_only.a Halide_multi_scale_L2_4_start_3_loss_only.a Halide_multi_scale_L2_4_start_4_loss_only.a Halide_multi_scale_L2_4_start_0.a Halide_multi_scale_L2_4_start_1.a Halide_multi_scale_L2_4_start_2.a Halide_multi_scale_L2_4_start_3.a Halide_multi_scale_L2_4_start_4.a Halide_multi_scale_L2_3_start_0_loss_only.a Halide_multi_scale_L2_3_start_1_loss_only.a Halide_multi_scale_L2_3_start_2_loss_only.a Halide_multi_scale_L2_3_start_3_loss_only.a Halide_multi_scale_L2_3_start_0.a Halide_multi_scale_L2_3_start_1.a Halide_multi_scale_L2_3_start_2.a Halide_multi_scale_L2_3_start_3.a Halide_multi_scale_L2_2_start_0_loss_only.a Halide_multi_scale_L2_2_start_1_loss_only.a Halide_multi_scale_L2_2_start_2_loss_only.a Halide_multi_scale_L2_2_start_0.a Halide_multi_scale_L2_2_start_1.a Halide_multi_scale_L2_2_start_2.a Halide_downsample_2.a Halide_gaussian_conv_05.a Halide_gaussian_conv_1.a Halide_gaussian_conv_2.a Halide_gaussian_conv_5.a Halide_reduce_single_tensor_2d.a Halide_reduce_single_tensor_start_0.a Halide_reduce_single_tensor_start_2.a Halide_multi_scale_L2_1_start_0_loss_only.a Halide_multi_scale_L2_1_start_1_loss_only.a Halide_multi_scale_L2_1_start_0.a Halide_multi_scale_L2_1_start_1.a Halide_reduce_single_tensor_start_5.a Halide_reduce_single_tensor_start_10.a runtime.a `libpng-config --cflags --ldflags` -ljpeg -ldl -lpthread -fPIC -Wall -shared -lpthread -lHalide -I ./ -o Halide_lib.o'
]

def main():
    if len(sys.argv) != 3:
        print('Usage: python script_compile_halide.py <halide_path> <build_path>')
        print('eg: python script_compile_halide.py /n/fs/shaderml/Halide_binary/Halide-10.0.0-x86-64-linux /n/fs/scratch/yutingy/Halide')
        return

    hl_path = sys.argv[1]
    path = sys.argv[2]
    
    if not os.path.isdir(path):
        os.makedirs(path)
        
    shutil.copyfile('Halide_lib.cpp', os.path.join(path, 'Halide_lib.cpp'))
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
        
    cmd += 'cp Halide_lib.o %s;\n' % cwd
    
    for single_cmd in cmd.split(';'):
        print(single_cmd)
        os.system(single_cmd)
        
if __name__ == '__main__':
    main()