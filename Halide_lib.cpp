// Compilation commands
// g++ Halide_lib.cpp -std=c++11 -I $HALIDE_INCLUDE_PATH -I $HALIDE_TOOL_PATH -L $HALIDE_LIB_PATH Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05.a Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1.a Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2.a Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_5.a Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_0.a Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_1.a Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_2.a Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_3.a Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4.a Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_5.a Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_5_start_0.a Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_5_start_1.a Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_5_start_2.a Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_5_start_3.a Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_5_start_4.a Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_5_start_5.a Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_5_start_6.a Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_5_start_7.a Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_5_start_8.a Halide_multi_scale_L2_4_sigma_05_1_2_5_start_0_loss_only.a Halide_multi_scale_L2_4_sigma_05_1_2_5_start_1_loss_only.a Halide_multi_scale_L2_4_sigma_05_1_2_5_start_2_loss_only.a Halide_multi_scale_L2_4_sigma_05_1_2_5_start_3_loss_only.a Halide_multi_scale_L2_4_sigma_05_1_2_5_start_4_loss_only.a Halide_multi_scale_L2_4_sigma_05_1_2_5_start_5_loss_only.a Halide_multi_scale_L2_4_sigma_05_1_2_5_start_6_loss_only.a Halide_multi_scale_L2_4_sigma_05_1_2_5_start_7_loss_only.a Halide_multi_scale_L2_4_sigma_05_1_2_5_start_8_loss_only.a Halide_multi_scale_L2_0_loss_only.a Halide_multi_scale_L2_4_start_0_loss_only.a Halide_multi_scale_L2_4_start_1_loss_only.a Halide_multi_scale_L2_4_start_2_loss_only.a Halide_multi_scale_L2_4_start_3_loss_only.a Halide_multi_scale_L2_4_start_4_loss_only.a Halide_multi_scale_L2_4_start_0.a Halide_multi_scale_L2_4_start_1.a Halide_multi_scale_L2_4_start_2.a Halide_multi_scale_L2_4_start_3.a Halide_multi_scale_L2_4_start_4.a Halide_multi_scale_L2_3_start_0_loss_only.a Halide_multi_scale_L2_3_start_1_loss_only.a Halide_multi_scale_L2_3_start_2_loss_only.a Halide_multi_scale_L2_3_start_3_loss_only.a Halide_multi_scale_L2_3_start_0.a Halide_multi_scale_L2_3_start_1.a Halide_multi_scale_L2_3_start_2.a Halide_multi_scale_L2_3_start_3.a Halide_multi_scale_L2_2_start_0_loss_only.a Halide_multi_scale_L2_2_start_1_loss_only.a Halide_multi_scale_L2_2_start_2_loss_only.a Halide_multi_scale_L2_2_start_0.a Halide_multi_scale_L2_2_start_1.a Halide_multi_scale_L2_2_start_2.a Halide_downsample_2.a Halide_gaussian_conv_05.a Halide_gaussian_conv_1.a Halide_gaussian_conv_2.a Halide_gaussian_conv_5.a Halide_reduce_single_tensor_2d.a Halide_reduce_single_tensor_start_0.a Halide_reduce_single_tensor_start_2.a Halide_multi_scale_L2_1_start_0_loss_only.a Halide_multi_scale_L2_1_start_1_loss_only.a Halide_multi_scale_L2_1_start_0.a Halide_multi_scale_L2_1_start_1.a Halide_reduce_single_tensor_start_5.a Halide_reduce_single_tensor_start_10.a runtime.a `libpng-config --cflags --ldflags` -ljpeg -ldl -lpthread -fPIC -Wall -shared -lpthread -lHalide -I ./ -o Halide_lib.o

#include "Halide_lib.h"

// Reduce sum ops
#include "Halide_reduce_single_tensor_2d.h"
#include "Halide_reduce_single_tensor_start_0.h"
#include "Halide_reduce_single_tensor_start_2.h"
#include "Halide_reduce_single_tensor_start_5.h"
#include "Halide_reduce_single_tensor_start_10.h"

// Basic deep learning library
#include "Halide_downsample_2.h"
#include "Halide_gaussian_conv_05.h"
#include "Halide_gaussian_conv_1.h"
#include "Halide_gaussian_conv_2.h"
#include "Halide_gaussian_conv_5.h"

// Kernels for multi scale L2

#include "Halide_multi_scale_L2_1_start_0_loss_only.h"
#include "Halide_multi_scale_L2_1_start_1_loss_only.h"

#include "Halide_multi_scale_L2_1_start_0.h"
#include "Halide_multi_scale_L2_1_start_1.h"

#include "Halide_multi_scale_L2_2_start_0_loss_only.h"
#include "Halide_multi_scale_L2_2_start_1_loss_only.h"
#include "Halide_multi_scale_L2_2_start_2_loss_only.h"

#include "Halide_multi_scale_L2_2_start_0.h"
#include "Halide_multi_scale_L2_2_start_1.h"
#include "Halide_multi_scale_L2_2_start_2.h"

#include "Halide_multi_scale_L2_3_start_0_loss_only.h"
#include "Halide_multi_scale_L2_3_start_1_loss_only.h"
#include "Halide_multi_scale_L2_3_start_2_loss_only.h"
#include "Halide_multi_scale_L2_3_start_3_loss_only.h"

#include "Halide_multi_scale_L2_3_start_0.h"
#include "Halide_multi_scale_L2_3_start_1.h"
#include "Halide_multi_scale_L2_3_start_2.h"
#include "Halide_multi_scale_L2_3_start_3.h"

#include "Halide_multi_scale_L2_4_start_0_loss_only.h"
#include "Halide_multi_scale_L2_4_start_1_loss_only.h"
#include "Halide_multi_scale_L2_4_start_2_loss_only.h"
#include "Halide_multi_scale_L2_4_start_3_loss_only.h"
#include "Halide_multi_scale_L2_4_start_4_loss_only.h"

#include "Halide_multi_scale_L2_4_start_0.h"
#include "Halide_multi_scale_L2_4_start_1.h"
#include "Halide_multi_scale_L2_4_start_2.h"
#include "Halide_multi_scale_L2_4_start_3.h"
#include "Halide_multi_scale_L2_4_start_4.h"

#include "Halide_multi_scale_L2_4_sigma_05_1_2_5_start_0_loss_only.h"
#include "Halide_multi_scale_L2_4_sigma_05_1_2_5_start_1_loss_only.h"
#include "Halide_multi_scale_L2_4_sigma_05_1_2_5_start_2_loss_only.h"
#include "Halide_multi_scale_L2_4_sigma_05_1_2_5_start_3_loss_only.h"
#include "Halide_multi_scale_L2_4_sigma_05_1_2_5_start_4_loss_only.h"
#include "Halide_multi_scale_L2_4_sigma_05_1_2_5_start_5_loss_only.h"
#include "Halide_multi_scale_L2_4_sigma_05_1_2_5_start_6_loss_only.h"
#include "Halide_multi_scale_L2_4_sigma_05_1_2_5_start_7_loss_only.h"
#include "Halide_multi_scale_L2_4_sigma_05_1_2_5_start_8_loss_only.h"
#include "Halide_multi_scale_L2_0_loss_only.h"

#include "Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05.h"
#include "Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1.h"
#include "Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2.h"
#include "Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_5.h"

#include "Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_5_start_0.h"
#include "Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_5_start_1.h"
#include "Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_5_start_2.h"
#include "Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_5_start_3.h"
#include "Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_5_start_4.h"
#include "Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_5_start_5.h"
#include "Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_5_start_6.h"
#include "Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_5_start_7.h"
#include "Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_5_start_8.h"

#include "Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_0.h"
#include "Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_1.h"
#include "Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_2.h"
#include "Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_3.h"
#include "Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4.h"
#include "Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_5.h"

using namespace Halide::Tools;
using namespace Halide;

void set_host_dirty(Buffer<float> input) {
    input.set_host_dirty();
}

void copy_to_host(Buffer<float> input) {
    input.copy_to_host();
}

int reduce_sum(int start_idx,
               int width, int height,
               Buffer<float> input,
               Buffer<float> output,
               bool check_ok) {
    
    int fail = 0;
    
    if (start_idx == 0) {
        if (!check_ok) {
            Halide_reduce_single_tensor_start_0(width, height, *input.get(), *output.get());
        }
    } else if (start_idx == 2) {
        if (!check_ok) {
            Halide_reduce_single_tensor_start_2(width, height, *input.get(), *output.get());
        }
    } else if (start_idx == 5) {
        if (!check_ok) {
            Halide_reduce_single_tensor_start_5(width, height, *input.get(), *output.get());
        }
    } else if (start_idx == 10) {
        if (!check_ok) {
            Halide_reduce_single_tensor_start_10(width, height, *input.get(), *output.get());
        }
    } else {
        fail = -1;
        printf("Error! Reduce kernel starting with idx = %d is not compiled\n", start_idx);
    }
    
    return fail;
}

int gaussian_conv(float sigma,
                  Buffer<float> input,
                  Buffer<float> output,
                  bool check_ok) {
    int fail = 0;
    
    if (sigma == 0.5f) {
        if (!check_ok) {
            Halide_gaussian_conv_05(*input.get(), *output.get());
        }
    } else if (sigma == 1.f) {
        if (!check_ok) {
            Halide_gaussian_conv_1(*input.get(), *output.get());
        }
    } else if (sigma == 2.f) {
        if (!check_ok) {
            Halide_gaussian_conv_2(*input.get(), *output.get());
        }
    } else if (sigma == 5.f) {
        if (!check_ok) {
            Halide_gaussian_conv_5(*input.get(), *output.get());
        }
    } else {
        fail = -1;
        printf("Error! Kernel for sigma = %f is not compiled\n", sigma);
    }
    
    return fail;
}

int downsample(int scale,
               Buffer<float> input,
               Buffer<float> output,
               bool check_ok) {
    
    int fail = 0;
    
    if (scale == 2) {
        if (!check_ok) {
            Halide_downsample_2(*input.get(), *output.get());
        }
    } else {
        fail = -1;
        printf("Error! Kernel for scale = %d is not compiled\n", scale);
    }
    
    return fail;
    
}

int nscale_L2(int nscale, 
              std::vector<float> sigmas,
              int width, int height,
              Buffer<float> input0, Buffer<float> input1, Buffer<float> input2, Buffer<float> input3, 
              Buffer<float> input4, Buffer<float> input5, Buffer<float> input6, Buffer<float> input7,
              Buffer<float> input8, Buffer<float> input9, Buffer<float> input10, Buffer<float> gradients, 
              Buffer<float> loss, bool get_loss,
              int start_stage,
              bool get_deriv,
              bool check_ok) {
    
    int fail = 0;
    
    if (nscale == 0 && sigmas.size() == 0 && start_stage == 0) {
        if (!check_ok) {
            if (get_deriv) {
                Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_0(
                    width, height,
                    *input0.get(), *input1.get(), *input2.get(), *input3.get(), *input4.get(), *input5.get(),
                    *input6.get(), *input7.get(), *input8.get(), *input9.get(), *input10.get(), *gradients.get());
            } else {
                Halide_multi_scale_L2_0_loss_only(
                    width, height,
                    *input0.get(), *input1.get(), *input2.get(), *input3.get(), *input4.get(), *input5.get(),
                    *input6.get(), *input7.get(), *input8.get(), *input9.get(), *input10.get(), *gradients.get());
            }
        }
    } else if (nscale == 4 && sigmas.size() == 4 && sigmas[0] == 0.5f && sigmas[1] == 1.f && sigmas[2] == 2.f && sigmas[3] == 5.f  && start_stage <= 8) {
        
        if (!check_ok) {
            if (start_stage == 0) {
                if (get_deriv) {
                    Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_5(
                        width, height,
                        *input0.get(), *input1.get(), *input2.get(), *input3.get(), *input4.get(), *input5.get(),
                        *input6.get(), *input7.get(), *input8.get(), *input9.get(), *input10.get(), *gradients.get());
                } else {
                    Halide_multi_scale_L2_4_sigma_05_1_2_5_start_0_loss_only(
                        width, height,
                        *input0.get(), *input1.get(), *input2.get(), *input3.get(), *input4.get(), *input5.get(),
                        *input6.get(), *input7.get(), *input8.get(), *input9.get(), *input10.get(), *gradients.get());
                }
            } else if (start_stage == 1) {
                if (get_deriv) {
                    Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_5_start_1(
                        width, height,
                        *input0.get(), *input1.get(), *input2.get(), *input3.get(), *input4.get(), *input5.get(),
                        *input6.get(), *input7.get(), *input8.get(), *input9.get(), *input10.get(), *gradients.get());
                } else {
                    Halide_multi_scale_L2_4_sigma_05_1_2_5_start_1_loss_only(
                        width, height,
                        *input0.get(), *input1.get(), *input2.get(), *input3.get(), *input4.get(), *input5.get(),
                        *input6.get(), *input7.get(), *input8.get(), *input9.get(), *input10.get(), *gradients.get());
                }
            } else if (start_stage == 2) {
                if (get_deriv) {
                    Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_5_start_2(
                        width, height,
                        *input0.get(), *input1.get(), *input2.get(), *input3.get(), *input4.get(), *input5.get(),
                        *input6.get(), *input7.get(), *input8.get(), *input9.get(), *input10.get(), *gradients.get());
                } else {
                    Halide_multi_scale_L2_4_sigma_05_1_2_5_start_2_loss_only(
                        width, height,
                        *input0.get(), *input1.get(), *input2.get(), *input3.get(), *input4.get(), *input5.get(),
                        *input6.get(), *input7.get(), *input8.get(), *input9.get(), *input10.get(), *gradients.get());
                }
            } else if (start_stage == 3) {
                if (get_deriv) {
                    Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_5_start_3(
                        width, height,
                        *input0.get(), *input1.get(), *input2.get(), *input3.get(), *input4.get(), *input5.get(),
                        *input6.get(), *input7.get(), *input8.get(), *input9.get(), *input10.get(), *gradients.get());
                } else {
                    Halide_multi_scale_L2_4_sigma_05_1_2_5_start_3_loss_only(
                        width, height,
                        *input0.get(), *input1.get(), *input2.get(), *input3.get(), *input4.get(), *input5.get(),
                        *input6.get(), *input7.get(), *input8.get(), *input9.get(), *input10.get(), *gradients.get());
                }
            } else if (start_stage == 4) {
                if (get_deriv) {
                    Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_5_start_4(
                        width, height,
                        *input0.get(), *input1.get(), *input2.get(), *input3.get(), *input4.get(), *input5.get(),
                        *input6.get(), *input7.get(), *input8.get(), *input9.get(), *input10.get(), *gradients.get());
                } else {
                    Halide_multi_scale_L2_4_sigma_05_1_2_5_start_4_loss_only(
                        width, height,
                        *input0.get(), *input1.get(), *input2.get(), *input3.get(), *input4.get(), *input5.get(),
                        *input6.get(), *input7.get(), *input8.get(), *input9.get(), *input10.get(), *gradients.get());
                }
            } else if (start_stage == 5) {
                if (get_deriv) {
                    Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_5_start_5(
                        width, height,
                        *input0.get(), *input1.get(), *input2.get(), *input3.get(), *input4.get(), *input5.get(),
                        *input6.get(), *input7.get(), *input8.get(), *input9.get(), *input10.get(), *gradients.get());
                } else {
                    Halide_multi_scale_L2_4_sigma_05_1_2_5_start_5_loss_only(
                        width, height,
                        *input0.get(), *input1.get(), *input2.get(), *input3.get(), *input4.get(), *input5.get(),
                        *input6.get(), *input7.get(), *input8.get(), *input9.get(), *input10.get(), *gradients.get());
                }
            } else if (start_stage == 6) {
                if (get_deriv) {
                    Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_5_start_6(
                        width, height,
                        *input0.get(), *input1.get(), *input2.get(), *input3.get(), *input4.get(), *input5.get(),
                        *input6.get(), *input7.get(), *input8.get(), *input9.get(), *input10.get(), *gradients.get());
                } else {
                    Halide_multi_scale_L2_4_sigma_05_1_2_5_start_6_loss_only(
                        width, height,
                        *input0.get(), *input1.get(), *input2.get(), *input3.get(), *input4.get(), *input5.get(),
                        *input6.get(), *input7.get(), *input8.get(), *input9.get(), *input10.get(), *gradients.get());
                }
            } else if (start_stage == 7) {
                if (get_deriv) {
                    Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_5_start_7(
                        width, height,
                        *input0.get(), *input1.get(), *input2.get(), *input3.get(), *input4.get(), *input5.get(),
                        *input6.get(), *input7.get(), *input8.get(), *input9.get(), *input10.get(), *gradients.get());
                } else {
                    Halide_multi_scale_L2_4_sigma_05_1_2_5_start_7_loss_only(
                        width, height,
                        *input0.get(), *input1.get(), *input2.get(), *input3.get(), *input4.get(), *input5.get(),
                        *input6.get(), *input7.get(), *input8.get(), *input9.get(), *input10.get(), *gradients.get());
                }
            } else if (start_stage == 8) {
                if (get_deriv) {
                    Halide_multi_scale_L2_manual_AD_global_read_Sioutas_gpu_4_sigma_05_1_2_5_start_8(
                        width, height,
                        *input0.get(), *input1.get(), *input2.get(), *input3.get(), *input4.get(), *input5.get(),
                        *input6.get(), *input7.get(), *input8.get(), *input9.get(), *input10.get(), *gradients.get());
                } else {
                    Halide_multi_scale_L2_4_sigma_05_1_2_5_start_8_loss_only(
                        width, height,
                        *input0.get(), *input1.get(), *input2.get(), *input3.get(), *input4.get(), *input5.get(),
                        *input6.get(), *input7.get(), *input8.get(), *input9.get(), *input10.get(), *gradients.get());
                }
            }
        }
    } else if (nscale == 4 && sigmas.size() == 0 && start_stage <= 4) {
        if (!check_ok) {
            if (start_stage == 0) {
                if (get_deriv) {
                    Halide_multi_scale_L2_4_start_0(
                        width, height,
                        *input0.get(), *input1.get(), *input2.get(), *input3.get(), *input4.get(), *input5.get(),
                        *input6.get(), *input7.get(), *input8.get(), *input9.get(), *input10.get(), *gradients.get());
                } else {
                    Halide_multi_scale_L2_4_start_0_loss_only(
                        width, height,
                        *input0.get(), *input1.get(), *input2.get(), *input3.get(), *input4.get(), *input5.get(),
                        *input6.get(), *input7.get(), *input8.get(), *input9.get(), *input10.get(), *gradients.get());
                }
            } else if (start_stage == 1) {
                if (get_deriv) {
                    Halide_multi_scale_L2_4_start_1(
                        width, height,
                        *input0.get(), *input1.get(), *input2.get(), *input3.get(), *input4.get(), *input5.get(),
                        *input6.get(), *input7.get(), *input8.get(), *input9.get(), *input10.get(), *gradients.get());
                } else {
                    Halide_multi_scale_L2_4_start_1_loss_only(
                        width, height,
                        *input0.get(), *input1.get(), *input2.get(), *input3.get(), *input4.get(), *input5.get(),
                        *input6.get(), *input7.get(), *input8.get(), *input9.get(), *input10.get(), *gradients.get());
                }
            } else if (start_stage == 2) {
                if (get_deriv) {
                    Halide_multi_scale_L2_4_start_2(
                        width, height,
                        *input0.get(), *input1.get(), *input2.get(), *input3.get(), *input4.get(), *input5.get(),
                        *input6.get(), *input7.get(), *input8.get(), *input9.get(), *input10.get(), *gradients.get());
                } else {
                    Halide_multi_scale_L2_4_start_2_loss_only(
                        width, height,
                        *input0.get(), *input1.get(), *input2.get(), *input3.get(), *input4.get(), *input5.get(),
                        *input6.get(), *input7.get(), *input8.get(), *input9.get(), *input10.get(), *gradients.get());
                }
            } else if (start_stage == 3) {
                if (get_deriv) {
                    Halide_multi_scale_L2_4_start_3(
                        width, height,
                        *input0.get(), *input1.get(), *input2.get(), *input3.get(), *input4.get(), *input5.get(),
                        *input6.get(), *input7.get(), *input8.get(), *input9.get(), *input10.get(), *gradients.get());
                } else {
                    Halide_multi_scale_L2_4_start_3_loss_only(
                        width, height,
                        *input0.get(), *input1.get(), *input2.get(), *input3.get(), *input4.get(), *input5.get(),
                        *input6.get(), *input7.get(), *input8.get(), *input9.get(), *input10.get(), *gradients.get());
                }
            } else if (start_stage == 4) {
                if (get_deriv) {
                    Halide_multi_scale_L2_4_start_4(
                        width, height,
                        *input0.get(), *input1.get(), *input2.get(), *input3.get(), *input4.get(), *input5.get(),
                        *input6.get(), *input7.get(), *input8.get(), *input9.get(), *input10.get(), *gradients.get());
                } else {
                    Halide_multi_scale_L2_4_start_4_loss_only(
                        width, height,
                        *input0.get(), *input1.get(), *input2.get(), *input3.get(), *input4.get(), *input5.get(),
                        *input6.get(), *input7.get(), *input8.get(), *input9.get(), *input10.get(), *gradients.get());
                }
            }
        }
    } else if (nscale == 3 && sigmas.size() == 0 && start_stage <= 3) {
        if (!check_ok) {
            if (start_stage == 0) {
                if (get_deriv) {
                    Halide_multi_scale_L2_3_start_0(
                        width, height,
                        *input0.get(), *input1.get(), *input2.get(), *input3.get(), *input4.get(), *input5.get(),
                        *input6.get(), *input7.get(), *input8.get(), *input9.get(), *input10.get(), *gradients.get());
                } else {
                    Halide_multi_scale_L2_3_start_0_loss_only(
                        width, height,
                        *input0.get(), *input1.get(), *input2.get(), *input3.get(), *input4.get(), *input5.get(),
                        *input6.get(), *input7.get(), *input8.get(), *input9.get(), *input10.get(), *gradients.get());
                }
            } else if (start_stage == 1) {
                if (get_deriv) {
                    Halide_multi_scale_L2_3_start_1(
                        width, height,
                        *input0.get(), *input1.get(), *input2.get(), *input3.get(), *input4.get(), *input5.get(),
                        *input6.get(), *input7.get(), *input8.get(), *input9.get(), *input10.get(), *gradients.get());
                } else {
                    Halide_multi_scale_L2_3_start_1_loss_only(
                        width, height,
                        *input0.get(), *input1.get(), *input2.get(), *input3.get(), *input4.get(), *input5.get(),
                        *input6.get(), *input7.get(), *input8.get(), *input9.get(), *input10.get(), *gradients.get());
                }
            } else if (start_stage == 2) {
                if (get_deriv) {
                    Halide_multi_scale_L2_3_start_2(
                        width, height,
                        *input0.get(), *input1.get(), *input2.get(), *input3.get(), *input4.get(), *input5.get(),
                        *input6.get(), *input7.get(), *input8.get(), *input9.get(), *input10.get(), *gradients.get());
                } else {
                    Halide_multi_scale_L2_3_start_2_loss_only(
                        width, height,
                        *input0.get(), *input1.get(), *input2.get(), *input3.get(), *input4.get(), *input5.get(),
                        *input6.get(), *input7.get(), *input8.get(), *input9.get(), *input10.get(), *gradients.get());
                }
            } else if (start_stage == 3) {
                if (get_deriv) {
                    Halide_multi_scale_L2_3_start_3(
                        width, height,
                        *input0.get(), *input1.get(), *input2.get(), *input3.get(), *input4.get(), *input5.get(),
                        *input6.get(), *input7.get(), *input8.get(), *input9.get(), *input10.get(), *gradients.get());
                } else {
                    Halide_multi_scale_L2_3_start_3_loss_only(
                        width, height,
                        *input0.get(), *input1.get(), *input2.get(), *input3.get(), *input4.get(), *input5.get(),
                        *input6.get(), *input7.get(), *input8.get(), *input9.get(), *input10.get(), *gradients.get());
                }
            }
        }
    } else if (nscale == 2 && sigmas.size() == 0 && start_stage <= 3) {
        if (!check_ok) {
            if (start_stage == 0) {
                if (get_deriv) {
                    Halide_multi_scale_L2_2_start_0(
                        width, height,
                        *input0.get(), *input1.get(), *input2.get(), *input3.get(), *input4.get(), *input5.get(),
                        *input6.get(), *input7.get(), *input8.get(), *input9.get(), *input10.get(), *gradients.get());
                } else {
                    Halide_multi_scale_L2_2_start_0_loss_only(
                        width, height,
                        *input0.get(), *input1.get(), *input2.get(), *input3.get(), *input4.get(), *input5.get(),
                        *input6.get(), *input7.get(), *input8.get(), *input9.get(), *input10.get(), *gradients.get());
                }
            } else if (start_stage == 1) {
                if (get_deriv) {
                    Halide_multi_scale_L2_2_start_1(
                        width, height,
                        *input0.get(), *input1.get(), *input2.get(), *input3.get(), *input4.get(), *input5.get(),
                        *input6.get(), *input7.get(), *input8.get(), *input9.get(), *input10.get(), *gradients.get());
                } else {
                    Halide_multi_scale_L2_2_start_1_loss_only(
                        width, height,
                        *input0.get(), *input1.get(), *input2.get(), *input3.get(), *input4.get(), *input5.get(),
                        *input6.get(), *input7.get(), *input8.get(), *input9.get(), *input10.get(), *gradients.get());
                }
            } else if (start_stage == 2) {
                if (get_deriv) {
                    Halide_multi_scale_L2_2_start_2(
                        width, height,
                        *input0.get(), *input1.get(), *input2.get(), *input3.get(), *input4.get(), *input5.get(),
                        *input6.get(), *input7.get(), *input8.get(), *input9.get(), *input10.get(), *gradients.get());
                } else {
                    Halide_multi_scale_L2_2_start_2_loss_only(
                        width, height,
                        *input0.get(), *input1.get(), *input2.get(), *input3.get(), *input4.get(), *input5.get(),
                        *input6.get(), *input7.get(), *input8.get(), *input9.get(), *input10.get(), *gradients.get());
                }
            }
        }
    } else if (nscale == 1 && sigmas.size() == 0 && start_stage <= 3) {
        if (!check_ok) {
            if (start_stage == 0) {
                if (get_deriv) {
                    Halide_multi_scale_L2_1_start_0(
                        width, height,
                        *input0.get(), *input1.get(), *input2.get(), *input3.get(), *input4.get(), *input5.get(),
                        *input6.get(), *input7.get(), *input8.get(), *input9.get(), *input10.get(), *gradients.get());
                } else {
                    Halide_multi_scale_L2_1_start_0_loss_only(
                        width, height,
                        *input0.get(), *input1.get(), *input2.get(), *input3.get(), *input4.get(), *input5.get(),
                        *input6.get(), *input7.get(), *input8.get(), *input9.get(), *input10.get(), *gradients.get());
                }
            } else if (start_stage == 1) {
                if (get_deriv) {
                    Halide_multi_scale_L2_1_start_1(
                        width, height,
                        *input0.get(), *input1.get(), *input2.get(), *input3.get(), *input4.get(), *input5.get(),
                        *input6.get(), *input7.get(), *input8.get(), *input9.get(), *input10.get(), *gradients.get());
                } else {
                    Halide_multi_scale_L2_1_start_1_loss_only(
                        width, height,
                        *input0.get(), *input1.get(), *input2.get(), *input3.get(), *input4.get(), *input5.get(),
                        *input6.get(), *input7.get(), *input8.get(), *input9.get(), *input10.get(), *gradients.get());
                }
            }
        }
    } else {
        fail = -1;
        std::string sigmas_str = "";
        
        for (int idx = 0; idx < sigmas.size(); idx++) {
            sigmas_str += std::to_string(sigmas[idx]) + ", ";
        }
        
        printf("Error! Kernel for nscale = %d and sigmas = %s is not compiled\n", nscale, sigmas_str.c_str());
    }
    
    if (fail == 0 && get_loss) {
        if (get_deriv) {
            Halide_reduce_single_tensor_2d(width, height, gradients.get()->sliced(2, 3), *loss.get());
        } else {
            Halide_reduce_single_tensor_2d(width, height, gradients.get()->sliced(2, 0), *loss.get());
        }
    }
    
    return fail;
} 