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
    
    __nscale_L2_main_body__
    else {
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