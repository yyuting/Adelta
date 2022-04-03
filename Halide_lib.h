#include "HalideBuffer.h"
#include "halide_benchmark.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <utility>

#include <iostream>
#include <string>
#include <vector>

#include "Halide.h"

#include "halide_image_io.h"

using namespace Halide::Tools;
using namespace Halide;

void set_host_dirty(Buffer<float> input);
void copy_to_host(Buffer<float> input);

int reduce_sum(int start_idx,
               int width, int height,
               Buffer<float> input,
               Buffer<float> output,
               bool check_ok=false);

int gaussian_conv(float sigma,
                  Buffer<float> input,
                  Buffer<float> output,
                  bool check_ok=false);

int downsample(int scale,
               Buffer<float> input,
               Buffer<float> output,
               bool check_ok=false);

int nscale_L2(int nscale, 
              std::vector<float> sigmas,
              int width, int height,
              Buffer<float> input0, Buffer<float> input1, Buffer<float> input2, Buffer<float> input3, 
              Buffer<float> input4, Buffer<float> input5, Buffer<float> input6, Buffer<float> input7,
              Buffer<float> input8, Buffer<float> input9, Buffer<float> input10, Buffer<float> gradients, 
              Buffer<float> loss, bool get_loss=true,
              int start_stage=0,
              bool get_deriv=true,
              bool check_ok=false);