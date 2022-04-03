// The actual forward + backward pass for the shader program

#include "Halide.h"
#include <stdio.h>

using namespace Halide;

class Layer : public Halide::Generator<Layer> {

public:
    
    Input<Buffer<float>> input{"input", 3};
    Output<Buffer<float>> output{"output", 3};
    
    GeneratorParam<float> sigma{"sigma", /* default value */ 0.5};
    
    std::vector<float> gkern(int half_ksize, float sigma) {
        
        std::vector<float> kernel = {1.f};
        
        float sum = 1.f;
        
        for (int idx = 1; idx <= half_ksize; idx++) {
            
            float current_val = std::exp(-0.5f * pow((float) (idx) / sigma, 2.f));
            
            kernel.push_back(current_val);
            kernel.insert(kernel.begin(), current_val);
            
            sum += current_val * 2.f;
        }
        
        for (int idx = 0; idx < kernel.size(); idx++) {
            kernel[idx] /= sum;
        }
        
        return kernel;
        
    }
            
    void generate() {
        
        Func input_bound = BoundaryConditions::constant_exterior(input, 0.f,
                                                                 {{0, input.width()}, {0, input.height()}});
        
        int half_ksize = (int) (std::ceil(sigma));
        
        std::vector<float> kernel = gkern(half_ksize, sigma);
                
        Expr val0 = 0.f;
                
        for (int k_idx = 0; k_idx < kernel.size(); k_idx++) {

            Expr current_u = u + k_idx - half_ksize;

            val0 += input_bound(current_u, v, c) * kernel[k_idx];
        }

        intermediate(u, v, c) = val0;
        
        Expr val1 = 0.f;
                
        for (int k_idx = 0; k_idx < kernel.size(); k_idx++) {

            Expr current_v = v + k_idx - half_ksize;

            val1 += intermediate(u, current_v, c) * kernel[k_idx];
        }

        output(u, v, c) = val1;
        
    }
    
    void schedule() {
        if (auto_schedule) {
            input.set_estimates({{0, 960}, {0, 640}, {0, 3}});
            output.set_estimates({{0, 960}, {0, 640}, {0, 3}});
        } else {
            
            Var u_i("u_i");
            Var u_o("u_o");
            Var v_i("v_i");
            Var v_o("v_o");
            
            intermediate
                .reorder(c, u, v)
                .compute_at(output, u_o)
                .gpu_threads(u)
                .gpu_threads(v);
            
            output
                .compute_root()
                .split(u, u_o, u_i, 4)
                .split(v, v_o, v_i, 8)
                .reorder(c, u_i, v_i, u_o, v_o)
                .gpu_threads(u_i)
                .gpu_threads(v_i)
                .gpu_blocks(v_o)
                .gpu_blocks(u_o);
        }
    }
    
private:
    
    Var u{"u"}, v{"v"}, c{"c"}, p{"p"};
    Func intermediate;
};

HALIDE_REGISTER_GENERATOR(Layer, layer)