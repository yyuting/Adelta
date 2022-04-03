// The actual forward + backward pass for the shader program

#include "Halide.h"
#include <stdio.h>

using namespace Halide;

class Layer : public Halide::Generator<Layer> {

public:
    
    Input<Buffer<float>> input{"input", 3};
    Output<Buffer<float>> output{"output", 3};
    
    GeneratorParam<int> scale{"scale", /* default value */ 2};
        
    void generate() {
        
        Func input_bound = BoundaryConditions::constant_exterior(input, 0.f,
                                                                 {{0, input.width()}, {0, input.height()}});
        
        RDom downsample(0, scale, 0, scale);
        
        Expr coef = 1.f / ((float) scale * (float) scale);
        
        output(u, v, c) += coef * input_bound(scale * u + downsample.x, scale * v + downsample.y, c);
    }
    
    void schedule() {
        if (auto_schedule) {
            input.set_estimates({{0, 960}, {0, 640}, {0, 3}});
            output.set_estimates({{0, 480}, {0, 320}, {0, 3}});
        } else {
            Pipeline pipeline = get_pipeline();

            Var u_i("u_i");
            Var u_o("u_o");
            Var v_i("v_i");
            Var v_o("v_o");

            Func output = pipeline.get_func(4);

            {
                Var u = output.args()[0];
                Var v = output.args()[1];
                RVar r16$x(output.update(0).get_schedule().rvars()[0].var);
                RVar r16$y(output.update(0).get_schedule().rvars()[1].var);
                output
                    .compute_root()
                    .split(u, u_o, u_i, 2)
                    .split(v, v_o, v_i, 2)
                    .reorder(c, u_i, v_i, u_o, v_o)
                    .gpu_threads(u_i)
                    .gpu_threads(v_i)
                    .gpu_blocks(v_o)
                    .gpu_blocks(u_o)
                    .compute_root();
                output.update(0)
                    .split(u, u_o, u_i, 2, TailStrategy::RoundUp)
                    .split(v, v_o, v_i, 2, TailStrategy::RoundUp)
                    .reorder(r16$x, r16$y, c, u_i, v_i, u_o, v_o)
                    .gpu_threads(u_i)
                    .gpu_threads(v_i)
                    .unroll(r16$x)
                    .unroll(r16$y)
                    .gpu_blocks(v_o)
                    .gpu_blocks(u_o);
            }
        }
    }
    
private:
    
    Var u{"u"}, v{"v"}, c{"c"}, p{"p"};
};

HALIDE_REGISTER_GENERATOR(Layer, layer)