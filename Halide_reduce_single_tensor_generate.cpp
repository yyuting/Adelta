#include "Halide.h"
#include <stdio.h>

using namespace Halide;

class Reduce : public Halide::Generator<Reduce> {
    
public:
    Input<int> width{"width"};
    Input<int> height{"height"};
    
    Input<Buffer<float>> input{"input", 3};
    Output<Buffer<float>> gradients{"gradients", 1};
    
    GeneratorParam<int> start_idx{"start_idx", /* default value */ 0};
    
    void generate() {
        
        gradients(i) = 0.f;
        
        RDom r0(0, width, 0, height);
        
        r = r0;
        
        gradients(i) += input(r.x, r.y, i + start_idx);
    }
    
    void schedule() {
        
        if (get_target().has_gpu_feature()) {
            intermediate = gradients.update().rfactor({{r.x, x}});
            Var xi;
            intermediate
                .in()
                .compute_root()
                .gpu_tile(x, xi, 32);
            Var io, ii;
            gradients
                .compute_root()
                .gpu_tile(i, io, ii, 1)
                .update()
                .gpu_tile(i, io, ii, 1);
        } else {
            intermediate = gradients.update().rfactor({{r.x, x}});
            intermediate.compute_at(gradients, i)
                .vectorize(x, 8)
                .update()
                .vectorize(x, 8);
            gradients.vectorize(i, 8)
                .update()
                .parallel(i);
        }
    }
    
private:
    
    Var i{"i"}, x{"x"}, y{"y"};
    Func intermediate;
    RDom r;

};

HALIDE_REGISTER_GENERATOR(Reduce, reduce)