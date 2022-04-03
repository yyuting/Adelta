#include "Halide.h"
#include <stdio.h>

using namespace Halide;

class Reduce : public Halide::Generator<Reduce> {
    
public:
    Input<int> width{"width"};
    Input<int> height{"height"};
    
    Input<Buffer<float>> input{"input", 2};
    Output<Buffer<float>> gradients{"gradients", 1};
    
    void generate() {
        
        gradients(i) = 0.f;
        
        RDom r0(0, width, 0, height);
        
        r = r0;
        
        gradients(i) += input(r.x, r.y);
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
            
            gradients.bound(i, 0, 1).unroll(i);

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
        
        gradients.print_loop_nest();
    }
    
private:
    
    Var i{"i"}, x{"x"}, y{"y"};
    Func intermediate;
    RDom r;

};

HALIDE_REGISTER_GENERATOR(Reduce, reduce)