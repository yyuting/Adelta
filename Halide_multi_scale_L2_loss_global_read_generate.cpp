// The actual forward + backward pass for the shader program

#include "Halide.h"
#include <stdio.h>

#define MAX_NINPUT1 10

using namespace Halide;

class Loss : public Halide::Generator<Loss> {
    
public:
    
    Input<int> width{"width"};
    Input<int> height{"height"};
    
    Input<Buffer<float>> input0{"input0", 3};
    Input<Buffer<float>[MAX_NINPUT1]> input1{"input1", 3};
    
    Output<Buffer<float>> dL_dcol{"dL_dcol", 3};
    
    GeneratorParam<int> nscale{"nscale", /* default value */ 0};
    GeneratorParam<std::string> smoothing_sigmas{"smoothing_sigmas", /* default value */ ""};
    GeneratorParam<int> start_stage{"start_stage", /* default value */ 0};
    
    GeneratorParam<bool> loss_only{"loss_only", /* default value */ false};
    
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
        
        c_bound = 4;
        
        std::string sigma_str = smoothing_sigmas;
        
        printf("%s\n", sigma_str.c_str());
        std::vector<int> valid_boundary;
        
        if (sigma_str.compare("") != 0) {
            std::stringstream ss(sigma_str);

            while( ss.good() )
            {
                std::string substr;
                getline( ss, substr, ',' );
                sigma_fl.push_back( std::stof(substr) );
            }
        }
        
        Func diff("diff");
        diff(u, v, c) = (input0(u, v, c) - input1[0](u, v, c));
        
        RDom downsample(0, 2, 0, 2);

        for (int level = 0; level < nscale; level++) {
            
            Func dx[2];
            
            if (level == 0) {
                dx[0](u, v, c) += 0.25f * input0(2 * u + downsample.x, 2 * v + downsample.y, c);
            } else {
                for (int i = 0; i < 1; i++) {
                    dx[i](u, v, c) += 0.25f * f_dx[level-1](2 * u + downsample.x, 2 * v + downsample.y, c);
                }
            }
            
            f_dx.push_back(dx[0]);
            
            Func current_diff("current_diff_" + std::to_string(level));
            current_diff(u, v, c) = (f_dx[level](u, v, c) - input1[level+1](u, v, c));
            
            current_diffs.push_back(current_diff);
        }
        
        if (sigma_fl.size() > 0) {
            
            Func input0_bound;
            
            Expr final_width, final_height;
            
            int scale = pow(2, nscale);
            
            final_width = Halide::cast<int> (width / scale);
            final_height = Halide::cast<int> (height / scale);
            
            Expr sum_blur_vals;
            Expr sum_blur_loss;
            
            if (nscale == 0) {
                input0_bound = BoundaryConditions::constant_exterior(input0, 0.f,
                                                                       {{0, final_width}, {0, final_height}});
                
                sum_blur_vals = 0.f;
                sum_blur_loss = 0.f;
                
            } else {
                input0_bound = BoundaryConditions::constant_exterior(f_dx[nscale-1], 0.f,
                                                                       {{0, final_width}, {0, final_height}});
                
                if ((int) start_stage <= (int) nscale) {
                    sum_blur_vals = current_diffs[nscale-1](u, v, c);
                    sum_blur_loss = 0.5f * current_diffs[nscale-1](u, v, c) * current_diffs[nscale-1](u, v, c);
                } else {
                    sum_blur_vals = 0.f;
                    sum_blur_loss = 0.f;
                }
            }

            for (int idx = 0; idx < sigma_fl.size(); idx++) {

                float current_sigma = sigma_fl[idx];

                printf("sigma: %f\n", current_sigma);

                int half_ksize = (int) (std::ceil(current_sigma));

                std::vector<float> kernel = gkern(half_ksize, current_sigma);
                
                Func intermediate("intermediate_" + std::to_string(idx));
                Func blurred("blurred_" + std::to_string(idx));
                
                Expr val0 = 0.f;
                
                for (int k_idx = 0; k_idx < kernel.size(); k_idx++) {

                    Expr current_u = u + k_idx - half_ksize;
                    
                    val0 += input0_bound(current_u, v, c) * kernel[k_idx];
                }
                
                intermediate(u, v, c) = val0;
                
                Expr val1 = 0.f;
                
                for (int k_idx = 0; k_idx < kernel.size(); k_idx++) {

                    Expr current_v = v + k_idx - half_ksize;

                    val1 += intermediate(u, current_v, c) * kernel[k_idx];
                }
                
                blurred(u, v, c) = val1;
                
                f_dx.push_back(blurred);
                intermediates.push_back(intermediate);
                
                Func current_diff("current_diff_blur_" + std::to_string(idx));
                
                current_diff(u, v, c) = (blurred(u, v, c) - input1[nscale+idx+1](u, v, c));
                
                current_diffs.push_back(current_diff);
                
                Func diff_bound = BoundaryConditions::constant_exterior(current_diff, 0.f, 
                                                                        {{0, final_width}, {0, final_height}});
                
                Expr blur_val0 = 0.f;
                
                for (int k_idx = 0; k_idx < kernel.size(); k_idx++) {

                    Expr current_v = v + k_idx - half_ksize;

                    blur_val0 += diff_bound(u, current_v, c) * kernel[k_idx];
                }
                
                Func d_intermediate("d_intermediate_" + std::to_string(idx));
                d_intermediate(u, v, c) = blur_val0;
                
                d_intermediates.push_back(d_intermediate);
                
                Expr blur_val1 = 0.f;
                
                for (int k_idx = 0; k_idx < kernel.size(); k_idx++) {

                    Expr current_u = u + k_idx - half_ksize;

                    blur_val1 += d_intermediate(current_u, v, c) * kernel[k_idx];
                }

                Func d_diff_blur("d_diff_blur_" + std::to_string(idx));

                d_diff_blur(u, v, c) = blur_val1;
                
                if (start_stage <= nscale + idx + 1) {
                    
                    sum_blur_vals += d_diff_blur(u, v, c);
                
                    //sum_blur_loss += 0.5f * d_diff_blur(u, v, c) * d_diff_blur(u, v, c);
                    sum_blur_loss += 0.5f * current_diff(u, v, c) * current_diff(u, v, c);
                }
            
                d_diff_blurs.push_back(d_diff_blur);

            }
            
            d_lowest_res(u, v, c, p) = mux(p, {sum_blur_vals, sum_blur_loss});
        }
        
        Expr val, loss;
        
        if (start_stage == 0) {
            val = diff(u, v, c);
            loss = 0.5f * diff(u, v, c) * diff(u, v, c);
        } else {
            val = 0.f;
            loss = 0.f;
        }
        
        for (int level = 0; level < nscale; level++) {

            int current_scale = pow(2, level + 1);
            
            if (level == nscale - 1 && sigma_fl.size() > 0) {
                val += d_lowest_res(u / current_scale, v / current_scale, c, 0);
                loss += d_lowest_res(u / current_scale, v / current_scale, c, 1);
            } else {
                if (start_stage <= level + 1) {
                    val += current_diffs[level](u / current_scale, v / current_scale, c);
                    loss += 0.5f * current_diffs[level](u / current_scale, v / current_scale, c) * \
                        current_diffs[level](u / current_scale, v / current_scale, c);
                }
            }
        }
        
        val /= Halide::cast<float> (3 * width * height);
        loss /= Halide::cast<float> (3 * width * height);
        
        printf("before defining per_channel_loss\n");
        
        per_channel_loss(u, v, c) = loss;
        per_channel_deriv(u, v, c) = val;
        
        printf("after defining per_channel_loss\n");
        
        if (loss_only) {
            dL_dcol(u, v, c) = select(c > 0,
                                      Halide::undef<float>(), 
                                      per_channel_loss(u, v, 0) + per_channel_loss(u, v, 1) + per_channel_loss(u, v, 2));
        }
        else if (sigma_fl.size() > 0) {
            printf("entering sigma_fl > 0\n");
            int current_scale = pow(2, nscale);
            //dL_dcol(u, v, c, p) = mux(p, {val, loss}) + d_lowest_res(u / current_scale, v / current_scale, c, p);
            //dL_dcol(u, v, c) = select(c < 3,
            //                          val,
            //                          per_channel_loss(u, v, 0) + per_channel_loss(u, v, 1) + per_channel_loss(u, v, 2));
            dL_dcol(u, v, c) = mux(c, {per_channel_deriv(u, v, 0),
                                       per_channel_deriv(u, v, 1),
                                       per_channel_deriv(u, v, 2),
                                       per_channel_loss(u, v, 0) + per_channel_loss(u, v, 1) + per_channel_loss(u, v, 2)});
                                          
        } else {
            //dL_dcol(u, v, c, p) = mux(p, {val, loss});
            dL_dcol(u, v, c) = mux(c, {per_channel_deriv(u, v, 0),
                                       per_channel_deriv(u, v, 1),
                                       per_channel_deriv(u, v, 2),
                                       per_channel_loss(u, v, 0) + per_channel_loss(u, v, 1) + per_channel_loss(u, v, 2)});
        }
    }
    
    void schedule() {
        if (auto_schedule) {

            input0.set_estimates({{0, 960}, {0, 640}, {0, 3}});
            
            for (int idx = 0; idx < MAX_NINPUT1; idx++) {
                int scale;
                if (idx <= nscale) {
                    scale = pow(2, idx);
                } else {
                    scale = pow(2, nscale);
                }
                input1[idx].set_estimates({{0, (int) (960 / scale)}, {0, (int) (640 / scale)}, {0, 3}});
            }
            
            dL_dcol.set_estimates({{0, 960}, {0, 640}, {0, 3}});
            width.set_estimate(960);
            height.set_estimate(640);
            

        } else {
            
            int p_bound = 2;
            
            printf("p_bound: %d, c_bound: %d", p_bound, c_bound);
            
            Var u_i("u_i");
            Var u_o("u_o");
            Var v_i("v_i");
            Var v_o("v_o");
            
            for (int idx = 0; idx < nscale; idx++) {
                Var u = f_dx[idx].args()[0];
                Var v = f_dx[idx].args()[1];
                RVar r40$x(f_dx[idx].update(0).get_schedule().rvars()[0].var);
                RVar r40$y(f_dx[idx].update(0).get_schedule().rvars()[1].var);
                f_dx[idx]
                    .compute_root()
                    .split(u, u_o, u_i, 16)
                    .split(v, v_o, v_i, 8)
                    .reorder(c, u_i, v_i, u_o, v_o)
                    .gpu_threads(u_i)
                    .gpu_threads(v_i)
                    .gpu_blocks(v_o)
                    .gpu_blocks(u_o)
                    .compute_root();
                f_dx[idx].update(0)
                    .split(u, u_o, u_i, 16, TailStrategy::RoundUp)
                    .split(v, v_o, v_i, 8, TailStrategy::RoundUp)
                    .reorder(r40$x, r40$y, c, u_i, v_i, u_o, v_o)
                    .gpu_threads(u_i)
                    .gpu_threads(v_i)
                    .unroll(r40$x)
                    .unroll(r40$y)
                    .gpu_blocks(v_o)
                    .gpu_blocks(u_o);
            }

            {
                Var u = dL_dcol.args()[0];
                Var v = dL_dcol.args()[1];
                
                dL_dcol.reorder(c, u, v).bound(c, 0, c_bound).unroll(c);
                
                dL_dcol
                    .compute_root()
                    .split(u, u_o, u_i, 32)
                    .split(v, v_o, v_i, 32)
                    //.reorder(c, u_i, v_i, u_o, v_o)
                    .reorder(c, u_i, v_i, u_o, v_o)
                    .gpu_threads(u_i)
                    .gpu_threads(v_i)
                    .gpu_blocks(v_o)
                    .gpu_blocks(u_o);
            }
            
            if (sigma_fl.size() > 0) {
                
                for (int idx = 0; idx < sigma_fl.size(); idx++) {
                    {
                        Var u = current_diffs[nscale+idx].args()[0];
                        Var v = current_diffs[nscale+idx].args()[1];
                        Var c = current_diffs[nscale+idx].args()[2];

                        current_diffs[nscale+idx]
                            .compute_root()
                            .split(u, u_o, u_i, 2)
                            .split(v, v_o, v_i, 2)
                            .reorder(c, u_i, v_i, u_o, v_o)
                            .gpu_threads(u_i)
                            .gpu_threads(v_i)
                            .gpu_blocks(v_o)
                            .gpu_blocks(u_o);
                    }
                    {
                        Var u = d_intermediates[idx].args()[0];
                        Var v = d_intermediates[idx].args()[1];
                        Var c = d_intermediates[idx].args()[2];
                        d_intermediates[idx]
                            .reorder(c, u, v)
                            .compute_at(d_lowest_res, u_o)
                            .gpu_threads(u)
                            .gpu_threads(v);
                    }
                    {
                        Var u = intermediates[idx].args()[0];
                        Var v = intermediates[idx].args()[1];
                        Var c = intermediates[idx].args()[2];
                        intermediates[idx]
                            .reorder(c, u, v)
                            .compute_at(current_diffs[nscale+idx], u_o)
                            .gpu_threads(u)
                            .gpu_threads(v);
                    }
                }
                
                {
                    Var u = d_lowest_res.args()[0];
                    Var v = d_lowest_res.args()[1];
                    
                    d_lowest_res.reorder(p, c, u, v).bound(p, 0, p_bound).unroll(p);
                    
                    d_lowest_res
                        .compute_root()
                        .split(u, u_o, u_i, 2)
                        .split(v, v_o, v_i, 2)
                        .reorder(c, u_i, v_i, u_o, v_o)
                        .gpu_threads(u_i)
                        .gpu_threads(v_i)
                        .gpu_blocks(v_o)
                        .gpu_blocks(u_o);
                }
                
            }
            
            /*
            {
                Var u = f0.args()[0];
                Var v = f0.args()[1];
                RVar r40$x(f0.update(0).get_schedule().rvars()[0].var);
                RVar r40$y(f0.update(0).get_schedule().rvars()[1].var);
                f0
                    .compute_root()
                    .split(u, u_o, u_i, 16)
                    .split(v, v_o, v_i, 8)
                    .reorder(c, u_i, v_i, u_o, v_o)
                    .gpu_threads(u_i)
                    .gpu_threads(v_i)
                    .gpu_blocks(v_o)
                    .gpu_blocks(u_o)
                    .compute_root();
                f0.update(0)
                    .split(u, u_o, u_i, 16, TailStrategy::RoundUp)
                    .split(v, v_o, v_i, 8, TailStrategy::RoundUp)
                    .reorder(r40$x, r40$y, c, u_i, v_i, u_o, v_o)
                    .gpu_threads(u_i)
                    .gpu_threads(v_i)
                    .unroll(r40$x)
                    .unroll(r40$y)
                    .gpu_blocks(v_o)
                    .gpu_blocks(u_o);
            }
            {
                Var u = f2.args()[0];
                Var v = f2.args()[1];
                RVar r40$x(f2.update(0).get_schedule().rvars()[0].var);
                RVar r40$y(f2.update(0).get_schedule().rvars()[1].var);
                f2
                    .compute_root()
                    .split(u, u_o, u_i, 16)
                    .split(v, v_o, v_i, 8)
                    .reorder(c, u_i, v_i, u_o, v_o)
                    .gpu_threads(u_i)
                    .gpu_threads(v_i)
                    .gpu_blocks(v_o)
                    .gpu_blocks(u_o)
                    .compute_root();
                f2.update(0)
                    .split(u, u_o, u_i, 16, TailStrategy::RoundUp)
                    .split(v, v_o, v_i, 8, TailStrategy::RoundUp)
                    .reorder(r40$x, r40$y, c, u_i, v_i, u_o, v_o)
                    .gpu_threads(u_i)
                    .gpu_threads(v_i)
                    .unroll(r40$x)
                    .unroll(r40$y)
                    .gpu_blocks(v_o)
                    .gpu_blocks(u_o);
            }
            {
                Var u = f4.args()[0];
                Var v = f4.args()[1];
                RVar r40$x(f4.update(0).get_schedule().rvars()[0].var);
                RVar r40$y(f4.update(0).get_schedule().rvars()[1].var);
                f4
                    .compute_root()
                    .split(u, u_o, u_i, 16)
                    .split(v, v_o, v_i, 8)
                    .reorder(c, u_i, v_i, u_o, v_o)
                    .gpu_threads(u_i)
                    .gpu_threads(v_i)
                    .gpu_blocks(v_o)
                    .gpu_blocks(u_o)
                    .compute_root();
                f4.update(0)
                    .split(u, u_o, u_i, 16, TailStrategy::RoundUp)
                    .split(v, v_o, v_i, 8, TailStrategy::RoundUp)
                    .reorder(r40$x, r40$y, c, u_i, v_i, u_o, v_o)
                    .gpu_threads(u_i)
                    .gpu_threads(v_i)
                    .unroll(r40$x)
                    .unroll(r40$y)
                    .gpu_blocks(v_o)
                    .gpu_blocks(u_o);
            }
            {
                Var u = f6.args()[0];
                Var v = f6.args()[1];
                RVar r40$x(f6.update(0).get_schedule().rvars()[0].var);
                RVar r40$y(f6.update(0).get_schedule().rvars()[1].var);
                f6
                    .compute_root()
                    .split(u, u_o, u_i, 16)
                    .split(v, v_o, v_i, 4)
                    .reorder(c, u_i, v_i, u_o, v_o)
                    .gpu_threads(u_i)
                    .gpu_threads(v_i)
                    .gpu_blocks(v_o)
                    .gpu_blocks(u_o)
                    .compute_root();
                f6.update(0)
                    .split(u, u_o, u_i, 16, TailStrategy::RoundUp)
                    .split(v, v_o, v_i, 4, TailStrategy::RoundUp)
                    .reorder(r40$x, r40$y, c, u_i, v_i, u_o, v_o)
                    .gpu_threads(u_i)
                    .gpu_threads(v_i)
                    .unroll(r40$x)
                    .unroll(r40$y)
                    .gpu_blocks(v_o)
                    .gpu_blocks(u_o);
            }
            {
                Var u = f8.args()[0];
                Var v = f8.args()[1];
                Var c = f8.args()[2];
                RVar r40$x(f8.update(0).get_schedule().rvars()[0].var);
                RVar r40$y(f8.update(0).get_schedule().rvars()[1].var);
                f8
                    .compute_root()
                    .split(u, u_o, u_i, 16)
                    .split(v, v_o, v_i, 4)
                    .reorder(c, u_i, v_i, u_o, v_o)
                    .gpu_threads(u_i)
                    .gpu_threads(v_i)
                    .gpu_blocks(v_o)
                    .gpu_blocks(u_o)
                    .compute_root();
                f8.update(0)
                    .split(u, u_o, u_i, 16, TailStrategy::RoundUp)
                    .split(v, v_o, v_i, 4, TailStrategy::RoundUp)
                    .reorder(r40$x, r40$y, c, u_i, v_i, u_o, v_o)
                    .gpu_threads(u_i)
                    .gpu_threads(v_i)
                    .unroll(r40$x)
                    .unroll(r40$y)
                    .gpu_blocks(v_o)
                    .gpu_blocks(u_o);
            } */
        }
        
        dL_dcol.print_loop_nest();
        
        printf("finished schedule\n");
    }
    
private:
    
    Var u{"u"}, v{"v"}, c{"c"}, p{"p"};
    
    int c_bound;
    
    std::vector<Func> current_diffs, f_dx, intermediates, d_intermediates, d_diff_blurs;
    Func d_lowest_res, l_lowest_res, per_channel_loss, per_channel_deriv;
    
    std::vector<float> sigma_fl;
    
};

HALIDE_REGISTER_GENERATOR(Loss, loss)
    