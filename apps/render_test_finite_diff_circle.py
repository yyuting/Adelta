"""
------------------------------------------------------------------------------------------------------------------------------
# generate gt

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ellipse --shader test_finite_diff_ellipse --init_values_pool apps/example_init_values/test_finite_diff_ellipse.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_col --render_size 256,256 --aa_nsamples 1000

------------------------------------------------------------------------------------------------------------------------------
# visualize gradient

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_circle --shader test_finite_diff_circle --init_values_pool apps/example_init_values/test_finite_diff_circle_extra_init_values.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_col --render_size 256,256

------------------------------------------------------------------------------------------------------------------------------
# optimization

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_circle --shader test_finite_diff_circle --init_values_pool apps/example_init_values/test_finite_diff_circle_extra_init_values.npy --modes optimization --metrics 1_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 256,256 --is_color --gt_file /n/fs/scratch/yutingy/test_finite_diff_ellipse/init00004.png --suffix _compare_diffvg --backend hl --quiet --no_reset_opt --save_all_loss --niters 100

------------------------------------------------------------------------------------------------------------------------------
# comparison with diffvg

python apps/diffvg_circle.py --dir /n/fs/scratch/yutingy/diffvg/results/single_circle_nsamples_2 --init_values_pool apps/example_init_values/test_finite_diff_circle_extra_init_values.npy --nsamples 2 --gt_file /n/fs/scratch/yutingy/test_finite_diff_ellipse/init00004.png

------------------------------------------------------------------------------------------------------------------------------
# plot

---------------------------------------------
# preprocess

python preprocess_raw_loss_data.py /n/fs/scratch/yutingy/test_finite_diff_circle /n/fs/scratch/yutingy/test_finite_diff_circle/ours_both_sides_1_scale_L2_adam_1.0e-02_compare_diffvg_all_loss.npy /n/fs/scratch/yutingy/diffvg/results/single_circle_nsamples_2/all_loss.npy

---------------------------------------------
# plot median 

python plot_opt_baseline.py --dir /n/fs/scratch/yutingy/test_finite_diff_circle --filenames ours_both_sides_1_scale_L2_adam_1.0e-02_compare_diffvg_all_loss.npy,all_loss.npy --labels Ours,DVG --scales 0.00066,0.033 --suffix _median --median --xlabel "Runtime (s)" --ylabel "L2 Error"

---------------------------------------------
# plot transparent

python plot_opt_baseline.py --dir /n/fs/scratch/yutingy/test_finite_diff_circle --filenames ours_both_sides_1_scale_L2_adam_1.0e-02_compare_diffvg_all_loss.npy,all_loss.npy --labels Ours,DVG --scales 0.00066,0.033 --suffix _transparent --transparent --xlabel "Runtime (s)" --ylabel "L2 Error"

"""

from render_util import *
from render_single import render_single

compiler.log_prefix_only = False
compiler.log_intermediates_less = True

def cmd_template():
    
    cmd = f"""python approx_gradient.py --shader test_finite_diff_circle --init_values_pool apps/example_init_values/test_finite_diff_circle_extra_init_values.npy --metrics 1_scale_L2 --render_size 256,256 --is_color --niters 100"""
    
    return cmd

init_name = 'test_finite_diff_ellipse'
default_size = [256,256]
default_opt_arg = '1_scale_L2'

nargs = 7
args_range = np.array([256., 256., 256., 1., 1., 1., 1.])

def test_finite_diff_circle(u, v, X, width=960, height=640):

    radius = X[0]
    origin_x = X[1]
    origin_y = X[2]
    
    fill_col = np.array([X[3], X[4], X[5]])
    alpha = X[6]
    fill_col = fill_col * alpha
    
    bg_col = np.array([0., 0., 0.])
    
    u_diff = Var('u_diff', u - origin_x)
    v_diff = Var('v_diff', v - origin_y)
    
    dist2 = (u_diff) ** 2 + (v_diff) ** 2
    
    radius2 = radius ** 2
    
    cond_diff = Var('cond_diff', dist2 - radius2)
    
    col = Var('col', select(cond_diff < 0, fill_col, bg_col))
    
    return col

shaders = [test_finite_diff_circle]
is_color = True

def sample_init(nsamples, target_width=256, target_height=256):
    
    init = np.zeros([nsamples, nargs])
    
    init[:, 0] = np.random.rand(nsamples) * 50 + 15
    init[:, 1:3] = np.random.rand(nsamples, 2) * 50 + 103
    init[:, 3:] = np.random.rand(nsamples, 4)
    
    return init
         
if __name__ == '__main__':
    init = sample_init(100)
    np.save('../test_finite_diff_circle_extra_init_values.npy', init)
                
