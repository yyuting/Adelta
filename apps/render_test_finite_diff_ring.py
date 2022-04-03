"""
------------------------------------------------------------------------------------------------------------------------------
# generate opt gt

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ellipse_ring --shader test_finite_diff_ellipse_ring --init_values_pool apps/example_init_values/test_finite_diff_ellipse_ring.npy --modes visualize_gradient --metrics naive_sum --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --is_col --render_size 256,256 --aa_nsamples 1000

------------------------------------------------------------------------------------------------------------------------------
# visualize gradient

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring --shader test_finite_diff_ring --init_values_pool apps/example_init_values/test_finite_diff_ring_ref.npy --modes visualize_gradient --metrics 1_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 256,256 --is_color --backend hl --quiet --no_reset_opt --save_all_loss --niters 200

# comparison with diffvg
python apps/diffvg_ring.py --dir /n/fs/scratch/yutingy/diffvg/results/single_circle_outline_nsamples_2 --init_values_pool apps/example_init_values/test_finite_diff_ring_extra_init_values.npy --nsamples 2 --gt_file /n/fs/scratch/yutingy/test_finite_diff_ellipse_ring/init00004.png

python visualize_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring --gradient_files /n/fs/scratch/yutingy/test_finite_diff_ring/gradient_map.npy,/n/fs/scratch/yutingy/diffvg/results/single_circle_outline_nsamples_2/fd_gradient_wrt_radius.npy,/n/fs/scratch/yutingy/diffvg/results/single_circle_outline_nsamples_2/diffvg_gradient_wrt_radius.npy --names ours,fd,diffvg

------------------------------------------------------------------------------------------------------------------------------
# optimization

python approx_gradient.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring --shader test_finite_diff_ring --init_values_pool apps/example_init_values/test_finite_diff_ring_extra_init_values.npy --modes optimization --metrics 1_scale_L2 --gradient_methods_optimization ours --learning_rate 0.01 --finite_diff_h 0.01 --finite_diff_both_sides --render_size 256,256 --is_color --gt_file /n/fs/scratch/yutingy/test_finite_diff_ellipse_ring/init00004.png --suffix _compare_diffvg --backend hl --quiet --no_reset_opt --save_all_loss --niters 100

------------------------------------------------------------------------------------------------------------------------------
# plot

---------------------------------------------
# preprocess

python preprocess_raw_loss_data.py /n/fs/scratch/yutingy/test_finite_diff_ring /n/fs/scratch/yutingy/test_finite_diff_ring/ours_both_sides_1_scale_L2_adam_1.0e-02_compare_diffvg_all_loss.npy /n/fs/scratch/yutingy/diffvg/results/single_circle_outline_nsamples_2/all_loss.npy

---------------------------------------------
# plot median 

python plot_opt_baseline.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring --filenames ours_both_sides_1_scale_L2_adam_1.0e-02_compare_diffvg_all_loss.npy,all_loss.npy --labels Ours,DVG --scales 0.00078,0.040 --suffix _median --median --xlabel "Runtime (s)" --ylabel "L2 Error"

---------------------------------------------
# plot transparent

python plot_opt_baseline.py --dir /n/fs/scratch/yutingy/test_finite_diff_ring --filenames ours_both_sides_1_scale_L2_adam_1.0e-02_compare_diffvg_all_loss.npy,all_loss.npy --labels Ours,DVG --scales 0.00078,0.040 --suffix _transparent --transparent --xlabel "Runtime (s)" --ylabel "L2 Error"

"""

from render_util import *
from render_single import render_single

compiler.log_prefix_only = False
compiler.log_intermediates_less = True

nargs = 8
args_range = np.array([256., 256., 256., 100, 1., 1., 1., 1.])

def test_finite_diff_ring(u, v, X, width=960, height=640):
    """
    X has shape nargs + 3
    first 3 entries are u, v, time
    the other entries are scalar tunable parameters
    """
    radius = X[0]
    origin_x = X[1]
    origin_y = X[2]
    stroke_width = X[3]
    
    stroke_col = np.array([X[4], X[5], X[6]])
    alpha = X[7]
    stroke_col = stroke_col * alpha
    
    bg_col = np.array([0., 0., 0.])
    
    u_diff = Var('u_diff', u - origin_x)
    v_diff = Var('v_diff', v - origin_y)
    
    dist2 = (u_diff) ** 2 + (v_diff) ** 2
    
    dist = dist2 ** 0.5
    
    cond0 = Var('cond0', dist - radius - stroke_width < 0)
    cond1 = Var('cond1', dist - radius + stroke_width > 0)
    
    col = Var('col', select(cond0 & cond1, stroke_col, bg_col))
    
    return col

shaders = [test_finite_diff_ring]
is_color = True

def sample_init(nsamples, target_width=256, target_height=256):
    
    init = np.zeros([nsamples, nargs])
    
    init[:, 0] = 2 * (np.random.rand(nsamples) * 50 + 15)
    init[:, 1:3] = np.random.rand(nsamples, 2) * 50 + 103
    init[:, 3] = 2 * (np.random.rand(nsamples) * 10 + 3)
    init[:, 4:] = np.random.rand(nsamples, 4)
    
    return init
         
if __name__ == '__main__':
    init = sample_init(100)
    np.save('../test_finite_diff_ring_extra_init_values.npy', init)
                