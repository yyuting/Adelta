import numpy as np
import os
import sys
sys.path += ['util']
import argparse_util
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import skimage.io
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
import matplotlib.font_manager
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

def scientific_format(val):
    base_strs = ('%.2e' % val).split('e')
    
    ans = '$%s \\times 10^{%d}$' % (base_strs[0], int(base_strs[1]))
    
    return ans

def parse():
    
    parser = argparse_util.ArgumentParser(description='metric compare line integral')
    
    parser.add_argument('--baseline_dir', dest='baseline_dir', default='', help='specifies the dir to find finite diff baseline datas')
    parser.add_argument('--eval_labels', dest='eval_labels', default='', help='specifies the labels for each method, seperated by comma')
    parser.add_argument('--visualization_thre', dest='visualization_thre', type=float, default=0.1, help='specifies the threshold used to visualize error')
    parser.add_argument('--max_half_len', dest='max_half_len', type=float, default=0.1, help='specifies half length of the line integral')
    parser.add_argument('--rhs_file', dest='rhs_file', default='', help='specifies the rhs file used to assemble err')
    parser.add_argument('--visualization_suffix', dest='visualization_suffix', default='', help='a suffix append to visualization filename')
    parser.add_argument('--deriv_metric_suffix', dest='deriv_metric_suffix', default='', help='specifies the suffix appending after the original error data')
    parser.add_argument('--visualize_img_name', dest='visualize_img_name', default='', help='if specified, use the file to visualize fw rendering')
    parser.add_argument('--ncols', dest='ncols', type=int, default=5, help='specifies the number of columns per row')
    parser.add_argument('--crop_ratio', dest='crop_ratio', type=float, default=0, help='specifies the ratio to crop edge')
                        
    
    args = parser.parse_args()
    
    return args
    
def collect_data(args):
        
    eval_labels = [val for val in args.eval_labels.split(',')]
        
    if ',' in args.deriv_metric_suffix:
        metric_suffixes = args.deriv_metric_suffix.split(',')
    else:
        metric_suffixes = [args.deriv_metric_suffix]
    
    assert args.rhs_file != ''
    derivs_rhs = np.load(args.rhs_file)
    
    
    
    err_datas = []
    
    def get_base_name(metric_suffix):

        ans = 'kernel_smooth_metric_debug_10000X1_len_%f_kernel_box_sigma_1.000000_0.100000' % (args.max_half_len)

        ans = '%s%s' % (ans, metric_suffix)
        
        return ans
        
    def get_err_filename(metric_suffix):
        suffix = ''

        filename = '%s%s.npy' % (get_base_name(metric_suffix), suffix)
            
        return filename
        
    for eval_idx in range(len(eval_labels)):
            
        metric_suffix = metric_suffixes[eval_idx]
        
        current_err = np.load(os.path.join(args.baseline_dir, get_err_filename(metric_suffix)))
        
        err_datas.append(current_err)
        
    return err_datas
        
def visualize_mixed(args, err_datas):
    
    eval_labels = [val for val in args.eval_labels.split(',')]
    
    assert args.rhs_file != ''
    derivs_rhs = np.load(args.rhs_file)
    
    if args.crop_ratio > 0:
        crops = [int(derivs_rhs.shape[0] * args.crop_ratio),
                 int(derivs_rhs.shape[1] * args.crop_ratio)]
        derivs_rhs = derivs_rhs[crops[0]:-crops[0], crops[1]:-crops[1]]
    else:
        crops = None
    
    visualize_img_name = args.visualize_img_name
    
    if os.path.exists(visualize_img_name):
        has_visualize = True
        visualize_img = skimage.io.imread(visualize_img_name)
        extra_subplot = 1
    else:
        has_visualize = False
        extra_subplot = 0
        print('visualization missing, ', visualize_img_name)
        
    if len(err_datas[0].shape) == 2:
        nchannels = 1
    else:
        assert len(err_datas[0].shape) == 3
        nchannels = err_datas[0].shape[2]
        assert nchannels in [1, 3]
        
    if nchannels == 3:
        is_color = True
    else:
        is_color = False
        
    err_thre = args.visualization_thre
    
    sample_count = 0

    non_sample_count = len(eval_labels)
    
    mixed_grid = False

    nsubplots = non_sample_count + extra_subplot

    nrows = int(np.ceil(nsubplots / args.ncols))

    if nrows > 1:
        ncols = args.ncols
        mixed_grid = True
    else:
        ncols = nsubplots

            
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))
    
    if nrows == 1:
        axes = np.array([axes])
        if ncols == 1:
            axes = np.array([axes])
    
    if has_visualize:
            
        if is_color:
            axes[0][0].imshow(visualize_img)
        else:
            axes[0][0].imshow(visualize_img, cmap='gray')
        axes[0][0].title.set_text('Shader Program Rendering')
        
        dummy_fig = plt.figure()
        plt.figure(dummy_fig.number)
        plt.imshow(visualize_img, cmap='gray')
        plt.axis('off')
        plt.savefig(os.path.join(args.baseline_dir, 'rendering_visualization.png'), bbox_inches='tight')
        
        plt.figure(fig.number)
        
    collected_mean = []
    flat_count = 0
        
    for idx in range(len(err_datas)):
        
        current_err = err_datas[idx]
        
        if crops is not None:
            current_err = current_err[crops[0]:-crops[0], crops[1]:-crops[1]]
        
        title = eval_labels[idx]

        if nrows == 1:
            ax = axes[0][idx + extra_subplot]
        else:
            current_row = flat_count // ncols
            current_col = flat_count % ncols
            flat_count += 1

            ax = axes[current_row][current_col]

        current_err = np.abs(current_err - derivs_rhs)

        if is_color:
            current_err = np.mean(current_err, -1)

        current_err = current_err.transpose()

        if err_thre > 0:
            im = ax.imshow(current_err, vmin=0, vmax=err_thre, cmap='hot')
        else:
            im = ax.imshow(-current_err, vmin=err_thre, vmax=0, cmap='hot')
        
        current_err_nonzero = current_err[current_err != 0]

        #mean_mean = np.mean(current_err_nonzero)
        mean_mean = np.mean(current_err)

        ax.title.set_text('%s\nmean err: %s' % (title, scientific_format(mean_mean)))
        
        dummy_fig = plt.figure()
        plt.figure(dummy_fig.number)
        if err_thre > 0:
            plt.imshow(current_err, vmin=0, vmax=err_thre, cmap='hot')
            plt.axis('off')
            plt.savefig(os.path.join(args.baseline_dir, '%s.png' % title), bbox_inches='tight')
        
        plt.figure(fig.number)
            
    if nrows == 1:
        axes = axes[0]
        
    for ax in axes.ravel().tolist():
        ax.set_xticks([])
        ax.set_yticks([])
            
    fig.colorbar(im, ax=axes.ravel().tolist(), location="bottom")
        
    plt.savefig(os.path.join(args.baseline_dir, 'deriv_metric_visualization_combined%s.png' % (args.visualization_suffix)))  
    
    # create a standalone colorbar
    # https://stackoverflow.com/questions/9295026/matplotlib-plots-removing-axis-legends-and-white-spaces
    dummy = np.array([[0,1]])
    plt.figure(figsize=(0.5, 9))
    if err_thre > 0:
        img = plt.imshow(dummy, cmap="hot", vmin=0, vmax=err_thre)
        plt.gca().set_visible(False)
        cax = plt.axes([0.1, 0.2, 0.8, 0.6])
        cbar = plt.colorbar(orientation="vertical", cax=cax)
        cbar.set_ticks([])
        plt.savefig(os.path.join(args.baseline_dir, "colorbar.png"), bbox_inches='tight')
            
def main():
    args = parse()
    err_datas = collect_data(args)
    visualize_mixed(args, err_datas)
    
if __name__ == '__main__':
    main()
                        