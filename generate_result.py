import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import sys
import numpy
import numpy as np
from preprocess_raw_loss_data import preprocess_file
from matplotlib import pyplot as plt
from matplotlib.ticker import LogLocator, FixedLocator, NullFormatter, FixedFormatter, NullLocator, MaxNLocator
import string
import shutil
import run_stored_cmd

ylim_thre = 0.1
success_thre = 0.2
fontsize = 20

thre = 2

cols = {'Ours': 'b',
        'FD': 'g',
        'SPSA': 'r',
        'Ours_no_random': 'purple',
        'DVG': 'orange'}

zorders = {'Ours': 1,
           'FD': 2,
           'SPSA': 3,
           'Ours_no_random': 4,
           'DVG': 5}
           

shader_files = [
    # DVG comparisons
    'apps/render_test_finite_diff_circle.py',
    'apps/test_finite_diff_ring.py',
    # TEG comparisons
    'apps/render_test_finite_diff_circle_bw.py',
    'apps/render_test_finite_diff_quadrant.py',
    'apps/render_test_finite_diff_rectangle_2step.py',
    # FD/SPSA comparisons
    'apps/render_test_finite_diff_olympic_vec_optional_update.py',
    'apps/render_test_finite_diff_ring_contour.py',
    'apps/render_test_finite_diff_raymarching_siggraph_cone.py',
    'apps/render_test_finite_diff_raymarching_structured_tf_logo.py',
    'apps/render_test_finite_diff_raytracing_structured_tf_logo.py'
]

datas_dvg = {
    
    'Circle':
    {
        'dir': 'test_finite_diff_circle',
        'Ours': '_compare_diffvg',
        'DVG': 'single_circle_outline_nsamples_2',
        'macro': 'circle'
    },
    
    'Ring':
    {
        'dir': 'test_finite_diff_ring',
        'Ours': '_compare_diffvg',
        'DVG': 'single_circle_outline_nsamples_2',
        'macro': 'circle'
    }
}

datas = {

    'Olympic Rings': 
    {
        'dir': 'test_finite_diff_olympic_vec_optional_update',
        'Ours': '_from_real_random',
        'Ours_no_random': '_from_real_no_random',
        'FD': ['_from_real_random_fd_h_%s',
              '_from_real_no_random_fd_h_%s'],
        'SPSA': [(1, '_from_real_random_fd_h_%s_scaled_niter'),
                 (65, '_from_real_random_fd_h_%s'),
                 (1, '_from_real_vanilla_fd_h_%s_scaled_niter'),
                 (40, '_from_real_vanilla_fd_h_%s'),
                 (1, '_from_real_no_random_fd_h_%s_scaled_niter'),
                 (40, '_from_real_no_random_fd_h_%s'),],
        #'Ours_no_random': '_from_real5_rings',
        #'FD': ['_from_real_random_5_rings_fd_h_%s'],
        #'SPSA': [(1, '_from_real_random_5_rings_fd_h_%s_scaled_niter'),
        #         (33, '_from_real_random_5_rings_fd_h_%s'),
        #         (1, '_from_real_vanilla_5_rings_fd_h_%s_scaled_niter'),
        #         (20, '_from_real_vanilla_5_rings_fd_h_%s')],
        'xmax': 52,
        'same_time_SPSA': 2,
        'max_halflen': 3e-3,
        'ref': '/n/fs/shaderml/global_opt/proj/apps/olympic_rgb.png',
        'macro': '\\olympic'
    },
    
    'Celtic Knot':
    {
        'dir': 'test_finite_diff_ring_contour',
        'Ours': '_from_real_random',
        'Ours_no_random': '_from_real_no_random',
        #'FD': '_from_real_random_fd_%s',
        'FD': ['_from_real_random_fd_%s',
               '_from_real_no_random_fd_%s'],
        'SPSA': [(1, '_from_real_random_fd_%s_scaled_niter'),
                 (42, '_from_real_random_fd_%s'),
                 (1, '_from_real_vanilla_fd_h_%s_scaled_niter'),
                 (21, '_from_real_vanilla_fd_h_%s'),
                 (1, '_from_real_no_random_fd_%s_scaled_niter'),
                 (21, '_from_real_no_random_fd_%s')],
        'xmax': 7,
        'same_time_SPSA': 2,
        'max_halflen': 2e-3,
        'ref': '/n/fs/shaderml/differentiable_compiler/celtic_knot.png',
        'macro': '\\celtic'
    },
    
    'SIGGRAPH':
    {
        'dir': 'test_finite_diff_raymarching_siggraph_cone',
        'Ours': '_from_real_random',
        'Ours_no_random': '_from_real_no_random',
        #'FD': '_from_real_random_fd_h_%s',
        'FD': ['_from_real_random_fd_h_%s',
               '_from_real_no_random_fd_h_%s'],
        'SPSA': [(1, '_from_real_random_fd_h_%s_scaled_niter'),
                 (27, '_from_real_random_fd_h_%s'),
                 (1, '_from_real_vanilla_fd_h_%s_scaled_niter'),
                 (22, '_from_real_vanilla_fd_h_%s'),
                 (1, '_from_real_no_random_fd_h_%s_scaled_niter'),
                 (22, '_from_real_no_random_fd_h_%s')],
        'xmax': 70,
        'same_time_SPSA': 2,
        'max_halflen': 2e-3,
        'ref': '/n/fs/shaderml/global_opt/proj/apps/siggraph_gradient.png',
        'macro': '\\siggraph'
    },
    
    'TF Raymarch':
    {
        'dir': 'test_finite_diff_raymarching_structured_tf_logo_updated',
        'Ours': '_from_real',
        'FD': ['_from_real_fd_h_%s'],
        'SPSA': [(1, '_from_real_fd_h_%s_scaled_niter'),
                 (16, '_from_real_fd_h_%s'),
                 (1, '_from_real_vanilla_fd_h_%s_scaled_niter'),
                 (16, '_from_real_vanilla_fd_h_%s')],
        'xmax': 10,
        'same_time_SPSA': 1,
        'max_halflen': 2e-3,
        'ref': '/n/fs/shaderml/differentiable_compiler/tf_logo.png',
        'macro': '\\tfmarch'
    },
    
    'TF Raycast':
    {
        'dir': 'test_finite_diff_raytracing_structured_tf_logo',
        'Ours': '_from_real_random',
        'Ours_no_random': '_from_real_no_random',
        #'FD': '_from_real_random_fd_%s',
        #'FD': '_from_real_vanilla_fd_h_%s',
        'FD': ['_from_real_random_fd_%s',
               '_from_real_no_random_fd_%s',
               '_from_real_vanilla_fd_h_%s'],
        'SPSA': [(1, '_from_real_random_fd_%s_scaled_niter'),
                 (27, '_from_real_random_fd_%s'),
                 (1, '_from_real_vanilla_fd_h_%s_scaled_niter'),
                 (16, '_from_real_vanilla_fd_h_%s'),
                 (1, '_from_real_no_random_fd_%s_scaled_niter'),
                 (16, '_from_real_no_random_fd_%s')],
        'xmax': 20,
        'same_time_SPSA': 4,
        'max_halflen': 2e-3,
        'ref': '/n/fs/shaderml/differentiable_compiler/tf_logo.png',
        'macro': '\\tfcast'
    },

}

table_header = """
\\begin{tabular}{c|c|c|c|c|c|c|c|c}
\\multirow{2}{*}{Shader} & \\multicolumn{4}{c|}{Med. Success Time} & \\multicolumn{4}{c}{Exp. Time to Success} \\\\ \\cline{2-9}
                        &   Ours  &  \\Owo & FD$^*$   &  SPSA$^*$  &  Ours  &  \\Owo & FD$^*$      &  SPSA$^*$           \\\\ \\thickhline
"""
table_end = """
\\end{tabular}
"""

metric_table_header = """
\\begin{tabular}{cccc}
& Ours & FD & SPSA \\\\
"""

result_table_header = """
\\begin{tabular}{ccc}
Target & Optimization & Modified \\\\ 
"""

result_table_tf_header = """
\\begin{tabular}{ccc}
"""

def scientific_format(val):
    base_strs = ('%.2e' % val).split('e')
    
    ans = '$%s \\times 10^{%d}$' % (base_strs[0], int(base_strs[1]))
    
    return ans
    

def get_median(loss):
    nvalid = loss.shape[1] - np.isnan(loss).sum(-1)
    max_nvalid = np.max(nvalid)

    last_loss = loss[np.arange(loss.shape[0]), nvalid - 1]

    need_fill_mask = np.isnan(loss[:, :max_nvalid])
    values_to_fill = np.tile(np.expand_dims(last_loss, 1), [1, max_nvalid])

    loss[:, :max_nvalid][need_fill_mask] = values_to_fill[need_fill_mask]

    median_val = np.median(loss, 0, keepdims=True)

    return median_val

def process_log(file):
    
    lines = open(file).read().split('\n')
    
    success = False
    found = False
    
    for line in lines[::-1]:
        if line.startswith('runtime per iter'):
            runtime = -float(line.replace('runtime per iter', ''))
            success = False
            found = True
        if line.startswith('99 '):
            if found:
                success = True
                break
        else:
            try:
                val = float(line[:2])
                success = False
                found = False
            except:
                pass
            
    assert success
    assert found
    return runtime

def draw_subplot(ax, shader, count, err_datas, success_median_times=None):
    
    #dummy = success_median_times
    #success_median_times = None
    
    L_min = 1e8
    L_max = -1e8
    
    for method in err_datas.keys():
        loss, *_ = err_datas[method]
        
        L_min = min(L_min, np.nanmin(loss))
        L_max = max(L_max, np.nanmax(loss))
        
    current_thre = thre
        
    if success_median_times is not None:
        L_min_ours = np.nanmin(err_datas['Ours'][0])
        current_thre = thre * L_min_ours / L_min
        ax.plot(np.arange(datas[shader]['xmax'] + 1), current_thre * np.ones_like(np.arange(datas[shader]['xmax'] + 1)), 'gray', linewidth=2)
        
    #L_min = np.nanmin(err_datas['Ours'][0])
        
    k = (L_max / L_min) ** (1 / 3)
    
    for method in err_datas.keys():

        if method in ['Ours', 'DVG', 'Ours_no_random']:
            label = method.replace('_', ' ')
        else:
            label = '%s$^*$' % method

        loss, scale, *_ = err_datas[method]
        loss /= L_min

        iterations = np.arange(loss.shape[1]) + 1

        for idx in range(loss.shape[0]):
            ax.plot(iterations * scale, loss[idx], cols[method], alpha=0.05, zorder=zorders[method])

        median_val = get_median(loss)[0]

        ax.plot(iterations * scale, median_val, 'w', alpha=0.5, linewidth=5, zorder=zorders[method])
        ax.plot(iterations * scale, median_val, cols[method], label=label, zorder=zorders[method])
        
    if success_median_times is not None:
        for method in list(err_datas.keys())[::-1]:
            success_median_time = err_datas[method][3]
            ax.scatter(success_median_time, current_thre, c=cols[method], s=50, zorder=200+zorders[method])
        
        #for idx in range(len(success_median_times)):
        #    ax.scatter(success_median_times[idx], thre, c=list(cols.values())[idx], zorder=200, s=50)

    ax.set_yscale('log')

    # Disable minor ticks, because they will be inconsistent for different k values
    #nminor_ticks = 10
    #ax.yaxis.set_minor_locator(LogLocator(base=k, subs=[i * k / nminor_ticks for i in range(1, nminor_ticks)]))
    #ax.yaxis.set_minor_formatter(NullFormatter())

    #ax.yaxis.set_minor_locator(NullLocator())

    if False:
        yticks = k ** np.arange(4)
        ax.yaxis.set_major_locator(FixedLocator(yticks))


        if count == 0:
            ax.yaxis.set_major_formatter(FixedFormatter(['$\epsilon^%d$' % idx for idx in range(4)]))
            ax.legend(fontsize=fontsize)
        else:
            ax.yaxis.set_major_formatter(NullFormatter())
    else:
        yticks = 10 ** np.arange(np.ceil(np.log10(L_max / L_min)))
        ax.yaxis.set_major_locator(FixedLocator(yticks))
        ax.yaxis.set_major_formatter(FixedFormatter(['%d' % idx for idx in range(yticks.shape[0])]))
        if shader in ['SIGGRAPH', 'Circle']:
            ax.legend(fontsize=fontsize)

    if success_median_times:
        if count in [1, 3]:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))

        if count == 0:
            ax.set_ylabel('Error (order of magnitude)', fontsize=fontsize)
            #ax.yaxis.set_label_position("right")
    else:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        if count == 0:
            ax.set_ylabel('Error (order of magnitude)', fontsize=fontsize)
            #ax.yaxis.set_label_position("right")

    if success_median_times is not None:
        # force last tick to dissapear
        ax.set_xlim(0, datas[shader]['xmax'] - 1e-8)
    ax.set_ylim(1 / k ** ylim_thre, L_max / L_min * k ** ylim_thre)

    if not shader.startswith('TF'):
        titlename = shader
    else:
        if shader == 'TF Raymarch':
            titlename = 'TF RayMarch'
        else:
            titlename = 'TF RayCast'
            
            
    if success_median_times is None:
        pass
        #ax.set_title('(' + string.ascii_lowercase[count] + ') ' + titlename, fontsize=fontsize, y=-0.2)
        
    #ax.set_title('(' + string.ascii_lowercase[count] + ') ' + titlename, fontsize=fontsize, y=-0.2)

    ax.tick_params(axis='both', which='major', labelsize=fontsize)

def generate_result():
    parent_dir = sys.argv[1]
    
    save_dir = os.path.join(parent_dir, 'result')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        
    fig, axes = plt.subplots(nrows=1, ncols=len(datas_dvg), figsize=(5 * len(datas_dvg), 5))
    count = 0
    
    for shader in datas_dvg.keys():
        current_dir = os.path.join(parent_dir, datas_dvg[shader]['dir'])
        err_datas = {}
        
        for method in ['Ours', 'DVG']:
            
            if method == 'Ours':
                file = 'ours_both_sides_1_scale_L2_adam_1.0e-02%s_all_loss.npy' % datas_dvg[shader][method]
                processed_file = os.path.join(current_dir, 'processed_' + file)
                file = os.path.join(current_dir, file)
                
                logfile = os.path.join(current_dir, 'log_ours_both_sides_1_scale_L2_adam_1.0e-02%s.txt' % datas_dvg[shader][method])
                scale = process_log(logfile)
            else:
                file = os.path.join(current_dir, '../diffvg/results/single_circle_nsamples_2/all_loss.npy')
                processed_file = os.path.join(current_dir, 'processed_all_loss.npy')
                
                logfile = os.path.join(current_dir, '../diffvg/results/single_circle_nsamples_2/log.txt')
                scale = -process_log(logfile)
            
            if os.path.exists(processed_file):
                loss = np.load(processed_file)
            else:
                loss = preprocess_file(file)
                np.save(processed_file, loss)
                
            err_datas[method] = (loss, 
                                 scale,
                                 file.replace('_all_loss.npy', ''))
            
        ax = axes[count]
        
        draw_subplot(ax, shader, count, err_datas)
        
        count += 1
        
    plt.subplots_adjust(wspace=0.1)
    plt.savefig(os.path.join(save_dir, 'error_plot_DVG.png'), bbox_inches='tight')
    plt.close()
                             
    
    fig, axes = plt.subplots(nrows=1, ncols=len(datas), figsize=(5 * len(datas), 5))
    
    if len(datas) == 1:
        axes = np.array([axes])
    
    count = 0
    
    table_str = table_header
    
    for shader in datas.keys():
        current_dir = os.path.join(parent_dir, datas[shader]['dir'])
        
        err_datas = {}
        L_min = 1e8
        L_max = -1e8
        
        for method in ['Ours', 'Ours_no_random', 'FD', 'SPSA']:
            
            loss_files = []
            log_files = []
            
            if method not in datas[shader].keys():
                continue
            
            if method.startswith('Ours'):
                loss_files.append('ours_both_sides_5_scale_L2_adam_1.0e-02%s_all_loss.npy' % datas[shader][method])
                log_files.append('log_ours_both_sides_5_scale_L2_adam_1.0e-02%s.txt' % datas[shader][method])
            elif method == 'FD':
                for suffix in datas[shader][method]:
                    for h_exp in range(5):
                        h_str = '0' * (h_exp + 1) + '1'
                        loss_files.append('finite_diff_both_sides_5_scale_L2_adam_1.0e-02%s_all_loss.npy' % (suffix % h_str))
                        log_files.append('log_finite_diff_both_sides_5_scale_L2_adam_1.0e-02%s.txt' % (suffix % h_str))
            else:
                for nsamples, suffix in datas[shader][method]:
                    for h_exp in range(5):
                        h_str = '0' * (h_exp + 1) + '1'
                        loss_files.append('finite_diff%d_both_sides_5_scale_L2_adam_1.0e-02%s_all_loss.npy' % (nsamples, suffix % h_str))
                        log_files.append('log_finite_diff%d_both_sides_5_scale_L2_adam_1.0e-02%s.txt' % (nsamples, suffix % h_str))
                        
            current_min = 1e8
            current_loss = None
            current_log = None
                        
            for idx in range(len(loss_files)):
                file = loss_files[idx]
                             
                processed_file = os.path.join(current_dir, 'processed_' + file)
                if os.path.exists(processed_file):
                    loss = np.load(processed_file)
                else:
                    loss = preprocess_file(os.path.join(current_dir, file))
                    np.save(processed_file, loss)
                    
                median_val = get_median(loss)
                
                #min_val = np.nanmin(median_val)
                min_val = np.nanmin(loss)
                
                if min_val < current_min:
                    current_loss = loss
                    current_log = log_files[idx]
                    current_min = min_val
                    
                if np.nanmin(loss) < L_min and method != 'Ours_no_random':
                    L_min = np.nanmin(loss)
                    
                if np.nanmax(loss) > L_max:
                    L_max = np.nanmax(loss)
                    
            err_datas[method] = (current_loss, 
                                 process_log(os.path.join(current_dir, current_log)),
                                 file.replace('_all_loss.npy', ''))
                    
        datas[shader]['err'] = err_datas
        
        # generate convergence plot and table
        
        
        #thre = (L_max / L_min) ** success_thre
        
        ax = axes[count]
        
        success_rates = []
        success_median_times = []
        expected_times_to_success = []
        expected_nsamples = 100
                
        for method in err_datas.keys():
            
            if method.startswith('Ours'):
                label = method.replace('_', ' ')
            else:
                label = '%s$^*$' % method
            
            loss, scale, _ = err_datas[method]
            loss /= L_min
                        
            iterations = np.arange(loss.shape[1]) + 1
                            
            success_rate = np.sum(np.nanmin(loss, 1) < thre)
            
            success_min_idx = np.argmax(loss < thre, 1)
            if np.sum(success_min_idx[success_min_idx != 0]):
                median_idx = np.median(success_min_idx[success_min_idx != 0])
            else:
                median_idx = np.inf
            success_median_time = median_idx * scale
            
            
            
            success_rates.append(success_rate)
            success_median_times.append(success_median_time)
            
            if success_rate > 0:
                sum_time = 0
                for _ in range(expected_nsamples):
                    err = 1e8
                    while err > thre:
                        current_run = loss[np.random.choice(np.arange(loss.shape[0]))]
                        if np.nanmin(current_run) < thre:
                            sum_time += np.where(current_run < thre)[0][0]
                            break
                        else:
                            sum_time += np.sum(np.logical_not(np.isnan(current_run)))
                avg_time = sum_time * scale / expected_nsamples
            else:
                avg_time = np.inf
                
            print(method, success_rate, success_median_time, avg_time)
                
            expected_times_to_success.append(avg_time)
            
            err_datas[method] += (success_median_time, avg_time)
        
        metric_median = []
        metric_expected = []
        
        min_idx_median = None
        min_idx_expected = None
        
        min_median = 1e8
        min_expected = 1e8
        
        method_count = 0
        for method in err_datas.keys():
            metric_median.append('%.1f'% err_datas[method][3] if not np.isinf(err_datas[method][3]) else '\\na')
            metric_expected.append('%.1f'% err_datas[method][4] if not np.isinf(err_datas[method][4]) else '\\na')
            
            if err_datas[method][3] < min_median:
                min_median = err_datas[method][3]
                min_idx_median = method_count
                
            if err_datas[method][4] < min_expected:
                min_expected = err_datas[method][4]
                min_idx_expected = method_count
                
            method_count += 1
            
        metric_median[min_idx_median] = '\\textbf{%s}' % metric_median[min_idx_median]
        metric_expected[min_idx_expected] = '\\textbf{%s}' % metric_expected[min_idx_expected]
        
        if shader == 'TF Raymarch':
            # Ours without random should be identical with Ours with random
            metric_median = metric_median[:1] * 2 + metric_median[1:]
            metric_expected = metric_expected[:1] * 2 + metric_expected[1:]
        
        time_str = ' & '.join(metric_median)
        expected_time_str = ' & '.join(metric_expected)
        
        #time_str = '\\textbf{%.1f} &' % err_datas['Ours'][3] + ' & '.join(['%.1f'% val if not np.isinf(val) else '\\na' for val in success_median_times[1:2]])
        #expected_time_str = '\\textbf{%.1f} &' % expected_times_to_success[0] + ' & '.join(['%.1f' % val if not np.isinf(val) else '\\infin' for val in expected_times_to_success[1:2]])
        
        shadername = datas[shader]['macro']
        
        table_str += f"""
{shadername} & {time_str} & {expected_time_str} \\\\ \\hline
        """
        
        #if 'Ours_no_random' in err_datas.keys():
        #    del err_datas['Ours_no_random']
        
        draw_subplot(ax, shader, count, err_datas, success_median_times)
        
        count += 1
        
    plt.subplots_adjust(wspace=0.1)
    plt.savefig(os.path.join(save_dir, 'error_plot.png'), bbox_inches='tight')
    plt.close()
    
    table_str += table_end
    
    open(os.path.join(save_dir, 'err_table.txt'), 'w').write(table_str)
    
    # generate error metric figures and table
    
    metric_table_str = metric_table_header
    
    for shader in datas.keys():
        current_dir = os.path.join(parent_dir, datas[shader]['dir'])
        
        rhs = np.load(os.path.join(current_dir, 
                                   'random_smooth_metric_2X100000_len_%f_2D_kernel_rhs.npy' % datas[shader]['max_halflen']))
        
        if shader == 'Olympic Rings':
            metric_table_str += """\\multirow{2}{*}{\\raisebox{1.8\\normalbaselineskip}[0pt][0pt]{\\rotatebox[origin=c]{90}{%s}}} """ % '\\olympicshort'
        else:
            metric_table_str += """\\raisebox{2.5\\normalbaselineskip}[0pt][0pt]{\\rotatebox[origin=c]{90}{%s}}""" % datas[shader]['macro']
        
        min_errs = []
        for method in ['Ours', 'FD', 'SPSA']:
            if method == 'Ours':
                suffixes = ['_2D_kernel']
            elif method == 'FD':
                suffixes = ['_2D_kernel_FD_finite_diff_%f' % 0.1 ** h for h in range(1, 6)]
            else:
                suffixes = ['_2D_kernel_SPSA_' + str(datas[shader]['same_time_SPSA']) + '_finite_diff_%f' % 0.1 ** h for h in range(1, 6)]
                
            min_err = 1e8
            min_map = None
            
            for suffix in suffixes:
                lhs = np.load(os.path.join(
                    current_dir, 
                    'kernel_smooth_metric_debug_10000X1_len_%f_kernel_box_sigma_1.000000_0.100000%s.npy' % 
                    (datas[shader]['max_halflen'], suffix)))   
                
                err_map = np.abs(lhs - rhs).transpose()
                
                #err = np.mean(err_map[err_map != 0])
                err = np.mean(err_map)
                
                if err < min_err:
                    min_err = err
                    min_map = err_map
                    
            if shader == 'Olympic Rings':
                pad = 20
                min_map2 = np.zeros((min_map.shape[0] + 2 * pad, min_map.shape[1] + 2 * pad))
                min_map2[pad:-pad, pad:-pad] = min_map
                min_map = min_map2
                
                min_err = np.mean(min_map)
                    
            min_errs.append(min_err)
                    
            plt.figure()
            plt.imshow(min_map, vmin=0, vmax=0.01, cmap='hot')
            plt.axis('off')
            err_name = '%s_%s.png' % (shader, method)
            plt.savefig(os.path.join(save_dir, err_name), bbox_inches='tight')
            plt.close()
            
            metric_table_str += """& \\includegraphics[width=\\w]{figures/metric/%s} """ % err_name
        
        #err_str = ' & '.join([''] + ['%.2e' % err for err in min_errs])
        #err_str = ' & '.join([''] + [scientific_format(err) for err in min_errs])
        err_str = ' & '.join([''] + [scientific_format(min_errs[idx]) if idx == 0 else '%.1fx' % (min_errs[idx] / min_errs[0]) for idx in range(len(min_errs))])
        
        metric_table_str += """ \\\\
%s \\\\
""" % err_str
        
    metric_table_str += """& \\multicolumn{3}{c}{
            \\includegraphics[width=3\\w]{figures/metric/colorbar.png}}
""" + table_end
    open(os.path.join(save_dir, 'metric_table.txt'), 'w').write(metric_table_str)
    
    dummy = np.array([[0,1]])
    plt.figure(figsize=(9, 0.5))
    img = plt.imshow(dummy, cmap="hot", vmin=0, vmax=0.01)
    plt.gca().set_visible(False)
    cax = plt.axes([0.1, 0.2, 0.8, 0.6])
    cbar = plt.colorbar(orientation="horizontal", cax=cax)
    cbar.set_ticks([0, 0.01])
    cbar.ax.tick_params(labelsize=fontsize)
    plt.savefig(os.path.join(save_dir, "colorbar.png"), bbox_inches='tight')
    
    # generate result figures and table
    
    result_table = result_table_header
    result_table_tf_raymarch = result_table_tf_header
    result_table_tf_raytrace = result_table_tf_header
    
    for shader in datas.keys():
        current_dir = os.path.join(parent_dir, datas[shader]['dir'])
            
        loss, scale, prefix, *_ = datas[shader]['err']['Ours']
        
        min_loss = np.nanmin(loss, 1)
        # first 5 might be modified by other processes
        sorted_idx = np.argsort(min_loss[5:]) + 5
            
        opt_file = os.path.join(current_dir, prefix + '_result%d_0.png' % sorted_idx[0])
        opt_dst = os.path.join(save_dir, 'opt_%s.png' % shader)
        
        if not os.path.exists(opt_dst):
            shutil.copyfile(opt_file, opt_dst)
            
        ref_dst = os.path.join(save_dir, 'ref_%s.png' % shader)
        
        if not os.path.exists(ref_dst):
            shutil.copyfile(datas[shader]['ref'], ref_dst)
            
        fig_str = """ 
\\includegraphics[width=\\w]{figures/result/ref_%s.png} & \\includegraphics[width=\\w]{figures/result/opt_%s.png} & \\includegraphics[width=\\w]{figures/result/anim_%s.png} \\\\
        """ % (shader, shader, shader)
        
        if shader == 'TF Raycast':
            result_table_tf_raytrace += fig_str + """
(a) Opt & (b) Modified & (c) Modified \\\\
                """
        elif shader == 'SIGGRAPH':
            # pass because already include its result in teaser figure
            pass
        else:
            result_table += fig_str + """ 
%s & & \\\\
            """ % datas[shader]['macro']
            
    result_table_tf_raymarch += table_end
    result_table_tf_raytrace += table_end
    result_table += table_end
    
    #open(os.path.join(save_dir, 'result_table_tf_raymarch.txt'), 'w').write(result_table_tf_raymarch)
    open(os.path.join(save_dir, 'result_table_tf_raytrace.txt'), 'w').write(result_table_tf_raytrace)
    open(os.path.join(save_dir, 'result_table.txt'), 'w').write(result_table)
    
def collect_result():
    for file in shader_files:
        run_stored_cmd.run(file, sys.argv[1])

def main():
    
    error = False
    if len(sys.argv) != 3:
        error = True
    elif sys.argv[2] not in ['generate', 'collect']:
        error = True
        
    if error:
        print('Usage: python generate_result <path> <mode>')
        print('mode = {generate, collect}')
        return
    
    if sys.argv[2] == 'generate':
        generate_result()
    else:
        collect_result()
                    
if __name__ == '__main__':
    main()