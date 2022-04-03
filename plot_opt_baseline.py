import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.colors
import sys
sys.path += ['util']
import argparse_util
import numpy as np
import os

def main():

    parser = argparse_util.ArgumentParser(description='plot basleines')
    parser.add_argument('--dir', dest='dir', default='', help='directory for task')
    parser.add_argument('--filenames', dest='filenames', default='', help='contains filenames that stores loss per iteration for different baselines, seperated by ,')
    parser.add_argument('--labels', dest='labels', default='', help='contains labels for each filename, seperated by ,')
    parser.add_argument('--scales', dest='scales', default='', help='if specified, scale the iterations of each method according to the list here, seperated by ,')
    parser.add_argument('--suffix', dest='suffix', default='', help='suffix to plot figure name')
    parser.add_argument('--subplot_idx', dest='subplot_idx', default='', help='if non-empty, only render the subplots specified')
    parser.add_argument('--figsize', dest='figsize', default='', help='specifies the figure size')
    parser.add_argument('--xmax', dest='xmax', default='', help='specifies xmax of the plot')
    parser.add_argument('--transparent', dest='transparent', action='store_true', help='if specified, plot all curves in the same subplot with transparent lines')
    parser.add_argument('--median', dest='median', action='store_true', help='if specified, plot median of all curves, fill early terminated curve using the last loss value')
    parser.add_argument('--fixed_err_resample', dest='fixed_err_resample', action='store_true', help='if specified, plot the runtime including emulated restart (from resampling another curve with replacement) needed to achieve a low enough loss')
    parser.add_argument('--err_lo', dest='err_lo', type=float, default=1e-4, help='specifies the lower end of err used for fixed_err_resample')
    parser.add_argument('--err_hi', dest='err_hi', type=float, default=0.1, help='specifies the higher end of err used for fixed_err_resample')
    parser.add_argument('--nerr', dest='nerr', type=int, default=10, help='specifies how many different error value should be sample')
    parser.add_argument('--err_nrepeat', dest='err_nrepeat', type=int, default=100, help='specifies how many times we should repeat the sampling process')
    parser.add_argument('--xlabel', dest='xlabel', default='', help='specifies x label to the figure')
    parser.add_argument('--ylabel', dest='ylabel', default='', help='specifies x label to the figure')
    
    args = parser.parse_args()
    
    filenames = args.filenames.split(',')
    filenames = ['processed_' + file for file in filenames]
    labels = args.labels.split(',')
    
    assert len(filenames) == len(labels)
        
    if args.scales == '':
        scales = np.ones(len(filenames))
    else:
        scales = args.scales.split(',')
        scales = [float(val) for val in scales]
        assert len(scales) == len(filenames)
    
    loss = np.load(os.path.join(args.dir, filenames[0]))
    
    assert len(loss.shape) == 2
    
    nexperiments = loss.shape[0]
    total_iter = loss.shape[1]
    nbaselines = len(filenames)
    
    losses = [loss]
    
    for i in range(1, nbaselines):
        losses.append(np.load(os.path.join(args.dir, filenames[i])))
        
    if args.xmax != '':
        xmax = [float(val) for val in args.xmax.split(',')]
        if scales is not None:
            assert len(xmax) == 1
        else:
            assert len(xmax) == 2
    else:
        xmax = None
        
    if args.median:
        # process losses
        nexperiments = 1
        for idx in range(len(losses)):
            loss = losses[idx]
            nvalid = loss.shape[1] - np.isnan(loss).sum(-1)
            max_nvalid = np.max(nvalid)
            
            last_loss = loss[np.arange(loss.shape[0]), nvalid - 1]
            
            need_fill_mask = np.isnan(loss[:, :max_nvalid])
            values_to_fill = np.tile(np.expand_dims(last_loss, 1), [1, max_nvalid])
            
            loss[:, :max_nvalid][need_fill_mask] = values_to_fill[need_fill_mask]
            
            median_val = np.median(loss, 0, keepdims=True)
            
            print(np.argsort(loss[:, max_nvalid - 1])[loss.shape[0] // 2])
            
            losses[idx] = median_val
    elif args.fixed_err_resample:
        
        err_ticks = np.logspace(np.log10(args.err_lo), np.log10(args.err_hi), base=10, num=args.nerr)
        sampled_filename = os.path.join(args.dir, 'sampled_niters_lo_%f_hi_%f_n_%d%s.npy' % (args.err_lo, args.err_hi, args.nerr, args.suffix))
        
        if os.path.exists(sampled_filename):
            sampled_iter = np.load(sampled_filename)
        else:
            sampled_iter = np.empty([nbaselines, err_ticks.shape[0], args.err_nrepeat])

            for nb in range(nbaselines):

                loss = losses[nb]

                loss_min = loss[np.logical_not(np.isnan(loss))].min()

                for err_idx in range(err_ticks.shape[0]):
                    err = err_ticks[err_idx]
                    if loss_min > err:
                        sampled_iter[nb, err_idx] = np.nan
                    else:
                        for repeat_idx in range(args.err_nrepeat):
                            # start sampling
                            achieved_loss = 1e8
                            niters = 0
                            while achieved_loss > err:
                                chosen_curve = np.random.choice(nexperiments)
                                last_valid_iter = loss.shape[1] - np.isnan(loss[chosen_curve]).sum() - 1

                                if loss[chosen_curve, last_valid_iter] < achieved_loss:
                                    achieved_loss = loss[chosen_curve, last_valid_iter]

                                if achieved_loss > err:
                                    niters += last_valid_iter + 1
                                else:
                                    first_acceptable_iter = np.where(loss[chosen_curve] < err)[0][0]
                                    niters += first_acceptable_iter + 1
                            sampled_iter[nb, err_idx, repeat_idx] = niters
            np.save(sampled_filename, sampled_iter)
        
        mean_sampled_iter = np.mean(sampled_iter, -1)
        if scales is not None:
            mean_sampled_iter = mean_sampled_iter * np.expand_dims(np.array(scales), -1)
        nexperiments = 1
        
        plt.clf()

        figsize = (10, 4)
        for nb in range(nbaselines):
            plt.plot(mean_sampled_iter[nb], err_ticks, label=labels[nb])
            
        if xmax is not None:
            plt.xlim(0, xmax[0])
            
        plt.yscale('log')
        
        plt.xlabel('Sampled Runtime')
        plt.ylabel('Error')
        
        plt.legend()
        plt.savefig(os.path.join(args.dir, 'baseline_err_runtime%s.png' % args.suffix))
        return
            
        

        
    single_plot = False
    if args.transparent or args.median or args.fixed_err_resample:
        single_plot = True
        
    if single_plot:
        ncols = 1
        assert nbaselines <= len(matplotlib.colors.TABLEAU_COLORS)
        colors = [matplotlib.colors.to_rgb(col) for col in matplotlib.colors.TABLEAU_COLORS.values()]
    elif scales is not None:
        ncols = 1
    else:
        ncols = 2

    for fig_idx in range(2):

        plt.clf()

        # Linear scale
        if single_plot:
            figsize = (10 * ncols, 4)
        else:
            figsize = (5 * ncols, 2 * nexperiments)
            
        fig = plt.figure(figsize=figsize)
        plots = [None] * nbaselines

        for ne in range(nexperiments):

            for col in range(ncols):
                
                if single_plot:
                    ax = plt.gca()
                else:
                    ax = plt.subplot(nexperiments, ncols, 1 + ne * ncols + col)

                    if ne != nexperiments - 1:
                        ax.set_xticklabels([])

                    if scales is None and ne == 0:
                        if col == 0:
                            ax.set_title('Scaled by runtime')
                        else:
                            ax.set_title('Scaled by number of op')

                for nb in range(nbaselines):
                    iterations = np.arange(losses[nb][ne].shape[0]) + 1

                    if fig_idx == 0:
                        # linear scale
                        data = losses[nb][ne]
                    else:
                        # log scale
                        #data = np.log10(losses[nb][ne])
                        data = losses[nb][ne]

                    if single_plot:
                        if args.transparent:
                            alpha = 0.1
                        else:
                            alpha = 1
                        pl = plt.plot(iterations * scales[nb], data, color=colors[nb], alpha=alpha)
                    else:
                        pl = plt.plot(iterations * scales[nb], data, label=labels[nb])
                    plots[nb] = pl[0]

                if xmax is not None:
                    ax.set_xlim(0, xmax[col])
                else:
                    ax.set_xlim(0, None)
                    
        
        if single_plot:
            plots = [None] * nbaselines
            for nb in range(nbaselines):
                if fig_idx == 0:
                    data = losses[nb][0][:2]
                else:
                    data = np.log10(losses[nb][0][:2])
                plots[nb] = plt.plot([-1, -1], data, color=colors[nb], label=labels[nb])[0]
                
        fontsize = 20

        fig.legend(plots, labels, loc="upper right", bbox_to_anchor=(0.9,0.87), fontsize=fontsize)
        
        
        if fig_idx == 1:
            plt.yscale('log')
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        
        if args.xlabel != '':
            plt.xlabel(args.xlabel, fontsize=fontsize)
        if args.ylabel != '':
            plt.ylabel(args.ylabel, fontsize=fontsize)

        if fig_idx == 0:
            plt.savefig(os.path.join(args.dir, 'baseline_comp%s.png' % args.suffix), bbox_inches='tight')
        else:
            plt.savefig(os.path.join(args.dir, 'baseline_comp_log%s.png' % args.suffix), bbox_inches='tight')

if __name__ == '__main__':
    main()
        