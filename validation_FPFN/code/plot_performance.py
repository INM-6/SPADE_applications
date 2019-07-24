# -*- coding: utf-8 -*-

import numpy as np
import quantities as pq
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import rc
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils import mkdirp, split_path


import random
import neo
# activate latex text rendering
rc('text', usetex=True)
plt.switch_backend('agg')

# Signifcance level (integer between 0 and 100)
alpha = 0.01

data_idxs = [0,1,2,3] # Index of non-stat data

FPs_all = []
FNs_all = []
Prec_all = []
Rec_all = []


# If save the figure
savefig = False
data_path = '../data/'
FPs_all = {}
FNs_all = {}
max_FPFNs_all = {}

for data_idx in data_idxs:
    path = '../results/results_data{data_idx}/alpha{alpha}'.format(data_idx=data_idx, alpha=alpha)
    # Load ground truth data
    # sts_rep = np.load(data_path + 'stp_data%i.npy' % data_idx, encoding='latin1').item()
    ##
    # Matrix in which will be stored the results
    FPs = np.zeros((8,8))
    FNs = np.zeros((8,8))
    FPs_shape = np.zeros((20,20))
    FPs_patt = []
    Precision = np.zeros((9,9))
    Recall = np.zeros((9,9))
    # Load the fca results
    results_all = np.load(path + 'result_filtered.npy', encoding='latin1')[0]
    # Loop all the combination of complexities and occurrences
    for xi in range(3,11):
        for occ in range (3,11):
            FPs_idx = []
            FNs_idx = []
            prec_list = []
            results = results_all['xi{xi}_occ{occ}'.format(
            xi=xi, occ=occ)]
            # Loop through all he 100 realizations
            for res_idx in results.keys():
                # winds_gt = ground_truth[int(res_idx)][0].rescale(pq.ms)
                # winds_gt = list(np.unique(np.floor(winds_gt[winds_gt<951])).magnitude)
                patterns = results[res_idx][0]
                num_FP = 0
                # Check that only one significant pattern was found
                for patt_idx, patt in enumerate(patterns):
                    if not list(np.sort(patt['neurons'])) == list(range(xi)):
                            # Checking that the spike trains in the pattern are the first xi
                            FPs_idx.append(res_idx)
                            num_FP += 1
                            FPs_patt.append(patt)
                            found_FP = True
                            # if data_idx == 2:
                                # print('(xi,occ)', (xi, occ))
                                # print('neurons', patt['neurons'])
                                # print('sgnt', patt['signature'])
                                # print('length', patt['lags'][-1])
                                # print('pval', patt['pvalue'])
                            break
                    elif not list(patt['lags'].magnitude) == [5.*i for i in range(1,xi)]:
                        # Checking that the spike trains in the pattern are the first xi
                        FPs_idx.append(res_idx)
                        num_FP += 1
                        FPs_patt.append(patt)
                        found_FP = True
                        print(patt['signature'])
                        break
                if len(patterns)==0 or len(patterns) == num_FP:
                    FNs_idx.append(int(res_idx))
                    prec_list.append(0.)
                else:
                    prec_list.append(1./(1 + num_FP))
            FPs_idx = np.unique(np.array(FPs_idx))
            FNs_idx = np.unique(np.array(FNs_idx))
            FPs_tot = len(FPs_idx)
            FNs_tot = len(FNs_idx)
            FNs[xi-3, occ-3] = FNs_tot
            FPs[xi-3, occ-3] = FPs_tot
            Precision[xi-3, occ-3] = np.average(prec_list)
            Recall[xi-3, occ-3] = (100. - FNs_tot) / 100
    # for fp in FPs_patt:
    max_FPFNs = np.zeros(FPs.shape)
    max_FPFNs[FPs>=FNs] = FPs[FPs>=FNs]
    max_FPFNs[FPs<FNs] = FNs[FPs<FNs]
    FPs_all[data_idx] = FPs/100.
    FNs_all[data_idx] = FNs/100.
    max_FPFNs_all[data_idx] = max_FPFNs/100.
##
## Loading data
# Set which rate model to load
sts_all = []
patterns_all = []
sts_back = []
for idx, data_idx in enumerate(data_idxs):
    # Loading a dictionary containing all the combination of parameters for one
    data_path = '../data/stp_data%i.npy' % data_idx
    sts_rep = np.load(data_path, encoding='latin1').item()

    # Selecting the parameter of the STP
    # Complexity: number of neurons (assumes values in [0,2,...,10])
    xi = 5
    # Number of occurrences of the STP (assumes values in [0,2...,10])
    occ = 5
    # repetition to select
    rep = 0
    # Storing all the 100 realizations of the data containing the background noise
    # and the STP (100 lists each containing 100 spiketrains, of which the first xi
    # each contain the STP)
    sts_back.append(sts_rep['sts_0occ_0xi'][rep])
    sts_all.append(sts_rep['sts_%iocc_%ixi' % (occ, xi)][rep])
    patterns_all.append(sts_rep['stp_%iocc_%ixi' % (occ, xi)][rep])

# Plot and save figure
inch2cm = 0.3937        # conversion from inches to centimeters
label_size = 6
text_size = 6
tick_size = 5
fig_FPFN = plt.figure(figsize=(12*inch2cm, 12*inch2cm))
fig_FPFN.subplots_adjust(top=.95, left=.1, right=.85, bottom=.1, hspace=0.01, wspace=.6)

for idx, sts in enumerate(sts_back):
    if idx==0 or idx==1 or idx == 3:
        ids = list(range(0, 20))
    if idx == 2:
        # ids = list(range(0,100,4))
        rates = [len(st) for st in sts]
        ids = np.argsort(rates)
    ax_raster = plt.subplot2grid((5,4), (1,idx), rowspan=1, colspan=1, aspect=0.05)
    for i, st in enumerate(np.array(sts_back[idx])[ids]):
        if idx==2:
            print(len(st))
        ax_raster.plot(st.rescale(pq.s), [i] * len(st), "k.", markersize=1)
    ax_raster.set_xlabel('time(s)', size = label_size)
    ax_raster.set_xticks(())
    # ax_raster.set_xticklabels(np.arange(0,1.1,0.5),size=tick_size)
    ax_raster.set_yticks(np.arange(0,21,5))
    ax_raster.set_yticklabels(np.arange(0,21,5),size=tick_size)
    ax_raster.set_ylim([-1,21])
    ax_raster.set_yticks(())
    if idx == 0:
        ax_raster.set_ylabel('neuron id', size = label_size)
    if idx == 0:
        ax_stationary = plt.subplot2grid((5,4), (0,idx), rowspan=1, colspan=1, aspect=0.015)
        ax_stationary.set_title('stationary', size=label_size)
        # Define a rate profile
        times = pq.s * np.arange(0, 1, 0.01)
        rate_profile = np.ones(times.shape)*25
        rate_profile = neo.AnalogSignal(rate_profile*pq.Hz, sampling_period=0.01*pq.s)
        ax_stationary.plot(rate_profile.times, rate_profile, color='k')
        ax_stationary.set_xlim([0, 1])
        ax_stationary.set_ylim([0, 65])
        ax_stationary.set_xticks(())
        # ax_stationary.set_xticklabels(np.arange(0, 1.1, 0.5), size=tick_size)
        ax_stationary.set_yticks(())
        ax_stationary.set_ylabel('rate (Hz)', size=label_size)
        ax_stationary.set_xlabel('time (s)', size=label_size)
    if idx == 1:
        ax_coherent = plt.subplot2grid((5,4), (0,idx), rowspan=1, colspan=1, aspect=0.015)
        ax_coherent.set_title('coherence', size=label_size)
        a0, a1 = 10 * pq.Hz, 60 * pq.Hz     # baseline and transient rates
        t1, t2 = 600 * pq.ms, 700 * pq.ms   # time segment of transient rate
        # Define a rate profile
        times = pq.s * np.arange(0, 1, 0.01)
        rate_profile = np.zeros(times.shape)
        rate_profile[np.any([times < t1, times > t2], axis=0)] = a0.magnitude
        rate_profile[np.all([times >= t1, times <= t2], axis=0)] = a1.magnitude
        rate_profile = rate_profile * a0.units
        rate_profile = neo.AnalogSignal(rate_profile, sampling_period=0.01*pq.s)
        ax_coherent.plot(rate_profile.times, rate_profile, color='k')
        ax_coherent.set_xlim([0, 1])
        ax_coherent.set_ylim([0, 65])
        ax_coherent.set_xticks(())
        # ax_coherent.set_xticklabels(np.arange(0, 1.1, 0.5), size=tick_size)
        ax_coherent.set_yticks(())
        ax_coherent.set_xlabel('time (s)', size=label_size)

    if idx == 2:
        ax_heterogeneous = plt.subplot2grid((5,4), (0,idx), rowspan=1, colspan=1, aspect=4.2)
        ax_heterogeneous.set_title('heterogeneity', size=label_size)
        rates = np.linspace(5, 25, 100)
        ax_heterogeneous.plot(np.arange(100)[::2], rates[::2], '.', color='k', ms=0.2, lw=0.5)
        ax_heterogeneous.set_xlim([0, 100+1])
        ax_heterogeneous.set_ylim([rates[0]*.9, rates[-1]*1.1])
        ax_heterogeneous.set_xticks(())
        # ax_heterogeneous.set_xticklabels([1, 25, 50, 75, 100], size=tick_size)
        ax_heterogeneous.set_yticks(())
        ax_heterogeneous.set_xlabel('neuron id', size=label_size)

    if idx == 3:
        ax_propagation = plt.subplot2grid((5,4), (0,idx), rowspan=1, colspan=1, aspect=0.18)
        ax_propagation.set_title('propagation', size=label_size)
        rate_profiles = [np.ones(100) for i in range(5)]
        for i in range(5):
            rate_profiles[i][[5+3*i,6+3*i,7+3*i,55+3*i,56+3*i, 57+3*i]] += .95
        N = len(rates)
        colors = ['r', 'k', 'lightblue', 'darkgreen', 'darkorange']
        for i, rate in enumerate(rate_profiles):
            color = colors[i]
            ax_propagation.step(np.arange(0,1,0.01), rate+i, '-', lw=1, color=color)
            if i == 0 or i == 4:
                text = 'group 1' if i == 0 else 'group l'
                x = 13 if i == 0 else 25
                y = 1.1+i if i == 0 else 1.3+i
                ax_propagation.text(x, y, text, color=color, size=tick_size)
        ax_propagation.set_ylim([0.5, 6.3])
        ax_propagation.set_xticks(())
        # ax_propagation.set_xticklabels(np.arange(0, 1.1, .5), size=tick_size)
        ax_propagation.set_yticks(())
        # ax_propagation.set_ylabel('rate (Hz)', size=label_size)
        ax_propagation.set_xlabel('time (s)', size=label_size)

def colorbar(mappable):
    ax = mappable.axes
    x, y, dx, dy = ax.get_position().bounds
    cax = plt.axes((x + .15, y + 0.019, 0.015, 0.13))
    fig = ax.figure
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="10%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)

# Plot FP FN
for FP_idx, data_idx in enumerate(data_idxs):
    FP = FPs_all[data_idx]
    ax_FP = plt.subplot2grid((5,len(data_idxs)), (2, FP_idx), rowspan=1, colspan=1, aspect=1)
    pcol_FP = ax_FP.pcolor(FP.T, norm=LogNorm(vmin=0.0001, vmax=1), cmap=plt.cm.Blues)
    ax_FP.plot(np.where(FP>0.05)[0]+0.5, np.where(FP>0.05)[1]+0.5, 'o',
    markerfacecolor="None", markeredgecolor='k', markersize=1.8)
    ax_FP.set_xticks(np.arange(0.5,9,1))
    ax_FP.tick_params(length=1.2)
    ax_FP.set_xticklabels(range(3,11), size=tick_size)
    ax_FP.set_xlim([0,8])
    ax_FP.set_yticks(np.arange(0.5,9,1))
    ax_FP.set_ylim([0,8])
    ax_FP.set_yticklabels(range(3,11), size=tick_size)
    if FP_idx == 0:
        ax_FP.set_ylabel('$\#$ occurrences$(c)$', fontsize=label_size)

    if FP_idx == len(data_idxs)-1:
        # cbar = fig_FPFN.colorbar(pcol_FP, ax=ax_FP)
        cbar = colorbar(pcol_FP)
        cbar.set_label('FP rate', size=label_size)
        cbar.ax.tick_params(size=0.,labelsize=tick_size)

for FN_idx, data_idx in enumerate(data_idxs):
    FN = FNs_all[data_idx]
    ax_FN = plt.subplot2grid((5,len(data_idxs)), (3, FN_idx), rowspan=1, colspan=1, aspect=1)
    pcol_FN = ax_FN.pcolor(FN.T,  norm=LogNorm(vmin=0.0001, vmax=1), cmap=plt.cm.Reds)
    ax_FN.plot(np.where(FN>0.05)[0]+0.5, np.where(FN>0.05)[1]+0.5, 'o',
    markerfacecolor="None", markeredgecolor='k', markersize=1.8)
    ax_FN.tick_params(length=1.2)
    ax_FN.set_xticks(np.arange(0.5,9,1))
    ax_FN.set_xticklabels(range(3,11), size=tick_size)
    ax_FN.set_xlim([0,8])
    ax_FN.set_yticks(np.arange(0.5,9,1))
    ax_FN.set_yticklabels(range(3,11), size=tick_size)
    ax_FN.set_ylim([0,8])
    if FN_idx == 0:
        ax_FN.set_ylabel('$\#$ occurrences$(c)$', size=label_size)
    if FN_idx == len(data_idxs)-1:
    # cbar = fig_FPFN.colorbar(pcol_FN, ax=ax_FN)
        cbar = colorbar(pcol_FN)
        cbar.set_label('FN rate', size=label_size)
        cbar.ax.tick_params(size=0.,labelsize=tick_size)

for max_FPFNs_idx, data_idx in enumerate(data_idxs):
    max_FPFNs = max_FPFNs_all[data_idx]
    ax_max_FPFNs = plt.subplot2grid((5,len(data_idxs)), (4, max_FPFNs_idx), rowspan=1, colspan=1, aspect=1)
    pcol_max_FPFNs = ax_max_FPFNs.pcolor(max_FPFNs.T, norm=LogNorm(vmin=0.0001, vmax=1), cmap=plt.cm.RdPu)
    ax_max_FPFNs.plot(np.where(max_FPFNs>0.05)[0]+0.5, np.where(max_FPFNs>0.05)[1]+0.5, 'o',
    markerfacecolor="None", markeredgecolor='k', markersize=1.8)
    ax_max_FPFNs.tick_params(length=1.2)
    ax_max_FPFNs.set_xticks(np.arange(0.5,9,1))
    ax_max_FPFNs.set_xticklabels(range(3,11), size=tick_size)
    ax_max_FPFNs.set_xlim([0,8])
    ax_max_FPFNs.set_xlabel('pattern size $(z)$', size=label_size)
    ax_max_FPFNs.set_yticks(np.arange(0.5,9,1))
    ax_max_FPFNs.set_yticklabels(range(3,11), size=tick_size)
    ax_max_FPFNs.set_ylim([0,8])
    if max_FPFNs_idx == 0:
        ax_max_FPFNs.set_ylabel('$\#$ occurrences$(c)$', size=label_size)
    if max_FPFNs_idx == len(data_idxs)-1:
        cbar = colorbar(pcol_max_FPFNs)
        # cbar = fig_FPFN.colorbar(pcol_max_FPFNs, ax=ax_max_FPFNs, fraction=.01)
        cbar.set_label('max(FP, FN)', size=label_size)
        cbar.ax.tick_params(size=0.,labelsize=tick_size)

figure_path = '../figures'
path_temp = './'
for folder in split_path(figure_path):
    path_temp = path_temp + '/' + folder
    mkdirp(path_temp)
fig_FPFN.savefig(figure_path + '/FPFN_performance.eps')
