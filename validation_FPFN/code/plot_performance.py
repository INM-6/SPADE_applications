# -*- coding: utf-8 -*-

import numpy as np
import quantities as pq
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import rc

# activate latex text rendering
rc('text', usetex=True)
plt.switch_backend('agg')
import os
# Function to create directory
def mkdirp(directory):
    if not os.path.isdir(directory):
        os.mkdir(directory)
# Function to split path to single folders
def split_path(path):
    folders = []
    while 1:
        path, folder = os.path.split(path)
        if folder != "":
            folders.append(folder)
        else:
            if path != "":
                folders.append(path)
            break
    folders.reverse()
    return folders

# Signifcance leve (integer between 0 and 100)
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
# Plot and save figure
inch2cm = 0.3937        # conversion from inches to centimeters
label_size = 6
text_size = 6
tick_size = 5
fig_FPFN = plt.figure(figsize=(12*inch2cm, 9*inch2cm))
fig_FPFN.subplots_adjust(top=.92, left=.1, right=.85, bottom=.2, hspace=0.2, wspace=0.02)


# Plot FP FN
for FP_idx, data_idx in enumerate(data_idxs):
    FP = FPs_all[data_idx]
    ax_FP = plt.subplot2grid((3,len(data_idxs)), (0, FP_idx), rowspan=1, colspan=1, aspect=1)
    pcol_FP = ax_FP.pcolor(FP.T, norm=LogNorm(vmin=0.0001, vmax=1), cmap=plt.cm.Blues)
    ax_FP.plot(np.where(FP>0.05)[0]+0.5, np.where(FP>0.05)[1]+0.5, 'o',
    markerfacecolor="None", markeredgecolor='k', markersize=1.8)
    if data_idx == 0:
        ax_FP.set_title('stationary', size=label_size)
    elif data_idx == 1:
        ax_FP.set_title('coherence', size=label_size)
    elif data_idx == 2:
        ax_FP.set_title('heterogeneity', size=label_size)
    elif data_idx == 3:
        ax_FP.set_title('propagation', size=label_size)
    ax_FP.set_xticks(np.arange(0.5,9,1))
    ax_FP.tick_params(length=1.2)
    ax_FP.set_xticklabels(())
    ax_FP.set_xlim([0,8])
    ax_FP.set_yticks(np.arange(0.5,9,1))
    ax_FP.set_ylim([0,8])
    if FP_idx == 0:
        ax_FP.set_ylabel('$\#$ occurrences$(c)$', fontsize=label_size)
        ax_FP.set_yticklabels(range(3,11), size=tick_size)
    else:
        ax_FP.set_yticklabels(())
    x, y, dx, dy = ax_FP.get_position().bounds
    cbar_ax = plt.axes((x+.2, y+0.0, 0.015, 0.21))
    cbar = fig_FPFN.colorbar(pcol_FP, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=tick_size)
    cbar.ax.text(6.2, .5, 'FP rate', va='center', ha='left', rotation=270, size=label_size)

for FN_idx, data_idx in enumerate(data_idxs):
    FN = FNs_all[data_idx]
    ax_FN = plt.subplot2grid((3,len(data_idxs)), (1, FN_idx), rowspan=1, colspan=1, aspect=1)
    pcol_FN = ax_FN.pcolor(FN.T,  norm=LogNorm(vmin=0.0001, vmax=1), cmap=plt.cm.Reds)
    ax_FN.plot(np.where(FN>0.05)[0]+0.5, np.where(FN>0.05)[1]+0.5, 'o',
    markerfacecolor="None", markeredgecolor='k', markersize=1.8)
    ax_FN.tick_params(length=1.2)
    ax_FN.set_xticks(())
    ax_FN.set_xticklabels(range(3,11), size=tick_size)
    ax_FN.set_xlim([0,8])
    ax_FN.set_yticks(np.arange(0.5,9,1))
    ax_FN.set_yticklabels(range(3,11), size=tick_size)
    ax_FN.set_ylim([0,8])
    if FN_idx == 0:
        ax_FN.set_ylabel('$\#$ occurrences$(c)$', size=label_size)
        ax_FN.set_yticklabels(range(3,11), size=tick_size)
    else:
        ax_FN.set_yticklabels(())
    x, y, dx, dy = ax_FN.get_position().bounds
    cbar_ax = plt.axes((x+.2, y+0.00, 0.015, 0.21))
    cbar = fig_FPFN.colorbar(pcol_FN, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=tick_size)
    cbar.ax.text(6.2, .5, 'FN rate', va='center', ha='left', rotation=270, size=label_size)

for max_FPFNs_idx, data_idx in enumerate(data_idxs):
    max_FPFNs = max_FPFNs_all[data_idx]
    ax_max_FPFNs = plt.subplot2grid((3,len(data_idxs)), (2, max_FPFNs_idx), rowspan=1, colspan=1, aspect=1)
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
        ax_max_FPFNs.set_yticklabels(range(3,11), size=tick_size)
    else:
        ax_max_FPFNs.set_yticklabels(())
    x, y, dx, dy = ax_max_FPFNs.get_position().bounds
    cbar_ax = plt.axes((x+.2, y+0.00, 0.015, 0.21))
    cbar = fig_FPFN.colorbar(pcol_max_FPFNs, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=tick_size)
    cbar.ax.text(6.2, .5, 'max FP FN', va='center', ha='left', rotation=270, size=label_size)

figure_path = '../figures'
path_temp = './'
for folder in split_path(figure_path):
    path_temp = path_temp + '/' + folder
    mkdirp(path_temp)
fig_FPFN.savefig(figure_path + '/FPFN_performance.eps')
