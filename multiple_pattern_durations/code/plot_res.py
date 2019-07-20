import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import rc
import yaml
import quantities as pq
# activate latex text rendering
rc('text', usetex=True)

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

# Load parameter from configfile
with open("configfile.yaml", 'r') as stream:
    config = yaml.load(stream)
spectra = config['spectra']
# Load parameters dictionary
param_dict = np.load('../data/art_data.npy', encoding='latin1').item()['params']
lengths = param_dict['lengths']
binsize = param_dict['binsize']
winlens = [int(l/binsize)+1 for l in lengths]

# Plotting parameters
inch2cm = 0.3937        # conversion from inches to centimeters
label_size = 6
text_size = 8
tick_size = 5

####### Raster plot of the data #######
# Loading spiketrains and patterns
data_file = np.load('../data/art_data.npy', encoding='latin1').item()
sts = data_file['data']
stps = data_file['patt']

figure = plt.figure(figsize=(12*inch2cm, 9*inch2cm))
figure.subplots_adjust(
    left=.08,
    right=.91,
    wspace=.8,
    hspace=1,
    top=.95,
    bottom=.1)
ax_raster = plt.subplot2grid((3, 2 * np.sum(winlens[:5])), (0, 0), rowspan=1, colspan=2 * np.sum(winlens[:5]))
for s_idx, s in enumerate(sts[:len(stps)]):
    s_sliced = s.time_slice(s.t_start + .15*pq.s, 0.4 * pq.s)
    ax_raster.plot(s_sliced, [s_idx] * len(s_sliced), 'k.', markersize=2)
for p_idx, p in enumerate(stps):
    p_sliced = p.time_slice(s.t_start + .15*pq.s, 0.4 * pq.s)
    if p_idx == len(stps) - 1:
        ax_raster.plot(p_sliced, [p_idx] * len(p_sliced), 'ro', label='STPs',
                 fillstyle='full',
                 markeredgecolor='red',
                 markersize=1.5)
    else:
        ax_raster.plot(p_sliced, [p_idx] * len(p_sliced), 'ro', fillstyle='full',
                 markeredgecolor='red',
                 markersize=1.5)
ax_raster.set_xlabel('time (s)', size=label_size)
ax_raster.set_ylabel('neuron id', size=label_size)
ax_raster.set_ylim(-0.5, 15)
ax_raster.tick_params(labelsize=tick_size)


########### Plotting SPADE results ##########
p_values_table = {}
for spectrum_idx, spectrum in enumerate(spectra):
    p_values_table[str(spectrum)] = {}
    ax_count = plt.subplot2grid((3, 2 * np.sum(winlens[:5])), (1, spectrum_idx*np.sum(winlens[:5])), rowspan=1, colspan=np.sum(winlens[:5]))
    # Plotting count and pval spectrum of patterns for each window length
    for w_idx, w in enumerate(winlens):
        # Load filtered results
        patterns, pval_spectrum, ns_sgnt, params = np.load(
            '../results/{}/winlen{}/filtered_patterns.npy'.format(spectrum,w),
             encoding='latin1')
        ax_count.bar(w_idx+1, len(patterns), color='k', width=0.5)
        if w_idx == len(winlens)-1:
            ax_count.plot(w_idx + 1, w_idx + 1, 'g^', markersize=4,
                          label='Number of STPs')
        else:
            ax_count.plot(w_idx + 1, w_idx + 1, 'g^', markersize=4)
        ax_count.set_yticks(np.arange(0, len(stps)+1))
        for tick in ax_count.xaxis.get_major_ticks():
            tick.label.set_fontsize(tick_size)
        # Plotting pvalue spectra
        pval_matrix = np.zeros((4, w))
        for sgnt in pval_spectrum:
            if spectrum == '#':
                for length in range(w):
                    pval_matrix[sgnt[1]-2,length] = sgnt[2]
            elif spectrum == '3d#':
                pval_matrix[sgnt[1]-2,sgnt[2]] = sgnt[3]
        ax_pval_spectrum = plt.subplot2grid(
            (3, 2 * np.sum(winlens[:5])),(2, spectrum_idx*np.sum(winlens[:5]) + sum(winlens[:w_idx])),
            colspan=sum(winlens[w_idx:w_idx + 1]))
        pcol = ax_pval_spectrum.pcolor(
            pval_matrix,  norm=LogNorm(vmin=0.0001, vmax=1), cmap=plt.cm.Greys)
        if spectrum == '#':
            for duration in range(pval_matrix.shape[1]):
                for occ in range(2, pval_matrix.shape[0]+2):
                    if (3, occ) not in ns_sgnt:
                        ax_pval_spectrum.plot(duration+0.5, occ-2+0.5, 'ro', markersize=1)
        elif spectrum == '3d#':
            for duration in range(pval_matrix.shape[1]):
                for occ in range(2, pval_matrix.shape[0]+2):
                    if (3, occ, duration) not in ns_sgnt:
                        ax_pval_spectrum.plot(duration+0.5, occ-2+0.5, 'ro', markersize=1)
        x_grid = np.arange(0, pval_matrix.shape[1]+1, 1)
        y_grid = np.arange(0, pval_matrix.shape[0]+1, 1)
        x_ticks = np.arange(0.5, pval_matrix.shape[1], 2)
        y_ticks = np.arange(0.5, pval_matrix.shape[0])
        x, y = np.meshgrid(x_grid, y_grid)
        c = np.ones_like(x)
        ax_pval_spectrum.pcolor(x, y, c, facecolor='none', edgecolor='k')
        ax_pval_spectrum.set_xticks(x_ticks)
        ax_pval_spectrum.set_xticklabels(range(0,pval_matrix.shape[1], 2), size=tick_size)
        if w_idx == len(winlens)//2+1:
            ax_pval_spectrum.set_xlabel('d (ms)', size=label_size)
        if w_idx == 0 and spectrum == '#':
            ax_pval_spectrum.set_yticks(y_ticks)
            ax_pval_spectrum.set_yticklabels(range(2, pval_matrix.shape[0] + 2), size=tick_size)
            ax_pval_spectrum.set_ylabel('c', size=label_size)
        else:
            ax_pval_spectrum.set_yticklabels(())
            ax_pval_spectrum.set_yticks(())
        if w_idx == len(winlens) - 1 and spectrum == '3d#':
            cbar = figure.colorbar(pcol, ticks=[0.0001, 0.001, 0.01, 0.1, 1], ax=ax_pval_spectrum)
            cbar.set_label('p-values', size=label_size)
            cbar.ax.tick_params(labelsize=tick_size)
            cbar.ax.minorticks_off()
        ax_pval_spectrum.set_xlim(0, pval_matrix.shape[1])
        ax_pval_spectrum.set_title(int(w*binsize.magnitude), size=label_size)
        if w == 13:
            # print(pval_matrix[-2, :])
            pattern_length = [0,2,6,8,12]
            p_val_arr = np.take(pval_matrix[2, :], pattern_length)
            p_values_table[str(spectrum)][str(w)] = p_val_arr
            # print(p_val_arr)

    if spectrum == '#':
        ax_count.set_title('2d-SPADE', size=text_size)
        ax_count.set_ylabel('detected STPs', size=label_size)
    elif spectrum == '3d#':
        ax_count.set_title('3d-SPADE', size=text_size)
        ax_count.set_yticks(())
    ax_count.set_xticks(range(1, len(winlens)+1))
    ax_count.set_xticklabels(winlens*int(binsize.magnitude))
    ax_count.tick_params(labelsize=tick_size)
    ax_count.set_ylim([0,len(winlens)+1])
    ax_count.set_xlabel('w (ms)', size=label_size)
    if spectrum_idx==0:
        ax_count.legend(loc='best', fontsize=label_size)
figure_path = '../figures'
path_temp = './'
for folder in split_path(figure_path):
    path_temp = path_temp + '/' + folder
    mkdirp(path_temp)
fig_formats = ['eps', 'png']
for format in fig_formats:
    figure.savefig(figure_path + '/raster_patt_count_spectra.{}'.format(format))
# save p_values_table (only for window = 13 and n_occ = 4)
np.save('../figures/pvalues_table.npy', p_values_table)
