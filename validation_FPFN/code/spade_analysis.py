import numpy as np
import os
import json
import quantities as pq
from mpi4py import MPI
import elephant.spade as spade
import argparse
import yaml
from utils import mkdirp, split_path


with open("configfile.yaml", 'r') as stream:
    config = yaml.load(stream)
# max. time window width in number of bins
winlen = config['winlen']
# resolution in ms
binsize = config['binsize'] * pq.ms
# Multiple of the binsize to dither data
dither = config['dither'] * binsize
# number of surrogates for psf
n_surr = config['n_surr']
# Spectrum ('#' for 2d-spectrum '3d#' for 3d-spectrum
spectrum = config['spectrum']
# Minimum size of pattern to analyze
min_spikes = config['min_xi']
# Minimum num occ pattern to analyze
min_occ = config['min_occ']


param = {'winlen':winlen, 'n_surr':n_surr, 'binsize':binsize,
         'spectrum':spectrum, 'min_spikes':min_spikes, 'min_occ':min_occ, 'dither':dither}

# Parsing data parameter
parser = argparse.ArgumentParser(description='Compute spade on artificial data '
                                             'for the given winlen and spectrum parameters')
parser.add_argument('data_idx', metavar='data_idx', type=int,
                   help='idx of the data to analyze (integer between 0 and 3)')
parser.add_argument('xi', metavar='xi', type=int,
                   help='size of injected pattern (integer between 3 and 10')
parser.add_argument('occ', metavar='occ', type=int,
                   help='number of occurrences of injected pattern (integer between 3 and 10')
args = parser.parse_args()
data_idx = args.data_idx
xi = args.xi
occ = args.occ

results = {}

# Loading data
data_path = '../data/stp_data%i.npy' % data_idx
if not os.path.exists(data_path):
    raise ValueError('Data path not existing')
datafile = np.load(data_path, encoding='latin1').item()

# MPI parameters
comm = MPI.COMM_WORLD   # create MPI communicator
rank = comm.Get_rank()  # get rank of current MPI task
size = comm.Get_size()  # get tot number of MPI tasks
print(size)

# Analyzing the 100 realization of data
for rep in range(100):
    print(("Doing rep", rep))
    # Selecting data
    spikedata = datafile['sts_%iocc_%ixi' % (occ, xi)][rep]
    # SPADE analysis
    if xi==0 and occ ==0 and rep == 0:
        output = spade.spade(spikedata, binsize, winlen, dither=dither,
                n_surr=n_surr, min_spikes=min_spikes, min_occ=min_occ, spectrum=spectrum)
    else:
        output = spade.spade(spikedata, binsize, winlen, dither=dither,
                             n_surr=0, min_spikes=min_spikes,
                             min_occ=min_occ, spectrum=spectrum)
    # Storing data
    if rank == 0:
        results[rep] = output
if rank == 0:
    # Storing results
    path = '../results/results_data{}'.format(data_idx)
    path_temp = './'
    for folder in split_path(path):
        path_temp = path_temp + '/' + folder
        mkdirp(path_temp)
    if xi == 0 and occ ==0:
        np.save(
            path + '/result_ind'.format(xi, occ), [results, param])
    else:
        np.save(
            path+'/result_xi{}_occ{}'.format(xi,occ), [results, param])
