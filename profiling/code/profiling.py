import elephant.spade as spade
import elephant.spike_train_generation as stg
import elephant.conversion as conv
import argparse
import quantities as pq
import numpy as np
import time
import os
import yaml

# Function to create new folders
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

# Initializing scripts parameters
parser = argparse.ArgumentParser(description='Profiling the SPADE functions' + 
                                             'using independent Poisson Processes')
parser.add_argument('expected_num_spikes', metavar='rate', type=int,
                   help='expected number of spikes of the Poisson processes')
args = parser.parse_args()
expected_num_spikes = args.expected_num_spikes

with open("configfile.yaml", 'r') as stream:
    config = yaml.load(stream)

winlen =config['winlen']
binsize = config['binsize'] * pq.ms
rates = config['rate']
t_stops = config['t_stop']
ns = config['n']

expected_num_spikes_dict = {}
for n in ns:
    for rate in rates:
        for t_stop in t_stops:
            expected_num_spikes_dict[n*rate*t_stop] = {
                'n':n, 't_stop':t_stop, 'rate':rate}

# Parsing parameters
rate = expected_num_spikes_dict[expected_num_spikes]['rate'] * pq.Hz
t_stop = expected_num_spikes_dict[expected_num_spikes]['t_stop'] * pq.s
n = expected_num_spikes_dict[expected_num_spikes]['n']

num_rep = 10
time_fast_fca = 0.
time_fpgrowth = 0.
for rep in range(num_rep):
    # Generating artificial data
    data = []
    for i in range(n):
        np.random.seed(0)
        data.append(stg.homogeneous_poisson_process(
            rate=rate, t_start=0*pq.s, t_stop=t_stop))

    # Extracting Closed Frequent Itemset with FP-Growth
    t0 = time.time()
    # Binning the data and clipping (binary matrix)
    binary_matrix = conv.BinnedSpikeTrain(data, binsize).to_bool_array()
    # Computing the context and the binary matrix encoding the relation between
    # objects (window positions) and attributes (spikes,
    # indexed with a number equal to  neuron idx*winlen+bin idx)
    context, transactions, rel_matrix = spade._build_context(binary_matrix, winlen)
    # Applying FP-Growth
    fim_results = [i for i in spade._fpgrowth(
                transactions,
                rel_matrix=rel_matrix,
                winlen=winlen)]
    time_fpgrowth += time.time() - t0

    # Extracting Closed Frequent Itemset with Fast_fca
    t1 = time.time()
    # Binning the data and clipping (binary matrix)
    binary_matrix = conv.BinnedSpikeTrain(data, binsize).to_bool_array()
    # Computing the context and the binary matrix encoding the relation between
    # objects (window positions) and attributes (spikes,
    # indexed with a number equal to  neuron idx*winlen+bin idx)
    context, transactions, rel_matrix = spade._build_context(binary_matrix, winlen)
    # Applying FP-Growth
    fim_results = spade._fast_fca(
                context,
                winlen=winlen)
    time_fast_fca += time.time() - t1

time_profiles = {'fp_growth':time_fpgrowth/num_rep, 'fast_fca':time_fast_fca/num_rep}

# Storing data
res_path = '../results/{}/'.format(expected_num_spikes)
# Create path is not already existing
path_temp = './'
for folder in split_path(res_path):
    path_temp = path_temp + '/' + folder
    mkdirp(path_temp)

np.save(res_path + '/profiling_results.npy',{'results':time_profiles,
        'parameters': {'rate':rate, 't_stop':t_stop, 'n':n, 'winlen':winlen,
        'binsize':binsize}})