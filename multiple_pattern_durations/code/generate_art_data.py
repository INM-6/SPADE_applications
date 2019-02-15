import numpy as np
import quantities as pq
import neo
import elephant.spike_train_generation as stg
import yaml
try:
    import matplotlib.pyplot as plt
    make_raster =True
    from matplotlib import rc
    # activate latex text rendering
    rc('text', usetex=True)
except:
    make_raster =False
import os
from scipy.stats import binom
import scipy

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

def generate_stp(occ, xi, t_stop, delays, t_start=0 * pq.s):
    '''
    Generate a spatio-temporal-pattern (STP). One pattern consists in a
    repeated sequence of spikes with fixed inter spikes intervals (delays).
    The starting time of the repetitions of the pattern are randomly generated.
    '''
    # Generating all the first spikes of the repetitions
    s1 = np.arange(0, (t_stop - t_start - delays[-1]).simplified.magnitude,
                   (t_stop - t_start - delays[-1]).simplified.magnitude/occ)
    s1 = np.array(
        [s + np.random.uniform(
            high=(t_stop - t_start - delays[-1]).simplified.magnitude/occ) for s in s1])

    # Using matrix algebra to add all the delays
    s1_matr = (s1 * np.ones([xi - 1, occ])).T
    delays_matr = np.ones(
        [occ, 1]) * delays.rescale(t_stop.units).magnitude.reshape([1, xi - 1])
    ss = s1_matr + delays_matr

    # Stacking the first and successive spikes
    stp = np.hstack((s1.reshape(occ, 1), ss))

    # Transofm in to neo SpikeTrain
    stp = [
        neo.core.SpikeTrain(
            t * t_stop.units + t_start, t_stop, t_start=t_start) for t in stp.T]
    return stp


# Load general parameters
with open("configfile.yaml", 'r') as stream:
    config = yaml.load(stream)

xi = config['xi']
winlen =config['winlen']
binsize = config['binsize'] * pq.ms
rate = config['rate'] * pq.Hz
t_stop = config['t_stop'] * pq.s
n = config['n']
alpha = config['alpha']
spectra = config['spectra']

lengths = list(
    range(0,(winlen +1) * int(binsize.magnitude), xi * int(
        binsize.magnitude))) * binsize.units

# Probability to have one repetion of the pattern assuming Poiss
p = (rate.simplified.magnitude * binsize.simplified.magnitude) ** xi

# Computing min_occ as 0.01 percentile of a binominal(n_bins, p)
occ = int(
    binom.isf(0.05, t_stop.simplified.magnitude * scipy.special.binom(n,xi), p))


# Generating the data
# Fixing the seed
np.random.seed(0)
# Generating Independent Background
background_sts = stg._n_poisson(rate=rate, t_stop=t_stop, n=n)
# Total number of patterns
num_neu_patt = len(lengths)*xi
sts = []
stps = []
# Raising errors if the number of neurons is smaller than number of patterns
if num_neu_patt>n:
    raise ValueError('Number of neurons involved in the patterns is larger than total number of patterns')
for l_idx, l in enumerate(lengths):
    lag_bins = int(l / (xi-1))
    delays = [i*lag_bins*binsize.magnitude for i in range(1,xi)]*binsize.units
    print(delays)
    # Generating patterns
    stp = generate_stp(occ=occ,xi=xi,t_stop=t_stop,delays=delays)
    stps.extend(stp)
    # Merging the background spiketrains to the patterns
    for p_idx, p in enumerate(stp):
        sts.append(stg._pool_two_spiketrains(
            background_sts[l_idx*xi + p_idx], p))
sts.extend(background_sts[num_neu_patt:])

data_path = '../data'
# Create path is not already existing
path_temp = './'
for folder in split_path(data_path):
    path_temp = path_temp + '/' + folder
    mkdirp(path_temp)
# Storing data
np.save(data_path + '/art_data.npy', {'patt':stps, 'data':sts, 'params':{
    'xi':xi, 'occ':occ, 'rate':rate, 't_stop':t_stop,'n':n, 'lengths':lengths,
    'binsize':binsize, 'alpha':alpha}})

