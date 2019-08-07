import elephant.spade as spade
import elephant.spike_train_generation as stg
import elephant.conversion as conv
import argparse
import quantities as pq
import numpy as np
import time
import os
import yaml
from utils import mkdirp, split_path, estimate_number_spikes


def compute_profiling_time(key, rate, t_stop, n, winlen, binsize, num_rep=10):
    """
    Function computing the profiling time needed to run SPADE on artificial
    poisson data of given rate, recording time, and number of neurons
    Parameters
    ----------
    key: list
        list of keys of the varying variable of the profiling analysis.
        Maximum of three keys, can be either 'neurons', 'time' and
        'rate'.
    rate: quantity
        rate of the poisson process
    t_stop: quantity
        duration of the spike trains
    n: int
        number of spike trains
    winlen: int
        window length for the SPADE analysis
    binsize: quantity
        binsize for the SPADE analysis
    """

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
        # Computing the context and the binary matrix encoding the relation
        # between objects (window positions) and attributes (spikes,
        # indexed with a number equal to  neuron idx*winlen+bin idx)
        context, transactions, rel_matrix = spade._build_context(binary_matrix,
                                                                 winlen)
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
        # Computing the context and the binary matrix encoding the relation
        # between objects (window positions) and attributes (spikes,
        # indexed with a number equal to  neuron idx*winlen+bin idx)
        context, transactions, rel_matrix = \
            spade._build_context(binary_matrix, winlen)
        # Applying FP-Growth
        fim_results = spade._fast_fca(context, winlen=winlen)
        time_fast_fca += time.time() - t1

    time_profiles = {'fp_growth': time_fpgrowth/num_rep,
                     'fast_fca': time_fast_fca/num_rep}

    # Storing data
    res_path = '../results/{}/{}/'.format(key, expected_num_spikes)
    # Create path is not already existing
    path_temp = './'
    for folder in split_path(res_path):
        path_temp = path_temp + '/' + folder
        mkdirp(path_temp)

    np.save(res_path + '/profiling_results.npy', {'results': time_profiles,
            'parameters': {'rate': rate, 't_stop': t_stop, 'n': n,
                           'winlen': winlen, 'binsize': binsize}})


if __name__ == '__main__':
    # Initializing scripts parameters
    parser = argparse.ArgumentParser(
        description='Profiling the SPADE functions' +
                    'using independent Poisson Processes')
    parser.add_argument('key', metavar='type', type=str,
                        help='varying parameter in data generation')
    parser.add_argument('expected_num_spikes', metavar='rate', type=int,
                        help='expected number of spikes of the Poisson '
                             'processes')

    args = parser.parse_args()
    expected_num_spikes = args.expected_num_spikes
    key = args.key

    with open("configfile.yaml", 'r') as stream:
        config = yaml.load(stream)

    winlen = config['winlen']
    binsize = config['binsize'] * pq.ms
    rates = config['rate']
    t_stops = config['t_stop']
    ns = config['n']
    keys = config['keys']

    expected_num_spikes_dict = estimate_number_spikes(keys=keys,
                                                      ns=ns,
                                                      rates=rates,
                                                      t_stops=t_stops)
    if key == 'rate':
        t_stop = t_stops[0] * pq.s
        n = ns[0]
        rate = expected_num_spikes_dict[key][expected_num_spikes][
                   'rate'] * pq.Hz
    elif key == 'neurons':
        t_stop = t_stops[0] * pq.s
        rate = rates[0] * pq.Hz
        n = expected_num_spikes_dict[key][expected_num_spikes]['n']
    elif key == 'time':
        rate = rates[0] * pq.Hz
        n = ns[0]
        t_stop = \
            expected_num_spikes_dict[key][expected_num_spikes]['t_stop'] * pq.s
    else:
        raise KeyError('key should be either rate or neurons or time')

    compute_profiling_time(key=key,
                           rate=rate,
                           t_stop=t_stop,
                           n=n,
                           winlen=winlen,
                           binsize=binsize)
