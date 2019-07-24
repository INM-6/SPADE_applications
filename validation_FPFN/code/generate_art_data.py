# -*- coding: utf-8 -*-

import elephant.spike_train_generation as stg
import quantities as pq
import time
import numpy as np
import neo
import os
import random
import yaml
from utils import mkdirp, split_path

t0 = time.time()


def generate_stp(occurr, xi, t_stop, delays, t_start=0 * pq.s):
    '''
    Generate a spatio-temporal-pattern (STP). One pattern consists in a
    repeated sequence of spikes with fixed inter spikes intervals (delays).
    The starting time of the repetitions of the pattern are randomly generated.
    '''
    # Generating all the first spikes of the repetitions
    s1 = np.sort(
        np.random.uniform(
            high=(t_stop - t_start - delays[-1]).magnitude, size=occurr))

    # Using matrix algebra to add all the delays
    s1_matr = (s1 * np.ones([xi - 1, occurr])).T
    delays_matr = np.ones(
        [occurr, 1]) * delays.rescale(t_stop.units).magnitude.reshape([1, xi - 1])
    ss = s1_matr + delays_matr

    # Stacking the first and successive spikes
    stp = np.hstack((s1.reshape(occurr, 1), ss))

    # Transorm in to neo SpikeTrain
    stp = [
        neo.core.SpikeTrain(
            t * t_stop.units + t_start, t_stop, t_start=t_start) for t in stp.T]
    return stp


def generate_sts(data_type, N=100):
    '''
    Generate a list of parallel spike trains with different statistics.

    The data are composed of background spiking activity plus possibly
    a repeated sequence of synchronous events (SSE).
    The background activity depends on the value of data_type.
    The size and occurrence count of the SSE is specified by sse_params.

    Parameters
    ----------
    data_type : int
        An integer specifying the type of background activity.
        At the moment the following types of background activity are
        supported (note: homog = across neurons; stat = over time):
        0 : 100 indep Poisson with rate 25 Hz
        1: 100 indep Poisson nonstat-step (10/60/10 Hz)
        2: 100 indep Poisson heterog (5->25 Hz), stat
        3 : 100 indep Poisson, rate increase with latency variability
    N: int 
        total number of neurons in the model. The default is N=100.
    Output
    ------
    sts : list of SpikeTrains
        a list of spike trains
    params : dict
        a dictionary of simulation parameters

    '''
    T = 1 * pq.s  # simulation time
    sampl_period = 10 * pq.ms           # sampling period of the rate profile
    params = {'nr_neurons': N, 'simul_time': T}
    # Indep Poisson homog, stat rate 25 Hz
    if data_type == 0:
        # Define a rate profile
        rate = 25 * pq.Hz
        # Generate data
        sts = stg._n_poisson(rate=rate, t_stop=T, n=N)
        # Storing rate parameter
        params['rate'] = rate
    # Indep Poisson,  homog, nonstat-step (10/60/10 Hz)
    elif data_type == 1:
        a0, a1 = 10 * pq.Hz, 60 * pq.Hz     # baseline and transient rates
        t1, t2 = 600 * pq.ms, 700 * pq.ms   # time segment of transient rate
        # Define a rate profile
        times = sampl_period.units * np.arange(
            0, T.rescale(sampl_period.units).magnitude, sampl_period.magnitude)
        rate_profile = np.zeros(times.shape)
        rate_profile[np.any([times < t1, times > t2], axis=0)] = a0.magnitude
        rate_profile[np.all([times >= t1, times <= t2], axis=0)] = a1.magnitude
        rate_profile = rate_profile * a0.units
        rate_profile = neo.AnalogSignal(
            rate_profile, sampling_period=sampl_period)
        # Generate data
        sts = [stg.inhomogeneous_poisson_process(
            rate_profile) for i in range(N)]
        # Storing rate parameter
        params['rate'] = rate_profile
    # Indep Poisson, heterog (5->15 Hz), stat
    elif data_type == 2:
        rate_min = 5 * pq.Hz     # min rate. Ensures that there is >=1 spike
        rate_max = 25 * pq.Hz    # max rate
        rates = np.linspace(rate_min.magnitude, rate_max.magnitude, N) * pq.Hz
        # Define a rate profile
        # Generate data
        sts = [stg.homogeneous_poisson_process(
            rate=rate, t_stop=T) for rate in rates]
        random.shuffle(sts)
        # Storing rate parameter
        params['rate'] = rates
    # Indep Poisson, rate increase sequence
    elif data_type == 3:
        l = 20  # 20 groups of neurons
        w = 5   # of 5 neurons each
        t0 = 50 * pq.ms  # the first of which increases the rate at time t0
        t00 = 500 * pq.ms  # and again at time t00
        ratechange_dur = 5 * pq.ms  # old: 10ms  # for a short period
        a0, a1 = 14 * pq.Hz, 100 * pq.Hz  # from rate a0 to a1
        ratechange_delay = 5 * pq.ms  # old: 10ms; followed with delay by next group
        # Define a rate profile
        times = sampl_period.units * np.arange(
            0, T.rescale(sampl_period.units).magnitude, sampl_period.magnitude)
        sts = []
        rate_profiles = []
        for i in range(l * w):
            t1 = t0 + (i // w) * ratechange_delay
            t2 = t1 + ratechange_dur
            t11 = t00 + (i // w) * ratechange_delay
            t22 = t11 + ratechange_dur
            rate_profile = np.zeros(times.shape)
            rate_profile[np.any([times < t1, times > t2], axis=0)] = \
                a0.magnitude
            rate_profile[np.all([times >= t1, times <= t2], axis=0)] = \
                a1.magnitude
            rate_profile[np.all([times >= t11, times <= t22], axis=0)] = \
                a1.magnitude
            rate_profile = rate_profile * a0.units
            rate_profile = neo.AnalogSignal(
                rate_profile, sampling_period=sampl_period)
            rate_profiles.append(rate_profile)
            # Generate data
            sts.append(stg.inhomogeneous_poisson_process(rate_profile))
        # Storing rate parameter
        params['rate'] = rate_profiles
    else:
        raise ValueError(
            'data_type %d not supported. Provide int from 0 to 10' % data_type)
    return sts, params
# Load general parameters
with open("configfile.yaml", 'r') as stream:
    config = yaml.load(stream)
data_idxs = config['data_idxs']
for data_idx in data_idxs:
    # Dictionary containing all the simulated data for a given rate
    sts_rep = {}
    # Iterating to generate many dataset for the one parameter setting
    for i in range(100):
        np.random.seed(i)
        # Generate the independent background of sts
        sts, params_background= generate_sts(data_idx)
        # Storing te independent sts
        if i == 0:
            sts_rep['sts_0occ_0xi'] = [sts]
        else:
            sts_rep['sts_0occ_0xi'].append(sts)
        # Iterating different complexities of the patterns
        for xi in range(2, 11):
            # Iterating different numbers of occurrences
            for occurr in range(3, 11):
                # Generating the stp
                np.random.seed(i * 100 + xi + occurr)
                stp = generate_stp(
                    occurr, xi, 1 * pq.s, np.arange(5, 5 * (xi), 5) * pq.ms)
                # Merging the stp in the first xi sts
                sts_pool = [0] * xi
                for st_id, st in enumerate(stp):
                    sts_pool[st_id] = stg._pool_two_spiketrains(
                        st, sts[st_id])
                # Storing datasets containg stps
                if i == 0:
                    sts_rep[
                        'sts_%iocc_%ixi' % (occurr, xi)] = [
                            sts_pool + sts[xi:]]
                    sts_rep['stp_%iocc_%ixi' % (occurr, xi)] = [stp]
                else:
                    sts_rep[
                        'sts_%iocc_%ixi' % (occurr, xi)].append(
                            sts_pool + sts[xi:])
                    sts_rep['stp_%iocc_%ixi' % (occurr, xi)].append(stp)
            sts_rep['params_background'] = params_background

    # Saving the datasets
    filepath = '../data/'
    path_temp = './'
    for folder in split_path(filepath):
        path_temp = path_temp + '/' + folder
        mkdirp(path_temp)

    filename = 'stp_data%i' % (data_idx)
    np.save(filepath + filename, sts_rep)

print((time.time() - t0))
