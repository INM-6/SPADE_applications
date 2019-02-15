# -*- coding: utf-8 -*-

import numpy as np
import argparse
import elephant.spade as spade
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

# Data to filter
parser = argparse.ArgumentParser(description='Compute spade on artificial data '
                                             'for the given winlen and spectrum parameters')
parser.add_argument('data_idx', metavar='data_idx', type=int,
                   help='idx of the data to analyze (integere between 0 and 3')
parser.add_argument('alpha', metavar='alpha', type=float,
                   help='significance threshold of psf')

args = parser.parse_args()
data_idx = args.data_idx
# Filtering parameters
alpha = args.alpha
with open("configfile.yaml", 'r') as stream:
    config = yaml.load(stream)
psr_param = config['psr_param']
correction = config['correction']
path = '../results/results_data{0}/'.format(data_idx)

res_ind, params = np.load(
                path + 'result_ind.npy', encoding='latin1')
pval_spectrum = res_ind[0]['pvalue_spectrum']
spectrum = params['spectrum']
if len(pval_spectrum) == 0:
    ns_sgnt = []
else:
    # Computing non-significant entries of the spectrum applying
    # the statistical correction
    ns_sgnt = spade.test_signature_significance(
        pval_spectrum, alpha, corr=correction, report='e',
        spectrum=spectrum)

filtered_results = {}
for xi in range(3,11):
    for occ in range(3,11):
        try:
            results, params = np.load(
                path + 'result_xi{xi}_occ{occ}.npy'.format(
                    xi=xi, occ=occ), encoding='latin1')
        except:
            print('not all res available')
            continue
        filtered_results['xi{xi}_occ{occ}'.format(
                xi=xi, occ=occ)] = {}
        # SPADE parameters
        spectrum = params['spectrum']
        min_spikes = params['min_spikes']
        n_surr = params['n_surr']
        winlen = params['winlen']
        binsize = params['binsize']
        min_occ = params['min_occ']
        for rep in results.keys():
            res_spade = results[rep]
            concepts = res_spade['patterns']
            ##### PSF filtering #####
            concepts_psf = list(filter(
                lambda c: spade._pattern_spectrum_filter(
                    c, ns_sgnt, spectrum, winlen), concepts))

            ##### PSR filtering ######
            # Decide whether filter the concepts using psr
            if psr_param is not None:
                # Filter using conditional tests (psr)
                concepts_psr = spade.pattern_set_reduction(concepts_psf, ns_sgnt,
                                                           winlen=winlen,
                                                           h=psr_param[0],
                                                           k=psr_param[1],
                                                           l=psr_param[2],
                                                           min_spikes=min_spikes,
                                                           min_occ=min_occ)
            else:
                concepts_psr = concepts_psf
            patterns = spade.concept_output_to_patterns(
                concepts=concepts_psr, pvalue_spectrum=pval_spectrum,
                winlen=winlen, binsize=binsize)
            # Storing filtered results
            params['alpha'] = alpha
            params['psr_param'] = psr_param
            params['correction'] = correction
            params['min_occ'] = min_occ
            filtered_results['xi{xi}_occ{occ}'.format(
                xi=xi, occ=occ)][rep] = [patterns, concepts_psr]
np.save(path + '/alpha{alpha}result_filtered.npy'.format(
        alpha=alpha), [filtered_results, params])
