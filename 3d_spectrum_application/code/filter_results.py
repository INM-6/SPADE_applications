import numpy as np
import elephant.spade as spade
import argparse
import yaml
# Function to filter patterns when the output format os spade function is 'patterns'
def _pattern_spectrum_filter(patterns, ns_signature, spectrum, winlen):
    '''Filter to select concept which signature is significant'''
    if spectrum == '3d#':
        keep_concept = patterns['signature'] + tuple([max(
                np.abs(np.diff(np.array(patterns['lags'])%winlen)))]) not in ns_signature
    else:
        keep_concept = patterns['signature'] not in ns_signature

    return keep_concept

# Load parameters dictionary
param_dict = np.load('../data/art_data.npy', encoding='latin1').item()['params']
lengths = param_dict['lengths']
binsize = param_dict['binsize']
winlens = [int(l/binsize)+1 for l in lengths]
print(winlens)
# Filtering parameters
# Load general parameters
with open("configfile.yaml", 'r') as stream:
    config = yaml.load(stream)
alpha = config['alpha']
psr_param = config['psr_param']
correction = config['correction']
min_occ = config['min_occ']
# Passing spectrum parameter
parser = argparse.ArgumentParser(description='Compute spade on artificial data '
                                             'for the given winlen and spectrum parameters')
parser.add_argument('spectrum', metavar='spectrum', type=str,
                   help='spectrum parameter of the spade function')
args = parser.parse_args()
spectrum = args.spectrum

# Filtering parameters for the different window length
for winlen in winlens:
    # Loading result
    try:
        res_spade, params = np.load('../results/{}/winlen{}/art_data_results.npy'.format(spectrum,winlen), encoding='latin1')
    except:
        print('Not all results available')
        continue
    concepts = res_spade['patterns']
    pval_spectrum  = res_spade['pvalue_spectrum']
    # SPADE parameters
    spectrum = params['spectrum']
    min_spikes = params['min_spikes']
    n_surr = params['n_surr']
    ##### PSF filtering #####
    if len(pval_spectrum) == 0:
        ns_sgnt = []
    else:
        # Computing non-significant entries of the spectrum applying
        # the statistical correction
        ns_sgnt = spade.test_signature_significance(
            pval_spectrum, alpha, corr=correction, report='e',
            spectrum=spectrum)
    concepts_psf = list(filter(
        lambda c: spade._pattern_spectrum_filter(
            c, ns_sgnt, spectrum, winlen), concepts))
    print(winlen)
    print(sorted(ns_sgnt))
    print(len(concepts_psf))
    #### PSR filtering ######
    # Decide whether filter the concepts using psr
    if psr_param is not None:
        # Filter using conditional tests (psr)
        if 0 < alpha < 1 and n_surr > 0:
            concepts_psr = spade.pattern_set_reduction(concepts_psf, ns_sgnt,
                                                       winlen=winlen,
                                                       h=psr_param[0],
                                                       k=psr_param[1],
                                                       l=psr_param[2],
                                                       min_spikes=min_spikes,
                                                       min_occ=min_occ)
        else:
            concepts_psr = spade.pattern_set_reduction(concepts_psf, [],
                                                       winlen=winlen,
                                                       h=psr_param[0],
                                                       k=psr_param[1],
                                                       l=psr_param[2],
                                                       min_spikes=min_spikes,
                                                       min_occ=min_occ)
        patterns = spade.concept_output_to_patterns(
            concepts_psr, winlen, binsize, pval_spectrum)
    else:
        patterns = spade.concept_output_to_patterns(
            concepts_psf, winlen, binsize, pval_spectrum)
    # Storing filtered results
    params['alpha'] = alpha
    params['psr_param'] = psr_param
    params['correction'] = correction
    params['min_occ'] = min_occ
    np.save(
        '../results/{}/winlen{}/filtered_patterns.npy'.format(
            spectrum, winlen), [patterns, pval_spectrum, ns_sgnt, params])