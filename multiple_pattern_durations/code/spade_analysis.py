import elephant.spade as spade
import numpy as np
import os
import argparse
from mpi4py import MPI  # for parallelized routines
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

data_file = np.load('../data/art_data.npy', encoding='latin1').item()

data = data_file['data']
data_param = data_file['params']

parser = argparse.ArgumentParser(description='Compute spade on artificial data '
                                             'for the given winlen and spectrum parameters')
parser.add_argument('winlen', metavar='winlen', type=int,
                   help='winlen parameter of the spade function')
parser.add_argument('spectrum', metavar='spectrum', type=str,
                   help='spectrum parameter of the spade function')
args = parser.parse_args()

min_spikes = data_param['xi']
max_spikes = data_param['xi']
with open("configfile.yaml", 'r') as stream:
    config = yaml.load(stream)
n_surr = config['n_surr']
binsize = data_param['binsize']
winlen = args.winlen
spectrum = args.spectrum
param = {'winlen':winlen, 'n_surr':n_surr, 'binsize':binsize,
         'spectrum':spectrum, 'min_spikes':min_spikes, 'max_spikes':max_spikes}
# Check MPI parameters
comm = MPI.COMM_WORLD   # create MPI communicator
rank = comm.Get_rank()  # get rank of current MPI task
size = comm.Get_size() # Get the total number of MPI processes
print('Number of processes:{}'.format(size))

# Compute spade res
print('Running spade')
spade_res = spade.spade(data, binsize=binsize, winlen=winlen, n_surr=n_surr,
                           min_spikes=min_spikes, max_spikes=max_spikes,
                        spectrum=spectrum, alpha=1, psr_param=None)


# Storing data
res_path = '../results/{}/winlen{}'.format(spectrum, winlen)
# Create path is not already existing
path_temp = './'
for folder in split_path(res_path):
    path_temp = path_temp + '/' + folder
    mkdirp(path_temp)

np.save(res_path + '/art_data_results.npy',[spade_res,param])