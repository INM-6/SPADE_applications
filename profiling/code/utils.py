import os


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


# function to estimate number of spikes
def estimate_number_spikes(keys, ns, rates, t_stops):
    expected_num_spikes = {}
    for key in keys:
        expected_num_spikes[key] = {}
        if key == 'neurons':
            for n in ns:
                expected_num_spikes[key][n*rates[0]*t_stops[0]] = {'n': n,
                                                                   't_stop': t_stops[0],
                                                                   'rate': rates[0]}
        elif key == 'rate':
            for rate in rates:
                expected_num_spikes[key][ns[0]*rate*t_stops[0]] = {'n': ns[0],
                                                                   't_stop': t_stops[0],
                                                                   'rate': rate}
        elif key == 'time':
            for t_stop in t_stops:
                expected_num_spikes[key][ns[0]*rates[0]*t_stop] = {'n': ns[0],
                                                                   't_stop': t_stop,
                                                                   'rate': rates[0]}
        else:
            raise KeyError('key not valid')
    return expected_num_spikes


def estimate_number_spikes(keys, ns, rates, t_stops):
    expected_num_spikes = {}
    for key in keys:
        expected_num_spikes[key] = {}
        if key == 'neurons':
            for n in ns:

                expected_num_spikes[key][n*rates[0]*t_stops[0]] = {'n': n,
                                                                   't_stop': t_stops[0],
                                                                   'rate': rates[0]}
        elif key == 'rate':
            for rate in rates:
                expected_num_spikes[key][ns[0]*rate*t_stops[0]] = {'n': ns[0],
                                                                   't_stop': t_stops[0],
                                                                   'rate': rate}
        elif key == 'time':
            for t_stop in t_stops:
                expected_num_spikes[key][ns[0]*rates[0]*t_stop] = {'n': ns[0],
                                                                   't_stop': t_stop,
                                                                   'rate': rates[0]}
        else:
            raise KeyError('key not valid')
    return expected_num_spikes


keys = ['time', 'neurons', 'rate']
rates = [15,30,45,60,75]
t_stops = [3,6,9,12,15]
ns = [100,200,300,400,500]
dict = estimate_number_spikes(keys=keys, ns=ns, rates=rates, t_stops=t_stops)