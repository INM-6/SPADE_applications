# coding=utf-8
"""
Plotting the profiling results for different times
"""
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import yaml
import quantities as pq
sns.set_style('ticks', {'xtick.direction': u'in', 'ytick.direction': u'in'})
sns.set_context("paper")

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


# Script variables
time_fpgrwoth = []
time_fast_fca = []

savefig = True

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
count_spikes = sorted(list(expected_num_spikes_dict.keys()))
# Loading routine
for num_spikes in count_spikes:
    res = np.load(
        "../results/{}/profiling_results.npy".format(num_spikes)).item()
    time_fpgrwoth.append(res['results']['fp_growth'])
    time_fast_fca.append(res['results']['fast_fca'])


def square(x, b, b0):
    """
    Defines a square function to fit the curve

    Parameters
    ----------
    x: numpy.ndarray:
        x of f(x)
    b: float
        Parameter to fit
    b0 : int
         y-intercept of the curve

    Returns
    -------
    f : numpy.ndarray
        Result of f(x)
    """
    return b * np.array(x) ** 2 + b0


def square2(x, b, b0):
    return b * np.array(np.sqrt(x)) ** 5 + b0


def root(x, b, b0):
    return b * np.sqrt(x)


def poly4(x, b, b0):
    """
    Defines a function with polynom 4 to fit the curve

    Parameters
    ----------
    x: numpy.ndarray:
        x of f(x)
    b: float
        Parameter to fit
    b0 : int
         y-intercept of the curve

    Returns
    -------
    f : numpy.ndarray
        Result of f(x)
    """
    return b * np.array(x) ** 4 + b0


def linear(x, b, b0):
    """
    Defines a linear function to fit the curve

    Parameters
    ----------
    x: numpy.ndarray:
        x of f(x)
    b: float
        Parameter to fit
    b0 : int
         y-intercept of the curve

    Returns
    -------
    f : numpy.ndarray
        Result of f(x)
    """
    return b * np.array(x) + b0


def extrapolate(x, y, k, offset=0):
    """
    Parameters
    ----------
    x : list
        x data
    y : list
        y data
    k : int
        Polynomial degree
    offset: int
        Determines how many points should be considered additionally when
        extrapolating. If not set (`k=0`) the length of y is taken and the points
        are then fitted only.

    Returns
    -------
    y_polated : numpy.ndarray
        Extrapolated result using univariate splines.
    """
    extrapolator = UnivariateSpline(x[:len(y)], y, k=k)
    y_polated = extrapolator(x[:(len(y) + offset)])
    return y_polated


def compute_xy(data, xaxis, ax=None, **kwargs):
    """
    Parameters
    ----------
    data : list
        List with datapoints
    xaxis : list
        Contains spike counts as x ticks
    ax : matplotlib.axes
        Axes of the figure
    kwargs :
        function: Function
            A function to be fit to
        label : string
            Label for data
        colors : Dictionary
            Contains colors
        marker : string
            Marker of the curve
        interpolate : bool
            Determines if the curve should be interpolated or not
        linesytle : string
            Set the linestyle of the plot

    Returns
    -------
    """
    if ax is None:
        ax = plt.gca()
    y = data
    if len(xaxis) != len(y):
        xaxis = np.array(xaxis)[:len(y)]
    if 'interpolate' in kwargs and kwargs['interpolate']:
        popt, pcov = curve_fit(kwargs['function'], xaxis, y, p0=[0.001, 0.1],
                               bounds=(0, np.inf))
        x_min_max = np.arange(min(xaxis) - 100, max(xaxis) + 1000, 0.1)
        ax.plot(x_min_max,
                np.array(kwargs['function'](
                    x_min_max, popt[0], popt[1])) / 60.0,
                color=kwargs['colors'], linewidth=0.7, linestyle=kwargs['linestyle'])
    ax.plot(xaxis, np.array(y) / 60.0, '.', color=kwargs['colors'], marker=kwargs['marker'],
            label=kwargs['label'], markersize=3., linestyle=kwargs['linestyle'])
    return ax


# Plot configurations
label_size = 6
tick_size = 5
inch2cm = 2.540  # conversion from inches to centimeters

colors = {
    "black": (0.0, 0.0, 0.0),
    "darkgray": (0.25, 0.25, 0.25),
    "midgray": (0.5, 0.5, 0.5),
    "lightgray": (0.65, 0.65, 0.65),
    "white": (1.0, 1.0, 1.0)
}
NUM_COLORS = 6
cm = plt.get_cmap('hsv')
cmap = sns.color_palette("husl", NUM_COLORS)
colores = {}
for i in range(1, NUM_COLORS + 1):
    color = cm(1. * i // NUM_COLORS)
    colores[i - 1] = color
colores = cmap
f, ax = plt.subplots(2, figsize=(
    8.5 / inch2cm, 10 / inch2cm), sharex=False)

interpolate = False
linestyle = ':'

# Plotting all functions
for idx, axes in enumerate(ax):
    # Plot FP-growth
    compute_xy(
        time_fpgrwoth, count_spikes, axes, function=square,
        label="FP-growth", colors=colores[3], marker="*", interpolate=interpolate,
        linestyle='-.')
    # Plot FCA
    compute_xy(
        time_fast_fca, count_spikes, axes, function=linear,
        label="Fast-FCA", colors=colores[1], marker="o",
        interpolate=interpolate, linestyle=linestyle)
    # Plot Spectra
    compute_xy(
        np.array(time_fpgrwoth)*2000, count_spikes, axes, function=poly4,
        label="2d spectrum FP-growth", colors=colores[2], marker="d",
        interpolate=interpolate, linestyle='-.')
    compute_xy(
        np.array(time_fpgrwoth) * 2000,
        count_spikes, axes, function=poly4,
        label="3d spectrum FP-growth", colors=colores[0], marker="v",
        interpolate=interpolate, linestyle=linestyle)
    compute_xy(
        np.array(time_fast_fca)*2000, count_spikes, axes, function=poly4,
        label="2d spectrum Fast-FCA", colors=colores[4], marker="d",
        interpolate=interpolate, linestyle='-.')
    compute_xy(
        np.array(time_fast_fca) * 2000,
        count_spikes, axes, function=poly4,
        label="3d spectrum Fast-FCA", colors=colores[5], marker="v",
        interpolate=interpolate, linestyle=linestyle)


# Axes specific things
# Ax 0
ax[0].set_ylim(-10, np.max(np.array(time_fast_fca) * 2000) / 60.0 + 100)
ax[0].set_ylabel("compute time (min)", size=label_size)
ax[0].set_xticks(count_spikes)
ax[0].set_xticklabels(count_spikes, size=tick_size)
ax[0].set_xlabel("number of spikes", size=label_size)
ax[0].tick_params(axis='both', length=2., labelsize=tick_size)
# Ax 1
ax[1].tick_params(axis='y', which='minor', left='off')
ax[1].set_xticks(count_spikes)
ax[1].set_xticklabels(count_spikes, size=tick_size)
ax[1].set_xlabel("number of spikes", size=label_size)
ax[1].set_ylabel('log (compute time)', size=label_size)
ax[1].set_yscale('log')
ax[1].tick_params(axis='both', length=2., labelsize=tick_size)

# Put legend position
axbox = ax[0].get_position()
legend = ax[0].legend(loc="best", numpoints=1,
                      markerscale=0.9, prop={"size": label_size - 1},
                      frameon=True, borderpad=0.1)
legend.get_frame().set_edgecolor('black')

# Comment this if you want set manually the space between edges and graph
# see above subplots_adjust
plt.tight_layout()
# plt.show()

if savefig is True:
    figname = 'profiling_times'
    figpath = '../figures/'
    path_temp = './'
    for folder in split_path(figpath):
        path_temp = path_temp + '/' + folder
        mkdirp(path_temp)
    f.savefig(figpath + figname + '.' + 'eps', format='eps')
