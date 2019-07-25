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
from utils import mkdirp, split_path, estimate_number_spikes
sns.set_style('ticks', {'xtick.direction': u'in', 'ytick.direction': u'in'})
sns.set_context("paper")

# Script variables
time_fpgrowth = {}
time_fast_fca = {}

savefig = True

with open("configfile.yaml", 'r') as stream:
    config = yaml.load(stream)

winlen =config['winlen']
binsize = config['binsize'] * pq.ms
rates = config['rate']
t_stops = config['t_stop']
ns = config['n']
keys = config['keys']

# check if the input is coherent
if not set(keys).issubset({'rate', 'neurons', 'time'}):
    raise ValueError('set of keys not valid')

expected_num_spikes_dict = estimate_number_spikes(keys=keys,
                                                  ns=ns,
                                                  rates=rates,
                                                  t_stops=t_stops)

# Loading routine
for key in keys:
    time_fast_fca[key] = []
    time_fpgrowth[key] = []
    # here infer the number of spikes for each key
    count_spikes = list(expected_num_spikes_dict[key].keys())
    for num_spikes in count_spikes:
        res = np.load(
            "../results/{}/{}/profiling_results.npy".format(key, num_spikes)).item()
        time_fpgrowth[key].append(res['results']['fp_growth'])
        time_fast_fca[key].append(res['results']['fast_fca'])


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
        linestyle : string
            Set the linestyle of the key

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
            label=kwargs['label'], markersize=4., linestyle=kwargs['linestyle'],
            markerfacecolor=kwargs['markerfacecolor'])
    return ax


# Plot configurations
label_size = 8
tick_size = 6
inch2cm = 2.540  # conversion from inches to centimeters

colors = {
    "black": (0.0, 0.0, 0.0),
    "darkgray": (0.25, 0.25, 0.25),
    "midgray": (0.5, 0.5, 0.5),
    "lightgray": (0.65, 0.65, 0.65),
    "white": (1.0, 1.0, 1.0)
}
NUM_COLORS = 10
cm = plt.get_cmap('hsv')
cmap = sns.color_palette("muted", NUM_COLORS)
colores = {}
for i in range(1, NUM_COLORS + 1):
    color = cm(1. * i // NUM_COLORS)
    colores[i - 1] = color
colores = cmap
f, ax = plt.subplots(2, 3, figsize=(
    18.4 / inch2cm, 9.2 / inch2cm))
f.subplots_adjust(wspace=0.15, hspace=0.05, bottom=0.2, left=0.07, right=0.98)

interpolate = False
linestyle = ':'

y1_lower_plot = np.array([])
y2_lower_plot = np.array([])

for key_idx, key in enumerate(keys):
    # Plotting all functions
    # Create plots by column
    axes_vertical = [ax[0][key_idx]] + [ax[1][key_idx]]
    for idx, axes in enumerate(axes_vertical):
        # Plot FP-growth
        compute_xy(
            time_fpgrowth[key], count_spikes, axes, function=square,
            label="FP-growth (C++)", colors=colores[1], marker="o",
            interpolate=interpolate, linestyle='-', markerfacecolor='None')
        # Plot FCA
        compute_xy(
            time_fast_fca[key], count_spikes, axes, function=linear,
            label="Fast-FCA (Python)", colors=colores[0], marker="o",
            interpolate=interpolate, linestyle=linestyle,
            markerfacecolor=colores[0])
        # Plot Spectra
        compute_xy(
            np.array(time_fpgrowth[key]) * 2000, count_spikes, axes, function=poly4,
            label="2d FP-growth", colors=colores[9], marker="o",
            interpolate=interpolate, linestyle='-', markerfacecolor='None')
        compute_xy(
            np.array(time_fpgrowth[key]) * 2000,
            count_spikes, axes, function=poly4,
            label="3d FP-growth", colors=colores[3], marker="o",
            interpolate=interpolate, linestyle=linestyle,
            markerfacecolor=colores[3])
        compute_xy(
            np.array(time_fast_fca[key]) * 2000, count_spikes, axes, function=poly4,
            label="2d Fast-FCA", colors=colores[4], marker="o",
            interpolate=interpolate, linestyle='-',
            markerfacecolor='None')
        compute_xy(
            np.array(time_fast_fca[key]) * 2000,
            count_spikes, axes, function=poly4,
            label="3d Fast-FCA", colors=colores[2], marker="o",
            interpolate=interpolate, linestyle=linestyle,
            markerfacecolor=colores[2])

    # Axes specific things
    # Ax 0

    ax[0][key_idx].set_xticks(count_spikes)
    ax[0][key_idx].set_xticklabels([], size=tick_size)
    # ax[0].set_xlabel("number of spikes", size=label_size)
    ax[0][key_idx].tick_params(axis='both', length=2., labelsize=tick_size)
    # Ax 1
    ax[1][key_idx].tick_params(axis='y', which='minor', left='off')
    ax[1][key_idx].set_xticks(count_spikes)
    ax[1][key_idx].set_xticklabels(count_spikes, size=tick_size)
    ax[1][key_idx].set_xlabel("$N_s$", size=label_size)
    ax[1][key_idx].set_yscale('log')
    ax[1][key_idx].tick_params(axis='both', length=2., labelsize=tick_size)

    # set titles only for upper plots
    if key == 'rate':
        title = 'T = ' + str(t_stops[0]) + 's, N = '+ str(ns[0])
        ax[0][key_idx].set_title(title, size=label_size)
    elif key == 'time':
        title = 'N = ' + str(ns[0]) + ', $\lambda$ = '+ str(rates[0]) + 'Hz'
        ax[0][key_idx].set_title(title, size=label_size)
    elif key == 'neurons':
        title = 'T = ' + str(t_stops[0]) + 's, $\lambda$ = '+ str(rates[0]) + 'Hz'
        ax[0][key_idx].set_title(title, size=label_size)
    else:
        raise ValueError('key not valid')

    if key_idx == 0:
        ax[0][key_idx].set_ylabel("compute time (min)", size=label_size)
        ax[1][key_idx].set_ylabel('log (compute time)', size=label_size)
        # Put legend position
        axbox = ax[0][key_idx].get_position()
        legend = ax[0][key_idx].legend(loc="best", numpoints=1,
                                       markerscale=0.9, prop={"size": label_size - 2},
                                       frameon=True, borderpad=0.5)
        legend.get_frame().set_edgecolor('grey')

    # Set second x-axis
    ax2 = ax[1][key_idx].twiny()

    # Decide the ticklabel position in the new x-axis,
    # then convert them to the position in the old x-axis
    if key == 'rate':
        newlabel = rates
        label_add_ax = '$\lambda$ (Hz)'
    elif key == 'time':
        newlabel = t_stops
        label_add_ax = 'T (s)'
    elif key == 'neurons':
        newlabel = ns
        label_add_ax = 'N'
    else:
        raise ValueError('key not valid')

    ax2.set_xticks(count_spikes)
    ax2.tick_params(length=2., labelsize=tick_size)
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 25))
    ax2.set_xlabel(label_add_ax, size=label_size)
    ax2.set_xlim(ax[1][key_idx].get_xlim())
    ax2.set_xticklabels(newlabel, size=tick_size)

    # make the lower share the same ylim
    y1, y2 = ax[1][key_idx].get_ylim()
    y1_lower_plot = np.append(y1_lower_plot, y1)
    y2_lower_plot = np.append(y2_lower_plot, y2)

ymin = np.max(y1_lower_plot)
ymax = np.max(y2_lower_plot)
for key_idx, key in enumerate(keys):
    ax[1][key_idx].set_ylim(ymin, ymax)

# Comment this if you want set manually the space between edges and graph
# see above subplots_adjust
# plt.tight_layout()
# plt.show()

if savefig is True:
    figname = 'profiling_times'
    figpath = '../figures/'
    path_temp = './'
    for folder in split_path(figpath):
        path_temp = path_temp + '/' + folder
        mkdirp(path_temp)
    f.savefig(figpath + figname + '.' + 'eps', format='eps')
