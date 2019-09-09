# coding=utf-8
"""
Plotting of the profiling results

The script produces:
Profiling results for different components of SPADE. Run times as a function
of the number of spikes by varying (as shown in the respective second x-axis)
1) the firing rates λ of the neurons (left panel, fixed number of neurons N
and duration T of the data sets),
2) the duration T of the data (central panel, constant rate λ and
N ) and
3) the number of parallel spike trains N (right panel, λ and T).
The profiling times for pattern mining using the Python (FCA) implementation
are shown in blue and for the C++ (FP-growth) implementation in orange.
The profiling times of the PSF using FCA is shown for the 2d in purple
and the 3d in green.
Similarly, the profiling times of the PSF using FP-growth for the 2d
in light blue and the 3d in red.
At the bottom, the same results are represented with logarithmic time axis.
The script reproduces Figure 6 of the publication:
Stella, A., Quaglio, P., Torre, E., & Grün, S. (2019). 3d-SPADE: Significance
Evaluation of Spatio-Temporal Patterns of Various Temporal Extents. Biosystems.

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


def plot(time_fpgrowth, time_fast_fca, keys, label_size=8, tick_size=6):
    """
    Function plotting the profiling time for all SPADE components (FIM and
    PSF), comparing the run time for the fpgrowth and the fast_fca algorithm
    (both implemented in SPADE).
    Parameters
    ----------
    time_fpgrowth : dictionary
        dictionary with profiling time of the fpgrowth algorithm,
        with keys depending on the parameters varying given by 'keys'
        and of the number of estimated spikes in the dataset
    time_fast_fca : dictionary
        dictionary similar to time_fpgrowth, but obtained by the run
        of fast_fca algorithm
    keys : list
        list of keys of the varying variable of the profiling analysis.
        Maximum of three keys, can be either 'neurons', 'time' and
        'rate'. Depending on the keys array, the function produces the
        corresponding plots (1 to 3 panels)
    label_size: int
        label size of the plot. Default is 8
    tick_size: int
        tick size of the plot. Default is 6

    """
    # Plot configurations
    inch2cm = 2.540  # conversion from inches to centimeters

    max_num_colors = 10
    cm = plt.get_cmap('hsv')
    cmap = sns.color_palette("muted", max_num_colors)
    colors = {}
    for i in range(1, max_num_colors + 1):
        color = cm(1. * i // max_num_colors)
        colors[i - 1] = color
    colors = cmap
    f, ax = plt.subplots(2, 3, figsize=(
        18.4 / inch2cm, 9.2 / inch2cm))
    f.subplots_adjust(wspace=0.15, hspace=0.05, bottom=0.2, left=0.07,
                      right=0.98)

    interpolate = False
    linestyle = ':'

    y1_lower_plot = np.array([])
    y2_lower_plot = np.array([])
    y2_upper_plot = np.array([])

    for key_idx, key in enumerate(keys):
        # Plotting all functions
        # Create plots by column
        axes_vertical = [ax[0][key_idx]] + [ax[1][key_idx]]
        for idx, axes in enumerate(axes_vertical):
            # Plot FP-growth
            compute_xy(
                time_fpgrowth[key], count_spikes, axes, function=square,
                label="FP-growth (C++)", colors=colors[1], marker="o",
                interpolate=interpolate, linestyle='-', markerfacecolor='None')
            # Plot FCA
            compute_xy(
                time_fast_fca[key], count_spikes, axes, function=linear,
                label="Fast-FCA (Python)", colors=colors[0], marker="o",
                interpolate=interpolate, linestyle=linestyle,
                markerfacecolor=colors[0])
            # Plot Spectra
            compute_xy(
                np.array(time_fpgrowth[key]) * 2000, count_spikes, axes,
                function=poly4, label="2d FP-growth", colors=colors[9],
                marker="o", interpolate=interpolate, linestyle='-',
                markerfacecolor='None')
            compute_xy(
                np.array(time_fpgrowth[key]) * 2000,
                count_spikes, axes, function=poly4,
                label="3d FP-growth", colors=colors[3], marker="o",
                interpolate=interpolate, linestyle=linestyle,
                markerfacecolor=colors[3])
            compute_xy(
                np.array(time_fast_fca[key]) * 2000, count_spikes, axes,
                function=poly4, label="2d Fast-FCA", colors=colors[4],
                marker="o", interpolate=interpolate, linestyle='-',
                markerfacecolor='None')
            compute_xy(
                np.array(time_fast_fca[key]) * 2000,
                count_spikes, axes, function=poly4,
                label="3d Fast-FCA", colors=colors[2], marker="o",
                interpolate=interpolate, linestyle=linestyle,
                markerfacecolor=colors[2])

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
            title = 'T = ' + str(t_stops[0]) + 's, N = ' + str(ns[0])
            ax[0][key_idx].set_title(title, size=label_size)
        elif key == 'time':
            title = 'N = ' + str(ns[0]) + ', $\lambda$ = ' + str(rates[0]) + \
                    'Hz'
            ax[0][key_idx].set_title(title, size=label_size)
        elif key == 'neurons':
            title = 'T = ' + str(t_stops[0]) + 's, $\lambda$ = ' + \
                    str(rates[0]) + 'Hz'
            ax[0][key_idx].set_title(title, size=label_size)
        else:
            raise ValueError('key not valid')

        if key_idx == 0:
            ax[0][key_idx].set_ylabel("compute time (min)", size=label_size)
            ax[1][key_idx].set_ylabel('log (compute time)', size=label_size)
            # Put legend position
            legend = ax[0][key_idx].legend(loc="best", numpoints=1,
                                           markerscale=0.9,
                                           prop={"size": label_size - 2},
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
        y1_l, y2_l = ax[1][key_idx].get_ylim()
        y1_lower_plot = np.append(y1_lower_plot, y1_l)
        y2_lower_plot = np.append(y2_lower_plot, y2_l)

        # make the time and the neurons plot share the same y axis (only in the
        # max value)
        if key == 'neurons' or key == 'time':
            y1_u, y2_u = ax[0][key_idx].get_ylim()
            y2_upper_plot = np.append(y2_upper_plot, y2_u)

    ymin_l = np.max(y1_lower_plot)
    ymax_l = np.max(y2_lower_plot)
    ymax_u = np.max(y2_upper_plot)
    for key_idx, key in enumerate(keys):
        ax[1][key_idx].set_ylim(ymin_l, ymax_l)
        if key == 'neurons' or key == 'time':
            y1_u, y2_u = ax[0][key_idx].get_ylim()
            ax[0][key_idx].set_ylim(y1_u, ymax_u)

    # Comment this if you want set manually the space between edges and graph
    # see above subplots_adjust
    # plt.tight_layout()

    figname = 'profiling_times'
    figpath = '../figures/'
    path_temp = './'
    for folder in split_path(figpath):
        path_temp = path_temp + '/' + folder
        mkdirp(path_temp)
    f.savefig(figpath + figname + '.' + 'eps', format='eps')


if __name__ == '__main__':
    # setting style of seaborn plot
    sns.set_style('ticks', {'xtick.direction': u'in',
                            'ytick.direction': u'in'})
    sns.set_context("paper")

    # Script variables
    time_fpgrowth = {}
    time_fast_fca = {}

    with open("configfile.yaml", 'r') as stream:
        config = yaml.load(stream)

    # define input variables from configfile
    winlen = config['winlen']
    binsize = config['binsize'] * pq.ms
    rates = config['rate']
    t_stops = config['t_stop']
    ns = config['n']
    keys = config['keys']

    # check if the input is coherent
    if not set(keys).issubset({'rate', 'neurons', 'time'}):
        raise ValueError('set of keys not valid')

    # estimation of the number of spikes, given by the input variables
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
                "../results/{}/{}/profiling_results.npy"
                    .format(key, num_spikes)).item()
            time_fpgrowth[key].append(res['results']['fp_growth'])
            time_fast_fca[key].append(res['results']['fast_fca'])

    # call of the plotting function
    plot(time_fpgrowth=time_fpgrowth, time_fast_fca=time_fast_fca, keys=keys)
