from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import inspect
import PySimpleGUI as sg
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import make_axes_locatable

from urllib.parse import urlencode
from urllib.request import urlretrieve

import numpy as np
import numpy.core.defchararray as np_f
import pandas as pd
import scipy as sci
from scipy.stats import f
from scipy.stats import kde
from subprocess import *
import os
import glob
from pathlib import Path
import re

from astropy.table import Table
from astropy import constants as const
from astropy import units as u
from astropy.io import fits
from astropy import coordinates as coords

import importlib
import tqdm

import time

import warnings

import ResearchTools.LCtools as LCtools
import VarStar_Vi_plot_functions as vi

from astropy.timeseries import LombScargle
from sklearn.cluster import MeanShift, estimate_bandwidth

spec_dir = "/Users/benjaminroulston/Dropbox/Research/TDSS/Variable_Stars/HARD_COPY_ORGINAL_DATA/SDSS_spec/02-26-2020/SDSSspec/"
CSS_LC_dir = "/Users/benjaminroulston/Dropbox/Research/TDSS/Variable_Stars/HARD_COPY_ORGINAL_DATA/CSS_LCs/csvs/"
ZTF_LC_dir = "/Users/benjaminroulston/Dropbox/Research/TDSS/Variable_Stars/HARD_COPY_ORGINAL_DATA/ZTF/DATA/06-24-2020/"

ZTF_filters = ['g', 'r']
ZTF_LC_file_names = [f'TDSS_SES+PREV_DR16DR12griLT20_GAIADR2_Drake2014PerVar_ZTF_{ZTF_filter}_epochGT10_GroupID.fits' for ZTF_filter in ZTF_filters]
ZTF_g_LCs = Table.read(ZTF_LC_dir + ZTF_LC_file_names[0])
ZTF_r_LCs = Table.read(ZTF_LC_dir + ZTF_LC_file_names[1])

prop_out_dir, CSS_LC_plot_dir, ZTF_LC_plot_dir, Vi_plots_dir, datestr = vi.makeViDirs()
nbins = 50
TDSSprop = vi.TDSSprop(nbins)
latestFullVartoolsRun_filename = "completed_Vi_prop_2020-07-16.csv"
latestFullVartoolsRun = vi.latestFullVartoolsRun(prop_out_dir + latestFullVartoolsRun_filename)

hasViRun, prop_id_last, properties = vi.checkViRun()  # if Vi has run, this will find where it let off and continue propid from there

prop_col_names_prefix = ['CSS_', 'ZTF_g_', 'ZTF_r_']
if hasViRun:
    pass
else:
    prop_id = 0
    prop_id_last = 0
    prop_col_names = ['lc_id', 'P', 'logProb', 'Amp', 'Mt', 'a95', 'lc_skew',
                      'Chi2', 'brtcutoff', 'brt10per', 'fnt10per', 'fntcutoff', 'errmn', 'ferrmn',
                      'ngood', 'nrejects', 'nabove', 'nbelow', 'Tspan100', 'Tspan95', 'isAlias', 'time_whittened',
                      'VarStat', 'Con', 'm', 'b_lin', 'chi2_lin', 'a', 'b_quad', 'c', 'chi2_quad']

    prop_col_names_full = [ii + jj for ii in prop_col_names_prefix for jj in prop_col_names]
    prop_col_names_full.insert(0, 'ViCompleted')
    prop_col_names_full.insert(0, 'dec')
    prop_col_names_full.insert(0, 'ra')
    prop_col_names_full.append('EqW')

    properties = np.zeros((len(TDSSprop.data), len(prop_col_names_full)))

    properties = Table(properties, names=prop_col_names_full)
    properties['ra'] = TDSSprop.data['ra']
    properties['dec'] = TDSSprop.data['dec']

runLS = True
plotLCerr = True
plt_resid = False
plt_subLC = True
plot_rejected = False
checkHarmonic = True
logProblimit = -10
Nepochs_required = 10
minP = 0.1
maxP = 100.0
nterms_LS = 1

prop = Table.read("/Users/benjaminroulston/Dropbox/Research/TDSS/Variable_Stars/HARD_COPY_ORGINAL_DATA/PROGRAM_SAMPLE/2020-06-24/FINAL_FILES/TDSS_SES+PREV_DR16DR12griLT20_GAIADR2_Drake2014PerVar_VSX_CSSID_ZTFIDs_LCpointer_PyHammer_VI_ALLPROP_07-27-2020.fits")

def freq2per(frequency, period_unit=u.d):
    return (frequency**-1).to(period_unit)


def per2freq(period, frequency_unit=u.microHertz):
    return (period**-1).to(frequency_unit)


prop_id = 5 # 12
ROW = TDSSprop.data[prop_id]

object_ra = ROW['ra']
object_dec = ROW['dec']
ra_string = '{:0>9.5f}'.format(object_ra)
dec_string = '{:0=+9.5f}'.format(object_dec)

is_CSS = ROW['CSSLC']
is_ZTF_g = np.isfinite(ROW['ZTF_g_GroupID'])
is_ZTF_r = np.isfinite(ROW['ZTF_r_GroupID'])

if is_CSS:
    lc_file = CSS_LC_dir+str(ROW['CSSID'])+'.dat'
    CSS_lc_data = Table.read(lc_file, format='ascii', names=['mjd', 'mag', 'magerr'])
    CSS_lc_data.sort('mjd')
if is_ZTF_g:
    ZTF_g_lc_data = ZTF_g_LCs[(ZTF_g_LCs['GroupID'] == ROW['ZTF_g_GroupID'])]['mjd', 'mag', 'magerr']
    ZTF_g_lc_data.sort('mjd')
if is_ZTF_r:
    ZTF_r_lc_data = ZTF_r_LCs[(ZTF_r_LCs['GroupID'] == ROW['ZTF_r_GroupID'])]['mjd', 'mag', 'magerr']
    ZTF_r_lc_data.sort('mjd')

# print(ra_string, dec_string)
# print(is_CSS, is_ZTF_g, is_ZTF_r)
# print(len(CSS_lc_data), len(ZTF_g_lc_data), len(ZTF_r_lc_data))

flc_data, LC_stat_properties = LCtools.process_LC(CSS_lc_data, fltRange=5.0)

goodQualIndex = np.where(flc_data['QualFlag'] == True)[0]
lc_mjd = flc_data['mjd'][goodQualIndex]
lc_mag = flc_data['mag'][goodQualIndex]
lc_err = flc_data['magerr'][goodQualIndex]

t_days = lc_mjd * u.day
y_mags = lc_mag * u.mag
dy_mags = lc_err * u.mag

P_range = [0.1 * u.d, np.inf * u.d]
Nf = 250000  # Number of frequencuies to check

minP = P_range[0].to(u.d)
maximum_frequency = (minP)**-1

maxP = P_range[1].to(u.d)
minimum_frequency = (maxP)**-1

freq_grid = np.linspace(minimum_frequency.value, maximum_frequency.value, num=Nf + 1)[1:] / u.d

# ls_window = LombScargle(t_days, np.ones(y_mags.size), fit_mean=False, center_data=False)
# window_power = ls_window.power(frequency=freq_grid)
# window_power[~np.isfinite(window_power)] = 0
# window_FAP_power_peak = np.nanstd(window_power).value * 4


ls = LombScargle(t_days, y_mags, dy_mags, fit_mean=True, center_data=True)
power = ls.power(frequency=freq_grid)
logFAP_limit = -10
FAP_power_peak = ls.false_alarm_level(10**logFAP_limit)
df = (1 * u.d)**-1

P = freq2per(freq_grid[np.argmax(power)]).value
FAP = ls.false_alarm_probability(power.max())
# print(f"The period with the largest period is: P = {P}d.")
# print(f"The corosponding log10(FAP) is: los10(FAP) = {np.log10(FAP)}")

sorted_P = np.flip(freq2per(freq_grid[np.argsort(power)]))
X = sorted_P[:100].reshape(-1,1)
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_
labels_unique, labels_index, labels_inverse, labels_inverse = np.unique(labels, return_index=True, return_inverse=True, return_counts=True)
n_clusters_ = len(labels_unique)
# print("number of estimated clusters : %d" % n_clusters_)

P1 = X.flatten()[np.where(labels == np.argsort(labels_index)[0])[0][0]].value
P2 = X.flatten()[np.where(labels == np.argsort(labels_index)[1])[0][0]].value
P3 = X.flatten()[np.where(labels == np.argsort(labels_index)[2])[0][0]].value

# print(f"The top 3 periods are: \n {P1}d \n {P2}d \n {P3}d")

title = "RA: {!s} DEC: {!s}".format(ra_string, dec_string)
LCtools.plot_LC_analysis_Pal2013(flc_data, P1, P2, P3, freq_grid, power, FAP_power_peak=FAP_power_peak, logFAP_limit=logFAP_limit, df=df, title=title)

#  The magic function that makes it possible.... glues together tkinter and pyplot using Canvas Widget
def draw_plot_Pal2013(lc_data, P1, P2, P3, frequency, power, FAP_power_peak=None, logFAP_limit=None, df=None, title=""):
    fig = plt.figure(figsize=(32, 18), constrained_layout=False)
    gs = GridSpec(8, 5, figure=fig)

    ax1 = fig.add_subplot(gs[0, :4])
    ax2 = fig.add_subplot(gs[1, :4])

    ax3 = fig.add_subplot(gs[:2, 4])

    ax4 = fig.add_subplot(gs[2:4, 0])
    ax5 = fig.add_subplot(gs[2:4, 1])
    ax6 = fig.add_subplot(gs[2:4, 2])
    ax7 = fig.add_subplot(gs[2:4, 3])
    ax8 = fig.add_subplot(gs[2:4, 4])

    ax9 = fig.add_subplot(gs[4:6, 0])
    ax10 = fig.add_subplot(gs[4:6, 1])
    ax11 = fig.add_subplot(gs[4:6, 2])
    ax12 = fig.add_subplot(gs[4:6, 3])
    ax13 = fig.add_subplot(gs[4:6, 4])

    ax14 = fig.add_subplot(gs[6:8, 0])
    ax15 = fig.add_subplot(gs[6:8, 1])
    ax16 = fig.add_subplot(gs[6:8, 2])
    ax17 = fig.add_subplot(gs[6:8, 3])
    ax18 = fig.add_subplot(gs[6:8, 4])

    LCtools.plot_powerspec(frequency, power, ax1=ax1, ax2=ax2, FAP_power_peak=FAP_power_peak, logFAP_limit=logFAP_limit, alias_df=df, title=title)

    LCtools.plt_any_lc_ax(lc_data, P1, is_Periodic=False, ax=ax3, title="", phasebin=True, bins=25)

    LCtools.plt_any_lc_ax(lc_data, (1 / 3) * P1, is_Periodic=True, ax=ax4, title="f = {!s}$\mu$Hz $|$ 0.33".format(np.round(((P1 * u.d)**-1).to(u.microHertz),2)), stdTitle=False, phasebin=True, bins=25)
    LCtools.plt_any_lc_ax(lc_data, (1 / 2) * P1, is_Periodic=True, ax=ax5, title="f = {!s}$\mu$Hz $|$ 0.5".format(np.round(((P1 * u.d)**-1).to(u.microHertz),2)), stdTitle=False, phasebin=True, bins=25)
    LCtools.plt_any_lc_ax(lc_data, 1.0 * P1, is_Periodic=True, ax=ax6, title="f = {!s}$\mu$Hz $|$ ".format(np.round(((P1 * u.d)**-1).to(u.microHertz),2)), stdTitle=False, phasebin=True, bins=25)
    LCtools.plt_any_lc_ax(lc_data, 2.0 * P1, is_Periodic=True, ax=ax7, title="f = {!s}$\mu$Hz $|$ 2".format(np.round(((P1 * u.d)**-1).to(u.microHertz),2)), stdTitle=False, phasebin=True, bins=25)
    LCtools.plt_any_lc_ax(lc_data, 3.0 * P1, is_Periodic=True, ax=ax8, title="f = {!s}$\mu$Hz $|$ 3".format(np.round(((P1 * u.d)**-1).to(u.microHertz),2)), stdTitle=False, phasebin=True, bins=25)

    LCtools.plt_any_lc_ax(lc_data, (1 / 3) * P2, is_Periodic=True, ax=ax9, title="f = {!s}$\mu$Hz $|$ 0.33".format(np.round(((P2 * u.d)**-1).to(u.microHertz),2)), stdTitle=False, phasebin=True, bins=25)
    LCtools.plt_any_lc_ax(lc_data, (1 / 2) * P2, is_Periodic=True, ax=ax10, title="f = {!s}$\mu$Hz $|$ 0.5".format(np.round(((P2 * u.d)**-1).to(u.microHertz),2)), stdTitle=False, phasebin=True, bins=25)
    LCtools.plt_any_lc_ax(lc_data, 1.0 * P2, is_Periodic=True, ax=ax11, title="f = {!s}$\mu$Hz $|$ ".format(np.round(((P2 * u.d)**-1).to(u.microHertz),2)), stdTitle=False, phasebin=True, bins=25)
    LCtools.plt_any_lc_ax(lc_data, 2.0 * P2, is_Periodic=True, ax=ax12, title="f = {!s}$\mu$Hz $|$ 2".format(np.round(((P2 * u.d)**-1).to(u.microHertz),2)), stdTitle=False, phasebin=True, bins=25)
    LCtools.plt_any_lc_ax(lc_data, 3.0 * P2, is_Periodic=True, ax=ax13, title="f = {!s}$\mu$Hz $|$ 3".format(np.round(((P2 * u.d)**-1).to(u.microHertz),2)), stdTitle=False, phasebin=True, bins=25)

    LCtools.plt_any_lc_ax(lc_data, (1 / 3) * P3, is_Periodic=True, ax=ax14, title="f = {!s}$\mu$Hz $|$ 0.33".format(np.round(((P3 * u.d)**-1).to(u.microHertz),2)), stdTitle=False, phasebin=True, bins=25)
    LCtools.plt_any_lc_ax(lc_data, (1 / 2) * P3, is_Periodic=True, ax=ax15, title="f = {!s}$\mu$Hz $|$ 0.5".format(np.round(((P3 * u.d)**-1).to(u.microHertz),2)), stdTitle=False, phasebin=True, bins=25)
    LCtools.plt_any_lc_ax(lc_data, 1.0 * P3, is_Periodic=True, ax=ax16, title="f = {!s}$\mu$Hz $|$ ".format(np.round(((P3 * u.d)**-1).to(u.microHertz),2)), stdTitle=False, phasebin=True, bins=25)
    LCtools.plt_any_lc_ax(lc_data, 2.0 * P3, is_Periodic=True, ax=ax17, title="f = {!s}$\mu$Hz $|$ 2".format(np.round(((P3 * u.d)**-1).to(u.microHertz),2)), stdTitle=False, phasebin=True, bins=25)
    LCtools.plt_any_lc_ax(lc_data, 3.0 * P3, is_Periodic=True, ax=ax18, title="f = {!s}$\mu$Hz $|$ 3".format(np.round(((P3 * u.d)**-1).to(u.microHertz),2)), stdTitle=False, phasebin=True, bins=25)

    ax3.set_title("")
    plt.tight_layout()
    return fig


def draw_plot(lc_data):
    fig = plt.figure()
    plt.scatter(lc_data['mjd'], lc_data['mag'])


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


def delete_figure_agg(figure_agg):
    figure_agg.get_tk_widget().forget()
    plt.close('all')


# -------------------------------- GUI Starts Here -------------------------------#
# fig = your figure you want to display.  Assumption is that 'fig' holds the      #
#       information to display.                                                   #
# --------------------------------------------------------------------------------#

fig = draw_plot_Pal2013(flc_data, P1, P2, P3, freq_grid, power, FAP_power_peak=FAP_power_peak, logFAP_limit=logFAP_limit, df=df, title=title)

# print(inspect.getsource(PyplotSimple))

sg.theme('Default')
figure_w, figure_h = 2048, 1280
# define the form layout
listbox_values = list(fig_dict)
col_listbox = [[sg.Listbox(values=listbox_values, change_submits=True, size=(28, len(listbox_values)), key='-LISTBOX-')],
               [sg.Text(' ' * 12), sg.Exit(size=(5, 2))]]

col_multiline = sg.Col([[sg.MLine(size=(70, 35), key='-MULTILINE-')]])
col_canvas = sg.Col([[sg.Canvas(size=(figure_w, figure_h), key='-CANVAS-')]])
col_instructions = sg.Col([[sg.Pane([col_canvas, col_multiline], size=(1536, 960))],
                           [sg.Text('Grab square above and slide upwards to view source code for graph')]])

# layout = [[sg.Text('Matplotlib Plot Test', font=('ANY 18'))],
#           [sg.Col(col_listbox), col_instructions], ]

layout = [[sg.Radio('My first Radio!     ', "P1", default=False)],
          [sg.Radio('My second Radio!     ', "P2", default=False)],
          [sg.Radio('My third Radio!     ', "P3", default=False)],
          [sg.Col(col_listbox), col_instructions]
          ]

# create the form and show it without the plot
window = sg.Window('TDSS Variable Star Visual Inspection - Period Aliasing',
                   layout, resizable=True, finalize=True)

canvas_elem = window['-CANVAS-']
multiline_elem = window['-MULTILINE-']
figure_agg = None

while True:
    event, values = window.read()
    if event in (sg.WIN_CLOSED, 'Exit'):
        break

    if figure_agg:
        # ** IMPORTANT ** Clean up previous drawing before drawing again
        delete_figure_agg(figure_agg)

    fig = draw_plot(lc_data)
    #fig = draw_plot_Pal2013(flc_data, P1, P2, P3, freq_grid, power, FAP_power_peak=FAP_power_peak, logFAP_limit=logFAP_limit, df=df, title=title)                                  # call function to get the figure
    figure_agg = draw_figure(window['-CANVAS-'].TKCanvas, fig)  # draw the figure
