# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 15:18:04 2026

@author: A R Fogg
"""

import os
import sys
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from numpy.lib.stride_tricks import sliding_window_view

from matplotlib.gridspec import GridSpec
#import pprint
# import matplotlib as mpl
# import scipy
#import sys
import calendar
# import string

#import plotting_spectrogram as pltsp
#import read_wind_waves_file
#import process_data 
# import wind_masked_akr_data
# import read_waters_masked_data
# import read_wind_position
# import read_fogg_burst_data
# import wind_raw_data
import pathlib
# import misc_wind_utility
# import run_integrate_intensity
# import read_integrated_power

import akr_burst_class
import spectrogram_plotter


sys.path.append(r'C:\Users\Alexandra\Documents\wind_waves_akr_code\wind_utility')
import read_waters_masked_data


data_dir = os.path.join(
     "C:" + os.sep, r'Users\Alexandra\Documents\data\fogg_burst_data\2026_refactor')


# Set up fontsizes
fontsize = 20
plt.rcParams['font.size'] = fontsize
plt.rcParams['axes.titlesize'] = fontsize
plt.rcParams['axes.labelsize'] = fontsize
plt.rcParams['xtick.labelsize'] = fontsize
plt.rcParams['ytick.labelsize'] = fontsize
plt.rcParams['legend.fontsize'] = fontsize


# Points to consider optimising
# 
# - start is first above threshold and end is first below. so start is
#   at the time of the burst, and end is just after. we should make them either
#   both on the above threshold (on the burst, preferable), or both on the
#   below threshold (around the burst)
# 
# - how to define frequency bins when the first time step doesn't meet our
#   criteria? perhaps we take from the next valid bin, i.e. extrapolate back
#
# - what about when no timestamps in the burst meet the packing density
#   threshold? so freq bounds are all NaNs
#
# - freq_width is centre of bin to centre of bin, should it be otherwise?



# run flux_tag all the way through

def test_search():

    akr_df = read_waters_masked_data.read_data_day(2002, 11, 1)

    # ID of the burst we want to freq diagnostic
    ID = 12
    # X axis start and end
    t_start, t_end = dt.datetime(2002,11,1,8), dt.datetime(2002,11,1,13)


    # Define constants
    threshold_number, time_gap, packing_density_threshold, min_length, min_f_bins = define_constants()

    akr_bursts, burst_stimes, burst_etimes, datetimes, n_filled, above_threshold_n = search(akr_df)
    
    burst_detection_diagnostic(akr_df, akr_bursts,
                               datetimes, n_filled, above_threshold_n,
                               threshold_number, time_gap,
                               packing_density_threshold, min_length,
                               min_f_bins, t_start, t_end)
    
    # Find the relevant burst
    akr_burst = next(b for b in akr_bursts if b.burst_ID == 12)
    fig_f, ax_f = plt.subplots()
    plot_freq_band_diagnostic(ax_f, akr_burst, akr_df)
    
    #breakpoint()
    return akr_df, akr_bursts, datetimes, n_filled, above_threshold_n,\
        threshold_number, time_gap, packing_density_threshold, min_length,\
        min_f_bins

def define_constants():
    
    # AKR burst definition thresholds and variables
    # Define threshold number of freq. bins that must be filled per time step
    threshold_number = 4 # below or equal to this is considered not enough

    # How long a gap in time will we allow in terms of meeting the threshold
    #   condition? Define this as 0 for now, so we can play with the number later
    time_gap = 2
   
    # Packing density threshold, used to determine upper and lower
    #   frequency boundaries
    packing_density_threshold = 80
    
    # Minimum number of bins filled after the first detection
    min_length = 3
    
    # Minimum number of frequency bins filled to define frequency bounds
    min_f_bins = 2

    return threshold_number, time_gap, packing_density_threshold, min_length, min_f_bins

def search(akr_df):
    
    code_stime = dt.datetime.now()
    
    # Define constants
    threshold_number, time_gap, packing_density_threshold, min_length, min_f_bins = define_constants()

    # Count the number of filled bins as a function of time
    datetimes, n_filled, above_threshold_n = count_filled_bins(akr_df, threshold_number)

    # Convert timestamps to unix time
    akr_df['unix'] = [pd.Timestamp(t).timestamp() for t in akr_df.datetime_ut]
    unix = [pd.Timestamp(t).timestamp() for t in datetimes]

    # Find starts and ends
    burst_stimes, burst_etimes = find_starts_and_ends(datetimes,
                                                        n_filled,
                                                        above_threshold_n)

    # Build burst classes and identify frequency bands
    akr_bursts = build_burst_objects(akr_df, burst_stimes, burst_etimes)


    code_etime = dt.datetime.now()
     
    print('Code finished, time elapsed: ', code_etime - code_stime)

    print('WARNING: NEED TO IMPLEMENT SMALL BURST SEARCH AND STICKER')
    print('WARNING: NEED TO IMPLEMENT CREATION OF BURST-MASKED DATA, including putting it into akr_burst class')
    return akr_bursts, burst_stimes, burst_etimes, datetimes, n_filled, above_threshold_n



def find_starts_and_ends(datetimes, n_filled, above_threshold_n):

    threshold_number, time_gap, packing_density_threshold, min_length, min_f_bins = define_constants()

    # Find Burst Starts ------------------------
    # How many time steps
    N = len(above_threshold_n)

    # Check if current bin is filled
    current = above_threshold_n[1:N-min_length]
    # Check the previous bin is empty
    previous = ~above_threshold_n[:N-(min_length+1)]

    # Check the future bins are filled up to min_length bins afterwards
    future = np.logical_and.reduce([
        above_threshold_n[1+i : N-(min_length-i)]
        for i in range(min_length)
    ])

    # Check all conditions are met
    starts = current & previous & future
    start_idx = np.where(starts)[0] + 1

    # burst_stimes_2 = datetimes[start_idx]
    
    
    #breakpoint()
    # Find Burst Ends ------------------------
    # Reverse the boolean: True is now below the threshold
    below_threshold_n = ~above_threshold_n
    windows = sliding_window_view(below_threshold_n, time_gap)
    quiet_runs = windows.all(axis=1)
    end_candidates = np.where(quiet_runs)[0]
    


    # Match Up Burst Starts and Ends ------------------------
    burst_stimes = datetimes[start_idx]
    burst_etimes = []

    for i in start_idx:
        future_ends = end_candidates[end_candidates > i]
        if future_ends.size == 0:
            continue
        burst_etimes.append(datetimes[future_ends[0]])

    # t4 = dt.datetime.now()    
    # print('new method: ', t4-t3)
    
    return burst_stimes, burst_etimes


def burst_detection_diagnostic(akr_df, akr_bursts,
                                datetimes, n_filled, above_threshold_n,
                                threshold_number, time_gap,
                                packing_density_threshold, min_length,
                                min_f_bins, t_start, t_end):
    # PLOT A DIAGNOSTIC
    fig, ax = plt.subplots(nrows=4, figsize=(10, 12))

    # N filled
    ax[0].plot(datetimes, n_filled, linewidth=1., color='grey')
    ind_gt, = np.where(n_filled >= threshold_number)
    ind_lt, = np.where(n_filled < threshold_number)
    ax[0].plot(datetimes[ind_gt], n_filled[ind_gt], linewidth=0,
               marker='o', fillstyle='none', color='teal',
               label='n $\geq$ '+str(threshold_number))
    ax[0].plot(datetimes[ind_lt], n_filled[ind_lt], linewidth=0,
               marker='o', fillstyle='none', color='orange',
               label='n < '+str(threshold_number))
    ax[0].axhline(threshold_number, linewidth=1.5, color='grey',
                  linestyle='dashed')
    ax[0].legend()

    # True/False timeseries
    ind_t, = np.where(above_threshold_n == True)
    ind_f, = np.where(above_threshold_n == False)
    ax[1].plot(datetimes[ind_t], above_threshold_n[ind_t], linewidth=0.,
               marker='^', color='teal', label='True')
    ax[1].plot(datetimes[ind_f], above_threshold_n[ind_f], linewidth=0.,
               marker='^', color='orange', label='False')
    ax[1].legend()

    # Spectrogram of Waters Mask, with polygon
    ax[2], x_arr, y_arr, z_arr = spectrogram_plotter.return_spectrogram(
        akr_df, ax[2], cmap='gray')
    # Indicate freq low and high
    for b in akr_bursts:
        ax[2].plot(b.burst_timestamp, b.freq_low, color='orange')
        ax[2].plot(b.burst_timestamp, b.freq_high, color='orange')

    # Adjust axis width for non-spectrogram axes
    pos_spec = ax[2].get_position().bounds
    for a in [ax[0], ax[1]]:
        pos = a.get_position().bounds
        a.set_position([pos_spec[0], pos[1], pos_spec[2], pos[3]])

    # Formatting axes
    for a in ax:
        # X limits
        a.set_xlim(t_start, t_end)
        # Indicate starts and ends
        for b in akr_bursts:
            a.axvline(b.stime, linestyle='dashed', color='purple')
            a.axvline(b.etime, linestyle='dotted', color='purple')
            
    # Frequency band diagnostic
    
    
    
    #fig_f, ax_f = plt.subplots()
    #plot_freq_band_diagnostic(freqs, flux, f_low, f_high, ax)

def plot_freq_band_diagnostic(akr_burst, akr_df,
                              timestamp_s_index=5, n_examples=4,
                              spec_time_buffer=pd.Timedelta(minutes=10),
                              time_label=None, flux_tag='akr_flux_si_1au',
                              cmap='gray'):
    """
    Diagnostic plot showing the selected frequency band.

    Parameters
    ----------

    """

    
    # Initialise figure
    fig_f = plt.figure(layout="constrained", figsize=(10, 12))
    gs = GridSpec(4, 1, figure=fig_f)
    ax_spec = fig_f.add_subplot(gs[0, :])
    ax_diag = fig_f.add_subplot(gs[1:, :])

    # Spectrogram of Waters Mask, with polygon
    ax_spec, x_arr, y_arr, z_arr = spectrogram_plotter.return_spectrogram(
        akr_df, ax_spec, cmap='gray')    
    ax_spec.set_xlim(akr_burst.stime - spec_time_buffer,
                     akr_burst.etime + spec_time_buffer)

    # Indicate burst detection
    ax_spec.axvline(akr_burst.stime, color='purple', linestyle='dashed')
    ax_spec.axvline(akr_burst.etime, color='purple', linestyle='dotted')
    ax_spec.plot(akr_burst.burst_timestamp, akr_burst.freq_low, color='orange')
    ax_spec.plot(akr_burst.burst_timestamp, akr_burst.freq_high, color='orange')

    # Indicate where the zoom in is
    ax_spec.axvline(akr_burst.burst_timestamp[timestamp_s_index], color='blue')
    ax_spec.axvline(akr_burst.burst_timestamp[timestamp_s_index + n_examples-1], color='blue')

    # Select spectrogram data for this burst
    burst_spec_df = akr_df.loc[(akr_df.datetime_ut >= akr_burst.stime) &
                               (akr_df.datetime_ut <= akr_burst.etime)]

    x_tick_pos = []
    x_tick_n = []
    for i, t in enumerate(np.array(
            akr_burst.burst_timestamp)[
                0 + timestamp_s_index:n_examples + timestamp_s_index]):

        demo_df = burst_spec_df.loc[(burst_spec_df.datetime_ut == t)]
        flux = demo_df[flux_tag].to_numpy()
        freqs = demo_df['freq'].to_numpy()
        f_low = akr_burst.freq_low[i + timestamp_s_index]
        f_high = akr_burst.freq_high[i + timestamp_s_index]

        # Define spectrogram arrays
        y = np.full(freqs.size + 1, np.nan)
        # Run through calculating bin edges, skipping the first and last bands
        # using midpoint definition x[i]+x[i+1])/2 
        for j in range(1, freqs.size):
            y[j] = (freqs[j-1] + freqs[j])/2
        # Calculate the edges for the first and last bins    
        y[0] = freqs[0] - ( y[1]-freqs[0] )
        y[-1] = freqs[-1] + (freqs[-1]-y[-2])

        # Create x positions
        real_dtime = demo_df.datetime_ut.unique()[0]
        dtime = 11.5 + (i * 10)
        dtime_edges = np.array([int(11 + (i * 10)), int(12 + (i * 10))])
        x_tick_pos.append(dtime)
        x_tick_n.append(real_dtime.strftime("%H:%M"))

        # Create meshgrid which defines the edges of the shapes
        x_arr, y_arr = np.meshgrid(dtime_edges, y)
        # Convert columns to NumPy arrays once
        times = np.full(len(demo_df), dtime)
        freqs_data = demo_df['freq'].to_numpy()
        flux = demo_df[flux_tag].to_numpy()

        # Find which time bin and freq bin each data point belongs to
        time_bin = np.searchsorted(dtime_edges, times) - 1
        freq_bin = np.searchsorted(y, freqs_data) - 1
        #breakpoint()
        # Create z array
        z_arr = np.full((len(y)-1, len(dtime_edges)-1), np.nan)
    
        # Fill the array
        valid = (
            (time_bin >= 0) & (time_bin < z_arr.shape[1]) &
            (freq_bin >= 0) & (freq_bin < z_arr.shape[0])
        )
        z_arr[freq_bin[valid], time_bin[valid]] = flux[valid]

        # Create image, normalising color on logscale
        im = ax_diag.pcolormesh(x_arr, y_arr, z_arr, cmap=cmap,
                                norm=mpl.colors.LogNorm(), edgecolors='blue')
        # Set y axis as log space
        ax_diag.set_yscale('log')
        
        # Indicate successful freq bins
        ax_diag.plot([dtime + 1., dtime + 1.], [f_low, f_high],
                     marker='x', color='red')
        ax_diag.text(dtime + 2., f_high,
                     f'{f_low:.1f}–\n{f_high:.1f} kHz',
                     ha='left', va='center', color='red', fontsize=15)

        # # Enable colour bar
        # if no_cbar == False:
        #     cbar = plt.colorbar(im_m, ax=ax, label='S (W m$^{-2}$ Hz$^{-1}$)')
        #     cbar.ax.yaxis.set_major_locator(mpl.ticker.LogLocator())
        #     cbar.ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    
        # # Labels and ticks
        # ax.set_ylabel('Frequency (kHz)')
        # ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%H%M"))


        # valid = ~np.isnan(flux)
        # empty = np.isnan(flux)
        # # # Fill empty bins with a value so they plot
        # # flux[np.isnan(flux)] = 100
        # # fig, ax = plt.subplots(figsize=(6, 4))
    
        # ax.plot(freqs[valid], np.full(freqs[valid].size, 1.), color='purple',
        #         marker='*', markersize=20, fillstyle='none', linewidth=0.,
        #         label='valid bins')
        # ax.plot(freqs[empty], np.full(freqs[empty].size, 1.), color='lightgray',
        #         marker='o', markersize=20, fillstyle='none', linewidth=0,
        #         label='empty bins')
    
        # # Shade selected band
        # if not np.isnan(f_low):
        #     plt.axvspan(f_low, f_high, color='orange', alpha=0.3,
        #                 label=f'Selected band ({f_low:.1f}–{f_high:.1f} kHz)')
    
        # plt.xlabel("Frequency (kHz)")
        # plt.ylabel("Flux")
        # title = "Frequency Band Detection"
        # if time_label is not None:
        #     title += f"\n{time_label}"
        # plt.title(title)
    
    ax_diag.set_yscale('log')
    ax_diag.set_xlim(0, 51)
    ax_diag.set_xticks(x_tick_pos, x_tick_n)


def plot_freq_band_diagnostic_old(#freqs, flux, f_low, f_high,
                              ax, akr_burst, akr_df,
                              time_label=None):
    """
    Diagnostic plot showing the selected frequency band.

    Parameters
    ----------
    freqs : np.ndarray
        Frequency array (sorted)
    flux : np.ndarray
        Flux array (NaNs allowed)
    f_low, f_high : float
        Selected frequency band limits
    time_label : str or datetime, optional
        Label for plot title
    """
    
    # spec_df = akr_df.loc[(akr_df.datetime_ut >= akr_burst.stime) &
    #                      (akr_df.datetime_ut <= akr_burst.etime)]

    spec_df = akr_df.loc[(akr_df.datetime_ut == akr_burst.stime)]
    flux = spec_df['akr_flux_si_1au'].to_numpy()
    freqs = spec_df['freq'].to_numpy()
    f_low = akr_burst.freq_low[0]
    f_high = akr_burst.freq_high[0]


    valid = ~np.isnan(flux)
    empty = np.isnan(flux)
    # # Fill empty bins with a value so they plot
    # flux[np.isnan(flux)] = 100
    # fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(freqs[valid], np.full(freqs[valid].size, 1.), color='purple',
            marker='*', markersize=20, fillstyle='none', linewidth=0.,
            label='valid bins')
    ax.plot(freqs[empty], np.full(freqs[empty].size, 1.), color='lightgray',
            marker='o', markersize=20, fillstyle='none', linewidth=0,
            label='empty bins')

    # # Plot all bins
    # ax.plot(freqs, flux, color='magenta', linewidth=0., marker="*", markersize=20, fillstyle='none', label='All bins')

    # # Highlight valid bins
    # ax.plot(freqs[valid], flux[valid], color='black', linewidth=0., marker='o', markersize=20, fillstyle='none', label='Valid bins')

    # Shade selected band
    if not np.isnan(f_low):
        plt.axvspan(f_low, f_high, color='orange', alpha=0.3,
                    label=f'Selected band ({f_low:.1f}–{f_high:.1f} kHz)')

    plt.xlabel("Frequency (kHz)")
    plt.ylabel("Flux")
    title = "Frequency Band Detection"
    if time_label is not None:
        title += f"\n{time_label}"
    plt.title(title)

    ax.set_xscale('log')


    plt.legend()
    plt.tight_layout()
    plt.show()




def extract_freq_bands_for_burst(akr_df, akr_burst, flux_tag="akr_flux_si_1au",
                                 packing_threshold=80.0, min_f_bins=2):

    # Select only this burst's data
    df = akr_df[
        (akr_df["datetime_ut"] >= akr_burst.stime) &
        (akr_df["datetime_ut"] <= akr_burst.etime)
    ]

    # Group by time
    times, freq_width, f_low, f_high = [], [], [], []
    for t, g in df.groupby("datetime_ut"):
        g = g.sort_values("freq")
        freqs = g["freq"].to_numpy()
        flux  = g[flux_tag].to_numpy()

        # find freq bounds for this time step
        w, (l, h) = find_freq_bound(
            freqs, flux,
            packing_threshold=packing_threshold,
            min_f_bins=min_f_bins
        )
        times.append(t)
        freq_width.append(w)
        f_low.append(l)
        f_high.append(h)

    akr_burst.record_freq_bands(times, freq_width, f_low, f_high)

def build_burst_objects(akr_df, burst_stimes, burst_etimes):

    bursts = []
    _, _, packing_density_threshold, _ , min_f_bins = define_constants()

    for i, (s, e) in enumerate(zip(burst_stimes, burst_etimes)):
        b = akr_burst_class.akr_burst(i, s, e)

        extract_freq_bands_for_burst(
            akr_df, b,
            packing_threshold=packing_density_threshold,
            min_f_bins=min_f_bins
        )

        bursts.append(b)

    return bursts


def find_freq_bound(freqs, flux, packing_threshold, min_f_bins=2):
    """
    Find the widest frequency band with non-NaN flux in packing_threshold
    percentage bins.

    Parameters
    ----------
    freqs : (N,) array
        Sorted frequencies (low to high)
    flux : (N,) array
        Flux values (NaNs allowed)
    packing_threshold : float
        Required % of valid bins
    min_bins : int, optional
        Minimum number of bins in a band

    Returns
    -------
    best_width : float
    f_low, f_high : float, float
    """

    N = len(freqs)
    # If we have less than min_bins frequency bands, return NaN
    if N < min_f_bins:
        return np.nan, np.nan

    # Identify bins with flux in
    valid = ~np.isnan(flux)

    # Initialise variables
    best_width = 0
    best_band = (np.nan, np.nan)

    # Sliding window sizes (i.e. w is band width in number of bins)
    for w in range(min_f_bins, N + 1):
        # Counts how many valid bins are inside every possible band of width w
        counts = np.convolve(valid.astype(int), np.ones(w, dtype=int), 'valid')
        packing = 100 * (counts / w)

        # Find starting indices of bands where packing threshold is met
        good = np.where(packing >= packing_threshold)[0]
        # Move on to next w
        if good.size == 0:
            continue

        # Find band that's widest in frequency
        widths = freqs[good + w - 1] - freqs[good]
        i = widths.argmax()

        # If better than what we've found already, store
        if widths[i] > best_width:
            best_width = widths[i]
            best_band = (freqs[good[i]], freqs[good[i] + w - 1])

    return best_width, best_band


def count_filled_bins(akr_df, threshold_number, ofile='_TEST', flux_tag='akr_flux_si_1au'):

     
    n_filled_csv = os.path.join(data_dir,'n_filled_summary' + ofile + '.csv')

    if (pathlib.Path(n_filled_csv).is_file()):
        print('Loading pre read n_filled data')
        df = pd.read_csv(n_filled_csv, delimiter=',',
                         parse_dates=['datetimes'],
                         float_precision='round_trip')
        datetimes = np.array(df.datetimes)
        n_filled = np.array(df.n_filled)
        above_threshold_n = np.array(df.above_threshold_n)
    else:

        t_s = dt.datetime.now()
        # Group data by datetime (i.e. individual sweeps)
        grouped = akr_df.groupby("datetime_ut")[flux_tag]
     
        # Count number of filled bins
        n_filled = grouped.apply(lambda x: x.notna().sum()).to_numpy()
        # Store the unique datetimes
        datetimes = grouped.size().index.to_numpy()

        # Assess if above threshold
        above_threshold_n = n_filled >= threshold_number

        print('Counting of filled bins complete, time elapsed: ',
              dt.datetime.now() - t_s)
        
        print('Saving n_filled summary to: ', n_filled_csv)
        n_df = pd.DataFrame({'datetimes':datetimes, 'n_filled':n_filled,
                             'above_threshold_n': above_threshold_n})
        n_df.to_csv(n_filled_csv, index=False)

    return datetimes, n_filled, above_threshold_n

