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
import matplotlib.pyplot as plt

from numpy.lib.stride_tricks import sliding_window_view
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

    burst_stimes, burst_etimes = search(akr_df)
    breakpoint()

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
    # Indicate starts and ends
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
        a.set_xlim(dt.datetime(2002,11,1,8), dt.datetime(2002,11,1,13))
        # Indicate starts and ends
        for s in burst_stimes:
            a.axvline(s, linestyle='dashed', color='purple')
        for e in burst_etimes:
            a.axvline(e, linestyle='dotted', color='purple')

    code_etime = dt.datetime.now()
     
    print('Code finished, time elapsed: ', code_etime - code_stime)

    print('WARNING: NEED TO IMPLEMENT SMALL BURST SEARCH AND STICKER')
    print('WARNING: NEED TO IMPLEMENT CREATION OF BURST-MASKED DATA')
    return burst_stimes, burst_etimes



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


def plot_freq_band_diagnostic(freqs, flux, f_low, f_high, time_label=None):
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

    valid = ~np.isnan(flux)
    empty = np.isnan(flux)
    # # Fill empty bins with a value so they plot
    # flux[np.isnan(flux)] = 100
    fig, ax = plt.subplots(figsize=(6, 4))

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

