# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 09:51:25 2026

@author: A R Fogg
"""

import numpy as np
# import pandas as pd
import datetime as dt
import matplotlib as mpl

import matplotlib.pyplot as plt




def return_spectrogram(akr_df, ax, no_cbar=False,
                       flux_tag='akr_flux_si_1au', cmap='viridis'):

    # Create and return a masked AKR spectrogram, given an input akr_df as returned by read_data()

    # Find unique frequency channels
    # (aka the freq bin centres, which are unevenly spaced)
    freqs = np.sort(akr_df['freq'].dropna().unique())
    
    # Define the frequency band edges by finding the midpoint between bins
    y = np.full(freqs.size+1, np.nan)

    # Run through calculating bin edges, skipping the first and last bands
    # using midpoint definition x[i]+x[i+1])/2 
    for i in range(1, freqs.size):
        y[i] = (freqs[i-1]+freqs[i])/2
    # Calculate the edges for the first and last bins    
    y[0] = freqs[0] - ( y[1]-freqs[0] )
    y[-1] = freqs[-1] + (freqs[-1]-y[-2])

    # Find unique times / time indexes (aka the time bin centres)
    dtimes = np.sort(akr_df['datetime_ut'].dropna().unique())

    # Data are not exactly 3 minute resolution, so we can't do it this simply
    #   we need to find the midpoint between each obs?
    # ??? needs updating
    dtime_edges = np.append(dtimes-np.timedelta64(90,'s'),
                            dtimes[-1]+np.timedelta64(90,'s'))

    # Create meshgrid which defines the edges of the shapes to be shaded
    # in sweep index vs frequency space
    x_arr, y_arr = np.meshgrid(dtime_edges,y)

    # Convert columns to NumPy arrays once
    times = akr_df['datetime_ut'].to_numpy()
    freqs_data = akr_df['freq'].to_numpy()
    flux = akr_df[flux_tag].to_numpy()

    # Find which time bin and freq bin each data point belongs to
    time_bin = np.searchsorted(dtime_edges, times) - 1
    freq_bin = np.searchsorted(y, freqs_data) - 1

    # Create z array
    z_arr = np.full((len(y)-1, len(dtime_edges)-1), np.nan)

    # Fill the array
    valid = (
        (time_bin >= 0) & (time_bin < z_arr.shape[1]) &
        (freq_bin >= 0) & (freq_bin < z_arr.shape[0])
    )
    z_arr[freq_bin[valid], time_bin[valid]] = flux[valid]

    # Create image, normalising color on logscale
    im_m = ax.pcolormesh(x_arr, y_arr, z_arr, cmap=cmap,
                         norm=mpl.colors.LogNorm())
    # Set y axis as log space
    ax.set_yscale('log')

    # Enable colour bar
    if no_cbar == False:
        cbar = plt.colorbar(im_m, ax=ax, label='S (W m$^{-2}$ Hz$^{-1}$)')
        cbar.ax.yaxis.set_major_locator(mpl.ticker.LogLocator())
        cbar.ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())

    # Labels and ticks
    ax.set_ylabel('Frequency (kHz)')
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%H%M"))

    return ax, x_arr, y_arr, z_arr
