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


sys.path.append(r'C:\Users\Alexandra\Documents\wind_waves_akr_code\wind_utility')
import read_waters_masked_data


data_dir = os.path.join(
     "C:" + os.sep, r'Users\Alexandra\Documents\data\fogg_burst_data\2026_refactor')


# Points to consider optimising
# - start is first above threshold and end is first below. so start is
#   at the time of the burst, and end is just after. we should make them either
#   both on the above threshold (on the burst, preferable), or both on the
#   below threshold (around the burst)



def test_search():

    akr_df = read_waters_masked_data.read_data_day(2002, 11, 1)

    search(akr_df)
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

    return threshold_number, time_gap, packing_density_threshold, min_length

def search(akr_df):
    
    code_stime = dt.datetime.now()
    
    # Define constants
    threshold_number, time_gap, packing_density_threshold, min_length = define_constants()

    # Count the number of filled bins as a function of time
    datetimes, n_filled, above_threshold_n = count_filled_bins(akr_df, threshold_number)

    # Convert timestamps to unix time
    akr_df['unix'] = [pd.Timestamp(t).timestamp() for t in akr_df.datetime_ut]
    unix = [pd.Timestamp(t).timestamp() for t in datetimes]

    # Find starts and ends
    burst_stimes, burst_etimes = find_starts_and_ends(datetimes,
                                                        n_filled,
                                                        above_threshold_n)


    # PLOT TEST
    fig, ax = plt.subplots(nrows=2, figsize=(8, 6))
    
    ax[0].plot(datetimes, n_filled, linewidth=1., color='grey')
    ind_gt, = np.where(n_filled >= threshold_number)
    ind_lt, = np.where(n_filled < threshold_number)
    ax[0].plot(datetimes[ind_gt], n_filled[ind_gt], linewidth=0,
               marker='o', fillstyle='none', color='teal')
    ax[0].plot(datetimes[ind_lt], n_filled[ind_lt], linewidth=0,
               marker='o', fillstyle='none', color='orange')
    ax[0].axhline(threshold_number, linewidth=1.5, color='grey',
                  linestyle='dashed')

    #breakpoint()
    ind_t, = np.where(above_threshold_n == True)
    ind_f, = np.where(above_threshold_n == False)
    ax[1].plot(datetimes[ind_t], above_threshold_n[ind_t], linewidth=0.,
               marker='^', color='teal')
    ax[1].plot(datetimes[ind_f], above_threshold_n[ind_f], linewidth=0.,
               marker='^', color='orange')

    for s in burst_stimes:
        ax[0].axvline(s, linestyle='dashed', color='green')
        ax[1].axvline(s, linestyle='dashed', color='green')
    for e in burst_etimes:
        ax[0].axvline(e, linestyle='dashed', color='red')
        ax[1].axvline(e, linestyle='dashed', color='red')

    #breakpoint()


    for a in ax:
        a.set_xlim(dt.datetime(2002,11,1,8), dt.datetime(2002,11,1,13))

    code_etime = dt.datetime.now()
     
    print('Code finished, time elapsed: ', code_etime - code_stime)


    return burst_stimes, burst_etimes



def find_starts_and_ends(datetimes, n_filled, above_threshold_n):

    # t1 = dt.datetime.now()
    # burst_stimes = []
    # burst_etimes = []
    # threshold_number, time_gap, packing_density_threshold, min_length = define_constants()
    # for i in range(1,n_filled.size-3):
    #     if n_filled[i] > threshold_number :
    #     #             # Yes
    #     #             # Is the previous bin also filled? - this needs to be robust for dealing with index 0!
    #     #             # Are we before a previously defined end time?
    #         if (n_filled[i-1] <= threshold_number) & (n_filled[i+1] > threshold_number) & (n_filled[i+2] > threshold_number) & (n_filled[i+3] > threshold_number) & ( all(datetimes[i] > a for a in burst_etimes) == True):
        
    #             #print('AKR Burst start time identified as: ',datetimes[i])
    #             burst_stimes.append(datetimes[i])
    
                
    #             # print('Looking for AKR Burst end time...')
    #             # Reset end_found
    #             end_found=False
    #             for j in range(i+1, n_filled.size):
    #                 # Looking for n_filled to drop below threshold
    #                 if n_filled[j] <= threshold_number:
    #                     #print('dropped below - first time | n is',n_filled[j], '| time is ', datetimes[j], '| int is ', j )
    #                     # Look to see if we pop back above the threshold again within the defined time gap?
    #                     for k in range(j,j+time_gap):
    #                         if n_filled[k] > threshold_number:
    #                             #print('popped back above',n_filled[k])
    #                             # We do pop above the threshold again at time k
    #                             # Keep looking until we drop below the threshold, and don't pop up again
    #                             #   within the allowed time gap
                                
    #                             # scanning after we have popped back up
    #                             l=k+1
    #                             while l in range(k+1, n_filled.size):
          
                                           
    #                                 if n_filled[l] <= threshold_number:
    #                                     # We are below the threshold again
                                       
    #                                     above_found=0
    #                                     for m in range(l+1, l+time_gap):
    #                                         #print('inside for, m is ', m, ' | n is ', n_filled[m], '| time is ', datetimes[m])
    #                                         # Do we pop back above the threshold again?
    #                                         if n_filled[m] > threshold_number:
    #                                             # Then continue to search in the m for loop another drop below
    #                                             #print('still above',n_filled[m])
    #                                             above_found=above_found+1
           
    #                                         if above_found == 0:
    #                                             end_found=True
                                                       
    #                                         if end_found==True:
    #                                             print('AKR Burst end time: ',datetimes[l])
    #                                             burst_etimes.append(datetimes[l])
    #                                             break  # Break out of the while loop if we found the end time    
    #                                         # If we got through all l:l+time_gap then skip to l+time_gap+1 and continue looking
    #                                         l=l+time_gap+1
                                                    
    #                                     else:
    #                                         l=l+1
        
    #                         # If we don't find anytime above threshold within the time gap, record j
    #                         if end_found==False:
    #                             # Burst ends at index j
    #                             print('AKR Burst end time: ',datetimes[j])
    #                             burst_etimes.append(datetimes[j])
                                
    #                         # Break out of end time search for loop (j)
    #                         break
    #                     else:
    #                         # In case we are out of time intervals to find a burst, strip off the burst
    #                         #   start time we found
    #                         if j == n_filled.size-1: 
    #                             print('Reach the end of the times given, so no burst end found')
    #                             print('Deleting burst start with undefined end')
    #                             burst_stimes=burst_stimes[0:-1]
      
            
            
    # t2 = dt.datetime.now()
    # print('original method: ', t2-t1)

    # t3 = dt.datetime.now()
    # starts = (
    #     # Not looking at the first or final three bins
    #     above_threshold_n[1:-3] &
    #     # Previous bin is False
    #     ~above_threshold_n[:-4] &
    #     # Next bin is True
    #     above_threshold_n[2:-2] &
    #     # Next bin is True
    #     above_threshold_n[3:-1] &
    #     # Next bin is True
    #     above_threshold_n[4:]
    # )
    # start_idx = np.where(starts)[0] + 1

    threshold_number, time_gap, packing_density_threshold, min_length = define_constants()

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

