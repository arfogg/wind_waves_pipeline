# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 15:14:25 2026

@author: A R Fogg
"""



class akr_burst():
    
    
    def __init__(self, index, stime, etime):

        self.burst_ID = index
        self.stime = stime
        self.etime = etime
        
    def record_freq_bands(self, timestamps, freq_width, freq_low, freq_high):
        self.burst_timestamp = timestamps # a list
        self.freq_width = freq_width # a list
        self.freq_low = freq_low # a list
        self.freq_high = freq_high  # a list