# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 15:14:25 2026

@author: A R Fogg
"""



class akr_burst():
    
    
    def __init__(self, index, stime):

        self.burst_ID = index
        self.stime = stime
        
    def record_end_time(self, etime):
        self.etime = etime
        
    # def record_freq_bands(self, freq_low, freq_high):
