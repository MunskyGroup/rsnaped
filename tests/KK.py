#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 09:51:53 2021

@author: luisub
"""
if 0==0:
    # System libraries
    import io
    import sys
    import os
    #import statistics
    from statistics import median_low
    import random
    import math
    from math import nan
    # To manipulate arrays
    import numpy as np  
    from numpy import unravel_index
    # For data frames
    import pandas as pd   
    # Ignoring warnings
    import warnings
    warnings.filterwarnings('ignore')
    # Skimage
    from skimage import img_as_float64, img_as_uint
    from skimage.filters.rank import entropy
    from skimage.morphology import disk
    from skimage.filters import threshold_minimum
    from skimage.morphology import binary_closing
    from skimage.measure import find_contours
    from skimage.draw import polygon2mask
    from skimage.draw import polygon
    from skimage.util import random_noise
    from skimage.transform import warp
    from skimage import transform
    from skimage.filters import difference_of_gaussians
    from skimage.filters import gaussian
    from skimage.draw import polygon_perimeter
    # Open cv
    import cv2
    # Parallel computing
    from joblib import Parallel, delayed
    import multiprocessing
    from tqdm import tqdm
    # Particle tracking
    import trackpy as tp 
    tp.quiet()  # Turn off progress reports for best performance
    # To create interactive elements
    import ipywidgets as widgets 
    from ipywidgets import interactive, HBox, Layout, GridspecLayout #, interact, fixed, interact_manual, Button, VBox
    import bqplot as bq
    from bqplot import LinearScale, ColorScale, HeatMap
    # Scipy 
    import scipy.stats as sps
    from scipy.signal import find_peaks #, peak_prominences, find_peaks_cwt
    from scipy.spatial import Delaunay
    from scipy.optimize import curve_fit
    # To read .tiff files
    import tifffile
    # to create gifs
    import imageio
    # time libraries
    from timeit import default_timer as timer
    import time
    # Cellpose
    from cellpose import models 
    # Plotting
    import matplotlib.pyplot as plt 
    import matplotlib.path as mpltPath
    from matplotlib import gridspec
    plt.style.use("dark_background")

# Conventions.
#number_timepoints
# number_channels
# index_channels
# index_time





class BandpassFilter(): 
    '''
    This class is inteded to apply high and low bandpass filters to the video. The format of the video must be [T,Y,X,C].
    
    Parameters
    ----------
    video : numpy array 
        array of images.
    min_percentile : int, optional
        Lower bound to normalize intensity. The default is 1.
    max_percentile : TYPE, optional
        Higher bound to normalize intensity. The default is 99.
    ignore_channel : int or None, optional
        Use this option to ignore the normalization of a given channel. The default is None.
    '''
    def __init__(self,video, low_pass = 0.5, high_pass = 10):
        # Making the values for the filters are odd numbers
        if (low_pass % 2) == 0:
            low_pass = low_pass+1
        if (high_pass % 2) == 0:
            high_pass = high_pass+1
        self.video = video
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.NUMBER_OF_CORES = multiprocessing.cpu_count()
                 
    def apply_filter(self):
        '''
        This method applies high and low bandpass filters to the video. The method uses **difference_of_gaussians** from skimage.filters.
        
        Returns
        -------
        video_filtered : np.uint16
            Filtered video resulting from the bandpass process. Array with format [T,Y,X,C].
        '''
        
        video_bp_filtered_float = np.zeros_like(self.video,dtype=np.float64) 
        number_timepoints, number_channels   = self.video.shape[0], self.video.shape[3]  
        for index_channels in range(0,number_channels):
            for index_time in range(0,number_timepoints):
                video_bp_filtered_float[index_time,:,:,index_channels]= difference_of_gaussians(self.video[index_time,:,:,index_channels], self.low_pass, self.high_pass)
        
        # temporal function that converts floats to uint
        def img_uint(image):
            temp_vid= img_as_uint(image)
            return temp_vid
        # returning the image normalized as uint. Notice that difference_of_gaussians converts the image into float.
        for index_channels in range(0,number_channels):
            init_video = Parallel(n_jobs=self.NUMBER_OF_CORES)(delayed(img_uint)(video_bp_filtered_float[i,:,:,index_channels]) for i in range(0,number_timepoints)) 
            video_filtered = np.asarray(init_video)
        
        return video_filtered



class RemoveExtrema():
    '''
    This class is inteded to remove extreme values from a video. The format of the video must be [T,Y,X,C].
    
    Parameters
    ----------
    video : numpy array 
        array of images.
    min_percentile : int, optional
        Lower bound to normalize intensity. The default is 1.
    max_percentile : TYPE, optional
        Higher bound to normalize intensity. The default is 99.
    ignore_channel : int or None, optional
        Use this option to ignore the normalization of a given channel. The default is None.
    '''
    def __init__(self, video, min_percentile=1, max_percentile=99,ignore_channel =None):
        self.video = video
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile
        self.ignore_channel =ignore_channel
        
    def remove_outliers(self):
        '''
        This method normalizes the values of a video by removing extreme values.
        
        Returns
        -------
        normalized_video : np.uint16
            Normalized video.
        '''
        normalized_video = np.copy(self.video, dtype=np.uint16)
        number_timepoints, _,_,number_channels = normalized_video.shape
        time_points, number_channels   = self.video.shape[0], self.video.shape[3]
        for index_channels in range (number_channels):
            if not self.ignore_channel ==index_channels:
                for index_time in range (number_timepoints):
                    normalized_video_temp = normalized_video[index_time,:,:,index_channels]
                    if not np.amax(normalized_video_temp) ==0: # this section detect that the channel is not empty to perform the normalization.
                        max_val = np.percentile(normalized_video_temp, self.max_percentile)
                        min_val = np.percentile(normalized_video_temp, self.min_percentile)
                        normalized_video_temp [normalized_video_temp> max_val] = max_val
                        normalized_video_temp [normalized_video_temp< min_val] = min_val
                        normalized_video_temp [normalized_video_temp<0]=0
        return normalized_video