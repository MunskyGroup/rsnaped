# -*- coding: utf-8 -*-
'''
**********
rSNAPed 
**********
'''

'''
A software for single-molecule image tracking, simulation and parameter estimation.
'''

# https://eikonomega.medium.com/getting-started-with-sphinx-autodoc-part-1-2cebbbca5365


#module_name, package_name, ClassName, method_name, 
# ExceptionName, function_name, GLOBAL_CONSTANT_NAME, 
# global_var_name, instance_var_name, function_parameter_name, local_var_name.

# Conventions.
#number_timepoints
# number_channels
# index_channels
# index_time


'''
**********
Libraries
**********
'''
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


'''
**********
Clases
**********
'''


class ConverToStandardFormat():
    '''
    This class contains two methods to:
    
    1. Transform any numpy array of images into the format [T,Y,X,C]. 
    2. Convert an image into an array video with a single time point (this last is necessary for compatibility).
    
    Parameters
    ----------
    video : numpy array 
        Array of images. This class accepts arrays with formats: [Y,X], [T,Y,X], [T,Y,X,C], or anyother permutation of channels, the user must specify the position of each dimension in the original video by deffining the parameters: time_position,height_position,width_position,channel_position.
    time_position : int, optional
        Position for the dimension for the time in the original video array. The default is 0.
    height_position : int, optional
        Position for the dimension for the y-axis (height) in the original video array. The default is 1.
    width_position : int, optional
        Position for the dimension for the x-axis (width) in the original video array. The default is 2.
    channel_position : int, optional
        Position for the dimension for the channels in the original video array. The default is 3.
    '''    
    def __init__(self, video, time_position = 0,height_position= 1,  width_position = 2, channel_position = 3):
        self.video = video
        self.time_dimension = time_position
        self.height_dimension = height_position
        self.width_dimension = width_position
        self.channel_dimension = channel_position
    def transpose_video(self):
        '''
        Method that transposes an unsorted video to the standard [T,Y,X,C]
        
        Returns
        -------
        video_correct_order : np.uint16
            Array with dimensions [T,Y,X,C].
        '''
        # making a copy of the video
        video_transposed = np.copy(self.video, dtype=np.uint16)
        # reshaping the video
        video_transposed = np.transpose(video_transposed, (self.time_dimension, self.height_dimension, self.width_dimension, self.channel_dimension))
        # calculating the video shape
        number_frames = video_transposed.shape[0]
        height = video_transposed.shape[1]
        width = video_transposed.shape[2]
        number_channels = video_transposed.shape[3]
        # Filling  with zeros the dimension with channel in case it has less than 3 colors video.
        if video_transposed.shape[3] <3:
            print ('The video has been transposed to the format [T,Y,X,C] and the channels are RGB')
            video_correct_order = np.zeros((number_frames,width,height,3),dtype=np.uint16)
            video_correct_order[:,:,:,:number_channels] = video_transposed
        elif video_transposed.shape[3] == 3:
            print ('The video has been transposed to the format [T,Y,X,C]')
            video_correct_order = video_transposed.copy()
        return video_correct_order
    
    def image_to_video(self):
        '''
        Method that converts an image into a video with one frame. This process is done for compatibility with the rest of the classes.
        
        Returns
        -------
        video_correct_order : np.uint16
            Array with dimensions [T,Y,X,C].
        '''
        # This section corrects the video to the dimensions. [T,Y,X,C] in case it is an image with 2D x,y.
        if len(self.video.shape)==2:
            video_temp = self.video.copy()
            video_correct_order = np.zeros((1,self.video.shape[0],self.video.shape[1],3), dtype=np.uint16)
            video_correct_order[0,:,:,0] = video_temp 
            print ('The video has been converted to the format [T,Y,X,C] from [Y,X]')  
        
        if len(self.video.shape)==3:
            video_temp = self.video.copy()
            video_correct_order = np.zeros((1,self.video.shape[0],self.video.shape[1],3), dtype=np.uint16)
            video_correct_order[0,:,:,:] = video_temp 
            print ('The video has been converted to the format [T,Y,X,C] from [Y,X,C]')  
        return video_correct_order
    
        
class RemoveExtrema():
    '''
    This class is inteded to remove extreme values from a video. The format of the video must be [T,Y,X,C].
    
    Parameters
    ----------
    video : numpy array 
        Array of images with dimensions [T,Y,X,C].
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
            Normalized video. Array with dimensions [T,Y,X,C].
        '''
        normalized_video = np.copy(self.video, dtype=np.uint16)
        number_timepoints, number_channels   = self.video.shape[0], self.video.shape[3]
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


class ScaleVideo():
    '''
    This class is intended to scale the intensity values in a video. The format of the video must be [T,Y,X,C].
    
    Parameters
    ----------
    video : numpy array 
        Array of images with dimensions [T,Y,X,C].
    scale_percentage : float, optional
        Percentage value (between 0 and 1, where 0.5 represents 50% scaling) to scale the video. The default is 1.
    ignore_channel : int or None, optional
        Use this option to ignore the normalization of a given channel. The default is None.
    '''
    def __init__(self, video, scale_percentage=1,ignore_channel =None):
        self.video = video
        self.scale_percentage = scale_percentage
        self.ignore_channel = ignore_channel
    def apply_scale(self):
        '''
        This method is intendent to scale the intensity values of a video.

        Returns
        -------
        scaled_video : np.uint16
            Scaled video. Array with dimensions [T,Y,X,C].
        '''
        scaled_video = np.copy(self.video, dtype=np.uint16)
        number_timepoints, number_channels   = self.video.shape[0], self.video.shape[3]
        for index_channels in range (number_channels):
            if not self.ignore_channel ==index_channels:
                for index_time in range (number_timepoints):
                    scaled_video_temp = scaled_video[index_time,:,:,index_channels]
                    if not np.amax(scaled_video_temp) ==0: # this section detect that the channel is not empty to perform the normalization.
                        scaled_video_temp = scaled_video_temp*self.scale_percentage
                        scaled_video_temp[scaled_video_temp<0]=0
        return scaled_video
    
    
class BandpassFilter(): 
    '''
    This class is intended to apply high and low bandpass filters to the video. The format of the video must be [T,Y,X,C].
    
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









































    
    
    
    