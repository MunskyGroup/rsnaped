# -*- coding: utf-8 -*-
'''
rSNAPed: A software for single-molecule image tracking, simulation and parameter estimation.
Created on Fri Jun 26 22:10:24 2020
@author: Luis Aguilera. luis.aguilera@colostate.edu
'''

# https://eikonomega.medium.com/getting-started-with-sphinx-autodoc-part-1-2cebbbca5365
#https://docs.anaconda.com/restructuredtext/detailed/

#module_name, package_name, ClassName, method_name, 
# ExceptionName, function_name, GLOBAL_CONSTANT_NAME, 
# global_var_name, instance_var_name, function_parameter_name, local_var_name.

# Conventions.
#number_timepoints
# number_channels
# index_channels
# index_time


import_libraries =1
if import_libraries ==1:
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


class ConverToStandardFormat():
    '''
    This class contains two methods to:
    
    1. Transform any numpy array of images into the format [T, Y, X, C]. 
    2. Convert an image into an array video with a single time point (this last is necessary for compatibility).
    
    Parameters
    ----------
    video : NumPy array 
        Array of images. This class accepts arrays with formats: [Y, X], [T, Y, X], [T, Y, X, C], or any other  permutation of channels, the user must specify the position of each dimension in the original video by defining the parameters: time_position,height_position,width_position,channel_position.
    time_position : int, optional
        Position for the dimension for the time in the original video array. The default is 0.
    height_position : int, optional
        Position for the dimension for the y-axis (height) in the original video array. The default is 1.
    width_position : int, optional
        Position for the dimension for the x-axis (width) in the original video array. The default is 2.
    channel_position : int, optional
        Position for the channel's dimension in the original video array. The default is 3.
    '''    
    def __init__(self, video, time_position = 0,height_position= 1,  width_position = 2, channel_position = 3):
        self.video = video
        self.time_dimension = time_position
        self.height_dimension = height_position
        self.width_dimension = width_position
        self.channel_dimension = channel_position
    def transpose_video(self):
        '''
        Method that transposes an unsorted video to the standard [T, Y, X, C]
        
        Returns
        -------
        video_correct_order : np.uint16
            Array with dimensions [T, Y, X, C].
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
            print ('The video has been transposed to the format [T, Y, X, C] and the channels are RGB')
            video_correct_order = np.zeros((number_frames,width,height,3),dtype=np.uint16)
            video_correct_order[:,:,:,:number_channels] = video_transposed
        elif video_transposed.shape[3] == 3:
            print ('The video has been transposed to the format [T, Y, X, C]')
            video_correct_order = video_transposed.copy()
        return video_correct_order
    def image_to_video(self):
        '''
        Method that converts an image into a video with one frame. This process is done for compatibility with the rest of the classes.
        
        Returns
        -------
        video_correct_order : np.uint16
            Array with dimensions [T, Y, X, C].
        '''
        # This section corrects the video to the dimensions. [T, Y, X, C] in case it is an image with 2D x,y.
        if len(self.video.shape)==2:
            video_temp = self.video.copy()
            video_correct_order = np.zeros((1,self.video.shape[0],self.video.shape[1],3), dtype=np.uint16)
            video_correct_order[0,:,:,0] = video_temp 
            print ('The video has been converted to the format [T, Y, X, C] from [Y, X]')  
        
        if len(self.video.shape)==3:
            video_temp = self.video.copy()
            video_correct_order = np.zeros((1,self.video.shape[0],self.video.shape[1],3), dtype=np.uint16)
            video_correct_order[0,:,:,:] = video_temp 
            print ('The video has been converted to the format [T, Y, X, C] from [Y, X, C]')  
        return video_correct_order
    
    
class RemoveExtrema():
    '''
    This class is intended to remove extreme values from a video. The format of the video must be [T, Y, X, C].
    
    Parameters
    ----------
    video : NumPy array 
        Array of images with dimensions [T, Y, X, C].
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
            Normalized video. Array with dimensions [T, Y, X, C].
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


class ScaleIntensity():
    '''
    This class is intended to scale the intensity values in a video. The format of the video must be [T, Y, X, C].
    
    Parameters
    ----------
    video : NumPy array 
        Array of images with dimensions [T, Y, X, C].
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
        This method is intended to scale the intensity values of a video.

        Returns
        -------
        scaled_video : np.uint16
            Scaled video. Array with dimensions [T, Y, X, C].
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
    This class is intended to apply high and low bandpass filters to the video. The format of the video must be [T, Y, X, C]. This class uses **difference_of_gaussians** from skimage.filters.
    
    Parameters
    ----------
    video : NumPy array 
        Array of images with dimensions [T, Y, X, C].
    min_percentile : int between 0 to 100, optional
        Lower bound to normalize intensity. The default is 1.
    max_percentile : int between 0 to 100, optional
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
        This method applies high and low bandpass filters to the video. 
        
        Returns
        -------
        video_filtered : np.uint16
            Filtered video resulting from the bandpass process. Array with format [T, Y, X, C].
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


class MaskingImage():
    '''
    This class is intended to apply a mask to the video. The video format must be [T, Y, X, C], and the format of the mask must be [Y, X].
    
    Parameters
    ----------
    video : NumPy array 
        Array of images with dimensions [T, Y, X, C].
    mask : NumPy array, with Boolean values, where 1 represents the masked area, and 0 represents the area outside the mask. 
        An array of images with dimensions [Y, X].
    '''
    def __init__(self,video,mask):
        self.mask = mask
        self.video = video
        def apply_mask(self):
            '''
            This method applies high and low bandpass filters to the video. The method uses **difference_of_gaussians** from skimage.filters.
        
            Returns
            -------
            video_removed_mask : np.uint16
            Video whit zero values outside the area delimited by the mask. Array with format [T, Y, X, C].
            '''
            video_removed_mask = np.einsum('ijkl,jk->ijkl', self.video, self.mask)
            return video_removed_mask


class BeadsAlignment():
    '''
    This class is intended to detected and align spots detected in the various channels of an image with dimensions [C, Y, X]. The class returns a homography matrix that can be used to align the images captured from two different cameras during the experiment. Notice that this class should only be used for images taken from a microscope setup that uses two cameras to image various channels. 
    
    Parameters
    ----------
    image_beads : NumPy array 
        Array with a simple image with dimensions [C, Y, X].
    spot_size : int, optional
        Average size of the beads,  The default is 5.
    min_intensity : int, optional
        Minimal expected intensity for the beads. The default is 400.        
    '''
    def __init__(self, image_beads,spot_size = 5, min_intensity =400):
        self.image_beads = image_beads
        self.spot_size = spot_size
        self.min_intensity = min_intensity
    def make_beads_alignment(self): 
        '''
        This method aligns a list of spots detected in an image with dimensions [C, Y, X] and returns a homography matrix.
        
        Returns
        -------
        homography_matrix : object 
            The homography matrix is a 3x3 matrix. This transformation matrix maps the points between two planes (images). 
        '''
        # Bandpass filter for the beads function
        low_pass_filter = 1 # low pass filter threshold 
        high_pass_filter = 71 # high pass filter threshold 
        self.image_beads[0,:,:]= tp.bandpass(self.image_beads[0,:,:], low_pass_filter, high_pass_filter, threshold=1, truncate=4) # Red channel
        self.image_beads[1,:,:]= tp.bandpass(self.image_beads[1,:,:], low_pass_filter, high_pass_filter, threshold=1, truncate=4) # Green channel
        # Locating beads in the image using "tp.locate" function from trackpy.
        f_red = tp.locate(self.image_beads[0,:,:],self.spot_size, self.min_intensity,maxsize=7,percentile=60) # data frame for the red channel
        f_green = tp.locate(self.image_beads[1,:,:],self.spot_size, self.min_intensity,maxsize=7,percentile=60)  # data frame for the green channel
        # Converting coordenates to float32 array for the red channel
        x_coord_red = np.array(f_red.x.values, np.float32)
        y_coord_red = np.array(f_red.y.values, np.float32)
        positions_red = np.column_stack((x_coord_red,y_coord_red ))
        # Converting coordenates to float32 array for the green channel
        x_coord_green = np.array(f_green.x.values, np.float32)
        y_coord_green = np.array(f_green.y.values, np.float32)
        positions_green = np.column_stack(( x_coord_green,y_coord_green ))
        # First step to remove of unmatched spots. Comparing Red versus Green channel.
        comparison_red = np.zeros((positions_red.shape[0]))
        comparison_green = np.zeros((positions_green.shape[0]))
        MIN_DISTANCE_TO_MATCH_BEADS=4
        for i in range (0, positions_red.shape[0]):
            idx = np.argmin(abs((positions_green[:,0] - positions_red[i,0])))
            comparison_red[i] = (abs(positions_green[idx,0] - positions_red[i,0]) <MIN_DISTANCE_TO_MATCH_BEADS) & (abs(positions_green [idx,1] - positions_red[i,1]) <MIN_DISTANCE_TO_MATCH_BEADS)
        for i in range (0, positions_green.shape[0]):
            idx = np.argmin(abs((positions_red[:,0] - positions_green[i,0])))
            comparison_green[i] = (abs(positions_red[idx,0] - positions_green[i,0]) <MIN_DISTANCE_TO_MATCH_BEADS) & (abs(positions_red [idx,1] - positions_green[i,1]) <MIN_DISTANCE_TO_MATCH_BEADS)
        positions_red = np.delete(positions_red, np.where( comparison_red ==0)[0], 0)
        positions_green = np.delete(positions_green, np.where(comparison_green ==0)[0], 0)
        # Second step to remove of unmatched spots. Comparing Green versus Red channel.
        comparison_red = np.zeros((positions_red.shape[0]))
        comparison_green = np.zeros((positions_green.shape[0]))
        for i in range (0, positions_green.shape[0]):
            idx = np.argmin(abs((positions_red[:,0] - positions_green[i,0])))
            comparison_green[i] = (abs(positions_red[idx,0] - positions_green[i,0]) <MIN_DISTANCE_TO_MATCH_BEADS) & (abs(positions_red [idx,1] - positions_green[i,1]) <MIN_DISTANCE_TO_MATCH_BEADS)
        for i in range (0, positions_red.shape[0]):
            idx = np.argmin(abs((positions_green[:,0] - positions_red[i,0])))
            comparison_red[i] = (abs(positions_green[idx,0] - positions_red[i,0]) <MIN_DISTANCE_TO_MATCH_BEADS) & (abs(positions_green [idx,1] - positions_red[i,1]) <MIN_DISTANCE_TO_MATCH_BEADS)
        positions_red = np.delete(positions_red, np.where( comparison_red ==0)[0], 0)
        positions_green = np.delete(positions_green, np.where(comparison_green ==0)[0], 0)
        print('The number of spots detected for the red channel are:')
        print(positions_red.shape)
        print('The number of spots detected for the green channel are:')
        print(positions_green.shape)
        # Calculating the minimum value of rows for the alignment
        no_spots_for_alignment = min(positions_red.shape[0],positions_green.shape[0])
        homography = transform.ProjectiveTransform()
        src = positions_red[:no_spots_for_alignment,:2]
        dst = positions_green[:no_spots_for_alignment,:2]
        homography.estimate(src, dst)
        homography_matrix = homography
        print('')
        print('The homography matrix is:')
        print (homography_matrix)
        return homography_matrix


class CamerasAlignment():
    '''
    This class is intended to align the images from multiple channels by applying a matrix transformation using the homography

    Parameters
    ----------
    video : NumPy array 
        Array of images with dimensions [T, Y, X, C].
    homography_matrix : object
        The homography matrix object generated with the class BeadsAlignment. 
    target_channels : List of int, optional
        Lower bound to normalize intensity. The default is [1].
    '''
    def __init__(self, video, homography_matrix, target_channels = [1]):
        self.video = video
        self.homography_matrix = homography_matrix
        self.target_channels = target_channels
    def make_video_alignment(self): 
        '''
        This method transforms the video by multiplying the target channels by the homography matrix.
        
        Returns
        -------
        transformed_video : np.uint16
            Transformed video. Array with dimensions [T, Y, X, C].
        '''
        transformed_video = np.zeros_like(self.video) 
        number_timepoints, height, width,number_channels =self.video.shape
        # Applying the alignment transformation to the whole video. Matrix multiplication to align the images from the two cameras.
        for index_channels in range(0,number_channels): # green and blue channels
            for index_time in range(0,number_timepoints):
                if index_channels in self.target_channels:
                    transformed_video[index_time,:,:,index_channels] = warp(self.video[index_time,:,:,index_channels], self.homography_matrix.params, output_shape=(height, width),preserve_range=True)
                else:
                    transformed_video[index_time,:,:,index_channels] = self.video[index_time,:,:,index_channels]
        return transformed_video


class VisualizerImage(): 
    '''
    This class is intended to visualize videos as 2D images. This class has the option to mark the particles that previously were selected by trackPy.  

    Parameters
    ----------
    list_videos : List of NumPy arrays or a single NumPy array 
        Images or videos to visualize. The format is a list of Numpy arrays where each array follows the convention [T, Y, X, C] or an image array with format [Y, X].
    list_videos_filtered : List of NumPy arrays or a single NumPy array or None
        Images or videos to visualize. The format is a list of Numpy arrays where each array follows the convention [T, Y, X, C]. The default is None.
    list_selected_particles_dataframe : pandas data frame, optional
        A pandas data frame containing the position of each spot in the image. The default is None.
    
    list_files_names : List of str or str, optional
        List of file names to display as the title on the image. The default is None.

    selected_channel : int, optional
        Allows the user to define the channel to visualize in the plotted images. The default is 0.
    selected_timepoint : int, optional
        Allows the user to define the time point or frame to display on the image. The default is 0.
    normalize : bool, optional
        Option to normalize the data by removing outliers. The code removes the 1 and 99 percentiles from the image. The default is False.
    individual_figure_size : int, optional
        Allows the user to change the size of each image. The default is 5.
    '''
    def __init__(self,list_videos,list_videos_filtered=None,list_selected_particles_dataframe=None,list_files_names=None,selected_channel=0,selected_timepoint=0,normalize=False,individual_figure_size=5):
        self.particle_size =7
        self.selected_timepoint = selected_timepoint
        self.selected_channel =selected_channel
        self.individual_figure_size = individual_figure_size        
        # Checkiing if the video is a list or a single video.
        if not (type(list_videos) is list):
            self.list_videos = [list_videos]
            self.number_videos = len(list_videos)
        else:
            self.number_videos = 1
        if not (type(list_files_names) is list):
            self.list_files_names = [list_files_names]
        else:
            self.list_files_names = list_files_names        
        if not (type(list_videos_filtered) is list):
            self.list_videos_filtered = [list_videos_filtered]
        if not (type(list_selected_particles_dataframe) is list):
            self.list_selected_particles_dataframe = [list_selected_particles_dataframe]        
        self.list_number_frames = [list_videos[i].shape[0] for i in range(0,self.number_videos)]
        maximum_timepoint_video =  min ( self.list_number_frames ) # Minimum of maximum size in the list of videos.
        if selected_timepoint > maximum_timepoint_video:
            self.selected_timepoint = maximum_timepoint_video
        # remove the 1 and 99 percentile if normalize == True
        if normalize == True:
            list_videos_normalized = []
            for index_video in range(0,self.number_videos):
                number_timepoints, _, _, number_channels = list_videos[index_video].shape
                temp_video = list_videos[index_video].copy()
                for index_channels in range (number_channels):
                    for index_time in range (number_timepoints):
                        temp_video[index_time,:,:,index_channels] = RemoveExtrema(temp_video[index_time,:,:,index_channels]).remove_outliers()
                list_videos_normalized.append(temp_video)
            self.list_videos = list_videos_normalized
        else:
            self.list_videos = list_videos
        # This section converts an image [Y, X] into a video with dimensions. [T, Y, X, C].
        if len(list_videos[0].shape)==2:
            list_videos_4D = []
            for index_video in range(0,self.number_videos):
                temp_video_shape = np.zeros((1,list_videos[index_video].shape[0],list_videos[index_video].shape[1],1), dtype=np.float32)
                temp_video_shape[0,:,:,0] = list_videos[index_video] 
                list_videos_4D.append(temp_video_shape)
            list_videos = list_videos_4D
            self.selected_channel =0
            self.selected_timepoint =0        
    def plot(self):
        '''
        This method plots a list of images as a grid. 

        Returns
        -------
        None.
        '''
        # Plotting only the cells 
        if ( self.list_selected_particles_dataframe[0] is None):
            NUM_COLUMNS = 3
            NUM_ROWS = int(math.ceil(len(self.list_videos) / NUM_COLUMNS))
            # Loop to plot multiple cells in a grid
            gs = gridspec.GridSpec(NUM_ROWS, NUM_COLUMNS)
            gs.update(wspace=0.1, hspace=0.1) # set the spacing between axes.
            fig = plt.figure(figsize=(self.individual_figure_size*NUM_COLUMNS, self.individual_figure_size*NUM_ROWS))
            for index_video in range(0,self.number_videos):
                ax = fig.add_subplot(gs[index_video])
                ax.imshow(self.list_videos[index_video][self.selected_timepoint,:,:,self.selected_channel],cmap='gray')
                ax.set_xticks([])
                ax.set_yticks([])
                if not ( self.list_files_names[0] is None):
                    ax.set(title=self.list_files_names[index_video][0:-4])
                else:
                    ax.set(title='Cell_'+str(index_video))
            fig.tight_layout()
            plt.show()
        # Plotting the cells and the detected spots
        if not ( self.list_selected_particles_dataframe[0] is None) and ( ( self.list_videos_filtered[0] is None)):
            NUM_COLUMNS = 3
            NUM_ROWS = int(math.ceil(len(self.list_videos) / NUM_COLUMNS))
            # Loop to plot multiple cells in a grid
            gs = gridspec.GridSpec(NUM_ROWS, NUM_COLUMNS)
            gs.update(wspace=0.1, hspace=0.1) # set the spacing between axes.
            fig = plt.figure(figsize=(self.individual_figure_size*NUM_COLUMNS, self.individual_figure_size*NUM_ROWS))
            counter = 0
            for index_video in range(0,self.number_videos):
                ax = fig.add_subplot(gs[index_video])
                ax.imshow(self.list_videos[index_video][self.selected_timepoint,:,:,self.selected_channel],cmap='gray')
                ax.set_xticks([])
                ax.set_yticks([])
                if not ( self.list_files_names[0] is None):
                    ax.set(title=self.list_files_names[index_video][0:-4])
                else:
                    ax.set(title='Cell_'+str(index_video))
                # main loop to mark spots
                number_particles  = None
                frames_part = None
                x_pos = None
                y_pos = None
                selected_particles_dataframe = self.list_selected_particles_dataframe[counter]
                if not len (self.list_selected_particles_dataframe[counter]) == 0:
                    number_particles = selected_particles_dataframe['particle'].nunique()
                    for k in range (0,number_particles):
                        frames_part =selected_particles_dataframe.loc[selected_particles_dataframe['particle']==selected_particles_dataframe['particle'].unique()[k]].frame.values  
                        index_time = self.selected_timepoint
                        if index_time in frames_part: # plotting the circles for each detected particle at a given time point
                            index_val=np.where(frames_part == index_time)
                            x_pos=int(selected_particles_dataframe.loc[selected_particles_dataframe['particle']==selected_particles_dataframe['particle'].unique()[k]].x.values[index_val])
                            y_pos=int(selected_particles_dataframe.loc[selected_particles_dataframe['particle']==selected_particles_dataframe['particle'].unique()[k]].y.values[index_val])
                        try:
                            circle = plt.Circle((x_pos, y_pos), self.particle_size/2, color='yellow', fill=False)
                            ax.add_artist(circle)
                        except:
                            pass
                counter +=1
            fig.tight_layout()
            plt.show()   
        # Plotting the cells and the detected spots and the filtered video.
        if (not ( self.list_selected_particles_dataframe[0] is None)) and (not ( self.list_videos_filtered[0] is None)) :
            NUM_COLUMNS = 3
            NUM_ROWS = self.number_videos
            gs = gridspec.GridSpec(NUM_ROWS, NUM_COLUMNS)
            gs.update(wspace=0.1, hspace=0.1) # set the spacing between axes.
            fig = plt.figure(figsize=(self.individual_figure_size*NUM_COLUMNS, self.individual_figure_size*NUM_ROWS))
            counter = 0
            for index_video in range(0, self.number_videos*3, 3):
                if not ( self.list_files_names[0] is None):
                    title_str = self.list_files_names[counter][0:-4]
                else:
                    title_str = 'Cell_'+str(counter)
                # Figure with raw video
                ax = fig.add_subplot(gs[index_video])
                ax.imshow(self.list_videos[counter][self.selected_timepoint,:,:,self.selected_channel],cmap='gray')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set(title= title_str + ' Original')
                # Figure with filtered video
                ax = fig.add_subplot(gs[index_video+1])
                ax.imshow(self.list_videos_filtered[counter][self.selected_timepoint,:,:,self.selected_channel],cmap='gray')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set(title=title_str + ' Filtered' )
                # Figure with filtered video and marking the spots
                ax = fig.add_subplot(gs[index_video+2])
                ax.imshow(self.list_videos_filtered[counter][self.selected_timepoint,:,:,self.selected_channel],cmap='gray')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set(title=title_str + ' Filtered' )
                # main loop to mark spots
                number_particles  = None
                frames_part = None
                x_pos = None
                y_pos = None
                selected_particles_dataframe = self.list_selected_particles_dataframe[counter]
                if not len (self.list_selected_particles_dataframe[counter]) == 0:
                    number_particles = selected_particles_dataframe['particle'].nunique()
                    for k in range (0,number_particles):
                        frames_part =selected_particles_dataframe.loc[selected_particles_dataframe['particle']==selected_particles_dataframe['particle'].unique()[k]].frame.values  
                        index_time = self.selected_timepoint
                        if index_time in frames_part: # plotting the circles for each detected particle at a given time point
                            index_val=np.where(frames_part == index_time)
                            x_pos=int(selected_particles_dataframe.loc[selected_particles_dataframe['particle']==selected_particles_dataframe['particle'].unique()[k]].x.values[index_val])
                            y_pos=int(selected_particles_dataframe.loc[selected_particles_dataframe['particle']==selected_particles_dataframe['particle'].unique()[k]].y.values[index_val])
                        try:
                            circle = plt.Circle((x_pos, y_pos), self.particle_size/2, color='yellow', fill=False)
                            ax.add_artist(circle)
                        except:
                            pass
                counter +=1
            fig.tight_layout()
            plt.show()    


class VisualizerVideo():   
    '''
    This class is intended to visualize videos as interactive widgets. This class has the option to mark the particles that previously were selected by trackPy.  
    
    Parameters
    ----------
    list_videos : List of NumPy arrays or a single NumPy array 
        Images or videos to visualize. The format is a list of Numpy arrays where each array follows the convention [T, Y, X, C] or an image array with format [Y, X].
    list_selected_particles_dataframe : pandas data frame, optional
        A pandas data frame containing the position of each spot in the image. The default is None.
    list_mask_array : List of NumPy arrays or a single NumPy array, with Boolean values, where 1 represents the masked area, and 0 represents the area outside the mask. 
        An array of images with dimensions [Y, X].      
    show_time_projection_spots : int, optional
       Allows the user to display the projection of all detected spots for all time points on the current image. The default is 0.
    normalize : bool, optional
        Option to normalize the data by removing outliers. The code removes the 1 and 99 percentiles from the image. The default is False.
    '''
    def __init__(self,list_videos,list_selected_particles_dataframe,list_mask_array=None,show_time_projection_spots=0,normalize=False ):
        self.particle_size =7
        self.show_time_projection_spots = show_time_projection_spots
        # Checking if the video is a list or a single video.
        if not (type(list_videos) is list):
            self.list_videos = [list_videos]
            self.number_videos = len(list_videos)
        else:
            self.number_videos = 1
        if not (type(list_mask_array) is list):
            self.list_mask_array = [list_mask_array]
        else:
            self.list_mask_array = list_mask_array 
        if not (type(list_selected_particles_dataframe) is list):
            self.list_selected_particles_dataframe = [list_selected_particles_dataframe]        
        self.list_number_frames = [list_videos[i].shape[0] for i in range(0,len(list_videos))] 
        self.min_time_all_cells = min(self.list_number_frames)        
        # remove the 1 and 99 percentile if normalize == True
        if normalize == True:
            list_videos_normalized = []
            for index_video in range(0,self.number_videos):
                number_timepoints, _, _, number_channels = list_videos[index_video].shape
                temp_video = list_videos[index_video].copy()
                for index_channels in range (number_channels):
                    for index_time in range (number_timepoints):
                        temp_video[index_time,:,:,index_channels] = RemoveExtrema(temp_video[index_time,:,:,index_channels]).remove_outliers()
                list_videos_normalized.append(temp_video)
            self.list_videos = list_videos_normalized
        else:
            self.list_videos = list_videos            
        n_channels = [list_videos[i].shape[3] for i in range(0,len(list_videos))] 
        self.min_num_channels = np.amin((n_channels))
    def make_video_app(self):
        '''
        This method returns two objects (controls and output) that can be used to display a widget.

        Returns
        -------
        controls : object
            Controls from interactive to use with ipywidgets **display**.
        output : object
            Output values from from interactive to use with ipywidgets **display**.
        '''
        def figure_viewer(drop_cell,index_time,drop_channel):
            video = self.list_videos[drop_cell]
            selected_particles_dataframe = self.list_selected_particles_dataframe[drop_cell]
            drop_size = self.particle_size
            plt.figure(1)
            ax = plt.gca()
            if drop_channel == 'Ch_0':
                channel =0
                plt.imshow(video[index_time,:,:,channel],cmap='gray')
            elif drop_channel == 'Ch_1':
                channel = 1
                plt.imshow(video[index_time,:,:,channel],cmap='gray')
            elif drop_channel == 'Ch_2':
                channel =2
                plt.imshow(video[index_time,:,:,channel],cmap='gray')
            elif drop_channel == 'Ch_3':
                channel =3
                plt.imshow(video[index_time,:,:,channel],cmap='gray')
            else :
                # Converting a np.uint16 array into float. 
                image = video[index_time,:,:,:].copy()
                min_image, max_image = np.min(image), np.max(image)
                image -= min_image
                image_float = np.array(image,'float32'); 
                image_float *= 255./(max_image-min_image); 
                image = np.asarray(np.round(image_float), 'uint8')
                plt.imshow(image)
            # Plots the detected spots.
            if not ( self.list_selected_particles_dataframe[0] is None):
                n_particles = selected_particles_dataframe['particle'].nunique()
                for k in range (0,n_particles):
                    frames_part =selected_particles_dataframe.loc[selected_particles_dataframe['particle']==selected_particles_dataframe['particle'].unique()[k]].frame.values  
                    if index_time in frames_part: # plotting the circles for each detected particle at a given time point
                        index_val=np.where(frames_part == index_time)
                        x_pos=int(selected_particles_dataframe.loc[selected_particles_dataframe['particle']==selected_particles_dataframe['particle'].unique()[k]].x.values[index_val])
                        y_pos=int(selected_particles_dataframe.loc[selected_particles_dataframe['particle']==selected_particles_dataframe['particle'].unique()[k]].y.values[index_val])
                    elif self.show_time_projection_spots==1: # In case the spot is not detected in a given time point, plot the closest the point
                        index_closest = np.abs(frames_part - index_time).argmin()
                        x_pos=int(selected_particles_dataframe.loc[selected_particles_dataframe['particle']==selected_particles_dataframe['particle'].unique()[k]].x.values[index_closest])
                        y_pos=int(selected_particles_dataframe.loc[selected_particles_dataframe['particle']==selected_particles_dataframe['particle'].unique()[k]].y.values[index_closest])
                    try:
                        circle = plt.Circle((x_pos, y_pos), drop_size/2, color='yellow', fill=False)
                        ax.add_artist(circle)
                    except:
                        pass
            # Plots the mask contour on the video
            if not ( self.list_mask_array[0] is None):
                mask_array = self.list_mask_array[drop_cell]
                if len(mask_array.shape) == 3: 
                    contuour_position = find_contours(mask_array[index_time,:,:], 0.8)
                elif len(mask_array.shape) == 2:
                    contuour_position = find_contours(mask_array[:,:], 0.8)
                temp = contuour_position[0][:,1]
                temp2 =contuour_position[0][:,0]
                plt.fill(temp,temp2, facecolor='none', edgecolor='yellow')
            plt.show()
        # This section deffines the drop menu for the number of channels in the video.
        if self.min_num_channels == 1:
            options_channels = ['Ch_0']
        if self.min_num_channels == 2:
            options_channels = ['Ch_0', 'Ch_1']
        if self.min_num_channels == 3:
            options_channels = ['Ch_0', 'Ch_1', 'Ch_2','All_Channels']
        if self.min_num_channels == 4:
            options_channels = ['Ch_0', 'Ch_1', 'Ch_2', 'Ch_3','All_Channels']    
        options_cells = list(range(0, self.number_videos))
        interactive_plot = interactive(figure_viewer,drop_cell= widgets.Dropdown(options=options_cells,description='Cell'),
                                       index_time = widgets.IntSlider(min=0,max=self.min_time_all_cells-1,step=1,value=0,description='Time'), 
                                       drop_channel = widgets.Dropdown(options=options_channels,description='Channel'))
        controls = HBox(interactive_plot.children[:-1], layout = Layout(flex_flow='row wrap'))
        output = interactive_plot.children[-1]
        return controls, output   
    

class VisualizerVideo3D():
    '''
    This class is intended to visualize 3d videos as interactive widgets. 
    
    Parameters
    ----------
    list_videos : List of NumPy arrays or a single NumPy array 
        Images or videos to visualize. The format is a list of Numpy arrays where each array follows the convention [T, Y, X, C] or an image array with format [Y, X].
    '''
    def __init__(self,list_videos):
        # Checking if the video is a list or a single video.
        if not (type(list_videos) is list):
            self.list_videos = [list_videos]
            self.number_videos = len(list_videos)
        else:
            self.number_videos = 1        
        self.list_number_frames = [list_videos[i].shape[0] for i in range(0,len(list_videos))] 
        self.min_time_all_cells = min(self.list_number_frames)
        list_number_z_slices = [list_videos[i].shape[1] for i in range(0,len(list_videos))] 
        self.min_z_slices= min(list_number_z_slices)
        n_channels = [list_videos[i].shape[4] for i in range(0,len(list_videos))] 
        self.min_num_channels = np.amin((n_channels))
        
    def make_video_app(self):
        '''
        This method returns two objects (controls and output) that can be used to display a widget.

        Returns
        -------
        controls : object
            Controls from interactive to use with ipywidgets **display**.
        output : object
            Output values from from interactive to use with ipywidgets **display**.
        '''
        def figure_viewer(drop_cell, drop_channel, index_z_axis, index_time):
            plt.figure(1)
            video = self.list_videos[drop_cell]
            if drop_channel == 'Ch_0':
                channel =0
                plt.imshow(video[index_time,index_z_axis,:,:,channel],cmap='gray')
            elif drop_channel == 'Ch_1':
                channel = 1
                plt.imshow(video[index_time,index_z_axis,:,:,channel],cmap='gray')
            elif drop_channel == 'Ch_2':
                channel =2
                plt.imshow(video[index_time,index_z_axis,:,:,channel],cmap='gray')
            elif drop_channel == 'Ch_3':
                channel =3
                plt.imshow(video[index_time,index_z_axis,:,:,channel],cmap='gray')
            elif drop_channel == 'All_Channels' :
                # Converting a np.uint16 array into float. 
                image = video[index_time,index_z_axis:,:,0:int(np.amin((3,self.min_num_channels)))].copy()
                min_image, max_image = np.min(image), np.max(image)
                image -= min_image; 
                image_float = np.array(image,'float32'); 
                image_float *= 255./(max_image-min_image); 
                image = np.asarray(np.round(image_float), 'uint8')
                plt.imshow(image)
            plt.axis('off')
            plt.show()               
        options_cell = list(range(0, self.number_videos))   
        if self.min_num_channels == 1:
            options_ch = ['Ch_0']
        if self.min_num_channels == 2:
            options_ch = ['Ch_0', 'Ch_1']
        if self.min_num_channels == 3:
            options_ch = ['Ch_0', 'Ch_1', 'Ch_2','All_Channels']
        if self.min_num_channels == 4:
            options_ch = ['Ch_0', 'Ch_1', 'Ch_2', 'Ch_3','All_Channels']
        interactive_plot = interactive(figure_viewer,   drop_cell= widgets.Dropdown(options=options_cell, description='Cell'), 
                                                        drop_channel = widgets.Dropdown(options=options_ch, description='Channel', value = 'Ch_0'),  
                                                        index_z_axis =  widgets.IntSlider(min=0, max=self.min_z_slices-1, step=1,value=0, description='z-slice'), 
                                                        index_time = widgets.IntSlider(min=0, max=self.min_time_all_cells-1, step=1, value=0, description='time') )
        controls = HBox(interactive_plot.children[:-1], layout = Layout(flex_flow='row wrap'))
        output = interactive_plot.children[-1]
        return controls, output


class VisualizerCrops():
    '''
    This class is intended to visualize the spots detected b trackPy as crops in an interactive widget.
    
    Parameters
    ----------
    list_videos : List of NumPy arrays or a single NumPy array.
        Images or videos to visualize. The format is a list of Numpy arrays where each array follows the convention [T, Y, X, C].
    list_selected_particles_dataframe : pandas data frame.
        A pandas data frame containing the position of each spot in the image. 
    particle_size : int, optional
        Allows the user to change the size of the crop. The default is 5.
    '''
    def __init__(self,list_videos, list_selected_particles_dataframe,particle_size=5):        
        if (particle_size % 2) ==0:
            self.particle_size = particle_size + 1
            print('Warning! Particle_size must be an odd number, this was automatically changed to: ', particle_size)
        else:
            self.particle_size = particle_size
        self.disk_size = int(particle_size/2) # size of the half of the crop
        self.crop_size = int(particle_size/2)+2
        # Checking if the video is a list or a single video.
        if not (type(list_videos) is list):
            self.list_videos = [list_videos]
            self.number_videos = len(list_videos)
        else:
            self.number_videos = 1
        
        if not (type(list_selected_particles_dataframe) is list):
            self.list_selected_particles_dataframe = [list_selected_particles_dataframe]        
        self.list_number_frames = [list_videos[i].shape[0] for i in range(0,len(list_videos))] 
        self.min_time_all_cells = min(self.list_number_frames) 
    def make_video_app(self):        
        '''
        This method returns two objects (controls and output) to display a widget.

        Returns
        -------
        controls : object
            Controls from interactive to use with ipywidgets **display**.
        output : object
            Output values from from interactive to use with ipywidgets **display**.
        ''' 
        def figure_viewer(drop_cell, track,index_time):            
            video = self.list_videos[drop_cell]
            selected_particles_dataframe = self.list_selected_particles_dataframe[drop_cell]
            n_particles = selected_particles_dataframe['particle'].nunique()
            frames_vector = np.zeros((n_particles, 2))            
            for index_particle in range(0,n_particles):
                frames_part = selected_particles_dataframe.loc[selected_particles_dataframe['particle']==selected_particles_dataframe['particle'].unique()[index_particle]].frame.values          
                frames_vector[index_particle,:] = frames_part[0],frames_part[-1]
            def return_crop(image,x_pos,y_pos,crop_size):
                # function that recenters the spots
                crop_image = image[y_pos-(crop_size):y_pos+(crop_size+1),x_pos-(crop_size):x_pos+(crop_size+1)]
                return crop_image
            number_channels  = video.shape[3]
            size_cropped_image = (100+(self.crop_size+1)) - (100-(self.crop_size)) # true size of crop in image
            red_image  = np.zeros((size_cropped_image,size_cropped_image))
            green_image  = np.zeros((size_cropped_image,size_cropped_image))
            blue_image  = np.zeros((size_cropped_image,size_cropped_image))
            frames_part =selected_particles_dataframe.loc[selected_particles_dataframe['particle']==selected_particles_dataframe['particle'].unique()[track]].frame.values  
            if index_time in frames_part: # detecting the position for the crop
                index_val=np.where(frames_part == index_time)
                x_pos=int(selected_particles_dataframe.loc[selected_particles_dataframe['particle']==selected_particles_dataframe['particle'].unique()[track]].x.values[index_val])
                y_pos=int(selected_particles_dataframe.loc[selected_particles_dataframe['particle']==selected_particles_dataframe['particle'].unique()[track]].y.values[index_val])
            else: #in case the code doesnt find a position it uses the closes time point value.
                index_closest = np.abs(frames_part - index_time).argmin()
                x_pos=int(selected_particles_dataframe.loc[selected_particles_dataframe['particle']==selected_particles_dataframe['particle'].unique()[track]].x.values[index_closest])
                y_pos=int(selected_particles_dataframe.loc[selected_particles_dataframe['particle']==selected_particles_dataframe['particle'].unique()[track]].y.values[index_closest])
            red_image[:,:] = return_crop(video[index_time,:,:,0],x_pos,y_pos,self.crop_size)
            green_image[:,:] = return_crop(video[index_time,:,:,1],x_pos,y_pos,self.crop_size)
            blue_image[:,:] = return_crop(video[index_time,:,:,2],x_pos,y_pos,self.crop_size)
            
            fig, ax = plt.subplots(1,number_channels, figsize=(10, 5))
            for index_channels in range(0,number_channels):
                if index_channels ==0:
                    ax[index_channels].imshow(red_image[index_time,:,:],origin='bottom',cmap='gray')
                    ax[index_channels].set_axis_off()
                    ax[index_channels].set(title ='Channel_0 (Red)');
                elif index_channels ==1:
                    ax[index_channels].imshow(green_image[index_time,:,:],origin='bottom',cmap='gray')
                    ax[index_channels].set_axis_off()
                    ax[index_channels].set(title ='Channel_1 (Green)');
                else:
                    ax[index_channels].imshow(blue_image[index_time,:,:],origin='bottom',cmap='gray')
                    ax[index_channels].set_axis_off()
                    ax[index_channels].set(title ='Channel_2 (Blue)');
            print('For track '+ str(track) + ' a spot is detected between time points : '+ str(int(frames_vector[track,0])) + ' and ' + str(int(frames_vector[track,1])) )
        options_cells = list(range(0, self.number_videos))
        interactive_plot = interactive(figure_viewer, drop_cell= widgets.Dropdown(options=options_cells,description='Cell'),
                                       track = widgets.IntSlider(min=0,max=self.selected_particles_dataframe['particle'].nunique()-1,step=1,value=0,description='track'),
                                       index_time = widgets.IntSlider(min=0,max=self.video.shape[0]-1,step=1,value=0,description='time'))
        controls = HBox(interactive_plot.children[:-1], layout = Layout(flex_flow='row wrap'))
        output = interactive_plot.children[-1]
        return controls, output 


class Cellpose():
    '''
    This class is intended to detect cells by image masking using **Cellpose** . The class uses optimization to maximize the number of cells or maximize the size of the detected cells.   
    
    Parameters
    ----------
    video : NumPy array 
        Array of images with dimensions [T, Y, X, C].
    num_iterations : int, optional
        Number of iterations for the optimization process. The default is 5.
    channels : List, optional
        List with the channels in the image. For gray images use [0,0], for RGB images with intensity for cytosol and nuclei use [0,1] . The default is [0,0].
    diameter : float, optional
        Average cell size. The default is 120.
    model_type : str, optional
        To detect between the two options: 'cyto' or 'nuclei'. The default is 'cyto'.
    selection_method : str, optional
        Option to use the optimization algorithm to maximize the number of cells or maximize the size options are 'max_area' or 'max_cells'. The default is 'max_area'.
    '''
    def __init__(self,video,num_iterations=5,channels=[0,0],diameter=120,model_type='cyto',selection_method = 'max_area'):
        self.video = video
        self.num_iterations = num_iterations
        self.minimumm_probaility = 0
        self.maximum_probaility = 5
        self.channels = channels 
        self.diameter = diameter
        self.model_type = model_type # options are 'cyto' or 'nuclei'
        self.selection_method = selection_method # options are 'max_area' or 'max_cells'
    def calculate_masks(self):
        '''
        This method performs the process of image masking using **Cellpose**.

        Returns
        -------
        selected_masks : List of NumPy arrays
            List of NumPy arrays with values between 0 and the number of detected cels in the image, where a number larger than zero represents the masked area for each cell, and 0 represents the area where no cells are detected.
        '''
        # Next two lines suppressing output from cellpose
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        model = models.Cellpose(gpu=0, model_type=self.model_type) # model_type='cyto' or model_type='nuclei'
        # Loop that test multiple probabilities in cell pose and returns the masks with the longest area.
        tested_probabilities = np.round(np.linspace(self.minimumm_probaility,self.maximum_probaility,self.num_iterations),2)
        area_longest_mask = np.zeros(self.num_iterations)
        if self.selection_method == 'max_area': 
            # Loop that iterates in the selected probabilities
            for idx, cell_prob in enumerate (tested_probabilities):
                masks, _, _, _ = model.eval(self.video,normalize=True,cellprob_threshold =cell_prob, diameter=self.diameter,min_size=-1, channels=self.channels, progress=None)
                n_masks = np.amax(masks)
                if n_masks >1: # detecting if more than 1 mask are detected per cell
                    size_mask = []
                    for nm in range (1,n_masks+1): # iterating for each mask in a given cell. The mask has values from 0 for background, to int n, where n is the number of detected masks.  
                        size_mask.append(np.sum(masks==nm)) # creating a list with the size of each mask
                    largest_mask=np.argmax(size_mask)+1 # detecting the mask with the largest value
                    temp_mask = np.zeros_like(masks) # making a copy of the image
                    selected_mask = temp_mask + (masks==largest_mask) # Selecting a single mask and making this mask equal to one and the background equal to zero.
                    area_longest_mask[idx] = np.sum(selected_mask)
                else: # do nothing if only a single mask is detected per image.
                    area_longest_mask[idx] = np.sum(masks)    
            # This section of the code selects the probability that should be used in cellpose.
            selected_probability = tested_probabilities[np.argmax(area_longest_mask)]
        if self.selection_method == 'max_cells':
            num_masks = np.zeros(self.num_iterations)
            # Loop that iterates in the selected probabilities
            for idx, cell_prob in enumerate (tested_probabilities):
                masks, _, _, _ = model.eval(self.video,normalize=True,cellprob_threshold =cell_prob, diameter=self.diameter,min_size=-1, channels=self.channels, progress=None)
                num_masks[idx] = np.amax(masks)
            # This section of the code selects the probability that should be used in cellpose.
            selected_probability = tested_probabilities[np.argmax(num_masks)]
        # This line re-runs the cellpose algorithm using the selected probability threshold
        selected_masks, _, _, _ = model.eval(self.video,normalize=True,cellprob_threshold =selected_probability, diameter=self.diameter,min_size=-1, channels=self.channels, progress=None)
        selected_masks[0,:]=0;selected_masks[:,0]=0;selected_masks[selected_masks.shape[0]-1,:]=0;selected_masks[:,selected_masks.shape[1]-1]=0#This line of code ensures that the corners are zeros.
        # reactivating outputs
        sys.stdout.close()
        sys.stdout = old_stdout
        return selected_masks


class CellposeFISH():
    '''
    This class is intended to detect cells in FISH images using **Cellpose**. The class uses optimization to generate the meta-parameters used by cellpose. This class segments the nucleus and cytosol for every cell detected in the image. 

    Parameters
    ----------
    video : NumPy array 
        Array of images with dimensions [Z, Y, X, C].
    channel_with_cytosol : int, optional
        DESCRIPTION. The default is [0,1].
    channel_with_nucleus : list or int, optional
        DESCRIPTION. The default is 2.
    selected_z_slice : int, optional
        DESCRIPTION. The default is 5.
    diameter_cytosol : float, optional
        Average cytosol size in pixels. The default is 150.
    diamter_nucleus : float, optional
        Average nucleus size in pixels. The default is 100.
    show_plot : bool, optional
        If true, it shows a plot with the detected masks. The default is 1.
    '''
    def __init__(self,video, channel_with_cytosol=[0,1], channel_with_nucleus=2, selected_z_slice=5, diameter_cytosol =150, diamter_nucleus=100, show_plot=1):

        self.video = video
        self.selected_z_slice = selected_z_slice
        self.channel_with_cytosol = channel_with_cytosol
        self.channel_with_nucleus = channel_with_nucleus
        self.diameter_cytosol = diameter_cytosol
        self.diamter_nucleus = diamter_nucleus
        self.show_plot = show_plot
    def calculate_masks(self):
        '''
        This method performs the process of cell detection for FISH images using **Cellpose**.

        Returns
        -------
        list_masks_complete_cells : List of NumPy arrays or a single NumPy array 
            Masks for every cell detected in the image. The list contains the mask arrays consisting of one or multiple Numpy arrays with format [Y, X].
        list_masks_nuclei : List of NumPy arrays or a single NumPy array 
            Masks for the nuclei for every cell detected in the image. The list contains the mask arrays consisting of one or multiple Numpy arrays with format [Y, X].
        list_masks_cytosol_no_nuclei : List of NumPy arrays or a single NumPy array 
            Masks for every nucleus and cytosol for every cell detected in the image. The list contains the mask arrays consisting of one or multiple Numpy arrays with format [Y, X].
        index_paired_masks: List of pairs of int
            List of pairs of integers that associates the detected nuclei and cytosol.
        '''
        # This function is intended to separate masks in list of submasks
        def separate_masks (masks):
            list_masks = []
            n_masks = np.amax(masks)
            if n_masks >1: # detecting if more than 1 mask are detected per cell
                #number_particles = []
                for nm in range (1,n_masks+1): # iterating for each mask in a given cell. The mask has values from 0 for background, to int n, where n is the number of detected masks.  
                    mask_copy = masks.copy()
                    tested_mask = np.where(mask_copy ==nm, 1, 0) # making zeros all elements outside each mask, and once all elements inside of each mask. 
                    list_masks.append(tested_mask)
            else:  # do nothing if only a single mask is detected per image.
                list_masks.append(masks)
            return list_masks
        # function that determines if the nuclus is in the cytosol
        def isNucleousInCytosol(mask_nucleus,mask_cyto):
            nucleousInCell =[]
            nucleousInCell.append(0) # Appending an empty value, to avoid errors
            contuour_n = find_contours(mask_nucleus, 0.5)
            contuour_c = find_contours(mask_cyto, 0.5)
            path = mpltPath.Path(contuour_c[0])
            # calculating the center of the mask applying a 10% reduction
            x_coord = [[i][0][1] for i in contuour_n[0]]
            y_coord = [[i][0][0] for i in contuour_n[0]]
            # Center of mask
            center_value = 0.5
            x_center = center_value* min(x_coord) + center_value * max(x_coord)
            y_center = center_value* min(y_coord) + center_value* max(y_coord)
            test_point = [y_center,x_center]
            if path.contains_point(test_point) == 1:
                return 1
            else:
                return 0
        # This funciton takes two list of images for masks and returns a list of lists with pairs indicating the nucleus masks that are contained in a cytosol
        def paired_masks(list_masks_nuclei,list_masks_cyto):
            n_masks_nuclei = len (list_masks_nuclei)
            n_masks_cyto = len(list_masks_cyto)
            array_paired_masks = np.zeros((n_masks_cyto,n_masks_nuclei)) # This array has dimensions n_masks_cyto,n_masks_nuclei and each entry indicate if the masks are paired
            for i in range(0,n_masks_cyto):
                for j in range (0, n_masks_nuclei):
                    array_paired_masks[i,j] = isNucleousInCytosol(list_masks_nuclei[j],list_masks_cyto[i])
            # vertical array with paired masks        
            itemindex = np.where(array_paired_masks==1)
            index_paired_masks = np.vstack((itemindex[0],itemindex[1])).T
            return index_paired_masks # This array has dimensions [n_masks_cyto,n_masks_nuclei] 
        # This function creates a mask for each cell.
        def generate_masks_complete_cell(index_paired_masks,list_separated_masks_cyto):
            list_masks_complete_cells =[]
            #list_idx =[]
            for i in range(0,index_paired_masks.shape[0]):
                sel_mask_c = index_paired_masks[i][0]
                #if sel_mask_c not in list_idx: # conditional expersion to check if the mask is not duplicated
                    #list_idx.append(index_paired_masks[i][0]) # creating a list of indexes of masks to avoid saving the same mask two times
                list_masks_complete_cells.append(list_separated_masks_cyto[sel_mask_c])
            return list_masks_complete_cells
        # This function creates a mask for each nuclei
        def generate_masks_nuclei(index_paired_masks,list_separated_masks_nuclei):
            list_masks_nuclei =[]
            for i in range(0,index_paired_masks.shape[0]):
                sel_mask_n = index_paired_masks[i][1]
                list_masks_nuclei.append(list_separated_masks_nuclei[sel_mask_n])
            return list_masks_nuclei
        # This function creates a mask for each cytosol without nucleus
        def generate_masks_cytosol_no_nuclei(index_paired_masks,list_masks_complete_cells,list_masks_nuclei):
            list_masks_cytosol_no_nuclei =[]
            for i in range(0,index_paired_masks.shape[0]):
                substraction = list_masks_complete_cells[i] - list_masks_nuclei[i]
                substraction[substraction<0] =0
                list_masks_cytosol_no_nuclei.append(substraction)
            return list_masks_cytosol_no_nuclei
        def join_nulcei_maks(index_paired_masks,list_masks_nuclei):
            # this code detects duplicated mask for the nucleus in the same cell and replaces with a joined mask. Also deletes from the list the duplicated elements.
            index_masks_cytosol = index_paired_masks[:,0]
            idxs_to_delete =[]
            for i in range(0, len (index_masks_cytosol)):
                duplicated_masks_idx = np.where(index_masks_cytosol == i)[0]
                if len(duplicated_masks_idx)>1:
                    joined_mask =np.zeros_like(list_masks_nuclei[0])
                    for j in range(0,len(duplicated_masks_idx)):
                        joined_mask = joined_mask + list_masks_nuclei[duplicated_masks_idx[j]]
                        joined_mask[joined_mask>0]=1
                    list_masks_nuclei[duplicated_masks_idx[0]] = joined_mask # replacing the first duplication occurence with the joined mask
                    idxs_to_delete.append(duplicated_masks_idx[1::][0])
            # creating a new list with the joined masks
            list_mask_joined = []
            for i in range(0, len (list_masks_nuclei)):
                if i not in idxs_to_delete:
                    list_mask_joined.append(list_masks_nuclei[i])
            # removing from index
            new_index_paired_masks = np.delete(index_paired_masks, idxs_to_delete,axis=0)
            return list_mask_joined, new_index_paired_masks
        ##### IMPLEMENTATION #####
        # Correcting the 3D video to 2D and normalized
        video_correct_order = np.zeros((1,self.video.shape[1],self.video.shape[2],self.video.shape[3]) ,dtype = np.uint16)
        video_correct_order[0,:,:,:] = self.video[self.selected_z_slice,:,:,:]
        video_normalized = RemoveExtrema(video_correct_order,1,99).remove_outliers()
        # Cellpose
        masks_cyto, _, _ = Cellpose(video_normalized[0,:,:,self.channel_with_cytosol],diameter=self.diameter_cytosol,model_type='cyto',selection_method = 'max_cells' ).calculate_masks()
        masks_nuclei, _, _ = Cellpose(video_normalized[0,:,:,self.channel_with_nucleus],diameter=self.diamter_nucleus,model_type='nuclei',selection_method = 'max_cells').calculate_masks()
        # Implementation
        list_separated_masks_nuclei = separate_masks(masks_nuclei)
        list_separated_masks_cyto = separate_masks(masks_cyto)
        # Array with paired masks
        index_paired_masks  =  paired_masks(list_separated_masks_nuclei,list_separated_masks_cyto)        
        # Optional section that joins multiple nucleus masks
        id_c =index_paired_masks[:,0].tolist()
        duplicated_nuclei_in_masks=any(id_c.count(x) > 1 for x in id_c)
        if duplicated_nuclei_in_masks == True:
            list_masks_nuclei,index_paired_masks = join_nulcei_maks(index_paired_masks,list_separated_masks_nuclei)
        # List of mask
        list_masks_complete_cells = generate_masks_complete_cell(index_paired_masks,list_separated_masks_cyto)
        list_masks_nuclei = generate_masks_nuclei(index_paired_masks,list_separated_masks_nuclei)
        list_masks_cytosol_no_nuclei= generate_masks_cytosol_no_nuclei(index_paired_masks,list_masks_complete_cells,list_masks_nuclei)        
        if self.show_plot ==1:
            fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 10))
            im = video_normalized[0,:,:,0:3].copy()
            imin, imax = np.min(im), np.max(im); im -= imin; 
            imf = np.array(im,'float32'); 
            imf *= 255./(imax-imin); 
            im = np.asarray(np.round(imf), 'uint8')
            axes[0].imshow(im)
            axes[0].set(title='All channels')
            axes[1].imshow(masks_cyto)   
            axes[1].set(title='Cytosol mask')
            axes[2].imshow(masks_nuclei)   
            axes[2].set(title='Nuclei mask')
            axes[3].imshow(im)
            for i in range(0,index_paired_masks.shape[0]):
                contuour_n = find_contours(list_masks_nuclei[i], 0.5)
                contuour_c = find_contours(list_masks_complete_cells[i], 0.5)
                axes[3].fill(contuour_n[0][:,1],contuour_n[0][:,0], facecolor='none', edgecolor='yellow') # mask nucleus
                axes[3].fill(contuour_c[0][:,1],contuour_c[0][:,0], facecolor='none', edgecolor='yellow') # mask cytosol
                axes[3].set(title='Paired masks')
            plt.show()
        return list_masks_complete_cells, list_masks_nuclei, list_masks_cytosol_no_nuclei, index_paired_masks

















    
    
    
    