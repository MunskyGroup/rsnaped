#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
rSNAPed: A software for single-molecule image tracking, simulation and parameter estimation.
Created on Fri Jun 26 22:10:24 2020
Authors: Luis U. Aguilera, William Raymond, Brooke Silagy, Brian Munsky.
'''

# https://eikonomega.medium.com/getting-started-with-sphinx-autodoc-part-1-2cebbbca5365
# https://docs.anaconda.com/restructuredtext/detailed/

# Conventions.
# module_name, package_name, ClassName, method_name,
# ExceptionName, function_name, GLOBAL_CONSTANT_NAME,
# global_var_name, instance_var_name, function_parameter_name, local_var_name.


import_libraries =1
if import_libraries ==1:
    # To manipulate arrays
    import pkg_resources
    pkg_resources.require("numpy>=`1.20.1")  #  to use specific numpy version
    import numpy as np
    from numpy import unravel_index
    # To run stochastic simulations
    import rsnapsim as rss
    # System libraries
    import io
    import sys
    import os
    #import statistics
    from statistics import median_low
    import random
    import math
    from math import nan
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
    from skimage.restoration import denoise_nl_means, estimate_sigma
    from skimage.morphology import square, dilation
    # Open cv

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
    from scipy import ndimage as nd
    # To read .tiff files
    import tifffile
    # to create gifs
    import imageio
    # time libraries
    import pathlib
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

    1. Transform any numpy array of images into the format [T,Y,X,C].
    2. Convert an image into an array video with a single time point (this last is necessary for compatibility).

    Parameters
    ----------
    video : NumPy array
        Array of images. This class accepts arrays with formats: [Y,X], [T,Y,X], [T,Y,X,C], or any other  permutation of channels, the user must specify the position of each dimension in the original video by defining the parameters: time_position,height_position,width_position,channel_position.
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
        Method that transposes an unsorted video to the standard [T,Y,X,C]

        Returns
        -------
        video_correct_order : np.uint16
            Array with dimensions [T,Y,X,C].
        '''
        # making a copy of the video
        video_transposed = np.copy(self.video)
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


class AugmentationVideo():
    '''
    This class is inteded to be used for data augmentation. It takes a video and perform random rotations in the X and Y axis.

    Parameters
    ----------
    video : NumPy array
        Array of images with dimensions [T,Y,X,C].
    '''
    def __init__(self,video):
     self.video = video

    def random_rotation(self):
        '''
        Method that performs random rotations of a video in the Y and X axis.

        Returns
        -------
        video_random_rotation : np.uint16
            Array with dimensions [T,Y,X,C].
        '''
        angles = [0,90,180,270,360]
        selected_angle = random.choice(angles)
        if selected_angle != 0:
            video_random_rotation = nd.rotate(self.video, angle=selected_angle, axes=(1,2))
            return video_random_rotation
        else:
            return self.video


class RemoveExtrema():
    '''
    This class is intended to remove extreme values from a video. The format of the video must be [T,Y,X,C].

    Parameters
    ----------
    video : NumPy array
        Array of images with dimensions [T,Y,X,C] or single image with diemnsions [Y,X].
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
            Normalized video. Array with dimensions [T,Y,X,C] or iamge with format [Y,X].
        '''
        normalized_video = np.copy(self.video)
        normalized_video = np.array(normalized_video,'float32');
        # Normalization code for image with format [Y,X]
        if len(self.video.shape) ==2:
            number_timepoints = 1
            number_channels = 1
            normalized_video_temp = normalized_video
            if not np.amax(normalized_video_temp) ==0: # this section detect that the channel is not empty to perform the normalization.
                max_val = np.percentile(normalized_video_temp, self.max_percentile)
                min_val = np.percentile(normalized_video_temp, self.min_percentile)
                normalized_video_temp [normalized_video_temp> max_val] = max_val
                normalized_video_temp [normalized_video_temp< min_val] = min_val
                normalized_video_temp [normalized_video_temp<0]=0
        # Normalization for video with format [T,Y,X,C].
        else:
            number_timepoints, number_channels   = self.video.shape[0], self.video.shape[3]
            for index_channels in range (number_channels):
                if not self.ignore_channel == index_channels:
                    for index_time in range (number_timepoints):
                        normalized_video_temp = normalized_video[index_time,:,:,index_channels]
                        if not np.amax(normalized_video_temp) ==0: # this section detect that the channel is not empty to perform the normalization.
                            max_val = np.percentile(normalized_video_temp, self.max_percentile)
                            min_val = np.percentile(normalized_video_temp, self.min_percentile)
                            normalized_video_temp [normalized_video_temp> max_val] = max_val
                            normalized_video_temp [normalized_video_temp< min_val] = min_val
                            normalized_video_temp [normalized_video_temp<0]=0
        return np.asarray(normalized_video, 'uint16')



class ScaleIntensity():
    '''
    This class is intended to scale the intensity values in a video. The format of the video must be [T,Y,X,C].

    Parameters
    ----------
    video : NumPy array
        Array of images with dimensions [T,Y,X,C].
    scale_maximum_value : int, optional
        Maximum intensity value to scaled the video. The default is 1000.
    ignore_channel : int or None, optional
        Use this option to ignore the normalization of a given channel. The default is None.
    '''
    def __init__(self, video, scale_maximum_value=1000,ignore_channel =None):
        self.video = video
        self.scale_maximum_value = scale_maximum_value
        self.ignore_channel = ignore_channel
    def apply_scale(self):
        '''
        This method is intended to scale the intensity values of a video.

        Returns
        -------
        scaled_video : np.uint16
            Scaled video. Array with dimensions [T,Y,X,C].
        '''
        scaled_video = np.copy(self.video)
        scaled_video = np.array(scaled_video,'float32');
        number_timepoints, number_channels   = self.video.shape[0], self.video.shape[3]
        for index_channels in range (number_channels):
            if not self.ignore_channel == index_channels:
                for index_time in range (number_timepoints):
                    max_val = np.amax(scaled_video[index_time,:,:,index_channels])
                    min_val = np.amin(scaled_video[index_time,:,:,index_channels])
                    if max_val != 0: # this section detect that the channel is not empty to perform the normalization.
                        scaled_video[index_time,:,:,index_channels] = (scaled_video[index_time,:,:,index_channels]-min_val) / (max_val-min_val)
                        scaled_video[index_time,:,:,index_channels] = scaled_video[index_time,:,:,index_channels]*self.scale_maximum_value
                        scaled_video[index_time,:,:,index_channels][scaled_video[index_time,:,:,index_channels]<0]=0
        return np.asarray(scaled_video, 'uint16')


class BandpassFilter():
    '''
    This class is intended to apply high and low bandpass filters to the video. The format of the video must be [T,Y,X,C]. This class uses **difference_of_gaussians** from skimage.filters.

    Parameters
    ----------
    video : NumPy array
        Array of images with dimensions [T,Y,X,C].
    low_pass : float, optional
        Lower pass filter intensity. The default is 0.5.
    high_pass : float, optional
        Higher pass filter intensity. The default is 10.

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


class MaskingImage():
    '''
    This class is intended to apply a mask to the video. The video format must be [T,Y,X,C], and the format of the mask must be [Y,X].

    Parameters
    ----------
    video : NumPy array
        Array of images with dimensions [T,Y,X,C].
    mask : NumPy array, with Boolean values, where 1 represents the masked area, and 0 represents the area outside the mask.
        An array of images with dimensions [Y,X].
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
            Video with zero values outside the area delimited by the mask. Array with format [T,Y,X,C].
        '''
        video_removed_mask = np.einsum('ijkl,jk->ijkl', self.video, self.mask)
        return video_removed_mask


class BeadsAlignment():
    '''
    This class is intended to detected and align spots detected in the various channels of an image with dimensions [C,Y, X]. The class returns a homography matrix that can be used to align the images captured from two different cameras during the experiment. Notice that this class should only be used for images taken from a microscope setup that uses two cameras to image various channels.

    Parameters
    ----------
    image_beads : NumPy array
        Array with a simple image with dimensions [C,Y, X].
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
        This method aligns a list of spots detected in an image with dimensions [C,Y, X] and returns a homography matrix.

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
        Array of images with dimensions [T,Y,X,C].
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
            Transformed video. Array with dimensions [T,Y,X,C].
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
        Images or videos to visualize. The format is a list of Numpy arrays where each array follows the convention [T,Y,X,C] or an image array with format [Y,X].
    list_videos_filtered : List of NumPy arrays or a single NumPy array or None
        Images or videos to visualize. The format is a list of Numpy arrays where each array follows the convention [T,Y,X,C]. The default is None.
    list_selected_particles_dataframe : pandas data frame, optional
        A pandas data frame containing the position of each spot in the image. The default is None.
    list_files_names : List of str or str, optional
        List of file names to display as the title on the image. The default is None.
    list_mask_array : List of NumPy arrays or a single NumPy array, with Boolean values, where 1 represents the masked area, and 0 represents the area outside the mask.
        An array of images with dimensions [Y,X].
    selected_channel : int, optional
        Allows the user to define the channel to visualize in the plotted images. The default is 0.
    selected_timepoint : int, optional
        Allows the user to define the time point or frame to display on the image. The default is 0.
    normalize : bool, optional
        Option to normalize the data by removing outliers. The code removes the 1 and 99 percentiles from the image. The default is False.
    individual_figure_size : int, optional
        Allows the user to change the size of each image. The default is 5.
    list_real_particle_positions : List of Pandas dataframes or a single dataframe, optional.
        A pandas data frame containing the position of each spot in the image. This dataframe is generated with class SimulatedCell, and it contains the true position for each spot. This option is only intended to be used to train algorithms for tracking and visualize real vs detected spots. The default is None.
    '''
    def __init__(self,list_videos,list_videos_filtered=None,list_selected_particles_dataframe=None,list_files_names=None,list_mask_array=None,list_real_particle_positions=None, selected_channel=0,selected_timepoint=0,normalize=False,individual_figure_size=5):
        self.particle_size =7

        self.selected_timepoint = selected_timepoint
        self.selected_channel =selected_channel
        self.individual_figure_size = individual_figure_size
        # Checkiing if the video is a list or a single video.
        if not (type(list_videos) is list):
            list_videos = [list_videos]
            self.list_videos = list_videos
            self.number_videos = 1
        else:
            self.list_videos = list_videos
            self.number_videos = len(list_videos)

        if not (type(list_mask_array) is list):
            list_mask_array = [list_mask_array]
            self.list_mask_array = list_mask_array
        else:
            self.list_mask_array = list_mask_array

        #### LIST REAL PARTICLES TO SHOW ON THE IMAGE
        if not (type(list_real_particle_positions) is list):
            list_real_particle_positions = [list_real_particle_positions]
            self.list_real_particle_positions = list_real_particle_positions
        else:
            self.list_real_particle_positions = list_real_particle_positions

        if not (type(list_files_names) is list):
            list_files_names = [list_files_names]
            self.list_files_names = list_files_names
        else:
            self.list_files_names = list_files_names

        if not (type(list_videos_filtered) is list):
            list_videos_filtered = [list_videos_filtered]
            self.list_videos_filtered = list_videos_filtered
        else:
            self.list_videos_filtered = list_videos_filtered

        if not (type(list_selected_particles_dataframe) is list):
            list_selected_particles_dataframe = [list_selected_particles_dataframe]
            self.list_selected_particles_dataframe = list_selected_particles_dataframe
        else:
            self.list_selected_particles_dataframe = list_selected_particles_dataframe

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
        # This section converts an image [Y,X] into a video with dimensions. [T,Y,X,C].
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
        NUM_COLUMNS = 8
        if ( self.list_selected_particles_dataframe[0] is None):
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
        # Plotting the cells and the detected spots
        if not ( self.list_selected_particles_dataframe[0] is None) and ( ( self.list_videos_filtered[0] is None)):
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
                if not (self.list_selected_particles_dataframe[0] is None):
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
                # main loop to mark spots ==> REAL SPOTS. USE FOR SIMULATED CELL
                number_particles  = None
                frames_part = None
                x_pos = None
                y_pos = None
                if not (self.list_real_particle_positions[0] is None):
                    selected_particles_dataframe = self.list_real_particle_positions[counter]
                    if not len (self.list_real_particle_positions[counter]) == 0:
                        number_particles = selected_particles_dataframe['particle'].nunique()
                        for k in range (0,number_particles):
                            frames_part =selected_particles_dataframe.loc[selected_particles_dataframe['particle']==selected_particles_dataframe['particle'].unique()[k]].frame.values
                            index_time = self.selected_timepoint
                            if index_time in frames_part: # plotting the circles for each detected particle at a given time point
                                index_val=np.where(frames_part == index_time)
                                x_pos=int(selected_particles_dataframe.loc[selected_particles_dataframe['particle']==selected_particles_dataframe['particle'].unique()[k]].x.values[index_val])
                                y_pos=int(selected_particles_dataframe.loc[selected_particles_dataframe['particle']==selected_particles_dataframe['particle'].unique()[k]].y.values[index_val])
                            try:
                                circle = plt.Circle((x_pos, y_pos), 2, color='orangered', fill=False)
                                ax.add_artist(circle)
                            except:
                                pass
                # Plots the mask contour on the video
                if not ( self.list_mask_array[0] is None):
                    mask_array = self.list_mask_array[index_video]
                    if len(mask_array.shape) == 3:
                        contuour_position = find_contours(mask_array[index_time,:,:], 0.8)
                    elif len(mask_array.shape) == 2:
                        contuour_position = find_contours(mask_array[:,:], 0.8)
                    temp = contuour_position[0][:,1]
                    temp2 =contuour_position[0][:,0]
                    plt.fill(temp,temp2, facecolor='none', edgecolor='yellow')
                counter +=1
        # Plotting the cells and the detected spots and the filtered video.
        if (not ( self.list_selected_particles_dataframe[0] is None)) and (not ( self.list_videos_filtered[0] is None)) :
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
                ax.imshow(self.list_videos[counter][self.selected_timepoint,:,:,self.selected_channel],cmap='gray',vmax=np.amax(self.list_videos[counter][self.selected_timepoint,:,:,self.selected_channel])*0.95)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set(title= title_str + ' Original')
                # Figure with filtered video
                ax = fig.add_subplot(gs[index_video+1])
                ax.imshow(self.list_videos_filtered[counter][self.selected_timepoint,:,:,self.selected_channel],cmap='gray',vmax=np.amax(self.list_videos[counter][self.selected_timepoint,:,:,self.selected_channel])*0.95)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set(title=title_str + ' Filtered' )
                # Figure with filtered video and marking the spots
                ax = fig.add_subplot(gs[index_video+2])
                ax.imshow(self.list_videos_filtered[counter][self.selected_timepoint,:,:,self.selected_channel],cmap='gray',vmax=np.amax(self.list_videos[counter][self.selected_timepoint,:,:,self.selected_channel])*0.95)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set(title=title_str + ' F + M' )
                # main loop to mark spots
                number_particles  = None
                frames_part = None
                x_pos = None
                y_pos = None
                # Section that plots the spots
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
                # main loop to mark spots ==> REAL SPOTS. USE FOR SIMULATED CELL
                number_particles  = None
                frames_part = None
                x_pos = None
                y_pos = None

                if not ( self.list_real_particle_positions[0] is None):
                    selected_particles_dataframe = self.list_real_particle_positions[counter]
                    if not len (self.list_real_particle_positions[counter]) == 0:
                        number_particles = selected_particles_dataframe['particle'].nunique()
                        for k in range (0,number_particles):
                            frames_part =selected_particles_dataframe.loc[selected_particles_dataframe['particle']==selected_particles_dataframe['particle'].unique()[k]].frame.values
                            index_time = self.selected_timepoint
                            if index_time in frames_part: # plotting the circles for each detected particle at a given time point
                                index_val=np.where(frames_part == index_time)
                                x_pos=int(selected_particles_dataframe.loc[selected_particles_dataframe['particle']==selected_particles_dataframe['particle'].unique()[k]].x.values[index_val])
                                y_pos=int(selected_particles_dataframe.loc[selected_particles_dataframe['particle']==selected_particles_dataframe['particle'].unique()[k]].y.values[index_val])
                            try:
                                circle = plt.Circle((x_pos, y_pos), 2, color='orangered', fill=True)
                                ax.add_artist(circle)
                            except:
                                pass
                # Plots the mask contour on the video
                if not ( self.list_mask_array[0] is None):
                    mask_array = self.list_mask_array[index_video]
                    if len(mask_array.shape) == 3:
                        contuour_position = find_contours(mask_array[index_time,:,:], 0.8)
                    elif len(mask_array.shape) == 2:
                        contuour_position = find_contours(mask_array[:,:], 0.8)
                    temp = contuour_position[0][:,1]
                    temp2 =contuour_position[0][:,0]
                    plt.fill(temp,temp2, facecolor='none', edgecolor='yellow')
                counter +=1
        fig.tight_layout()
        plt.show()


class VisualizerVideo():
    '''
    This class is intended to visualize videos as interactive widgets. This class has the option to mark the particles that previously were selected by trackPy.

    Parameters
    ----------
    list_videos : List of NumPy arrays or a single NumPy array
        Images or videos to visualize. The format is a list of Numpy arrays where each array follows the convention [T,Y,X,C] or an image array with format [Y,X].
    dataframe_particles : pandas data frame, optional
        A pandas data frame containing the position of each spot in the image. The default is None.
    list_mask_array : List of NumPy arrays or a single NumPy array, with Boolean values, where 1 represents the masked area, and 0 represents the area outside the mask.
        An array of images with dimensions [Y,X].
    show_time_projection_spots : int, optional
       Allows the user to display the projection of all detected spots for all time points on the current image. The default is 0.
    normalize : bool, optional
        Option to normalize the data by removing outliers. The code removes the 1 and 99 percentiles from the image. The default is False.
    step_size_in_sec : float, optional
        Step size in seconds. The default is 1.
    '''
    def __init__(self,list_videos,dataframe_particles=None,list_mask_array=None,show_time_projection_spots=0,normalize=False, step_size_in_sec=1):
        self.particle_size =7
        self.show_time_projection_spots = show_time_projection_spots
        # Checking if the video is a list or a single video.
        if not (type(list_videos) is list):
            list_videos = [list_videos]
            self.list_videos = list_videos
            self.number_videos = 1
        else:
            self.number_videos = len(list_videos)
        if not (type(list_mask_array) is list):
            list_mask_array = [list_mask_array]
            self.list_mask_array = list_mask_array
        else:
            self.list_mask_array = list_mask_array
        if not (type(dataframe_particles) is list):
            dataframe_particles = [dataframe_particles]
            self.dataframe_particles = dataframe_particles
        else:
            self.dataframe_particles = dataframe_particles
        self.list_number_frames = [list_videos[i].shape[0] for i in range(0,self.number_videos)]
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
            self.list_videos = self.list_videos
        n_channels = [self.list_videos[i].shape[3] for i in range(0,self.number_videos)][0]
        self.min_num_channels = np.amin((n_channels))
        self.step_size_in_sec = step_size_in_sec
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
        def figure_viewer(drop_cell,index_time_slider,drop_channel):
            video = self.list_videos[drop_cell]
            selected_particles_dataframe = self.dataframe_particles[drop_cell]
            drop_size = self.particle_size
            index_time = int(index_time_slider/self.step_size_in_sec)
            plt.figure(1)
            ax = plt.gca()
            if drop_channel == 'Ch_0':
                channel =0
                plt.imshow(video[index_time,:,:,channel],cmap='gray',vmax=np.amax(video[index_time,:,:,channel])*0.95)
            elif drop_channel == 'Ch_1':
                channel = 1
                plt.imshow(video[index_time,:,:,channel],cmap='gray',vmax=np.amax(video[index_time,:,:,channel])*0.95)
            elif drop_channel == 'Ch_2':  # vmax=np.mean(video[index_time,:,:,channel])+3*np.std(video[index_time,:,:,channel])
                channel =2
                plt.imshow(video[index_time,:,:,channel],cmap='gray',vmax=np.amax(video[index_time,:,:,channel])*0.95)
            elif drop_channel == 'Ch_3':
                channel =3
                plt.imshow(video[index_time,:,:,channel],cmap='gray',vmax=np.amax(video[index_time,:,:,channel])*0.95)
            else :
                # Converting a np.uint16 array into float.
                image = video[index_time,:,:,:].copy()
                min_image, max_image = np.min(image), np.max(image)
                image -= min_image
                image_float = np.array(image,'float32');
                image_float *= 255./(max_image-min_image);
                image = np.asarray(np.round(image_float), 'uint8')
                plt.imshow(image,vmax=np.amax(image)*0.95)
            # Plots the detected spots.
            if not ( self.dataframe_particles[0] is None):
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
                                       index_time_slider = widgets.IntSlider(min=0,max=(self.min_time_all_cells-1)*self.step_size_in_sec ,step=self.step_size_in_sec,value=0,description='Time'),
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
        Images or videos to visualize. The format is a list of Numpy arrays where each array follows the convention [T,Y,X,C] or an image array with format [Y,X].
    '''
    def __init__(self,list_videos):
        # Checking if the video is a list or a single video.
        if not (type(list_videos) is list):
            list_videos = [list_videos]
            self.list_videos = list_videos
            self.number_videos = 1
        else:
            self.number_videos = len(list_videos)
        self.list_number_frames = [list_videos[i].shape[0] for i in range(0,self.number_videos)]
        self.min_time_all_cells = min(self.list_number_frames)
        list_number_z_slices = [list_videos[i].shape[1] for i in range(0,self.number_videos)]
        self.min_z_slices= min(list_number_z_slices)
        n_channels = [list_videos[i].shape[3] for i in range(0,self.number_videos)][0]
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
        Images or videos to visualize. The format is a list of Numpy arrays where each array follows the convention [T,Y,X,C].
    list_selected_particles_dataframe : pandas data frame.
        A pandas data frame containing the position of each spot in the image.
    particle_size : int, optional
        Average particle size. The default is 5.
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
            list_videos = [list_videos]
            self.list_videos = list_videos
            self.number_videos = 1
        else:
            self.number_videos = len(list_videos)

        if not (type(list_selected_particles_dataframe) is list):
            list_selected_particles_dataframe = [list_selected_particles_dataframe]
            self.list_selected_particles_dataframe = list_selected_particles_dataframe
        self.list_number_frames = [list_videos[i].shape[0] for i in range(0,self.number_videos)]
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
        Array of images with dimensions [T,Y,X,C].
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
        self.maximum_probaility = 4
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

        if np.amax(selected_masks)==0:
            selected_masks = None
            print('No cells detected on the image')
        sys.stdout.close()
        sys.stdout = old_stdout
        return selected_masks


class CellposeFISH():
    '''
    This class is intended to detect cells in FISH images using **Cellpose**. The class uses optimization to generate the meta-parameters used by cellpose. This class segments the nucleus and cytosol for every cell detected in the image.

    Parameters
    ----------
    video : NumPy array
        Array of images with dimensions [Z,Y,X,C].
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
            Masks for every cell detected in the image. The list contains the mask arrays consisting of one or multiple Numpy arrays with format [Y,X].
        list_masks_nuclei : List of NumPy arrays or a single NumPy array
            Masks for the nuclei for every cell detected in the image. The list contains the mask arrays consisting of one or multiple Numpy arrays with format [Y,X].
        list_masks_cytosol_no_nuclei : List of NumPy arrays or a single NumPy array
            Masks for every nucleus and cytosol for every cell detected in the image. The list contains the mask arrays consisting of one or multiple Numpy arrays with format [Y,X].
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
        masks_cyto = Cellpose(video_normalized[0,:,:,self.channel_with_cytosol],diameter=self.diameter_cytosol,model_type='cyto',selection_method = 'max_cells' ).calculate_masks()
        masks_nuclei = Cellpose(video_normalized[0,:,:,self.channel_with_nucleus],diameter=self.diamter_nucleus,model_type='nuclei',selection_method = 'max_cells').calculate_masks()
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


class CellposeSelection():
    '''
    This class is intended to select the cell with the maximum area (max_area) or with the maximum number of spots (max_spots) from a masked image previously detect cells by **Cellpose**

    Parameters
    ----------
    mask : NumPy array
        Arrays with values between 0 and the number of detected cells in the image, where a number larger than zero represents the masked area for each cell, and 0 represents the area where no cells are detected. An array of images with dimensions [Y,X].
    video : NumPy array
        An array of images with dimensions [T,Y,X,C].
    slection_method : str, optional
        Options used by the optimization algorithm to select a cell based on the number of cells or the number of spots. The options are: 'max_area' or 'max_spots'. The default is 'maximum_area'.
    particle_size : int, optional
        Average particle size. The default is 5.
    selected_channel : int, optional
        Channel where the particles are detected and tracked. The default is 0.
    '''
    def __init__(self,mask,video, slection_method = 'max_area', particle_size=5, selected_channel=0):
        self.mask = mask
        self.video = video
        self.num_frames = video.shape[0]
        self.selected_channel = selected_channel      # selected channel
        self.particle_size = particle_size            # according to the documentation it must be an even number 3,5,7,9 etc.
        self.slection_method = slection_method
        self.minimal_frames = int(video.shape[0]*0.9)

    def select_mask(self):
        '''
        This method selects the cell with the maximum area (max_area) or with the maximum number of spots (max_spots) from a masked image.

        Returns
        -------
        selected_mask : NumPy array, with Boolean values, where 1 represents the masked area, and 0 represents the area outside the mask.
            An array of images with dimensions [Y,X].
        '''
        if self.slection_method == 'max_area':
            # Iterating for each mask to select the mask with the largest area.
            n_masks = np.amax(self.mask)
            if n_masks >1: # detecting if more than 1 mask are detected per cell
                size_mask = []
                for nm in range (1,n_masks+1): # iterating for each mask in a given cell. The mask has values from 0 for the background to int n, where n is the number of detected masks.
                    size_mask.append(np.sum(self.mask==nm)) # creating a list with the size of each mask
                largest_mask=np.argmax(size_mask)+1 # detecting the mask with the largest value
                temp_mask = np.zeros_like(self.mask) # making a copy of the image
                selected_mask = temp_mask + (self.mask==largest_mask) # Selecting a single mask and making this mask equal to one and the background equal to zero.
            else: # do nothing if only a single mask is detected per image.
                selected_mask = self.mask
        if self.slection_method == 'max_spots':
            # Iterating for each mask to select the mask with the largest area.
            n_masks = np.amax(self.mask)
            if n_masks >1: # detecting if more than 1 mask are detected per cell
                number_particles = []
                for nm in range (1,n_masks+1): # iterating for each mask in a given cell. The mask has values from 0 for the background to int n, where n is the number of detected masks.
                    # # Apply mask to a given time point
                    mask_copy = self.mask.copy()
                    tested_mask = np.where(mask_copy ==nm, 1, 0) # making zeros all elements outside each mask, and once all elements inside of each mask.
                    video_minimal_time = np.amin((int(self.num_frames/3),5,self.num_frames))
                    _, number_detected_trajectories, _, _ = Trackpy(self.video[0:video_minimal_time,:,:,:],tested_mask,particle_size=self.particle_size, selected_channel=self.selected_channel ,minimal_frames=self.minimal_frames, show_plot =0).perform_tracking()
                    number_particles.append(number_detected_trajectories)
                pre_selected_mask=np.argmax(number_particles)+1 # detecting the mask with the largest value
                temp_mask = np.zeros_like(self.mask,dtype = np.uint16) # making a copy of the image
                selected_mask = temp_mask + (self.mask==pre_selected_mask) # Selecting a single mask and making this mask equal to one and the background equal to zero.
            else:  # do nothing if only a single mask is detected per image.
                selected_mask = self.mask

            if np.amax(selected_mask)==0:
                selected_mask = None
                print('No mask was selected in the image.')
                # This section dilates the mask to connect areas that are isolated.
        mask_int=np.where(selected_mask> 0.5, 1, 0).astype(np.int)
        dilated_image = dilation(mask_int, square(15))
        mask_final = np.where(dilated_image> 0.5, 1, 0).astype(np.int)
        mask_final[0,:]=0;mask_final[:,0]=0;mask_final[mask_final.shape[0]-1,:]=0;mask_final[:,mask_final.shape[1]-1]=0#This line of code ensures that the corners are zeros.

        return mask_final


class Trackpy():
    '''
    This class is intended to detect spots in the video by using **Trackpy**.

    Parameters
    ----------
    video : NumPy array
        Array of images with dimensions [T,Y,X,C]. In case a FISH image is used, the format must be [Z,Y,X,C], and the user must specify the parameter FISH_image =1.
    mask : NumPy array, with Boolean values, where 1 represents the masked area, and 0 represents the area outside the mask.
        An array of images with dimensions [Y,X].
    particle_size : int, optional
        Average particle size. The default is 5.
    selected_channel : int, optional
        Channel where the particles are detected and tracked. The default is 0.
    minimal_frames : int, optional
        This parameter defines the minimal number of frames that a particle should appear on the video to be considered on the final count. The default is 5.
    optimization_iterations : int, optional
        Number of iterations for the optimization process to select the best filter. The default is 30.
    use_defaul_filter : bool, optional
        This option allows the user to use a default filter if TRUE. Else, it uses an optimization process to select the best filter for the image. The default is= 1.
    show_plot : bool, optional
        Allows the user to show a plot for the optimization process. The default is 1.
    FISH_image : bool, optional.
        This parameter allows the user to use FISH images and connect spots detected along multiple z-slices. The default is 0.
    '''
    def __init__(self, video, mask, particle_size =5, selected_channel=0, minimal_frames=5,optimization_iterations =40,use_defaul_filter = 1, FISH_image = 0, show_plot =1):
        self.num_cores = multiprocessing.cpu_count()
        self.time_points = video.shape[0]
        self.selected_channel = selected_channel
        # function that remove outliers from the video
        #video = RemoveExtrema(video, min_percentile=10, max_percentile=100,ignore_channel =None).remove_outliers()
        #video = RemoveExtrema(video, min_percentile=85, max_percentile=100,ignore_channel =None).remove_outliers()
        video = RemoveExtrema(video, min_percentile=0.5, max_percentile=99.9,ignore_channel =None).remove_outliers()

        # Function to convert the video to uint
        def img_uint(image):
            temp_vid= img_as_uint(image)
            return temp_vid
        init_video = Parallel(n_jobs=self.num_cores)(delayed(img_uint)(video[i,:,:,self.selected_channel]) for i in range(0,self.time_points))
        self.video = np.asarray(init_video)

        self.video_complete = video.copy()
        self.mask = mask
        if (particle_size % 2) ==0:
            particle_size = particle_size + 1
            print('particle_size must be an odd number, this was automatically changed to: ', particle_size)
        self.particle_size = particle_size # according to the documentation must be an even number 3,5,7,9 etc.

        if self.time_points< 10:
            self.min_time_particle_vanishes = 0
            self.max_distance_particle_moves = 5
        else:
            self.min_time_particle_vanishes = 1
            self.max_distance_particle_moves = 7

        if minimal_frames> self.time_points:     # this line is making sure that "minimal_frames" is always less or equal than the total number of frames
            minimal_frames = self.time_points
        self.minimal_frames = minimal_frames
        self.optimization_iterations = optimization_iterations
        self.show_plot = show_plot

        self.use_defaul_filter =  use_defaul_filter
        # parameters for the filters
        self.low_pass_filter = 0.1
        self.default_highpass = 10
        #self.gaussian_filter_sigma = 0.5
        #self.median_filter_size = 5
        #self.max_highpass = 20 #10
        #self.min_highpass = 0.1
        self.perecentile_intensity_slection = 70 #Not modify
        self.default_threshold_int_std =1.5
        self.MAX_NUM_STD_OPTIMIZATION = 1.5

        # This section detects if a FISH image is passed and it adjust accordingly.
        self.FISH_image = FISH_image
        if self.FISH_image ==1:
            self.min_time_particle_vanishes = 0
            self.max_distance_particle_moves = 1
            self.minimal_frames = minimal_frames
    def perform_tracking(self):
        '''
        This method

        Returns
        -------
        trackpy_dataframe : pandas data frame.
            Pandas data frame from trackpy with fields [x, y, mass, size, ecc, signal, raw_mass, ep, frame, particle].
        number_particles : int.
            The total number of detected particles in the data frame.
        video_filtered : np.uint16.
            Filtered video resulting from the bandpass process. Array with format [T,Y,X,C].
        '''
        num_detected_particles = np.zeros((self.optimization_iterations),dtype=np.float)
        #vector_highpass_filters = np.linspace(self.min_highpass,self.max_highpass,self.optimization_iterations).astype(np.uint16)
        min_int_vector = np.round(np.linspace(0,self.MAX_NUM_STD_OPTIMIZATION,self.optimization_iterations),1) # range of std to test for optimization
        percentile = self.perecentile_intensity_slection
        # Function that rounds an array to the nearest multiple of five
        def round_to_5(num):
            if num> 3:
                if num % 5 == 0:
                    return int(num)
                elif num % 5 < 2.5:
                     return int(num - num % 5)
                else:
                     return int(num + (5 - num % 5))
            else:
                return int(0)
        # Funtions with the bandpass and gaussian filters
        def bandpass_filter (image, lowfilter, highpass):
            temp_vid= difference_of_gaussians(image, lowfilter, highpass,truncate=3.0)
            return img_as_uint(temp_vid)
        def gaussian_filter(image, sigma=0.1):
            temp_image= img_as_float64(image)
            filtered_image = gaussian(temp_image, sigma=sigma, output=None, mode='nearest', cval=0, multichannel=None, preserve_range=True, truncate=4.0)
            return img_as_uint(filtered_image)
        # non-linear filter
        def nl_filter(image):
            temp_image= img_as_float64(image)
            sigma_est = np.mean(estimate_sigma(temp_image, multichannel=True))
            denoise_img = denoise_nl_means(temp_image, h=sigma_est, fast_mode=True,patch_size=10, patch_distance=3, multichannel=False)
            return img_as_uint(denoise_img)

        def median_filter (image, size=1):
            temp_image= img_as_float64(image)
            filtered_image = nd.median_filter(temp_image,size=size)
            return img_as_uint(filtered_image)

        if self.use_defaul_filter ==1: # This section uses a default value for the filter size.
            temp_vid_dif_filter = Parallel(n_jobs=self.num_cores)(delayed(bandpass_filter)(self.video[i,:,:],self.low_pass_filter,self.default_highpass) for i in range(0,self.time_points))
            temp_video_bp_filtered = np.asarray(temp_vid_dif_filter)
            #temp_video_bp_filtered =self.video
            video_removed_mask = np.einsum('ijk,jk->ijk', temp_video_bp_filtered, self.mask)
            f_init = tp.locate(video_removed_mask[0,:,:], self.particle_size, minmass=0, max_iterations=100,preprocess=False, percentile=percentile)
            try:
                min_int_in_video = np.amax( (0, np.mean(f_init.mass.values) + self.default_threshold_int_std *np.std(f_init.mass.values)))
            except:
                min_int_in_video = 0
            try:
                f = tp.batch(video_removed_mask[:,:,:],self.particle_size, minmass=min_int_in_video, processes='auto',max_iterations=1000,preprocess=False, percentile=percentile)
                t = tp.link_df(f,(self.max_distance_particle_moves,self.max_distance_particle_moves), memory=self.min_time_particle_vanishes, adaptive_stop = 1,link_strategy='auto') # tp.link_df(data_frame, min_distance_particle_moves, min_time_particle_vanish).
                t_sel = tp.filter_stubs(t, self.minimal_frames)  # selecting trajectories that appear in at least 10 frames.
                number_particles = t_sel['particle'].nunique()
                trackpy_dataframe = t_sel
            except:
                number_particles = []
                trackpy_dataframe = None
        else: # This section uses optimization to select the optimal value for the filter size.
            #for index_p,highpass_filters in enumerate(vector_highpass_filters):
            temp_vid_dif_filter = Parallel(n_jobs=self.num_cores)(delayed(bandpass_filter)(self.video[i,:,:],self.low_pass_filter,self.default_highpass) for i in range(0,self.time_points))
            temp_video_bp_filtered = np.asarray(temp_vid_dif_filter)
            video_removed_mask = np.einsum('ijk,jk->ijk', temp_video_bp_filtered, self.mask)
            f_init = tp.locate(video_removed_mask[0,:,:], self.particle_size, minmass=0, max_iterations=100,preprocess=False, percentile=percentile)
            for index_p,int_optimization_value in enumerate(min_int_vector):
                #### OPTIMIZATION VARIABLE:  min_int_in_video
                try:
                    min_int_in_video = np.amax( (0,  np.mean(f_init.mass.values) + int_optimization_value *np.std(f_init.mass.values)))
                    f = tp.batch(video_removed_mask[:,:,:],self.particle_size, minmass=min_int_in_video, processes='auto',max_iterations=1000,preprocess=False, percentile=percentile)
                    t = tp.link_df(f,(self.max_distance_particle_moves,self.max_distance_particle_moves), memory=self.min_time_particle_vanishes, adaptive_stop = 1,link_strategy='auto') # tp.link_df(data_frame, min_distance_particle_moves, min_time_particle_vanish).
                    t_sel = tp.filter_stubs(t, self.minimal_frames)  # selecting trajectories that appear in at least 10 frames.
                    num_detected_particles[index_p] = t_sel['particle'].nunique()
                except:
                    num_detected_particles[index_p] = 0

            if self.optimization_iterations <=10:
                window_size = 2
            elif (self.optimization_iterations >10) and (self.optimization_iterations <= 20):
                window_size = 3
            else:
                window_size = 4
            conv_num_detected_particles = np.round( np.convolve(num_detected_particles, np.ones(window_size)/window_size, mode='same'))
            rounded_num_particules = np.array([round_to_5(conv_num_detected_particles[x]) for x in range(len(conv_num_detected_particles))],dtype=np.float32)
            try:
                threshold = 3 # threshold to reject if
                num_detected_particles_with_nans = rounded_num_particules.copy()
                num_detected_particles_with_nans[num_detected_particles_with_nans<=threshold]=np.nan
                mode_detected_particles = sps.mode(num_detected_particles_with_nans, nan_policy='omit')[0][0] # calculaitng the mode of nonzero num_detected_particles.
                center_modes = int (len(np.where(num_detected_particles_with_nans == mode_detected_particles)[0])/2)-1 #
                if center_modes<0:
                    center_modes =0
                index_containin_central_mode = np.where(num_detected_particles_with_nans == mode_detected_particles)[0][center_modes]
                selected_int_optimized = min_int_vector[index_containin_central_mode]
                #print(index_containin_central_mode)
                #print(selected_int_optimized)
                #selected_filter = vector_highpass_filters[index_containin_central_mode] # selecting the intensity for the first instance where the mode appears in the vector_intensities
                #temp_vid_dif_filter = Parallel(n_jobs=self.num_cores)(delayed(bandpass_filter)(self.video[i,:,:],self.low_pass_filter,self.default_highpass) for i in range(0,self.time_points))
                #temp_vid_dif_filter = Parallel(n_jobs=self.num_cores)(delayed(bandpass_filter)(self.video[i,:,:],self.low_pass_filter,selected_filter) for i in range(0,self.time_points))
                #temp_video_bp_filtered = np.asarray(temp_vid_dif_filter)
                #temp_video_bp_filtered = self.video
                #video_removed_mask = np.einsum('ijk,jk->ijk', temp_video_bp_filtered, self.mask)
                #f_init = tp.locate(video_removed_mask[0,:,:], self.particle_size, minmass=0, max_iterations=100,preprocess=False, percentile=percentile)
                #### OPTIMIZATION VARIABLE:  min_int_in_video
                min_int_in_video =np.amax( (0,  np.mean(f_init.mass.values) + selected_int_optimized *np.std(f_init.mass.values)))
                f = tp.batch(video_removed_mask[:,:,:],self.particle_size, minmass=min_int_in_video, processes='auto',max_iterations=1000,preprocess=False, percentile=percentile)
                t = tp.link_df(f,(self.max_distance_particle_moves,self.max_distance_particle_moves), memory=self.min_time_particle_vanishes, adaptive_stop = 1,link_strategy='auto') # tp.link_df(data_frame, min_distance_particle_moves, min_time_particle_vanish).
                t_sel = tp.filter_stubs(t, self.minimal_frames)  # selecting trajectories that appear in at least 10 frames.
                number_particles = t_sel['particle'].nunique()
                trackpy_dataframe = t_sel
            except:
                 number_particles = 0
                 trackpy_dataframe = None
                 self.show_plot = 0
                 selected_int_optimized =0
            if self.show_plot ==1:
                plt.figure(figsize=(7,7))
                plt.plot(min_int_vector, num_detected_particles, 'w-', linewidth=2, label = 'real')
                plt.plot(min_int_vector[1:-1], conv_num_detected_particles[1:-1], 'g-', linewidth = 1, label = 'smoothed')
                plt.plot([selected_int_optimized,selected_int_optimized],[np.amin(num_detected_particles),np.amax(num_detected_particles)],'r-',linewidth=2,label='Automatic selection')
                #plt.plot([vector_intensities[0],vector_intensities[-1]],[threshold,threshold],'k--',linewidth=1,label='min. selection threshold')
                plt.legend(loc='best')
                plt.xlabel('minimal value (int.)')
                plt.ylabel('Detected Spots')
                plt.title('Optimization: detecting the longest plateau')
                plt.show()
                print('')
                print('The number of detected trajectories is: ', number_particles)
                print('')
                print('')
        video_filtered =  self.video_complete.copy()
        video_filtered[:,:,:,self.selected_channel] = video_removed_mask
        return trackpy_dataframe, int(number_particles), video_filtered



class Intensity():
    '''
    This class is intended to calculate the intensity in the detected spots.

    Parameters
    ----------
    video : NumPy array
        Array of images with dimensions [T,Y,X,C].
    particle_size : int, optional
        Average particle size. The default is 5.
    trackpy_dataframe : pandas data frame or None (if not given).
        Pandas data frame from trackpy with fields [x, y, mass, size, ecc, signal, raw_mass, ep, frame, particle]. The default is None
    spot_positions_movement : NumPy array  or None (if not given).
        Array of images with dimensions [T, S, x_y_positions].  The default is None
    method : str, optional
        Method to calculate intensity the options are : 'total_intensity' , 'disk_donut' and 'gaussian_fit'. The default is 'disk_donut'.
    step_size : float, optional
        Frame rate in seconds. The default is 1 frame per second.
    show_plot : bool, optional
        Allows the user to show a plot for the optimization process. The default is 1.
    '''
    def __init__(self,video,particle_size=5, trackpy_dataframe=None,spot_positions_movement=None, method ='disk_donut',step_size=1,show_plot=1):
        #print(spot_positions_movement)
        if particle_size <5:
            particle_size = 5 # minimal size allowed for detection

        if (particle_size % 2) ==0:
            particle_size = particle_size + 1
            print('particle_size must be an odd number, this was automatically changed to: ', particle_size)
        self.video = video
        self.trackpy_dataframe = trackpy_dataframe
        #self.disk_size = int(particle_size/2) # size of the half of the crop
        #self.crop_size = int(particle_size/2)+2

        self.disk_size = int(particle_size/2) # size of the half of the crop
        self.crop_size = self.disk_size+10

        self.show_plot = show_plot
        self.method = method # options are : 'total_intensity' , 'disk_donut' and 'gaussian_fit'
        self.particle_size = particle_size
        self.spot_positions_movement = spot_positions_movement
        # the number of spots is determined by the dataframe or numpy array passed by the user
        if not ( trackpy_dataframe is None):
            self.n_particles = self.trackpy_dataframe['particle'].nunique()
        if not ( spot_positions_movement is None):
            self.n_particles = spot_positions_movement.shape[1]
        if (trackpy_dataframe is None) and (spot_positions_movement is None):
            print ('Error a trackpy_dataframe or spot_positions_movement should be given')
            raise
        self.step_size =step_size
    def calculate_intensity(self):
        '''
        This method calculates the spot intensity.

        Returns
        -------
        dataframe_particles : pandas dataframe
            Dataframe with fields [cell_number, particle, frame, red_int_mean, green_int_mean, blue_int_mean, red_int_std, green_int_std, blue_int_std, x, y].
        array_intensities_mean : Numpy array
            Array with dimensions [S,T,C].
        time_vector : Numpy array
            1D array.
        mean_intensities: Numpy array
            Array with dimensions [S,T,C].
        std_intensities : Numpy array
            Array with dimensions [S,T,C].
        mean_intensities_normalized : Numpy array
            Array with dimensions [S,T,C].
        std_intensities_normalized : Numpy array
            Array with dimensions [S,T,C].

        '''
        time_points, number_channels  = self.video.shape[0], self.video.shape[3]
        array_intensities_mean = np.zeros((self.n_particles,time_points,number_channels))
        array_intensities_std = np.zeros((self.n_particles,time_points,number_channels))
        def gaussian_fit(test_im):
            size_spot = test_im.shape[0]
            image_flat = test_im.ravel()
            def gaussian_function(size_spot, offset,sigma):
                ax = np.linspace(-(size_spot - 1) / 2., (size_spot - 1) / 2., size_spot)
                xx, yy = np.meshgrid(ax, ax)
                kernel =  offset *(np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma)))
                return kernel.ravel()
            p0 = (np.mean(image_flat) ,int(size_spot/2))
            popt, _ = curve_fit(gaussian_function,size_spot, image_flat, p0=p0)
            spot_intensity_gaussian = popt[0] # Amplitude
            spot_intensity_gaussian_std = popt[1]
            return spot_intensity_gaussian, spot_intensity_gaussian_std

        # def disk_donut(test_im,disk_size):
        #     #plt.figure()
        #     #plt.imshow(test_im)
        #     #plt.colorbar()
        #     #plt.show()
        #     center_coordenates = int(test_im.shape[0]/2)
        #     #print('image_size',test_im.shape[1])
        #     #print('disk',disk_size)
        #     #print('center',center_coordenates)
        #     # mean intensity in disk
        #     image_in_disk = test_im[center_coordenates-disk_size:center_coordenates+disk_size+1,center_coordenates-disk_size:center_coordenates+disk_size+1]
        #     mean_intensity_disk = np.mean(image_in_disk)
        #     # mean intensity in donut.  The center is set to zeros and then the mean is calculated ignoring the zeros.
        #     recentered_image_donut = test_im.copy()
        #     mean_outside = [np.mean( recentered_image_donut[0,:]),np.mean(recentered_image_donut[-1,:]),np.mean(recentered_image_donut[:,0]),np.mean(recentered_image_donut[:,-1])]
        #     mean_outside = np.mean(mean_outside)
        #     #print(mean_outside)
        #     recentered_image_donut[center_coordenates-disk_size:center_coordenates+disk_size+1,center_coordenates-disk_size:center_coordenates+disk_size+1] = 0
        #     # plt.figure()
        #     # plt.imshow(recentered_image_donut)
        #     # plt.colorbar()
        #     # plt.show()
        #     spot_intensity_disk_donut_std = recentered_image_donut[recentered_image_donut!=0].std() # mean calculation ignoring zeros
        #     spot_intensity_disk_donut_std= np.nan_to_num(spot_intensity_disk_donut_std)
        #     # substracting background minus center intensity
        #     #print((mean_intensity_disk - mean_intensity_donut), mean_intensity_disk , mean_intensity_donut)
        #     #spot_intensity_disk_donut = np.array( mean_intensity_disk - mean_intensity_donut, dtype=np.float32)
        #     spot_intensity_disk_donut = np.array( mean_intensity_disk - mean_outside, dtype=np.float32)
        #     spot_intensity_disk_donut[spot_intensity_disk_donut<0]=0
        #     spot_intensity_disk_donut[np.isnan(spot_intensity_disk_donut)] = 0 # replacing nans with zero
        #     return spot_intensity_disk_donut, spot_intensity_disk_donut_std

        def disk_donut(test_im,disk_size):
                # Function that calculates intensity using the disk and donut method
                center_coordenates = int(test_im.shape[0]/2)
                recentered_image_donut = test_im.copy().astype(np.float32)
                # mean intensity in disk
                image_in_disk = test_im[center_coordenates-disk_size:center_coordenates+disk_size+1,center_coordenates-disk_size:center_coordenates+disk_size+1]
                #mean_intensity_disk = image_in_disk.mean() # mean calculation ignoring zeros
                mean_intensity_disk = np.amax(image_in_disk.flatten())
                #mean_intensity_disk = np.mean(image_in_disk)
                spot_intensity_disk_donut_std = np.std(image_in_disk)
                # mean intensity in donut.  The center is set to zeros and then the mean is calculated ignoring the zeros.
                #recentered_image_donut[center_coordenates-disk_size:center_coordenates+disk_size+1,center_coordenates-disk_size:center_coordenates+disk_size+1] = 0
                #mean_intensity_donut = recentered_image_donut[recentered_image_donut!=0].mean() # mean calculation ignoring zeros
                recentered_image_donut[center_coordenates-disk_size:center_coordenates+disk_size+1,center_coordenates-disk_size:center_coordenates+disk_size+1] = np.nan
                mean_intensity_donut = np.nanmedian(recentered_image_donut.flatten()) # mean calculation ignoring zeros
                # substracting background minus center intensity
                spot_intensity_disk_donut = np.array( mean_intensity_disk - mean_intensity_donut, dtype=np.float32)
                spot_intensity_disk_donut[spot_intensity_disk_donut<0]=0
                spot_intensity_disk_donut[np.isnan(spot_intensity_disk_donut)] = 0 # replacing nans with zero
                return spot_intensity_disk_donut, spot_intensity_disk_donut_std

        def return_crop(image,x_pos,y_pos,crop_size):
            # function that recenters the spots
            crop_image = image[y_pos-(crop_size):y_pos+(crop_size+1),x_pos-(crop_size):x_pos+(crop_size+1)]
            return crop_image
        # Section that marks particles if a numpy array with spot positions is passed.

        if not ( self.spot_positions_movement is None):
            frames_part = self.spot_positions_movement.shape[0]
            for k in range (0,self.n_particles):
                for j in range(0,frames_part):
                    for i in range(0,number_channels):
                        x_pos = self.spot_positions_movement[j,k,1]
                        y_pos = self.spot_positions_movement[j,k,0]
                        #x_pos = self.spot_positions_movement[j,k,0]
                        #y_pos = self.spot_positions_movement[j,k,1]
                        crop_image = return_crop(self.video[j,:,:,i],x_pos,y_pos,self.crop_size) # NOT RECENTERING IMAGE
                        if self.method == 'disk_donut':
                            #print('self_crop',self.crop_size)
                            #print('self_particle_size',self.particle_size)
                            #print('--------------------------')
                            array_intensities_mean[k,j,i], array_intensities_std[k,j,i] = disk_donut(crop_image,self.disk_size)
                        elif self.method == 'total_intensity':
                            array_intensities_mean[k,j,i] = np.amax((0,np.mean(crop_image)))# mean intensity in the crop
                            array_intensities_std[k,j,i] = np.amax((0,np.std(crop_image)))# std intensity in the crop
                        elif self.method == 'gaussian_fit':
                            array_intensities_mean[k,j,i], array_intensities_std[k,j,i] = gaussian_fit(crop_image)# disk_donut(crop_image,self.disk_size)
        if not ( self.trackpy_dataframe is None):
            for k in range (0,self.n_particles):
                frames_part =self.trackpy_dataframe.loc[self.trackpy_dataframe['particle']==self.trackpy_dataframe['particle'].unique()[k]].frame.values
                for j in range(0,len(frames_part)):
                    for i in range(0,number_channels):
                        current_frame = frames_part[j]
                        try:
                            x_pos=int(self.trackpy_dataframe.loc[self.trackpy_dataframe['particle']==self.trackpy_dataframe['particle'].unique()[k]].x.values[j])
                            y_pos=int(self.trackpy_dataframe.loc[self.trackpy_dataframe['particle']==self.trackpy_dataframe['particle'].unique()[k]].y.values[j])
                        except:
                            x_pos=int(self.trackpy_dataframe.loc[self.trackpy_dataframe['particle']==self.trackpy_dataframe['particle'].unique()[k]].x.values[frames_part[0]])
                            y_pos=int(self.trackpy_dataframe.loc[self.trackpy_dataframe['particle']==self.trackpy_dataframe['particle'].unique()[k]].y.values[frames_part[0]])
                        if self.method == 'disk_donut':
                            crop_image = return_crop(self.video[j,:,:,i],x_pos,y_pos,self.crop_size) # NOT RECENTERING IMAGE
                            array_intensities_mean[k,current_frame,i],array_intensities_std[k,current_frame,i] = disk_donut(crop_image,self.disk_size)
                        elif self.method == 'total_intensity':
                            crop_image = return_crop(self.video[j,:,:,i],x_pos,y_pos,self.crop_size) # NOT RECENTERING IMAGE
                            array_intensities_mean[k,current_frame,i] = np.amax((0,np.mean(crop_image))) # mean intensity in image
                            array_intensities_std[k,current_frame,i] = np.amax((0,np.std(crop_image))) # std intensity in image
                        elif self.method == 'gaussian_fit':
                            crop_image = return_crop(self.video[j,:,:,i],x_pos,y_pos,self.crop_size) # NOT RECENTERING IMAGE
                            array_intensities_mean[k,current_frame,i],array_intensities_std[k,current_frame,i] = gaussian_fit(crop_image)# disk_donut(crop_image,self.disk_size)
        # Calculate mean intensities.
        mean_intensities = np.nanmean(array_intensities_mean, axis = 0, dtype=np.float32)
        std_intensities = np.nanstd(array_intensities_mean, axis = 0, dtype=np.float32)
        # Calculate mean intensities normalized
        array_mean_intensities_normalized = np.zeros_like(array_intensities_mean)*nan
        for k in range (0,self.n_particles):
                for i in range(0,number_channels):
                    if np.nanmax( array_intensities_mean[k,:,i]) > 0:
                        array_mean_intensities_normalized[k,:,i] = array_intensities_mean[k,:,i]/ np.nanmax( array_intensities_mean[k,:,i])
        mean_intensities_normalized = np.nanmean(array_mean_intensities_normalized, axis = 0, dtype=np.float32)
        mean_intensities_normalized=np.nan_to_num(mean_intensities_normalized)
        std_intensities_normalized = np.nanstd(array_mean_intensities_normalized, axis = 0, dtype=np.float32)
        std_intensities_normalized=np.nan_to_num(std_intensities_normalized)
        time_vector = np.arange(0,time_points,1)*self.step_size
        if self.show_plot ==1:
            fig, ax = plt.subplots(3,1, figsize=(16, 4))
            for id in range (0,self.n_particles):
                frames_part =self.trackpy_dataframe.loc[self.trackpy_dataframe['particle']==self.trackpy_dataframe['particle'].unique()[id]].frame.values
                ax[0].plot(self.trackpy_dataframe.loc[self.trackpy_dataframe['particle']==self.trackpy_dataframe['particle'].unique()[id]].frame.values, array_intensities_mean[id,frames_part,0],'r')
                ax[1].plot(self.trackpy_dataframe.loc[self.trackpy_dataframe['particle']==self.trackpy_dataframe['particle'].unique()[id]].frame.values, array_intensities_mean[id,frames_part,1],'g')
                ax[2].plot(self.trackpy_dataframe.loc[self.trackpy_dataframe['particle']==self.trackpy_dataframe['particle'].unique()[id]].frame.values, array_intensities_mean[id,frames_part,2],'b')
            ax[0].set(title='Trajectories: Intensity vs Time')
            ax[2].set(xlabel='Time (a.u.)')
            ax[1].set(ylabel='Intensity (a.u.)')
            ax[0].set_xlim([0, time_points-1])
            ax[1].set_xlim([0, time_points-1])
            ax[2].set_xlim([0, time_points-1])
            ax[0].set_xticks([]);ax[1].set_xticks([]);
            plt.show()
            fig, ax = plt.subplots(3,1, figsize=(16, 4))
            for id in range (0,self.n_particles):
                time_vector = np.arange(0,time_points,1)*self.step_size
                ax[0].plot(time_vector,mean_intensities[:,0],'r')
                ax[1].plot(time_vector,mean_intensities[:,1],'g')
                ax[2].plot(time_vector,mean_intensities[:,2],'b')
                ax[0].fill_between(time_vector, mean_intensities[:,0] - std_intensities[:,0], mean_intensities[:,0] + std_intensities[:,0], color='lightgray', alpha=0.9)
                ax[1].fill_between(time_vector, mean_intensities[:,1] - std_intensities[:,1], mean_intensities[:,1] + std_intensities[:,1], color='lightgray', alpha=0.9)
                ax[2].fill_between(time_vector, mean_intensities[:,2] - std_intensities[:,2], mean_intensities[:,2] + std_intensities[:,2], color='lightgray', alpha=0.9)
            ax[0].set(title='Mean values')
            ax[2].set(xlabel='Time (a.u.)')
            ax[1].set(ylabel='Intensity (a.u.)')
            ax[0].set_xlim([0, time_points-1])
            ax[1].set_xlim([0, time_points-1])
            ax[2].set_xlim([0, time_points-1])
            ax[0].set_xticks([]);ax[1].set_xticks([]);
            plt.show()
            fig, ax = plt.subplots(3,1, figsize=(16, 4))
            for id in range (0,self.n_particles):
                time_vector = np.arange(0,time_points,1)*self.step_size
                ax[0].plot(time_vector,mean_intensities_normalized[:,0],'r')
                ax[1].plot(time_vector,mean_intensities_normalized[:,1],'g')
                ax[2].plot(time_vector,mean_intensities_normalized[:,2],'b')
                ax[0].fill_between(time_vector, mean_intensities_normalized[:,0] - std_intensities_normalized[:,0], mean_intensities_normalized[:,0] + std_intensities_normalized[:,0], color='lightgray', alpha=0.9)
                ax[1].fill_between(time_vector, mean_intensities_normalized[:,1] - std_intensities_normalized[:,1], mean_intensities_normalized[:,1] + std_intensities_normalized[:,1], color='lightgray', alpha=0.9)
                ax[2].fill_between(time_vector, mean_intensities_normalized[:,2] - std_intensities_normalized[:,2], mean_intensities_normalized[:,2] + std_intensities_normalized[:,2], color='lightgray', alpha=0.9)
            ax[0].set(title='Mean values normalized by max intensity of each trajectory')
            ax[2].set(xlabel='Time (a.u.)')
            ax[1].set(ylabel='Intensity (a.u.)')
            ax[0].set_xlim([0, time_points-1])
            ax[1].set_xlim([0, time_points-1])
            ax[2].set_xlim([0, time_points-1])
            ax[0].set_xticks([]);ax[1].set_xticks([]);
            plt.show()
        # Initialize a dataframe
        init_dataFrame = {'cell_number': [],
            'particle': [],
            'frame': [],
            'red_int_mean': [],
            'green_int_mean': [],
            'blue_int_mean': [],
            'red_int_std': [],
            'green_int_std': [],
            'blue_int_std': [],
            'x': [],
            'y': []}
        dataframe_particles = pd.DataFrame(init_dataFrame)
        # Iterate for each spot and save time courses in the data frame
        counter = 0
        for id in range (0,self.n_particles):
            # Loop that populates the dataframes
            if not ( self.trackpy_dataframe is None):
                temporal_frames_vector = np.around(self.trackpy_dataframe.loc[self.trackpy_dataframe['particle']==self.trackpy_dataframe['particle'].unique()[id]].frame.values)  # time_(sec)
                temporal_x_position_vector = self.trackpy_dataframe.loc[self.trackpy_dataframe['particle']==self.trackpy_dataframe['particle'].unique()[id]].x.values
                temporal_y_position_vector = self.trackpy_dataframe.loc[self.trackpy_dataframe['particle']==self.trackpy_dataframe['particle'].unique()[id]].y.values

            else:
                counter_time_vector = np.arange(0,time_points,1)
                #time_vector = np.arange(0,time_points,1)*self.step_size
                temporal_frames_vector = counter_time_vector
                temporal_x_position_vector = self.spot_positions_movement[:,id,1]
                temporal_y_position_vector = self.spot_positions_movement[:,id,0]

            temporal_red_vector = array_intensities_mean[id, temporal_frames_vector,0] # red
            temporal_green_vector = array_intensities_mean[id,temporal_frames_vector,1] # green
            temporal_blue_vector = array_intensities_mean[id,temporal_frames_vector,2] # blue
            temporal_red_vector_std = array_intensities_std[id, temporal_frames_vector,0] # red
            temporal_green_vector_std = array_intensities_std[id,temporal_frames_vector,1] # green
            temporal_blue_vector_std = array_intensities_std[id,temporal_frames_vector,2] # blue
            temporal_spot_number_vector = np.around([counter] * len(temporal_frames_vector))
            temporal_cell_number_vector = np.around([0] * len(temporal_frames_vector))
                        # Section that append the information for each spots
            temp_data_frame = {'cell_number': temporal_cell_number_vector,
                'particle': temporal_spot_number_vector,
                'frame': temporal_frames_vector*self.step_size,
                'red_int_mean': temporal_red_vector,
                'green_int_mean': temporal_green_vector,
                'blue_int_mean': temporal_blue_vector,
                'red_int_std': temporal_red_vector_std,
                'green_int_std': temporal_green_vector_std,
                'blue_int_std': temporal_blue_vector_std,
                'x': temporal_x_position_vector,
                'y': temporal_y_position_vector}
            counter +=1
            temp_DataFrame = pd.DataFrame(temp_data_frame)
            dataframe_particles = dataframe_particles.append(temp_DataFrame, ignore_index=True)
            dataframe_particles = dataframe_particles.astype({"cell_number": int, "particle": int, "frame": int}) # specify data type as integer for some columns

        return dataframe_particles, array_intensities_mean, time_vector, mean_intensities,std_intensities, mean_intensities_normalized, std_intensities_normalized

class SimulatedCell():
    '''
    This class takes a base video, and it draws simulated spots on top of the image. The intensity for each simulated spot is proportional to the stochastic simulation given by the user.

    Parameters
    ----------
    base_video : NumPy array
        Array of images with dimensions [T,Y,X,C].
    video_for_mask : NumPy array, optional
        Array of images with dimensions [T,Y,X,C]. Use only if the base video has been edited, and an empty video is needed to calculate the mask. The default is None.
    number_spots : int
        Number of simulated spots in the cell. The default is 10.
    number_frames : int
        The number of frames or time points to simulate. In seconds. The default is 20.
    step_size : float, optional
        Step size for the simulation. In seconds. The default is 1.
    diffusion_coefficient : float, optional
        The diffusion coefficient for the particles' Brownian motion. The default is 0.01.
    simulated_trajectories_ch0 : NumPy array, optional
        An array of simulated trajectories with dimensions [S,T] for channel 0. The default is None and indicates that the intensity will be generated, drawing random numbers in a range.
    size_spot_ch0 : int, optional
        Spot size in pixels for channel 0. The default is 5.
    spot_sigma_ch0 : int, optional
        Sigma value for the simulated spots in channel 0, the units are pixels. The default is 2.
    simulated_trajectories_ch1 : NumPy array, optional
        An array of simulated trajectories with dimensions [S,T] for channel 1. The default is None and indicates that the intensity will be generated, drawing random numbers in a range.
    size_spot_ch1 : int, optional
        Spot size in pixels for channel 1. The default is 5.
    spot_sigma_ch1 : int, optional
        Sigma value for the simulated spots in channel 1, the units are pixels. The default is 2.
    simulated_trajectories_ch2 : NumPy array, optional
        An array of simulated trajectories with dimensions [S,T] for channel 2. The default is None and indicates that the intensity will be generated, drawing random numbers in a range.
    size_spot_ch2 : int, optional
        Spot size in pixels for channel 2. The default is 5.
    spot_sigma_ch2 : int, optional
        Sigma value for the simulated spots in channel 2, the units are pixels. The default is 2.
    ignore_ch0 : bool, optional
        A flag that ignores channel 0 returning a NumPy array filled with zeros. The default is 0.
    ignore_ch1 : bool, optional
        A flag that ignores channel 1 returning a NumPy array filled with zeros. The default is 1.
    ignore_ch2 : bool, optional
        A flag that ignores channel 2 returning a NumPy array filled with zeros. The default is 1.
    save_as_tif_uint8 : bool, optional
        If true, it generates and saves a uint8 (low) quality image tif file for the simulation. The default is 0.
    save_as_tif : bool, optional
        If true, it generates and saves a uint16 (High) quality image tif file for the simulation. The default is 0.
    save_as_gif : bool, optional
        If true, it generates and saves a gif animation for the simulation. The default is 0.
    save_dataframe : bool, optional
        If true, it generates and saves a pandas dataframe with the simulation. Dataframe with fields [cell_number, particle, frame,red_int_mean, green_int_mean, blue_int_mean, red_int_std, green_int_std, blue_int_std, x, y]. The default is 0.
    saved_file_name : str, optional
        The file name for the simulated cell output files (tif images, gif images, data frames). The default is 'temp'.
    create_temp_folder : bool, optional
        Creates a folder with the simulation output. The default is True.
    intensity_calculation_method : str, optional
        Method to calculate intensity the options are : 'total_intensity' , 'disk_donut' and 'gaussian_fit'. The default is 'disk_donut'.
    using_for_multiplexing : bool, optional
        Flag that indicates that multiple genes are simulated per cell.
    min_int_multiplexing : float or None, optional
        Indicates the minimal SSA value for all simulated genes in a multplexing experiment. The default is None.
    max_int_multiplexing : float or None, optional
        Indicates the maximum SSA value for all simulated genes in a multplexing experiment. The default is None.
    perform_video_augmentation : bool, optional
        If true, it performs random rotations the initial video. The default is 1.
    frame_selection_empty_video : str, optional
        Method to select the frames from the empty video, the options are : 'constant' , 'shuffle' and 'loop'. The default is 'shuffle'.
    '''
    def __init__(self, base_video,video_for_mask = None, number_spots=10, number_frames=20, step_size =1, diffusion_coefficient =0.01, simulated_trajectories_ch0=None, size_spot_ch0=5, spot_sigma_ch0=2, simulated_trajectories_ch1=[0], size_spot_ch1=5, spot_sigma_ch1=2, simulated_trajectories_ch2=[0], size_spot_ch2=5, spot_sigma_ch2=2, ignore_ch0=0,ignore_ch1=1, ignore_ch2=1,save_as_tif_uint8 =0, save_as_tif=0, save_as_gif=0, save_dataframe=0, saved_file_name='temp',create_temp_folder = True,intensity_calculation_method='disk_donut',using_for_multiplexing=0,min_int_multiplexing=None, max_int_multiplexing=None,perform_video_augmentation=0,frame_selection_empty_video='shuffle'):
        if (perform_video_augmentation ==1) and (video_for_mask is None):
            base_video = AugmentationVideo(base_video).random_rotation()
        self.intensity_calculation_method = intensity_calculation_method
        MAXIMUM_INTENSITY_IN_BASE_VIDEO = 1000
        if using_for_multiplexing == 0:
            base_video = RemoveExtrema(base_video, min_percentile=0, max_percentile=99.8,ignore_channel =None).remove_outliers()
            base_video = ScaleIntensity(base_video, scale_maximum_value=MAXIMUM_INTENSITY_IN_BASE_VIDEO,ignore_channel =None).apply_scale()
        self.base_video = base_video
        if not (video_for_mask is None):
            video_for_mask = RemoveExtrema(video_for_mask, min_percentile=0, max_percentile=99.8,ignore_channel =None).remove_outliers()
            video_for_mask = ScaleIntensity(video_for_mask, scale_maximum_value=MAXIMUM_INTENSITY_IN_BASE_VIDEO,ignore_channel =None).apply_scale()
            self.video_for_mask = video_for_mask
            video_for_mask = video_for_mask
        else:
            self.video_for_mask = base_video
            video_for_mask = base_video
        self.number_spots = number_spots
        self.number_frames = number_frames
        self.step_size = step_size
        self.diffusion_coefficient = diffusion_coefficient
        self.image_size = [base_video.shape[1],base_video.shape[2]]
        self.simulated_trajectories_ch0 = simulated_trajectories_ch0
        self.size_spot_ch0 = size_spot_ch0
        self.spot_sigma_ch0 = spot_sigma_ch0
        self.simulated_trajectories_ch1 = simulated_trajectories_ch1
        self.size_spot_ch1 = size_spot_ch1
        self.spot_sigma_ch1 = spot_sigma_ch1
        self.simulated_trajectories_ch2 = simulated_trajectories_ch2
        self.size_spot_ch2 = size_spot_ch2
        self.spot_sigma_ch2 = spot_sigma_ch2
        self.ignore_ch0 = ignore_ch0
        self.ignore_ch1 = ignore_ch1
        self.ignore_ch2 = ignore_ch2
        self.n_channels = 3
        self.z_slices =1
        self.time_vector = np.linspace(0, number_frames*step_size, num=number_frames)
        self.save_as_tif_uint8 = save_as_tif_uint8
        self.save_as_tif = save_as_tif
        self.save_as_gif = save_as_gif
        self.save_dataframe = save_dataframe
        self.saved_file_name = saved_file_name
        self.create_temp_folder = create_temp_folder
        self.min_int_multiplexing = min_int_multiplexing
        self.max_int_multiplexing = max_int_multiplexing
        self.frame_selection_empty_video = frame_selection_empty_video
        # The following two constants are weights used to define a range of intensities for the simulated spots.
        self.MIN_INTENSITY_SPOT_WEIGHT = 1.2 # 1.05 lower weight that multiplays the mean intensity value in the image to define the simulated spot intensity.
        self.MAX_INTENSITY_SPOT_WEIGHT = 3 # 1.6 higher weight that multiplays the mean intensity value in the image to define the simulated spot intensity.
        self.MAX_STD_INT_IMAGE = 4 # maximum number of standard deviations avobe the mean that are permited to draw an spot.
        # This function is intended to detect the mask and then reduce the mask by a given percentage. This reduction ensures that the simulated spots are inclosed inside the cell.
        def mask_reduction(polygon_array, percentage_reduction= 0.2):
            # Reducing the size of the mask to plot only inside the cell.
            x_reduction = percentage_reduction+0.1
            y_reduction = percentage_reduction+0.1
            x_coord = [[i][0][1] for i in polygon_array]
            y_coord = [[i][0][0] for i in polygon_array]
            # Center of mask
            center_value = 0.5
            x_center = center_value* min(x_coord) + center_value * max(x_coord)
            y_center = center_value* min(y_coord) + center_value* max(y_coord)
            # Reducing the size of the mask
            reduced_x = [(i - x_center) * (1 - x_reduction) + x_center for i in x_coord]
            reduced_y = [(i - y_center) * (1 - y_reduction) + y_center for i in y_coord]
            # create list of new coordinates
            mask_coordinates =  np.vstack((reduced_y,reduced_x)).T
            mask_coordinates.shape
            return mask_coordinates
        # section that uses cellpose to calculate the mask
        selected_image = self.video_for_mask[0,:,:,1] # selecting for the mask the first time point
        selected_masks = Cellpose(selected_image,num_iterations=10, channels=[0],diameter=200,model_type='cyto',selection_method = 'max_area').calculate_masks() # options are 'max_area' or 'max_cells'
        if np.amax(selected_masks) ==0:
            print('Error, no masks were found on the image')
            raise
        selected_mask  = CellposeSelection(selected_masks,selected_image, slection_method = 'max_area').select_mask()
        # section that converts the mask in contours and resduces the size of the mask to ensure the spots are generated inside the cell.
        try:
            contours = np.array(find_contours(selected_mask, 0.5),dtype = int)
        except:
            # this section extends the mask to conect isolated areas in the mask
            plt.imshow(selected_mask)
            dilated_image = dilation(selected_mask, square(20))
            dilated_image[0,:]=0;dilated_image[:,0]=0;dilated_image[dilated_image.shape[0]-1,:]=0;dilated_image[:,dilated_image.shape[1]-1]=0#This line of code ensures that the corners are zeros.
            contours = np.array(find_contours(dilated_image, 0.5),dtype = int)
        polygon_array = contours[0]
        self.polygon_array = mask_reduction(polygon_array, percentage_reduction= 0.2)
        # Removing mask to the video
        self.video_removed_mask = MaskingImage(base_video,selected_mask).apply_mask()

    def make_simulation (self):
        '''
        This method generates the simulated cell.

        Returns
        -------
        tensor_video : NumPy array uint16
            Array with dimensions [T,Y,X,C]
        tensor_for_image_j : NumPy array uint16
            Array with dimensions [T,Z,C,Y,X]
        spot_positions_movement : NumPy array
            Array with dimensions [T,Spots, position(y,x)]
        tensor_mean_intensity_in_figure : NumPy array, np.float
            Array with dimensions [T,Spots]
        tensor_std_intensity_in_figure : NumPy array, np.float
            Array with dimensions [T,Spots]
        dataframe_particles : pandas dataframe
            Dataframe with fields [cell_number, particle, frame, red_int_mean, green_int_mean, blue_int_mean, red_int_std, green_int_std, blue_int_std, x, y].
        '''
        def calculate_intensity_in_figure(tensor_image,time_vector,number_spots, spot_positions_movement, size_spot):
            # Function that allows the user to
            if (size_spot % 2) ==0:
                size_spot = size_spot + 1
            disk_size =  int(size_spot/2)
            crop_size = disk_size+4
            tensor_mean_intensity_in_figure = np.zeros((len(time_vector),number_spots),dtype='float')
            tensor_std_intensity_in_figure = np.zeros((len(time_vector),number_spots),dtype='float')

            def gaussian_fit(test_im):
                size_spot = test_im.shape[0]
                image_flat = test_im.ravel()
                def gaussian_function(size_spot, offset,sigma):
                    ax = np.linspace(-(size_spot - 1) / 2., (size_spot - 1) / 2., size_spot)
                    xx, yy = np.meshgrid(ax, ax)
                    kernel =  offset *(np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma)))
                    return kernel.ravel()
                p0 = (np.mean(image_flat) ,int(size_spot/2))
                popt, _ = curve_fit(gaussian_function,size_spot, image_flat, p0=p0)
                spot_intensity_gaussian = popt[0] # Amplitude
                std_intensity_gaussian = popt[1] # sigma
                return spot_intensity_gaussian, std_intensity_gaussian

            def disk_donut(test_im,disk_size):
                # Function that calculates intensity using the disk and donut method
                center_coordenates = int(test_im.shape[0]/2)
                # mean intensity in disk
                image_in_disk = test_im[center_coordenates-disk_size:center_coordenates+disk_size+1,center_coordenates-disk_size:center_coordenates+disk_size+1]
                mean_intensity_disk = np.mean(image_in_disk)
                std_intensity_disk = np.std(image_in_disk)
                # mean intensity in donut.  The center is set to zeros and then the mean is calculated ignoring the zeros.
                recentered_image_donut = test_im.copy()
                recentered_image_donut[center_coordenates-disk_size:center_coordenates+disk_size+1,center_coordenates-disk_size:center_coordenates+disk_size+1] = 0
                mean_intensity_donut = recentered_image_donut[recentered_image_donut!=0].mean() # mean calculation ignoring zeros
                # substracting background minus center intensity
                spot_intensity_disk_donut = np.array( mean_intensity_disk - mean_intensity_donut, dtype=np.float32)
                spot_intensity_disk_donut[spot_intensity_disk_donut<0]=0
                spot_intensity_disk_donut[np.isnan(spot_intensity_disk_donut)] = 0 # replacing nans with zero
                return spot_intensity_disk_donut, std_intensity_disk
            def return_crop(image,x_pos,y_pos,crop_size):
                # Function that return the crop from a given image and coordenates
                crop_image = image[y_pos-(crop_size):y_pos+(crop_size+1),x_pos-(crop_size):x_pos+(crop_size+1)]
                return crop_image
            for t_p,_ in enumerate(time_vector):
                center_positions_vector = spot_positions_movement[t_p,:,:]
                temp_tensor_image = tensor_image[t_p,:,:]
                for point_index in range(0,len(center_positions_vector)):
                    center_position = center_positions_vector[point_index]
                    crop_with_extra_area = return_crop(temp_tensor_image,center_position[1],center_position[0],crop_size)
                    #mean_intensity_donut = np.mean(crop_with_extra_area)
                    #tensor_mean_intensity_in_figure[t_p,point_index],tensor_std_intensity_in_figure[t_p,point_index]  = gaussian_fit(crop_with_extra_area)
                    #if tensor_mean_intensity_in_figure[t_p,point_index] < 0:
                    tensor_mean_intensity_in_figure[t_p,point_index],tensor_std_intensity_in_figure[t_p,point_index]  = disk_donut(crop_with_extra_area,disk_size)
                    if tensor_mean_intensity_in_figure[t_p,point_index] < 0:
                        tensor_mean_intensity_in_figure[t_p,point_index] = 0
            return tensor_mean_intensity_in_figure, tensor_std_intensity_in_figure

        def make_replacement_pixelated_spots(matrix_background, center_positions_vector, size_spot, spot_sigma, using_ssa, simulated_trajectories_time_point,min_SSA_value,max_SSA_value):
            #This funciton is intended to replace a kernel gaussian matrix for each spot position. The kernel gaussian matrix is scaled with the values obtained from the SSA o with the values given in a range.
            if size_spot%2 ==0:
                size_spot = size_spot+1
            # Copy the matrix_background
            pixelated_image = matrix_background.copy()
            pixelated_image_no_spots = matrix_background.copy()
            # Defining constant values
            MIN_INTENSITY_ALL_SPOTS = 4000 # basal value for each spot
            MAX_INTENSITY_ALL_SPOTS = 5000 # maximum allowed intensity for a given spot
            MAX_INTENSITY_IN_UINT16 = 65535 # maximum value in a unint16 image

            for point_index in range(0,len(center_positions_vector)):
                # Section that creates the Gaussian Kernel Matrix
                ax = np.linspace(-(size_spot - 1) / 2., (size_spot - 1) / 2., size_spot)
                xx, yy = np.meshgrid(ax, ax)
                kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(spot_sigma))
                #kernel = kernel / np.sum(kernel) # nrrmalized to one
                # creating a position for each spot
                center_position = center_positions_vector[point_index]
                pixel_size_arround_spot = size_spot + 5
                #try:
                # this section calculates the basal intensity arround an spots
                temp_int_arround_spot = pixelated_image_no_spots[center_position[0]-int(pixel_size_arround_spot/2): center_position[0]+int(pixel_size_arround_spot/2)+1 ,center_position[1]-int(pixel_size_arround_spot/2): center_position[1]+int(pixel_size_arround_spot/2)+1 ]
                temp_int_arround_spot.flatten()
                temp_mean = np.mean(temp_int_arround_spot)
                #temp_std = np.std(temp_int_arround_spot)
                #dist_mean = abs(temp_int_arround_spot - temp_mean)
                #max_dev = 4
                #idx_inside_elements = dist_mean < max_dev * temp_std
                #inside_elements = temp_int_arround_spot[idx_inside_elements]
                #mean_int_arround_spot = np.mean(inside_elements)
                mean_int_arround_spot = temp_mean
                # except:
                #     mean_int_arround_spot = pixelated_image_no_spots[center_position[0],center_position[1]]
                if using_ssa == 1 :
                    min_int = np.amin((mean_int_arround_spot *self.MIN_INTENSITY_SPOT_WEIGHT,MIN_INTENSITY_ALL_SPOTS))
                    max_int = np.amin((mean_int_arround_spot *self.MAX_INTENSITY_SPOT_WEIGHT,MAX_INTENSITY_ALL_SPOTS))  # 65535 is the maximum value for np.uint16   # bast value 1.5
                    if max_int == MAX_INTENSITY_IN_UINT16:
                         print('Warning! Spot genrated in an area with high intensity, mean_int = ',mean_int_arround_spot)
                    # Scaling the spot kernel matrix with the intensity value form the SSA.
                    int_ssa = (simulated_trajectories_time_point[point_index] - min_SSA_value) / (max_SSA_value-min_SSA_value) # intensity normalized to min and max values in the SSA
                    int_tp = min_int + (int_ssa * (max_int-min_int)) # Scaling to the min and max values in the image
                    spot_intensity = np.amax((0, int_tp))
                    kernel_value_intensity = (kernel*spot_intensity)
                else:
                    min_int = np.amin((mean_int_arround_spot *self.MIN_INTENSITY_SPOT_WEIGHT,MIN_INTENSITY_ALL_SPOTS))
                    max_int = np.amin((mean_int_arround_spot *self.MAX_INTENSITY_SPOT_WEIGHT,MAX_INTENSITY_ALL_SPOTS))
                    # Scaling the spot kernel matrix with a mean value obtained from uniformly random distribution.
                    spot_intensity = int(random.uniform(min_int,max_int))
                    kernel_value_intensity = (kernel*spot_intensity)
                # Setting thresholds
                kernel_value_intensity[kernel_value_intensity < min_int] = min_int # making spots with less intensity than the background equal to the background.
                kernel_value_intensity[kernel_value_intensity > MAX_INTENSITY_ALL_SPOTS] = MAX_INTENSITY_ALL_SPOTS # making spots with more intensity than maximum allowed intensity equal to max_int
                # Defining the current area in the matrix that will be replaced by the spot kernel.
                pixelated_image[center_position[0]-int(size_spot/2): center_position[0]+int(size_spot/2)+1 ,center_position[1]-int(size_spot/2): center_position[1]+int(size_spot/2)+1 ] = kernel_value_intensity
            return pixelated_image # final_image



        def make_spots_movement(polygon_array, number_spots, time_vector, step_size, image_size, diffusion_coefficient,internal_base_video=None):
            # Function that creates the simulated spots inside a given polygon
            #print('mean int in cell',np.mean(base_video))
            #print('std int in cell',np.std(base_video))
            path = mpltPath.Path(polygon_array)
            initial_points_in_polygon = np.zeros((number_spots,2), dtype = 'int')
            counter_number_spots = 0
            conter_security = 0
            MAX_ITERATIONS = 5000
            min_position = 20 # minimal position in pixels
            max_position = image_size[1]-20 # maximal position in pixels
            while (counter_number_spots < number_spots) and (conter_security<MAX_ITERATIONS):
                test_points = (int(random.uniform(min_position,max_position)), int(random.uniform(min_position,max_position)))
                # testing if the spot is located in an area of high intensity?
                if not ( internal_base_video is None):
                    selected_image= internal_base_video
                    #MAX_INTENSITY_TO_DRAW_SPOT = 3500
                    #max_allowed_int_image = np.amin( (np.mean(selected_image) + MAX_STD*np.std(selected_image),MAX_INTENSITY_TO_DRAW_SPOT))
                    max_allowed_int_image = np.mean(selected_image) + self.MAX_STD_INT_IMAGE*np.std(selected_image)
                    min_allowed_int_image = np.mean(selected_image) #- self.MAX_STD_INT_IMAGE*np.std(selected_image)
                    size_spot =5
                    pixel_size_arround_spot = size_spot + 5
                    temp_crop_arround_spot = selected_image[test_points[0]-int(pixel_size_arround_spot/2): test_points[0]+int(pixel_size_arround_spot/2)+1 ,test_points[1]-int(pixel_size_arround_spot/2): test_points[1]+int(pixel_size_arround_spot/2)+1 ]
                    mean_int_tested_spot = np.mean(temp_crop_arround_spot)
                    if (mean_int_tested_spot>min_allowed_int_image) and (mean_int_tested_spot<max_allowed_int_image):
                        int_test =1
                    else:
                        int_test =0
                else:
                    int_test =1
                conter_security +=1
                if path.contains_point(test_points) == 1 and int_test ==1:
                    counter_number_spots+=1
                    initial_points_in_polygon[counter_number_spots-1,:] = np.asarray(test_points)
                if conter_security>MAX_ITERATIONS:
                    print('error generating spots')
            ## Brownian motion
            # scaling factor for Brownian motion.
            brownian_movement = math.sqrt(2*diffusion_coefficient*step_size)
            # Prealocating memory
            y_positions = np.array(initial_points_in_polygon[:,0],dtype='int') #  x_position for selected spots inside the polygon
            x_positions = np.array(initial_points_in_polygon[:,1],dtype='int') #  y_position for selected spots inside the polygon
            temp_Position_y = np.zeros_like(y_positions,dtype='int')
            temp_Position_x = np.zeros_like(x_positions,dtype='int')
            newPosition_y = np.zeros_like(y_positions,dtype='int')
            newPosition_x = np.zeros_like(x_positions,dtype='int')
            spot_positions_movement = np.zeros((len(time_vector),number_spots,2),dtype='int')
            # Main loop that computes the random motion and new spot positions
            for t_p,_ in enumerate(time_vector):
                for i_p in range (0, number_spots):
                    if t_p == 0:
                        temp_Position_y[i_p]= y_positions[i_p]
                        temp_Position_x[i_p]= x_positions[i_p]
                    else:
                        temp_Position_y[i_p]= newPosition_y[i_p] + int(brownian_movement * np.random.randn(1))
                        temp_Position_x[i_p]= newPosition_x[i_p] + int(brownian_movement * np.random.randn(1))
                    while path.contains_point((temp_Position_y[i_p], temp_Position_x[i_p])) == 0:
                        temp_Position_y[i_p]= newPosition_y[i_p]
                        temp_Position_x[i_p]= newPosition_x[i_p]
                    newPosition_y[i_p]= temp_Position_y[i_p]
                    newPosition_x[i_p]= temp_Position_x[i_p]
                spot_positions_movement [t_p,:,:]= np.vstack((newPosition_y, newPosition_x)).T
                #print(spot_positions_movement)
            return spot_positions_movement # vector with dimensions (time, spot, y, x )

        def make_simulation(base_video_slected_channel,masked_video_slected_channel, spot_positions_movement, time_vector, polygon_array, image_size,size_spot, spot_sigma, simulated_trajectories,frame_selection_empty_video):
            # Main function that makes the simulated cell by calling multiple function.
            temp_image = masked_video_slected_channel[0,:,:]
            temp_image_nonzeros = temp_image.copy()
            temp_image_nonzeros.flatten
            base_video_slected_channel_copy = base_video_slected_channel.copy()
            # Calculate initial background intensity for each spot.
            tensor_image = np.zeros((len(time_vector),image_size[0],image_size[1]),dtype=np.uint32)
            len_empty_video = base_video_slected_channel.shape[0]
            empty_video_index = np.arange(len_empty_video, dtype=np.int32)
            # array that stores the frames to be selected from the empty vector
            if frame_selection_empty_video ==  'constant': # selects the first time point
                index_frame_selection = np.zeros((len(time_vector)), dtype=np.int32)
            if frame_selection_empty_video ==  'loop':
                index_frame_selection = np.resize(empty_video_index,len(time_vector))
            if frame_selection_empty_video ==  'shuffle':
                index_frame_selection = np.random.randint(0,high=len_empty_video, size=len(time_vector), dtype=np.int32)
            for t_p,_ in enumerate(time_vector):
                matrix_background = base_video_slected_channel_copy[index_frame_selection[t_p],:,:]
                if not ( simulated_trajectories is None):
                    using_ssa = 1
                    simulated_trajectories_tp = simulated_trajectories[:,t_p]
                    if  not (self.min_int_multiplexing is None):
                        max_SSA_value=self.max_int_multiplexing
                        min_SSA_value=self.min_int_multiplexing
                    else:
                        max_SSA_value = simulated_trajectories.max()
                        min_SSA_value = simulated_trajectories.min()
                else:
                    using_ssa = 0
                    simulated_trajectories_tp = 0
                    max_SSA_value = 0
                    min_SSA_value =0
                # Making the pixelated spots
                tensor_image[t_p,:,:] = make_replacement_pixelated_spots(matrix_background, spot_positions_movement[t_p,:,:], size_spot, spot_sigma, using_ssa, simulated_trajectories_tp,min_SSA_value,max_SSA_value)
                #counter += 1
            return tensor_image

        # Create the spots for all channels. Return array with 3-dimensions: T,Sop, XY-Coord
        spot_positions_movement = make_spots_movement(self.polygon_array, self.number_spots, self.time_vector, self.step_size, self.image_size, self.diffusion_coefficient,self.base_video[0,:,:,1])


        # This section of the code runs the for each channel
        if self.ignore_ch0 == 0:
            tensor_image_ch0 = make_simulation(self.base_video[:,:,:,0], self.video_removed_mask[:,:,:,0], spot_positions_movement, self.time_vector, self.polygon_array, self.image_size, self.size_spot_ch0, self.spot_sigma_ch0, self.simulated_trajectories_ch0,self.frame_selection_empty_video)
            tensor_mean_intensity_in_figure_ch0, tensor_std_intensity_in_figure_ch0 = calculate_intensity_in_figure(tensor_image_ch0,self.time_vector,self.number_spots, spot_positions_movement, self.size_spot_ch0)
        else:
            tensor_image_ch0 = np.zeros((self.number_frames,self.image_size[0],self.image_size[1]),dtype=np.uint16)
            tensor_mean_intensity_in_figure_ch0 = np.zeros(( len(self.time_vector),self.number_spots),dtype=np.uint16)
            tensor_std_intensity_in_figure_ch0 = np.zeros((len(self.time_vector),self.number_spots),dtype=np.uint16)
        # Channel 1
        if self.ignore_ch1 == 0:
            tensor_image_ch1 = make_simulation(self.base_video[:,:,:,1],self.video_removed_mask[:,:,:,1],spot_positions_movement, self.time_vector, self.polygon_array, self.image_size, self.size_spot_ch1, self.spot_sigma_ch1, self.simulated_trajectories_ch1,self.frame_selection_empty_video)
            tensor_mean_intensity_in_figure_ch1, tensor_std_intensity_in_figure_ch1 = calculate_intensity_in_figure(tensor_image_ch1,self.time_vector,self.number_spots, spot_positions_movement, self.size_spot_ch1)
        else:
            tensor_image_ch1 = np.zeros((self.number_frames,self.image_size[0],self.image_size[1]),dtype=np.uint16)
            tensor_mean_intensity_in_figure_ch1 = np.zeros((len(self.time_vector),self.number_spots),dtype=np.uint16)
            tensor_std_intensity_in_figure_ch1 = np.zeros((len(self.time_vector),self.number_spots),dtype=np.uint16)
        # Channel 2
        if self.ignore_ch2 == 0:
            tensor_image_ch2 = make_simulation(self.base_video[:,:,:,2], self.video_removed_mask[:,:,:,2], spot_positions_movement, self.time_vector, self.polygon_array, self.image_size, self.size_spot_ch2, self.spot_sigma_ch2, self.simulated_trajectories_ch2,self.frame_selection_empty_video)
            tensor_mean_intensity_in_figure_ch2, tensor_std_intensity_in_figure_ch2 = calculate_intensity_in_figure(tensor_image_ch2,self.time_vector,self.number_spots, spot_positions_movement, self.size_spot_ch2)
        else:
            tensor_image_ch2 = np.zeros((self.number_frames,self.image_size[0],self.image_size[1]),dtype=np.uint16)
            tensor_mean_intensity_in_figure_ch2 = np.zeros((len(self.time_vector),self.number_spots),dtype=np.uint16)
            tensor_std_intensity_in_figure_ch2 = np.zeros((len(self.time_vector),self.number_spots),dtype=np.uint16)
        # Creating a tensor with the final video as a tensor with 4D the order TXYC
        tensor_video =  np.zeros((len(self.time_vector),self.image_size[0],self.image_size[1], self.n_channels),dtype=np.uint16)
        tensor_video [:,:,:,0] =  tensor_image_ch0
        tensor_video [:,:,:,1] =  tensor_image_ch1
        tensor_video [:,:,:,2] =  tensor_image_ch2
    # This section saves the tensor as a imagej array 5D. In the orderd :  TZCYX
        tensor_for_image_j = np.zeros((len(self.time_vector), self.z_slices, self.n_channels, self.image_size[0], self.image_size[1]),dtype=np.uint16)
        for i,_ in enumerate(self.time_vector):
            for ch in range(0, 3):
                if ch ==0:
                    tensor_for_image_j [i,0,0,:,:] = tensor_image_ch0 [i,:,:]
                if ch ==1:
                    tensor_for_image_j [i,0,1,:,:] = tensor_image_ch1 [i,:,:]
                if ch ==2:
                    tensor_for_image_j [i,0,2,:,:] = tensor_image_ch2 [i,:,:]
        # Creating tensors with real intensity values in the order TSC
        tensor_mean_intensity_in_figure = np.zeros((len(self.time_vector),self.number_spots, self.n_channels),dtype=np.uint16)
        tensor_mean_intensity_in_figure[:,:,0] = tensor_mean_intensity_in_figure_ch0
        tensor_mean_intensity_in_figure[:,:,1] = tensor_mean_intensity_in_figure_ch1
        tensor_mean_intensity_in_figure[:,:,2] = tensor_mean_intensity_in_figure_ch2
        # The same for the std
        tensor_std_intensity_in_figure = np.zeros((len(self.time_vector),self.number_spots, self.n_channels),dtype=np.uint16)
        tensor_std_intensity_in_figure[:,:,0] = tensor_std_intensity_in_figure_ch0
        tensor_std_intensity_in_figure[:,:,1] = tensor_std_intensity_in_figure_ch1
        tensor_std_intensity_in_figure[:,:,2] = tensor_std_intensity_in_figure_ch2
        
        if (self.save_as_tif_uint8==1) or (self.save_as_gif==1):
            if self.create_temp_folder == True:
                save_to_path = pathlib.Path().absolute().joinpath('temp')
                if not os.path.exists(str(save_to_path)):
                    os.makedirs(str(save_to_path))
                print ("The output is saved in the directory: " , str(save_to_path))
            else:
                save_to_path== pathlib.Path().absolute()
            tensor_video_copy = tensor_video.copy()
            normalized_tensor = np.zeros_like(tensor_video_copy,dtype='uint8')
            num_images_for_gif = tensor_video_copy.shape[0]
            for ch in range(0,tensor_video_copy.shape[3]):
                for i in range(0,num_images_for_gif):
                    # CONVERSION TO UINT
                    image = tensor_video_copy[i,:,:,ch].copy()
                    min_image, max_image = np.min(image), np.max(image);
                    image -= min_image;
                    image_float = np.array(image,'float32');
                    image_float *= 255./(max_image-min_image);
                    image = np.asarray(np.round(image_float), 'uint8')
                    normalized_tensor[i,:,:,ch] = image
            if self.save_as_tif_uint8==1:
                tifffile.imwrite(str(save_to_path.joinpath(self.saved_file_name+'_unit8_'+'.tif')), normalized_tensor)

            if self.save_as_gif==1:
                # Saving the simulation as a gif. Complete image
                with imageio.get_writer(str(save_to_path.joinpath(self.saved_file_name+'_unit8_'+'.gif')), mode='I') as writer:
                    for i in range(0,num_images_for_gif):
                        image = normalized_tensor[i,:,:,0:1]
                        writer.append_data(image)
        if self.save_as_tif==1:
            if self.create_temp_folder == True:
                save_to_path = pathlib.Path().absolute().joinpath('temp')
                if not os.path.exists(str(save_to_path)):
                    os.makedirs(str(save_to_path))
                print ("The output is saved in the directory: " , str(save_to_path))
            else:
                save_to_path= pathlib.Path().absolute()
            tifffile.imwrite(str(save_to_path.joinpath(self.saved_file_name+'.tif')), tensor_video)
        dataframe_particles, _, _, _,_, _, _ = Intensity(tensor_video,particle_size=self.size_spot_ch0,spot_positions_movement=spot_positions_movement,method=self.intensity_calculation_method,step_size =self.step_size,show_plot=0 ).calculate_intensity()

        if self.save_dataframe==1:
            if self.create_temp_folder == True:
                save_to_path = pathlib.Path().absolute().joinpath('temp')
                if not os.path.exists(str(save_to_path)):
                    os.makedirs(str(save_to_path))
                print ("The output is saved in the directory: " , str(save_to_path))
            else:
                save_to_path=pathlib.Path().absolute()
            dataframe_particles.to_csv(str(save_to_path.joinpath(self.saved_file_name +'_df'+ '.csv')), index = True)
        return tensor_video , tensor_for_image_j , spot_positions_movement, tensor_mean_intensity_in_figure, tensor_std_intensity_in_figure, dataframe_particles


class SimulatedCellMultiplexing ():
    '''
    This class takes a base video and simulates a multiplexing experiment, and it draws simulated spots on top of the image. The intensity for each simulated spot is proportional to the stochastic simulation given by the user.

    Parameters
    ----------
    inial_video :  NumPy array
        Array of images with dimensions [T,Y,X,C].
    list_gene_sequences : List of strings
        List where every element is a gene sequence file.
    list_number_spots : List of int
        List where every element represents the number of spots to simulate for each gene.
    list_target_channels : List of int with a range from 0 to 2
        List where every element represents the specific channel where the spots will be located.
    list_diffusion_coefficients : List of floats
        List where every element represents the diffusion coefficient for every gene.
    list_label_names : List of str
        List where every element contains the label for each gene.
    list_elongation_rates : List of floats
        List where every element represents the elongation rate for every gene.
    list_initation_rates : List of floats
        List where every element represents the initation rate for every gene.
    simulation_time_in_sec : int
        The simulation time in seconds. The default is 20.
    step_size_in_sec : float, optional
        Step size for the simulation. In seconds. The default is 1.
    save_as_tif : bool, optional
        If true, it generates and saves a uint16 (High) quality image tif file for the simulation. The default is 0.
    save_dataframe : bool, optional
        If true, it generates and saves a pandas dataframe with the simulation. Dataframe with fields [cell_number, particle, frame,red_int_mean, green_int_mean, blue_int_mean, red_int_std, green_int_std, blue_int_std, x, y]. The default is 0.
    saved_file_name : str, optional
        The file name for the simulated cell output files (tif images, gif images, data frames). The default is 'temp'.
    create_temp_folder : bool, optional
        Creates a folder with the simulation output. The default is True.
    cell_number : int, optional
        Cell number used as an index for the data frame. The default is 0.
    save_as_gif : bool, optional
        If true, it generates and saves a .gif with the simulation. The default is 0.
    perform_video_augmentation : bool, optional
        If true, it performs random rotations the initial video. The default is 1.
    frame_selection_empty_video : str, optional
        Method to select the frames from the empty video, the options are : 'constant' , 'shuffle' and 'loop'. The default is 'shuffle'.
    '''
    def __init__(self,inial_video,list_gene_sequences,list_number_spots,list_target_channels,list_diffusion_coefficients,list_label_names,list_elongation_rates,list_initation_rates,simulation_time_in_sec,step_size_in_sec,save_as_tif, save_dataframe, saved_file_name,create_temp_folder,cell_number=0,save_as_gif=0,perform_video_augmentation=1,frame_selection_empty_video='shuffle'):
        if perform_video_augmentation ==1:
            self.inial_video = AugmentationVideo(inial_video).random_rotation()
        else:
            self.inial_video = inial_video
        self.list_gene_sequences = list_gene_sequences
        self.list_number_spots = list_number_spots
        self.list_target_channels = list_target_channels
        self.list_diffusion_coefficients = list_diffusion_coefficients
        self.list_label_names = list_label_names
        self.list_elongation_rates = list_elongation_rates
        self.list_initation_rates = list_initation_rates
        self.simulation_time_in_sec = simulation_time_in_sec
        self.step_size_in_sec = step_size_in_sec
        self.number_genes = len(list_gene_sequences)
        self.save_as_tif = save_as_tif
        self.save_dataframe = save_dataframe
        self.saved_file_name = saved_file_name
        self.create_temp_folder = create_temp_folder
        self.cell_number = cell_number
        self.save_as_gif=save_as_gif
        self.frame_selection_empty_video = frame_selection_empty_video

    def make_simulation (self):
        '''
        Method that runs the simulations for the multiplexing experiment.

        Returns
        -------
        tensor_video : NumPy array uint16
            Array with dimensions [T,Y,X,C]
        dataframe_particles : pandas dataframe
            Dataframe with fields [cell_number, particle, frame, red_int_mean, green_int_mean, blue_int_mean, red_int_std, green_int_std, blue_int_std, x, y].
        list_ssa : List of NumPy arrays
            List of numpy arrays with the stochastic simulations for each gene. The format is [S,T], where the dimentsions are S = spots and T = time.

        '''
        # FUNCTION THAT RUNS THE SSA IN rSNAPsim
        def rsnapsim_ssa(gene_file,ke ,ki,simulation_time_in_sec=100,n_traj=20, frame_rate =self.step_size_in_sec):
            poi_strs, poi_objs, tagged_pois,raw_seq = rss.seqmanip.open_seq_file(gene_file)
            gene_obj = tagged_pois['1'][0]
            gene_obj.ke = ke
            rss.solver.protein = gene_obj #pass the protein object
            t_burnin = 1000
            t = np.linspace(0,t_burnin+simulation_time_in_sec, int((t_burnin+simulation_time_in_sec)/frame_rate) )   # ask Will how to pass the step_size. (start, stop, num)
            ssa_solution = rss.solver.solve_ssa(gene_obj.kelong, t, ki=ki, kt = ke, low_memory=False,record_stats=False,n_traj=n_traj)
            time_ssa = ssa_solution.time[int(t_burnin/frame_rate)-1:-1]
            time_ssa = time_ssa-t_burnin
            ssa_int =  ssa_solution.intensity_vec[0,int(t_burnin/frame_rate)-1 :-1,:].T
            return time_ssa, ssa_int

        # Wrapper for the simulated cell
        def wrapper_SimulatedCell (base_video,video_for_mask=None, ssa=None, target_channel=1, diffusion_coefficient=0.05, step_size=self.step_size_in_sec,spot_size=5, spot_sigma=2,intensity_calculation_method='disk_donut',using_for_multiplexing=0,min_int_multiplexing =0 , max_int_multiplexing=0,save_as_gif=0,frame_selection_empty_video=self.frame_selection_empty_video):
            if target_channel ==0:
                ignore_ch0=0; ignore_ch1=1; ignore_ch2=1
            elif target_channel ==1:
                ignore_ch0=0; ignore_ch1=0; ignore_ch2=1
            elif target_channel ==2:
                ignore_ch0=0; ignore_ch1=1; ignore_ch2=0
            number_spots_per_cell = ssa.shape[0]
            tensor_video , _ , _, _, _, DataFrame_particles_intensities = SimulatedCell( base_video=base_video,video_for_mask=video_for_mask, number_spots = number_spots_per_cell, number_frames=ssa.shape[1], step_size=step_size, diffusion_coefficient =diffusion_coefficient, simulated_trajectories_ch0=None, size_spot_ch0=spot_size, spot_sigma_ch0=spot_sigma, simulated_trajectories_ch1=ssa, size_spot_ch1=spot_size, spot_sigma_ch1=spot_sigma, simulated_trajectories_ch2=ssa, size_spot_ch2=spot_size, spot_sigma_ch2=spot_sigma, ignore_ch0=ignore_ch0,ignore_ch1=ignore_ch1, ignore_ch2=ignore_ch2,save_as_tif_uint8=0,save_as_tif =0,save_as_gif=save_as_gif, save_dataframe=0,create_temp_folder = 1,intensity_calculation_method=intensity_calculation_method,using_for_multiplexing=using_for_multiplexing,min_int_multiplexing=min_int_multiplexing, max_int_multiplexing=max_int_multiplexing,frame_selection_empty_video=frame_selection_empty_video).make_simulation()
            DataFrame_particles_intensities['cell_number'] = DataFrame_particles_intensities['cell_number'].replace([0],self.cell_number)
            return tensor_video, DataFrame_particles_intensities  # [cell_number, particle, frame, red_int_mean, green_int_mean, blue_int_mean, red_int_std, green_int_std, blue_int_std, x, y].
        # Runs the SSA and the simulated cell functions
        list_ssa = []
        list_min_ssa =[]
        list_max_ssa =[]
        for i in range(0,self.number_genes):
            _ , ssa_solution = rsnapsim_ssa(self.list_gene_sequences[i],ke =self.list_elongation_rates[i], ki=self.list_initation_rates[i],simulation_time_in_sec=self.simulation_time_in_sec,n_traj=self.list_number_spots[i])
            list_ssa.append(ssa_solution)
            list_min_ssa.append(ssa_solution.min())
            list_max_ssa.append(ssa_solution.max())
        # Creating the videos
        list_DataFrame_particles_intensities= []
        for i in range(0,self.number_genes):
            if i == 0 :
                tensor_video , DataFrame_particles_intensities = wrapper_SimulatedCell(self.inial_video, video_for_mask = self.inial_video, ssa=list_ssa[i], target_channel=self.list_target_channels[i], diffusion_coefficient=self.list_diffusion_coefficients[i],min_int_multiplexing =min(list_min_ssa) , max_int_multiplexing= max(list_max_ssa),save_as_gif=self.save_as_gif,frame_selection_empty_video=self.frame_selection_empty_video)
            else:
                tensor_video , DataFrame_particles_intensities = wrapper_SimulatedCell(tensor_video, video_for_mask = self.inial_video, ssa=list_ssa[i], target_channel=self.list_target_channels[i], diffusion_coefficient=self.list_diffusion_coefficients[i],using_for_multiplexing=1,min_int_multiplexing =min(list_min_ssa) , max_int_multiplexing= max(list_max_ssa),save_as_gif=self.save_as_gif,frame_selection_empty_video='loop') # notice that for the multiplexing frame_selection_empty_video has to be 'loop', becuase the intial video deffines the inital background image.
            list_DataFrame_particles_intensities.append(DataFrame_particles_intensities)
        # Adding a classification column to all dataframes
        for i in range(0,self.number_genes):
            classification = self.list_label_names[i]
            list_DataFrame_particles_intensities[i]['Classification'] = classification
        # Increasing the particle number
        for i in range(0,self.number_genes):
            if i > 0:
                number_spots_for_previous_genes = np.sum(self.list_number_spots[0:i])
                list_DataFrame_particles_intensities[i]['particle'] = list_DataFrame_particles_intensities[i]['particle'] + number_spots_for_previous_genes
        # Merging multiple dataframes in a single one.
        dataframe_simulated_cell = pd.concat(list_DataFrame_particles_intensities)        
        
        # saving the simulated video and data frame
        
        if self.save_as_tif==1:
            if self.create_temp_folder == True:
                save_to_path = pathlib.Path().absolute().joinpath('temp')
                if not os.path.exists(str(save_to_path)):
                    os.makedirs(str(save_to_path))
                print ("The output is saved in the directory: " , str(save_to_path))
            else:
                save_to_path=pathlib.Path().absolute()
            tifffile.imwrite(str(save_to_path.joinpath(self.saved_file_name+'.tif')), tensor_video)
        if self.save_dataframe==1:
            if self.create_temp_folder == True:
                save_to_path = pathlib.Path().absolute().joinpath('temp')
                if not os.path.exists(str(save_to_path)):
                    os.makedirs(str(save_to_path))
                print ("The output is saved in the directory: " , str(save_to_path))
            else:
                save_to_path=pathlib.Path().absolute()
            dataframe_simulated_cell.to_csv(str(save_to_path.joinpath(self.saved_file_name +'_df'+ '.csv')), index = True)
        return tensor_video, dataframe_simulated_cell, list_ssa

class PipelineTracking():
    '''
    A pipeline that allows cell segmentation, spot detection, and tracking of spots.

    Parameters
    ----------
    video : NumPy array
        Array of images with dimensions [T,Y,X,C].
    particle_size : int, optional
        Average particle size. The default is 5.
    file_name : str, optional
        The file name for the simulated cell. The default is 'Cell.tif'.
    selected_channel : int, optional
        Allows the user to define the channel to visualize in the plotted images. The default is 0.
    intensity_calculation_method : str, optional
        Method to calculate intensity the options are : 'total_intensity' , 'disk_donut' and 'gaussian_fit'. The default is 'disk_donut'..
    mask_selection_method :  str, optional
        Option to use the optimization algorithm to maximize the number of cells or maximize the size. The options are 'max_area' or 'max_cells'. The default is 'max_area'.
    show_plot : bool, optional
        Allows the user to show a plot for the optimization process. The default is 1.
    use_optimization_for_tracking : bool, optional
        Option to run an optimization process to slect the best filter for the tracking process. The default is 0.
    real_positions_dataframe : Pandas dataframe.
        A pandas data frame containing the position of each spot in the image. This dataframe is generated with class SimulatedCell, and it contains the true position for each spot. This option is only intended to be used to train algorithms for tracking and visualize real vs detected spots. The default is None.
    average_cell_diameter : float, optional
        Average cell size. The default is 120.
    print_process_times : bool, optional
        Allows the user the times taken during each process. The default is 0.

    '''
    def __init__(self,video,particle_size=5,file_name='Cell.tif',selected_channel=0,intensity_calculation_method ='disk_donut', mask_selection_method = 'max_spots',show_plot=1, use_optimization_for_tracking =1,real_positions_dataframe = None,average_cell_diameter=120,print_process_times=0):
        self.video = video
        self.particle_size = particle_size
        self.image = video[1,:,:,:]
        self.num_frames = video.shape[0]
        self.file_name = file_name
        self.intensity_calculation_method = intensity_calculation_method  # options are : 'total_intensity' and 'disk_donut'
        self.mask_selection_method = mask_selection_method # options are : 'max_spots' and 'max_area'
        self.selected_channel = selected_channel
        self.show_plot = show_plot
        self.use_optimization_for_tracking = use_optimization_for_tracking
        self.real_positions_dataframe = real_positions_dataframe
        self.average_cell_diameter = average_cell_diameter
        self.print_process_times = print_process_times
        # Iterations
        self.NUM_ITERATIONS_CELLPOSE =10
        self.NUM_ITERATIONS_TRACKING = 100
        self.MN_PERCENTAGE_FRAMES_FOR_TRACKING = 0.3

    def run(self):
        '''
        Runs the pipeline.

        Returns
        -------
        dataframe_particles : pandas dataframe
            Dataframe with fields [cell_number, particle, frame, red_int_mean, green_int_mean, blue_int_mean, red_int_std, green_int_std, blue_int_std, x, y].
        array_intensities : Numpy array
            Array with dimensions [S,T,C].
        time_vector : Numpy array
            1D array.
        mean_intensities: Numpy array
            Array with dimensions [S,T,C].
        std_intensities : Numpy array
            Array with dimensions [S,T,C].
        mean_intensities_normalized : Numpy array
            Array with dimensions [S,T,C].
        std_intensities_normalized : Numpy array
            Array with dimensions [S,T,C].

        '''
        start = timer()
        selected_masks = Cellpose(self.image,num_iterations=self.NUM_ITERATIONS_CELLPOSE, selection_method='max_cells',diameter=self.average_cell_diameter ).calculate_masks() # options are 'max_area' or 'max_cells'
        if not ( selected_masks is None):
            selected_mask  = CellposeSelection(selected_masks,self.video, slection_method = self.mask_selection_method, particle_size=self.particle_size, selected_channel=self.selected_channel).select_mask()
        else:
            selected_mask= None
        end = timer()
        if self.print_process_times ==1:
            print('mask time:',round(end - start), ' sec')
        if not ( selected_mask is None):
            # Tracking
            start = timer()
            if self.num_frames >20:
                minimal_frames =  int(self.num_frames*self.MN_PERCENTAGE_FRAMES_FOR_TRACKING) # minimal number of frames to consider a trajectory
            else:
                minimal_frames =  int(self.num_frames*0.2) # minimal number of frames to consider a trajectory
            if self.use_optimization_for_tracking ==1:
                use_defaul_filter = 0
            else:
                use_defaul_filter = 1
            try:
                Dataframe_trajectories, number_detected_trajectories, filtered_video = Trackpy(self.video,selected_mask,particle_size=self.particle_size, selected_channel=self.selected_channel,minimal_frames=minimal_frames,optimization_iterations = self.NUM_ITERATIONS_TRACKING,use_defaul_filter=use_defaul_filter, show_plot =0).perform_tracking()
            except:
                Dataframe_trajectories = None
                filtered_video = self.video
            end = timer()
            if self.print_process_times ==1:
                print('tracking time:',round(end - start), ' sec')
            # Intensity calculation
            start = timer()
            if not ( Dataframe_trajectories is None):
                dataframe_particles, array_intensities, time_vector, mean_intensities,std_intensities, mean_intensities_normalized, std_intensities_normalized = Intensity(self.video,self.particle_size,Dataframe_trajectories,method=self.intensity_calculation_method,show_plot=0 ).calculate_intensity()
            else:
                dataframe_particles = None
                array_intensities = None
                time_vector = None
                mean_intensities = None
                std_intensities = None
                mean_intensities_normalized = None
                std_intensities_normalized = None
            end = timer()
            if self.print_process_times ==1:
                print('intensity calculation time:',round(end - start), ' sec')
            if (self.show_plot ==1):
                VisualizerImage(self.video,filtered_video,Dataframe_trajectories,self.file_name,list_mask_array = selected_mask,selected_channel=self.selected_channel,selected_timepoint=0,normalize=False,individual_figure_size=7,list_real_particle_positions=self.real_positions_dataframe).plot()
        else:
            dataframe_particles = None
            array_intensities = None
            time_vector = None
            mean_intensities = None
            std_intensities = None
            mean_intensities_normalized = None
            std_intensities_normalized = None
        return dataframe_particles, array_intensities, time_vector, mean_intensities,std_intensities, mean_intensities_normalized, std_intensities_normalized
