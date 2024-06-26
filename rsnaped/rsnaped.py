#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
rSNAPed: A software for single-molecule image tracking, simulation and parameter estimation.
Created on Fri Jun 26 22:10:24 2020
Authors: Luis U. Aguilera, William Raymond, Brooke Silagy, Brian Munsky.
'''


# Conventions.
# module_name, package_name, ClassName, method_name,
# ExceptionName, function_name, GLOBAL_CONSTANT_NAME,
# global_var_name, instance_var_name, function_parameter_name, local_var_name.

# To manipulate arrays
import pkg_resources
#pkg_resources.require("numpy >= `1.20.1")  #  to use specific numpy version
import numpy as np
from numpy import ndarray
from numpy import unravel_index
# To run stochastic simulations
try:
    import rsnapsim as rss
except:
    pass
    #print('Problems importing rsnapsim')
try:
    import cv2
except:
    print('Problems importing cv2')
# System libraries
import io
import sys
import datetime
import getpass
import socket
import platform
#import statistics
from statistics import median_low
import random
from random import randrange
import math
from math import nan
# For data typing
from typing import Union
from typing import List
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
from skimage.restoration import denoise_nl_means, estimate_sigma, denoise_wavelet
from skimage.morphology import square, dilation
from skimage.io import imread
from scipy.ndimage import gaussian_filter1d
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
# scipy
import scipy.stats as sps
from scipy.signal import find_peaks #, peak_prominences, find_peaks_cwt
from scipy.spatial import Delaunay
from scipy.optimize import curve_fit
from scipy import ndimage as nd 
from scipy.special import erf
from scipy.ndimage import gaussian_laplace
# To read .tiff files
import tifffile
# to create gifs
import imageio
# time libraries
from timeit import default_timer as timer
import time
# Cellpose
try:
    from cellpose import models
except:
    print('Problems importing cellpose')
# Plotting
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
from matplotlib import gridspec
# To work with files
import os
from os import listdir
from os.path import isfile, join
import re # to iterate in files
import glob # to iterate in files
import pathlib
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import shutil
from fpdf import FPDF
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.patches import Rectangle
from matplotlib.patches import ConnectionPatch
from scipy.ndimage.measurements import center_of_mass


#import statsmodels.tsa.stattools as stattools
#https://www.statsmodels.org/devel/generated/statsmodels.tsa.stattools.acf.html#statsmodels.tsa.stattools.acf

class Banner():
    def __init__(self,show=True):
        self.show = show
    def print_banner(self):
        if self.show == True:
            print(" \n"
                "  ██████╗░░██████╗███╗░░██╗░█████╗░██████╗░███████╗██████╗░ \n" 
                "  ██╔══██╗██╔════╝████╗░██║██╔══██╗██╔══██╗██╔════╝██╔══██╗ \n" 
                "  ██████╔╝╚█████╗░██╔██╗██║███████║██████╔╝█████╗░░██║░░██║ \n" 
                "  ██╔══██╗░╚═══██╗██║╚████║██╔══██║██╔═══╝░██╔══╝░░██║░░██║ \n" 
                "  ██║░░██║██████╔╝██║░╚███║██║░░██║██║░░░░░███████╗██████╔╝ \n" 
                "             by : L. Aguilera, T. Stasevich, and B. Munsky ")


class SSA_rsnapsim():
    '''
    This class uses rsnapsim to simulate the single-molecule translation dynamics of any gene.
    
    Parameters

    gene_file : str, 
        Path to the location of a FASTA file.
    ke : float, optional.
        Elongation rate. The default is 10.0.
    ki: float, optional.
        Initiation rate. The default is 0.03.
    frames: int, optional.
        Total number of simulation frames in seconds. The default is 300.
    n_traj: int, optional.
        Number of trajectories to simulate. The default is 20.
    frame_rate : int, optional.
        Frame rate per second. The default is 1.
    t_burnin : int , optional
        time of burnin. The default is 1000
    use_Harringtonin: bool, optional
        Flag to specify if Harringtonin is used in the experiment. The default is False.
    use_FRAP: bool
        Flag to specify if FRAP is used in the experiment. The default is False.
    perturbation_time_start: int, optional.
        Time to start the inhibition. The default is 0.
    perturbation_time_stop : int, opt.
        Time to start the inhibition. The default is None.

    Outputs:
    '''  
    def __init__(self,gene_file,mrna_object=None, ke=10,ki=0.03,frames=300,frame_rate=1,n_traj=20,t_burnin=1000,use_Harringtonin=False,use_FRAP=False, perturbation_time_start=0,perturbation_time_stop=None):
        self.gene_file=gene_file
        self.ke=ke
        self.ki=ki
        self.frames=frames
        self.frame_rate=frame_rate
        self.n_traj=n_traj
        self.t_burnin=t_burnin
        self.use_Harringtonin=use_Harringtonin
        self.use_FRAP=use_FRAP
        self.perturbation_time_start=perturbation_time_start
        self.perturbation_time_stop=perturbation_time_stop
        self.NUMBER_OF_CORES = multiprocessing.cpu_count()
        self.mrna_object = mrna_object

    def simulate(self):
        '''
        Method runs rSNAPsim and simulates the single molecule translation dynamics.

        Returns

        ssa_int : NumPy array.
            Contains the SSA trajectories with dimensions [Time_points, simulated_trajectories].
        ssa_ump : NumPy array.
            SSA trajectories in UMP(units of mature protein). SSA trajectories normalized by the number of probes in the sequence.  Array with dimensions [Time_points, simulated_trajectories].
        time_vector: NumPy array with dimensions [1, Time_points].
            Time vector used in the simulation.
        '''
        
        ##TODO ADD THE MRNA OBJECT SOLVER VERSION
        
        t = np.linspace(0,self.t_burnin+self.frames,(self.t_burnin+self.frames+1)*(self.frame_rate))
        _, _, tagged_pois,raw_seq = rss.seqmanip.open_seq_file(str(self.gene_file))
        try:
            gene_obj = tagged_pois['0'][0]
        except:
            gene_obj = tagged_pois['1'][0]
        gene_obj.ke_mu = self.ke
        number_probes = np.max(gene_obj.probe_vec)
        gene_length = len(raw_seq)
        if not ( self.perturbation_time_stop is None):
            t_stop_perturbation = self.perturbation_time_stop+self.t_burnin
        else:
            t_stop_perturbation = self.t_burnin+self.frames
        perturbation_list = [self.use_FRAP, self.use_Harringtonin,self.perturbation_time_start+self.t_burnin,t_stop_perturbation]
        rss.solver.protein = gene_obj #pass the protein object
        ssa_solution = rss.solver.solve_ssa(gene_obj.kelong,t, perturb=perturbation_list, ki=self.ki, low_memory=True, n_traj=self.n_traj )
        ssa = np.transpose( ssa_solution.intensity_vec[:,self.t_burnin*self.frame_rate:-1,:]) [:,:,0]
        ssa_ump = ssa/number_probes
        return ssa, ssa_ump, t,gene_length


class ReadImages():
    '''
    This class reads all .tif images in a given folder and returns the names of these files, path, and number of files.
    
    Parameters

    directory: str or PosixPath
        Directory containing the images to merge.
    '''

    def __init__(self, directory:str ):
        self.directory = Path(directory)
    def read(self):
        '''
        Method takes all the videos in the folder and merge those with similar names.

        Returns

        list_images : List of NumPy arrays. 
            List of NumPy arrays with format np.uint16 and dimensions [Z, Y, X, C] or [T, Y, X, C] . 
        list_file_names : List of strings 
            List with strings of names.
        number_files : int. 
            Number of images in the folder.
        '''
        path_files, list_files_names, list_images, number_images = Utilities.read_files_in_directory(directory= self.directory, extension_of_files_to_look_for = 'tif')
        return list_images, path_files, list_files_names, number_images


class MergeChannels():
    '''
    This class takes images as arrays with format [Z,Y,X] and merge then in a numpy array with format [Z, Y, X, C].
    It recursively merges the channels in a new dimension in the array. Minimal number of Channels 2 maximum is 4
    
    Parameters

    directory: str or PosixPath
        Directory containing the images to merge.
    substring_to_detect_in_file_name: str
        String with the prefix to detect in the files names. 
    save_figure: bool, optional
        Flag to save the merged images as .tif. The default is False. 
    '''
    def __init__(self, directory:str ,substring_to_detect_in_file_name:str = '.*_C0.tif', save_figure:bool = False ):
        if type(directory)== pathlib.PosixPath:
            self.directory = directory
        else:
            self.directory = Path(directory)
        self.substring_to_detect_in_file_name = substring_to_detect_in_file_name
        self.save_figure=save_figure
    
    def merge(self):
        '''
        Method takes all the videos in the folder and merge those with similar names.

        Returns

        list_file_names : List of strings 
            List with strings of names.
        list_merged_images : List of NumPy arrays. 
            List of NumPy arrays with format np.uint16 and dimensions [Z, Y, X, C].
        number_files : int. 
            Number of merged images in the folder.
        '''
        list_file_names =[]
        list_merged_images =[]  # list that stores all files belonging to the same image in a sublist
        ending_string = re.compile(self.substring_to_detect_in_file_name)  # detecting files ending in _C0.tif
        save_to_path = self.directory.joinpath('merged')
        for _, _, files in os.walk(self.directory):
            for file in files:
                if ending_string.match(file):
                    prefix = file.rpartition('_')[0]  # stores a string with the first part of the file name before the last underscore character in the file name string.
                    list_files_per_image = sorted ( glob.glob( str(self.directory.joinpath(prefix)) + '*.tif'))
                    list_file_names.append(prefix)
                    merged_img = np.concatenate([ imread(list_files_per_image[i])[..., np.newaxis] for i,_ in enumerate(list_files_per_image)],axis=-1).astype('uint16')
                    list_merged_images.append(merged_img) 
                    if self.save_figure ==1:
                        if not os.path.exists(str(save_to_path)):
                            os.makedirs(str(save_to_path))
                        tifffile.imsave(str(save_to_path.joinpath(prefix+'_merged'+'.tif')), merged_img, metadata={'axes': 'ZYXC'})
        number_files = len(list_file_names)
        return list_file_names, list_merged_images, number_files,save_to_path


class ConvertToStandardFormat():
    '''
    This class contains two methods to:

    1. Transform any numpy array of images into the format [T, Y, X, C].
    2. Convert an image into an array video with a single time point (this last is necessary for compatibility).

    Parameters

    video : NumPy array
        Array of images. This class accepts arrays with formats: [Y, X], [T, Y, X], [T, Y, X, C], or any other  permutation of channels, the user must specify the position of each dimension in the original video by defining the parameters: time_position, height_position, width_position, channel_position.
    time_position : int, optional
        Position for the dimension for the time in the original video array. The default is 0.
    height_position : int, optional
        Position for the dimension for the y-axis (height) in the original video array. The default is 1.
    width_position : int, optional
        Position for the dimension for the x-axis (width) in the original video array. The default is 2.
    channel_position : int, optional
        Position for the channel's dimension in the original video array. The default is 3.
    '''

    def __init__(self, video:np.ndarray, time_position:int = 0, height_position:int = 1,  width_position:int = 2, channel_position:int = 3):
        self.video = video
        self.time_dimension = time_position
        self.height_dimension = height_position
        self.width_dimension = width_position
        self.channel_dimension = channel_position
    def transpose_video(self):
        '''
        Method that transposes an unsorted video to the standard [T, Y, X, C]

        Returns

        video_correct_order : np.uint16
            Array with dimensions [T, Y, X, C].
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
        if video_transposed.shape[3] < 3:
            print ('The video has been transposed to the format [T, Y, X, C] and the channels are RGB')
            video_correct_order = np.zeros((number_frames, width, height, 3), dtype = np.uint16)
            video_correct_order[:, :, :, :number_channels] = video_transposed
        elif video_transposed.shape[3] == 3:
            print ('The video has been transposed to the format [T, Y, X, C]')
            video_correct_order = np.copy(video_transposed)
        return video_correct_order
    def image_to_video(self):
        '''
        Method that converts an image into a video with one frame. This process is done for compatibility with the rest of the classes.

        Returns

        video_correct_order : np.uint16
            Array with dimensions [T, Y, X, C].
        '''
        # This section corrects the video to the dimensions. [T, Y, X, C] in case it is an image with 2D x, y.
        if len(self.video.shape) == 2:
            video_temp = np.copy(self.video)
            video_correct_order = np.zeros((1, self.video.shape[0], self.video.shape[1], 3), dtype = np.uint16)
            video_correct_order[0, :, :, 0] = video_temp
            print ('The video has been converted to the format [T, Y, X, C] from [Y, X]')
        if len(self.video.shape) == 3:
            video_temp = np.copy(self.video)
            video_correct_order = np.zeros((1, self.video.shape[0], self.video.shape[1], 3), dtype = np.uint16)
            video_correct_order[0, :, :, :] = video_temp
            print ('The video has been converted to the format [T, Y, X, C] from [Y, X, C]')
        return video_correct_order


class AugmentationVideo():
    '''
    This class is intended to be used for data augmentation. It takes a video and perform random rotations in the X and Y axis.
    
    Parameters

    video : NumPy array
        Array of images with dimensions [T, Y, X, C].
    predefined_angle : int, with values 0, 90, 180, 270, 360, or None. Optional, None is the default.
        Indicates the specific angle of rotation.
    '''
    def __init__(self, video:np.ndarray,predefined_angle=None):
        self.video = video
        self.predefined_angle=predefined_angle

    def random_rotation(self):
        '''
        Method that performs random rotations of a video in the Y and X axis.

        Returns

        video_random_rotation : np.uint16
            Array with dimensions [T, Y, X, C].
        '''
        if (self.predefined_angle is None):
            angles = [0, 90, 180, 270, 360]
            selected_angle = random.choice(angles)
        else:
            selected_angle=self.predefined_angle
        if selected_angle != 0:
            if len(self.video.shape) > 3:
                video_random_rotation = nd.rotate(self.video, angle = selected_angle, axes = (1, 2))
            else:
                video_random_rotation = nd.rotate(self.video, angle = selected_angle, axes = (0, 1))
            return video_random_rotation , selected_angle
        else:
            return self.video, selected_angle
        

class RemoveExtrema():
    '''
    This class is intended to remove extreme values from a video. The format of the video must be [Y, X] , [Y, X, C] , [Z, Y, X, C] or [T, Y, X, C].

    Parameters

    video : NumPy array
        Array of images with dimensions [Y, X] , [Y, X, C] , [Z, Y, X, C] or [T, Y, X, C].
    min_percentile : float, optional
        Lower bound to normalize intensity. The default is 1.
    max_percentile : float, optional
        Higher bound to normalize intensity. The default is 99.
    selected_channels : List or None, optional
        Use this option to select a list channels to remove extrema. The default is None and applies the removal of extrema to all the channels.
    format_video : str, optional,
        Use this option to select the format of the video. The default is 'TYXC'. Options are 'YX' , 'YXC' , 'ZYXC' or 'TYXC'.
    '''

    def __init__(self, video:np.ndarray, min_percentile:float = 1, max_percentile:float = 99, selected_channels = None, format_video = 'TYXC'):
        self.video = video
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile
        if not (type(selected_channels) is list):
                self.selected_channels = [selected_channels]
        else:
            self.selected_channels =selected_channels
        self.format_video = format_video

    def remove_outliers(self):
        '''
        This method normalizes the values of a video by removing extreme values.

        Returns

        normalized_video : np.uint16
            Normalized video. Array with dimensions [T, Y, X, C] or image with format [Y, X].
        '''
        
        def image_extrema_removal(image):
            max_val = np.percentile(image, self.max_percentile)
            min_val = np.min ( (0, np.percentile(image, self.min_percentile)))
            image [image > max_val] = max_val
            image [image < min_val] = min_val
            image [image < 0] = 0
            return image
        # Preallocate array
        temp_video = np.zeros(self.video.shape)
        # Remove extrema depending on the format of the video
        if self.format_video == 'YX':
            if not np.max(self.video) == 0: 
                temp_video = image_extrema_removal(self.video)
            else:
                temp_video = self.video
        elif self.format_video == 'YXC':
            for i in range(0, self.video.shape[2]):
                temp_img = self.video[:,:,i]
                if not np.max(temp_img) == 0:
                    temp_img = image_extrema_removal(temp_img)
                    temp_video[:,:,i] = temp_img
                else:
                    temp_video[:,:,i] = temp_img
        elif self.format_video == 'TYX':
            for i in range(0, self.video.shape[0]):
                temp_img = self.video[i,:,:]
                if not np.max(temp_img) == 0:
                    temp_img = image_extrema_removal(temp_img)
                    temp_video[i,:,:] = temp_img
                else:
                    temp_video[i,:,:] = temp_img
        elif self.format_video == 'TYXC' or self.format_video == 'ZYXC':
            for i in range(0, self.video.shape[0]):
                for j in range(0, self.video.shape[3]):
                    temp_img = self.video[i,:,:,j]
                    if not np.max(temp_img) == 0:
                        temp_img = image_extrema_removal(temp_img)
                        temp_video[i,:,:,j] = temp_img
                    else:
                        temp_video[i,:,:,j] = temp_img
        return np.asarray(temp_video, 'uint16')

# class RemoveExtrema():
#     '''
#     This class is intended to remove extreme values from a video. The format of the video must be [Y, X] , [Y, X, C] , [Z, Y, X, C] or [T, Y, X, C].

#     Parameters

#     video : NumPy array
#         Array of images with dimensions [Y, X] , [Y, X, C] , [Z, Y, X, C] or [T, Y, X, C].
#     min_percentile : float, optional
#         Lower bound to normalize intensity. The default is 1.
#     max_percentile : float, optional
#         Higher bound to normalize intensity. The default is 99.
#     selected_channels : List or None, optional
#         Use this option to select a list channels to remove extrema. The default is None and applies the removal of extrema to all the channels.
#     '''

#     def __init__(self, video:np.ndarray, min_percentile:float = 1, max_percentile:float = 99, selected_channels:Union[list, None] = None):
#         self.video = video
#         self.min_percentile = min_percentile
#         self.max_percentile = max_percentile
#         if not (type(selected_channels) is list):
#                 self.selected_channels = [selected_channels]
#         else:
#             self.selected_channels =selected_channels

#     def remove_outliers(self):
#         '''
#         This method normalizes the values of a video by removing extreme values.

#         Returns

#         normalized_video : np.uint16
#             Normalized video. Array with dimensions [T, Y, X, C] or image with format [Y, X].
#         '''
        
#         normalized_video = np.copy(self.video)
#         normalized_video_temp = np.zeros_like(normalized_video)
#         # Normalization code for image with format [Y, X]
#         if len(self.video.shape) == 2:
#             number_time_points = 1
#             number_channels = 1
#             #normalized_video_temp = normalized_video
#             if not np.max(normalized_video) == 0: # this section detect that the channel is not empty to perform the normalization.
#                 max_val = np.percentile(normalized_video, self.max_percentile)
#                 min_val = np.min ( (0, np.percentile(normalized_video, self.min_percentile)))
#                 normalized_video [normalized_video > max_val] = max_val
#                 normalized_video [normalized_video < min_val] = min_val
#                 normalized_video [normalized_video < 0] = 0
#                 normalized_video_temp = normalized_video
#         # Normalization for video with format [Y, X, C].
#         if len(self.video.shape) == 3:
#             number_channels   = self.video.shape[-1]
#             for index_channels in range (number_channels):
#                 if (index_channels in self.selected_channels) or (self.selected_channels is None) :
#                     #normalized_video_temp = normalized_video[ :, :, index_channels]
#                     if not np.max(normalized_video_temp) == 0: # this section detect that the channel is not empty to perform the normalization.
#                         max_val = np.percentile(normalized_video_temp, self.max_percentile)
#                         min_val = np.min ( (0,np.percentile(normalized_video_temp, self.min_percentile)))
#                         normalized_video_temp [normalized_video_temp > max_val] = max_val
#                         normalized_video_temp [normalized_video_temp < min_val] =  min_val
#                         normalized_video_temp [normalized_video_temp < 0] = 0
#                         normalized_video[ :, :, index_channels] = normalized_video_temp[ :, :, index_channels] 
#         # Normalization for video with format [T, Y, X, C] or [Z, Y, X, C].
#         if len(self.video.shape) == 4:
#             number_time_points, number_channels   = self.video.shape[0], self.video.shape[-1]
#             for index_channels in range (number_channels):
#                 if (index_channels in self.selected_channels) or (self.selected_channels is None) :
#                     for index_time in range (number_time_points):
#                         normalized_video_temp = normalized_video[index_time, :, :, index_channels].copy()
#                         if not np.max(normalized_video_temp) == 0: # this section detect that the channel is not empty to perform the normalization.
#                             max_val = np.percentile(normalized_video_temp, self.max_percentile)
#                             min_val = np.min ( (0,np.percentile(normalized_video_temp, self.min_percentile)))
#                             normalized_video_temp [normalized_video_temp > max_val] = max_val
#                             normalized_video_temp [normalized_video_temp < min_val] = min_val
#                             normalized_video_temp [normalized_video_temp < 0] = 0
#                             normalized_video[ index_time, :, :, index_channels] = normalized_video_temp
#             normalized_video[normalized_video<0]=0
#         return np.asarray(normalized_video, 'uint16')


class ScaleIntensity():
    '''
    This class is intended to scale the intensity values in a video. The format of the video must be [T, Y, X, C].

    Parameters

    video : NumPy array
        Array of images with dimensions [T, Y, X, C].
    scale_maximum_value : int, optional
        Maximum intensity value to scaled the video. The default is 1000.
    ignore_channel : int or None, optional
        Use this option to ignore the normalization of a given channel. The default is None.
    '''
    def __init__(self, video:np.ndarray, scale_maximum_value:int = 1000, ignore_channel: Union[bool, None] = None):
        self.video = video
        self.scale_maximum_value = scale_maximum_value
        self.ignore_channel = ignore_channel
    def apply_scale(self):
        '''
        This method is intended to scale the intensity values of a video.

        Returns

        scaled_video : np.uint16
            Scaled video. Array with dimensions [T, Y, X, C].
        '''
        scaled_video = np.copy(self.video)
        scaled_video = np.array(scaled_video, 'float32')
        number_time_points, number_channels   = self.video.shape[0], self.video.shape[3]
        for index_channels in range (number_channels):
            if not self.ignore_channel == index_channels:
                max_val = np.max(scaled_video[:, :, :, index_channels])
                min_val = np.min(( 0, np.min(scaled_video[:, :, :, index_channels]) ) )
                for index_time in range (number_time_points):
                    if max_val != 0: # this section detect that the channel is not empty to perform the normalization.
                        scaled_video[index_time, :, :, index_channels] = ((scaled_video[index_time, :, :, index_channels])-min_val) / (max_val-min_val)
                        scaled_video[scaled_video<0]=0
                        scaled_video[index_time, :, :, index_channels] = np.multiply( scaled_video[index_time, :, :, index_channels] , self.scale_maximum_value)                         
        scaled_video_int = scaled_video.astype('uint16')
        
        return scaled_video_int


class GaussianLaplaceFilter():
    '''
    This class is intended to apply high and low bandpass filters to the video. The format of the video must be [T, Y, X, C]. This class uses **difference_of_gaussians** from skimage.filters.

    Parameters

    video : NumPy array
        Array of images with dimensions [T, Y, X, C].
    low_pass : float, optional
        Lower pass filter intensity. The default is 0.5.
    high_pass : float, optional
        Higher pass filter intensity. The default is 10.

    '''
    def __init__(self, video:np.ndarray, sigma:float = 1):
        # Making the values for the filters are odd numbers
        self.video = video
        self.sigma = sigma
        self.NUMBER_OF_CORES = multiprocessing.cpu_count()
    def apply_filter(self):
        '''
        This method applies high and low bandpass filters to the video.

        Returns

        video_filtered : np.uint16
            Filtered video resulting from the bandpass process. Array with format [T, Y, X, C].
        '''
        # Pre-allocating arrays
        number_time_points, number_channels   = self.video.shape[0], self.video.shape[3]
        video_filtered = np.zeros_like(self.video, dtype = np.uint16)
        # Applying the filter
        for index_channels in range(0, number_channels):
            for index_time in range(0, number_time_points):
                video_filtered[index_time, :, :, index_channels] = gaussian_laplace(self.video[index_time, :, :, index_channels], self.sigma)
        return video_filtered

class GaussianFilter():
    '''
    This class is intended to apply high and low bandpass filters to the video. The format of the video must be [T, Y, X, C]. This class uses **difference_of_gaussians** from skimage.filters.

    Parameters

    video : NumPy array
        Array of images with dimensions [T, Y, X, C].
    low_pass : float, optional
        Lower pass filter intensity. The default is 0.5.
    high_pass : float, optional
        Higher pass filter intensity. The default is 10.

    '''
    def __init__(self, video:np.ndarray, sigma:float = 1):
        # Making the values for the filters are odd numbers
        self.video = video
        self.sigma = sigma
        self.NUMBER_OF_CORES = multiprocessing.cpu_count()
    def apply_filter(self):
        '''
        This method applies high and low bandpass filters to the video.

        Returns
        
        video_filtered : np.uint16
            Filtered video resulting from the bandpass process. Array with format [T, Y, X, C].
        '''
        video_bp_filtered_float = np.zeros_like(self.video, dtype = np.float64)
        video_filtered = np.zeros_like(self.video, dtype = np.uint16)
        number_time_points, number_channels   = self.video.shape[0], self.video.shape[3]
        for index_channels in range(0, number_channels):
            for index_time in range(0, number_time_points):
                video_bp_filtered_float[index_time, :, :, index_channels] = gaussian(self.video[index_time, :, :, index_channels], self.sigma)
        # returning the image normalized as uint. Notice that difference_of_gaussians converts the image into float.
        for index_channels in range(0, number_channels):
            init_video = Parallel(n_jobs = self.NUMBER_OF_CORES)(delayed(Utilities.img_uint)(video_bp_filtered_float[i, :, :, index_channels]) for i in range(0, number_time_points))
            video_filtered[:,:,:,index_channels] = np.asarray(init_video)
        return video_filtered


class BandpassFilter():
    '''
    This class is intended to apply high and low bandpass filters to the video. The format of the video must be [T, Y, X, C]. This class uses **difference_of_gaussians** from skimage.filters.

    Parameters

    video : NumPy array
        Array of images with dimensions [T, Y, X, C].
    low_pass : float, optional
        Lower pass filter intensity. The default is 0.5.
    high_pass : float, optional
        Higher pass filter intensity. The default is 10.

    '''
    def __init__(self, video:np.ndarray, low_pass:float = 0.5, high_pass:float = 10):
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

        video_filtered : np.uint16
            Filtered video resulting from the bandpass process. Array with format [T, Y, X, C].
        '''
        # Pre-allocating arrays
        number_time_points, number_channels   = self.video.shape[0], self.video.shape[3]
        video_bp_filtered_float = np.zeros_like(self.video, dtype = np.float64)
        video_float = np.zeros_like(self.video, dtype = np.float64)
        video_filtered = np.zeros_like(self.video, dtype = np.uint16)
        # returning the image normalized as uint. Notice that difference_of_gaussians converts the image into float.
        for index_channels in range(0, number_channels):
            temp_video = Parallel(n_jobs = self.NUMBER_OF_CORES)(delayed(Utilities.img_float)(self.video [i, :, :, index_channels]) for i in range(0, number_time_points))
            video_float[:,:,:,index_channels] = np.asarray(temp_video)
        # Applying the filter
        for index_channels in range(0, number_channels):
            for index_time in range(0, number_time_points):
                video_bp_filtered_float[index_time, :, :, index_channels] = difference_of_gaussians(video_float[index_time, :, :, index_channels], self.low_pass, self.high_pass)
        # returning the image normalized as uint. Notice that difference_of_gaussians converts the image into float.
        for index_channels in range(0, number_channels):
            init_video = Parallel(n_jobs = self.NUMBER_OF_CORES)(delayed(Utilities.img_uint)(video_bp_filtered_float[i, :, :, index_channels]) for i in range(0, number_time_points))
            video_filtered[:,:,:,index_channels] = np.asarray(init_video)
        return video_filtered


class MaskingImage():
    '''
    This class is intended to apply a mask to the video. The video format must be [T, Y, X, C], and the format of the mask must be [Y, X].

    Parameters

    video : NumPy array
        Array of images with dimensions [T, Y, X, C].
    mask : NumPy array, with Boolean values, where 1 represents the masked area, and 0 represents the area outside the mask.
        An array of images with dimensions [Y, X].
    '''
    def __init__(self, video:np.ndarray, mask:np.ndarray):
        self.mask = mask
        self.video = video
    def apply_mask(self):
        '''
        This method applies high and low bandpass filters to the video. The method uses **difference_of_gaussians** from skimage.filters.

        Returns

        video_removed_mask : np.uint16
            Video with zero values outside the area delimited by the mask. Array with format [T, Y, X, C].
        '''
        video_removed_mask = np.einsum('ijkl, jk -> ijkl', self.video, self.mask)
        return video_removed_mask


class MaskManual_createMask():
    def __init__(self,video,mask_object,time_point_selected=0,selected_channel=1,show_plot=True):
        self.video = video
        self.mask_object = mask_object
        self.time_point_selected = time_point_selected
        self.selected_channel = selected_channel
        self.number_channels = video.shape
        self.video_removed_mask= np.zeros_like(video) 
        self.show_plot=show_plot
        
    def make_mask(self):
        time_points, height, width, number_channels = self.video.shape[0],self.video.shape[1], self.video.shape[2], self.video.shape[3]
        array_points_coordinates = np.array([self.mask_object.selected_points],'int')
        mask = cv2.fillPoly(np.zeros(self.video[self.time_point_selected,:,:,self.selected_channel].shape,np.uint8),array_points_coordinates,[1,1,1])
        mask_array = np.zeros((time_points, height, width))
        for i in range(0,number_channels):
            for k in range(0,time_points):
                self.video_removed_mask[k,:height,:width,i] = np.multiply(self.video[k,:height,:width,i], mask)
                mask_array [k,:,:] = mask
        if self.show_plot ==1:
        # Plotting
            plt.rcParams["figure.figsize"] = (5,5)
            plt.imshow(self.video_removed_mask[self.time_point_selected,:,:,self.selected_channel], cmap=plt.cm.cividis)
            plt.show()        
        return self.video_removed_mask, mask_array


class BeadsAlignment():
    '''
    This class is intended to detected and align spots detected in the various channels of an image with dimensions [C, Y, X]. The class returns a homography matrix that can be used to align the images captured from two different cameras during the experiment. Notice that this class should only be used for images taken from a microscope setup that uses two cameras to image various channels.
    
    Parameters

    first_image_beads : NumPy array
        Array with a simple image with dimensions [ Y, X].
    second_image_beads : NumPy array
        Array with a simple image with dimensions [ Y, X].
    spot_size : int, optional
        Average size of the beads,  The default is 5.
    min_intensity : float, optional
        Minimal expected intensity for the beads. The default is 400.
    show_plot : Bool, optional
        Shows a plot with the detected beads in the image. The default is True.
    '''
    def __init__(self, first_image_beads:np.ndarray,second_image_beads:np.ndarray, spot_size:int = 5, min_intensity:float = 100,show_plot=True):
        self.first_image_beads = first_image_beads
        self.second_image_beads = second_image_beads
        self.spot_size = spot_size
        self.min_intensity = min_intensity
        self.show_plot = show_plot
    def make_beads_alignment(self):
        '''
        This method aligns a list of spots detected in an image with dimensions [C, Y, X] and returns a homography matrix.

        Returns
        
        homography_matrix : object
            The homography matrix is a 3x3 matrix. This transformation matrix maps the points between two planes (images).
        '''
        # Applying a log filter to the image
        MIN_DISTANCE_TO_MATCH_BEADS = 4
        filtered_first_image_beads= Utilities.log_filter(self.first_image_beads, sigma=1.5)
        filtered_first_image_beads= Utilities.bandpass_filter(filtered_first_image_beads, lowfilter=0.5, highpass=10)
        filtered_second_image_beads = Utilities.log_filter(self.second_image_beads, sigma=1.5)
        filtered_second_image_beads = Utilities.bandpass_filter(filtered_second_image_beads, lowfilter=0.5, highpass=10)
        # Locating beads in the image using "tp.locate" function from trackpy.
        df0 = tp.locate(filtered_first_image_beads, diameter = self.spot_size, minmass=self.min_intensity, maxsize=self.spot_size*2, preprocess=False,max_iterations=100) # data frame for the first channel
        df1 = tp.locate(filtered_second_image_beads, diameter= self.spot_size, minmass=self.min_intensity, maxsize=self.spot_size*2, preprocess=False,max_iterations=100)  # data frame for the second channel
        # retrieving the coordinates for spots type 0 and 1 for each cell 
        array_spots_0 = np.asarray( df0[['y','x']]) # coordinates for spot_type_0 with shape [num_spots_type_0, 3]
        array_spots_1 = np.asarray( df1[['y','x']]) # coordinates for spot_type_1 with shape [num_spots_type_1, 3]
        total_spots0 = array_spots_0.shape[0]
        # Concatenating arrays from spots 0 and 1
        array_all_spots = np.concatenate((array_spots_0,array_spots_1), axis=0) 
        # Calculating a distance matrix. 
        distance_matrix = np.zeros( (array_all_spots.shape[0], array_all_spots.shape[0])) #  the distance matrix is an square matrix resulting from the concatenation of both spot  types.
        for i in range(len(array_all_spots)):
            for j in range(len(array_all_spots)):
                if j<i:
                    distance_matrix[i,j] = np.linalg.norm( ( array_all_spots[i,:]-array_all_spots[j,:] )  )
        # masking the distance matrix. Ones indicate the distance is less or equal than threshold_distance
        mask_distance_matrix = (distance_matrix <= MIN_DISTANCE_TO_MATCH_BEADS) 
        # Selecting the right-lower quadrant as a subsection of the distance matrix that compares one spot type versus the other. 
        subsection_mask_distance_matrix = mask_distance_matrix[total_spots0:, 0:total_spots0].copy()
        # Calculating each type of spots in cell
        indices = np.where(subsection_mask_distance_matrix)
        number_colocalized_spots = indices[0].shape[0]
        positions_in_first_image =  np.zeros((number_colocalized_spots, 2))
        positions_in_second_image = np.zeros((number_colocalized_spots, 2))
        for i in range (number_colocalized_spots):
            positions_in_first_image [i,:] = array_spots_0[indices[1][i],:] # indices[1] represents the columns in the right-lower quadrant as a subsection of the distance matrix that compares one spot type versus the other. 
            positions_in_second_image [i,:] = array_spots_1[indices[0][i],:] # indices[0] represents the rows in the right-lower quadrant as a subsection of the distance matrix that compares one spot type versus the other. 
        # this step changes the order of the spots form [n_spots, [Y,X]] to  [n_spots, [X,Y]]
        positions_in_first_image = positions_in_first_image[:, [1,0]]
        positions_in_second_image = positions_in_second_image[:, [1,0]]
        number_spots_first_image = positions_in_first_image.shape[0]
        number_spots_second_image = positions_in_second_image.shape[0]
        distance_matrix_after = np.zeros( (number_spots_first_image)) #  the distance matrix is an square matrix resulting from the concatenation of both spot  types.
        # This code calculates the distance matrix between the spots detected in the two images
        for i in range(number_spots_first_image):
            for j in range(number_spots_first_image):
                if j==i:
                    distance_matrix_after[i] = np.linalg.norm( ( positions_in_first_image[i,:]-positions_in_second_image[j,:] )  )
        print('sum of dist ',np.mean(distance_matrix_after))
        print('Calculating the homography matrix between the two images.')
        print('_______ ')
        print(' # Spots in first image : ', number_spots_first_image, '  # Spots in second image : ',number_spots_second_image, '\n')
        print('Spots detected in the first image: ')
        print(np.round(positions_in_first_image[0:np.min( (5, number_spots_first_image)), :] ,1))
        #print('The number of spots detected for the second image are: ', number_spots_second_image, '\n')
        print('Spots detected in the second image:')
        print(np.round(positions_in_second_image[0:np.min((5, number_spots_second_image)),:],1))
        print('_______ ')
        # Plotting the images with the detected spots
        if self.show_plot == True:
            Plots.plot_beads_alignment(self.first_image_beads,filtered_first_image_beads,positions_in_first_image, self.second_image_beads,filtered_second_image_beads,positions_in_second_image,self.spot_size )
        # Calculating the minimum value of rows for the alignment
        no_spots_for_alignment = min(positions_in_first_image.shape[0], positions_in_second_image.shape[0])
        src = positions_in_first_image[:no_spots_for_alignment, :2].reshape((no_spots_for_alignment, 2))
        dst = positions_in_second_image[:no_spots_for_alignment, :2].reshape((no_spots_for_alignment, 2))
        homography_matrix = transform.estimate_transform('projective', src, dst)
        print('')
        print('Calculated homography matrix: ')
        print (homography_matrix)
        return homography_matrix


class CamerasAlignment():
    '''
    This class is intended to align the images from multiple channels by applying a matrix transformation using the homography

    Parameters

    video : NumPy array
        Array of images with dimensions [T, Y, X, C].
    homography_matrix : object
        The homography matrix object generated with the class BeadsAlignment.
    target_channels : List of int, optional
        Lower bound to normalize intensity. The default is [1].
    '''
    def __init__(self, video:np.ndarray, homography_matrix, target_channels: list = [1]):
        self.video = video
        self.homography_matrix = homography_matrix
        if not isinstance(target_channels, list):
            target_channels =[target_channels]
        self.target_channels = target_channels
    def make_video_alignment(self):
        '''
        This method transforms the video by multiplying the target channels by the homography matrix.

        Returns

        transformed_video : np.uint16
            Transformed video. Array with dimensions [T, Y, X, C].
        '''
        transformed_video = np.zeros_like(self.video)
        number_time_points, height, width, number_channels = self.video.shape
        # Applying the alignment transformation to the whole video. Matrix multiplication to align the images from the two cameras.
        for index_channels in range(0, number_channels): # green and blue channels
            for index_time in range(0, number_time_points):
                if index_channels in self.target_channels:
                    transformed_video[index_time, :, :, index_channels] = warp(self.video[index_time, :, :, index_channels], self.homography_matrix.params, output_shape = (height, width), preserve_range = True).astype(np.uint16)
                else:
                    transformed_video[index_time, :, :, index_channels] = self.video[index_time, :, :, index_channels].astype(np.uint16)
        return transformed_video



class Cellpose():
    '''
    This class is intended to detect cells by image masking using **Cellpose** . The class uses optimization to maximize the number of cells or maximize the size of the detected cells.

    Parameters

    video : NumPy array
        Array of images with dimensions [T, Y, X].
    num_iterations : int, optional
        Number of iterations for the optimization process. The default is 5.
    channels : List, optional
        List with the channels in the image. For gray images use [0, 0], for RGB images with intensity for cytosol and nuclei use [0, 1] . The default is [0, 0].
    diameter : float, optional
        Average cell size. The default is 120.
    model_type : str, optional
        To detect between the two options: 'cyto' or 'nuclei'. The default is 'cyto'.
    selection_method : str, optional
        Option to use the optimization algorithm to maximize the number of cells or maximize the size options are 'max_area' or 'max_cells' or 'max_cells_and_area'. The default is 'max_cells_and_area'.
    '''
    def __init__(self, video:np.ndarray, num_iterations:int = 5, channels:list = [0, 0], diameter:float = 120, model_type:str = 'cyto', selection_method:str = 'max_cells_and_area',minimum_cell_area=1000):
        #self.video = video
        self.num_iterations = num_iterations
        self.minimum_probability = 0
        self.maximum_probability = 4
        self.channels = channels
        self.diameter = diameter
        self.model_type = model_type # options are 'cyto' or 'nuclei'
        self.selection_method = selection_method # options are 'max_area' or 'max_cells'
        self.optimization_parameter = np.round(np.linspace(self.minimum_probability, self.maximum_probability, self.num_iterations), 2)
        self.minimum_cell_area =minimum_cell_area
        # removing pixels that over the 98 percentile. This operation is only performed for cell segmentation.
        if len(video.shape) > 3:  
            raise ValueError ('A multicolor image is passed for segmentation. Please select a color channel for segmentation and pass the image with one of the following formats [T, Y, X] or [Y, X]')
        elif len(video.shape) ==3:
            self.video = RemoveExtrema(video, min_percentile = 0.5, max_percentile = 99,format_video='TYX').remove_outliers()
        else:
            self.video = RemoveExtrema(video, min_percentile = 0.5, max_percentile = 99,format_video='YX').remove_outliers()
                        
    def calculate_masks(self):
        '''
        This method performs the process of image masking using **Cellpose**.

        Returns

        selected_masks : List of NumPy arrays
            List of NumPy arrays with values between 0 and the number of detected cells in the image, where a number larger than zero represents the masked area for each cell, and 0 represents the area where no cells are detected.
        '''
        # Next two lines suppressing output from Cellpose
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        model = models.Cellpose(gpu = 1, model_type = self.model_type) # model_type = 'cyto' or model_type = 'nuclei'
        # Loop that test multiple probabilities in cell pose and returns the masks with the longest area.
        def cellpose_max_area( optimization_threshold):
            try:
                masks, _, _, _ = model.eval(self.video, normalize = True, cellprob_threshold = optimization_threshold, diameter = self.diameter, channels = self.channels, progress = None)
                masks=Utilities().remove_artifacts_from_mask_image(masks,minimal_mask_area_size=self.minimum_cell_area)
            except:
                masks =0
            n_masks = np.max(masks)
            if n_masks > 1: # detecting if more than 1 mask are detected per cell
                size_mask = []
                for nm in range (1, n_masks+1): # iterating for each mask in a given cell. The mask has values from 0 for background, to int n, where n is the number of detected masks.
                    size_mask.append(np.sum(masks == nm)) # creating a list with the size of each mask
                largest_mask = np.argmax(size_mask)+1 # detecting the mask with the largest value
                temp_mask = np.zeros_like(masks) # making a copy of the image
                selected_mask = temp_mask + (masks == largest_mask) # Selecting a single mask and making this mask equal to one and the background equal to zero.
                return np.sum(selected_mask)
            else: # do nothing if only a single mask is detected per image.
                return np.sum(masks)
        def cellpose_max_cells(optimization_threshold):
            try:
                masks, _, _, _ = model.eval(self.video, normalize = True, cellprob_threshold = optimization_threshold, diameter =self.diameter, channels = self.channels, progress = None)
                masks=Utilities().remove_artifacts_from_mask_image(masks,minimal_mask_area_size=self.minimum_cell_area)
            except:
                masks =0
            return np.max(masks)
        def cellpose_max_cells_and_area( optimization_threshold):
            try:
                masks, _, _, _ = model.eval(self.video, normalize = True, cellprob_threshold = optimization_threshold, diameter = self.diameter, channels = self.channels, progress = None)
                masks=Utilities().remove_artifacts_from_mask_image(masks,minimal_mask_area_size=self.minimum_cell_area)
            except:
                masks =0
            n_masks = np.max(masks)
            if n_masks > 1: # detecting if more than 1 mask are detected per cell
                size_mask = []
                for nm in range (1, n_masks+1): # iterating for each mask in a given cell. The mask has values from 0 for background, to int n, where n is the number of detected masks.
                    size_mask.append(np.sum(masks == nm)) # creating a list with the size of each mask
                number_masks= np.max(masks)
                metric = np.sum(np.asarray(size_mask)) * number_masks
                return metric
            if n_masks == 1: # do nothing if only a single mask is detected per image.
                return np.sum(masks == 1)
            else:  # return zero if no mask are detected
                return 0     
        if self.selection_method == 'max_area':
            list_metrics_masks = [cellpose_max_area(optimization_probability ) for i, optimization_probability in enumerate(self.optimization_parameter)]
            evaluated_metric_for_masks = np.asarray(list_metrics_masks)
        if self.selection_method == 'max_cells':
            list_metrics_masks = [cellpose_max_cells(optimization_probability) for i,optimization_probability in enumerate(self.optimization_parameter)]
            evaluated_metric_for_masks = np.asarray(list_metrics_masks)
        if self.selection_method == 'max_cells_and_area':
            list_metrics_masks = [cellpose_max_cells_and_area(optimization_probability ) for i,optimization_probability in enumerate(self.optimization_parameter)]
            evaluated_metric_for_masks = np.asarray(list_metrics_masks)
        
        if np.max(evaluated_metric_for_masks) >0:
            selected_conditions = self.optimization_parameter[np.argmax(evaluated_metric_for_masks)]
            selected_masks, _, _, _ = model.eval(self.video, normalize = True, cellprob_threshold = selected_conditions, diameter = self.diameter, min_size = -1, channels = self.channels, progress = None)
            selected_masks=Utilities().remove_artifacts_from_mask_image(selected_masks,minimal_mask_area_size=self.minimum_cell_area)
            selected_masks[0:10, :] = 0;selected_masks[:, 0:10] = 0;selected_masks[selected_masks.shape[0]-10:selected_masks.shape[0]-1, :] = 0; selected_masks[:, selected_masks.shape[1]-10: selected_masks.shape[1]-1 ] = 0#This line of code ensures that the corners are zeros.
        else:
            selected_masks = None
            print('No cells detected on the image \n')
        sys.stdout.close()
        sys.stdout = old_stdout
        return selected_masks


class CellposeSelection():
    '''
    This class is intended to select the cell with the maximum area (max_area) or with the maximum number of spots (max_spots) from a masked image previously detect cells by **Cellpose**

    Parameters

    mask : NumPy array
        Arrays with values between 0 and the number of detected cells in the image, where a number larger than zero represents the masked area for each cell, and 0 represents the area where no cells are detected. An array of images with dimensions [Y, X].
    video : NumPy array
        An array of images with dimensions [T, Y, X, C].
    selection_method : str, optional
        Options used by the optimization algorithm to select a cell based on the number of cells or the number of spots. The options are: 'all_cells_in_image','max_area' or 'max_spots'. The default is 'maximum_area'.
    particle_size : int, optional
        Average particle size. The default is 5.
    selected_channel : int, optional
        Channel where the particles are detected and tracked. The default is 0.
    '''
    def __init__(self, mask:np.ndarray, video:np.ndarray, selection_method:str = 'max_area', particle_size:int = 5, selected_channel:int = 0):
        self.mask = mask
        self.video = video
        self.num_frames = video.shape[0]
        self.selected_channel = selected_channel      # selected channel
        self.particle_size = particle_size            # according to the documentation it must be an even number 3, 5, 7, 9 etc.
        self.selection_method = selection_method
        self.minimal_frames = int(video.shape[0]*0.9)

    def select_mask(self):
        '''
        This method selects the cell with the maximum area (max_area) or with the maximum number of spots (max_spots) from a masked image.

        Returns

        selected_mask : NumPy array, with Boolean values, where 1 represents the masked area, and 0 represents the area outside the mask.
            An array of images with dimensions [Y, X].
        '''
        if self.selection_method == 'max_area':
            # Iterating for each mask to select the mask with the largest area.
            n_masks = np.max(self.mask)
            if n_masks > 1: # detecting if more than 1 mask are detected per cell
                size_mask = []
                for nm in range (1, n_masks+1): # iterating for each mask in a given cell. The mask has values from 0 for the background to int n, where n is the number of detected masks.
                    size_mask.append(np.sum(self.mask == nm)) # creating a list with the size of each mask
                largest_mask = np.argmax(size_mask)+1 # detecting the mask with the largest value
                temp_mask = np.zeros_like(self.mask) # making a copy of the image
                selected_mask = temp_mask + (self.mask == largest_mask) # Selecting a single mask and making this mask equal to one and the background equal to zero.
            else: # do nothing if only a single mask is detected per image.
                selected_mask = self.mask
        
        if self.selection_method == 'all_cells_in_image':
            # Returning all masks in the image
            selected_mask = self.mask
        
        if self.selection_method == 'max_spots':
            # Iterating for each mask to select the mask with the largest area.
            n_masks = np.max(self.mask)
            if n_masks > 1: # detecting if more than 1 mask are detected per cell
                number_particles = []
                for nm in range (1, n_masks+1): # iterating for each mask in a given cell. The mask has values from 0 for the background to int n, where n is the number of detected masks.
                    # # Apply mask to a given time point
                    mask_copy = np.copy(self.mask)
                    tested_mask = np.where(mask_copy == nm, 1, 0) # making zeros all elements outside each mask, and once all elements inside of each mask.
                    video_minimal_time = np.min((int(self.num_frames/3), 5, self.num_frames))
                    _, number_detected_trajectories, _ = Trackpy(self.video[0:video_minimal_time, :, :, :], tested_mask, particle_size = self.particle_size, selected_channel = self.selected_channel , minimal_frames = self.minimal_frames, show_plot = 0).perform_tracking()                    
                    number_particles.append(number_detected_trajectories)
                pre_selected_mask = np.argmax(number_particles)+1 # detecting the mask with the largest value
                temp_mask = np.zeros_like(self.mask, dtype = np.uint16) # making a copy of the image
                selected_mask = temp_mask + (self.mask == pre_selected_mask) # Selecting a single mask and making this mask equal to one and the background equal to zero.
            else:  # do nothing if only a single mask is detected per image.
                selected_mask = self.mask
            if np.max(selected_mask) == 0:
                selected_mask = None
                print('No mask was selected in the image.')
                # This section dilates the mask to connect areas that are isolated.
        if np.max(selected_mask) == 1:
            mask_int = np.where(selected_mask > 0.5, 1, 0).astype(np.int)
            dilated_image = dilation(mask_int, square(20))
            mask_final = np.where(dilated_image > 0.5, 1, 0).astype(np.int)
        else:
            mask_final = selected_mask
        mask_final[0:10, :] = 0;mask_final[:, 0:10] = 0;mask_final[mask_final.shape[0]-10:mask_final.shape[0]-1, :] = 0; mask_final[:, mask_final.shape[1]-10: mask_final.shape[1]-1 ] = 0#This line of code ensures that the corners are zeros.
        return mask_final


class Trackpy():
    '''
    This class is intended to detect spots in the video by using **Trackpy**.

    Parameters

    video : NumPy array
        Array of images with dimensions [T, Y, X, C]. In case a FISH image is used, the format must be [Z, Y, X, C], and the user must specify the parameter FISH_image = 1.
    mask : NumPy array, with Boolean values, where 1 represents the masked area, and 0 represents the area outside the mask.
        An array of images with dimensions [Y, X].
    particle_size : int, optional
        Average particle size. The default is 5.
    selected_channel : int, optional
        Channel where the particles are detected and tracked. The default is 0.
    minimal_frames : int, optional
        This parameter defines the minimal number of frames that a particle should appear on the video to be considered on the final count. The default is 5.
    optimization_iterations : int, optional
        Number of iterations for the optimization process to select the best filter. The default is 30.
    use_default_filter : bool, optional
        This option allows the user to use a default filter if TRUE. Else, it uses an optimization process to select the best filter for the image. The default is = True.
    show_plot : bool, optional
        Allows the user to show a plot for the optimization process. The default is True.
    FISH_image : bool, optional.
        This parameter allows the user to use FISH images and connect spots detected along multiple z-slices. The default is 0.
    intensity_selection_threshold_int_std : float, optional. The default is None, and it uses a default value or an optimization method if use_optimization_for_tracking is set to TRUE. 
        Threshold intensity for tracking
    intensity_threshold_tracking : float or None, optional.
        Intensity threshold value to be used during tracking. If no value is passed, the code attempts to calculate this value. The default is None.
    image_name : str , optional
        Name for the image for tracking. The default is 'temp_tracking.png'.

    '''
    def __init__(self, video:np.ndarray, mask:np.ndarray, particle_size:int = 5, selected_channel:int = 0, minimal_frames:int = 5, optimization_iterations:int = 1000, use_default_filter:bool = True, FISH_image: bool = False, show_plot:bool = True,intensity_threshold_tracking=None,image_name:str='temp_tracking.png'):
        self.NUMBER_OF_CORES = multiprocessing.cpu_count()
        self.time_points = video.shape[0]
        self.selected_channel = selected_channel
        ini_video = np.asarray(Parallel(n_jobs = self.NUMBER_OF_CORES)(delayed(Utilities.img_uint)(video[i, :, :, self.selected_channel]) for i in range(0, self.time_points)) )
        def filter_video(video, tracking_filter,frames_to_track):
            # function that remove outliers from the video
            video = RemoveExtrema(video, min_percentile = 0.5, max_percentile = 99.9,format_video='TYX').remove_outliers()
            # selecting the filter to apply
            if tracking_filter == 'bandpass_filter':
                temp_vid_dif_filter = Parallel(n_jobs = self.NUMBER_OF_CORES)(delayed(Utilities.bandpass_filter)(video[i, :, :], self.low_pass_filter, self.highpass_filter) for i in range(0, frames_to_track))
            elif tracking_filter == 'log_filter':       
                temp_vid_dif_filter = Parallel(n_jobs = self.NUMBER_OF_CORES)(delayed(Utilities.log_filter)(video[i, :, :], sigma=1.5) for i in range(0, frames_to_track))
            elif tracking_filter == 'all':  
                temp_vid_filter = Parallel(n_jobs = self.NUMBER_OF_CORES)(delayed(Utilities.bandpass_filter)(video[i, :, :], self.low_pass_filter, self.highpass_filter) for i in range(0, frames_to_track))
                temp_vid = np.asarray(temp_vid_filter)
                temp_vid_dif_filter = Parallel(n_jobs = self.NUMBER_OF_CORES)(delayed(Utilities.log_filter)(temp_vid[i, :, :], sigma=1.5) for i in range(0, frames_to_track))
            video_filtered = np.asarray(temp_vid_dif_filter)
            return video_filtered
        self.image_name=image_name
        self.intensity_threshold_tracking = intensity_threshold_tracking   
        self.mask = mask
        if (particle_size % 2) == 0:
            particle_size = particle_size + 1
            print('particle_size must be an odd number, this was automatically changed to: ', particle_size)
        self.particle_size = particle_size # according to the documentation must be an even number 3, 5, 7, 9 etc.
        if self.time_points < 10:
            self.min_time_particle_vanishes = 0
            self.max_distance_particle_moves = 5
        else:
            self.min_time_particle_vanishes = 1
            self.max_distance_particle_moves = 10
        if minimal_frames > self.time_points:     # this line is making sure that "minimal_frames" is always less or equal than the total number of frames
            minimal_frames = self.time_points
        self.minimal_frames = minimal_frames
        self.optimization_iterations = optimization_iterations
        self.show_plot = show_plot
        self.use_default_filter =  use_default_filter
        # parameters for the filters
        self.low_pass_filter = 0.1
        self.highpass_filter = 5
        self.percentile_intensity_selection = 70 #Not modify
        self.default_threshold_int_std = 1 #0.5  # very important parameter. 1 works well
        # This section detects if a FISH image is passed and it adjust accordingly.
        self.FISH_image = FISH_image
        self.MAX_INT_OPTIMIZATION_DEFAULT_VALUE = 400
        if self.FISH_image == True:
            self.min_time_particle_vanishes = 1
            self.max_distance_particle_moves = 1
            self.minimal_frames = minimal_frames
        self.tracking_filter = 'all'
        self.video_filtered = filter_video(video=ini_video, tracking_filter=self.tracking_filter,frames_to_track=self.time_points)
        self.MIN_INT_OPTIMIZATION = 1 
        if use_default_filter ==0:
            f_init = tp.locate(self.video_filtered[0, :, :], self.particle_size, minmass = 1, max_iterations = 100, preprocess = False, percentile = 70)
            if not f_init.empty:
                self.MAX_INT_OPTIMIZATION = np.max( (1, np.round( np.max(f_init.mass.values) ) ) )
            else:
                self.MAX_INT_OPTIMIZATION = self.MAX_INT_OPTIMIZATION_DEFAULT_VALUE
        else:
            self.MAX_INT_OPTIMIZATION = self.MAX_INT_OPTIMIZATION_DEFAULT_VALUE
        if self.optimization_iterations <= 10:
            self.sigma_for_1d_gaussian_filter = 1
        elif (self.optimization_iterations > 20) and (self.optimization_iterations <= 200):
            self.sigma_for_1d_gaussian_filter = 2
        else:
            self.sigma_for_1d_gaussian_filter = 5     
        
    def perform_tracking(self):
        '''
        This method performs the tracking of the particles using trackpy.

        Returns

        trackpy_dataframe : pandas data frame.
            Pandas data frame from trackpy with fields [x, y, mass, size, ecc, signal, raw_mass, ep, frame, particle].
        number_particles : int.
            The total number of detected particles in the data frame.
        video_filtered : np.uint16.
            Filtered video resulting from the bandpass process. Array with format [T, Y, X, C].
        '''
        min_int_vector = np.round(np.linspace(self.MIN_INT_OPTIMIZATION, self.MAX_INT_OPTIMIZATION, self.optimization_iterations), 0) # range of std to test for optimization
        min_int_vector = np.unique(min_int_vector)
        def spots_in_mask(dataframe,mask):
            # extracting the contours in the image
            coords = np.array([dataframe.y, dataframe.x]).T # These are the points detected by trackpy
            coords_int = np.round(coords).astype(int)  # or np.floor, depends
            values_at_coords = mask[tuple(coords_int.T)] # If 1 the value is in the mask
            dataframe['In Mask'] = values_at_coords # Check if pts are on/in polygon mask  
            return dataframe 
        
        # main function that performs the particle tracking
        def video_tracking(video, mask, min_int = None, flag_for_optimization=0):
            if min_int == None:
                f_init = tp.locate(video, self.particle_size, minmass = 0, max_iterations = 100, preprocess = False, percentile = self.percentile_intensity_selection)
                try:
                    min_int = np.max( (0, np.round( np.mean(f_init.mass.values) + self.default_threshold_int_std *np.std(f_init.mass.values))))
                except:
                    min_int = 0
            if flag_for_optimization ==1:
                trackpy_dataframe = tp.locate(video[0,:,:], self.particle_size, minmass = min_int, max_iterations = 100, preprocess = False, percentile = self.percentile_intensity_selection)
                number_particles = len(trackpy_dataframe)
            else:
                # detecting spots in all frames
                dataframe_with_spots_all_frames = tp.batch(video, self.particle_size, minmass = min_int, processes = 'auto', max_iterations = 1000, preprocess = False, percentile = self.percentile_intensity_selection)
                # Adding a column indicating if the spots are located inside the mask 
                dataframe_with_label_in_mask = spots_in_mask(dataframe_with_spots_all_frames,mask)
                # Selecting only the spots located inside the mask
                dataframe_particles_in_mask = dataframe_with_label_in_mask[dataframe_with_label_in_mask['In Mask']==True]
                # Linking particles
                dataframe_linked_particles = tp.link_df(dataframe_particles_in_mask, self.max_distance_particle_moves, memory = self.min_time_particle_vanishes, adaptive_stop = 1, link_strategy = 'auto') # tp.link_df(data_frame, min_distance_particle_moves, min_time_particle_vanish).
                # Selecting trajectories that appear in at least 10 frames.
                trackpy_dataframe = tp.filter_stubs(dataframe_linked_particles, self.minimal_frames)  
                number_particles = trackpy_dataframe['particle'].nunique()            
            return trackpy_dataframe, number_particles        
        # Tracking using a given intensity_threshold_tracking
        if not (self.intensity_threshold_tracking is None):
            trackpy_dataframe, number_particles = video_tracking(video=self.video_filtered, mask=self.mask, min_int= self.intensity_threshold_tracking)
        # This section uses optimization to select the optimal value for the filter size.
        else: 
            list_dataframe_and_number_paricles =   Parallel(n_jobs = self.NUMBER_OF_CORES)(delayed(video_tracking)(video=self.video_filtered, mask=self.mask, min_int= min_int_optimization, flag_for_optimization=1) for _ , min_int_optimization in enumerate(min_int_vector) )
            num_particles= [element[1] for element in list_dataframe_and_number_paricles] # extracting the second element (list_dataframe_and_number_paricles) in list obtained from parallel processing            
            num_detected_particles = np.asarray(num_particles)
            smooth_vector_detected_spots = gaussian_filter1d(num_detected_particles, self.sigma_for_1d_gaussian_filter)
            log_num_spots =  np.log(smooth_vector_detected_spots +1)
            NUMBER_OF_LEFT_BINS_IGNORED_FOR_OPTIMIZATION  = 40
            derivative_vector_detected_spots = np.gradient(log_num_spots[NUMBER_OF_LEFT_BINS_IGNORED_FOR_OPTIMIZATION:])      #  derivative
            index_max_second_derivative = derivative_vector_detected_spots.argmax()+NUMBER_OF_LEFT_BINS_IGNORED_FOR_OPTIMIZATION #+ self.ADDED_INDEX_TO_OPTIMIZED_SELECTION 
            selected_int_optimized = min_int_vector[index_max_second_derivative]  # + self.ADDED_INTENSITY_TO_OPTIMIZED_SELECTION
            trackpy_dataframe, number_particles = video_tracking(video=self.video_filtered, mask=self.mask, min_int= selected_int_optimized)
            if self.show_plot ==True:
                plt.figure(figsize =(5,5))
                plt.plot(min_int_vector, log_num_spots, label='norm detected_spots',linewidth=5,color='lime')
                plt.plot(min_int_vector[index_max_second_derivative], log_num_spots[index_max_second_derivative], 'o',label='selected threshold', markersize=20, markerfacecolor='orangered')
                plt.xlabel('Threshold intensity', size=16)
                plt.ylabel('log (number of spots)', size=16)
                plt.savefig(self.image_name,bbox_inches='tight')
                plt.show()
                print('The number of detected trajectories is: ', number_particles)
                print('The selected intensity threshold is: ', str(selected_int_optimized), '\n' )
        video_filtered = np.expand_dims(self.video_filtered,axis=3)
        return trackpy_dataframe, int(number_particles), video_filtered


class ParticleMotion():
    '''
    This class is intended to calculate the mean square displacement in the detected spots. This class uses trackpy.motion.emsd. 

    Parameters

    trackpy_dataframe : pandas data frame or None (if not given).
        Pandas data frame from trackpy with fields [x, y, mass, size, ecc, signal, raw_mass, ep, frame, particle]. The default is None
    
    microns_per_pixel : float
        Factors to converts microns per pixel.
    step_size_in_sec : float
        Factor that sets the frames per second in the video.
    max_lagtime : int, optional
        Is the time interval used to compute the mean square displacement. Trackpy uses a default of 100.
    show_plot : bool, optional
        If True, it displays a plot of MSD vs time. The default is True.
    remove_drift : bool, optional
        This flags removes the drift in the dataframe.
        
    '''
    def __init__(self,trackpy_dataframe,microns_per_pixel,step_size_in_sec,max_lagtime=100,show_plot=True,remove_drift=False):
        self.trackpy_dataframe= trackpy_dataframe
        self.microns_per_pixel = microns_per_pixel
        self.step_size_in_sec = step_size_in_sec
        self.max_lagtime = max_lagtime
        self.show_plot = show_plot 
        self.remove_drift=remove_drift
        
    def calculate_msd(self):
        '''
        This method calculates the MSD for all the spots in the dataframe.

        Returns
        calculated_diffusion_coefficient : float
            Calculated mean square displacement for the particle ensemble.
        MSD_series : pandas dataframe
            Dataframe with fields [msd, time].

        '''
        # This section removes drift from the original dataframe before calculating the MSD.
        if self.remove_drift == True:
            temp_trackpy_df = self.trackpy_dataframe.copy()
            drift = tp.compute_drift(temp_trackpy_df)
            trackpy_df = tp.subtract_drift(temp_trackpy_df.copy(), drift)
            if self.show_plot==True: 
                drift.plot()
                plt.show()
        else:
            trackpy_df = self.trackpy_dataframe.copy()
        # This section calculates the Mean square displacement for all the particles in the system.
        #MSD_series = tp.emsd(trackpy_df, mpp = self.microns_per_pixel, fps = self.step_size_in_sec, max_lagtime = self.max_lagtime )
        # This section calculates the mean square displacement in the image.
        # for this it use the formula At^n
        # where n is approximated to 1
        # the value of A = 4D where D is the diffusion coefficient. 4 comes from Einstein equation for particle diffusion in a 2D system.
        em = tp.emsd(trackpy_df,mpp=self.microns_per_pixel, fps=self.step_size_in_sec,  max_lagtime = self.max_lagtime)
        slope =  np.linalg.lstsq(em.index[:, np.newaxis],em)[0][0]
        calculated_diffusion_coefficient = slope / 4 
        print(r'The diffusion constant is {0:.5f} μm²/s'.format(calculated_diffusion_coefficient))
        if self.show_plot == True:
            plt.style.use(['default', 'fivethirtyeight'])
            fig, ax = plt.subplots(figsize=(6,4))
            ax = em.plot(style='o', label='MSD')
            time_range = np.arange(start=0, stop= self.max_lagtime/self.step_size_in_sec, step = 1 ) 
            model_fit = slope * time_range
            ax.plot(time_range       , slope * time_range , label='linear fit')
            ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]', xlabel='lag time $t$')
            #ax.set(xlim=(0, 100))
            ax.legend(loc='upper left')
        return calculated_diffusion_coefficient, em, time_range, model_fit, trackpy_df
    
class Intensity():
    '''
    This class is intended to calculate the intensity in the detected spots.

    Parameters

    video : NumPy array
        Array of images with dimensions [T, Y, X, C].
    particle_size : int, optional
        Average particle size. The default is 5.
    trackpy_dataframe : pandas data frame or None (if not given).
        Pandas data frame from trackpy with fields [x, y, mass, size, ecc, signal, raw_mass, ep, frame, particle]. The default is None
    spot_positions_movement : NumPy array  or None (if not given).
        Array of images with dimensions [T, S, y_x_positions].  The default is None
    dataframe_format : str, optional
        Format for the dataframe the options are : 'short' , and 'long'. The default is 'short'.
        "long" format generates this dataframe: [image_number, cell_number, particle, frame, ch0_int_mean, ch1_int_mean, ch2_int_mean, ch0_int_std, ch1_int_std, ch2_int_std, x, y, ch0_SNR,ch1_SNR,ch2_SNR].
        "short" format generates this dataframe: [image_number, cell_number, particle, frame, ch0_int_mean, ch1_int_mean, ch2_int_mean, x, y].
    method : str, optional
        Method to calculate intensity the options are : 'total_intensity' , 'disk_donut' and 'gaussian_fit'. The default is 'disk_donut'.
    step_size : float, optional
        Frame rate in seconds. The default is 1 frame per second.
    show_plot : bool, optional
        Allows the user to show a plot for the optimization process. The default is True.
    '''
    def __init__(self, video:np.ndarray, particle_size:int = 5, trackpy_dataframe: Union[object , None ] = None, spot_positions_movement: Union[np.ndarray, None] = None,dataframe_format:str = 'short',   method:str = 'disk_donut', step_size:float = 1, show_plot:bool = True,cell_counter:int=0,image_index:int=0):
        if particle_size < 3:
            particle_size = 3 # minimal size allowed for detection
        if (particle_size % 2) == 0:
            particle_size = particle_size + 1
            print('particle_size must be an odd number, this was automatically changed to: ', particle_size)
        self.video = video
        self.trackpy_dataframe = trackpy_dataframe
        self.disk_size = int(np.round(particle_size/2)) # size of the half of the crop
        self.crop_size = int(np.round(particle_size/2))*2
        self.spots_range_to_replace = np.linspace(-(particle_size - 1) / 2, (particle_size - 1) / 2, particle_size,dtype=int)
        PIXELS_AROUND_SPOT = 6 # THIS HAS TO BE AN EVEN NUMBER
        self.crop_range_to_replace = np.linspace(-(particle_size+PIXELS_AROUND_SPOT - 1) / 2, (particle_size+PIXELS_AROUND_SPOT - 1) / 2, particle_size+PIXELS_AROUND_SPOT,dtype=int)
        self.show_plot = show_plot
        self.dataframe_format = dataframe_format # options are : 'short' and 'long'
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
        self.step_size = step_size
        #If the video is longer than 1000 frames  avoid using parallel computing. Necessary to avoid issues with the memory.
        if video.shape[0] < 1000:
            self.NUMBER_OF_CORES = multiprocessing.cpu_count()
        else:
            self.NUMBER_OF_CORES =1
            
        self.cell_counter = cell_counter
        self.image_index=image_index
    
    def calculate_intensity(self):
        '''
        This method calculates the spot intensity.

        Returns

        dataframe_particles : pandas dataframe
            Dataframe with fields [image_number, cell_number, particle, frame, ch0_int_mean, ch1_int_mean, ch2_int_mean, ch0_int_std, ch1_int_std, ch2_int_std, x, y, ch0_SNR,ch1_SNR,ch2_SNR].
        array_intensities_mean : Numpy array
            Array with dimensions [S, T, C].
        time_vector : Numpy array
            1D array.
        mean_intensities: Numpy array
            Array with dimensions [S, T, C].
        std_intensities : Numpy array
            Array with dimensions [S, T, C].
        mean_intensities_normalized : Numpy array
            Array with dimensions [S, T, C].
        std_intensities_normalized : Numpy array
            Array with dimensions [S, T, C].
        '''
        
        time_points, number_channels  = self.video.shape[0], self.video.shape[3]
        array_intensities_mean = np.zeros((self.n_particles, time_points, number_channels))*np.nan
        array_intensities_std = np.zeros((self.n_particles, time_points, number_channels))*np.nan
        array_intensities_snr = np.zeros((self.n_particles, time_points, number_channels))*np.nan
        array_intensities_background_mean = np.zeros((self.n_particles, time_points, number_channels))*np.nan
        array_intensities_background_std = np.zeros((self.n_particles, time_points, number_channels))*np.nan
        def gaussian_fit(test_im):
            size_spot = test_im.shape[0]
            image_flat = test_im.ravel()
            def gaussian_function(size_spot, offset, sigma):
                ax = np.linspace(-(size_spot - 1) / 2., (size_spot - 1) / 2., size_spot)
                xx, yy = np.meshgrid(ax, ax)
                kernel =  offset *(np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma)))
                return kernel.ravel()
            p0 = (np.min(image_flat) , np.std(image_flat) ) # int(size_spot/2))
            optimized_parameters, _ = curve_fit(gaussian_function, size_spot, image_flat, p0 = p0)
            spot_intensity_gaussian = optimized_parameters[0] # Amplitude
            spot_intensity_gaussian_std = optimized_parameters[1]
            return spot_intensity_gaussian, spot_intensity_gaussian_std
        
        def return_crop(image:np.ndarray, x:int, y:int,spot_range):
            crop_image = image[y+spot_range[0]:y+(spot_range[-1]+1), x+spot_range[0]:x+(spot_range[-1]+1)].copy()
            return crop_image

        def reduce_dataframe(df,number_channels):
            # This function creates a list with the column names for the dataframe. This list values with the number_channels.
            list_columns_to_drop = ['ch'+str(c)+'_int_std' for c in range(0, number_channels)] + ['ch'+str(c)+'_SNR' for c in range(0, number_channels)]  + ['ch'+str(c)+'_bg_int_mean' for c in range(0, number_channels)] + ['ch'+str(c)+'_bg_int_std' for c in range(0, number_channels)]
            # This function is intended to reduce the columns that are not used in the ML process.
            return df.drop(list_columns_to_drop, axis = 1)  
            #return df.drop(['ch0_int_std', 'ch1_int_std','ch2_int_std','ch0_SNR', 'ch1_SNR', 'ch2_SNR','ch0_bg_int_mean','ch1_bg_int_mean','ch2_bg_int_mean','ch0_bg_int_std','ch1_bg_int_std','ch2_bg_int_std'], axis = 1)
        
        def return_donut(image, spot_size):
            tem_img = image.copy().astype('float')
            center_coordinates = int(tem_img.shape[0]/2)
            range_to_replace = np.linspace(-(spot_size - 1) / 2, (spot_size - 1) / 2, spot_size,dtype=int)
            min_index = center_coordinates+range_to_replace[0]
            max_index = (center_coordinates +range_to_replace[-1])+1
            tem_img[min_index: max_index , min_index: max_index] *= np.nan
            removed_center_flat = tem_img.copy().flatten()
            donut_values = removed_center_flat[~np.isnan(removed_center_flat)]
            return donut_values.astype('uint16')
        
        def signal_to_noise_ratio(values_disk,values_donut):
            mean_intensity_disk = np.mean(values_disk.flatten().astype('float'))            
            mean_intensity_donut = np.mean(values_donut.flatten().astype('float')) # mean calculation ignoring zeros
            std_intensity_donut = np.std(values_donut.flatten().astype('float')) # mean calculation ignoring zeros
            SNR = (mean_intensity_disk-mean_intensity_donut) / std_intensity_donut
            mean_background_int = mean_intensity_donut
            std_background_int = std_intensity_donut
            return SNR, mean_background_int,std_background_int
        
        def disk_donut(values_disk, values_donut):
            mean_intensity_disk = np.mean(values_disk.flatten().astype('float'))
            spot_intensity_disk_donut_std = np.std(values_disk.flatten().astype('float'))
            mean_intensity_donut = np.mean(values_donut.flatten().astype('float')) # mean calculation ignoring zeros
            spot_intensity_disk_donut = mean_intensity_disk - mean_intensity_donut
            #spot_intensity_disk_donut[np.isnan(spot_intensity_disk_donut)] = 0 # replacing nans with zero
            return spot_intensity_disk_donut, spot_intensity_disk_donut_std
        
        # Section that marks particles if a numpy array with spot positions is passed.
        def intensity_from_position_movement(particle_index , frames_part ,time_points, number_channels ):
            intensities_mean = np.zeros((time_points, number_channels))*np.nan
            intensities_std = np.zeros((time_points, number_channels))*np.nan
            intensities_snr = np.zeros((time_points, number_channels))*np.nan
            intensities_background_mean = np.zeros((time_points, number_channels))*np.nan
            intensities_background_std = np.zeros((time_points, number_channels))*np.nan
            for j in range(0, frames_part):
                for i in range(0, number_channels):
                    x_pos = int(np.round(self.spot_positions_movement[j,particle_index, 1]))
                    y_pos = int(np.round(self.spot_positions_movement[j,particle_index, 0]))
                    crop_with_disk_and_donut = return_crop(self.video[j, :, :, i], x_pos, y_pos,spot_range=self.crop_range_to_replace) # 
                    values_disk = return_crop(self.video[j, :, :, i], x_pos, y_pos, spot_range=self.spots_range_to_replace) 
                    values_donut = return_donut( crop_with_disk_and_donut,spot_size=self.particle_size )
                    intensities_snr[j, i]  , intensities_background_mean [j, i], intensities_background_std [j, i] = signal_to_noise_ratio(values_disk,values_donut) # SNR
                    if self.method == 'disk_donut':
                        intensities_mean[j, i], intensities_std[j, i] = disk_donut(values_disk,values_donut )
                    elif self.method == 'total_intensity':
                        intensities_mean[j, i] = np.max((0, np.mean(values_disk)))# mean intensity in the crop
                        intensities_std[ j, i] = np.max((0, np.std(values_disk)))# std intensity in the crop
                    elif self.method == 'gaussian_fit':
                        intensities_mean[j, i], intensities_std[j, i] = gaussian_fit(values_disk)# disk_donut(crop_image, self.disk_size
            return intensities_mean, intensities_std, intensities_snr, intensities_background_mean, intensities_background_std
        def intensity_from_dataframe(particle_index ,time_points, number_channels ):
            frames_part = self.trackpy_dataframe.loc[self.trackpy_dataframe['particle'] == self.trackpy_dataframe['particle'].unique()[particle_index]].frame.values
            intensities_mean = np.zeros((time_points, number_channels))*np.nan
            intensities_std = np.zeros((time_points, number_channels))*np.nan
            intensities_snr = np.zeros((time_points, number_channels))*np.nan
            intensities_background_mean = np.zeros((time_points, number_channels))*np.nan
            intensities_background_std = np.zeros((time_points, number_channels))*np.nan
            for j in range(0, len(frames_part)):
                for i in range(0, number_channels):
                    current_frame = frames_part[j]
                    try:
                        x_pos = int(np.round(self.trackpy_dataframe.loc[self.trackpy_dataframe['particle'] == self.trackpy_dataframe['particle'].unique()[particle_index]].x.values[j]))
                        y_pos = int(np.round(self.trackpy_dataframe.loc[self.trackpy_dataframe['particle'] == self.trackpy_dataframe['particle'].unique()[particle_index]].y.values[j]))
                    except:
                        x_pos = int(np.round(self.trackpy_dataframe.loc[self.trackpy_dataframe['particle'] == self.trackpy_dataframe['particle'].unique()[particle_index]].x.values[frames_part[0]]))
                        y_pos = int(np.round(self.trackpy_dataframe.loc[self.trackpy_dataframe['particle'] == self.trackpy_dataframe['particle'].unique()[particle_index]].y.values[frames_part[0]]))
                    crop_with_disk_and_donut = return_crop(self.video[j, :, :, i], x_pos, y_pos,spot_range=self.crop_range_to_replace) # 
                    values_disk = return_crop(self.video[j, :, :, i], x_pos, y_pos, spot_range=self.spots_range_to_replace) 
                    values_donut = return_donut( crop_with_disk_and_donut ,spot_size=self.particle_size)
                    intensities_snr[current_frame, i] , intensities_background_mean [current_frame, i], intensities_background_std [current_frame, i] = signal_to_noise_ratio(values_disk,values_donut) # SNR
                    if self.method == 'disk_donut':
                        intensities_mean[current_frame, i], intensities_std[current_frame, i] = disk_donut(values_disk, values_donut )
                    elif self.method == 'total_intensity':
                        intensities_mean[ current_frame, i] = np.max((0, np.mean(values_disk))) # mean intensity in image
                        intensities_std[ current_frame, i] = np.max((0, np.std(values_disk))) # std intensity in image
                    elif self.method == 'gaussian_fit':
                        intensities_mean[current_frame, i], intensities_std[ current_frame, i] = gaussian_fit(values_disk)# disk_donut(crop_image, disk_size)
            #intensities_mean[np.isnan(intensities_mean)] = 0 # replacing nans with zeros
            #intensities_std[np.isnan(intensities_std)] = 0 # replacing nans with zeros
            return intensities_mean, intensities_std, intensities_snr , intensities_background_mean, intensities_background_std
        if not ( self.spot_positions_movement is None):
            frames_part = self.spot_positions_movement.shape[0]
            list_intensities_mean_std_snr = Parallel(n_jobs = self.NUMBER_OF_CORES)(delayed(intensity_from_position_movement)(i,frames_part, time_points, number_channels  ) for i in range(0,  self.n_particles))
            array_intensities_mean = np.asarray([list_intensities_mean_std_snr[i][0]  for i in range(0,len(list_intensities_mean_std_snr))]   )
            array_intensities_std = np.asarray([list_intensities_mean_std_snr[i][1]  for i in range(0,len(list_intensities_mean_std_snr))]   )
            array_intensities_snr = np.asarray([list_intensities_mean_std_snr[i][2]  for i in range(0,len(list_intensities_mean_std_snr))]   )
            array_intensities_background_mean = np.asarray([list_intensities_mean_std_snr[i][3]  for i in range(0,len(list_intensities_mean_std_snr))]   )
            array_intensities_background_std = np.asarray([list_intensities_mean_std_snr[i][4]  for i in range(0,len(list_intensities_mean_std_snr))]   )        
        if not ( self.trackpy_dataframe is None):
            list_intensities_mean_std_snr = Parallel(n_jobs = self.NUMBER_OF_CORES)(delayed(intensity_from_dataframe)(i, time_points, number_channels  ) for i in range(0,  self.n_particles))
            array_intensities_mean = np.asarray([list_intensities_mean_std_snr[i][0]  for i in range(0,len(list_intensities_mean_std_snr))]   )
            array_intensities_std = np.asarray([list_intensities_mean_std_snr[i][1]  for i in range(0,len(list_intensities_mean_std_snr))]   )
            array_intensities_snr = np.asarray([list_intensities_mean_std_snr[i][2]  for i in range(0,len(list_intensities_mean_std_snr))]   )
            array_intensities_background_mean = np.asarray([list_intensities_mean_std_snr[i][3]  for i in range(0,len(list_intensities_mean_std_snr))]   )
            array_intensities_background_std = np.asarray([list_intensities_mean_std_snr[i][4]  for i in range(0,len(list_intensities_mean_std_snr))]   )
        # Calculate mean intensities.
        mean_intensities = np.nanmean(array_intensities_mean, axis = 0, dtype = np.float32)
        std_intensities = np.nanstd(array_intensities_mean, axis = 0, dtype = np.float32)
        # Calculate mean intensities normalized
        array_mean_intensities_normalized = np.zeros_like(array_intensities_mean)*np.nan
        for k in range (0, self.n_particles):
                for i in range(0, number_channels):
                    if np.nanmax( array_intensities_mean[k, :, i]) > 0:
                        array_mean_intensities_normalized[k, :, i] = array_intensities_mean[k, :, i]/ np.nanmax( array_intensities_mean[k, :, i])
        mean_intensities_normalized = np.nanmean(array_mean_intensities_normalized, axis = 0, dtype = np.float32)
        #mean_intensities_normalized = np.nan_to_num(mean_intensities_normalized)
        std_intensities_normalized = np.nanstd(array_mean_intensities_normalized, axis = 0, dtype = np.float32)
        #std_intensities_normalized = np.nan_to_num(std_intensities_normalized)
        time_vector = np.arange(0, time_points, 1)*self.step_size
        
        if (self.show_plot == True) and not(self.trackpy_dataframe is None):
            Plots.plot_tracking_spots(self.trackpy_dataframe, mean_intensities, mean_intensities_normalized, array_intensities_mean, std_intensities, std_intensities_normalized, self.step_size,time_points)
            
        # Initialize a dataframe
        # init_constant_dataframe = {'image_number': [], 
        #     'cell_number': [], 
        #     'particle': [], 
        #     'frame': [], 
        #     'x': [], 
        #     'y': []}
        # dataframe_particles_constant = pd.DataFrame(init_constant_dataframe)
        
        list_constant_fields = ['image_number', 'cell_number', 'particle', 'frame', 'x', 'y']
        dataframe_particles_constant = pd.DataFrame(columns=list_constant_fields)
        number_constant_columns = len(list_constant_fields)
        
        list_variable_fields =  ['ch'+str(c)+'_int_mean' for c in range(0, number_channels)] + \
                                        ['ch'+str(c)+'_int_std' for c in range(0, number_channels)] + \
                                        ['ch'+str(c)+'_SNR' for c in range(0, number_channels)] + \
                                        ['ch'+str(c)+'_bg_int_mean' for c in range(0, number_channels)] + \
                                        ['ch'+str(c)+'_bg_int_std' for c in range(0, number_channels)]
        dataframe_particles_variable = pd.DataFrame(columns=list_variable_fields)
        number_variable_columns = len(list_variable_fields) 
        
        complete_dataframe = pd.concat([dataframe_particles_constant, dataframe_particles_variable], axis=1)        
        number_total_columns = number_constant_columns + number_variable_columns    
    
        
        #array_complete = np.zeros((1,number_total_columns))
        #init_variable_dataframe ={
        #    'ch0_int_mean': [], 
        #     'ch1_int_mean': [], 
        #     'ch2_int_mean': [], 
        #     'ch0_int_std': [], 
        #     'ch1_int_std': [], 
        #     'ch2_int_std': [], 
        #     'ch0_SNR':[],
        #     'ch1_SNR':[],
        #     'ch2_SNR':[],
        #     'ch0_bg_int_mean':[],
        #     'ch1_bg_int_mean':[],
        #     'ch2_bg_int_mean':[],
        #     'ch0_bg_int_std':[],
        #     'ch1_bg_int_std':[],
        #     'ch2_bg_int_std':[]
        # }
        #dataframe_particles_variable = pd.DataFrame(init_variable_dataframe)
        dataframe_particles = complete_dataframe.copy()
        # Iterate for each spot and save time courses in the data frame
        counter = 0
        for id in range (0, self.n_particles):
            # Loop that populates the dataframes
            if not ( self.trackpy_dataframe is None):
                temporal_frames_vector = np.around(self.trackpy_dataframe.loc[self.trackpy_dataframe['particle'] == self.trackpy_dataframe['particle'].unique()[id]].frame.values)  # time_(sec)
                temporal_x_position_vector = self.trackpy_dataframe.loc[self.trackpy_dataframe['particle'] == self.trackpy_dataframe['particle'].unique()[id]].x.values
                temporal_y_position_vector = self.trackpy_dataframe.loc[self.trackpy_dataframe['particle'] == self.trackpy_dataframe['particle'].unique()[id]].y.values
            elif not ( self.spot_positions_movement is None):
                counter_time_vector = np.arange(0, time_points, 1)
                temporal_frames_vector = counter_time_vector
                temporal_x_position_vector = self.spot_positions_movement[:, id, 1]
                temporal_y_position_vector = self.spot_positions_movement[:, id, 0]
            else:
                temporal_frames_vector = np.array([1])
                temporal_x_position_vector = 0
                temporal_y_position_vector = 0
            
            temporal_image_number_vector = [self.image_index] * len(temporal_frames_vector)
            temporal_cell_number_vector = [self.cell_counter] * len(temporal_frames_vector)
            temporal_spot_number_vector = [counter] * len(temporal_frames_vector)
            
            # Prealocating memory
            array_complete = np.zeros((len(temporal_frames_vector),number_total_columns)).astype(float)    
            array_complete[:,0] = temporal_image_number_vector # image_number' 
            array_complete[:,1] = temporal_cell_number_vector # cell_number
            array_complete[:,2] = temporal_spot_number_vector # particle
            array_complete[:,3] = temporal_frames_vector*self.step_size # 'frame'
            array_complete[:,4] = temporal_x_position_vector #     'x'
            array_complete[:,5] = temporal_y_position_vector #     'y'
            
            # Populating fields for all colors
            for c in range(0, number_channels):
                array_complete[:,number_constant_columns+c]   = array_intensities_mean[id, temporal_frames_vector, c]   # mean intensities
                array_complete[:,number_constant_columns+number_channels+c] = array_intensities_std[id, temporal_frames_vector, c]   # std intensities
                array_complete[:,number_constant_columns+number_channels*2+c] = array_intensities_snr[id, temporal_frames_vector, c]   # SNR 
                array_complete[:,number_constant_columns+number_channels*3+c] = array_intensities_background_mean[id, temporal_frames_vector, c]   # BG mean intensities
                array_complete[:,number_constant_columns+number_channels*4+c] = array_intensities_background_std[id, temporal_frames_vector, c]   # BG std 
            
            # temporal_ch0_vector =  array_intensities_mean[id, temporal_frames_vector, 0]  # ch0
            # temporal_ch1_vector = array_intensities_mean[id, temporal_frames_vector, 1]  # ch1
            # temporal_ch2_vector =  array_intensities_mean[id, temporal_frames_vector, 2]  # ch2
            
            # temporal_ch0_vector_std =  array_intensities_std[id, temporal_frames_vector, 0]  # ch0
            # temporal_ch1_vector_std =  array_intensities_std[id, temporal_frames_vector, 1]  # ch1
            # temporal_ch2_vector_std =  array_intensities_std[id, temporal_frames_vector, 2]  # ch2
            
            # temporal_ch0_SNR =  array_intensities_snr[id, temporal_frames_vector, 0] # ch0
            # temporal_ch1_SNR =  array_intensities_snr[id, temporal_frames_vector, 1]  # ch1
            # temporal_ch2_SNR =  array_intensities_snr[id, temporal_frames_vector, 2]  # ch2
            
            # temporal_ch0_bg_int_mean  = array_intensities_background_mean [id, temporal_frames_vector, 0]  # ch0
            # temporal_ch1_bg_int_mean = array_intensities_background_mean [id, temporal_frames_vector, 1]  # ch1
            # temporal_ch2_bg_int_mean=  array_intensities_background_mean [id, temporal_frames_vector, 2]  # ch2
            
            # temporal_ch0_bg_int_std  = array_intensities_background_std[id, temporal_frames_vector, 0]  # ch0
            # temporal_ch1_bg_int_std = array_intensities_background_std[id, temporal_frames_vector, 1]  # ch1
            # temporal_ch2_bg_int_std = array_intensities_background_std[id, temporal_frames_vector, 2]  # ch2
            
            
            # Section that append the information for each spots
            # temp_data_frame = {'image_number': temporal_image_number_vector, 
            #     'cell_number': temporal_cell_number_vector, 
            #     'particle': temporal_spot_number_vector, 
            #     'frame': temporal_frames_vector*self.step_size, 
            #     'x': temporal_x_position_vector, 
            #     'y': temporal_y_position_vector,
            #     'ch0_int_mean': np.round( temporal_ch0_vector ,2), 
            #     'ch1_int_mean': np.round( temporal_ch1_vector ,2), 
            #     'ch2_int_mean': np.round( temporal_ch2_vector ,2), 
            #     'ch0_int_std': np.round( temporal_ch0_vector_std ,2), 
            #     'ch1_int_std': np.round( temporal_ch1_vector_std ,2), 
            #     'ch2_int_std': np.round( temporal_ch2_vector_std, 2), 
            #     'ch0_SNR' : np.round( temporal_ch0_SNR ,2),
            #     'ch1_SNR': np.round( temporal_ch1_SNR ,2),
            #     'ch2_SNR': np.round( temporal_ch2_SNR ,2),
            #     'ch0_bg_int_mean': np.round( temporal_ch0_bg_int_mean ,2),
            #     'ch1_bg_int_mean': np.round( temporal_ch1_bg_int_mean ,2),
            #     'ch2_bg_int_mean': np.round( temporal_ch2_bg_int_mean ,2),
            #     'ch0_bg_int_std': np.round( temporal_ch0_bg_int_std ,2),
            #     'ch1_bg_int_std': np.round( temporal_ch1_bg_int_std ,2),
            #     'ch2_bg_int_std': np.round( temporal_ch2_bg_int_std ,2) }
            counter += 1
            # temp_DataFrame = pd.DataFrame(temp_data_frame)
            
            dataframe_particles = dataframe_particles.append(pd.DataFrame(array_complete, columns=complete_dataframe.columns), ignore_index=True)
            #dataframe_particles = dataframe_particles.append(temp_DataFrame, ignore_index = True)
            dataframe_particles = dataframe_particles.astype({"image_number": int, "cell_number": int, "particle": int, "frame": int, "x": int, "y": int}) # specify data type as integer for some columns

        if self.dataframe_format == 'short':
            dataframe_particles = reduce_dataframe(dataframe_particles,number_channels)
        return dataframe_particles, array_intensities_mean, time_vector, mean_intensities, std_intensities, mean_intensities_normalized, std_intensities_normalized


class Covariance():
    '''
    This class calculates the auto-covariance from an intensity time series.
    
    Parameters

    dataframe_particles : pandas dataframe
        Dataframe with fields [image_number, cell_number, particle, frame, ch0_int_mean, ch1_int_mean, ch2_int_mean, ch0_int_std, ch1_int_std, ch2_int_std, x, y, ch0_SNR, ch1_SNR, ch2_SNR].
    selected_field : str, optinal
        Field in the datafre to be used to calculate autocovariance. The deafault is 'ch1_int_mean'
    max_lagtime : int, optional
        Is the time interval used to compute the mean square displacement. Trackpy uses a default of 100.
    show_plot : bool, optional
        If True, it displays a plot of Covariance vs time. The default is True.
    step_size : int, optional.
        Frame rate in seconds. The default is 1 frame per second.

    '''
    def __init__(self, intensity_array =None, dataframe_particles=None,selected_field='ch1_int_mean', max_lagtime= 100, show_plot= True,figure_size=(6,4),step_size=1):
        self.intensity_array = intensity_array
        self.dataframe_particles = dataframe_particles
        self.max_lagtime = max_lagtime
        self.show_plot = show_plot
        self.selected_field = selected_field
        self.figure_size=figure_size
        self.step_size = step_size
    
        
    def calculate_autocovariance(self):
        '''
        Method that runs the simulations for the multiplexing experiment.

        Returns

        mean_acf_data : NumPy array
            2D Array with dimensions [particle, Time]
        err_acf_data : NumPy array 
            2D Array with dimensions [particle, Time]
        lags : NumPy array
            1D Array with dimensions [Time]
        decorrelation_time : float
            Float indicating the decorrelation time.
        auto_correlation_matrix : Numpy array
            2D Array with dimenssions [particle, lags].

        '''
        
        def df_to_array(dataframe_simulated_cell,selected_field):
            '''
            This function takes the dataframe and extracts the information from it. 

            Input
                dataframe_simulated_cell : pandas dataframe
                    Dataframe with fields [image_number, cell_number, particle, frame, ch0_int_mean, ch1_int_mean, ch2_int_mean, ch0_int_std, ch1_int_std, ch2_int_std, x, y, ch0_SNR, ch1_SNR, ch2_SNR].
                selected_field : str,
                    selected field to extract data.

            Returns

                intensity_array : Intensities for each particle in the green channel. NumPy array with dimensions [number_particles, time_points]. The maximum time points are defined by the longest trajectory. Short trajectories are populated with zeros.
            '''
            # get the total number of particles in all cells
            total_particles = 0
            for cell in set(dataframe_simulated_cell['image_number']):
                total_particles += len(set(dataframe_simulated_cell[dataframe_simulated_cell['image_number'] == 0]['particle'] ))
            #preallocate numpy array sof n_particles by nframes
            intensity_array = np.zeros([total_particles, np.max(dataframe_simulated_cell['frame'])+1] )  #intensity green
            k = 0
            # For loops that iterate for each particle and stores the data in the previously pre-alocated arrays.
            for cell in set(dataframe_simulated_cell['image_number']):  #for every cell 
                for particle in set(dataframe_simulated_cell[dataframe_simulated_cell['image_number'] == 0]['particle'] ): #for every particle
                    temporal_dataframe = dataframe_simulated_cell[(dataframe_simulated_cell['image_number'] == cell) & (dataframe_simulated_cell['particle'] == particle)]  #slice the dataframe
                    frame_values = temporal_dataframe['frame'].values
                    intensity_array[k, frame_values] = temporal_dataframe[selected_field].values  #fill the arrays to return out
                    k+=1 #iterate over k (total particles)
            return intensity_array 
        
        def get_autocorrelation(data, g0='G0', norm='individual'):
            n_traj = data.shape[0]
            acf_vec = np.zeros(data.shape)
            def get_acc_fft(signal):
                N = len(signal)
                fvi = np.fft.fft(signal, n=2*N)
                acf = fvi*np.conjugate(fvi)
                acf = np.fft.ifft(acf)
                acf = np.real(acf[:N])/float(N)
                return acf
            global_mean = np.mean(data)
            global_var = np.var(data)
            for i in range(n_traj):
                if norm == 'individual':
                    if np.mean(data[i] == 0):
                        signal = (data[i] - 1e-6) / 1e-6
                    else:
                        signal = (data[i] - np.mean(data[i])) / np.var(data[i])
                else:
                    signal = (data[i] - global_mean) / global_var
                if g0 == 'G1':
                    g1 = get_acc_fft(signal)[1]
                    acf_vec[i] = get_acc_fft(signal)/g1
                if g0 == 'G0':
                    g = get_acc_fft(signal)[0]
                    acf_vec[i] = get_acc_fft(signal)/g
                # returns an autocorrelation matrix with the shape [particle, lags]
            return acf_vec

        if not (self.intensity_array is None):
            intensity_array = self.intensity_array
        elif not (self.dataframe_particles is None):
            intensity_array = df_to_array(self.dataframe_particles, selected_field=self.selected_field )
        auto_correlation_matrix = get_autocorrelation(intensity_array, g0='G0', norm='individual')
        mean_acf_data = np.mean(auto_correlation_matrix, axis=0)
        err_acf_data = np.std(auto_correlation_matrix, axis=0) 
        lags = np.arange(start=0, stop=len(mean_acf_data)*self.step_size, step=self.step_size)
        number_lags= lags.shape[0]
        print('nlags',number_lags)
        try:
            decorrelation_time = lags[np.where(mean_acf_data<0)[0][0]]
            print('The dwell (decorrelation) time is ', str(decorrelation_time) , 'seconds')
        except:
            print('It was not possible to estimate the dwell (decorrelation) times.')
            decorrelation_time = 0
        # Plotting the ACF and its mean.
        if self.show_plot == True:
            fig, ax = plt.subplots(1,1, figsize=self.figure_size)
            ax.fill_between(lags, mean_acf_data - err_acf_data, mean_acf_data + err_acf_data, color='grey', alpha=0.3)
            ax.plot(lags, mean_acf_data, '-', linewidth=3, color='#1C00FE', label='mean ACF')
            ax.set_ylim((-0.4, 1))
            if self.max_lagtime < number_lags:
                ax.set_xlim((0, self.max_lagtime*self.step_size))
            ax.set_xlabel('tau')
            ax.set_ylabel(r'G/G(0)')
            plt.show()
        
        return mean_acf_data,err_acf_data,lags, decorrelation_time, auto_correlation_matrix

class SimulateRNA():
    
    '''
    This class simulates RNA intensity.
    
    Parameters

    shape_output_array: tuple
        Desired shape of the array, a tuple with two elements where the first element is the number of trajectories and the second element represents the number of time points.
    rna_intensity_method : str, optional.
        Method to generate intensity. The options are 'constant' and 'random'. The default is 'constant'.
    min_int : float, optional.
        Value representing the minimal intensity in the output array. the default is zero.
    max_int : float, optional.
        Value representing the maximum intensity in the output array. the default is 10.
    mean_int : float, optional.
        Value representing the mean intensity in the output array. the default is 5.
        
    '''

    def __init__(self, shape_output_array, rna_intensity_method='constant',mean_int=10 ):
        self.shape_output_array = shape_output_array
        self.rna_intensity_method = rna_intensity_method
        self.mean_int = mean_int 
        
    def simulate(self):
        '''
        Method that simulates the RNA intensity

        Returns

        rna_intensities : NumPy. 
            NumPy arrays with format np.int32 and dimensions equal to shape_output_array. 

        '''
        if self.rna_intensity_method =='random':
            rna_intensities = np.random.normal(loc=self.mean_int, scale=2, size=self.shape_output_array) 
        elif self.rna_intensity_method =='constant':
            rna_intensities = np.full(shape=self.shape_output_array, fill_value= self.mean_int)
        return rna_intensities


class SimulatedCell():
    '''
    This class takes a base video, and it draws simulated spots on top of the image. The intensity for each simulated spot is proportional to the stochastic simulation given by the user.

    Parameters
    
    base_video : NumPy array
        Array of images with dimensions [T, Y, X, C].
    video_for_mask : NumPy array, optional
        Array of images with dimensions [T, Y, X, C]. Use only if the base video has been edited, and an empty video is needed to calculate the mask. The default is None.
    mask_image : NumPy array, optional    
        Numpy Array with dimensions [Y, X]. This image is used as a mask for the simulated video. The mask_image has to represent the same image as the base_video and video_for_mask  
    number_spots : int
        Number of simulated spots in the cell. The default is 10.
    number_frames : int
        The number of frames or time points to simulate. In seconds. The default is 20.
    step_size : float, optional
        Step size for the simulation. In seconds. The default is 1.
    diffusion_coefficient_pixel_per_second : float, optional
        The diffusion coefficient for the particles' Brownian motion. The default is 0.01.
    simulated_trajectories_ch0 : NumPy array, optional
        An array of simulated trajectories with dimensions [S, T] for channel 0. The default is None and indicates that the intensity will be generated, drawing random numbers in a range.
    size_spot_ch0 : int, optional
        Spot size in pixels for channel 0. The default is 5.
    spot_sigma_ch0 : int, optional
        Sigma value for the simulated spots in channel 0, the units are pixels. The default is 2.
    simulated_trajectories_ch1 : NumPy array, optional
        An array of simulated trajectories with dimensions [S, T] for channel 1. The default is None and indicates that the intensity will be generated, drawing random numbers in a range.
    size_spot_ch1 : int, optional
        Spot size in pixels for channel 1. The default is 5.
    spot_sigma_ch1 : int, optional
        Sigma value for the simulated spots in channel 1, the units are pixels. The default is 2.
    simulated_trajectories_ch2 : NumPy array, optional
        An array of simulated trajectories with dimensions [S, T] for channel 2. The default is None and indicates that the intensity will be generated, drawing random numbers in a range.
    size_spot_ch2 : int, optional
        Spot size in pixels for channel 2. The default is 5.
    spot_sigma_ch2 : int, optional
        Sigma value for the simulated spots in channel 2, the units are pixels. The default is 2.
    ignore_ch0 : bool, optional
        A flag that ignores channel 0 returning a NumPy array filled with zeros. The default is 0.
    ignore_ch1 : bool, optional
        A flag that ignores channel 1 returning a NumPy array filled with zeros. The default is 0.
    ignore_ch2 : bool, optional
        A flag that ignores channel 2 returning a NumPy array filled with zeros. The default is 0.
    intensity_calculation_method : str, optional
        Method to calculate intensity the options are : 'total_intensity' , 'disk_donut' and 'gaussian_fit'. The default is 'disk_donut'.
    using_for_multiplexing : bool, optional
        Flag that indicates that multiple genes are simulated per cell.
    perform_video_augmentation : bool, optional
        If true, it performs random rotations the initial video. The default is 1.
    frame_selection_empty_video : str, optional
        Method to select the frames from the empty video, the options are : 'constant' , 'shuffle', 'loop', and 'linear_interpolation'. The default is 'shuffle'.
    ignore_trajectories_ch0 : bool, optional
        A flag that ignores plotting trajectories in channel 0. The default is False.
    ignore_trajectories_ch1 : bool, optional
        A flag that ignores plotting trajectories in channel 1. The default is False.
    ignore_trajectories_ch2 : bool, optional
        A flag that ignores plotting trajectories in channel 2. The default is False.
    intensity_scale_ch0 : float , optional
        Scaling factor for channel 0 that converts the intensity in the stochastic simulations to the intensity in the image.
    intensity_scale_ch1 : float , optional
        Scaling factor for channel 1 that converts the intensity in the stochastic simulations to the intensity in the image.
    intensity_scale_ch2 : float , optional
        Scaling factor for channel 2 that converts the intensity in the stochastic simulations to the intensity in the image.
    dataframe_format : str, optional
        Format for the dataframe the options are : 'short' , and 'long'. The default is 'short'.
        "long" format generates this dataframe: [image_number, cell_number, particle, frame, ch0_int_mean, ch1_int_mean, ch2_int_mean, ch0_int_std, ch1_int_std, ch2_int_std, x, y, ch0_SNR,ch1_SNR,ch2_SNR].
        "short" format generates this dataframe: [image_number, cell_number, particle, frame, ch0_int_mean, ch1_int_mean, ch2_int_mean, x, y].
    
    '''
    def __init__(self, base_video:np.ndarray,
                video_for_mask:Union[np.ndarray, None] = None,  
                mask_image:Union[np.ndarray, None] = None, 
                number_spots:int = 10, 
                number_frames:int = 20, 
                step_size:float = 1, 
                diffusion_coefficient_pixel_per_second:Union[np.ndarray, float] = 0.01, 
                simulated_trajectories_ch0:Union[np.ndarray, None]  = None, 
                size_spot_ch0:int = 5, 
                spot_sigma_ch0:int = 1, 
                simulated_trajectories_ch1:Union[np.ndarray, None] = None, 
                size_spot_ch1:int = 5, 
                spot_sigma_ch1:int = 1, 
                simulated_trajectories_ch2:Union[np.ndarray, None] = None, 
                size_spot_ch2:int = 5, 
                spot_sigma_ch2:int = 1, 
                ignore_ch0: bool = False, 
                ignore_ch1: bool = False, 
                ignore_ch2: bool = False, 
                intensity_calculation_method :str = 'disk_donut', 
                perform_video_augmentation: bool = 0, 
                frame_selection_empty_video:str = 'shuffle',
                ignore_trajectories_ch0:bool =False, 
                ignore_trajectories_ch1:bool =False, 
                ignore_trajectories_ch2:bool =False,
                intensity_scale_ch0:float = 1,
                intensity_scale_ch1:float = 1,
                intensity_scale_ch2:float = 1,
                dataframe_format:str = 'short',
                photobleaching_parameters:Union[np.ndarray, None] = None,
                photobleaching_model:Union[str, None] = None ):
        if (perform_video_augmentation == True) and (video_for_mask is None):
            preprocessed_base_video,selected_angle = AugmentationVideo(base_video).random_rotation()
            if not(mask_image is None):
                self.mask_image,selected_angle = AugmentationVideo(mask_image,selected_angle).random_rotation()
            else:
                self.mask_image=mask_image
        else:
            preprocessed_base_video = base_video
            self.mask_image = mask_image
        self.intensity_calculation_method = intensity_calculation_method
        self.base_video = preprocessed_base_video
        if not (video_for_mask is None):
            video_for_mask = RemoveExtrema(video_for_mask, min_percentile = 0, max_percentile = 99.8,format_video='TYXC').remove_outliers()
            self.video_for_mask = video_for_mask
        else:
            self.video_for_mask = self.base_video
        self.number_spots = number_spots
        self.number_frames = number_frames
        self.step_size = step_size
        self.diffusion_coefficient_pixel_per_second = diffusion_coefficient_pixel_per_second
        self.image_size = [base_video.shape[1], base_video.shape[2]]
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
        self.ignore_trajectories_ch0 = ignore_trajectories_ch0
        self.ignore_trajectories_ch1 = ignore_trajectories_ch1
        self.ignore_trajectories_ch2 = ignore_trajectories_ch2
        self.intensity_scale_ch0 = intensity_scale_ch0
        self.intensity_scale_ch1 = intensity_scale_ch1
        self.intensity_scale_ch2 = intensity_scale_ch2
        self.n_channels = 3
        self.z_slices = 1
        self.time_vector = np.linspace(0, number_frames*step_size, num = number_frames)
        self.frame_selection_empty_video = frame_selection_empty_video
        self.dataframe_format =dataframe_format
        self.photobleaching_parameters = photobleaching_parameters
        self.photobleaching_model = photobleaching_model
        self.MAX_VALUE_uint16 = int(65535*0.8)
        # The following two constants are weights used to define a range of intensities for the simulated spots.
        self.MAX_STD_INT_IMAGE = 4 # maximum number of standard deviations above the mean that are allowed to draw an spot.
        # This function is intended to detect the mask and then reduce the mask by a given percentage. This reduction ensures that the simulated spots are inclosed inside the cell.
        def mask_reduction(polygon_array, percentage_reduction:float = 0.2):
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
            mask_coordinates =  np.vstack((reduced_y, reduced_x)).T
            mask_coordinates.shape
            return mask_coordinates
        if (self.mask_image is None):
            # section that uses cellpose to calculate the mask
            selected_image = np.max(self.video_for_mask[:, :, :, 1],axis=0) # selecting for the mask the first time point
            selected_masks = Cellpose(selected_image, num_iterations = 10, channels = [0, 0], diameter = 200, model_type = 'cyto', selection_method = 'max_area').calculate_masks() # options are 'max_area' or 'max_cells'
            if np.max(selected_masks) == 0:
                print('Error, no masks were found on the image')
                raise
            selected_mask  = CellposeSelection(selected_masks, selected_image, selection_method = 'max_area').select_mask()
        else:
            selected_mask = self.mask_image
        # section that converts the mask in contours and resduces the size of the mask to ensure the spots are generated inside the cell.
        try:
            contours = np.array(find_contours(selected_mask, 0.5), dtype = int)
        except:
            # this section extends the mask to connect isolated areas in the mask
            dilated_image = dilation(selected_mask, square(20))
            dilated_image[0, :] = 0;dilated_image[:, 0] = 0;dilated_image[dilated_image.shape[0]-1, :] = 0;dilated_image[:, dilated_image.shape[1]-1] = 0#This line of code ensures that the corners are zeros.
            contours = np.array(find_contours(dilated_image, 0.5), dtype = int)
        polygon_array = contours[0]
        self.polygon_array = mask_reduction(polygon_array, percentage_reduction = 0.2)
        # Removing mask to the video
        self.video_removed_mask = MaskingImage(base_video, selected_mask).apply_mask()

    def make_simulation (self):
        '''
        This method generates the simulated cell.

        Returns
        
        tensor_video : NumPy array uint16
            Array with dimensions [T, Y, X, C]
        spot_positions_movement : NumPy array
            Array with dimensions [T, Spots, position(y, x)]
        tensor_mean_intensity_in_figure : NumPy array, np.float
            Array with dimensions [T, Spots]
        tensor_std_intensity_in_figure : NumPy array, np.float
            Array with dimensions [T, Spots]
        dataframe_particles : pandas dataframe
            Dataframe with fields [image_number, cell_number, particle, frame, ch0_int_mean, ch1_int_mean, ch2_int_mean, ch0_int_std, ch1_int_std, ch2_int_std, x, y, ch0_SNR,ch1_SNR,ch2_SNR].
        '''
        def make_replacement_pixelated_spots(matrix_background:np.ndarray, center_positions_vector:np.ndarray, size_spot:int, spot_sigma:int, using_ssa:bool, simulated_trajectories_time_point:np.ndarray,intensity_scale:float):
            #This function is intended to replace a kernel gaussian matrix for each spot position. The kernel gaussian matrix is scaled with the values obtained from the SSA o with the values given in a range.
            if size_spot%2 == 0:
                print('The size of the spot must be an odd number')
                raise
            # Copy the matrix_background
            pixelated_image = matrix_background.copy()
            #half_spot_size = int(np.round(size_spot/2))
            spots_range_to_replace = np.linspace(-(size_spot - 1) / 2, (size_spot - 1) / 2, size_spot,dtype=int)
            
            # Section that creates the Gaussian Kernel Matrix
            def pdf_pixel_resolution( ax, spot_sigma=2):
                xx, yy = np.meshgrid(ax, ax)
                kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(spot_sigma))
                return kernel #kernel/np.max(kernel)
            
            def gaussian_subpixel_erf(point, size_spot=5, spot_sigma=2):
                '''
                get a gaussian kernel from a point in a subpixel of the center

                point: iterable of x and y e.g. [x,y]
                size_spot: size of the kernel to generate **MUST BE ODD** to use the center pixel - consider adding check
                spot_sigma: std of the point spread function

                returns: size_spot x size_spot gaussian kernel with subpixel of frac(point) in the center
                '''
                x,y = point # get the point
                x_p = x - int(x) # get the fraction value, subpixel value of center pixel
                y_p = y - int(y) 
                # generate the N+1 x N+1 pixel grid with N/2 x N/2 as the center pixel
                pixelgrid = np.array([np.linspace( -(size_spot+1)/2+1, (size_spot+1)/2 , size_spot+1)])
                xbar = pixelgrid-x_p #subtract the subpixel value
                ybar = pixelgrid-y_p
                # get the gaussian cdf sum of this bin
                Fx = .5*(1+erf(xbar /(spot_sigma*np.sqrt(2))) ) 
                Fy = .5*(1+erf(ybar /(spot_sigma*np.sqrt(2))) )
                dFx = Fx[:,1:] - Fx[:,:-1] #subtract the differences of the cdfs in each direction
                dFy = Fy[:,1:] - Fy[:,:-1]
                K = dFx.T @ dFy # dot together to generate the NxN kernel
                return K*(1/np.sum(K))  #/np.max(K) # (K-np.min(K))/(np.max(K)-np.min(K))
            
            for point_index in range(0, len(center_positions_vector)):
                # creating a position for each spot
                center_position = center_positions_vector[point_index]
                kernel=gaussian_subpixel_erf(point=center_position, size_spot=size_spot, spot_sigma=spot_sigma)
                #kernel=pdf_pixel_resolution( ax=spots_range_to_replace, spot_sigma=spot_sigma)
                if using_ssa == True :
                    int_ssa = simulated_trajectories_time_point[point_index] #- min_SSA_value) / (max_SSA_value-min_SSA_value) # intensity normalized to min and max values in the SSA
                    int_tp = int_ssa* intensity_scale 
                    spot_intensity = np.max((0, int_tp))
                else:
                    spot_intensity = intensity_scale
                kernel_value_intensity = (kernel*spot_intensity).astype(int)
                center_position = np.round(center_position).astype(int)
                #selected_area = pixelated_image[center_position[0]-half_spot_size: center_position[0]+half_spot_size+1 , center_position[1]-half_spot_size: center_position[1]+half_spot_size+1 ]
                selected_area = pixelated_image[center_position[0]+spots_range_to_replace[0]: center_position[0]+spots_range_to_replace[-1]+1 , center_position[1]+spots_range_to_replace[0]: center_position[1]+spots_range_to_replace[-1]+1 ].copy()
                selected_area = selected_area+ kernel_value_intensity
                selected_area[selected_area>self.MAX_VALUE_uint16] = self.MAX_VALUE_uint16  # maxximum range for int uint16 = 65535
                pixelated_image[center_position[0]+spots_range_to_replace[0]: (center_position[0]+spots_range_to_replace[-1])+1 , center_position[1]+spots_range_to_replace[0]: center_position[1]+(spots_range_to_replace[-1])+1 ] = selected_area
                
            return pixelated_image # final_image

        def make_spots_movement(polygon_array, number_spots:int, time_vector:np.ndarray, step_size: float, image_size:np.ndarray, diffusion_coefficient:float, internal_base_video:Union[np.ndarray, None] = None):
            # Function that creates the simulated spots inside a given polygon
            path = mpltPath.Path(polygon_array)
            initial_points_in_polygon = np.zeros((number_spots, 2)) # , dtype = 'int'
            counter_number_spots = 0
            conter_security = 0
            MAX_ITERATIONS = 5000
            # Defining the maximum and  minimum value in the borders to draw spots in the image.
            min_position = 20 # minimal position in pixels
            max_position = image_size[1]-min_position # maximal position in pixels
            while (counter_number_spots < number_spots) and (conter_security < MAX_ITERATIONS):
                test_points = (int(random.uniform(min_position, max_position)), int(random.uniform(min_position, max_position)))
                # testing if the spot is located in an area of high intensity?
                if not ( internal_base_video is None):
                    selected_image = internal_base_video
                    max_allowed_int_image = np.median(selected_image) + self.MAX_STD_INT_IMAGE*np.std(selected_image)
                    min_allowed_int_image = np.median(selected_image) - np.std(selected_image)
                    pixel_size_around_spot = 9
                    half_area_around_spot = int(np.round(pixel_size_around_spot/2))
                    temp_crop_around_spot = selected_image[test_points[0]-half_area_around_spot: test_points[0]+half_area_around_spot+1 , test_points[1]-half_area_around_spot: test_points[1]+half_area_around_spot+1 ]
                    mean_int_tested_spot = np.mean(temp_crop_around_spot)
                    if (mean_int_tested_spot > min_allowed_int_image) and (mean_int_tested_spot < max_allowed_int_image) and (mean_int_tested_spot<self.MAX_VALUE_uint16):
                        int_test = 1
                    else:
                        int_test = 0
                else:
                    int_test = 1
                conter_security += 1
                if path.contains_point(test_points) == True and int_test == True:
                    counter_number_spots += 1
                    initial_points_in_polygon[counter_number_spots-1, :] = np.asarray(test_points)
                if conter_security > MAX_ITERATIONS:
                    print('error generating spots')

            ## Brownian motion
            # scaling factor for Brownian motion.
            # here we check if the user passed a single float as a diffusion coefficent, 
            # if they did, use that value as the diffusion for all spots
            # if they passed an array, then each spot has its own diffusion rate
            # this array could be over time or not, if its only size of number of spots fill it 
            # in over time.
            isfloat_D = False
            if isinstance(diffusion_coefficient, float):
                brownian_movement = math.sqrt(2*diffusion_coefficient*step_size)
                isfloat_D = True
            else:
                brownian_movement = np.sqrt(2*diffusion_coefficient*step_size)
                #brownian_movement = np.random.pareto(1.2)*np.sqrt(2*diffusion_coefficient*step_size)
              
            # Preallocating memory
            y_positions = np.array(initial_points_in_polygon[:, 0], dtype = np.single) #  x_position for selected spots inside the polygon , dtype = 'int'
            x_positions = np.array(initial_points_in_polygon[:, 1], dtype = np.single) #  y_position for selected spots inside the polygon
            temp_Position_y = np.zeros_like(y_positions, dtype = np.single)
            temp_Position_x = np.zeros_like(x_positions, dtype = np.single)
            newPosition_y = np.zeros_like(y_positions, dtype = np.single)
            newPosition_x = np.zeros_like(x_positions, dtype = np.single)
            spot_positions_movement = np.zeros((len(time_vector), number_spots, 2))
            # Main loop that computes the random motion and new spot positions
            
            #check if the motion array is over time or not
            
            #TODO THIS IS UGLY FIX POTENTIALLY IN THE FUTURE
            isnot_overtimeD = False
            if not isfloat_D:
                if brownian_movement.ndim > 1:
                    if brownian_movement.shape[1] == len(time_vector):
                        isnot_overtimeD = False
                    else:
                        brownian_movement = brownian_movement[0]
                        isnot_overtimeD = True
                else:
                    if brownian_movement.shape[0] == number_spots:
                        isnot_overtimeD = True
                    
            
            for t_p, _ in enumerate(time_vector):
                for i_p in range (0, number_spots):
                    if t_p == 0:
                        temp_Position_y[i_p] = y_positions[i_p]
                        temp_Position_x[i_p] = x_positions[i_p]
                    else:
                        if isfloat_D:
                            temp_Position_y[i_p] = newPosition_y[i_p] + brownian_movement * np.random.randn(1)
                            temp_Position_x[i_p] = newPosition_x[i_p] + brownian_movement * np.random.randn(1)
                        else:
                            if not isnot_overtimeD:
                                temp_Position_y[i_p] = newPosition_y[i_p] + brownian_movement[i_p] * np.random.randn(1)
                                temp_Position_x[i_p] = newPosition_x[i_p] + brownian_movement[i_p] * np.random.randn(1)
                                
                            if isnot_overtimeD:
                                temp_Position_y[i_p] = newPosition_y[i_p] + brownian_movement[i_p,t_p] * np.random.randn(1)
                                temp_Position_x[i_p] = newPosition_x[i_p] + brownian_movement[i_p,t_p] * np.random.randn(1)                          
                        
                    while path.contains_point((temp_Position_y[i_p], temp_Position_x[i_p])) == 0:
                        temp_Position_y[i_p] = newPosition_y[i_p]
                        temp_Position_x[i_p] = newPosition_x[i_p]
                    newPosition_y[i_p] = temp_Position_y[i_p]
                    newPosition_x[i_p] = temp_Position_x[i_p]
                spot_positions_movement [t_p, :, :] = np.vstack((newPosition_y, newPosition_x)).T
                #print(spot_positions_movement)
            return spot_positions_movement # vector with dimensions (time, spot, y, x )
            
        def make_simulation(base_video_selected_channel:np.ndarray, masked_video_selected_channel:np.ndarray, spot_positions_movement:np.ndarray, time_vector:np.ndarray, polygon_array, image_size:np.ndarray, size_spot:int, spot_sigma:int, simulated_trajectories, frame_selection_empty_video,ignore_trajectories,intensity_scale):
            
            def generate_gaussian_video(original_video, num_requested_frames, quantile=.95, scale=1.0):
                # Take a given video and approximate its per pixel Gaussian distribution
                # in this case just take the means and std over all pixels for generating the new frame
                #frames_in_orginal_video = original_video.shape[0]
                x_dim = original_video.shape[2]
                y_dim = original_video.shape[1]
                video_means = np.mean(original_video,axis=0) #per_pixel_mean per time
                video_std = np.std(original_video,axis=0) #per_pixel_std per time
                video_std[video_std > np.quantile(video_std, .95)] = np.quantile(video_std, quantile)
                generated_video_gaussian = np.zeros((num_requested_frames,y_dim,x_dim), dtype=np.uint16)
                for j in range(x_dim):
                    for k in range(y_dim):
                        generated_video_gaussian[:,j,k] = np.random.randn(num_requested_frames)*video_std[j,k]*scale + video_means[j,k]
                return generated_video_gaussian

            def generate_gaussian_tau_video(original_video, num_requested_frames, quantile=.95, scale=1.0, shot_noise_percent=.4, tau=10):
                # shot noise percent, alpha, is the percent that each frame is only shot noise
                # tau is how long each pixel is correlated with itself in frames
                
                # Take a given video and approximate its per pixel Gaussian distribution for the shot noise
                # in this case just take the means and std over all pixels for generating the new frame
                #frames_in_orginal_video = original_video.shape[0]
                x_dim = original_video.shape[2]
                y_dim = original_video.shape[1]
                video_means = np.mean(original_video,axis=0) #per_pixel_mean per time
                video_std = np.std(original_video,axis=0) #per_pixel_std per time
                video_std[video_std > np.quantile(video_std, .95)] = np.quantile(video_std, quantile)
                generated_video = np.zeros((num_requested_frames,y_dim,x_dim), dtype=np.uint16)
                
                log_window = np.logspace(0,1,tau)/np.sum(np.logspace(0,1,tau))
                for j in range(x_dim):
                    for k in range(y_dim):
                            
                        N0 = np.random.randn(num_requested_frames)*video_std[j,k]*scale + video_means[j,k]
                        N1 = np.random.randn(num_requested_frames+tau)*video_std[j,k]*scale + video_means[j,k]
                        z = num_requested_frames+tau
                        N1_sliding_average = np.sum((N1[np.arange(num_requested_frames)[None, :] + np.arange(num_requested_frames)[:, None]] )*log_window,axis=1)
                        
                        generated_video[:,j,k] = np.uint16(N0*shot_noise_percent + (1-shot_noise_percent)*N1_sliding_average)
                
                
                
                return generated_video


            def generate_poisson_video(original_video, num_requested_frames):
                # Take a given video and approximate its per pixel poission distribution
                # in this case just take the means of each pixel over all time frames as the lambda for poission dist
                #frames_in_orginal_video = original_video.shape[0]
                x_dim = original_video.shape[2]
                y_dim = original_video.shape[1]
                video_means = np.mean(original_video,axis=0) #per_pixel_mean per time
                generated_video = np.zeros((num_requested_frames,y_dim,x_dim), dtype=np.uint16)
                for j in range(x_dim):
                    for k in range(y_dim):
                        generated_video[:,j,k] = np.random.poisson(lam= video_means[j,k], size=(num_requested_frames,))
                return generated_video
            def function_interpolate_video(orignal_video,num_requested_frames):
                frames_in_orginal_video = orignal_video.shape[0]
                x_dim = orignal_video.shape[2]
                y_dim = orignal_video.shape[1]
                # prealocating array for interpolated video
                interpolated_video = np.zeros((num_requested_frames,y_dim,x_dim ))
                # Populating array for end elements in the array
                interpolated_video [0]=orignal_video[0]
                interpolated_video [num_requested_frames-1]=orignal_video[frames_in_orginal_video-1]
                interpolated_indexes = np.linspace(1, frames_in_orginal_video-1, num_requested_frames-2).astype('int')
                proportion_to_interpolate=0
                for counter, interpolated_index_value in enumerate(interpolated_indexes):
                    num_rep_interpolated_index_value = np.count_nonzero(interpolated_indexes == interpolated_index_value)
                    proportion_to_interpolate =  np.round( proportion_to_interpolate + (1/num_rep_interpolated_index_value) , 3)
                    if num_rep_interpolated_index_value >1:    
                        reverse_proportion = np.max( (0, 1-proportion_to_interpolate))
                        interpolated_video [counter+1]= proportion_to_interpolate*(orignal_video[interpolated_index_value+1]) + reverse_proportion*orignal_video[1-interpolated_index_value]
                        proportion_to_interpolate+proportion_to_interpolate
                    else:
                        interpolated_video [counter+1]=orignal_video[interpolated_index_value]
                    if proportion_to_interpolate >= 1:
                        proportion_to_interpolate=0
                return interpolated_video
            # Main function that makes the simulated cell by calling multiple function.
            temp_image = masked_video_selected_channel[0, :, :]
            temp_image_nonzeros = temp_image.copy()
            temp_image_nonzeros.flatten
            base_video_selected_channel_copy = base_video_selected_channel.copy()
            # Calculate initial background intensity for each spot.
            tensor_image = np.zeros((len(time_vector), image_size[0], image_size[1]), dtype = np.uint32)
            len_empty_video = base_video_selected_channel.shape[0]
            empty_video_index = np.arange(len_empty_video, dtype = np.int32)
            # array that stores the frames to be selected from the empty vector
            if frame_selection_empty_video ==  'linear_interpolation': # selects the first time point
                interpolated_video = function_interpolate_video(orignal_video=base_video_selected_channel.copy(), num_requested_frames=len(time_vector))
                index_frame_selection = range(0, len(time_vector))
                base_video_selected_channel_copy = interpolated_video
            elif frame_selection_empty_video ==  'poisson': # selects the first time point
                generated_video = generate_poisson_video(original_video=base_video_selected_channel.copy(), num_requested_frames=len(time_vector))
                index_frame_selection = range(0, len(time_vector))
                base_video_selected_channel_copy = generated_video
            elif frame_selection_empty_video ==  'gaussian': # selects the first time point
                generated_video = generate_gaussian_video(original_video=base_video_selected_channel.copy(), num_requested_frames=len(time_vector))
                index_frame_selection = range(0, len(time_vector))
                base_video_selected_channel_copy = generated_video

            elif frame_selection_empty_video ==  'gaussian_w_tau_correlation': # selects the first time point
                generated_video = generate_gaussian_tau_video(original_video=base_video_selected_channel.copy(), num_requested_frames=len(time_vector))
                index_frame_selection = range(0, len(time_vector))
                base_video_selected_channel_copy = generated_video
                                
             
            elif frame_selection_empty_video ==  'constant': # selects the first time point
                index_frame_selection = np.zeros((len(time_vector)), dtype = np.int32)
            elif frame_selection_empty_video ==  'loop':
                index_frame_selection = np.resize(empty_video_index, len(time_vector))
            elif frame_selection_empty_video ==  'shuffle':
                index_frame_selection = np.random.randint(0, high = len_empty_video, size = len(time_vector), dtype = np.int32)
            else:
                # If not method to generate video is passed, the gaussian is default.
                generated_video = generate_gaussian_video(original_video=base_video_selected_channel.copy(), num_requested_frames=len(time_vector))
                index_frame_selection = range(0, len(time_vector))
                base_video_selected_channel_copy = generated_video
            for t_p, _ in enumerate(time_vector):
                matrix_background = base_video_selected_channel_copy[index_frame_selection[t_p], :, :]
                if not ( simulated_trajectories is None):
                    using_ssa = 1
                    simulated_trajectories_tp = simulated_trajectories[:, t_p]
                else:
                    using_ssa = 0
                    simulated_trajectories_tp = 0
                # Making the pixelated spots
                if ignore_trajectories ==1:
                    tensor_image[t_p, :, :] = matrix_background
                else:
                    tensor_image[t_p, :, :] = make_replacement_pixelated_spots(matrix_background, spot_positions_movement[t_p, :, :], size_spot, spot_sigma, using_ssa, simulated_trajectories_tp,intensity_scale)
            return tensor_image
        # Create the spots for all channels. Return array with 3-dimensions: T, Sop, XY-Coord
        spot_positions_movement = make_spots_movement(self.polygon_array, self.number_spots, self.time_vector, self.step_size, self.image_size, self.diffusion_coefficient_pixel_per_second, self.video_removed_mask[0, :, :, 1])
        # This section of the code runs the for each channel
        if self.ignore_ch0 == 0:
            tensor_image_ch0 = make_simulation(self.base_video[:, :, :, 0], self.video_removed_mask[:, :, :, 0], spot_positions_movement, self.time_vector, self.polygon_array, self.image_size, self.size_spot_ch0, self.spot_sigma_ch0, self.simulated_trajectories_ch0, self.frame_selection_empty_video,self.ignore_trajectories_ch0,self.intensity_scale_ch0 )
        else:
            tensor_image_ch0 = np.zeros((self.number_frames, self.image_size[0], self.image_size[1]), dtype = np.uint16)
        # Channel 1
        if self.ignore_ch1 == 0:
            tensor_image_ch1 = make_simulation(self.base_video[:, :, :, 1], self.video_removed_mask[:, :, :, 1], spot_positions_movement, self.time_vector, self.polygon_array, self.image_size, self.size_spot_ch1, self.spot_sigma_ch1, self.simulated_trajectories_ch1, self.frame_selection_empty_video,self.ignore_trajectories_ch1,self.intensity_scale_ch1)
        else:
            tensor_image_ch1 = np.zeros((self.number_frames, self.image_size[0], self.image_size[1]), dtype = np.uint16)
        # Channel 2
        if self.ignore_ch2 == 0:
            if self.base_video.shape[3] <3 :   
                tensor_image_ch2 = make_simulation(self.base_video[:, :, :, 1], self.video_removed_mask[:, :, :, 1], spot_positions_movement, self.time_vector, self.polygon_array, self.image_size, self.size_spot_ch2, self.spot_sigma_ch2, self.simulated_trajectories_ch2, self.frame_selection_empty_video,self.ignore_trajectories_ch2,self.intensity_scale_ch2)
            else:
                tensor_image_ch2 = make_simulation(self.base_video[:, :, :, 2], self.video_removed_mask[:, :, :, 2], spot_positions_movement, self.time_vector, self.polygon_array, self.image_size, self.size_spot_ch2, self.spot_sigma_ch2, self.simulated_trajectories_ch2, self.frame_selection_empty_video,self.ignore_trajectories_ch2,self.intensity_scale_ch2)
        else:
            tensor_image_ch2 = np.zeros((self.number_frames, self.image_size[0], self.image_size[1]), dtype = np.uint16)
        
        # Creating a tensor with the final video as a tensor with 4D the order TXYC
        tensor_video =  np.zeros((len(self.time_vector), self.image_size[0], self.image_size[1], self.n_channels), dtype = np.uint16)
        tensor_video [:, :, :, 0] =  tensor_image_ch0
        tensor_video [:, :, :, 1] =  tensor_image_ch1
        tensor_video [:, :, :, 2] =  tensor_image_ch2
        
        # APPLICATION OF PHOTOBLEACHING POST VIDEO GENERATION
        if isinstance(self.photobleaching_model,str): #if theres a photobleaching model selected
            for channel in range(self.n_channels):
                mu = self.photobleaching_parameters[channel,0]
                sigma = self.photobleaching_parameters[channel,1]
                print(mu)
                print(sigma)
                if sigma != 0:
                    photobleaching_array = PhotobleachScaler(model=self.photobleaching_model, shape=[self.image_size[0], self.image_size[1]], t=self.time_vector, mu=mu, sigma=sigma,).generate_bleaching_array()
                    print(photobleaching_array.shape)
                    print(np.swapaxes(photobleaching_array,-1,0).shape)
                    tensor_video [:, :, :, channel] =  tensor_video [:, :, :, channel]*photobleaching_array #*np.swapaxes(photobleaching_array,-1,0) 
                else:
                    photobleaching_array = PhotobleachScaler(model=self.photobleaching_model, shape=[], t=self.time_vector, mu=mu, sigma=0,).generate_bleaching_array()
                    tensor_video [:, :, :, channel] = np.swapaxes(np.swapaxes(tensor_video [:, :, :, channel] ,0,-1)*photobleaching_array,-1,0) #apply to video

                

        # Creating dataframes.
        # converting spot position to int
        spot_positions_movement_int = np.round(spot_positions_movement).astype('int')
        dataframe_particles, _, _, _, _, _, _ = Intensity(tensor_video, particle_size = self.size_spot_ch0, spot_positions_movement = spot_positions_movement_int, method = self.intensity_calculation_method, step_size = self.step_size, show_plot = 0,dataframe_format =self.dataframe_format ).calculate_intensity()
        # Adding SSA Channels
        #number_elements = np.prod(self.simulated_trajectories_ch0.shape)
        if not (self.simulated_trajectories_ch0 is None):
            ssa_ch0 = self.simulated_trajectories_ch0.flatten(order='C')
        else:
            ssa_ch0=np.zeros(shape=(self.number_spots*len(self.time_vector)))
            
        if not (self.simulated_trajectories_ch1 is None):
            ssa_ch1 = self.simulated_trajectories_ch1.flatten(order='C')
        else:
            ssa_ch1=np.zeros(shape=(self.number_spots*len(self.time_vector)))
        
        if not (self.simulated_trajectories_ch2 is None):
            ssa_ch2 = self.simulated_trajectories_ch2.flatten(order='C')
        else:
            ssa_ch2=np.zeros(shape=(self.number_spots*len(self.time_vector)))
        ssa_complete_trajectories = np.stack((ssa_ch0,ssa_ch1,ssa_ch2),axis=-1)
        ssa_columns = ['ch0_SSA_UMP','ch1_SSA_UMP','ch2_SSA_UMP']
        dataframe_particles[ssa_columns] = ssa_complete_trajectories        
        return tensor_video , spot_positions_movement_int, dataframe_particles


class SimulatedCellDispatcher():
    '''
    This class takes a base video and simulates a multiplexing experiment, and it draws simulated spots on top of the image. The intensity for each simulated spot is proportional to the stochastic simulation given by the user.

    Parameters

    initial_video :  NumPy array
        Array of images with dimensions [T, Y, X, C].
    list_gene_sequences : List of strings
        List where every element is a gene sequence file.
    list_number_spots : List of int
        List where every element represents the number of spots to simulate for each gene.
    list_target_channels_proteins : List of int with a range from 0 to 2
        List where every element represents the specific channel where the spots for the nascent proteins will be located.
    list_target_channels_mRNA : List of int with a range from 0 to 2
        List where every element represents the specific channel where the spots for the mRNA signals will be located.
    list_diffusion_coefficients : List of floats
        List where every element represents the diffusion coefficient for every gene.
    list_label_names : List of str
        List where every element contains the label for each gene.
    list_elongation_rates : List of floats
        List where every element represents the elongation rate for every gene.
    list_initiation_rates : List of floats
        List where every element represents the initiation rate for every gene.
    simulation_time_in_sec : int
        The simulation time in seconds. The default is 20.
    step_size_in_sec : float, optional
        Step size for the simulation. In seconds. The default is 1.
    mask_image : NumPy array, optional    
        Numpy Array with dimensions [Y, X]. This image is used as a mask for the simulated video. The mask_image has to represent the same image as the base_video and video_for_mask.
    image_number : int, optional
        Cell number used as an index for the data frame. The default is 0.
    perform_video_augmentation : bool, optional
        If true, it performs random rotations the initial video. The default is True.
    frame_selection_empty_video : str, optional
        Method to select the frames from the empty video, the options are : 'constant' , 'shuffle' and 'loop'. The default is 'shuffle'.
    spot_size : int, optional
        Spot size in pixels. The default is 5.
    spot_sigma : int, optional.
        Sigma value used to generate a gaussian Point Spread Fucntion. The default is 1.
    intensity_scale_ch0 : float , optional
        Scaling factor for channel 0 that converts the intensity in the stochastic simulations to the intensity in the image.
    intensity_scale_ch1 : float , optional
        Scaling factor for channel 1 that converts the intensity in the stochastic simulations to the intensity in the image.
    intensity_scale_ch2 : float , optional
        Scaling factor for channel 2 that converts the intensity in the stochastic simulations to the intensity in the image.
    simulated_RNA_intensities_method : str, optinal
        Method used to simulate RNA intensities in the image. The optiions are 'constant' or 'random'. The default is 'constant'
    dataframe_format : str, optional
        Format for the dataframe the options are : 'short' , and 'long'. The default is 'short'.
        "long" format generates this dataframe: [image_number, cell_number, particle, frame, ch0_int_mean, ch1_int_mean, ch2_int_mean, ch0_int_std, ch1_int_std, ch2_int_std, x, y, ch0_SNR,ch1_SNR,ch2_SNR].
        "short" format generates this dataframe: [image_number, cell_number, particle, frame, ch0_int_mean, ch1_int_mean, ch2_int_mean, x, y].
    ignore_ch0 : bool, optional
        A flag that ignores channel 0 returning a NumPy array filled with zeros. The default is 0.
    ignore_ch1 : bool, optional
        A flag that ignores channel 1 returning a NumPy array filled with zeros. The default is 0.
    ignore_ch2 : bool, optional
        A flag that ignores channel 2 returning a NumPy array filled with zeros. The default is 0.
    scale_intensity_in_base_video : bool, optional
        Flag to scale intensity to a maximum value of 10000. This arbritary value is selected based on the maximum intensities obtained from the original images. The default is False.
    basal_intensity_in_background_video : int, optional
        if the base video is rescaled, this value indicates the maximum value to rescale the original video. The default is 20000    
    use_Harringtonin: bool, optional
        Flag to specify if Harringtonin is used in the experiment. The default is False.
    use_FRAP: bool
        Flag to specify if FRAP is used in the experiment. The default is False.
    perturbation_time_start: int, optional.
        Time to start the inhibition. The default is 0.
    perturbation_time_stop : int, opt.
        Time to start the inhibition. The default is None.
    photobleaching_parameters : ndarray, opt.
        Numpy array of photobleaching parameters Nchannels x Npars (min 2 parameters, mu and sigma). The default is None.
    perturbation_time_stop : str, opt.
        Type of photobleaching model to use, loss or exponential. The default is None.
    '''
    def __init__(self, 
                initial_video:np.ndarray, 
                list_gene_sequences:list, 
                list_number_spots:list, 
                list_target_channels_proteins:list, 
                list_target_channels_mRNA:list,
                list_diffusion_coefficients:list, 
                list_label_names:list, 
                list_elongation_rates:list, 
                list_initiation_rates:list, 
                simulation_time_in_sec:float, 
                step_size_in_sec:float, 
                mask_image:Union[np.ndarray, None] = None, 
                image_number:int = 0, 
                perform_video_augmentation:bool = True, 
                frame_selection_empty_video:str = 'shuffle', 
                spot_size:int = 5 ,
                intensity_scale_ch0 = 1,
                intensity_scale_ch1 = 1,
                intensity_scale_ch2 = 1,
                dataframe_format='short',
                simulated_RNA_intensities_method='constant',
                spot_sigma=1,ignore_ch0: bool = False, 
                ignore_ch1: bool = False, 
                ignore_ch2: bool = False, 
                scale_intensity_in_base_video: bool = False, 
                basal_intensity_in_background_video : int = 20000,
                microns_per_pixel : float = 1.,
                use_Harringtonin= False,
                perturbation_time_start=0,
                perturbation_time_stop=None,
                use_FRAP=False,
                photobleaching_parameters=None,
                photobleaching_model=None):
        if perform_video_augmentation == True:
            preprocessed_base_video,selected_angle = AugmentationVideo(initial_video).random_rotation()
            if not(mask_image is None):
                self.mask_image,selected_angle = AugmentationVideo(mask_image,selected_angle).random_rotation()
            else:
                self.mask_image = mask_image    
        else:
            preprocessed_base_video  = initial_video
            self.mask_image = mask_image
        preprocessed_base_video = RemoveExtrema(preprocessed_base_video, min_percentile = 0, max_percentile = 99,format_video='TYXC').remove_outliers()
        if scale_intensity_in_base_video == True:
            preprocessed_base_video = ScaleIntensity(video=preprocessed_base_video, scale_maximum_value=basal_intensity_in_background_video).apply_scale()
        # Calculating statistics from initial video
        mean_int_in_video = [ np.median(preprocessed_base_video[1,:,:,i]) for i in range(preprocessed_base_video.shape[3]) ]
        if len(mean_int_in_video)<3:
            mean_int_in_video.append(mean_int_in_video[1])
        self.mean_int_in_video = np.array(mean_int_in_video)        
        self.initial_video = preprocessed_base_video
        self.list_gene_sequences = list_gene_sequences
        self.list_number_spots = list_number_spots
        self.list_target_channels_proteins = list_target_channels_proteins
        self.list_target_channels_mRNA = list_target_channels_mRNA
        self.list_diffusion_coefficients = list_diffusion_coefficients
        self.list_label_names = list_label_names
        self.list_elongation_rates = list_elongation_rates
        self.list_initiation_rates = list_initiation_rates
        self.simulation_time_in_sec = simulation_time_in_sec
        self.step_size_in_sec = step_size_in_sec
        self.number_genes = len(list_gene_sequences)
        self.image_number = image_number
        self.frame_selection_empty_video = frame_selection_empty_video
        self.spot_size = spot_size
        self.intensity_scale_ch0 = intensity_scale_ch0
        self.intensity_scale_ch1 = intensity_scale_ch1
        self.intensity_scale_ch2 = intensity_scale_ch2
        self.dataframe_format =dataframe_format
        self.simulated_RNA_intensities_method = simulated_RNA_intensities_method
        self.spot_sigma = spot_sigma
        if max(list_target_channels_proteins)>2:
            raise ValueError('The target channel in the list should be a int between 0 and 2.')
        if max(list_target_channels_mRNA)>2:
            raise ValueError('The target channel in the list should be a int between 0 and 2.')    
        self.ignore_ch0 = ignore_ch0
        self.ignore_ch1 = ignore_ch1
        self.ignore_ch2 = ignore_ch2
        self.microns_per_pixel = microns_per_pixel
        self.use_Harringtonin = use_Harringtonin
        self.use_FRAP=use_FRAP
        self.perturbation_time_start = perturbation_time_start
        self.perturbation_time_stop = perturbation_time_stop
        self.photobleaching_parameters = photobleaching_parameters
        self.photobleaching_model = photobleaching_model
        
    def make_simulation (self):
        '''
        Method that runs the simulations for the multiplexing experiment.

        Returns

        tensor_video : NumPy array uint16
            Array with dimensions [T, Y, X, C]
        dataframe_particles : pandas dataframe
            Dataframe with fields [cell_number, particle, frame, ch0_int_mean, ch1_int_mean, ch2_int_mean, ch0_int_std, ch1_int_std, ch2_int_std, x, y, ch0_SNR,ch1_SNR,ch2_SNR].
        list_ssa : List of NumPy arrays
            List of numpy arrays with the stochastic simulations for each gene. The format is [S, T], where the dimensions are S = spots and T = time.
        '''
        # Wrapper for the simulated cell
        def wrapper_simulated_cell (base_video, video_for_mask = None, ssa_protein = None, rna_intensity=None, target_channel_protein = 1,target_channel_mRNA =0,  diffusion_coefficient = 0.05, step_size = self.step_size_in_sec, spot_size = self.spot_size, intensity_calculation_method = 'disk_donut', frame_selection_empty_video = self.frame_selection_empty_video,int_scale_to_snr=np.array([100,100,100]),microns_per_pixel=1 ):
            if target_channel_protein == 0 and target_channel_mRNA==1:
                ignore_trajectories_ch0 = 0; ignore_trajectories_ch1 = 0; ignore_trajectories_ch2 = 1
                simulated_trajectories_ch0 = ssa_protein 
                simulated_trajectories_ch1 = rna_intensity
                simulated_trajectories_ch2 = None
            elif target_channel_protein == 0 and target_channel_mRNA==2:
                ignore_trajectories_ch0 = 0; ignore_trajectories_ch1 = 1; ignore_trajectories_ch2 = 0
                simulated_trajectories_ch0 = ssa_protein
                simulated_trajectories_ch1 = None
                simulated_trajectories_ch2 = rna_intensity
            elif target_channel_protein == 1 and target_channel_mRNA==0:
                ignore_trajectories_ch0 = 0; ignore_trajectories_ch1 = 0; ignore_trajectories_ch2 = 1
                simulated_trajectories_ch0 = rna_intensity
                simulated_trajectories_ch1 = ssa_protein
                simulated_trajectories_ch2 = None
            elif target_channel_protein == 1 and target_channel_mRNA==2:
                ignore_trajectories_ch0 = 1; ignore_trajectories_ch1 = 0; ignore_trajectories_ch2 = 0
                simulated_trajectories_ch0 = None
                simulated_trajectories_ch1 = ssa_protein
                simulated_trajectories_ch2 = rna_intensity
            elif target_channel_protein == 2 and target_channel_mRNA==0:
                ignore_trajectories_ch0 = 0; ignore_trajectories_ch1 = 1; ignore_trajectories_ch2 = 0
                simulated_trajectories_ch0 = rna_intensity
                simulated_trajectories_ch1 = None
                simulated_trajectories_ch2 = ssa_protein
            elif target_channel_protein == 2 and target_channel_mRNA==1:            
                ignore_trajectories_ch0 = 1; ignore_trajectories_ch1 = 0; ignore_trajectories_ch2 = 0
                simulated_trajectories_ch0 = None
                simulated_trajectories_ch1 = rna_intensity
                simulated_trajectories_ch2 = ssa_protein
            number_spots_per_cell = ssa_protein.shape[0]
            # converting the diffusion coefficient from micrones_per_second to pixeles_per_second
            diffusion_coefficient_pixel_per_second =  diffusion_coefficient / (microns_per_pixel**2)
            # Running simulated cell
            tensor_video, _,DataFrame_particles_intensities = SimulatedCell( base_video = base_video, 
                                                                            video_for_mask = video_for_mask, 
                                                                            mask_image=self.mask_image, 
                                                                            number_spots = number_spots_per_cell, 
                                                                            number_frames = ssa_protein.shape[1], 
                                                                            step_size = step_size, 
                                                                            diffusion_coefficient_pixel_per_second = diffusion_coefficient_pixel_per_second, 
                                                                            simulated_trajectories_ch0 = simulated_trajectories_ch0, 
                                                                            size_spot_ch0 = spot_size, 
                                                                            spot_sigma_ch0 = self.spot_sigma, 
                                                                            simulated_trajectories_ch1 = simulated_trajectories_ch1, 
                                                                            size_spot_ch1 = spot_size, 
                                                                            spot_sigma_ch1 = self.spot_sigma, 
                                                                            simulated_trajectories_ch2 = simulated_trajectories_ch2, 
                                                                            size_spot_ch2 = spot_size, 
                                                                            spot_sigma_ch2 = self.spot_sigma, 
                                                                            intensity_calculation_method = intensity_calculation_method, 
                                                                            frame_selection_empty_video = frame_selection_empty_video, 
                                                                            ignore_trajectories_ch0 = ignore_trajectories_ch0, 
                                                                            ignore_trajectories_ch1 = ignore_trajectories_ch1,
                                                                            ignore_trajectories_ch2 = ignore_trajectories_ch2,
                                                                            intensity_scale_ch0 = int_scale_to_snr[0],
                                                                            intensity_scale_ch1 = int_scale_to_snr[1],
                                                                            intensity_scale_ch2 = int_scale_to_snr[2],
                                                                            dataframe_format =self.dataframe_format,
                                                                            ignore_ch0 = self.ignore_ch0,
                                                                            ignore_ch1 = self.ignore_ch1,
                                                                            ignore_ch2 = self.ignore_ch2,
                                                                            photobleaching_parameters = self.photobleaching_parameters,
                                                                            photobleaching_model = self.photobleaching_model).make_simulation()
            DataFrame_particles_intensities['cell_number'] = DataFrame_particles_intensities['cell_number'].replace([0], self.image_number)
            DataFrame_particles_intensities['image_number'] = DataFrame_particles_intensities['image_number'].replace([0], self.image_number)

            return tensor_video, DataFrame_particles_intensities  # [cell_number, particle, frame, ch0_int_mean, ch1_int_mean, ch2_int_mean, ch0_int_std, ch1_int_std, ch2_int_std, x, y].
        # Runs the SSA and the simulated cell functions
        list_ssa = []
        list_min_ssa = []
        list_max_ssa = []
        RNA_INTENSITY_MAX_VALUE =10 # this variable defines a value of units of RNA
        for i in range(0, self.number_genes):
            # # Simulations for intensity
            ssa,ssa_ump,_,_ = SSA_rsnapsim( gene_file = self.list_gene_sequences[i], 
                                        ke = self.list_elongation_rates[i],
                                        ki = self.list_initiation_rates[i],
                                        frames = self.simulation_time_in_sec,
                                        frame_rate = 1,
                                        n_traj = self.list_number_spots[i],
                                        use_Harringtonin = self.use_Harringtonin,
                                        use_FRAP =  self.use_FRAP,
                                        perturbation_time_start = self.perturbation_time_start,
                                        perturbation_time_stop = self.perturbation_time_stop).simulate() 
            simulated_trajectories_RNA= SimulateRNA(shape_output_array=(self.list_number_spots[i], self.simulation_time_in_sec), 
                                                                        rna_intensity_method = self.simulated_RNA_intensities_method,
                                                                        mean_int=RNA_INTENSITY_MAX_VALUE ).simulate()
            # appending simulated data
            list_ssa.append(ssa_ump)
            list_min_ssa.append(ssa_ump.min())
            list_max_ssa.append(ssa_ump.max())
        vector_int_scales  = np.array ([self.intensity_scale_ch0,self.intensity_scale_ch1, self.intensity_scale_ch2])
        # Calculating the estimated elongation rates based on parameter values
        calculated_mean_int_in_ssa = np.zeros(len(self.list_gene_sequences))+0.001
        for g in range(len(self.list_gene_sequences)):
            _, _,tagged_pois,raw_seq = rss.seqmanip.open_seq_file(str(self.list_gene_sequences[g]))
            gene_len = len(raw_seq)
            try:
                gene_obj = tagged_pois['0'][0]
            except:
                gene_obj = tagged_pois['1'][0]
            number_probes = np.max(gene_obj.probe_vec)
            calculated_mean_int_in_ssa[g] = (gene_len * self.list_initiation_rates[g]) / (self.list_elongation_rates[g] * number_probes)
        # Calculated Scaling factors for intensity
        int_scale_to_snr = np.zeros(3)
        for i in range (len(self.list_target_channels_proteins)):
            int_scale_to_snr[self.list_target_channels_proteins[i]] = (vector_int_scales[self.list_target_channels_proteins[i]] * self.mean_int_in_video[self.list_target_channels_proteins[i]]) / calculated_mean_int_in_ssa[i]
        # Intensity scale for RNA Channel
        for i in range (len(self.list_target_channels_mRNA)):
            int_scale_to_snr[self.list_target_channels_mRNA[i]] = (vector_int_scales[self.list_target_channels_mRNA[i]] * self.mean_int_in_video[self.list_target_channels_mRNA[i]]) / RNA_INTENSITY_MAX_VALUE        
        # Creating the videos
        list_DataFrame_particles_intensities = []
        for i in range(0, self.number_genes):
            if i == 0 :
                tensor_video , DataFrame_particles_intensities = wrapper_simulated_cell(self.initial_video, 
                                                                                        video_for_mask = self.initial_video, 
                                                                                        ssa_protein = list_ssa[i], 
                                                                                        rna_intensity = simulated_trajectories_RNA,
                                                                                        target_channel_protein = self.list_target_channels_proteins[i],
                                                                                        target_channel_mRNA =  self.list_target_channels_mRNA[i], 
                                                                                        diffusion_coefficient = self.list_diffusion_coefficients[i], 
                                                                                        frame_selection_empty_video = self.frame_selection_empty_video,
                                                                                        int_scale_to_snr=int_scale_to_snr,
                                                                                        microns_per_pixel=self.microns_per_pixel)
            else:
                tensor_video , DataFrame_particles_intensities = wrapper_simulated_cell(tensor_video, 
                                                                                        video_for_mask = self.initial_video, 
                                                                                        ssa_protein = list_ssa[i],
                                                                                        rna_intensity =  simulated_trajectories_RNA,
                                                                                        target_channel_protein = self.list_target_channels_proteins[i], 
                                                                                        target_channel_mRNA = self.list_target_channels_mRNA[i] , 
                                                                                        diffusion_coefficient = self.list_diffusion_coefficients[i], 
                                                                                        frame_selection_empty_video = 'loop',  # notice that for the multiplexing frame_selection_empty_video has to be 'loop', because the initial video deffines the initial background image.
                                                                                        int_scale_to_snr=int_scale_to_snr,
                                                                                        microns_per_pixel=self.microns_per_pixel)
            list_DataFrame_particles_intensities.append(DataFrame_particles_intensities)
        # Adding a classification column to all dataframes
        for i in range(0, self.number_genes):
            classification = self.list_label_names[i]
            list_DataFrame_particles_intensities[i]['Classification'] = classification
        # Increasing the particle number
        for i in range(0, self.number_genes):
            if i > 0:
                number_spots_for_previous_genes = np.sum(self.list_number_spots[0:i])
                list_DataFrame_particles_intensities[i]['particle'] = list_DataFrame_particles_intensities[i]['particle'] + number_spots_for_previous_genes
        # Merging multiple dataframes in a single one.
        dataframe_simulated_cell = pd.concat(list_DataFrame_particles_intensities)        
        return tensor_video, dataframe_simulated_cell, list_ssa, self.mask_image


class PipelineTracking():
    '''
    A pipeline that allows cell segmentation, spot detection, and tracking of spots.

    Parameters

    video : NumPy array
        Array of images with dimensions [T, Y, X, C].
    mask : NumPy array, optional
        Array of images with dimensions [ Y, X]. The default is None, and it will perform segmentation using Cellpose.
    particle_size : int, optional
        Average particle size. The default is 5.
    file_name : str, optional
        The file name for the simulated cell. The default is 'Cell.tif'.
    selected_channel_tracking : int, optional
        Integer indicating the channel used for particle tracking. The default is 0.
    selected_channel_segmentation: int, optional
        Integer indicating the channel used for segmenting the cytosol. The default is 0.
    intensity_calculation_method : str, optional
        Method to calculate intensity the options are : 'total_intensity' , 'disk_donut' and 'gaussian_fit'. The default is 'disk_donut'..
    mask_selection_method :  str, optional
        Option to use the optimization algorithm to maximize the number of cells or maximize the size. The options are 'max_area' or 'max_cells'. The default is 'max_area'.
    show_plot : bool, optional
        Allows the user to show a plot for the optimization process. The default is 1.
    use_optimization_for_tracking : bool, optional
        Option to run an optimization process to select the best filter for the tracking process. The default is 0.
    real_positions_dataframe : Pandas dataframe.
        A pandas data frame containing the position of each spot in the image. This dataframe is generated with class SimulatedCell, and it contains the true position for each spot. This option is only intended to be used to train algorithms for tracking and visualize real vs detected spots. The default is None.
    average_cell_diameter : float, optional
        Average cell size. The default is 120.
    print_process_times : bool, optional
        Allows the user the times taken during each process. The default is 0.
    intensity_selection_threshold_int_std : float, optional. The default is None, and it uses a default value or an optimization method if use_optimization_for_tracking is set to TRUE. 
        Threshold intensity for tracking
    min_percentage_time_tracking = float, optional
        Value that indicates the minimal (normalized) percentage of time to consider a particle as a detected trajectory during the tracking process. The number must be between 0 and 1. The default is 0.3.
    intensity_threshold_tracking : float or None, optional.
        Intensity threshold value to be used during tracking. If no value is passed, the code attempts to calculate this value. The default is None.
    dataframe_format : str, optional
        Format for the dataframe the options are : 'short' , and 'long'. The default is 'short'.
        "long" format generates this dataframe: [image_number, cell_number, particle, frame, ch0_int_mean, ch1_int_mean, ch2_int_mean, ch0_int_std, ch1_int_std, ch2_int_std, x, y, ch0_SNR,ch1_SNR,ch2_SNR].
        "short" format generates this dataframe: [image_number, cell_number, particle, frame, ch0_int_mean, ch1_int_mean, ch2_int_mean, x, y].
    create_pdf : bool, optional
        Flag to indicate if a pdf report is needed. The default is True.
    path_temporal_results : path or str, optional.
        Path used to store results to create the PDF. The default is None.
    image_index : int, optional
        Index indicating a counter for the image. The default is 0.
    '''
    def __init__(self, video:np.ndarray, mask:np.ndarray = None, particle_size:int = 5, file_name:str = 'Cell.tif', selected_channel_tracking:int = 0,selected_channel_segmentation:int = 0,  intensity_calculation_method:str = 'disk_donut', mask_selection_method:str = 'max_spots', show_plot:bool = 1, use_optimization_for_tracking: bool = 1, real_positions_dataframe = None, average_cell_diameter: float = 120, print_process_times:bool = 0,min_percentage_time_tracking=0.4,intensity_threshold_tracking=None,dataframe_format='short',create_pdf=True,path_temporal_results=None,image_index:int=0):
        self.video = video
        self.mask = mask
        self.particle_size = particle_size
        self.image = np.max(video,axis=0)
        self.num_frames = video.shape[0]
        self.file_name = file_name
        self.intensity_calculation_method = intensity_calculation_method  # options are : 'total_intensity' and 'disk_donut'
        self.mask_selection_method = mask_selection_method # options are : 'max_spots' and 'max_area'
        self.selected_channel_tracking = selected_channel_tracking
        self.selected_channel_segmentation = selected_channel_segmentation
        self.show_plot = show_plot
        self.use_optimization_for_tracking = use_optimization_for_tracking
        self.real_positions_dataframe = real_positions_dataframe
        self.average_cell_diameter = average_cell_diameter
        self.print_process_times = print_process_times
        self.NUM_ITERATIONS_CELLPOSE = 10
        self.NUM_ITERATIONS_TRACKING = 1000
        self.MIN_PERCENTAGE_FRAMES_FOR_TRACKING = min_percentage_time_tracking
        self.intensity_threshold_tracking=intensity_threshold_tracking
        self.dataframe_format=dataframe_format
        self.create_pdf=create_pdf
        self.path_temporal_results =path_temporal_results
        self.image_index=image_index
        if create_pdf == True:
            self.save_image_as_file=True
        else:
            self.save_image_as_file=False
        
    def run(self):
        '''
        Runs the pipeline.

        Returns

        dataframe_particles : pandas dataframe
            Dataframe with fields [image_number, cell_number, particle, frame, ch0_int_mean, ch1_int_mean, ch2_int_mean, ch0_int_std, ch1_int_std, ch2_int_std, x, y].
        selected_mask : Numpy array
            Array with the selected mask. Where zeros represents the background and one represent the area in the cell.
        array_intensities : Numpy array
            Array with dimensions [S, T, C].
        time_vector : Numpy array
            1D array.
        mean_intensities: Numpy array
            Array with dimensions [S, T, C].
        std_intensities : Numpy array
            Array with dimensions [S, T, C].
        mean_intensities_normalized : Numpy array
            Array with dimensions [S, T, C].
        std_intensities_normalized : Numpy array
            Array with dimensions [S, T, C].
        '''
        
        image_name_tracking = self.path_temporal_results.joinpath('tracking_'+str(self.image_index) +'.png')
        image_name_visualization = self.path_temporal_results.joinpath('visualization_'+str(self.image_index) +'.png')
        if self.print_process_times ==True:
            start = timer()
        if self.mask is None:
            selected_masks = Cellpose(self.image, num_iterations = self.NUM_ITERATIONS_CELLPOSE, selection_method = 'max_cells_and_area', diameter = self.average_cell_diameter ).calculate_masks() # options are 'max_area' or 'max_cells'
            if not ( selected_masks is None):
                selected_mask  = CellposeSelection(selected_masks, self.video, selection_method = self.mask_selection_method, particle_size = self.particle_size, selected_channel = self.selected_channel_segmentation).select_mask()
            else:
                selected_mask = None
        else:
            selected_mask = self.mask
        if self.print_process_times ==True:
            end = timer()
        # test if segmentation was succcesful
        MINIMAL_NUMBER_OF_PIXELS_IN_MASK = 10000
        if not(selected_mask is None):
            detected_mask_pixels =np.count_nonzero(selected_mask.flatten())
            if  detected_mask_pixels > MINIMAL_NUMBER_OF_PIXELS_IN_MASK:
                segmentation_succesful = True
                tracking_succesful = True
            else:
                segmentation_succesful = False  
                tracking_succesful = False
        else:
            segmentation_succesful = False
            tracking_succesful = False
        if self.print_process_times == True:
            print('mask time:', round(end - start), ' sec')
                
        if not ( selected_mask is None) and (segmentation_succesful == True):
            number_masks = np.max(selected_mask)
            if self.print_process_times == True:
                start = timer()
            dataframe_particles_all_cells = pd.DataFrame()
            for i in range (1,number_masks+1):
                mask = np.zeros_like(selected_mask)
                mask[selected_mask==i] = 1
                # Tracking
                if self.print_process_times ==True:
                    start = timer()
                if self.num_frames > 20:
                    minimal_frames =  int(self.num_frames*self.MIN_PERCENTAGE_FRAMES_FOR_TRACKING) # minimal number of frames to consider a trajectory
                else:
                    minimal_frames =  int(self.num_frames*0.4) # minimal number of frames to consider a trajectory
                if self.use_optimization_for_tracking == True:
                    use_default_filter = 0
                else:
                    use_default_filter = 1
                Dataframe_trajectories, _, filtered_video = Trackpy(self.video, mask=mask, particle_size = self.particle_size, selected_channel = self.selected_channel_tracking, minimal_frames = minimal_frames, optimization_iterations = self.NUM_ITERATIONS_TRACKING, use_default_filter = use_default_filter, show_plot = self.show_plot,intensity_threshold_tracking=self.intensity_threshold_tracking, image_name=image_name_tracking).perform_tracking()
                # Intensity calculation
                
                if not ( Dataframe_trajectories is None):
                    dataframe_particles, array_intensities, time_vector, mean_intensities, std_intensities, mean_intensities_normalized, std_intensities_normalized = Intensity(self.video, self.particle_size, Dataframe_trajectories, method = self.intensity_calculation_method, show_plot = 0, dataframe_format=self.dataframe_format,cell_counter=i-1,image_index=self.image_index).calculate_intensity()
                    # This flag makes segmentation_succesful flase if less than 4 particles are detected in the dataframe_particles.
                    # This option avoids problem while calculating the next steps.
                    if array_intensities.shape[0]<1:
                        segmentation_succesful =False
                        tracking_succesful =False
                else:
                    dataframe_particles =Intensity(self.video, self.particle_size, Dataframe_trajectories, method = self.intensity_calculation_method, show_plot = 0, dataframe_format=self.dataframe_format,cell_counter=i-1,image_index=self.image_index).calculate_intensity()[0]
                    #dataframe_particles = None
                    array_intensities = None
                    time_vector = None
                    mean_intensities = None
                    std_intensities = None
                    mean_intensities_normalized = None
                    std_intensities_normalized = None
                #dataframe_particles
                dataframe_particles_all_cells = dataframe_particles_all_cells.append(dataframe_particles, ignore_index = True)
                # Visualization
                if (self.show_plot == True) or (self.create_pdf==True):
                    if tracking_succesful == True:
                        VisualizerImage(self.video, filtered_video, Dataframe_trajectories, self.file_name, list_mask_array = mask, selected_channel = self.selected_channel_tracking, selected_time_point = 0, normalize = False, individual_figure_size = 7, list_real_particle_positions = self.real_positions_dataframe,image_name=image_name_visualization,show_plot=self.show_plot,save_image_as_file=self.save_image_as_file).plot()
                    else:
                        VisualizerImage(self.video, filtered_video, None, self.file_name, list_mask_array = mask, selected_channel = self.selected_channel_tracking, selected_time_point = 0, normalize = False, individual_figure_size = 7, list_real_particle_positions = self.real_positions_dataframe,image_name=image_name_visualization,show_plot=self.show_plot,save_image_as_file=self.save_image_as_file).plot()
            # Printing processing time
            if self.print_process_times == True:
                end = timer()
                print('tracking time:', round(end - start), ' sec')
        else:
            dataframe_particles_all_cells = None
            array_intensities = None
            time_vector = None
            mean_intensities = None
            std_intensities = None
            mean_intensities_normalized = None
            std_intensities_normalized = None
        return dataframe_particles_all_cells,selected_mask, array_intensities, time_vector, mean_intensities, std_intensities, mean_intensities_normalized, std_intensities_normalized, segmentation_succesful


class PhotobleachScaler():
    '''
    This class is intended to provide a photobleaching array to any size array
    
    Model_options:
        
        exponential - I0*np.exp(-alpha*t)
        loss - I0*(1-alpha)^(nframes)

    Parameters

    model : str
        type of photobleaching model to use. Options are 'exponential' or 'loss' corresponding to an exponential curve or percentage lost per frame.
            exponential: $I_0*e^(-\alpha*t)$
            loss:  $I_0*(1-\alpha)^(len(t))$
    shape : list
        shape of the corresponding photobleaching array to generate neglecting time axis. Providing no shape provides one photobleaching curve. Providing [512,512] 
        generates a photobleaching curve of size (512,512,len(t)) for a per pixel bleaching rate.
    t : np.ndarray
        Time vector to generate photobleaching over.
    mu : float
        photobleaching average, $\alpha$ for each model
    sigma : float
        photobleaching standard deviation, if left as 0, mu is used for every photobleaching curve.

    Usage

    To generate a per pixel normally distributed N(.001,.0001^2) %loss for a video of shape [100,512,512], T, X, Y:
        bleaching_array = PhotobleachScaler(model='loss', shape=[512,512], t=np.linspace(0,100,100), mu=0.001, sigma=.0001 ).generate_bleaching_array()
        bleaching_array.shape = (100,512,512)

    To generate a single consistent %loss curve to use for an entire video:
        bleaching_array = PhotobleachScaler(model='loss', shape=[], t=np.linspace(0,100,100), mu=0.001, sigma=0).generate_bleaching_array()
        bleaching_array.shape = (100,)

    '''
    def __init__(self, model:str = 'loss', shape:list = [], t:np.ndarray = np.array([]), mu:float = 0.001, sigma:float = 0.0001,  ):
        self.model = model
        self.photobleaching_array = np.zeros(shape)
        self.t = t
        self.shape = shape
        self.mu = mu
        self.sigma = sigma
        self.ntimes = len(t)
    
    def generate_bleaching_array(self):
        # generate the bleaching array for a percentage loss per frame model
        if self.model.lower() == 'loss':
            if self.sigma != 0: #if the user included a sigma, simulate a random normal
                # this is the cumulative product of a normally distributed random % loss (clipped to avoid negatives)
                bleaching_array = np.cumprod(1-(np.random.normal(loc=self.mu,
                                                                scale=self.sigma,
                                                                size=([self.ntimes] + self.shape ))).clip(min=0),axis=0)
            else:
                bleaching_array = (1-self.mu)**np.arange(self.ntimes) #just a single curve is needed if theres no noise
        
        if self.model.lower() in ['exponential','exp']:
            
            if self.sigma != 0: #if the user included a sigma, simulate a random normal
                # use the e ^ -alpha * t  model
                bleaching_array = np.swapaxes(np.swapaxes(np.exp(-np.random.normal(loc=self.mu,
                                                                scale=self.sigma,
                                                                size=(self.shape + [self.ntimes] )).clip(min=0)*self.t),-1,0),-1,1)
                
            else:
                bleaching_array = np.exp(-self.mu*self.t)  #just a single curve is needed if theres no noise
                    
        return bleaching_array
        


class PhotobleachingCalculation():
    '''
    This class is intended to calculate the intensity in the detected spots.

    Parameters

    video : NumPy array
        Array of images with dimensions [T, Y, X, C].
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
    def __init__(self, video:np.ndarray,mask:np.ndarray,step_size:int=1, selected_channel:int =1,show_plot:bool =1):
        self.video = video
        self.show_plot = show_plot
        self.mask = mask
        self.step_size = step_size
        self.selected_channel = selected_channel
    def visualize_photobleaching(self): 
        # https://stackoverflow.com/questions/37713691/python-fitting-exponential-decay-curve-from-recorded-values
        # https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.optimize.curve_fit.html
        # https://stackoverflow.com/questions/52356128/how-to-set-up-the-initial-value-for-curve-fit-to-find-the-best-optimizing-not-j
        time_points, height, width, number_channels = self.video.shape[0],self.video.shape[1], self.video.shape[2], self.video.shape[3]
        # define type of function to search
        def fun_model_exp(t_p, photobleaching_basal_constant, photobleaching_exp_constant):
            return photobleaching_basal_constant * np.exp(-photobleaching_exp_constant*t_p)
        # array with time points
        t_p = np.arange(0,time_points,self.step_size) 
        # saving the mean intensity for each frame
        mean_cell_intensity = np.zeros((number_channels, time_points)) 
        for i in range(0,number_channels):
            for k in range(0,time_points):
                temp_cell_int = self.video[k,:,:,i]*self.mask
                mean_cell_intensity[i,k] = temp_cell_int[np.nonzero(temp_cell_int)].mean()
        # Optimization routine to fit data to an exponential 
        num_initial_guesses = 50
        p0_vector_exp = np.logspace(np.log10(0.0001), np.log10(1), num=num_initial_guesses, endpoint=True, base=10) # vector of initial guesses
        p0_vector_basal = np.linspace(1,1000, num=num_initial_guesses, endpoint=True) # vector of initial guesses
        rmse_list = []
        opt_param_list = np.zeros((len(p0_vector_exp),2))
        for i in range(len(p0_vector_exp)): # the optimization is tested for all initial guesses
            try:
                opt_parameters, _ = curve_fit(fun_model_exp,t_p,mean_cell_intensity[self.selected_channel,:],p0=[p0_vector_basal[i],p0_vector_exp[i]]) # curve_fit is used to fit the model to the function "fun_model_exp"
            except:
                opt_parameters = [0,0] # and exception is generated in case the method cannot fit the data.
            # Model evaluation
            model_data = fun_model_exp(t_p, opt_parameters[0],  opt_parameters[1]) # model data are generated by using "fun_model_exp" with the selected parameter value  opt_parameters[0,0] notice that we only have a single parameter
            # objective function
            rmse_list.append(np.sqrt(np.mean(np.square(model_data - mean_cell_intensity[self.selected_channel,:])))) # rmse is calculated comparing model and data.
            opt_param_list[i,:] = opt_parameters 
        index_selected_parameter  = np.argmin(rmse_list)
        selected_parameters =  opt_param_list[index_selected_parameter,:]   # selecting the parameter set that minimizes the rmse            
        model_data_sel_parameter = fun_model_exp(t_p, selected_parameters[0],selected_parameters[1])
        photobleaching_necessary =0
        if rmse_list[index_selected_parameter]  < 5:
            photobleaching_necessary = 1
        if abs(selected_parameters[1]) < 1e-4:
            photobleaching_necessary = 0        
        if self.show_plot == True:
            if len(t_p)> 1000:
                downsample = 50
            elif len(t_p)> 500 and len(t_p) < 1000:
                downsample = 20
            elif len(t_p)> 200 and len(t_p) < 500:
                downsample = 10
            elif len(t_p)> 50 and len(t_p) < 200:
                downsample = 3
            else:
                downsample = 1
            plt.figure(1)
            if self.selected_channel ==0:
                col_line = 'r'
            elif self.selected_channel ==1:
                col_line = 'g'
            else:
                col_line = 'b'
            if photobleaching_necessary  ==1:
                plt.plot(t_p[::downsample], model_data_sel_parameter[::downsample], color='black', marker='o', markerfacecolor='black',linestyle='None', markersize=10, label='fit: $f(t) = %.1f * e^{-%.3f t}$' % (selected_parameters[0],selected_parameters[1]))
            else:
                plt.plot(t_p[::downsample], model_data_sel_parameter[::downsample], color='black', marker='o', markerfacecolor='black',linestyle='None', markersize=10, label='No meaningful fit was possible.' )
            plt.plot(t_p, mean_cell_intensity[self.selected_channel,:], color=col_line,linewidth = 3, label='Mean intensity channel %d' % (self.selected_channel)  )
            plt.legend(loc='best')
            plt.show()
        return selected_parameters, mean_cell_intensity[self.selected_channel,:], t_p



class PhotobleachingCorrection():
    def __init__(self, video:np.ndarray,mask:np.ndarray,step_size:int=1, selected_channel:int=1,show_plot:bool=1):
        self.video = video
        self.show_plot = show_plot
        self.mask = mask
        self.step_size = step_size
        self.selected_channel = selected_channel
        
    def apply_photobleaching_correction(self): 
        time_points, height, width, number_channels   = self.video.shape[0],self.video.shape[1], self.video.shape[2], self.video.shape[3]
        # define type of function to search
        def fun_model_exp(t_p, photobleaching_basal_constant, photobleaching_exp_constant):
            return photobleaching_basal_constant * np.exp(-photobleaching_exp_constant*t_p)
        # array with time points
        t_p = np.arange(0,time_points,self.step_size) 
        # saving the mean intensity for each frame
        mean_cell_intensity = np.zeros((number_channels, time_points)) 
        for i in range(0,number_channels):
            for k in range(0,time_points):
                temp_cell_int = self.video[k,:,:,i]*self.mask
                mean_cell_intensity[i,k] = temp_cell_int[np.nonzero(temp_cell_int)].mean()
        video_photobleached_corrected = 0
        return video_photobleached_corrected
    
    def correct_photobleaching_video(video, mask_array, normalized=False, guess = None, niter=10000):
        # get the average intensity in the mask for each channel
        average_ints = np.zeros([video.shape[0], video.shape[-1],]   )
        for i in range(video.shape[0]):
            mask =mask_array[i]
            for j in range(video.shape[-1]):
                average_ints[i,j] = np.mean(video[i,:,:,j][mask == 1])
        try:  #generate an inital guess
            guess[0] #### ASK WILL ABOUT THIS CODE
        except:
            guess = (np.mean(average_ints[0])*.5, .3,)
        
        def expo(x,a,c):        
            '''
            exponential function.
            '''
            #c = np.abs(c)
            return a*np.exp(-c*x)
        corrected_video = np.zeros(video.shape) #preallocate final video
        tvec = range(video.shape[0]) #time vector for fitting
        exponentials = np.zeros(average_ints.shape) #array for exponentials
        for i in range(video.shape[-1]): # for each trajectory
            # fit the exponential curve
            opt,cov = curve_fit(expo,tvec,average_ints[:,i],p0=guess,maxfev=niter)
            exponentials[:,i] = expo(tvec,opt[0],opt[1])
            if normalized:
                exponentials[:,i] = expo(tvec,opt[0],opt[1])
            else:
                exponentials[:,i] = expo(tvec,1.0,opt[1])
        #correct each frame for each channel
        for i in range(video.shape[-1]):
            for j in range(video.shape[0]):
                corrected_video[j,:,:,i] = video[j,:,:,i]/exponentials[j,i] 

        return corrected_video, exponentials, opt, cov
    
    
    
    

class ReportPDF():
    '''
    This class intended to create a PDF report including the images generated during the pipeline for segmentation and tracking.
    
    Parameters
    
    directory_results: str or PosixPath
        Directory containing the images to include in the report.
    pdf_report_name  : str
        Name of the pdf repot.
    list_segmentation_succesful : list
        List indicating if the segmentation was sucessful for the image.
    list_file_names : list,
        List with the file names.
    This PDF file is generated, and it contains the processing steps for each image in the folder.
    
    '''    
    def __init__(self, directory, pdf_report_name,list_file_names,list_segmentation_succesful):
        self.directory = directory
        self.pdf_report_name=pdf_report_name
        self.list_segmentation_succesful =list_segmentation_succesful
        self.list_file_names=list_file_names
        self.num_images = len(list_file_names)
    def create_report(self):
        
        '''
        This method creates a PDF with the original images, images for cell segmentation and images for the spot detection.
        '''
        pdf = FPDF()
        WIDTH = 210
        HEIGHT = 297
        pdf.add_page()
        pdf.set_font('Courier', 'B', 12)
        for i,temp_file_name in enumerate(self.list_file_names):
            pdf.cell(w=0, h=10, txt='Original image: ' + temp_file_name,ln =2,align = 'L')
            # code that returns the path of the original image
            temp_original_img_name = self.directory.joinpath( 'visualization_' + str(i) +'.png' )
            pdf.image(str(temp_original_img_name), x=10, y=20, w=WIDTH-30)
            # creating some space
            for text_idx in range(0, 12):
                pdf.cell(w=0, h=10, txt='',ln =1,align = 'L')
            pdf.cell(w=0, h=10, txt='Tracking: ' + temp_file_name,ln =1,align = 'L')
            # code that returns the path of the segmented image
            if self.list_segmentation_succesful[i]==True:
                temp_tracking_img_name = self.directory.joinpath( 'tracking_' + str(i) +'.png' )
                pdf.image(str(temp_tracking_img_name), x=10, y=HEIGHT/2, w=WIDTH-150)
                if i< self.num_images-1:
                    pdf.add_page()
            else:
                pdf.cell(w=0, h=20, txt='Segmentation was not possible for image: ' + temp_file_name,ln =1,align = 'L')
                if i< self.num_images-1:
                    pdf.add_page()
        pdf.output(self.pdf_report_name, 'F')
        return None



class MetadataSimulatedCell():
    '''
    This class is intended to generate a metadata file containing used dependencies, user information, and parameters used to generate the simulated cell.
    
    Parameters
    
    The parameters for this class are defined in the SimultedCell class
    '''
    def __init__(self, 
                metadata_filename,
                video_dir, 
                masks_dir, 
                list_gene_sequences,
                list_number_spots,
                list_target_channels_proteins,
                list_target_channels_mRNA, 
                list_diffusion_coefficients,
                list_label_names,
                list_elongation_rates,
                list_initiation_rates,
                number_cells = 1,
                simulation_time_in_sec = 100,
                step_size_in_sec = 1,
                frame_selection_empty_video='gaussian',
                spot_size = 7 ,
                spot_sigma=1,
                intensity_scale_ch0 = None,
                intensity_scale_ch1 = None,
                intensity_scale_ch2 = None,
                simulated_RNA_intensities_method='constant',
                basal_intensity_in_background_video= 20000,
                list_files_names_outputs=[]):
        self.metadata_filename = metadata_filename
        self.video_dir = video_dir
        self.masks_dir = masks_dir
        self.list_gene_sequences = list_gene_sequences
        self.list_number_spots = list_number_spots
        self.list_target_channels_proteins = list_target_channels_proteins
        self.list_target_channels_mRNA =  list_target_channels_mRNA
        self.list_diffusion_coefficients = list_diffusion_coefficients
        self.list_label_names = list_label_names
        self.list_elongation_rates = list_elongation_rates
        self.list_initiation_rates = list_initiation_rates
        self.number_cells = number_cells
        self.simulation_time_in_sec = simulation_time_in_sec
        self.step_size_in_sec = step_size_in_sec
        self.frame_selection_empty_video = frame_selection_empty_video
        self.spot_size = spot_size
        self.spot_sigma = spot_sigma
        self.intensity_scale_ch0 = intensity_scale_ch0
        self.intensity_scale_ch1 = intensity_scale_ch1
        self.intensity_scale_ch2 = intensity_scale_ch2
        self.simulated_RNA_intensities_method = simulated_RNA_intensities_method
        self.basal_intensity_in_background_video = basal_intensity_in_background_video
        self.list_files_names = list_files_names_outputs

    def write_metadata(self):
        '''
        This method writes the metadata file.
        '''
        installed_modules = [str(module).replace(" ","==") for module in pkg_resources.working_set]
        important_modules = ['rsnapsim','rsnaped', 'cellpose','trackpy', 'scipy','pathlib','re','glob',  'cv2','imageio','tqdm', 'torch','tifffile', 'setuptools', 'scipy', 'scikit-learn', 'scikit-image', 'pysmb', 'pyfiglet', 'pip', 'Pillow', 'pandas', 'opencv-python-headless', 'numpy', 'numba', 'natsort', 'mrc', 'matplotlib', 'llvmlite', 'jupyter-core', 'jupyter-client', 'joblib', 'ipython', 'ipython-genutils', 'ipykernel']
        def create_data_file(filename):
            if sys.platform == 'linux' or sys.platform == 'darwin':
                os.system('touch  ' + filename)
            elif sys.platform == 'win32':
                os.system('echo , > ' + filename)
        number_spaces_pound_sign = 75
        def write_data_in_file(filename):
            with open(filename, 'w') as fd:
                fd.write('#' * (number_spaces_pound_sign)) 
                fd.write('\nAUTHOR INFORMATION  ')
                fd.write('\n    Author: ' + getpass.getuser())
                fd.write('\n    Created at: ' + datetime.datetime.today().strftime('%d %b %Y'))
                fd.write('\n    Time: ' + str(datetime.datetime.now().hour) + ':' + str(datetime.datetime.now().minute) )
                fd.write('\n    Operative System: ' + sys.platform )
                fd.write('\n    Hostname: ' + socket.gethostname() + '\n')
                fd.write('#' * (number_spaces_pound_sign) ) 
                fd.write('\nPARAMETERS USED  ')
                fd.write('\n    number_simulated_cells: '+ str(self.number_cells) )
                fd.write('\n    simulation_time_in_sec: '+ str(self.simulation_time_in_sec ) )
                fd.write('\n    step_size_in_sec: '+ str(self.step_size_in_sec ) )
                fd.write('\n    frame_selection_empty_video: '+ str(self.frame_selection_empty_video ) )
                fd.write('\n    spot_size: '+ str(self.spot_size ) )
                fd.write('\n    spot_sigma: '+ str(self.spot_sigma ) )
                fd.write('\n    intensity_scale_ch0: '+ str(self.intensity_scale_ch0 ) )
                fd.write('\n    intensity_scale_ch1: '+ str(self.intensity_scale_ch1 ) )
                fd.write('\n    intensity_scale_ch2: '+ str(self.intensity_scale_ch2 ) )
                fd.write('\n    simulated_RNA_intensities_method: '+ str(self.simulated_RNA_intensities_method ) )
                fd.write('\n    basal_intensity_in_background_video: '+ str(self.basal_intensity_in_background_video) )
                fd.write('\n    Parameters for each gene')
                for k in range (0,len(self.list_gene_sequences)):
                    fd.write('\n      Gene File Name: ' + str(pathlib.Path(self.list_gene_sequences[k]).name ) )
                    fd.write('\n        number_spots: ' + str(self.list_number_spots[k]) )
                    fd.write('\n        target_channel_protein: ' + str(self.list_target_channels_proteins[k]) )
                    fd.write('\n        target_channel_mRNA: ' + str(self.list_target_channels_mRNA[k]) )
                    fd.write('\n        diffusion_coefficient: ' + str(self.list_diffusion_coefficients[k]) )
                    fd.write('\n        elongation_rate: ' + str(self.list_elongation_rates[k]) )
                    fd.write('\n        initiation_rates: ' + str(self.list_initiation_rates[k]) )
                    fd.write('\n        label_name: ' + str(self.list_label_names[k]) )
                fd.write('\n') 
                fd.write('#' * (number_spaces_pound_sign) ) 
                fd.write('\n FILES AND DIRECTORIES USED ')
                fd.write('\n    Original video directory: ' + str(self.video_dir) )
                fd.write('\n    Masks directory : ' + str(self.masks_dir)  )
                # for loop for all the images.
                fd.write('\n    Images in the directory :'  )
                counter=0
                for indx, img_name in enumerate (self.list_files_names):
                    fd.write('\n        '+ img_name +  '   - Image Id Number:  ' + str(indx ))
                    counter+=1
                fd.write('\n')  
                fd.write('#' * (number_spaces_pound_sign)) 
                fd.write('\nREPRODUCIBILITY ')
                fd.write('\n    Platform: \n')
                fd.write('        Python: ' + str(platform.python_version()) )
                fd.write('\n    Dependencies: ')
                # iterating for all modules
                for module_name in installed_modules:
                    if any(module_name[0:4] in s for s in important_modules):
                        fd.write('\n        '+ module_name)
                fd.write('\n') 
                fd.write('#' * (number_spaces_pound_sign) ) 
        create_data_file(self.metadata_filename)
        write_data_in_file(self.metadata_filename)
        return None


class MetadataImageProcessing():
    '''
    This class is intended to generate a metadata file containing used dependencies, user information, and parameters used to process single molecule gene expression experiments.
    
    Parameters
    
    The parameters for this class are defined in the SimultedCell class
    '''
    def __init__(self, 
                metadata_filename_ip,
                list_video_paths, 
                files_dir_path_processing,
                particle_size,
                selected_channel_tracking ,
                selected_channel_segmentation ,
                intensity_calculation_method , 
                mask_selection_method ,
                use_optimization_for_tracking,
                average_cell_diameter,
                min_percentage_time_tracking,
                dataframe_format,
                list_segmentation_succesful,
                list_frames_videos):
        self.metadata_filename_ip=metadata_filename_ip
        self.list_video_paths=list_video_paths
        self.files_dir_path_processing=files_dir_path_processing
        self.particle_size=particle_size
        self.selected_channel_tracking=selected_channel_tracking 
        self.selected_channel_segmentation=selected_channel_segmentation 
        self.intensity_calculation_method=intensity_calculation_method 
        self.mask_selection_method=mask_selection_method 
        self.use_optimization_for_tracking=use_optimization_for_tracking
        self.average_cell_diameter=average_cell_diameter
        self.min_percentage_time_tracking=min_percentage_time_tracking
        self.dataframe_format=dataframe_format
        self.list_segmentation_succesful=list_segmentation_succesful
        self.list_frames_videos=list_frames_videos

    def write_metadata(self):
        '''
        This method writes the metadata file.
        '''
        installed_modules = [str(module).replace(" ","==") for module in pkg_resources.working_set]
        important_modules = ['rsnapsim','rsnaped', 'cellpose','trackpy', 'scipy','pathlib','re','glob',  'cv2','imageio','tqdm', 'torch','tifffile', 'setuptools', 'scipy', 'scikit-learn', 'scikit-image', 'pysmb', 'pyfiglet', 'pip', 'Pillow', 'pandas', 'opencv-python-headless', 'numpy', 'numba', 'natsort', 'mrc', 'matplotlib', 'llvmlite', 'jupyter-core', 'jupyter-client', 'joblib', 'ipython', 'ipython-genutils', 'ipykernel']
        def create_data_file(filename):
            if sys.platform == 'linux' or sys.platform == 'darwin':
                os.system('touch  ' + filename)
            elif sys.platform == 'win32':
                os.system('echo , > ' + filename)
        number_spaces_pound_sign = 75
        def write_data_in_file(filename):
            with open(filename, 'w') as fd:
                fd.write('#' * (number_spaces_pound_sign)) 
                fd.write('\nAUTHOR INFORMATION  ')
                fd.write('\n    Author: ' + getpass.getuser())
                fd.write('\n    Created at: ' + datetime.datetime.today().strftime('%d %b %Y'))
                fd.write('\n    Time: ' + str(datetime.datetime.now().hour) + ':' + str(datetime.datetime.now().minute) )
                fd.write('\n    Operative System: ' + sys.platform )
                fd.write('\n    Hostname: ' + socket.gethostname() + '\n')
                fd.write('#' * (number_spaces_pound_sign) ) 
                fd.write('\nPARAMETERS USED  ')
                fd.write('\n    number_processed_images: '+ str(len(self.list_video_paths)) )
                fd.write('\n    particle_size: '+ str(self.particle_size ) )
                fd.write('\n    selected_channel_tracking: '+ str(self.selected_channel_tracking ) )
                fd.write('\n    selected_channel_segmentation: '+ str(self.selected_channel_segmentation ) )
                fd.write('\n    intensity_calculation_method: '+ str(self.intensity_calculation_method ) )
                fd.write('\n    mask_selection_method: '+ str(self.mask_selection_method ) )
                fd.write('\n    use_optimization_for_tracking: '+ str(self.use_optimization_for_tracking ) )
                fd.write('\n    average_cell_diameter: '+ str(self.average_cell_diameter) )
                fd.write('\n    min_percentage_time_tracking: '+ str(self.min_percentage_time_tracking ) )
                fd.write('\n    dataframe_format: '+ str(self.dataframe_format) )
                fd.write('#' * (number_spaces_pound_sign) ) 
                fd.write('\n FILES AND DIRECTORIES USED ')
                fd.write('\n    processed_directory: '+ str(self.files_dir_path_processing ) )
                # for loop for all the images.
                fd.write('\n    Images in the directory :'  )
                counter=0
                for indx, img_name in enumerate (self.list_video_paths):
                    fd.write('\n        '+ pathlib.Path(img_name).name +  '   - Image Id :  ' + str(indx ) +  '   - Frames :  ' + str(self.list_frames_videos[indx]) +  '   - Processing Successful:  '  + str( bool(self.list_segmentation_succesful[indx])) )
                    counter+=1
                fd.write('\n')  
                fd.write('#' * (number_spaces_pound_sign)) 
                fd.write('\nREPRODUCIBILITY ')
                fd.write('\n    Platform: \n')
                fd.write('        Python: ' + str(platform.python_version()) )
                fd.write('\n    Dependencies: ')
                # iterating for all modules
                for module_name in installed_modules:
                    if any(module_name[0:4] in s for s in important_modules):
                        fd.write('\n        '+ module_name)
                fd.write('\n') 
                fd.write('#' * (number_spaces_pound_sign) ) 
        create_data_file(self.metadata_filename_ip)
        write_data_in_file(self.metadata_filename_ip)
        return None



class VisualizerImage():
    '''
    This class is intended to visualize videos as 2D images. This class has the option to mark the particles that previously were selected by trackPy.

    Parameters

    list_videos : List of NumPy arrays or a single NumPy array
        Images or videos to visualize. The format is a list of Numpy arrays where each array follows the convention [T, Y, X, C] or an image array with format [Y, X].
    list_videos_filtered : List of NumPy arrays or a single NumPy array or None
        Images or videos to visualize. The format is a list of Numpy arrays where each array follows the convention [T, Y, X, C]. The default is None.
    list_selected_particles_dataframe : pandas data frame, optional
        A pandas data frame containing the position of each spot in the image. The default is None.
    list_files_names : List of str or str, optional
        List of file names to display as the title on the image. The default is None.
    list_mask_array : List of NumPy arrays or a single NumPy array, with Boolean values, where 1 represents the masked area, and 0 represents the area outside the mask.
        An array of images with dimensions [Y, X].
    selected_channel : int, optional
        Allows the user to define the channel to visualize in the plotted images. The default is 0.
    selected_time_point : int, optional
        Allows the user to define the time point or frame to display on the image. The default is 0.
    normalize : bool, optional
        Option to normalize the data by removing outliers. The code removes the 1 and 99 percentiles from the image. The default is False.
    individual_figure_size : float, optional
        Allows the user to change the size of each image. The default is 5.
    list_real_particle_positions : List of Pandas dataframes or a single dataframe, optional.
        A pandas data frame containing the position of each spot in the image. This dataframe is generated with class SimulatedCell, and it contains the true position for each spot. This option is only intended to be used to train algorithms for tracking and visualize real vs detected spots. The default is None.
    image_name : str, optional.
        Name for the image. The default is 'temp_image.png'.
    show_plot: bool, optional
        Flag to display the image to screen. The default is True.
    save_image_as_file : bool, optional,
        Flag to save image as png. The default is False.
    '''
    def __init__(self, list_videos: list, list_videos_filtered: Union[list, None] = None, list_selected_particles_dataframe: Union[list, None] = None, list_files_names: Union[list, None] = None, list_mask_array: Union[list, None] = None, list_real_particle_positions: Union[list, None] = None, selected_channel:int = 0, selected_time_point:int = 0, normalize:bool = False, individual_figure_size:float = 5,image_name:str='temp_image.png',show_plot:bool=True,save_image_as_file:bool=False,colormap='plasma'):
        self.particle_size = 7
        self.selected_time_point = selected_time_point
        self.selected_channel = selected_channel
        self.individual_figure_size = individual_figure_size
        self.image_name =image_name
        self.show_plot=show_plot
        self.save_image_as_file=save_image_as_file
        self.colormap= colormap
        
        # Checking if the video is a list or a single video.
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
        self.list_number_frames = [list_videos[i].shape[0] for i in range(0, self.number_videos)]
        maximum_time_point_video =  min ( self.list_number_frames ) # Minimum of maximum size in the list of videos.
        if selected_time_point > maximum_time_point_video:
            self.selected_time_point = maximum_time_point_video
        # remove the 1 and 99 percentile if normalize == True
        if normalize == True:
            list_videos_normalized = []
            for index_video in range(0, self.number_videos):
                number_time_points, _, _, number_channels = list_videos[index_video].shape
                temp_video = np.copy(list_videos[index_video])
                for index_channels in range (number_channels):
                    for index_time in range (number_time_points):
                        temp_video[index_time, :, :, index_channels] = RemoveExtrema(temp_video[index_time, :, :, index_channels],format_video='YX').remove_outliers()
                list_videos_normalized.append(temp_video)
            self.list_videos = list_videos_normalized
        else:
            self.list_videos = list_videos
        # This section converts an image [Y, X] into a video with dimensions. [T, Y, X, C].
        if len(list_videos[0].shape) == 2:
            list_videos_4D = []
            for index_video in range(0, self.number_videos):
                temp_video_shape = np.zeros((1, list_videos[index_video].shape[0], list_videos[index_video].shape[1], 1), dtype = np.float32)
                temp_video_shape[0, :, :, 0] = list_videos[index_video]
                list_videos_4D.append(temp_video_shape)
            list_videos = list_videos_4D
            self.selected_channel = 0
            self.selected_time_point = 0
    def plot(self):
        '''
        This method plots a list of images as a grid.

        Returns

        None.
        '''
        # Plotting only the cells
        NUM_COLUMNS = 8
        if ( self.list_selected_particles_dataframe[0] is None):
            NUM_ROWS = int(math.ceil(len(self.list_videos) / NUM_COLUMNS))
            # Loop to plot multiple cells in a grid
            gs = gridspec.GridSpec(NUM_ROWS, NUM_COLUMNS)
            gs.update(wspace = 0.1, hspace = 0.1) # set the spacing between axes.
            fig = plt.figure(figsize = (self.individual_figure_size*NUM_COLUMNS, self.individual_figure_size*NUM_ROWS))
            for index_video in range(0, self.number_videos):
                ax = fig.add_subplot(gs[index_video])
                ax.imshow(self.list_videos[index_video][self.selected_time_point, :, :, self.selected_channel], cmap = 'gray')
                ax.set_xticks([])
                ax.set_yticks([])
                if not ( self.list_files_names[0] is None):
                    ax.set(title = self.list_files_names[index_video][0:-4])
                else:
                    ax.set(title = 'Cell_'+str(index_video))
        # Plotting the cells and the detected spots
        if not ( self.list_selected_particles_dataframe[0] is None) and ( ( self.list_videos_filtered[0] is None)):
            NUM_ROWS = int(math.ceil(len(self.list_videos) / NUM_COLUMNS))
            # Loop to plot multiple cells in a grid
            gs = gridspec.GridSpec(NUM_ROWS, NUM_COLUMNS)
            gs.update(wspace = 0.1, hspace = 0.1) # set the spacing between axes.
            fig = plt.figure(figsize = (self.individual_figure_size*NUM_COLUMNS, self.individual_figure_size*NUM_ROWS))
            counter = 0
            for index_video in range(0, self.number_videos):
                ax = fig.add_subplot(gs[index_video])
                ax.imshow(self.list_videos[index_video][self.selected_time_point, :, :, self.selected_channel], cmap = 'gray')
                ax.set_xticks([])
                ax.set_yticks([])
                if not ( self.list_files_names[0] is None):
                    ax.set(title = self.list_files_names[index_video][0:-4])
                else:
                    ax.set(title = 'Cell_'+str(index_video))
                # main loop to mark spots
                number_particles  = None
                frames_part = None
                x_pos = None
                y_pos = None
                if not (self.list_selected_particles_dataframe[0] is None):
                    selected_particles_dataframe = self.list_selected_particles_dataframe[counter]
                    if not len (self.list_selected_particles_dataframe[counter]) == 0:
                        number_particles = selected_particles_dataframe['particle'].nunique()
                        for k in range (0, number_particles):
                            frames_part = selected_particles_dataframe.loc[selected_particles_dataframe['particle'] == selected_particles_dataframe['particle'].unique()[k]].frame.values
                            index_time = self.selected_time_point
                            if index_time in frames_part: # plotting the circles for each detected particle at a given time point
                                index_val = np.where(frames_part == index_time)
                                x_pos = int(selected_particles_dataframe.loc[selected_particles_dataframe['particle'] == selected_particles_dataframe['particle'].unique()[k]].x.values[index_val])
                                y_pos = int(selected_particles_dataframe.loc[selected_particles_dataframe['particle'] == selected_particles_dataframe['particle'].unique()[k]].y.values[index_val])
                            try:
                                circle = plt.Circle((x_pos, y_pos), self.particle_size/2, color = 'yellow', fill = False)
                                ax.add_artist(circle)
                            except:
                                pass
                # main loop to mark spots ==  > REAL SPOTS. USE FOR SIMULATED CELL
                number_particles  = None
                frames_part = None
                x_pos = None
                y_pos = None
                if not (self.list_real_particle_positions[0] is None):
                    selected_particles_dataframe = self.list_real_particle_positions[counter]
                    if not len (self.list_real_particle_positions[counter]) == 0:
                        number_particles = selected_particles_dataframe['particle'].nunique()
                        for k in range (0, number_particles):
                            frames_part = selected_particles_dataframe.loc[selected_particles_dataframe['particle'] == selected_particles_dataframe['particle'].unique()[k]].frame.values
                            index_time = self.selected_time_point
                            if index_time in frames_part: # plotting the circles for each detected particle at a given time point
                                index_val = np.where(frames_part == index_time)
                                x_pos = int(selected_particles_dataframe.loc[selected_particles_dataframe['particle'] == selected_particles_dataframe['particle'].unique()[k]].x.values[index_val])
                                y_pos = int(selected_particles_dataframe.loc[selected_particles_dataframe['particle'] == selected_particles_dataframe['particle'].unique()[k]].y.values[index_val])
                            try:
                                circle = plt.Circle((x_pos, y_pos), 2, color = 'orangered', fill = False)
                                ax.add_artist(circle)
                            except:
                                pass
                # Plots the mask contour on the video
                if not ( self.list_mask_array[0] is None):
                    mask_array = self.list_mask_array[index_video]
                    if len(mask_array.shape) == 3:
                        contuour_position = find_contours(mask_array[index_time, :, :], 0.8)
                    elif len(mask_array.shape) == 2:
                        contuour_position = find_contours(mask_array[:, :], 0.8)
                    temp = contuour_position[0][:, 1]
                    temp2 = contuour_position[0][:, 0]
                    plt.fill(temp, temp2, facecolor = 'none', edgecolor = 'yellow')
                counter += 1
        # Plotting the cells and the detected spots and the filtered video.
        if (not ( self.list_selected_particles_dataframe[0] is None)) and (not ( self.list_videos_filtered[0] is None)) :
            NUM_ROWS = self.number_videos
            gs = gridspec.GridSpec(NUM_ROWS, NUM_COLUMNS)
            gs.update(wspace = 0.1, hspace = 0.1) # set the spacing between axes.
            fig = plt.figure(figsize = (self.individual_figure_size*NUM_COLUMNS, self.individual_figure_size*NUM_ROWS))
            counter = 0
            for index_video in range(0, self.number_videos*3, 3):
                if not ( self.list_files_names[0] is None):
                    title_str = self.list_files_names[counter][0:-4]
                else:
                    title_str = 'Cell_'+str(counter)
                # Figure with raw video
                ax = fig.add_subplot(gs[index_video])
                ax.imshow(self.list_videos[counter][self.selected_time_point, :, :, self.selected_channel], cmap = self.colormap, vmax = np.max(self.list_videos[counter][self.selected_time_point, :, :, self.selected_channel])*0.95)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set(title = title_str + ' Original')
                # Figure with filtered video
                ax = fig.add_subplot(gs[index_video+1])
                ax.imshow(self.list_videos_filtered[counter][self.selected_time_point, :, :, self.selected_channel], cmap = self.colormap )
                #ax.imshow(self.list_videos_filtered[counter][self.selected_time_point, :, :, self.selected_channel], cmap = 'Greys', vmin = 0 ,vmax = np.max(self.list_videos[counter][self.selected_time_point, :, :, self.selected_channel]))
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set(title = title_str + ' Filtered' )
                # Figure with original video and marking the spots
                ax = fig.add_subplot(gs[index_video+2])
                ax.imshow(self.list_videos[counter][self.selected_time_point, :, :, self.selected_channel], cmap = self.colormap, vmax = np.max(self.list_videos[counter][self.selected_time_point, :, :, self.selected_channel])*0.95)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set(title = title_str + ' Detected Spots' )
                # main loop to mark spots
                number_particles  = None
                frames_part = None
                x_pos = None
                y_pos = None
                # Section that plots the spots
                selected_particles_dataframe = self.list_selected_particles_dataframe[counter]
                if not len (self.list_selected_particles_dataframe[counter]) == 0:
                    number_particles = selected_particles_dataframe['particle'].nunique()
                    for k in range (0, number_particles):
                        frames_part = selected_particles_dataframe.loc[selected_particles_dataframe['particle'] == selected_particles_dataframe['particle'].unique()[k]].frame.values
                        index_time = self.selected_time_point
                        if index_time in frames_part: # plotting the circles for each detected particle at a given time point
                            index_val = np.where(frames_part == index_time)
                            x_pos = int(selected_particles_dataframe.loc[selected_particles_dataframe['particle'] == selected_particles_dataframe['particle'].unique()[k]].x.values[index_val])
                            y_pos = int(selected_particles_dataframe.loc[selected_particles_dataframe['particle'] == selected_particles_dataframe['particle'].unique()[k]].y.values[index_val])
                        try:
                            circle = plt.Circle((x_pos, y_pos), self.particle_size/2, color = 'w', fill = False)
                            ax.add_artist(circle)
                        except:
                            pass
                # main loop to mark spots ==  > REAL SPOTS. USE FOR SIMULATED CELL
                number_particles  = None
                frames_part = None
                x_pos = None
                y_pos = None
                if not ( self.list_real_particle_positions[0] is None):
                    selected_particles_dataframe = self.list_real_particle_positions[counter]
                    if not len (self.list_real_particle_positions[counter]) == 0:
                        number_particles = selected_particles_dataframe['particle'].nunique()
                        for k in range (0, number_particles):
                            frames_part = selected_particles_dataframe.loc[selected_particles_dataframe['particle'] == selected_particles_dataframe['particle'].unique()[k]].frame.values
                            index_time = self.selected_time_point
                            if index_time in frames_part: # plotting the circles for each detected particle at a given time point
                                index_val = np.where(frames_part == index_time)
                                x_pos = int(selected_particles_dataframe.loc[selected_particles_dataframe['particle'] == selected_particles_dataframe['particle'].unique()[k]].x.values[index_val])
                                y_pos = int(selected_particles_dataframe.loc[selected_particles_dataframe['particle'] == selected_particles_dataframe['particle'].unique()[k]].y.values[index_val])
                            try:
                                circle = plt.Circle((x_pos, y_pos), 2, color = 'yellow', fill = True)
                                ax.add_artist(circle)
                            except:
                                pass
                # Plots the mask contour on the video
                if not ( self.list_mask_array[0] is None):
                    mask_array = self.list_mask_array[index_video]
                    if len(mask_array.shape) == 3:
                        contuour_position = find_contours(mask_array[index_time, :, :], 0.8)
                    elif len(mask_array.shape) == 2:
                        contuour_position = find_contours(mask_array[:, :], 0.8)
                    temp = contuour_position[0][:, 1]
                    temp2 = contuour_position[0][:, 0]
                    plt.fill(temp, temp2, facecolor = 'none', edgecolor = 'yellow')
                counter += 1
        fig.tight_layout()
        if self.save_image_as_file == True:
            print(self.image_name)
            plt.savefig(self.image_name,bbox_inches='tight')
        if self.show_plot ==True:
            plt.show()
        else:
            plt.close()
        return None


class VisualizerVideo():
    '''
    This class is intended to visualize videos as interactive widgets. This class has the option to mark the particles that previously were selected by trackPy.

    Parameters

    list_videos : List of NumPy arrays or a single NumPy array
        Images or videos to visualize. The format is a list of Numpy arrays where each array follows the convention [T, Y, X, C] or an image array with format [Y, X].
    dataframe_particles : pandas data frame, optional
        A pandas data frame containing the position of each spot in the image. The default is None.
    list_mask_array : List of NumPy arrays or a single NumPy array, with Boolean values, where 1 represents the masked area, and 0 represents the area outside the mask.
        An array of images with dimensions [Y, X].
    show_time_projection_spots : int, optional
        Allows the user to display the projection of all detected spots for all time points on the current image. The default is False.
    normalize : bool, optional
        Option to normalize the data by removing outliers. The code removes the 1 and 99 percentiles from the image. The default is False.
    step_size_in_sec : float, optional
        Step size in seconds. The default is 1.
    '''
    def __init__(self, list_videos:list, dataframe_particles = None, list_mask_array:list = None, show_time_projection_spots:bool = False, normalize:bool = False, step_size_in_sec:float = 1):
        self.particle_size = 7
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
        self.list_number_frames = [list_videos[i].shape[0] for i in range(0, self.number_videos)]
        self.min_time_all_cells = min(self.list_number_frames)
        # remove the 1 and 99 percentile if normalize == True
        if normalize == True:
            list_videos_normalized = []
            for index_video in range(0, self.number_videos):
                number_time_points, _, _, number_channels = list_videos[index_video].shape
                temp_video = np.copy(list_videos[index_video])
                for index_channels in range (number_channels):
                    for index_time in range (number_time_points):
                        temp_video[index_time, :, :, index_channels] = RemoveExtrema(temp_video[index_time, :, :, index_channels],format_video='YX').remove_outliers()
                list_videos_normalized.append(temp_video)
            self.list_videos = list_videos_normalized
        else:
            self.list_videos = self.list_videos
        n_channels = [self.list_videos[i].shape[3] for i in range(0, self.number_videos)][0]
        self.min_num_channels = np.min((n_channels))
        self.step_size_in_sec = step_size_in_sec
    def make_video_app(self):
        '''
        This method returns two objects (controls and output) that can be used to display a widget.

        Returns

        controls : object
            Controls from interactive to use with ipywidgets **display**.
        output : object
            Output values from from interactive to use with ipywidgets **display**.
        '''
        def figure_viewer(drop_cell:int, index_time_slider:int, drop_channel:int):
            video = self.list_videos[drop_cell]
            selected_particles_dataframe = self.dataframe_particles[drop_cell]
            drop_size = self.particle_size
            index_time = int(index_time_slider/self.step_size_in_sec)
            plt.figure(1)
            ax = plt.gca()
            if drop_channel == 'Ch_0':
                channel = 0
                plt.imshow(video[index_time, :, :, channel], cmap = 'gray', vmax = np.max(video[index_time, :, :, channel])*0.95)
            elif drop_channel == 'Ch_1':
                channel = 1
                plt.imshow(video[index_time, :, :, channel], cmap = 'gray', vmax = np.max(video[index_time, :, :, channel])*0.95)
            elif drop_channel == 'Ch_2':  # vmax = np.mean(video[index_time, :, :, channel])+3*np.std(video[index_time, :, :, channel])
                channel = 2
                plt.imshow(video[index_time, :, :, channel], cmap = 'gray', vmax = np.max(video[index_time, :, :, channel])*0.95)
            elif drop_channel == 'Ch_3':
                channel = 3
                plt.imshow(video[index_time, :, :, channel], cmap = 'gray', vmax = np.max(video[index_time, :, :, channel])*0.95)
            else :
                # Converting a np.uint16 array into float.
                image = np.copy(video[index_time, :, :, :])
                min_image, max_image = np.min(image), np.max(image)
                image -= min_image
                image_float = np.array(image, 'float32')
                image_float *= 255./(max_image-min_image)
                image = np.asarray(np.round(image_float), 'uint8')
                plt.imshow(image, vmax = np.max(image)*0.95)
            # Plots the detected spots.
            if not ( self.dataframe_particles[0] is None):
                n_particles = selected_particles_dataframe['particle'].nunique()
                for k in range (0, n_particles):
                    frames_part = selected_particles_dataframe.loc[selected_particles_dataframe['particle'] == selected_particles_dataframe['particle'].unique()[k]].frame.values
                    if index_time in frames_part: # plotting the circles for each detected particle at a given time point
                        index_val = np.where(frames_part == index_time)
                        x_pos = int(selected_particles_dataframe.loc[selected_particles_dataframe['particle'] == selected_particles_dataframe['particle'].unique()[k]].x.values[index_val])
                        y_pos = int(selected_particles_dataframe.loc[selected_particles_dataframe['particle'] == selected_particles_dataframe['particle'].unique()[k]].y.values[index_val])
                    elif self.show_time_projection_spots == True: # In case the spot is not detected in a given time point, plot the closest the point
                        index_closest = np.abs(frames_part - index_time).argmin()
                        x_pos = int(selected_particles_dataframe.loc[selected_particles_dataframe['particle'] == selected_particles_dataframe['particle'].unique()[k]].x.values[index_closest])
                        y_pos = int(selected_particles_dataframe.loc[selected_particles_dataframe['particle'] == selected_particles_dataframe['particle'].unique()[k]].y.values[index_closest])
                    try:
                        circle = plt.Circle((x_pos, y_pos), drop_size/2, color = 'yellow', fill = False)
                        ax.add_artist(circle)
                    except:
                        pass
            # Plots the mask contour on the video
            if not ( self.list_mask_array[0] is None):
                mask_array = self.list_mask_array[drop_cell]
                if len(mask_array.shape) == 3:
                    contuour_position = find_contours(mask_array[index_time, :, :], 0.8)
                elif len(mask_array.shape) == 2:
                    contuour_position = find_contours(mask_array[:, :], 0.8)
                temp = contuour_position[0][:, 1]
                temp2 = contuour_position[0][:, 0]
                plt.fill(temp, temp2, facecolor = 'none', edgecolor = 'yellow')
            plt.show()
        # This section defines the drop menu for the number of channels in the video.
        if self.min_num_channels == 1:
            options_channels = ['Ch_0']
        if self.min_num_channels == 2:
            options_channels = ['Ch_0', 'Ch_1']
        if self.min_num_channels == 3:
            options_channels = ['Ch_0', 'Ch_1', 'Ch_2', 'All_Channels']
        if self.min_num_channels == 4:
            options_channels = ['Ch_0', 'Ch_1', 'Ch_2', 'Ch_3', 'All_Channels']
        options_cells = list(range(0, self.number_videos))
        interactive_plot = interactive(figure_viewer, drop_cell = widgets.Dropdown(options = options_cells, description = 'Cell'), 
                                        index_time_slider = widgets.IntSlider(min = 0, max = (self.min_time_all_cells-1)*self.step_size_in_sec , step = self.step_size_in_sec, value = 0, description = 'Time'), 
                                        drop_channel = widgets.Dropdown(options = options_channels, description = 'Channel'))
        controls = HBox(interactive_plot.children[:-1], layout = Layout(flex_flow = 'row wrap'))
        output = interactive_plot.children[-1]
        return controls, output


class VisualizerVideo3D():
    '''
    This class is intended to visualize 3d videos as interactive widgets.

    Parameters

    list_videos : List of NumPy arrays or a single NumPy array
        Images or videos to visualize. The format is a list of Numpy arrays where each array follows the convention [T, Y, X, C] or an image array with format [Y, X].
    '''
    def __init__(self, list_videos:list):
        # Checking if the video is a list or a single video.
        if not (type(list_videos) is list):
            list_videos = [list_videos]
            self.list_videos = list_videos
            self.number_videos = 1
        else:
            self.number_videos = len(list_videos)
        self.list_number_frames = [list_videos[i].shape[0] for i in range(0, self.number_videos)]
        self.min_time_all_cells = min(self.list_number_frames)
        list_number_z_slices = [list_videos[i].shape[1] for i in range(0, self.number_videos)]
        self.min_z_slices = min(list_number_z_slices)
        n_channels = [list_videos[i].shape[3] for i in range(0, self.number_videos)][0]
        self.min_num_channels = np.min((n_channels))
    def make_video_app(self):
        '''
        This method returns two objects (controls and output) that can be used to display a widget.

        Returns

        controls : object
            Controls from interactive to use with ipywidgets **display**.
        output : object
            Output values from from interactive to use with ipywidgets **display**.
        '''
        def figure_viewer(drop_cell:int, drop_channel:int, index_z_axis:int, index_time:int):
            plt.figure(1)
            video = self.list_videos[drop_cell]
            if drop_channel == 'Ch_0':
                channel = 0
                plt.imshow(video[index_time, index_z_axis, :, :, channel], cmap = 'gray')
            elif drop_channel == 'Ch_1':
                channel = 1
                plt.imshow(video[index_time, index_z_axis, :, :, channel], cmap = 'gray')
            elif drop_channel == 'Ch_2':
                channel = 2
                plt.imshow(video[index_time, index_z_axis, :, :, channel], cmap = 'gray')
            elif drop_channel == 'Ch_3':
                channel = 3
                plt.imshow(video[index_time, index_z_axis, :, :, channel], cmap = 'gray')
            elif drop_channel == 'All_Channels' :
                # Converting a np.uint16 array into float.
                image = np.copy(video[index_time, index_z_axis:, :, 0:int(np.min((3, self.min_num_channels)))])
                min_image, max_image = np.min(image), np.max(image)
                image -= min_image
                image_float = np.array(image, 'float32')
                image_float *= 255./(max_image-min_image)
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
            options_ch = ['Ch_0', 'Ch_1', 'Ch_2', 'All_Channels']
        if self.min_num_channels == 4:
            options_ch = ['Ch_0', 'Ch_1', 'Ch_2', 'Ch_3', 'All_Channels']
        interactive_plot = interactive(figure_viewer,   drop_cell = widgets.Dropdown(options = options_cell, description = 'Cell'), 
                                                        drop_channel = widgets.Dropdown(options = options_ch, description = 'Channel', value = 'Ch_0'), 
                                                        index_z_axis =  widgets.IntSlider(min = 0, max = self.min_z_slices-1, step = 1, value = 0, description = 'z-slice'), 
                                                        index_time = widgets.IntSlider(min = 0, max = self.min_time_all_cells-1, step = 1, value = 0, description = 'time') )
        controls = HBox(interactive_plot.children[:-1], layout = Layout(flex_flow = 'row wrap'))
        output = interactive_plot.children[-1]
        return controls, output


class VisualizerCrops():
    '''
    This class is intended to visualize the spots detected b trackPy as crops in an interactive widget.

    Parameters

    list_videos : List of NumPy arrays or a single NumPy array.
        Images or videos to visualize. The format is a list of Numpy arrays where each array follows the convention [T, Y, X, C].
    list_selected_particles_dataframe : pandas data frame.
        A pandas data frame containing the position of each spot in the image.
    particle_size : int, optional
        Average particle size. The default is 5.
    '''
    def __init__(self, list_videos:list, list_selected_particles_dataframe:list, particle_size:int = 5):
        if (particle_size % 2) == 0:
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
        self.list_number_frames = [list_videos[i].shape[0] for i in range(0, self.number_videos)]
        self.min_time_all_cells = min(self.list_number_frames)
    def make_video_app(self):
        '''
        This method returns two objects (controls and output) to display a widget.

        Returns

        controls : object
            Controls from interactive to use with ipywidgets **display**.
        output : object
            Output values from from interactive to use with ipywidgets **display**.
        '''
        def figure_viewer(drop_cell:int, track:int, index_time:int):
            video = self.list_videos[drop_cell]
            selected_particles_dataframe = self.list_selected_particles_dataframe[drop_cell]
            n_particles = selected_particles_dataframe['particle'].nunique()
            frames_vector = np.zeros((n_particles, 2))
            for index_particle in range(0, n_particles):
                frames_part = selected_particles_dataframe.loc[selected_particles_dataframe['particle'] == selected_particles_dataframe['particle'].unique()[index_particle]].frame.values
                frames_vector[index_particle, :] = frames_part[0], frames_part[-1]
            def return_crop(image:np.ndarray, x_pos:int, y_pos:int, crop_size:int):
                # function that recenter the spots
                crop_image = image[y_pos-(crop_size):y_pos+(crop_size+1), x_pos-(crop_size):x_pos+(crop_size+1)]
                return crop_image
            number_channels  = video.shape[3]
            size_cropped_image = (100+(self.crop_size+1)) - (100-(self.crop_size)) # true size of crop in image
            ch0_image  = np.zeros((size_cropped_image, size_cropped_image))
            ch1_image  = np.zeros((size_cropped_image, size_cropped_image))
            ch2_image  = np.zeros((size_cropped_image, size_cropped_image))
            frames_part = selected_particles_dataframe.loc[selected_particles_dataframe['particle'] == selected_particles_dataframe['particle'].unique()[track]].frame.values
            if index_time in frames_part: # detecting the position for the crop
                index_val = np.where(frames_part == index_time)
                x_pos = int(selected_particles_dataframe.loc[selected_particles_dataframe['particle'] == selected_particles_dataframe['particle'].unique()[track]].x.values[index_val])
                y_pos = int(selected_particles_dataframe.loc[selected_particles_dataframe['particle'] == selected_particles_dataframe['particle'].unique()[track]].y.values[index_val])
            else: #in case the code doesn't find a position it uses the closes time point value.
                index_closest = np.abs(frames_part - index_time).argmin()
                x_pos = int(selected_particles_dataframe.loc[selected_particles_dataframe['particle'] == selected_particles_dataframe['particle'].unique()[track]].x.values[index_closest])
                y_pos = int(selected_particles_dataframe.loc[selected_particles_dataframe['particle'] == selected_particles_dataframe['particle'].unique()[track]].y.values[index_closest])
            ch0_image[:, :] = return_crop(video[index_time, :, :, 0], x_pos, y_pos, self.crop_size)
            ch1_image[:, :] = return_crop(video[index_time, :, :, 1], x_pos, y_pos, self.crop_size)
            ch2_image[:, :] = return_crop(video[index_time, :, :, 2], x_pos, y_pos, self.crop_size)
            _, ax = plt.subplots(1, number_channels, figsize = (10, 5))
            for index_channels in range(0, number_channels):
                if index_channels == 0:
                    ax[index_channels].imshow(ch0_image[index_time, :, :], origin = 'bottom', cmap = 'gray')
                    ax[index_channels].set_axis_off()
                    ax[index_channels].set(title = 'Channel_0 (Red)')
                elif index_channels == 1:
                    ax[index_channels].imshow(ch1_image[index_time, :, :], origin = 'bottom', cmap = 'gray')
                    ax[index_channels].set_axis_off()
                    ax[index_channels].set(title = 'Channel_1 (Green)')
                else:
                    ax[index_channels].imshow(ch2_image[index_time, :, :], origin = 'bottom', cmap = 'gray')
                    ax[index_channels].set_axis_off()
                    ax[index_channels].set(title = 'Channel_2 (Blue)')
            print('For track '+ str(track) + ' a spot is detected between time points : '+ str(int(frames_vector[track, 0])) + ' and ' + str(int(frames_vector[track, 1])) )
        options_cells = list(range(0, self.number_videos))
        interactive_plot = interactive(figure_viewer, drop_cell = widgets.Dropdown(options = options_cells, description = 'Cell'), 
                                        track = widgets.IntSlider(min = 0, max = self.selected_particles_dataframe['particle'].nunique()-1, step = 1, value = 0, description = 'track'), 
                                        index_time = widgets.IntSlider(min = 0, max = self.video.shape[0]-1, step = 1, value = 0, description = 'time'))
        controls = HBox(interactive_plot.children[:-1], layout = Layout(flex_flow = 'row wrap'))
        output = interactive_plot.children[-1]
        return controls, output
    
class Plots():
    '''
    This class contains miscellaneous plots that are constantly used during the generation of the simulated data.
    '''
    def __init__(self):
        pass
    
    def plot_beads_alignment(first_image_beads,filtered_first_image_beads,positions_in_first_image, second_image_beads,filtered_second_image_beads,positions_in_second_image,spot_size ):
        _, ax = plt.subplots(2,2, figsize=(10, 10))
        ax[0,0].imshow(first_image_beads,cmap='Greys_r')
        ax[0,0].set_xticks([]); ax[0,0].set_yticks([])
        ax[0,0].set_title('Original first image')
        ax[0,1].imshow(filtered_first_image_beads,cmap='Greys_r')
        ax[0,1].set_xticks([]); ax[0,1].set_yticks([])
        ax[0,1].set_title('Filtered image')
        for i in range(0, positions_in_first_image.shape[0]):
            circle1=plt.Circle((positions_in_first_image[i,0], positions_in_first_image[i,1]), spot_size, color = 'yellow', fill = False)
            ax[0,1].add_artist(circle1)        
        ax[1,0].imshow(second_image_beads,cmap='Greys_r')
        ax[1,0].set_xticks([]); ax[1,0].set_yticks([])
        ax[1,0].set_title('Original second image')
        ax[1,1].imshow(filtered_second_image_beads,cmap='Greys_r')
        ax[1,1].set_xticks([]); ax[1,1].set_yticks([])
        ax[1,1].set_title('Filtered image')
        for i in range(0, positions_in_second_image.shape[0]):
            circle2=plt.Circle((positions_in_second_image[i,0], positions_in_second_image[i,1]), spot_size, color = 'yellow', fill = False)
            ax[1,1].add_artist(circle2)
        plt.show()
    
    def plot_tracking_spots(trackpy_dataframe, mean_intensities, mean_intensities_normalized, array_intensities_mean, std_intensities, std_intensities_normalized, step_size,time_points):
        n_particles = trackpy_dataframe['particle'].nunique()
        _, ax = plt.subplots(3, 1, figsize = (16, 4))
        for id in range (0, n_particles):
            frames_part = trackpy_dataframe.loc[trackpy_dataframe['particle'] == trackpy_dataframe['particle'].unique()[id]].frame.values
            ax[0].plot(trackpy_dataframe.loc[trackpy_dataframe['particle'] == trackpy_dataframe['particle'].unique()[id]].frame.values, array_intensities_mean[id, frames_part, 0], 'r')
            ax[1].plot(trackpy_dataframe.loc[trackpy_dataframe['particle'] == trackpy_dataframe['particle'].unique()[id]].frame.values, array_intensities_mean[id, frames_part, 1], 'g')
            ax[2].plot(trackpy_dataframe.loc[trackpy_dataframe['particle'] == trackpy_dataframe['particle'].unique()[id]].frame.values, array_intensities_mean[id, frames_part, 2], 'b')
        ax[0].set(title = 'Trajectories: Intensity vs Time')
        ax[2].set(xlabel = 'Time (a.u.)')
        ax[1].set(ylabel = 'Intensity (a.u.)')
        ax[0].set_xlim([0, time_points-1])
        ax[1].set_xlim([0, time_points-1])
        ax[2].set_xlim([0, time_points-1])
        ax[0].set_xticks([])
        ax[1].set_xticks([])
        plt.show()
        _, ax = plt.subplots(3, 1, figsize = (16, 4))
        for id in range (0, n_particles):
            time_vector = np.arange(0, time_points, 1)*step_size
            ax[0].plot(time_vector, mean_intensities[:, 0], 'r')
            ax[1].plot(time_vector, mean_intensities[:, 1], 'g')
            ax[2].plot(time_vector, mean_intensities[:, 2], 'b')
            ax[0].fill_between(time_vector, mean_intensities[:, 0] - std_intensities[:, 0], mean_intensities[:, 0] + std_intensities[:, 0], color = 'lightgray', alpha = 0.9)
            ax[1].fill_between(time_vector, mean_intensities[:, 1] - std_intensities[:, 1], mean_intensities[:, 1] + std_intensities[:, 1], color = 'lightgray', alpha = 0.9)
            ax[2].fill_between(time_vector, mean_intensities[:, 2] - std_intensities[:, 2], mean_intensities[:, 2] + std_intensities[:, 2], color = 'lightgray', alpha = 0.9)
        ax[0].set(title = 'Mean values')
        ax[2].set(xlabel = 'Time (a.u.)')
        ax[1].set(ylabel = 'Intensity (a.u.)')
        ax[0].set_xlim([0, time_points-1])
        ax[1].set_xlim([0, time_points-1])
        ax[2].set_xlim([0, time_points-1])
        ax[0].set_xticks([])
        ax[1].set_xticks([])
        plt.show()
        _, ax = plt.subplots(3, 1, figsize = (16, 4))
        for id in range (0, n_particles):
            time_vector = np.arange(0, time_points, 1)*step_size
            ax[0].plot(time_vector, mean_intensities_normalized[:, 0], 'r')
            ax[1].plot(time_vector, mean_intensities_normalized[:, 1], 'g')
            ax[2].plot(time_vector, mean_intensities_normalized[:, 2], 'b')
            ax[0].fill_between(time_vector, mean_intensities_normalized[:, 0] - std_intensities_normalized[:, 0], mean_intensities_normalized[:, 0] + std_intensities_normalized[:, 0], color = 'lightgray', alpha = 0.9)
            ax[1].fill_between(time_vector, mean_intensities_normalized[:, 1] - std_intensities_normalized[:, 1], mean_intensities_normalized[:, 1] + std_intensities_normalized[:, 1], color = 'lightgray', alpha = 0.9)
            ax[2].fill_between(time_vector, mean_intensities_normalized[:, 2] - std_intensities_normalized[:, 2], mean_intensities_normalized[:, 2] + std_intensities_normalized[:, 2], color = 'lightgray', alpha = 0.9)
        ax[0].set(title = 'Mean values normalized by max intensity of each trajectory')
        ax[2].set(xlabel = 'Time (a.u.)')
        ax[1].set(ylabel = 'Intensity (a.u.)')
        ax[0].set_xlim([0, time_points-1])
        ax[1].set_xlim([0, time_points-1])
        ax[2].set_xlim([0, time_points-1])
        ax[0].set_xticks([])
        ax[1].set_xticks([])
        plt.show()
            
    def plot_image_channels(image, selected_time_point = 0):
        '''
        This function plots all the channels for the original image.
        '''
        
        if len(image.shape)<4:
            image = np.expand_dims(image, axis = 0)
        number_channels = image.shape[3]
        _ , axes = plt.subplots(nrows=1, ncols=number_channels, figsize=(15, 5))
        for i in range (0,number_channels ):
            img_2d = image[selected_time_point,:,:,i]
            img_2d_rescaled = RemoveExtrema(img_2d, min_percentile = 0.1, max_percentile= 99.5,format_video='YX').remove_outliers()
            axes[i].imshow(img_2d_rescaled, cmap='viridis') 
            axes[i].set_title('Channel_'+str(i))
        plt.show()
    
    def plot_cell_trajectories_ssa_crops(image, df,spot_size=5,selected_channel=1, selected_trajectory=None,microns_per_pixel=None, name_plot=None,perturbation_label=None,perturbation=False,perturbation_time=None):
        # creating temporal folder to save images
        if not (name_plot is None):
            current_dir = pathlib.Path().absolute()
            save_to_dir =  current_dir.joinpath('temp_images' )
            Utilities.test_if_directory_exist_if_not_create(save_to_dir,remove_if_already_exist=False)
            name_plot = save_to_dir.joinpath(name_plot)
        # Extracting vector sizes
        simulation_time_in_sec = df['frame'].values.max()+1
        number_spots_per_cell = df['particle'].values.max()+1
        # name for the vectors based on the user defined channel
        int_mean_values = 'ch'+str(selected_channel)+'_int_mean'
        ssa_values = 'ch'+str(selected_channel)+'_SSA_UMP'
        # extracting data from frame zero
        df_zero = df[['y','x','particle']][(df["frame"] == 0) ]
        if (selected_trajectory is None):
            selected_trajectory = df_zero['particle'].loc[ df_zero[['x']].idxmax()  ].values[0] # This selects the spots that is more to the right side of the image
        spot_size_crop = spot_size+2
        selected_color = '#1C00FE' #'orangered' #'royalblue'
        number_bins = 20
        number_selected_spots =30
        linewidth_value = 1.5
        position_selected_spot = df[['y','x']][(df["frame"] == 0) & (df["particle"] == selected_trajectory)].values[0]
        # Crops
        spot_range = np.linspace(-(spot_size_crop - 1) / 2, (spot_size_crop - 1) / 2, spot_size_crop,dtype=int)
        def return_crop(image, y, x,spot_range):
            crop_image = image[y+spot_range[0]:y+(spot_range[-1]+1), x+spot_range[0]:x+(spot_range[-1]+1)].copy()
            return crop_image
        crop_array = np.zeros(( spot_size_crop, spot_size_crop*number_selected_spots ))
        time_array_crops = np.linspace(0, simulation_time_in_sec-1, number_selected_spots,dtype=int)
        # Creating crops
        counter = 0
        for i,time_crop in enumerate (time_array_crops):
            position_selected_spot_crop = df[['y','x']][(df["frame"] == time_crop) & (df["particle"] == selected_trajectory)].values[0].astype('int')
            crop_array[:spot_size_crop, counter:spot_size_crop+counter] = return_crop (image[time_crop,:,:,selected_channel],position_selected_spot_crop[0],position_selected_spot_crop[1], spot_range)
            counter += spot_size_crop
        # Extracting intensity values
        intensity_values_in_image = np.zeros((number_spots_per_cell,simulation_time_in_sec)) # pre-allocating memory for intensity
        for i in range(number_spots_per_cell):
            intensity_values_in_image[i,:] = df[int_mean_values][df['particle'] ==i].values
        # Extracting SSA values
        ssa_ump = np.zeros((number_spots_per_cell,simulation_time_in_sec)) # pre-allocating memory for intensity
        for i in range(number_spots_per_cell):
            ssa_ump[i,:] = df[ssa_values][df['particle'] ==i].values
        # Plotting
        widths = [1.5, 2, 0.3]
        heights = [0.2, 1, 1]
        plt.style.use(['default'])
        fig = plt.figure(figsize=(11, 5),constrained_layout=True,dpi=300)
        plt.tight_layout() 
        gs = fig.add_gridspec(ncols=3, nrows=3, width_ratios=widths,height_ratios=heights)
        #simulated cell
        f_ax1 = fig.add_subplot(gs[:, 0]); f_ax1.axis('off')
        f_ax1.imshow(image[0,:,:,selected_channel],cmap='Greys_r')
        f_ax1.add_patch(Rectangle((position_selected_spot[1]-7,position_selected_spot[0]-7), 14,14, fill=False,edgecolor=selected_color,lw=1.5))
        if not (microns_per_pixel is None):
            scalebar = ScaleBar(dx = microns_per_pixel, units= 'um', length_fraction=0.2,location='lower right',box_color='k',color='w')
            f_ax1.add_artist(scalebar)
        # Crops
        f_ax2 = fig.add_subplot(gs[0, 1:-1]); 
        f_ax2.set_title('Translation spots')
        f_ax2.imshow(crop_array,cmap='Greys_r')
        f_ax2.set_yticklabels([])
        f_ax2.set_xticklabels([])
        f_ax2.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
        # connections
        connection_top = ConnectionPatch(xyA=(position_selected_spot[1]+spot_size_crop, position_selected_spot[0]-spot_size_crop), xyB=(0,0), coordsA="data", coordsB="data", axesA=f_ax1, axesB=f_ax2, color=selected_color)
        f_ax1.add_artist(connection_top)
        connection_bottom = ConnectionPatch(xyA=(position_selected_spot[1]+spot_size_crop, position_selected_spot[0]+spot_size_crop), xyB=(0,spot_size_crop-1), coordsA="data", coordsB="data", axesA=f_ax1, axesB=f_ax2, color=selected_color)
        f_ax1.add_artist(connection_bottom)
        # trajectories image
        f_ax3 = fig.add_subplot(gs[1, 1])
        f_ax3.plot(intensity_values_in_image.T,'grey',alpha = .1)
        f_ax3.plot(intensity_values_in_image[selected_trajectory,:].T,'-', linewidth = linewidth_value, color = selected_color,label = 'representative')
        #f_ax3.set_xlabel('Time (s)')
        f_ax3.set_ylabel('Intensity (au)')
        f_ax3.set_ylim((intensity_values_in_image.min(),intensity_values_in_image.max()))
        f_ax3.set_title('Intensities from simulated video', color ='k')
        #f_ax3.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        f_ax3.set_xlim((-2,simulation_time_in_sec+2))
        if perturbation == True:
            f_ax3.axvline(x=perturbation_time,linestyle='-', linewidth = 2, color = 'orangered',label = perturbation_label)
        f_ax3.legend(loc='upper right',fontsize=8)
        # dist image
        f_ax4 = fig.add_subplot(gs[1, -1])
        f_ax4.hist(intensity_values_in_image[:,:].flatten() , bins = number_bins, color = selected_color, orientation = 'horizontal')
        f_ax4.set_ylim((intensity_values_in_image.min(),intensity_values_in_image.max()))
        f_ax4.set_xlabel('Count')
        f_ax4.set_yticklabels([]); f_ax4.set_yticks([])
        #trajectories SSA
        f_ax5 = fig.add_subplot(gs[2, 1])
        f_ax5.plot(ssa_ump.T,'grey',alpha = .1)
        f_ax5.plot(ssa_ump[selected_trajectory,:],'-', linewidth = linewidth_value, color = selected_color,label = 'representative')
        f_ax5.set_xlabel('Time (s)')
        f_ax5.set_ylabel('Intensity (au)')
        f_ax5.set_ylim((-1,ssa_ump.max()))
        f_ax5.set_xlim((-2,simulation_time_in_sec+2))
        f_ax5.set_title('Intensities from SSA', color ='k')
        if perturbation == True:
            f_ax5.axvline(x=perturbation_time, linestyle='-', linewidth = 2, color = 'orangered',label = perturbation_label)
        f_ax5.legend(loc='upper right',fontsize=8)
        # Dist SSA
        f_ax6 = fig.add_subplot(gs[-1, -1])
        f_ax6.hist(ssa_ump[:,:].flatten() , bins = number_bins, color = selected_color, orientation = 'horizontal')
        f_ax6.set_ylim((ssa_ump.min(),ssa_ump.max()))
        f_ax6.set_xlabel('Count')
        f_ax6.set_yticklabels([]); f_ax6.set_yticks([])
        plt.subplots_adjust(left=1,right=10,wspace=0.0001, hspace=0.5)
        plt.tight_layout(pad=0.02) 
        if not (name_plot is None):
            plt.savefig(name_plot, transparent=False,dpi=1200, bbox_inches = 'tight', format='pdf')
        plt.show()



class Utilities():
    '''
    This class contains miscellaneous methods to perform tasks needed in multiple classes. No parameters are necessary for this class.
    '''
    def __init__(self):
        pass
    
    # Function that reorder the index to make it continuos 
    def reorder_mask_image(self,mask_image_tested):
        number_masks = np.max(mask_image_tested)
        mask_new =np.zeros_like(mask_image_tested)
        if number_masks>0:
            counter = 0
            for index_mask in range(1,number_masks+1):
                if index_mask in mask_image_tested:
                    counter = counter + 1
                    if counter ==1:
                        mask_new = np.where(mask_image_tested == index_mask, -counter, mask_image_tested)
                    else:
                        mask_new = np.where(mask_new == index_mask, -counter, mask_new)
            reordered_mask = np.absolute(mask_new)
        else:
            reordered_mask = mask_new
        return reordered_mask  
    # Function that reorder the index to make it continuos 
    def remove_artifacts_from_mask_image(self,mask_image_tested, minimal_mask_area_size = 5000):
        number_masks = np.max(mask_image_tested)
        if number_masks>0:
            for index_mask in range(1,number_masks+1):
                mask_size = np.sum(mask_image_tested == index_mask)
                if mask_size <= minimal_mask_area_size:
                    mask_image_tested = np.where(mask_image_tested == index_mask,0,mask_image_tested )
            reordered_mask = Utilities().reorder_mask_image(mask_image_tested)
        else:
            reordered_mask=mask_image_tested
        return reordered_mask 
    
    def log_LL_fun(real_data,simulation_data,nbins=30):
        hist_exp_data, hist_exp_bins = np.histogram( real_data , bins=nbins)
        dist_sim_data, dist_sim_bins = np.histogram(simulation_data, bins=hist_exp_bins, density=True)
        dist_sim_data[dist_sim_data ==0] = 1e-7
        LL_int_distb = np.dot(hist_exp_data, np.log(dist_sim_data))    # likelihood function for comparing distributions
        return -1*LL_int_distb
    
    def remove_extrema_in_vector(vector ,max_percentile = 99):
        '''This function is intended to remove extrema data given by the max percentiles specified by the user'''
        vector = vector [vector>0]
        max_val = np.percentile(vector, max_percentile)
        new_vector = vector [vector< max_val] # = np.percentile(vector,max_percentile)
        return new_vector
    
    def variable_to_list(tested_variable):
        if (isinstance(tested_variable, tuple)==False) and (isinstance(tested_variable, list)==False):
            tested_variable = [tested_variable]
        return tested_variable
    
    def convert_str_to_path(path_to_test):
        if not isinstance(path_to_test, pathlib.PurePath):
            path_to_test = pathlib.Path(path_to_test)
        return path_to_test
    
    def test_if_file_exist(path_to_test):
        path_to_test=Utilities.convert_str_to_path(path_to_test)
        if path_to_test.is_file()== True:
            is_a_file = True
        else:
            is_a_file = False
        return is_a_file
    
    def test_if_directory_exist_if_not_create(path_to_test, remove_if_already_exist = False):
        # making sure the path is a pathlib object
        path_to_test=Utilities.convert_str_to_path(path_to_test)
        # Test if the file or folder exist exist
        if Path.exists(path_to_test):
            if remove_if_already_exist == True:
                # removing existing dir
                shutil.rmtree(str(path_to_test))
                # creating new directory
                path_to_test.mkdir(parents=True, exist_ok=True)
        else:
            path_to_test.mkdir(parents=True, exist_ok=True)
        return None
    
    def remove_folder(path_to_test):
        # making sure the path is a pathlib object
        path_to_test=Utilities.convert_str_to_path(path_to_test)
        if Path.exists(path_to_test):
            # removing existing dir
            shutil.rmtree(str(path_to_test))
        return None
    
    def variable_is_None(tested_variable):
        if tested_variable in (None, 'None', 'none',['None'],['none'],[None]):
            return True
        else:
            return False
    
    def read_files_in_directory(directory, extension_of_files_to_look_for = 'tif',return_images_in_list=True):
        directory = Utilities.convert_str_to_path(directory)
        ## Reads the folder and returns the files inside of the folder as lists
        list_files_names = sorted([f for f in listdir(directory) if isfile(join(directory, f)) and ('.'+extension_of_files_to_look_for) in f], key=str.lower)  # reading all files in the folder
        list_files_names.sort(key=lambda f: int(re.sub('\D', '', f)))  # sorting the index in numerical order
        path_files = sorted([ str(directory.joinpath(f).resolve()) for f in list_files_names if '.'+extension_of_files_to_look_for in f] , key=str.lower)# creating the complete path for each file
        path_files.sort(key=lambda f: int(re.sub('\D', '', f)))  # sorting the index in numerical order
        number_images = len(path_files)
        if return_images_in_list == True:
            list_images = [imread(f) for f in path_files]
        else:
            list_images = None
        return path_files, list_files_names, list_images, number_images
    
    def convert_directory_to_standard_format(directory, time_position = 0, height_position = 1,  width_position = 2, channel_position = 3):
        path_files, _,list_images, number_images = Utilities.read_files_in_directory(directory= directory, extension_of_files_to_look_for = 'tif',return_images_in_list=True)
        directory.joinpath('standard_format').mkdir(parents=True, exist_ok=True)
        path_to_images_in_standard_format = directory.joinpath('standard_format')
        # Showing the simulated images
        for i in range (number_images):
            real_video = ConvertToStandardFormat(video=list_images[i], time_position = time_position, height_position = height_position,  width_position = width_position, channel_position = channel_position ).transpose_video()
            tifffile.imwrite(str(path_to_images_in_standard_format.joinpath(pathlib.Path(path_files[i]).stem+'.tif')), real_video)
        print('The videos converted to standard format are stored in this directory: ', path_to_images_in_standard_format )
        return path_to_images_in_standard_format
    
    def convert_to_int8(image,rescale=True):
        '''
        This method converts images from int16 to uint8. Optionally, the image can be rescaled and stretched.
        
        Parameters
        
        image : NumPy array
            NumPy array with dimensions [T,Y, X, C]. The code expects 3 channels (RGB). If less than 3 values are passed, the array is padded with zeros.
        rescale : bool, optional
            If True it rescales the image to the min and max intensity to 0 and 255. The default is True. 
        '''
        if rescale == True:
            image_new= np.copy(image)
            image_uint8= np.zeros_like(image,dtype='float32')
            for i in range(0, image_new.shape[3]):  # iterate for each channel
                min_intensity, max_intensity = np.min(image_new[:,:,:,i]), np.max(image_new[:,:,:,i])
                image_uint8[:,:,:,i] = image_new[:,:,:,i] - min_intensity
                image_uint8[:,:,:,i] = (image_uint8[:,:,:,i]*255)/(max_intensity-min_intensity)
            image_uint8[image_uint8<0]=0
            image_uint8 = np.uint8(image_uint8)
        else:
            image_new= np.copy(image)
            image_uint8= np.zeros_like(image)
            for i in range(0, image.shape[3]):  # iterate for each channel
                image_new[:,:,:,i]= (image[:,:,:,i]/ image[:,:,:,i].max()) *255
            image_uint8 = np.uint8(image_new)
        # padding with zeros the channel dimenssion.
        if image_uint8.shape[3]<3:
            # padding until having 3 color channels
            while image_new.shape[3]<3:
                zeros_plane = np.zeros_like(image_new[:,:,:,0])
                image_new = np.concatenate((image_new,zeros_plane[:,:,:,np.newaxis]),axis=3)
        # If more than 3 color channels are passed. It only selects the first three color channels.
        elif image_uint8.shape[3]> 3:
            image_uint8 = image_uint8[:,:,:,0:3]
        return image_uint8
    
    
        # Function to convert the video to uint
    def img_uint(image):
        temp_vid = img_as_uint(image)
        return temp_vid

    # temporal function that converts uint to float
    def img_float(image):
        temp_vid = img_as_float64(image)
        return temp_vid

    # Functions with the bandpass and gaussian filters
    def bandpass_filter (image: np.ndarray, lowfilter, highpass):
        temp_vid = difference_of_gaussians(image, lowfilter, highpass, truncate = 3.0)
        return img_as_uint(temp_vid)

    def gaussian_filter(image: np.ndarray, sigma:float = 0.1):
        temp_image = img_as_float64(image)
        filtered_image = gaussian(temp_image, sigma = sigma, output = None, mode = 'nearest', cval = 0, multichannel = None, preserve_range = True, truncate = 4.0)
        return img_as_uint(filtered_image)

    def log_filter(image: np.ndarray, sigma=1):
        temp_image = img_as_float64(image)
        temp_vid = gaussian_laplace(temp_image, sigma=sigma)
        temp_vid = np.clip(-temp_vid, a_min=0, a_max=None)
        return img_as_uint(temp_vid)

    def extract_field_from_dataframe(dataframe_path=None, dataframe=None,selected_time=None,selected_field='ch1_int_mean',use_nan_for_padding=True):
        '''
        This function extracts the selected_field as a vector for a given frame. If selected_time is None, the code will return the extracted 
        data as a NumPy array with dimensions [number_particles, max_time_points]. The maximum time points are defined by the longest trajectory.
        
        Input
            dataframe_path: Patlibpath or str, optional.    
                Path to the selected dataframe.
            dataframe : pandas dataframe
                Dataframe with fields [image_number, cell_number, particle, frame, ch0_int_mean, ch1_int_mean, ch2_int_mean, ch0_int_std, ch1_int_std, ch2_int_std, x, y, ch0_SNR,ch1_SNR,ch2_SNR].
            selected_field : str,
                selected field to extract data.
            selected_time : int, optional
                Selected time point to extract information. The default is None, and indicates to select all time points.
            selected_field : str, optional.
                selected dataframe to extract data.
            use_nan_for_padding: bool, optional
                Option to fill the array with Nans instead of zeros. The default is True.
        Returns

            extracted_data : Selected Field for each particle. NumPy array with dimensions [number_particles, max_time_points] if selected_time is None. The maximum time points are defined by the longest trajectory. Short trajectories are populated with zeros or Nans.
                If selected_time is giive the code will return all existing vlues that meet that condtion.
            
        '''
        def df_to_array(dataframe,selected_field,use_nan_for_padding):
            '''
            This function takes the dataframe and extracts the information from it. 
            
            '''
            # get the total number of particles in all cells
            total_particles = 0
            for image_id in set(dataframe['image_number']):
                for cell in set(dataframe[dataframe['image_number'] == image_id]['cell_number'] ):
                    total_particles += len(set(dataframe[ (dataframe['image_number'] == image_id) & ( dataframe['cell_number'] == cell) ]   ['particle'] ))
            #preallocate numpy array sof n_particles by nframes
            field_as_array = np.zeros([total_particles, np.max(dataframe['frame'])+1] ) 
            if use_nan_for_padding == True:
                field_as_array[:] = np.nan
            k = 0
            # For loops that iterate for each particle and stores the data in the previously pre-alocated arrays.
            for image_id in set(dataframe['image_number']):  #for every cell 
                for cell in set(dataframe[dataframe['image_number'] == image_id]['cell_number'] ):
                    
                    for particle in set( dataframe[(dataframe['image_number'] == image_id)& (dataframe['cell_number'] == cell)    ]['particle'] ): #for every particle
                        temporal_dataframe = dataframe[(dataframe['image_number'] == image_id)  & ( dataframe['cell_number'] == cell)  & (dataframe['particle'] == particle)]  #slice the dataframe
                        frame_values = temporal_dataframe['frame'].values
                        field_as_array[k, frame_values] = temporal_dataframe[selected_field].values  #fill the arrays to return out
                        k+=1 #iterate over k (total particles)
            return field_as_array 
        if not(dataframe_path is None):
            temporal_dataframe = pd.read_csv(dataframe_path)
        else:
            temporal_dataframe = dataframe.copy()
        if not(selected_time is None):
            extracted_data = temporal_dataframe.loc[(temporal_dataframe['frame']==selected_time)][selected_field].values
        else:
            extracted_data = df_to_array(temporal_dataframe,selected_field,use_nan_for_padding)
        return extracted_data
    
    
    
    def remove_spots_from_image(image, x_values, y_values,spot_size):
        img_removed_spots = image.copy()
        for i in range(len(x_values)):
            img_removed_spots[ y_values[i]-spot_size//2:y_values[i]+(spot_size//2)+1,  x_values[i]-spot_size//2:x_values[i]+(spot_size//2)+1 ] = 0
        return img_removed_spots
    def save_video_as_tif(video, saved_file_name ='temp', save_to_path=None):
        if save_to_path == None:
            save_to_path = pathlib.Path().absolute()
        if isinstance(save_to_path, pathlib.PurePath) == False:
            save_to_path = pathlib.Path(save_to_path)
        tifffile.imwrite(str(save_to_path.joinpath(saved_file_name+'.tif')), video)
        print('the video is save in dir:' , str(save_to_path.joinpath(saved_file_name+'.tif')) )
    def save_as_gif (video, saved_file_name ='temp', save_to_path=None, max_frames = None ):
        if save_to_path == None:
            save_to_path = pathlib.Path().absolute()
        if isinstance(save_to_path, pathlib.PurePath) == False:
            save_to_path = pathlib.Path(save_to_path)
        video = Utilities.convert_to_int8(image=video)
        if not (max_frames is None):
            num_images_for_gif = max_frames
        else:
            num_images_for_gif = video.shape[0]
        num_channels_to_plot_in_gif = np.min((3, video.shape[3])) 
        with imageio.get_writer(str(save_to_path.joinpath(saved_file_name+'_unit8'+'.gif')), mode = 'I') as writer:
                for i in range(0, num_images_for_gif):
                    image = video[i, :, :, 0:num_channels_to_plot_in_gif]
                    writer.append_data(image)
        print('the video is save in dir:' ,str(save_to_path.joinpath(saved_file_name+'_unit8'+'.gif')) )



#########################################################
###### LARGE FUNCTIONS TO PERFORM COMPLEX TASKS #########
#########################################################

def simulate_cell ( video_dir, 
                    list_gene_sequences,
                    list_number_spots,
                    list_target_channels_proteins,
                    list_target_channels_mRNA, 
                    list_diffusion_coefficients,
                    list_elongation_rates,
                    list_initiation_rates,
                    list_label_names=None,
                    masks_dir = None,
                    number_cells = 1,
                    simulation_time_in_sec = 100,
                    step_size_in_sec = 1,
                    save_as_tif = True, 
                    save_dataframe = True, 
                    save_as_gif=False,
                    frame_selection_empty_video='gaussian',
                    spot_size = 7 ,
                    spot_sigma=1,
                    intensity_scale_ch0 = None,
                    intensity_scale_ch1 = None,
                    intensity_scale_ch2 = None,
                    dataframe_format = 'long',
                    simulated_RNA_intensities_method='constant',
                    store_videos_in_memory= False,
                    scale_intensity_in_base_video =False,
                    basal_intensity_in_background_video= 20000,
                    microns_per_pixel=1,
                    select_background_cell_index= None,
                    perform_video_augmentation= True,
                    use_Harringtonin = False,
                    use_FRAP = False,
                    perturbation_time_start = 0,
                    perturbation_time_stop = None,
                    save_metadata=False,
                    name_folder=None,
                    photobleaching_parameters = None, #N channels x N pars
                    photobleaching_model = None): #loss or exponential

    '''
    This function is intended to simulate single-molecule translation dynamics in a cell. The result of this simultion is a .tif video and a dataframe containing the transation spot characteristics.

    Parameters

    video_dir :  NumPy array
        Path to an initial video file with the following format Array of images with dimensions [T, Y, X, C].
    masks_dir :  NumPy array
        Path to an image with containing the mask used to segment the original video. The image has the following dimensions [Y, X]. Optional, the  default is None.
    list_gene_sequences : List of strings
        List where every element is a gene sequence file.
    list_number_spots : List of int
        List where every element represents the number of spots to simulate for each gene.
    list_target_channels_proteins : List of int with a range from 0 to 2
        List where every element represents the specific channel where the spots for the nascent proteins will be located.
    list_target_channels_mRNA : List of int with a range from 0 to 2
        List where every element represents the specific channel where the spots for the mRNA signals will be located.
    list_diffusion_coefficients : List of floats
        List where every element represents the diffusion coefficient for every gene. The units are microns^2 per second. The code automatically convert it to pixels^2 per second during the simulation.
    list_label_names : List of str or int
        List where every element contains the label for each gene. Optional the default is None. None will assign an integer label to each gene from 0 to n, ehere n is the number of genes-1.
    list_elongation_rates : List of floats
        List where every element represents the elongation rate for every gene.
    list_initiation_rates : List of floats
        List where every element represents the initiation rate for every gene.
    simulation_time_in_sec : int
        The simulation time in seconds. The default is 20.
    step_size_in_sec : float, optional
        Step size for the simulation. In seconds. The default is 1.
    save_as_tif : bool, optional
        If true, it generates and saves a uint16 (High) quality image tif file for the simulation. The default is True.
    save_dataframe : bool, optional
        If true, it generates and saves a pandas dataframe with the simulation. Dataframe with fields [image_number, cell_number, particle, frame, ch0_int_mean, ch1_int_mean, ch2_int_mean, ch0_int_std, ch1_int_std, ch2_int_std, x, y]. The default is 0.
    save_as_gif : bool, optional
        If true, it generates and saves a .gif with the simulation. The default is 0.
    frame_selection_empty_video : str, optional
        Method to select the frames from the empty video, the options are : 'constant' , 'shuffle' and 'loop'. The default is 'shuffle'.
    spot_size : int, optional
        Spot size in pixels. The default is 5.
    spot_sigma : int, optional.
        Sigma value used to generate a gaussian Point Spread Fucntion. The default is 1.
    intensity_scale_ch0 : float , optional
        Scaling factor for channel 0 that converts the intensity in the stochastic simulations to the intensity in the image.
    intensity_scale_ch1 : float , optional
        Scaling factor for channel 1 that converts the intensity in the stochastic simulations to the intensity in the image.
    intensity_scale_ch2 : float , optional
        Scaling factor for channel 2 that converts the intensity in the stochastic simulations to the intensity in the image.
    simulated_RNA_intensities_method : str, optinal
        Method used to simulate RNA intensities in the image. The optiions are 'constant' or 'random'. The default is 'constant'
    dataframe_format : str, optional
        Format for the dataframe the options are : 'short' , and 'long'. The default is 'short'.
        "long" format generates this dataframe: [image_number, cell_number, particle, frame, ch0_int_mean, ch1_int_mean, ch2_int_mean, ch0_int_std, ch1_int_std, ch2_int_std, x, y, ch0_SNR,ch1_SNR,ch2_SNR].
        "short" format generates this dataframe: [image_number, cell_number, particle, frame, ch0_int_mean, ch1_int_mean, ch2_int_mean, x, y].
    scale_intensity_in_base_video : bool, optional
        Flag to scale intensity to a maximum value of 10000. This arbritary value is selected based on the maximum intensities obtained from the original images. The default is False.
    basal_intensity_in_background_video : int, optional
        if the base video is rescaled, this value indicates the maximum value to rescale the original video. The default is 20000    
    select_background_cell_index : int in range 0 to 8, optional
        Index to select an specific cell for the background. Integer in range 0 to 8, or use None to select a random value. 
    perform_video_augmentation : bool, optional
        If true, it performs random rotations the initial video. The default is True.
    use_Harringtonin: bool, optional
        Flag to specify if Harringtonin is used in the experiment. The default is False.
    use_FRAP: bool
        Flag to specify if FRAP is used in the experiment. The default is False.
    perturbation_time_start: int, optional.
        Time to start the inhibition. The default is 0.
    perturbation_time_stop : int, opt.
        Time to start the inhibition. The default is None.
    name_folder : None or str, optional
        This string indicates the name of the solve where the results are stored. The default is None and it generates a name based on the parameters used for the simulation.

    
    Returns

    dataframe_particles : pandas dataframe
        Dataframe with fields [image_number, cell_number, particle, frame, ch0_int_mean, ch1_int_mean, ch2_int_mean, ch0_int_std, ch1_int_std, ch2_int_std, x, y].
    selected_mask : Numpy array
        Array with the selected mask. Where zeros represents the background and one represent the area in the cell.
    array_intensities : Numpy array
        Array with dimensions [S, T, C].
    time_vector : Numpy array
        1D array.
    mean_intensities: Numpy array
        Array with dimensions [S, T, C].
    std_intensities : Numpy array
        Array with dimensions [S, T, C].
    mean_intensities_normalized : Numpy array
        Array with dimensions [S, T, C].
    std_intensities_normalized : Numpy array
        Array with dimensions [S, T, C].
    '''
    
    # running the simulation
    #start = timer()
    # Testing if the user passed parameters as lists. If not the code conver the parameters into lists
    list_gene_sequences = Utilities.variable_to_list(list_gene_sequences)
    list_number_spots = Utilities.variable_to_list (list_number_spots)
    list_target_channels_proteins = Utilities.variable_to_list (list_target_channels_proteins)
    list_target_channels_mRNA = Utilities.variable_to_list(list_target_channels_mRNA)
    list_diffusion_coefficients = Utilities.variable_to_list (list_diffusion_coefficients)
    list_elongation_rates = Utilities.variable_to_list(list_elongation_rates)
    list_initiation_rates = Utilities.variable_to_list(list_initiation_rates)
    # Creating a list of labels
    if not (list_label_names is None):
        list_label_names = Utilities.variable_to_list(list_label_names)
    else:
        list_label_names = [i for i in range(len(list_gene_sequences) )]
    # Testing if some of the intensities is None. If true, the channel is ignored during the simulation. Resulting in tensor full with zeros.
    if intensity_scale_ch0 is None:
        ignore_ch0 = True
    else:
        ignore_ch0 = False
    if intensity_scale_ch1 is None:
        ignore_ch1 = True
    else:
        ignore_ch1 = False
    if intensity_scale_ch2 is None:
        ignore_ch2 = True
    else:
        ignore_ch2 = False        
    # creating the folder name
    if name_folder is None:
        name_folder = 'bg_' + frame_selection_empty_video 
        name_folder+='_rna_' + simulated_RNA_intensities_method
        name_folder+='_ke_'
        temp_list_ke = ''.join([str(list_elongation_rates[j])+'_' for j in range(len(list_gene_sequences))])
        name_folder += temp_list_ke+'ki_'
        temp_list_ki = ''.join([str(list_initiation_rates[j])+'_' for j in range(len(list_gene_sequences))])
        name_folder+= temp_list_ki+ 'kd_'
        temp_list_kd = ''.join([str(list_diffusion_coefficients[j])+'_' for j in range(len(list_gene_sequences))]) 
        name_folder+= temp_list_kd + 'spots_'
        temp_list_ns= ''.join([str(list_number_spots[j])+'_' for j in range(len(list_gene_sequences))])
        name_folder+= temp_list_ns + 'time_' + str(simulation_time_in_sec) + '_cells_' + str(number_cells)
        name_folder+='_int0_' +str(intensity_scale_ch0)+'_int1_' +str(intensity_scale_ch1)+'_int2_' +str(intensity_scale_ch2)
        name_folder = name_folder.replace(".", "_")
    else:
        name_folder = name_folder
    metadata_name = 'metadata.txt'
    folder_dataframe = 'dataframe_' + name_folder
    folder_video = 'videos_'  + name_folder
    folder_video_int_8 = 'videos_int8' 
    # Functions to create folder to save simulated cells
    current_dir = pathlib.Path().absolute()
    save_to_path_df =  current_dir.joinpath('temp_simulation' ,name_folder, folder_dataframe )
    save_to_path_video =  current_dir.joinpath('temp_simulation',name_folder , folder_video )
    save_to_path_video_int8 =  current_dir.joinpath('temp_simulation',name_folder , folder_video_int_8 )
    # Creating directories
    if save_dataframe == True:
        Utilities.test_if_directory_exist_if_not_create(save_to_path_df,remove_if_already_exist=True)
    if save_as_tif == True:
        Utilities.test_if_directory_exist_if_not_create(save_to_path_video,remove_if_already_exist=True)
    if save_as_gif == True:
        Utilities.test_if_directory_exist_if_not_create(save_to_path_video_int8,remove_if_already_exist=True)
    
    # function  that simulates the multiplexing experiments    
    # Pre-alocating arrays
    list_dataframe_simulated_cell =[]
    list_ssa_all_cells_and_genes =[]
    list_videos = []
    list_files_names_outputs = []
    list_masks  = []
    # Reading all empty cells in directory
    path_files, list_files_names, list_images, num_cell_shapes = Utilities.read_files_in_directory(directory=video_dir, extension_of_files_to_look_for = 'tif',return_images_in_list=True)
    for i in range(0,number_cells): 
        saved_file_name = 'sim_cell_' + str(i)  # if the video or dataframe are save, this variable assigns the name to the files
        if (select_background_cell_index is None):
            selected_video = randrange(num_cell_shapes)
        else:
            if select_background_cell_index < num_cell_shapes:
                selected_video = select_background_cell_index
            else:
                selected_video = 0
        initial_video = list_images[selected_video] #imread(str(path_files[selected_video])) # video with empty cell
        mask_image = imread(masks_dir.joinpath('mask_cell_shape_'+str(selected_video)+'.tif'))
        video, single_dataframe_simulated_cell, list_ssa, mask_used = SimulatedCellDispatcher(initial_video,
                                                                                    list_gene_sequences,
                                                                                    list_number_spots,
                                                                                    list_target_channels_proteins,
                                                                                    list_target_channels_mRNA, 
                                                                                    list_diffusion_coefficients,
                                                                                    list_label_names,
                                                                                    list_elongation_rates,
                                                                                    list_initiation_rates,
                                                                                    simulation_time_in_sec,
                                                                                    step_size_in_sec,
                                                                                    mask_image=mask_image,
                                                                                    image_number =i,
                                                                                    frame_selection_empty_video=frame_selection_empty_video,
                                                                                    spot_size =spot_size ,
                                                                                    spot_sigma=spot_sigma,
                                                                                    intensity_scale_ch0 = intensity_scale_ch0,
                                                                                    intensity_scale_ch1 = intensity_scale_ch1,
                                                                                    intensity_scale_ch2 = intensity_scale_ch2,
                                                                                    dataframe_format=dataframe_format,
                                                                                    simulated_RNA_intensities_method=simulated_RNA_intensities_method,
                                                                                    ignore_ch0 = ignore_ch0,
                                                                                    ignore_ch1 = ignore_ch1,
                                                                                    ignore_ch2 = ignore_ch2,
                                                                                    scale_intensity_in_base_video=scale_intensity_in_base_video,
                                                                                    basal_intensity_in_background_video=basal_intensity_in_background_video,
                                                                                    microns_per_pixel=microns_per_pixel,
                                                                                    perform_video_augmentation=perform_video_augmentation,
                                                                                    use_Harringtonin = use_Harringtonin,
                                                                                    perturbation_time_start = perturbation_time_start,
                                                                                    use_FRAP=use_FRAP,
                                                                                    perturbation_time_stop=perturbation_time_stop,
                                                                                    photobleaching_parameters=photobleaching_parameters,
                                                                                    photobleaching_model=photobleaching_model).make_simulation()
        if save_as_tif == True:
            tifffile.imwrite(str(save_to_path_video.joinpath(saved_file_name+'.tif')), video)
        if save_as_gif == True:
            video_int_8 = Utilities.convert_to_int8(image=video)
            tifffile.imwrite(str(save_to_path_video_int8.joinpath(saved_file_name+'_unit8'+'.tif')), video_int_8)
            num_images_for_gif = video_int_8.shape[0]
            num_channels_to_plot_in_gif = np.min((3, video_int_8.shape[3])) 
            with imageio.get_writer(str(save_to_path_video_int8.joinpath(saved_file_name+'_unit8'+'.gif')), mode = 'I') as writer:
                    for i in range(0, num_images_for_gif):
                        image = video_int_8[i, :, :, 0:num_channels_to_plot_in_gif]
                        writer.append_data(image)
            del video_int_8, image
        if store_videos_in_memory == False:
            video = []
        # appending dataframes for each cell
        list_dataframe_simulated_cell.append(single_dataframe_simulated_cell)
        list_ssa_all_cells_and_genes.append(list_ssa)
        list_videos.append(video)
        list_masks.append(mask_used)
        # list file names
        list_files_names_outputs.append(saved_file_name+'.tif')
    # Concatenating results
    merged_dataframe_simulated_cells = pd.concat(list_dataframe_simulated_cell)
    ssa_trajectories = np.array(list_ssa_all_cells_and_genes)
    # Saving dataframes to folder
    if save_dataframe == True:
        # saving the dataframe
        merged_dataframe_simulated_cells.to_csv( save_to_path_df.joinpath('dataframe_sim_cell.csv'), float_format="%.2f")
        # saving the ssa
        #np.save(save_to_path_df.joinpath('ssas_sim_cell.npy') , ssa_trajectories)
        # creating zip
        shutil.make_archive( base_name = save_to_path_df, format = 'zip', root_dir = save_to_path_df.parents[0], base_dir =save_to_path_df.name )
        #shutil.rmtree(save_to_path_df)
        print('The simulation dataframes are stored here:', str(save_to_path_df))
    # Creating metadata file
    if save_metadata == True:
        metadata_filename= str(current_dir.joinpath('temp_simulation',name_folder,metadata_name))
        MetadataSimulatedCell( metadata_filename,
                                video_dir, 
                                masks_dir, 
                                list_gene_sequences,
                                list_number_spots,
                                list_target_channels_proteins,
                                list_target_channels_mRNA, 
                                list_diffusion_coefficients,
                                list_label_names,
                                list_elongation_rates,
                                list_initiation_rates,
                                number_cells,
                                simulation_time_in_sec,
                                step_size_in_sec,
                                frame_selection_empty_video,
                                spot_size,
                                spot_sigma,
                                intensity_scale_ch0,
                                intensity_scale_ch1,
                                intensity_scale_ch2,
                                simulated_RNA_intensities_method,
                                basal_intensity_in_background_video,
                                list_files_names_outputs).write_metadata()
    #end = timer()
    #print('Time to generate simulated data:',round(end - start), ' sec')
    return list_videos, list_masks, list_dataframe_simulated_cell, merged_dataframe_simulated_cells, ssa_trajectories, list_files_names_outputs, save_to_path_video, save_to_path_df.joinpath('dataframe_sim_cell.csv')


def image_processing(files_dir_path_processing=None,
                    video=None,
                    list_masks = None,
                    particle_size=14,
                    selected_channel_tracking = 0,
                    selected_channel_segmentation = 0,
                    intensity_calculation_method ='disk_donut', 
                    mask_selection_method = 'max_area',
                    show_plot=False,
                    use_optimization_for_tracking=True,
                    real_positions_dataframe = None,
                    average_cell_diameter=200,
                    print_process_times=False,
                    min_percentage_time_tracking=0.3,
                    intensity_threshold_tracking=None,
                    dataframe_format='long',
                    create_pdf=False,
                    create_metadata=False):
    #start = timer()
    list_DataFrame_particles_intensities= []
    list_array_intensities = []
    list_time_vector = []
    list_selected_mask = []
    list_segmentation_succesful=[]
    list_file_names =[]
    list_frames_videos=[]
    
    ## Reads the folder with the results and import the simulations as lists
    if not (files_dir_path_processing is None):
        is_a_file = Utilities.test_if_file_exist(files_dir_path_processing)
        if is_a_file == False:
            path_files, _, _, number_images = Utilities.read_files_in_directory(directory= files_dir_path_processing, extension_of_files_to_look_for = 'tif',return_images_in_list=False)
        else:
            path_files = [Utilities.convert_str_to_path(files_dir_path_processing)]
            number_images = 1
        processing_path_name = files_dir_path_processing.name
    else:
        path_files = None
        number_images = 1
        processing_path_name = 'temp'
        
    # Creating directory to store tracking images.
    current_dir = pathlib.Path().absolute()
    save_to_path_ip =  current_dir.joinpath('temp_processing',processing_path_name )
    Utilities.test_if_directory_exist_if_not_create(save_to_path_ip,remove_if_already_exist=False)
    path_temporal_results =  save_to_path_ip.joinpath('temp_results')
    Utilities.test_if_directory_exist_if_not_create(path_temporal_results,remove_if_already_exist=True)
    list_video_paths = []
    
    for i in range(0,number_images): 
        if not (list_masks is None):
            mask = list_masks[i]
        else:
            mask=None
        if not (path_files is None):
            selected_video = imread(path_files[i]) # Loading the video
            file_name = pathlib.Path(path_files[i]).name
        else:
            selected_video = video
            file_name = 'temp'
            
        frames_in_video = selected_video.shape[0]
        
        if not (real_positions_dataframe is None):
            image_real_positions_dataframe= Utilities.variable_to_list(real_positions_dataframe)
            image_real_positions_dataframe = image_real_positions_dataframe[i]
        else:
            image_real_positions_dataframe = None
        DataFrame_particles_intensities, selected_mask, array_intensities, time_vector, _,_, _, _,segmentation_succesful = PipelineTracking(video=selected_video,
                                                                                                                    mask = mask,
                                                                                                                    particle_size=particle_size,
                                                                                                                    file_name=file_name,
                                                                                                                    selected_channel_tracking=selected_channel_tracking ,
                                                                                                                    selected_channel_segmentation=selected_channel_segmentation ,
                                                                                                                    intensity_calculation_method=intensity_calculation_method , 
                                                                                                                    mask_selection_method=mask_selection_method ,
                                                                                                                    show_plot=show_plot,
                                                                                                                    use_optimization_for_tracking=use_optimization_for_tracking,
                                                                                                                    real_positions_dataframe=image_real_positions_dataframe ,
                                                                                                                    average_cell_diameter=average_cell_diameter,
                                                                                                                    print_process_times=print_process_times,
                                                                                                                    min_percentage_time_tracking=min_percentage_time_tracking,
                                                                                                                    intensity_threshold_tracking=intensity_threshold_tracking,
                                                                                                                    dataframe_format=dataframe_format,
                                                                                                                    create_pdf=create_pdf,
                                                                                                                    path_temporal_results=path_temporal_results,
                                                                                                                    image_index=i).run()    
        list_DataFrame_particles_intensities.append(DataFrame_particles_intensities)
        list_array_intensities.append(array_intensities)
        list_time_vector.append(time_vector)
        list_selected_mask.append(selected_mask)
        if not (path_files is None):
            list_video_paths.append(path_files[i])
        else:
            list_video_paths.append([])
        list_segmentation_succesful.append(segmentation_succesful)
        list_file_names.append(file_name)
        list_frames_videos.append(frames_in_video)
        #print('Progress: ',str(i+1),'/',str(number_images))
    # Creating metadata file
    if create_metadata == True:
        metadata_filename_ip= str(save_to_path_ip.joinpath('metadata_image_processing.txt'))
        MetadataImageProcessing( metadata_filename_ip=metadata_filename_ip,
                                list_video_paths=list_video_paths, 
                                files_dir_path_processing=files_dir_path_processing,
                                particle_size=particle_size,
                                selected_channel_tracking=selected_channel_tracking ,
                                selected_channel_segmentation=selected_channel_segmentation ,
                                intensity_calculation_method=intensity_calculation_method , 
                                mask_selection_method=mask_selection_method ,
                                use_optimization_for_tracking=use_optimization_for_tracking,
                                average_cell_diameter=average_cell_diameter,
                                min_percentage_time_tracking=min_percentage_time_tracking,
                                dataframe_format=dataframe_format,
                                list_segmentation_succesful=list_segmentation_succesful,
                                list_frames_videos=list_frames_videos).write_metadata()
    # Create PDF
    if (create_pdf == True) and (show_plot == False):
        print("To create a pdf with results, It is needed to create plots. Please use the argument 'show_plot=True' when running funtion image_processing. ")
    if (create_pdf == True) and (show_plot == True):
        pdf_report_name = save_to_path_ip.joinpath('processing_report.pdf')
        ReportPDF(directory=path_temporal_results , pdf_report_name=pdf_report_name, list_file_names=list_file_names, list_segmentation_succesful=list_segmentation_succesful ).create_report()
    #end = timer()
    #print('Time to process data:',round(end - start), ' sec')
    return list_DataFrame_particles_intensities, list_array_intensities, list_time_vector, list_selected_mask



# Class spot classification

# Class to calculate tracking quality

# Statistics and PDF

# Autoflorescence removal    

# Class to calculate MSD

# 