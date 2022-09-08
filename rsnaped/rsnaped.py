# !/usr/bin/env python3
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

# To manipulate arrays
import pkg_resources
pkg_resources.require("numpy >= `1.20.1")  #  to use specific numpy version
import numpy as np
from numpy import ndarray
from numpy import unravel_index
# To run stochastic simulations
try:
    import rsnapsim as rss
except:
    print('rsnapsim is not installed')
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
    print('cellpose is not installed')
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
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
plt.style.use("dark_background")
import shutil



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
    def __init__(self,gene_file,ke=10,ki=0.03,frames=300,frame_rate=1,n_traj=20,t_burnin=1000,use_Harringtonin=False,use_FRAP=False, perturbation_time_start=0,perturbation_time_stop=None):
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
        t = np.linspace(0,self.t_burnin+self.frames,(self.t_burnin+self.frames+1)*(self.frame_rate))
        _, _, tagged_pois,raw_seq = rss.seqmanip.open_seq_file(str(self.gene_file))
        gene_obj = tagged_pois['1'][0]
        gene_obj.ke_mu = self.ke
        number_probes = np.amax(gene_obj.probe_vec)
        gene_length = len(raw_seq)

        if not ( self.perturbation_time_stop is None):
            t_stop_perturbation = self.perturbation_time_stop+self.t_burnin
        else:
            t_stop_perturbation = self.t_burnin+self.frames
        perturbation_list = [self.use_FRAP, self.use_Harringtonin,self.perturbation_time_start+self.t_burnin,t_stop_perturbation]

        def ssa_parallel(gene_obj,t,t_burnin,ki ):
            rss.solver.protein = gene_obj #pass the protein object
            ssa_solution = rss.solver.solve_ssa(gene_obj.kelong,t,perturb=perturbation_list,ki=ki, low_memory=True, n_traj=1 )
            return np.transpose( ssa_solution.intensity_vec[0,t_burnin*self.frame_rate:-1,:]) 
        list_ssa = Parallel(n_jobs=self.NUMBER_OF_CORES)(delayed(ssa_parallel)(gene_obj,t,self.t_burnin,self.ki) for i in range(0,self.n_traj)) 
        ssa = np.concatenate( list_ssa, axis=0 )
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
        if type(directory)== pathlib.PosixPath:
            self.directory = directory
        else:
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

        list_files_names = sorted([f for f in listdir(self.directory) if isfile(join(self.directory, f)) and ('.tif') in f], key=str.lower)  # reading all tif files in the folder
        list_files_names.sort(key=lambda f: int(re.sub('\D', '', f)))  # sorting the index in numerical order
        path_files = [ str(self.directory.joinpath(f).resolve()) for f in list_files_names ] # creating the complete path for each file
        number_files = len(path_files)
        list_images = [imread(f) for f in path_files]
        return list_images, path_files, list_files_names, number_files


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
    '''

    def __init__(self, video:np.ndarray, min_percentile:float = 1, max_percentile:float = 99, selected_channels:Union[list, None] = None):
        self.video = video
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile
        if not (type(selected_channels) is list):
                self.selected_channels = [selected_channels]
        else:
            self.selected_channels =selected_channels

    def remove_outliers(self):
        '''
        This method normalizes the values of a video by removing extreme values.

        Returns

        normalized_video : np.uint16
            Normalized video. Array with dimensions [T, Y, X, C] or image with format [Y, X].
        '''

        normalized_video = np.copy(self.video)
        #normalized_video = np.array(normalized_video, 'float32')
        # Normalization code for image with format [Y, X]
        if len(self.video.shape) == 2:
            number_time_points = 1
            number_channels = 1
            normalized_video_temp = normalized_video
            if not np.max(normalized_video_temp) == 0: # this section detect that the channel is not empty to perform the normalization.
                max_val = np.percentile(normalized_video_temp, self.max_percentile)
                min_val = np.min ( (0, np.percentile(normalized_video_temp, self.min_percentile)))
                normalized_video_temp [normalized_video_temp > max_val] = max_val
                normalized_video_temp [normalized_video_temp < min_val] = min_val
                normalized_video_temp [normalized_video_temp < 0] = 0
                normalized_video = normalized_video_temp
        # Normalization for video with format [Y, X, C].
        if len(self.video.shape) == 3:
            number_channels   = self.video.shape[2]
            for index_channels in range (number_channels):
                if (index_channels in self.selected_channels) or (self.selected_channels is None) :
                    normalized_video_temp = normalized_video[ :, :, index_channels]
                    if not np.max(normalized_video_temp) == 0: # this section detect that the channel is not empty to perform the normalization.
                        max_val = np.percentile(normalized_video_temp, self.max_percentile)
                        min_val = np.min ( (0,np.percentile(normalized_video_temp, self.min_percentile)))
                        normalized_video_temp [normalized_video_temp > max_val] = max_val
                        normalized_video_temp [normalized_video_temp < min_val] =  min_val
                        normalized_video_temp [normalized_video_temp < 0] = 0
                        normalized_video[ :, :, index_channels] = normalized_video_temp[ :, :, index_channels] 
        # Normalization for video with format [T, Y, X, C] or [Z, Y, X, C].
        if len(self.video.shape) == 4:
            number_time_points, number_channels   = self.video.shape[0], self.video.shape[3]
            for index_channels in range (number_channels):
                if (index_channels in self.selected_channels) or (self.selected_channels is None) :
                    for index_time in range (number_time_points):
                        normalized_video_temp = normalized_video[index_time, :, :, index_channels]
                        if not np.max(normalized_video_temp) == 0: # this section detect that the channel is not empty to perform the normalization.
                            max_val = np.percentile(normalized_video_temp, self.max_percentile)
                            min_val = np.min ( (0,np.percentile(normalized_video_temp, self.min_percentile)))
                            normalized_video_temp [normalized_video_temp > max_val] = max_val
                            normalized_video_temp [normalized_video_temp < min_val] = min_val
                            normalized_video_temp [normalized_video_temp < 0] = 0
                            normalized_video[ index_time, :, :, index_channels] = normalized_video_temp
            normalized_video[normalized_video<0]=0
        return np.asarray(normalized_video, 'uint16')


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
                        #scaled_video[index_time, :, :, index_channels][scaled_video[index_time, :, :, index_channels] < 0] = 0
                        scaled_video[scaled_video<0]=0
                        scaled_video[index_time, :, :, index_channels] = np.multiply( scaled_video[index_time, :, :, index_channels] , self.scale_maximum_value)                         
        scaled_video_int = scaled_video.astype(int)
        
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
        # temporal function that converts floats to uint
        def img_uint(image):
            temp_vid = img_as_uint(image)
            return temp_vid
        # temporal function that converts uint to float
        def img_float(image):
            temp_vid = img_as_float64(image)
            return temp_vid
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
        # temporal function that converts floats to uint
        def img_uint(image):
            temp_vid = img_as_uint(image)
            return temp_vid
        # returning the image normalized as uint. Notice that difference_of_gaussians converts the image into float.
        for index_channels in range(0, number_channels):
            init_video = Parallel(n_jobs = self.NUMBER_OF_CORES)(delayed(img_uint)(video_bp_filtered_float[i, :, :, index_channels]) for i in range(0, number_time_points))
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
        def img_uint(image):
            temp_vid = img_as_uint(image)
            return temp_vid
        # temporal function that converts uint to float
        def img_float(image):
            temp_vid = img_as_float64(image)
            return temp_vid
        # Pre-allocating arrays
        number_time_points, number_channels   = self.video.shape[0], self.video.shape[3]
        video_bp_filtered_float = np.zeros_like(self.video, dtype = np.float64)
        video_float = np.zeros_like(self.video, dtype = np.float64)
        video_filtered = np.zeros_like(self.video, dtype = np.uint16)
        # returning the image normalized as uint. Notice that difference_of_gaussians converts the image into float.
        for index_channels in range(0, number_channels):
            temp_video = Parallel(n_jobs = self.NUMBER_OF_CORES)(delayed(img_float)(self.video [i, :, :, index_channels]) for i in range(0, number_time_points))
            video_float[:,:,:,index_channels] = np.asarray(temp_video)
        # Applying the filter
        for index_channels in range(0, number_channels):
            for index_time in range(0, number_time_points):
                video_bp_filtered_float[index_time, :, :, index_channels] = difference_of_gaussians(video_float[index_time, :, :, index_channels], self.low_pass, self.high_pass)
        # returning the image normalized as uint. Notice that difference_of_gaussians converts the image into float.
        for index_channels in range(0, number_channels):
            init_video = Parallel(n_jobs = self.NUMBER_OF_CORES)(delayed(img_uint)(video_bp_filtered_float[i, :, :, index_channels]) for i in range(0, number_time_points))
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


#/***************************************************************************************
#    This class has been modified from the following reference.
#    Title: Select ROI within Jupyter Notebook.  bbox_select
#    Author: Prateek Khandelwal
#    Date: July 14, 2019
#    Availability: https://gist.github.com/Pked01/83cdef1dfe49e4004f5af78708767850
class ManualMask():
    def __init__(self,video,time_point_selected,selected_channel):
        normalized_video = RemoveExtrema(video, min_percentile= 0, max_percentile = 99,selected_channels=selected_channel).remove_outliers()
        self.image = normalized_video[time_point_selected,:,:,selected_channel]
        self.selected_points = []
        self.figure_to_draw_points , axes_in_figure = plt.subplots()
        self.new_image = axes_in_figure.imshow(np.copy(self.image),cmap='Spectral_r')
        self.cli = self.figure_to_draw_points.canvas.mpl_connect('button_press_event', self.onclick)
    # Function to draw the polygon in the figure
    def polygon(self,new_image,points_in_polygon):
        points_in_polygon = np.array(points_in_polygon, np.int32)
        points_in_polygon = points_in_polygon.reshape((-1,1,2))
        cv2.polylines(new_image,pts=[points_in_polygon],isClosed=True,color=(0,0,0),thickness=3)
        return new_image
    # Event handling
    def onclick(self, event):
        self.selected_points.append([event.xdata,event.ydata])
        if len(self.selected_points)>=1:
            self.figure_to_draw_points
            self.new_image.set_data(self.polygon(np.copy(self.image),self.selected_points))
#***************************************************************************************/


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
        filtered_first_image_beads= Utilities.log_filter(self.first_image_beads, sigma=1.5)
        filtered_first_image_beads= Utilities.bandpass_filter(filtered_first_image_beads, lowfilter=0.5, highpass=10)
        filtered_second_image_beads = Utilities.log_filter(self.second_image_beads, sigma=1.5)
        filtered_second_image_beads = Utilities.bandpass_filter(filtered_second_image_beads, lowfilter=0.5, highpass=10)
        # Locating beads in the image using "tp.locate" function from trackpy.
        dataframe_spots_in_first_image = tp.locate(filtered_first_image_beads, diameter = self.spot_size, minmass=self.min_intensity, maxsize=self.spot_size*2, preprocess=False,max_iterations=100) # data frame for the first channel
        dataframe_spots_in_second_image = tp.locate(filtered_second_image_beads, diameter= self.spot_size, minmass=self.min_intensity, maxsize=self.spot_size*2, preprocess=False,max_iterations=100)  # data frame for the second channel
        # Converting coordinates to float32 array for the first channel
        x_coord_in_first_image= np.array(dataframe_spots_in_first_image.x.values, np.float32)
        y_coord_in_first_image = np.array(dataframe_spots_in_first_image.y.values, np.float32)
        positions_in_first_image = np.column_stack((x_coord_in_first_image, y_coord_in_first_image ))
        # Converting coordinates to float32 array for the second channel
        x_coord_in_second_image = np.array(dataframe_spots_in_second_image.x.values, np.float32)
        y_coord_in_second_image = np.array(dataframe_spots_in_second_image.y.values, np.float32)
        positions_in_second_image = np.column_stack(( x_coord_in_second_image, y_coord_in_second_image ))
        # First step to remove of unmatched spots. Comparing first versus second image.
        comparison_fist_image = np.zeros((positions_in_first_image.shape[0]))
        comparison_second_image = np.zeros((positions_in_second_image.shape[0]))
        MIN_DISTANCE_TO_MATCH_BEADS = 5
        for i in range (0, positions_in_first_image.shape[0]):
            idx = np.argmin(abs((positions_in_second_image[:, 0] - positions_in_first_image[i, 0])))
            comparison_fist_image[i] = (abs(positions_in_second_image[idx, 0] - positions_in_first_image[i, 0]) < MIN_DISTANCE_TO_MATCH_BEADS) or (abs(positions_in_second_image [idx, 1] - positions_in_first_image[i, 1]) < MIN_DISTANCE_TO_MATCH_BEADS)
        for i in range (0, positions_in_second_image.shape[0]):
            idx = np.argmin(abs((positions_in_first_image[:, 0] - positions_in_second_image[i, 0])))
            comparison_second_image[i] = (abs(positions_in_first_image[idx, 0] - positions_in_second_image[i, 0]) < MIN_DISTANCE_TO_MATCH_BEADS) or (abs(positions_in_first_image [idx, 1] - positions_in_second_image[i, 1]) < MIN_DISTANCE_TO_MATCH_BEADS)
        positions_in_first_image = np.delete(positions_in_first_image, np.where( comparison_fist_image == 0)[0], 0)
        positions_in_second_image = np.delete(positions_in_second_image, np.where(comparison_second_image == 0)[0], 0)
        # Second step to remove of unmatched spots. Comparing second versus first image.
        comparison_fist_image = np.zeros((positions_in_first_image.shape[0]))
        comparison_second_image = np.zeros((positions_in_second_image.shape[0]))
        for i in range (0, positions_in_second_image.shape[0]):
            idx = np.argmin(abs((positions_in_first_image[:, 0] - positions_in_second_image[i, 0])))
            comparison_second_image[i] = (abs(positions_in_first_image[idx, 0] - positions_in_second_image[i, 0]) < MIN_DISTANCE_TO_MATCH_BEADS) or (abs(positions_in_first_image [idx, 1] - positions_in_second_image[i, 1]) < MIN_DISTANCE_TO_MATCH_BEADS)
        for i in range (0, positions_in_first_image.shape[0]):
            idx = np.argmin(abs((positions_in_second_image[:, 0] - positions_in_first_image[i, 0])))
            comparison_fist_image[i] = (abs(positions_in_second_image[idx, 0] - positions_in_first_image[i, 0]) < MIN_DISTANCE_TO_MATCH_BEADS) or (abs(positions_in_second_image [idx, 1] - positions_in_first_image[i, 1]) < MIN_DISTANCE_TO_MATCH_BEADS)
        positions_in_first_image = np.delete(positions_in_first_image, np.where( comparison_fist_image == 0)[0], 0)
        positions_in_second_image = np.delete(positions_in_second_image, np.where(comparison_second_image == 0)[0], 0)
        
        number_spots_first_image = positions_in_first_image.shape[0]
        number_spots_second_image = positions_in_second_image.shape[0]
        print('Calculating the homography matrix between the two images.')
        print('_______ ')
        print(' # Spots in first image : ', number_spots_first_image, '  # Spots in second image : ',number_spots_second_image, '\n')
        print('Spots detected in the first image: ')
        print(np.round(positions_in_first_image[0:np.min( (3, number_spots_first_image)), :] ,1))
        #print('The number of spots detected for the second image are: ', number_spots_second_image, '\n')
        print('Spots detected in the second image:')
        print(np.round(positions_in_second_image[0:np.min((3, number_spots_second_image)),:],1))
        print('_______ ')
        
        if self.show_plot == True:
        # Plotting the detected spots.
            fig, ax = plt.subplots(2,2, figsize=(10, 10))
            ax[0,0].imshow(self.first_image_beads,cmap='Greys_r')
            ax[0,0].set_xticks([]); ax[0,0].set_yticks([])
            ax[0,0].set_title('Original first image')
            ax[0,1].imshow(filtered_first_image_beads,cmap='Greys_r')
            ax[0,1].set_xticks([]); ax[0,1].set_yticks([])
            ax[0,1].set_title('Filtered image')
            for i in range(0, positions_in_first_image.shape[0]):
                circle1=plt.Circle((positions_in_first_image[i,0], positions_in_first_image[i,1]), self.spot_size, color = 'yellow', fill = False)
                ax[0,1].add_artist(circle1)        
            ax[1,0].imshow(self.second_image_beads,cmap='Greys_r')
            ax[1,0].set_xticks([]); ax[1,0].set_yticks([])
            ax[1,0].set_title('Original second image')
            ax[1,1].imshow(filtered_second_image_beads,cmap='Greys_r')
            ax[1,1].set_xticks([]); ax[1,1].set_yticks([])
            ax[1,1].set_title('Filtered image')
            for i in range(0, positions_in_second_image.shape[0]):
                circle2=plt.Circle((positions_in_second_image[i,0], positions_in_second_image[i,1]), self.spot_size, color = 'yellow', fill = False)
                ax[1,1].add_artist(circle2)
        
        # Calculating the minimum value of rows for the alignment
        no_spots_for_alignment = min(positions_in_first_image.shape[0], positions_in_second_image.shape[0])
        homography = transform.ProjectiveTransform()
        src = positions_in_first_image[:no_spots_for_alignment, :2]
        dst = positions_in_second_image[:no_spots_for_alignment, :2]
        homography.estimate(src, dst)
        homography_matrix = homography
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
                    transformed_video[index_time, :, :, index_channels] = warp(self.video[index_time, :, :, index_channels], self.homography_matrix.params, output_shape = (height, width), preserve_range = True)
                else:
                    transformed_video[index_time, :, :, index_channels] = self.video[index_time, :, :, index_channels]
        return transformed_video


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
    '''
    def __init__(self, list_videos: list, list_videos_filtered: Union[list, None] = None, list_selected_particles_dataframe: Union[list, None] = None, list_files_names: Union[list, None] = None, list_mask_array: Union[list, None] = None, list_real_particle_positions: Union[list, None] = None, selected_channel:int = 0, selected_time_point:int = 0, normalize:bool = False, individual_figure_size:float = 5):
        self.particle_size = 7
        self.selected_time_point = selected_time_point
        self.selected_channel = selected_channel
        self.individual_figure_size = individual_figure_size
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
                        temp_video[index_time, :, :, index_channels] = RemoveExtrema(temp_video[index_time, :, :, index_channels]).remove_outliers()
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
                ax.imshow(self.list_videos[counter][self.selected_time_point, :, :, self.selected_channel], cmap = 'viridis', vmax = np.amax(self.list_videos[counter][self.selected_time_point, :, :, self.selected_channel])*0.95)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set(title = title_str + ' Original')
                # Figure with filtered video
                ax = fig.add_subplot(gs[index_video+1])
                ax.imshow(self.list_videos_filtered[counter][self.selected_time_point, :, :, self.selected_channel], cmap = 'Greys' )
                #ax.imshow(self.list_videos_filtered[counter][self.selected_time_point, :, :, self.selected_channel], cmap = 'Greys', vmin = 0 ,vmax = np.amax(self.list_videos[counter][self.selected_time_point, :, :, self.selected_channel]))
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set(title = title_str + ' Filtered' )
                # Figure with original video and marking the spots
                ax = fig.add_subplot(gs[index_video+2])
                ax.imshow(self.list_videos[counter][self.selected_time_point, :, :, self.selected_channel], cmap = 'viridis', vmax = np.amax(self.list_videos[counter][self.selected_time_point, :, :, self.selected_channel])*0.95)
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
                            circle = plt.Circle((x_pos, y_pos), self.particle_size/2, color = 'yellow', fill = False)
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
                                circle = plt.Circle((x_pos, y_pos), 2, color = 'orangered', fill = True)
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
        plt.show()


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
                        temp_video[index_time, :, :, index_channels] = RemoveExtrema(temp_video[index_time, :, :, index_channels]).remove_outliers()
                list_videos_normalized.append(temp_video)
            self.list_videos = list_videos_normalized
        else:
            self.list_videos = self.list_videos
        n_channels = [self.list_videos[i].shape[3] for i in range(0, self.number_videos)][0]
        self.min_num_channels = np.amin((n_channels))
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
                plt.imshow(video[index_time, :, :, channel], cmap = 'gray', vmax = np.amax(video[index_time, :, :, channel])*0.95)
            elif drop_channel == 'Ch_1':
                channel = 1
                plt.imshow(video[index_time, :, :, channel], cmap = 'gray', vmax = np.amax(video[index_time, :, :, channel])*0.95)
            elif drop_channel == 'Ch_2':  # vmax = np.mean(video[index_time, :, :, channel])+3*np.std(video[index_time, :, :, channel])
                channel = 2
                plt.imshow(video[index_time, :, :, channel], cmap = 'gray', vmax = np.amax(video[index_time, :, :, channel])*0.95)
            elif drop_channel == 'Ch_3':
                channel = 3
                plt.imshow(video[index_time, :, :, channel], cmap = 'gray', vmax = np.amax(video[index_time, :, :, channel])*0.95)
            else :
                # Converting a np.uint16 array into float.
                image = np.copy(video[index_time, :, :, :])
                min_image, max_image = np.min(image), np.max(image)
                image -= min_image
                image_float = np.array(image, 'float32')
                image_float *= 255./(max_image-min_image)
                image = np.asarray(np.round(image_float), 'uint8')
                plt.imshow(image, vmax = np.amax(image)*0.95)
            # Plots the detected spots.
            if not ( self.dataframe_particles[0] is None):
                n_particles = selected_particles_dataframe['particle'].nunique()
                for k in range (0, n_particles):
                    frames_part = selected_particles_dataframe.loc[selected_particles_dataframe['particle'] == selected_particles_dataframe['particle'].unique()[k]].frame.values
                    if index_time in frames_part: # plotting the circles for each detected particle at a given time point
                        index_val = np.where(frames_part == index_time)
                        x_pos = int(selected_particles_dataframe.loc[selected_particles_dataframe['particle'] == selected_particles_dataframe['particle'].unique()[k]].x.values[index_val])
                        y_pos = int(selected_particles_dataframe.loc[selected_particles_dataframe['particle'] == selected_particles_dataframe['particle'].unique()[k]].y.values[index_val])
                    elif self.show_time_projection_spots == 1: # In case the spot is not detected in a given time point, plot the closest the point
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
        self.min_num_channels = np.amin((n_channels))
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
                image = np.copy(video[index_time, index_z_axis:, :, 0:int(np.amin((3, self.min_num_channels)))])
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
            red_image  = np.zeros((size_cropped_image, size_cropped_image))
            green_image  = np.zeros((size_cropped_image, size_cropped_image))
            blue_image  = np.zeros((size_cropped_image, size_cropped_image))
            frames_part = selected_particles_dataframe.loc[selected_particles_dataframe['particle'] == selected_particles_dataframe['particle'].unique()[track]].frame.values
            if index_time in frames_part: # detecting the position for the crop
                index_val = np.where(frames_part == index_time)
                x_pos = int(selected_particles_dataframe.loc[selected_particles_dataframe['particle'] == selected_particles_dataframe['particle'].unique()[track]].x.values[index_val])
                y_pos = int(selected_particles_dataframe.loc[selected_particles_dataframe['particle'] == selected_particles_dataframe['particle'].unique()[track]].y.values[index_val])
            else: #in case the code doesn't find a position it uses the closes time point value.
                index_closest = np.abs(frames_part - index_time).argmin()
                x_pos = int(selected_particles_dataframe.loc[selected_particles_dataframe['particle'] == selected_particles_dataframe['particle'].unique()[track]].x.values[index_closest])
                y_pos = int(selected_particles_dataframe.loc[selected_particles_dataframe['particle'] == selected_particles_dataframe['particle'].unique()[track]].y.values[index_closest])
            red_image[:, :] = return_crop(video[index_time, :, :, 0], x_pos, y_pos, self.crop_size)
            green_image[:, :] = return_crop(video[index_time, :, :, 1], x_pos, y_pos, self.crop_size)
            blue_image[:, :] = return_crop(video[index_time, :, :, 2], x_pos, y_pos, self.crop_size)
            _, ax = plt.subplots(1, number_channels, figsize = (10, 5))
            for index_channels in range(0, number_channels):
                if index_channels == 0:
                    ax[index_channels].imshow(red_image[index_time, :, :], origin = 'bottom', cmap = 'gray')
                    ax[index_channels].set_axis_off()
                    ax[index_channels].set(title = 'Channel_0 (Red)')
                elif index_channels == 1:
                    ax[index_channels].imshow(green_image[index_time, :, :], origin = 'bottom', cmap = 'gray')
                    ax[index_channels].set_axis_off()
                    ax[index_channels].set(title = 'Channel_1 (Green)')
                else:
                    ax[index_channels].imshow(blue_image[index_time, :, :], origin = 'bottom', cmap = 'gray')
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


class Cellpose():
    '''
    This class is intended to detect cells by image masking using **Cellpose** . The class uses optimization to maximize the number of cells or maximize the size of the detected cells.

    Parameters

    video : NumPy array
        Array of images with dimensions [T, Y, X, C].
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
    def __init__(self, video:np.ndarray, num_iterations:int = 5, channels:list = [0, 0], diameter:float = 120, model_type:str = 'cyto', selection_method:str = 'max_cells_and_area'):
        self.video = video
        self.num_iterations = num_iterations
        self.minimum_probability = 0
        self.maximum_probability = 4
        self.channels = channels
        self.diameter = diameter
        self.model_type = model_type # options are 'cyto' or 'nuclei'
        self.selection_method = selection_method # options are 'max_area' or 'max_cells'
        self.optimization_parameter = np.round(np.linspace(self.minimum_probability, self.maximum_probability, self.num_iterations), 2)

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
                masks, _, _, _ = model.eval(self.video, normalize = True, cellprob_threshold = optimization_threshold, diameter = self.diameter, min_size = 1000, channels = self.channels, progress = None)
            except:
                masks =0
            n_masks = np.amax(masks)
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
                masks, _, _, _ = model.eval(self.video, normalize = True, cellprob_threshold = optimization_threshold, diameter =self.diameter, min_size = 1000, channels = self.channels, progress = None)
            except:
                masks =0
            return np.amax(masks)
        def cellpose_max_cells_and_area( optimization_threshold):
            try:
                masks, _, _, _ = model.eval(self.video, normalize = True, cellprob_threshold = optimization_threshold, diameter = self.diameter, min_size = 1000, channels = self.channels, progress = None)
            except:
                masks =0
            n_masks = np.amax(masks)
            if n_masks > 1: # detecting if more than 1 mask are detected per cell
                size_mask = []
                for nm in range (1, n_masks+1): # iterating for each mask in a given cell. The mask has values from 0 for background, to int n, where n is the number of detected masks.
                    size_mask.append(np.sum(masks == nm)) # creating a list with the size of each mask
                number_masks= np.amax(masks)
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
        if np.amax(evaluated_metric_for_masks) >0:
            selected_conditions = self.optimization_parameter[np.argmax(evaluated_metric_for_masks)]
            selected_masks, _, _, _ = model.eval(self.video, normalize = True, cellprob_threshold = selected_conditions, diameter = self.diameter, min_size = -1, channels = self.channels, progress = None)
            selected_masks[0:10, :] = 0;selected_masks[:, 0:10] = 0;selected_masks[selected_masks.shape[0]-10:selected_masks.shape[0]-1, :] = 0; selected_masks[:, selected_masks.shape[1]-10: selected_masks.shape[1]-1 ] = 0#This line of code ensures that the corners are zeros.
        else:
            selected_masks = None
            print('No cells detected on the image')
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
        Options used by the optimization algorithm to select a cell based on the number of cells or the number of spots. The options are: 'max_area' or 'max_spots'. The default is 'maximum_area'.
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
            n_masks = np.amax(self.mask)
            if n_masks > 1: # detecting if more than 1 mask are detected per cell
                size_mask = []
                for nm in range (1, n_masks+1): # iterating for each mask in a given cell. The mask has values from 0 for the background to int n, where n is the number of detected masks.
                    size_mask.append(np.sum(self.mask == nm)) # creating a list with the size of each mask
                largest_mask = np.argmax(size_mask)+1 # detecting the mask with the largest value
                temp_mask = np.zeros_like(self.mask) # making a copy of the image
                selected_mask = temp_mask + (self.mask == largest_mask) # Selecting a single mask and making this mask equal to one and the background equal to zero.
            else: # do nothing if only a single mask is detected per image.
                selected_mask = self.mask
        if self.selection_method == 'max_spots':
            # Iterating for each mask to select the mask with the largest area.
            n_masks = np.amax(self.mask)
            if n_masks > 1: # detecting if more than 1 mask are detected per cell
                number_particles = []
                for nm in range (1, n_masks+1): # iterating for each mask in a given cell. The mask has values from 0 for the background to int n, where n is the number of detected masks.
                    # # Apply mask to a given time point
                    mask_copy = np.copy(self.mask)
                    tested_mask = np.where(mask_copy == nm, 1, 0) # making zeros all elements outside each mask, and once all elements inside of each mask.
                    video_minimal_time = np.amin((int(self.num_frames/3), 5, self.num_frames))
                    _, number_detected_trajectories, _ = Trackpy(self.video[0:video_minimal_time, :, :, :], tested_mask, particle_size = self.particle_size, selected_channel = self.selected_channel , minimal_frames = self.minimal_frames, show_plot = 0).perform_tracking()                    
                    number_particles.append(number_detected_trajectories)
                pre_selected_mask = np.argmax(number_particles)+1 # detecting the mask with the largest value
                temp_mask = np.zeros_like(self.mask, dtype = np.uint16) # making a copy of the image
                selected_mask = temp_mask + (self.mask == pre_selected_mask) # Selecting a single mask and making this mask equal to one and the background equal to zero.
            else:  # do nothing if only a single mask is detected per image.
                selected_mask = self.mask
            if np.amax(selected_mask) == 0:
                selected_mask = None
                print('No mask was selected in the image.')
                # This section dilates the mask to connect areas that are isolated.
        
        mask_int = np.where(selected_mask > 0.5, 1, 0).astype(np.int)
        dilated_image = dilation(mask_int, square(20))
        mask_final = np.where(dilated_image > 0.5, 1, 0).astype(np.int)
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
    '''
    def __init__(self, video:np.ndarray, mask:np.ndarray, particle_size:int = 5, selected_channel:int = 0, minimal_frames:int = 5, optimization_iterations:int = 10, use_default_filter:bool = True, FISH_image: bool = False, show_plot:bool = True,intensity_threshold_tracking=None):
        self.NUMBER_OF_CORES = multiprocessing.cpu_count()
        self.time_points = video.shape[0]
        self.selected_channel = selected_channel
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
        
        # Function to convert the video to uint
        def img_uint(image):
            temp_vid = img_as_uint(image)
            return temp_vid
        ini_video = np.asarray(Parallel(n_jobs = self.NUMBER_OF_CORES)(delayed(img_uint)(video[i, :, :, self.selected_channel]) for i in range(0, self.time_points)) )
        
        def filter_video(video, tracking_filter,frames_to_track):
            # function that remove outliers from the video
            video = RemoveExtrema(video, min_percentile = 0.5, max_percentile = 99.9).remove_outliers()
            # selecting the filter to apply
            if tracking_filter == 'bandpass_filter':
                temp_vid_dif_filter = Parallel(n_jobs = self.NUMBER_OF_CORES)(delayed(bandpass_filter)(video[i, :, :], self.low_pass_filter, self.highpass_filter) for i in range(0, frames_to_track))
            elif tracking_filter == 'log_filter':       
                temp_vid_dif_filter = Parallel(n_jobs = self.NUMBER_OF_CORES)(delayed(log_filter)(video[i, :, :], sigma=1.5) for i in range(0, frames_to_track))
            elif tracking_filter == 'all':  
                temp_vid_filter = Parallel(n_jobs = self.NUMBER_OF_CORES)(delayed(bandpass_filter)(video[i, :, :], self.low_pass_filter, self.highpass_filter) for i in range(0, frames_to_track))
                temp_vid = np.asarray(temp_vid_filter)
                temp_vid_dif_filter = Parallel(n_jobs = self.NUMBER_OF_CORES)(delayed(log_filter)(temp_vid[i, :, :], sigma=1.5) for i in range(0, frames_to_track))
            video_filtered = np.asarray(temp_vid_dif_filter)
            return video_filtered
        
        
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
        if self.FISH_image == 1:
            self.min_time_particle_vanishes = 1
            self.max_distance_particle_moves = 1
            self.minimal_frames = minimal_frames
        #self.tracking_filter = 'bandpass_filter' # 'bandpass_filter' , 'log_filter', 'all'
        #self.tracking_filter = 'log_filter'
        self.tracking_filter = 'all'
        self.video_filtered = filter_video(video=ini_video, tracking_filter=self.tracking_filter,frames_to_track=self.time_points)
        self.MIN_INT_OPTIMIZATION = 1 
        if use_default_filter ==0:
            f_init = tp.locate(self.video_filtered[0, :, :], self.particle_size, minmass = 1, max_iterations = 100, preprocess = False, percentile = 70)
            self.MAX_INT_OPTIMIZATION = np.max( (1, np.round( np.max(f_init.mass.values) ) ) )
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
                    min_int = np.amax( (0, np.round( np.mean(f_init.mass.values) + self.default_threshold_int_std *np.std(f_init.mass.values))))
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
            ignored_edges = 40
            derivative_vector_detected_spots = np.gradient(log_num_spots[ignored_edges:])      #  derivative
            index_max_second_derivative = derivative_vector_detected_spots.argmax()+ignored_edges #+ self.ADDED_INDEX_TO_OPTIMIZED_SELECTION 
            selected_int_optimized = min_int_vector[index_max_second_derivative]  # + self.ADDED_INTENSITY_TO_OPTIMIZED_SELECTION
            trackpy_dataframe, number_particles = video_tracking(video=self.video_filtered, mask=self.mask, min_int= selected_int_optimized)
            if self.show_plot == 1:
                plt.figure(figsize =(5,5))
                plt.plot(min_int_vector, log_num_spots, label='norm detected_spots',linewidth=5,color='lime')
                plt.plot(min_int_vector[index_max_second_derivative], log_num_spots[index_max_second_derivative], 'o',label='selected threshold', markersize=20, markerfacecolor='orangered')
                #plt.plot(min_int_vector,normalized_second_derivative, label=r"$f''(spots)$",color='orangered',linewidth=1)
                plt.xlabel('Threshold intensity', size=16)
                plt.ylabel('log (number of spots)', size=16)
                plt.show()
                print('The number of detected trajectories is: ', number_particles)
                print('The selected intensity threshold is: ', str(selected_int_optimized), '\n' )
        video_filtered = np.expand_dims(self.video_filtered,axis=3)
        return trackpy_dataframe, int(number_particles), video_filtered


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
        Array of images with dimensions [T, S, x_y_positions].  The default is None
    dataframe_format : str, optional
        Format for the dataframe the options are : 'short' , and 'long'. The default is 'short'.
        "long" format generates this dataframe: [cell_number, particle, frame, red_int_mean, green_int_mean, blue_int_mean, red_int_std, green_int_std, blue_int_std, x, y, SNR_red,SNR_green,SNR_blue].
        "short" format generates this dataframe: [cell_number, particle, frame, red_int_mean, green_int_mean, blue_int_mean, x, y].
    method : str, optional
        Method to calculate intensity the options are : 'total_intensity' , 'disk_donut' and 'gaussian_fit'. The default is 'disk_donut'.
    step_size : float, optional
        Frame rate in seconds. The default is 1 frame per second.
    show_plot : bool, optional
        Allows the user to show a plot for the optimization process. The default is True.
    '''
    def __init__(self, video:np.ndarray, particle_size:int = 5, trackpy_dataframe: Union[object , None ] = None, spot_positions_movement: Union[np.ndarray, None] = None,dataframe_format:str = 'short',   method:str = 'disk_donut', step_size:float = 1, show_plot:bool = True):
        if particle_size < 3:
            particle_size = 3 # minimal size allowed for detection
        if (particle_size % 2) == 0:
            particle_size = particle_size + 1
            print('particle_size must be an odd number, this was automatically changed to: ', particle_size)
        self.video = video
        self.trackpy_dataframe = trackpy_dataframe
        self.disk_size = int(particle_size/2) # size of the half of the crop
        self.crop_size = int(particle_size/2)*2
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
    
    def calculate_intensity(self):
        '''
        This method calculates the spot intensity.

        Returns

        dataframe_particles : pandas dataframe
            Dataframe with fields [cell_number, particle, frame, red_int_mean, green_int_mean, blue_int_mean, red_int_std, green_int_std, blue_int_std, x, y, SNR_red,SNR_green,SNR_blue].
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
        array_intensities_mean = np.zeros((self.n_particles, time_points, number_channels))
        array_intensities_std = np.zeros((self.n_particles, time_points, number_channels))
        array_intensities_snr = np.zeros((self.n_particles, time_points, number_channels))
        array_intensities_background_mean = np.zeros((self.n_particles, time_points, number_channels))
        array_intensities_background_std = np.zeros((self.n_particles, time_points, number_channels))
        def gaussian_fit(test_im):
            size_spot = test_im.shape[0]
            image_flat = test_im.ravel()
            def gaussian_function(size_spot, offset, sigma):
                ax = np.linspace(-(size_spot - 1) / 2., (size_spot - 1) / 2., size_spot)
                xx, yy = np.meshgrid(ax, ax)
                kernel =  offset *(np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma)))
                return kernel.ravel()
            p0 = (np.amin(image_flat) , np.std(image_flat) ) # int(size_spot/2))
            optimized_parameters, _ = curve_fit(gaussian_function, size_spot, image_flat, p0 = p0)
            spot_intensity_gaussian = optimized_parameters[0] # Amplitude
            spot_intensity_gaussian_std = optimized_parameters[1]
            return spot_intensity_gaussian, spot_intensity_gaussian_std
        def signal_to_noise_ratio(test_im:np.ndarray, disk_size:int):
            # Function that calculates intensity using the disk and donut method
            center_coordinates = int(test_im.shape[0]/2)
            # mean intensity in disk
            image_in_disk = test_im[center_coordinates-disk_size:center_coordinates+disk_size+1, center_coordinates-disk_size:center_coordinates+disk_size+1]
            mean_intensity_disk = np.mean(image_in_disk.flatten())            
            # Calculate SD in the donut
            recentered_image_donut = test_im.copy().astype(np.float32)
            recentered_image_donut[center_coordinates-disk_size:center_coordinates+disk_size+1, center_coordinates-disk_size:center_coordinates+disk_size+1] = np.nan
            mean_intensity_donut = np.nanmean(recentered_image_donut.flatten()) # mean calculation ignoring zeros
            std_intensity_donut = np.nanstd(recentered_image_donut.flatten()) # mean calculation ignoring zeros
            # Calculate SNR
            if std_intensity_donut >0:
                calculated_signal_to_noise_ratio = (mean_intensity_disk-mean_intensity_donut) / std_intensity_donut
            else:
                calculated_signal_to_noise_ratio = 0
            # Calculation using a gaussian filter
            mean_background_int = mean_intensity_donut
            std_background_int = std_intensity_donut
            return calculated_signal_to_noise_ratio, mean_background_int,std_background_int
        def disk_donut(test_im:np.ndarray, disk_size:int):
            # Function that calculates intensity using the disk and donut method
            center_coordinates = int(test_im.shape[0]/2)
            recentered_image_donut = test_im.copy().astype(np.float32)
            # mean intensity in disk
            image_in_disk = test_im[center_coordinates-disk_size:center_coordinates+disk_size+1, center_coordinates-disk_size:center_coordinates+disk_size+1]
            mean_intensity_disk = np.mean(image_in_disk)
            spot_intensity_disk_donut_std = np.std(image_in_disk)
            recentered_image_donut[center_coordinates-disk_size:center_coordinates+disk_size+1, center_coordinates-disk_size:center_coordinates+disk_size+1] = np.nan
            mean_intensity_donut = np.nanmedian(recentered_image_donut.flatten()) # mean calculation ignoring zeros
            # subtracting background minus center intensity
            spot_intensity_disk_donut = np.array( mean_intensity_disk - mean_intensity_donut, dtype = np.float32)
            #spot_intensity_disk_donut[spot_intensity_disk_donut < 0] = 0
            spot_intensity_disk_donut[np.isnan(spot_intensity_disk_donut)] = 0 # replacing nans with zero
            return spot_intensity_disk_donut, spot_intensity_disk_donut_std
        def return_crop(image:np.ndarray, x_pos:int, y_pos:int, crop_size:int):
            # function that recenter the spots
            crop_image = image[y_pos-(crop_size):y_pos+(crop_size+1), x_pos-(crop_size):x_pos+(crop_size+1)]
            return crop_image
        # Section that marks particles if a numpy array with spot positions is passed.
        def intensity_from_position_movement(particle_index , frames_part ,time_points, number_channels ):
            intensities_mean = np.zeros((time_points, number_channels))
            intensities_std = np.zeros((time_points, number_channels))
            intensities_snr = np.zeros((time_points, number_channels))
            intensities_background_mean = np.zeros((time_points, number_channels))
            intensities_background_std = np.zeros((time_points, number_channels))
            for j in range(0, frames_part):
                for i in range(0, number_channels):
                    x_pos = self.spot_positions_movement[j,particle_index, 1]
                    y_pos = self.spot_positions_movement[j,particle_index, 0]
                    if self.method == 'disk_donut':
                        crop_image = return_crop(self.video[j, :, :, i], x_pos, y_pos, self.crop_size) # NOT RE-CENTERING IMAGE
                        intensities_mean[j, i], intensities_std[j, i] = disk_donut(crop_image,self.disk_size)
                        intensities_snr[j, i]  , intensities_background_mean [j, i], intensities_background_std [j, i] = signal_to_noise_ratio(crop_image,self.disk_size) # SNR
                    elif self.method == 'total_intensity':
                        crop_image = return_crop(self.video[j, :, :, i], x_pos, y_pos, self.crop_size) # NOT RE-CENTERING IMAGE
                        intensities_mean[j, i] = np.amax((0, np.mean(crop_image)))# mean intensity in the crop
                        intensities_std[ j, i] = np.amax((0, np.std(crop_image)))# std intensity in the crop
                        intensities_snr[j, i] , intensities_background_mean [j, i], intensities_background_std [j, i] = signal_to_noise_ratio(crop_image,self.disk_size) # SNR
                    elif self.method == 'gaussian_fit':
                        crop_image = return_crop(self.video[j, :, :, i], x_pos, y_pos, self.crop_size) # NOT RE-CENTERING IMAGE
                        intensities_snr[j, i] , intensities_background_mean [j, i], intensities_background_std [j, i] = signal_to_noise_ratio(crop_image,self.disk_size) # SNR
                        particle_half_size = int(self.particle_size/2)
                        crop_image_gaussian = return_crop(self.video[j, :, :, i], x_pos, y_pos, particle_half_size) # NOT RE-CENTERING IMAGE
                        intensities_mean[j, i], intensities_std[j, i] = gaussian_fit(crop_image_gaussian)# disk_donut(crop_image, self.disk_size
            return intensities_mean, intensities_std, intensities_snr, intensities_background_mean, intensities_background_std
        def intensity_from_dataframe(particle_index ,time_points, number_channels ):
            frames_part = self.trackpy_dataframe.loc[self.trackpy_dataframe['particle'] == self.trackpy_dataframe['particle'].unique()[particle_index]].frame.values
            intensities_mean = np.zeros((time_points, number_channels))
            intensities_std = np.zeros((time_points, number_channels))
            intensities_snr = np.zeros((time_points, number_channels))
            intensities_background_mean = np.zeros((time_points, number_channels))
            intensities_background_std = np.zeros((time_points, number_channels))
            for j in range(0, len(frames_part)):
                for i in range(0, number_channels):
                    current_frame = frames_part[j]
                    try:
                        x_pos = int(self.trackpy_dataframe.loc[self.trackpy_dataframe['particle'] == self.trackpy_dataframe['particle'].unique()[particle_index]].x.values[j])
                        y_pos = int(self.trackpy_dataframe.loc[self.trackpy_dataframe['particle'] == self.trackpy_dataframe['particle'].unique()[particle_index]].y.values[j])
                    except:
                        x_pos = int(self.trackpy_dataframe.loc[self.trackpy_dataframe['particle'] == self.trackpy_dataframe['particle'].unique()[particle_index]].x.values[frames_part[0]])
                        y_pos = int(self.trackpy_dataframe.loc[self.trackpy_dataframe['particle'] == self.trackpy_dataframe['particle'].unique()[particle_index]].y.values[frames_part[0]])
                    if self.method == 'disk_donut':
                        crop_image = return_crop(self.video[j, :, :, i], x_pos, y_pos, self.crop_size) # NOT RECENTERING IMAGE
                        intensities_mean[current_frame, i], intensities_std[current_frame, i] = disk_donut(crop_image, self.disk_size)
                        intensities_snr[current_frame, i] , intensities_background_mean [current_frame, i], intensities_background_std [current_frame, i] = signal_to_noise_ratio(crop_image,self.disk_size) # SNR
                        if np.isnan(intensities_mean[current_frame, i]) ==1:
                            print(crop_image)
                    elif self.method == 'total_intensity':
                        crop_image = return_crop(self.video[j, :, :, i], x_pos, y_pos, self.crop_size) # NOT RECENTERING IMAGE
                        intensities_mean[ current_frame, i] = np.amax((0, np.mean(crop_image))) # mean intensity in image
                        intensities_std[ current_frame, i] = np.amax((0, np.std(crop_image))) # std intensity in image
                        intensities_snr[current_frame, i] , intensities_background_mean [current_frame, i], intensities_background_std [current_frame, i] = signal_to_noise_ratio(crop_image,self.disk_size) # SNR
                    elif self.method == 'gaussian_fit':
                        crop_image = return_crop(self.video[j, :, :, i], x_pos, y_pos, self.crop_size) # NOT RECENTERING IMAGE
                        intensities_snr[current_frame, i] , intensities_background_mean [current_frame, i], intensities_background_std [current_frame, i] = signal_to_noise_ratio(crop_image,self.disk_size) # SNR
                        particle_half_size = int(self.particle_size/2)
                        crop_image_gaussian = return_crop(self.video[j, :, :, i], x_pos, y_pos, particle_half_size) # NOT RECENTERING IMAGE
                        intensities_mean[current_frame, i], intensities_std[ current_frame, i] = gaussian_fit(crop_image_gaussian)# disk_donut(crop_image, disk_size)
            intensities_mean[np.isnan(intensities_mean)] = 0 # replacing nans with zeros
            intensities_std[np.isnan(intensities_std)] = 0 # replacing nans with zeros
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
        array_mean_intensities_normalized = np.zeros_like(array_intensities_mean)*nan
        for k in range (0, self.n_particles):
                for i in range(0, number_channels):
                    if np.nanmax( array_intensities_mean[k, :, i]) > 0:
                        array_mean_intensities_normalized[k, :, i] = array_intensities_mean[k, :, i]/ np.nanmax( array_intensities_mean[k, :, i])
        mean_intensities_normalized = np.nanmean(array_mean_intensities_normalized, axis = 0, dtype = np.float32)
        mean_intensities_normalized = np.nan_to_num(mean_intensities_normalized)
        std_intensities_normalized = np.nanstd(array_mean_intensities_normalized, axis = 0, dtype = np.float32)
        std_intensities_normalized = np.nan_to_num(std_intensities_normalized)
        time_vector = np.arange(0, time_points, 1)*self.step_size
        if self.show_plot == 1:
            _, ax = plt.subplots(3, 1, figsize = (16, 4))
            for id in range (0, self.n_particles):
                frames_part = self.trackpy_dataframe.loc[self.trackpy_dataframe['particle'] == self.trackpy_dataframe['particle'].unique()[id]].frame.values
                ax[0].plot(self.trackpy_dataframe.loc[self.trackpy_dataframe['particle'] == self.trackpy_dataframe['particle'].unique()[id]].frame.values, array_intensities_mean[id, frames_part, 0], 'r')
                ax[1].plot(self.trackpy_dataframe.loc[self.trackpy_dataframe['particle'] == self.trackpy_dataframe['particle'].unique()[id]].frame.values, array_intensities_mean[id, frames_part, 1], 'g')
                ax[2].plot(self.trackpy_dataframe.loc[self.trackpy_dataframe['particle'] == self.trackpy_dataframe['particle'].unique()[id]].frame.values, array_intensities_mean[id, frames_part, 2], 'b')
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
            for id in range (0, self.n_particles):
                time_vector = np.arange(0, time_points, 1)*self.step_size
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
            for id in range (0, self.n_particles):
                time_vector = np.arange(0, time_points, 1)*self.step_size
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
            'y': [],
            'SNR_red':[],
            'SNR_green':[],
            'SNR_blue':[],
            'background_int_mean_red':[],
            'background_int_mean_green':[],
            'background_int_mean_blue':[],
            'background_int_std_red':[],
            'background_int_std_green':[],
            'background_int_std_blue':[] }
        dataframe_particles = pd.DataFrame(init_dataFrame)
        # Iterate for each spot and save time courses in the data frame
        counter = 0
        for id in range (0, self.n_particles):
            # Loop that populates the dataframes
            if not ( self.trackpy_dataframe is None):
                temporal_frames_vector = np.around(self.trackpy_dataframe.loc[self.trackpy_dataframe['particle'] == self.trackpy_dataframe['particle'].unique()[id]].frame.values)  # time_(sec)
                temporal_x_position_vector = self.trackpy_dataframe.loc[self.trackpy_dataframe['particle'] == self.trackpy_dataframe['particle'].unique()[id]].x.values
                temporal_y_position_vector = self.trackpy_dataframe.loc[self.trackpy_dataframe['particle'] == self.trackpy_dataframe['particle'].unique()[id]].y.values
            else:
                counter_time_vector = np.arange(0, time_points, 1)
                temporal_frames_vector = counter_time_vector
                temporal_x_position_vector = self.spot_positions_movement[:, id, 1]
                temporal_y_position_vector = self.spot_positions_movement[:, id, 0]
            
            temporal_red_vector =  array_intensities_mean[id, temporal_frames_vector, 0]  # red
            temporal_green_vector = array_intensities_mean[id, temporal_frames_vector, 1]  # green
            temporal_blue_vector =  array_intensities_mean[id, temporal_frames_vector, 2]  # blue
            temporal_red_vector_std =  array_intensities_std[id, temporal_frames_vector, 0]  # red
            temporal_green_vector_std =  array_intensities_std[id, temporal_frames_vector, 1]  # green
            temporal_blue_vector_std =  array_intensities_std[id, temporal_frames_vector, 2]  # blue
            temporal_spot_number_vector = [counter] * len(temporal_frames_vector)
            temporal_cell_number_vector = [0] * len(temporal_frames_vector)
            temporal_SNR_red =  array_intensities_snr[id, temporal_frames_vector, 0] # red
            temporal_SNR_green =  array_intensities_snr[id, temporal_frames_vector, 1]  # green
            temporal_SNR_blue =  array_intensities_snr[id, temporal_frames_vector, 2]  # blue
            temporal_background_int_mean_red  = array_intensities_background_mean [id, temporal_frames_vector, 0]  # red
            temporal_background_int_mean_green = array_intensities_background_mean [id, temporal_frames_vector, 1]  # green
            temporal_background_int_mean_blue=  array_intensities_background_mean [id, temporal_frames_vector, 2]  # blue
            temporal_background_int_std_red  = array_intensities_background_std[id, temporal_frames_vector, 0]  # red
            temporal_background_int_std_green = array_intensities_background_std[id, temporal_frames_vector, 1]  # green
            temporal_background_int_std_blue = array_intensities_background_std[id, temporal_frames_vector, 2]  # blue
            # Section that append the information for each spots
            temp_data_frame = {'cell_number': temporal_cell_number_vector, 
                'particle': temporal_spot_number_vector, 
                'frame': temporal_frames_vector*self.step_size, 
                'red_int_mean': np.round( temporal_red_vector ,2), 
                'green_int_mean': np.round( temporal_green_vector ,2), 
                'blue_int_mean': np.round( temporal_blue_vector ,2), 
                'red_int_std': np.round( temporal_red_vector_std ,2), 
                'green_int_std': np.round( temporal_green_vector_std ,2), 
                'blue_int_std': np.round( temporal_blue_vector_std, 2), 
                'x': temporal_x_position_vector, 
                'y': temporal_y_position_vector,
                'SNR_red' : np.round( temporal_SNR_red ,2),
                'SNR_green': np.round( temporal_SNR_green ,2),
                'SNR_blue': np.round( temporal_SNR_blue ,2),
                'background_int_mean_red': np.round( temporal_background_int_mean_red ,2),
                'background_int_mean_green': np.round( temporal_background_int_mean_green ,2),
                'background_int_mean_blue': np.round( temporal_background_int_mean_blue ,2),
                'background_int_std_red': np.round( temporal_background_int_std_red ,2),
                'background_int_std_green': np.round( temporal_background_int_std_green ,2),
                'background_int_std_blue': np.round( temporal_background_int_std_blue ,2) }
            counter += 1
            temp_DataFrame = pd.DataFrame(temp_data_frame)
            dataframe_particles = dataframe_particles.append(temp_DataFrame, ignore_index = True)
            dataframe_particles = dataframe_particles.astype({"cell_number": int, "particle": int, "frame": int, "x": int, "y": int}) # specify data type as integer for some columns
        def reduce_dataframe(df):
            # This function is intended to reduce the columns that are not used in the ML process.
            return df.drop(['red_int_std', 'green_int_std','blue_int_std','SNR_red', 'SNR_green', 'SNR_blue','background_int_mean_red','background_int_mean_green','background_int_mean_blue','background_int_std_red','background_int_std_green','background_int_std_blue'], axis = 1)
        if self.dataframe_format == 'short':
            dataframe_particles = reduce_dataframe(dataframe_particles)
            #dataframe_particles = dataframe_particles.astype({"red_int_mean": int, "green_int_mean": int, "blue_int_mean": int, "x": int, "y": int}) 
        return dataframe_particles, array_intensities_mean, time_vector, mean_intensities, std_intensities, mean_intensities_normalized, std_intensities_normalized


class SimulateRNA():
    
    '''
    This class simulates RNA intensity.
    
    Parameters

    shape_output_array: tuple
        Desired shape of the array, a tuple with two elements where the first element is the number of trajectories and the second element represents the number of time points.
    rna_intensity_method : str, optional.
        Method to generate intensity. The options are 'constant' and 'random_values'. The default is 'constant'.
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
        if self.rna_intensity_method =='random_values':
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
    diffusion_coefficient : float, optional
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
    save_as_tif_uint8 : bool, optional
        If true, it generates and saves a uint8 (low) quality image tif file for the simulation. The default is 0.
    save_as_tif : bool, optional
        If true, it generates and saves a uint16 (High) quality image tif file for the simulation. The default is 0.
    save_as_gif : bool, optional
        If true, it generates and saves a gif animation for the simulation. The default is 0.
    save_dataframe : bool, optional
        If true, it generates and saves a pandas dataframe with the simulation. Dataframe with fields [cell_number, particle, frame, red_int_mean, green_int_mean, blue_int_mean, red_int_std, green_int_std, blue_int_std, x, y]. The default is 0.
    saved_file_name : str, optional
        The file name for the simulated cell output files (tif images, gif images, data frames). The default is 'temp'.
    create_temp_folder : bool, optional
        Creates a folder with the simulation output. The default is True.
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
        "long" format generates this dataframe: [cell_number, particle, frame, red_int_mean, green_int_mean, blue_int_mean, red_int_std, green_int_std, blue_int_std, x, y, SNR_red,SNR_green,SNR_blue].
        "short" format generates this dataframe: [cell_number, particle, frame, red_int_mean, green_int_mean, blue_int_mean, x, y].
    
    '''
    def __init__(self, base_video:np.ndarray,
                video_for_mask:Union[np.ndarray, None] = None,  
                mask_image:Union[np.ndarray, None] = None, 
                number_spots:int = 10, 
                number_frames:int = 20, 
                step_size:float = 1, 
                diffusion_coefficient:float = 0.01, 
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
                save_as_tif_uint8: bool = False, 
                save_as_tif: bool = False, 
                save_as_gif: bool = False, 
                save_dataframe: bool = False, 
                saved_file_name :str = 'temp', 
                create_temp_folder: bool = True, 
                intensity_calculation_method :str = 'disk_donut', 
                perform_video_augmentation: bool = 0, 
                frame_selection_empty_video:str = 'shuffle',
                ignore_trajectories_ch0:bool =False, 
                ignore_trajectories_ch1:bool =False, 
                ignore_trajectories_ch2:bool =False,
                intensity_scale_ch0:float = 1,
                intensity_scale_ch1:float = 1,
                intensity_scale_ch2:float = 1,
                dataframe_format:str = 'short' ):
        if (perform_video_augmentation == 1) and (video_for_mask is None):
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
            video_for_mask = RemoveExtrema(video_for_mask, min_percentile = 0, max_percentile = 99.8).remove_outliers()
            self.video_for_mask = video_for_mask
        else:
            self.video_for_mask = self.base_video
        self.number_spots = number_spots
        self.number_frames = number_frames
        self.step_size = step_size
        self.diffusion_coefficient = diffusion_coefficient
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
        self.save_as_tif_uint8 = save_as_tif_uint8
        self.save_as_tif = save_as_tif
        self.save_as_gif = save_as_gif
        self.save_dataframe = save_dataframe
        self.saved_file_name = saved_file_name
        self.create_temp_folder = create_temp_folder
        self.frame_selection_empty_video = frame_selection_empty_video
        self.dataframe_format =dataframe_format
        # The following two constants are weights used to define a range of intensities for the simulated spots.
        self.MAX_STD_INT_IMAGE = 5 # maximum number of standard deviations above the mean that are allowed to draw an spot.
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
            if np.amax(selected_masks) == 0:
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
            Dataframe with fields [cell_number, particle, frame, red_int_mean, green_int_mean, blue_int_mean, red_int_std, green_int_std, blue_int_std, x, y, SNR_red,SNR_green,SNR_blue].
        '''
        def make_replacement_pixelated_spots(matrix_background:np.ndarray, center_positions_vector:np.ndarray, size_spot:int, spot_sigma:int, using_ssa:bool, simulated_trajectories_time_point:np.ndarray,intensity_scale:float):
            #This function is intended to replace a kernel gaussian matrix for each spot position. The kernel gaussian matrix is scaled with the values obtained from the SSA o with the values given in a range.
            if size_spot%2 == 0:
                print('The size of the spot must be an odd number')
                raise
            # Copy the matrix_background
            pixelated_image = matrix_background.copy()
            for point_index in range(0, len(center_positions_vector)):
                # Section that creates the Gaussian Kernel Matrix
                ax = np.linspace(-(size_spot - 1) / 2., (size_spot - 1) / 2., size_spot)
                xx, yy = np.meshgrid(ax, ax)
                kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(spot_sigma))
                # creating a position for each spot
                center_position = center_positions_vector[point_index]
                if using_ssa == 1 :
                    int_ssa = simulated_trajectories_time_point[point_index] #- min_SSA_value) / (max_SSA_value-min_SSA_value) # intensity normalized to min and max values in the SSA
                    int_tp = int_ssa* intensity_scale 
                    spot_intensity = np.amax((0, int_tp))
                else:
                    spot_intensity = intensity_scale
                kernel_value_intensity = (kernel*spot_intensity)
                selected_area = pixelated_image[center_position[0]-int(size_spot/2): center_position[0]+int(size_spot/2)+1 , center_position[1]-int(size_spot/2): center_position[1]+int(size_spot/2)+1 ]
                selected_area = selected_area+ kernel_value_intensity
                pixelated_image[center_position[0]-int(size_spot/2): center_position[0]+int(size_spot/2)+1 , center_position[1]-int(size_spot/2): center_position[1]+int(size_spot/2)+1 ] = selected_area
            return pixelated_image # final_image

        def make_spots_movement(polygon_array, number_spots:int, time_vector:np.ndarray, step_size: float, image_size:np.ndarray, diffusion_coefficient:float, internal_base_video:Union[np.ndarray, None] = None):
            # Function that creates the simulated spots inside a given polygon
            path = mpltPath.Path(polygon_array)
            initial_points_in_polygon = np.zeros((number_spots, 2), dtype = 'int')
            counter_number_spots = 0
            conter_security = 0
            MAX_ITERATIONS = 5000
            NUMBER_SPACIAL_DIMENSSIONS_IN_SIMULATION = 2
            min_position = 20 # minimal position in pixels
            max_position = image_size[1]-20 # maximal position in pixels
            while (counter_number_spots < number_spots) and (conter_security < MAX_ITERATIONS):
                test_points = (int(random.uniform(min_position, max_position)), int(random.uniform(min_position, max_position)))
                # testing if the spot is located in an area of high intensity?
                if not ( internal_base_video is None):
                    selected_image = internal_base_video
                    max_allowed_int_image = np.mean(selected_image) + self.MAX_STD_INT_IMAGE*np.std(selected_image)
                    min_allowed_int_image = np.mean(selected_image) - 0.5*np.std(selected_image)
                    pixel_size_around_spot = 10
                    temp_crop_around_spot = selected_image[test_points[0]-int(pixel_size_around_spot/2): test_points[0]+int(pixel_size_around_spot/2)+1 , test_points[1]-int(pixel_size_around_spot/2): test_points[1]+int(pixel_size_around_spot/2)+1 ]
                    mean_int_tested_spot = np.mean(temp_crop_around_spot)
                    if (mean_int_tested_spot > min_allowed_int_image) and (mean_int_tested_spot < max_allowed_int_image):
                        int_test = 1
                    else:
                        int_test = 0
                else:
                    int_test = 1
                conter_security += 1
                if path.contains_point(test_points) == 1 and int_test == 1:
                    counter_number_spots += 1
                    initial_points_in_polygon[counter_number_spots-1, :] = np.asarray(test_points)
                if conter_security > MAX_ITERATIONS:
                    print('error generating spots')
            ## Brownian motion
            # scaling factor for Brownian motion.
            brownian_movement = math.sqrt(2*NUMBER_SPACIAL_DIMENSSIONS_IN_SIMULATION*diffusion_coefficient*step_size)
            # Preallocating memory
            y_positions = np.array(initial_points_in_polygon[:, 0], dtype = 'int') #  x_position for selected spots inside the polygon
            x_positions = np.array(initial_points_in_polygon[:, 1], dtype = 'int') #  y_position for selected spots inside the polygon
            temp_Position_y = np.zeros_like(y_positions, dtype = 'int')
            temp_Position_x = np.zeros_like(x_positions, dtype = 'int')
            newPosition_y = np.zeros_like(y_positions, dtype = 'int')
            newPosition_x = np.zeros_like(x_positions, dtype = 'int')
            spot_positions_movement = np.zeros((len(time_vector), number_spots, 2), dtype = 'int')
            # Main loop that computes the random motion and new spot positions
            for t_p, _ in enumerate(time_vector):
                for i_p in range (0, number_spots):
                    if t_p == 0:
                        temp_Position_y[i_p] = y_positions[i_p]
                        temp_Position_x[i_p] = x_positions[i_p]
                    else:
                        temp_Position_y[i_p] = newPosition_y[i_p] + int(brownian_movement * np.random.randn(1))
                        temp_Position_x[i_p] = newPosition_x[i_p] + int(brownian_movement * np.random.randn(1))
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
                # test if num_requested_frames >  frames_in_orginal_video
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
                        reverse_proportion = np.amax( (0, 1-proportion_to_interpolate))
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
            if frame_selection_empty_video ==  'generate_from_poisson': # selects the first time point
                generated_video = generate_poisson_video(original_video=base_video_selected_channel.copy(), num_requested_frames=len(time_vector))
                index_frame_selection = range(0, len(time_vector))
                base_video_selected_channel_copy = generated_video
            if frame_selection_empty_video ==  'generate_from_gaussian': # selects the first time point
                generated_video = generate_gaussian_video(original_video=base_video_selected_channel.copy(), num_requested_frames=len(time_vector))
                index_frame_selection = range(0, len(time_vector))
                base_video_selected_channel_copy = generated_video
            if frame_selection_empty_video ==  'constant': # selects the first time point
                index_frame_selection = np.zeros((len(time_vector)), dtype = np.int32)
            if frame_selection_empty_video ==  'loop':
                index_frame_selection = np.resize(empty_video_index, len(time_vector))
            if frame_selection_empty_video ==  'shuffle':
                index_frame_selection = np.random.randint(0, high = len_empty_video, size = len(time_vector), dtype = np.int32)
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
        spot_positions_movement = make_spots_movement(self.polygon_array, self.number_spots, self.time_vector, self.step_size, self.image_size, self.diffusion_coefficient, self.base_video[0, :, :, 1])
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
        # This section saves dataframes and simulated images
        if (self.save_as_tif_uint8 == 1) or (self.save_as_gif == 1):
            if self.create_temp_folder == True:
                save_to_path = pathlib.Path().absolute().joinpath('temp')
                if not os.path.exists(str(save_to_path)):
                    os.makedirs(str(save_to_path))
                print ("The output is saved in the directory: " , str(save_to_path))
            else:
                save_to_path = pathlib.Path().absolute()
            tensor_video_copy = tensor_video.copy()
            normalized_tensor = np.zeros_like(tensor_video_copy, dtype = 'uint8')
            num_images_for_gif = tensor_video_copy.shape[0]
            for ch in range(0, tensor_video_copy.shape[3]):
                for i in range(0, num_images_for_gif):
                    # CONVERSION TO UINT
                    image = tensor_video_copy[i, :, :, ch].copy()
                    min_image, max_image = np.min(image), np.max(image)
                    image -= min_image
                    image_float = np.array(image, 'float32')
                    image_float *= 255./(max_image-min_image)
                    image = np.asarray(np.round(image_float), 'uint8')
                    normalized_tensor[i, :, :, ch] = image
            if self.save_as_tif_uint8 == 1:
                tifffile.imwrite(str(save_to_path.joinpath(self.saved_file_name+'_unit8_'+'.tif')), normalized_tensor)
            if self.save_as_gif == 1:
                # Saving the simulation as a gif. Complete image
                with imageio.get_writer(str(save_to_path.joinpath(self.saved_file_name+'_unit8_'+'.gif')), mode = 'I') as writer:
                    for i in range(0, num_images_for_gif):
                        image = normalized_tensor[i, :, :, 0:1]
                        writer.append_data(image)
        if self.save_as_tif == True:
            if self.create_temp_folder == True:
                save_to_path = pathlib.Path().absolute().joinpath('temp')
                if not os.path.exists(str(save_to_path)):
                    os.makedirs(str(save_to_path))
                print ("The output is saved in the directory: " , str(save_to_path))
            else:
                save_to_path = pathlib.Path().absolute()
            tifffile.imwrite(str(save_to_path.joinpath(self.saved_file_name+'.tif')), tensor_video)
        
        dataframe_particles, _, _, _, _, _, _ = Intensity(tensor_video, particle_size = self.size_spot_ch0, spot_positions_movement = spot_positions_movement, method = self.intensity_calculation_method, step_size = self.step_size, show_plot = 0,dataframe_format =self.dataframe_format ).calculate_intensity()
        if self.save_dataframe == 1:
            if self.create_temp_folder == True:
                save_to_path = pathlib.Path().absolute().joinpath('temp')
                if not os.path.exists(str(save_to_path)):
                    os.makedirs(str(save_to_path))
                print ("The output is saved in the directory: " , str(save_to_path))
            else:
                save_to_path = pathlib.Path().absolute()
            dataframe_particles.to_csv(str(save_to_path.joinpath(self.saved_file_name +'_df'+ '.csv')), index = True)
            
        return tensor_video , spot_positions_movement, dataframe_particles


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
    save_as_tif : bool, optional
        If true, it generates and saves a uint16 (High) quality image tif file for the simulation. The default is 0.
    save_dataframe : bool, optional
        If true, it generates and saves a pandas dataframe with the simulation. Dataframe with fields [cell_number, particle, frame, red_int_mean, green_int_mean, blue_int_mean, red_int_std, green_int_std, blue_int_std, x, y]. The default is 0.
    saved_file_name : str, optional
        The file name for the simulated cell output files (tif images, gif images, data frames). The default is 'temp'.
    create_temp_folder : bool, optional
        Creates a folder with the simulation output. The default is True.
    mask_image : NumPy array, optional    
        Numpy Array with dimensions [Y, X]. This image is used as a mask for the simulated video. The mask_image has to represent the same image as the base_video and video_for_mask.
    cell_number : int, optional
        Cell number used as an index for the data frame. The default is 0.
    save_as_gif : bool, optional
        If true, it generates and saves a .gif with the simulation. The default is 0.
    perform_video_augmentation : bool, optional
        If true, it performs random rotations the initial video. The default is 1.
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
        Method used to simulate RNA intensities in the image. The optiions are 'constant' or 'random_values'. The default is 'constant'
    dataframe_format : str, optional
        Format for the dataframe the options are : 'short' , and 'long'. The default is 'short'.
        "long" format generates this dataframe: [cell_number, particle, frame, red_int_mean, green_int_mean, blue_int_mean, red_int_std, green_int_std, blue_int_std, x, y, SNR_red,SNR_green,SNR_blue].
        "short" format generates this dataframe: [cell_number, particle, frame, red_int_mean, green_int_mean, blue_int_mean, x, y].
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
                save_as_tif:bool=False, 
                save_dataframe:bool=False, 
                saved_file_name:str = 'temp', 
                create_temp_folder:bool = False, 
                mask_image:Union[np.ndarray, None] = None, 
                cell_number:int = 0, 
                save_as_gif:bool = False, 
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
                basal_intensity_in_background_video : int = 20000):
        if perform_video_augmentation == True:
            preprocessed_base_video,selected_angle = AugmentationVideo(initial_video).random_rotation()
            if not(mask_image is None):
                self.mask_image,selected_angle = AugmentationVideo(mask_image,selected_angle).random_rotation()
            else:
                self.mask_image = mask_image    
        else:
            preprocessed_base_video  = initial_video
            self.mask_image = mask_image
            
        preprocessed_base_video = RemoveExtrema(preprocessed_base_video, min_percentile = 0, max_percentile = 99).remove_outliers()
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
        self.save_as_tif = save_as_tif
        self.save_dataframe = save_dataframe
        self.saved_file_name = saved_file_name
        self.create_temp_folder = create_temp_folder
        self.cell_number = cell_number
        self.save_as_gif = save_as_gif
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
    def make_simulation (self):
        '''
        Method that runs the simulations for the multiplexing experiment.

        Returns

        tensor_video : NumPy array uint16
            Array with dimensions [T, Y, X, C]
        dataframe_particles : pandas dataframe
            Dataframe with fields [cell_number, particle, frame, red_int_mean, green_int_mean, blue_int_mean, red_int_std, green_int_std, blue_int_std, x, y, SNR_red,SNR_green,SNR_blue].
        list_ssa : List of NumPy arrays
            List of numpy arrays with the stochastic simulations for each gene. The format is [S, T], where the dimensions are S = spots and T = time.
        '''

        # Wrapper for the simulated cell
        def wrapper_simulated_cell (base_video, video_for_mask = None, ssa_protein = None, rna_intensity=None, target_channel_protein = 1,target_channel_mRNA =0,  diffusion_coefficient = 0.05, step_size = self.step_size_in_sec, spot_size = self.spot_size, intensity_calculation_method = 'disk_donut', save_as_gif = 0, frame_selection_empty_video = self.frame_selection_empty_video,int_scale_to_snr=np.array([100,100,100]) ):
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
            
            # Running simulated cell
            tensor_video, _,DataFrame_particles_intensities = SimulatedCell( base_video = base_video, 
                                                                            video_for_mask = video_for_mask, 
                                                                            mask_image=self.mask_image, 
                                                                            number_spots = number_spots_per_cell, 
                                                                            number_frames = ssa_protein.shape[1], 
                                                                            step_size = step_size, 
                                                                            diffusion_coefficient = diffusion_coefficient, 
                                                                            simulated_trajectories_ch0 = simulated_trajectories_ch0, 
                                                                            size_spot_ch0 = spot_size, 
                                                                            spot_sigma_ch0 = self.spot_sigma, 
                                                                            simulated_trajectories_ch1 = simulated_trajectories_ch1, 
                                                                            size_spot_ch1 = spot_size, 
                                                                            spot_sigma_ch1 = self.spot_sigma, 
                                                                            simulated_trajectories_ch2 = simulated_trajectories_ch2, 
                                                                            size_spot_ch2 = spot_size, 
                                                                            spot_sigma_ch2 = self.spot_sigma, 
                                                                            save_as_tif_uint8 = 0, 
                                                                            save_as_tif = 0, 
                                                                            save_as_gif = save_as_gif, 
                                                                            save_dataframe = 0, 
                                                                            create_temp_folder = 0, 
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
                                                                            ignore_ch2 = self.ignore_ch2).make_simulation()
            
            DataFrame_particles_intensities['cell_number'] = DataFrame_particles_intensities['cell_number'].replace([0], self.cell_number)
            return tensor_video, DataFrame_particles_intensities  # [cell_number, particle, frame, red_int_mean, green_int_mean, blue_int_mean, red_int_std, green_int_std, blue_int_std, x, y].
        # Runs the SSA and the simulated cell functions
        list_ssa = []
        list_min_ssa = []
        list_max_ssa = []
        RNA_INTENSITY_MAX_VALUE =10 # this variable defines a value of units of RNA
        for i in range(0, self.number_genes):
            # # Simulations for intensity
            _,ssa_ump,_,_ = SSA_rsnapsim( gene_file = self.list_gene_sequences[i], 
                                        ke = self.list_elongation_rates[i],
                                        ki = self.list_initiation_rates[i],
                                        frames = self.simulation_time_in_sec,
                                        frame_rate = 1,
                                        n_traj = self.list_number_spots[i]).simulate() 
                        
            simulated_trajectories_RNA= SimulateRNA(shape_output_array=(self.list_number_spots[i], self.simulation_time_in_sec), 
                                                                            rna_intensity_method = self.simulated_RNA_intensities_method,
                                                                            mean_int=RNA_INTENSITY_MAX_VALUE ).simulate()
            # appending simulated data
            list_ssa.append(ssa_ump)
            list_min_ssa.append(ssa_ump.min())
            list_max_ssa.append(ssa_ump.max())
        
        ####
        vector_int_scales  = np.array ([self.intensity_scale_ch0,self.intensity_scale_ch1, self.intensity_scale_ch2])
        ############
        
        # Calculating the estimated elongation rates based on parameter values
        calculated_mean_int_in_ssa = np.zeros(len(self.list_gene_sequences))+0.001
        for g in range(len(self.list_gene_sequences)):
            _, _,tagged_pois,raw_seq = rss.seqmanip.open_seq_file(str(self.list_gene_sequences[g]))
            gene_len = len(raw_seq)
            gene_obj = tagged_pois['1'][0]
            number_probes = np.amax(gene_obj.probe_vec)
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
                                                                                        save_as_gif = self.save_as_gif, 
                                                                                        frame_selection_empty_video = self.frame_selection_empty_video,
                                                                                        int_scale_to_snr=int_scale_to_snr)
            else:
                tensor_video , DataFrame_particles_intensities = wrapper_simulated_cell(tensor_video, 
                                                                                        video_for_mask = self.initial_video, 
                                                                                        ssa_protein = list_ssa[i],
                                                                                        rna_intensity =  simulated_trajectories_RNA,
                                                                                        target_channel_protein = self.list_target_channels_proteins[i], 
                                                                                        target_channel_mRNA = self.list_target_channels_mRNA[i] , 
                                                                                        diffusion_coefficient = self.list_diffusion_coefficients[i], 
                                                                                        save_as_gif = self.save_as_gif, 
                                                                                        frame_selection_empty_video = 'loop',
                                                                                        int_scale_to_snr=int_scale_to_snr) # notice that for the multiplexing frame_selection_empty_video has to be 'loop', because the initial video deffines the initial background image.

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
        # saving the simulated video and data frame
        if self.save_as_tif == 1:
            if self.create_temp_folder == True:
                save_to_path = pathlib.Path().absolute().joinpath('temp')
                if not os.path.exists(str(save_to_path)):
                    os.makedirs(str(save_to_path))
                print ("The output is saved in the directory: " , str(save_to_path))
            else:
                save_to_path = pathlib.Path().absolute()
            tifffile.imwrite(str(save_to_path.joinpath(self.saved_file_name+'.tif')), tensor_video)
        if self.save_dataframe == 1:
            if self.create_temp_folder == True:
                save_to_path = pathlib.Path().absolute().joinpath('temp')
                if not os.path.exists(str(save_to_path)):
                    os.makedirs(str(save_to_path))
                print ("The output is saved in the directory: " , str(save_to_path))
            else:
                save_to_path = pathlib.Path().absolute()
            dataframe_simulated_cell.to_csv(str(save_to_path.joinpath(self.saved_file_name +'_df'+ '.csv')), index = True)
        return tensor_video, dataframe_simulated_cell, list_ssa


class PipelineTracking():
    '''
    A pipeline that allows cell segmentation, spot detection, and tracking of spots.

    Parameters

    video : NumPy array
        Array of images with dimensions [T, Y, X, C].
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
        "long" format generates this dataframe: [cell_number, particle, frame, red_int_mean, green_int_mean, blue_int_mean, red_int_std, green_int_std, blue_int_std, x, y, SNR_red,SNR_green,SNR_blue].
        "short" format generates this dataframe: [cell_number, particle, frame, red_int_mean, green_int_mean, blue_int_mean, x, y].

    '''
    def __init__(self, video:np.ndarray, particle_size:int = 5, file_name:str = 'Cell.tif', selected_channel_tracking:int = 0,selected_channel_segmentation:int = 0,  intensity_calculation_method:str = 'disk_donut', mask_selection_method:str = 'max_spots', show_plot:bool = 1, use_optimization_for_tracking: bool = 1, real_positions_dataframe = None, average_cell_diameter: float = 120, print_process_times:bool = 0,min_percentage_time_tracking=0.4,intensity_threshold_tracking=None,dataframe_format='short'):
        self.video = video
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
        # Iterations
        self.NUM_ITERATIONS_CELLPOSE = 10
        self.NUM_ITERATIONS_TRACKING = 1000
        self.MIN_PERCENTAGE_FRAMES_FOR_TRACKING = min_percentage_time_tracking
        self.intensity_threshold_tracking=intensity_threshold_tracking
        self.dataframe_format=dataframe_format
    def run(self):
        '''
        Runs the pipeline.

        Returns

        dataframe_particles : pandas dataframe
            Dataframe with fields [cell_number, particle, frame, red_int_mean, green_int_mean, blue_int_mean, red_int_std, green_int_std, blue_int_std, x, y].
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
        start = timer()
        selected_masks = Cellpose(self.image, num_iterations = self.NUM_ITERATIONS_CELLPOSE, selection_method = 'max_cells_and_area', diameter = self.average_cell_diameter ).calculate_masks() # options are 'max_area' or 'max_cells'
        if not ( selected_masks is None):
            selected_mask  = CellposeSelection(selected_masks, self.video, selection_method = self.mask_selection_method, particle_size = self.particle_size, selected_channel = self.selected_channel_segmentation).select_mask()
        else:
            selected_mask = None
        end = timer()
        if self.print_process_times == 1:
            print('mask time:', round(end - start), ' sec')
        if not ( selected_mask is None):
            # Tracking
            start = timer()
            if self.num_frames > 20:
                minimal_frames =  int(self.num_frames*self.MIN_PERCENTAGE_FRAMES_FOR_TRACKING) # minimal number of frames to consider a trajectory
            else:
                minimal_frames =  int(self.num_frames*0.4) # minimal number of frames to consider a trajectory
            if self.use_optimization_for_tracking == 1:
                use_default_filter = 0
            else:
                use_default_filter = 1
            Dataframe_trajectories, _, filtered_video = Trackpy(self.video, selected_mask, particle_size = self.particle_size, selected_channel = self.selected_channel_tracking, minimal_frames = minimal_frames, optimization_iterations = self.NUM_ITERATIONS_TRACKING, use_default_filter = use_default_filter, show_plot = self.show_plot,intensity_threshold_tracking=self.intensity_threshold_tracking).perform_tracking()
            end = timer()
            if self.print_process_times == 1:
                print('tracking time:', round(end - start), ' sec')
            # Intensity calculation
            start = timer()
            if not ( Dataframe_trajectories is None):
                dataframe_particles, array_intensities, time_vector, mean_intensities, std_intensities, mean_intensities_normalized, std_intensities_normalized = Intensity(self.video, self.particle_size, Dataframe_trajectories, method = self.intensity_calculation_method, show_plot = 0, dataframe_format=self.dataframe_format).calculate_intensity()
            else:
                dataframe_particles = None
                array_intensities = None
                time_vector = None
                mean_intensities = None
                std_intensities = None
                mean_intensities_normalized = None
                std_intensities_normalized = None
            end = timer()
            if self.print_process_times == 1:
                print('intensity calculation time:', round(end - start), ' sec')
            if (self.show_plot == 1):
                VisualizerImage(self.video, filtered_video, Dataframe_trajectories, self.file_name, list_mask_array = selected_mask, selected_channel = self.selected_channel_tracking, selected_time_point = 0, normalize = False, individual_figure_size = 7, list_real_particle_positions = self.real_positions_dataframe).plot()
        else:
            dataframe_particles = None
            array_intensities = None
            time_vector = None
            mean_intensities = None
            std_intensities = None
            mean_intensities_normalized = None
            std_intensities_normalized = None
        return dataframe_particles,selected_mask, array_intensities, time_vector, mean_intensities, std_intensities, mean_intensities_normalized, std_intensities_normalized


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
        if self.show_plot == 1:
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




class Utilities():
    '''
    This class contains miscellaneous methods to perform tasks needed in multiple classes. No parameters are necessary for this class.
    '''
    def __init__(self):
        pass
    
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
    # Function to convert the video to uint
    def img_uint(image):
        temp_vid = img_as_uint(image)
        return temp_vid


    def compute_msd(trajectory):
        '''
        This function is intended to calculate the mean square displacement of a given trajectory.
        msd(τ)  = <[r(t+τ) - r(t)]^2>
        
        Parameters:
            trajectory: list of temporal evolution of a centers of mass .  [Y_val_particle_i_tp_0, X_val_particle_i_tp_0]   , ... , [Y_val_particle_i_tp_n, X_val_particle_i_tp_n] ]

        Returns:
            msd: mean square displacement
            rmsd: root mean square displacement
        '''
        total_length_trajectory=len(trajectory)
        msd=[]
        for i in range(total_length_trajectory-1):
            tau=i+1
            # Distance that a particle moves for each time point (tau) divided by time
            # msd(τ)                 = <[r(t+τ)  -    r(t)]^2>
            msd.append(np.sum((trajectory[0:-tau]-trajectory[tau::])**2)/(total_length_trajectory-tau)) # Reverse Indexing 
        # Converting list to np.array
        msd=np.array(msd)   # array with shape Nspots vs time_points
        rmsd = np.sqrt(msd)
        return msd, rmsd 


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
                    save_as_tif = False, 
                    save_dataframe = False, 
                    save_as_gif=False,
                    frame_selection_empty_video='generate_from_gaussian',
                    spot_size = 7 ,
                    spot_sigma=1,
                    intensity_scale_ch0 = None,
                    intensity_scale_ch1 = None,
                    intensity_scale_ch2 = None,
                    dataframe_format = 'long',
                    simulated_RNA_intensities_method='constant',
                    store_videos_in_memory= False,
                    scale_intensity_in_base_video =False,
                    basal_intensity_in_background_video= 20000):

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
        List where every element represents the diffusion coefficient for every gene.
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
        If true, it generates and saves a uint16 (High) quality image tif file for the simulation. The default is 0.
    save_dataframe : bool, optional
        If true, it generates and saves a pandas dataframe with the simulation. Dataframe with fields [cell_number, particle, frame, red_int_mean, green_int_mean, blue_int_mean, red_int_std, green_int_std, blue_int_std, x, y]. The default is 0.
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
        Method used to simulate RNA intensities in the image. The optiions are 'constant' or 'random_values'. The default is 'constant'
    dataframe_format : str, optional
        Format for the dataframe the options are : 'short' , and 'long'. The default is 'short'.
        "long" format generates this dataframe: [cell_number, particle, frame, red_int_mean, green_int_mean, blue_int_mean, red_int_std, green_int_std, blue_int_std, x, y, SNR_red,SNR_green,SNR_blue].
        "short" format generates this dataframe: [cell_number, particle, frame, red_int_mean, green_int_mean, blue_int_mean, x, y].
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

    Returns

    dataframe_particles : pandas dataframe
        Dataframe with fields [cell_number, particle, frame, red_int_mean, green_int_mean, blue_int_mean, red_int_std, green_int_std, blue_int_std, x, y].
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
    start = timer()
    
    # Testing if the user passed parameters as lists. If not the code conver the parameters into lists
    def test_if_list(tested_list):
        if isinstance(tested_list, list):
            return tested_list
        else:
            return [tested_list]
    list_gene_sequences = test_if_list (list_gene_sequences)
    list_number_spots = test_if_list (list_number_spots)
    list_target_channels_proteins = test_if_list (list_target_channels_proteins)
    list_target_channels_mRNA = test_if_list(list_target_channels_mRNA)
    list_diffusion_coefficients = test_if_list (list_diffusion_coefficients)
    list_elongation_rates = test_if_list(list_elongation_rates)
    list_initiation_rates = test_if_list(list_initiation_rates)
    
    if not (list_label_names is None):
        list_label_names = test_if_list(list_label_names)
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
    name_folder = '_bg_' + frame_selection_empty_video 
    name_folder+='_ke_'
    temp_list_ke = ''.join([str(list_elongation_rates[j])+'_' for j in range(len(list_gene_sequences))])
    name_folder += temp_list_ke+'ki_'
    temp_list_ki = ''.join([str(list_initiation_rates[j])+'_' for j in range(len(list_gene_sequences))])
    name_folder+= temp_list_ki+ 'kd_'
    temp_list_kd = ''.join([str(list_diffusion_coefficients[j])+'_' for j in range(len(list_gene_sequences))]) 
    name_folder+= temp_list_kd + 'num_spots_'
    temp_list_ns= ''.join([str(list_number_spots[j])+'_' for j in range(len(list_gene_sequences))])
    name_folder+= temp_list_ns + 'time_' + str(simulation_time_in_sec) + '_num_cells_' + str(number_cells)
    name_folder+='_int0_' +str(intensity_scale_ch0)+'_int1_' +str(intensity_scale_ch1)+'_int2_' +str(intensity_scale_ch2)
    name_folder = name_folder.replace(".", "_")
    metadata_filename = 'metadata'+ name_folder + '.txt'
    folder_dataframe = 'dataframe' + name_folder
    folder_video = 'videos' + name_folder
    folder_video_int_8 = 'videos_int8' + name_folder
        
    # Functions to create folder to save simulated cells
    current_dir = pathlib.Path().absolute()
    
    if save_dataframe == True:
        save_to_path_df =  current_dir.joinpath('temp' , folder_dataframe )
        if not os.path.exists(str(save_to_path_df)):
            os.makedirs(str(save_to_path_df))
        else:
            shutil.rmtree(str(save_to_path_df))
            os.makedirs(str(save_to_path_df))
    
    if save_as_tif == True:
        save_to_path_video =  current_dir.joinpath('temp' , folder_video )
        if not os.path.exists(str(save_to_path_video)):
            os.makedirs(str(save_to_path_video))
        else:
            shutil.rmtree(str(save_to_path_video))
            os.makedirs(str(save_to_path_video))
    
    if save_as_gif == True:
        save_to_path_video_int8 =  current_dir.joinpath('temp' , folder_video_int_8 )
        if not os.path.exists(str(save_to_path_video_int8)):
            os.makedirs(str(save_to_path_video_int8))
        else:
            shutil.rmtree(str(save_to_path_video_int8))
            os.makedirs(str(save_to_path_video_int8))
    
    # function  that simulates the multiplexing experiments    
    # Pre-alocating arrays
    list_dataframe_simulated_cell =[]
    list_ssa_all_cells_and_genes =[]
    list_videos = []
    list_files_names_outputs = []
    # Reading all empty cells in directory
    list_files_names = sorted([f for f in listdir(video_dir) if isfile(join(video_dir, f)) and ('.tif') in f], key=str.lower)  # reading all tif files in the folder
    list_files_names.sort(key=lambda f: int(re.sub('\D', '', f)))  # sorting the index in numerical order
    path_files = [ str(video_dir.joinpath(f).resolve()) for f in list_files_names ] # creating the complete path for each file
    num_cell_shapes = len(path_files)
    
    for i in range(0,number_cells): 
        saved_file_name = 'sim_cell_' + str(i)  # if the video or dataframe are save, this variable assigns the name to the files
        selected_video = randrange(num_cell_shapes)
        initial_video = imread(str(path_files[selected_video])) # video with empty cell
        mask_image = imread(masks_dir.joinpath('mask_cell_shape_'+str(selected_video)+'.tif'))
        video, single_dataframe_simulated_cell, list_ssa = SimulatedCellDispatcher(initial_video,
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
                                                                                    cell_number =i,
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
                                                                                    basal_intensity_in_background_video=basal_intensity_in_background_video).make_simulation()
        
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
        
        # list file names
        list_files_names_outputs.append(saved_file_name+'.tif')
    
    merged_dataframe_simulated_cells = pd.concat(list_dataframe_simulated_cell)
    ssa_trajectories = np.array(list_ssa_all_cells_and_genes)
    
    # Saving dataframes to folder
    if save_dataframe == True:
        # saving the dataframe
        merged_dataframe_simulated_cells.to_csv( save_to_path_df.joinpath('dataframe_sim_cell.csv'), float_format="%.2f")
        # saving the ssa
        np.save(save_to_path_df.joinpath('ssas_sim_cell.npy') , ssa_trajectories)
        # creating zip
        shutil.make_archive( base_name = save_to_path_df, format = 'zip', root_dir = save_to_path_df.parents[0], base_dir =save_to_path_df.name )
        #shutil.rmtree(save_to_path_df)
        print('The simulation dataframes are stored here:', str(save_to_path_df))

    # Creating metadata file
    metadata_filename= str(current_dir.joinpath('temp',metadata_filename))
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
    
    end = timer()
    print('Time to generate simulated data:',round(end - start), ' sec')
    
    return list_videos, list_dataframe_simulated_cell, merged_dataframe_simulated_cells, ssa_trajectories, list_files_names_outputs, save_to_path_video, save_to_path_df.joinpath('dataframe_sim_cell.csv')



def image_processing(files_dir_path,
                    particle_size=14,
                    selected_channel_tracking = 0,
                    selected_channel_segmentation = 0,
                    intensity_calculation_method ='disk_donut', 
                    mask_selection_method = 'max_area',
                    show_plot=False,
                    use_optimization_for_tracking=True,
                    real_positions_dataframe = None,
                    average_cell_diameter=200,
                    print_process_times=1,
                    min_percentage_time_tracking=0.3,
                    dataframe_format='long'):
    
    start = timer()
    list_DataFrame_particles_intensities= []
    list_array_intensities = []
    list_time_vector = []
    list_selected_mask = []
    
    ## Reads the folder with the results and import the simulations as lists
    list_files_names = sorted([f for f in listdir(files_dir_path) if isfile(join(files_dir_path, f)) and ('.tif') in f], key=str.lower)  # reading all tif files in the folder
    list_files_names.sort(key=lambda f: int(re.sub('\D', '', f)))  # sorting the index in numerical order
    path_files = sorted([ str(files_dir_path.joinpath(f).resolve()) for f in list_files_names ] , key=str.lower)# creating the complete path for each file
    print(path_files)
    # # Reading the microscopy data
    number_images = len(list_files_names)
        
    for i in range(0,number_images): 
        selected_video = imread(path_files[i]) # Loading the video
        DataFrame_particles_intensities, selected_mask, array_intensities, time_vector, _,_, _, _ = PipelineTracking(video=selected_video,
                                                                                                                    particle_size=particle_size,
                                                                                                                    file_name=list_files_names[i],
                                                                                                                    selected_channel_tracking=selected_channel_tracking ,
                                                                                                                    selected_channel_segmentation=selected_channel_segmentation ,
                                                                                                                    intensity_calculation_method=intensity_calculation_method , 
                                                                                                                    mask_selection_method=mask_selection_method ,
                                                                                                                    show_plot=show_plot,
                                                                                                                    use_optimization_for_tracking=use_optimization_for_tracking,
                                                                                                                    real_positions_dataframe=real_positions_dataframe[i] ,
                                                                                                                    average_cell_diameter=average_cell_diameter,
                                                                                                                    print_process_times=print_process_times,
                                                                                                                    min_percentage_time_tracking=min_percentage_time_tracking,
                                                                                                                    dataframe_format=dataframe_format).run()    
        list_DataFrame_particles_intensities.append(DataFrame_particles_intensities)
        list_array_intensities.append(array_intensities)
        list_time_vector.append(time_vector)
        list_selected_mask.append(selected_mask)
        print('Progress: ',str(i+1),'/',str(number_images))
    # PDF
    
    # Metadata 
        # 
        
    end = timer()
    print('Time to process data:',round(end - start), ' sec')
    return list_DataFrame_particles_intensities, list_array_intensities, list_time_vector, list_selected_mask

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
                frame_selection_empty_video='generate_from_gaussian',
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
                    fd.write('\n        target_channel_mrna: ' + str(self.list_target_channels_mRNA[k]) )
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
                frame_selection_empty_video='generate_from_gaussian',
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
                    fd.write('\n        target_channel_mrna: ' + str(self.list_target_channels_mRNA[k]) )
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



# Class spot classification

# Class to calculate tracking quality

# Statistics and PDF

# Autoflorescence removal    

# Class to calculate MSD

# 