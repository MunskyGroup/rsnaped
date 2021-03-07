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
# number_timepoints
# number_channels
# number_videos
# number_images

# index_channels
# index_time
# index_video
# index_image
#index_particle

# self.NUMBER_OF_CORES = multiprocessing.cpu_count()
#min_intensity
# spot_size





class CellposeSelection():
    """
      This class is intended to allow the user to select a mask from multiple masks detected by cellpose
    """     
    def __init__(self,mask,video, slection_method = 'maximum_area', particle_size=5, selected_channel=0):
        self.mask = mask
        self.video = video
        self.num_frames = video.shape[0]
        self.selected_channel = selected_channel      # selected channel
        self.particle_size = particle_size            # according to the documentation it must be an even number 3,5,7,9 etc.
        self.slection_method = slection_method
        self.minimal_frames = int(video.shape[0]*0.9)
                
    def select_mask(self):
        if self.slection_method == 'maximum_area':
            # Iterating for each mask to select the mask with the largest area.
            n_masks = np.amax(self.mask)
            if n_masks >1: # detecting if more than 1 mask are detected per cell
                size_mask = []
                for nm in range (1,n_masks+1): # iterating for each mask in a given cell. The mask has values from 0 for background, to int n, where n is the number of detected masks.  
                    size_mask.append(np.sum(self.mask==nm)) # creating a list with the size of each mask
                largest_mask=np.argmax(size_mask)+1 # detecting the mask with the largest value
                temp_mask = np.zeros_like(self.mask) # making a copy of the image
                selected_mask = temp_mask + (self.mask==largest_mask) # Selecting a single mask and making this mask equal to one and the background equal to zero. 
            else: # do nothing if only a single mask is detected per image.
                selected_mask = self.mask
        if self.slection_method == 'maximum_spots':
            # Iterating for each mask to select the mask with the largest area.
            n_masks = np.amax(self.mask)
            if n_masks >1: # detecting if more than 1 mask are detected per cell
                number_particles = []
                for nm in range (1,n_masks+1): # iterating for each mask in a given cell. The mask has values from 0 for background, to int n, where n is the number of detected masks.  
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
        return selected_mask