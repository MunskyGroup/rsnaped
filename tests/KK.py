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



# 
class Simulated_cell_from_video():
    def __init__(self, base_video, number_spots, number_frames, step_size =1, diffusion_coefficient =0.01, simulated_trajectories_ch0=None, size_spot_ch0=5, spot_sigma_ch0=2, simulated_trajectories_ch1=[0], size_spot_ch1=5, spot_sigma_ch1=2, simulated_trajectories_ch2=[0], size_spot_ch2=5, spot_sigma_ch2=2, ignore_ch0=0,ignore_ch1=1, ignore_ch2=1,save_as_tif_uint8 =0, save_as_tif=0, save_as_gif=0, save_dataframe=0, saved_file_name='temp',create_temp_folder = True):        
        '''
        This class takes a base video and it draws simulated spots on top of the image. The intensity for each simulated spot is proportional to the stochastic simulation given by the user. 

        Parameters
        ----------
        base_video : NumPy array 
            Array of images with dimensions [T,Y,X,C]. 
        number_spots : int
            Number of simulated spots in the cell.
        number_frames : int
            Number of frames or time points to simulate. In seconds. 
        step_size : int, optional
            Step size for the simulation. In seconds. The default is 1.
        diffusion_coefficient : float, optional
            Diffusion coefficient for the particles' brownian motion. The default is 0.01.
        simulated_trajectories_ch0 : NumPy array, optional
            Array of simulated trajectories with dimensions [trajectories,time_points] for channel 0. The default is None, and indicates that the intensity will be generated drawing random numbers in a range.
        size_spot_ch0 : int, optional
            Spot size in pixels for channel 0. The default is 5.
        spot_sigma_ch0 : int, optional
            Sigma value for the simulated spots in channel 0, the units are pixels. The default is 2.
        simulated_trajectories_ch1 : NumPy array, optional
            Array of simulated trajectories with dimensions [trajectories,time_points] for channel 1. The default is None, and indicates that the intensity will be generated drawing random numbers in a range.
        size_spot_ch1 : int, optional
            Spot size in pixels for channel 1. The default is 5.
        spot_sigma_ch1 : int, optional
            Sigma value for the simulated spots in channel 1, the units are pixels. The default is 2.
        simulated_trajectories_ch2 : NumPy array, optional
            Array of simulated trajectories with dimensions [trajectories,time_points] for channel 2. The default is None, and indicates that the intensity will be generated drawing random numbers in a range.
        size_spot_ch2 : int, optional
            Spot size in pixels for channel 2. The default is 5.
        spot_sigma_ch2 : int, optional
            Sigma value for the simulated spots in channel 2, the units are pixels. The default is 2.
        ignore_ch0 : bool, optional
            Flag that ignores channel 0 returning a numpy array filled with zeros. The default is 0.
        ignore_ch1 : bool, optional
            Flag that ignores channel 1 returning a numpy array filled with zeros. The default is 1.
        ignore_ch2 : bool, optional
            Flag that ignores channel 2 returning a numpy array filled with zeros. The default is 1.
        save_as_tif_uint8 : bool, optional
            If true, it generates and saves a uint8 (low) quality image tif file for the simulation. The default is 0.
        save_as_tif : bool, optional
            If true, it generates and saves a uint16 (High) quality image tif file for the simulation. The default is 0.
        save_as_gif : bool, optional
            If true, it generates and saves a gif animation for the simulation. The default is 0.
        save_dataframe : bool, optional
            If true, it generates and saves a pandas dataframe with the simulation. Dataframe with fields: [cell_number,spot_number,frame,red_int_mean,green_int_mean,blue_int_mean,red_int_std,green_int_std,blue_int_std,x_position,y_position]. The default is 0.
        saved_file_name : str, optional
            File name for the simulated cell output files (tif images, gif images, dataframes). The default is 'temp'.
        create_temp_folder : bool, optional
            Creates a folder with the simulation output. The default is True.
        '''
        self.base_video = base_video
        self.number_spots = number_spots
        self.number_frames = number_frames
        self.step_size  = step_size
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
        self.time_vector = np.arange(0,number_frames,step_size)
        self.save_as_tif_uint8 = save_as_tif_uint8
        self.save_as_tif = save_as_tif
        self.save_as_gif = save_as_gif
        self.save_dataframe = save_dataframe
        self.saved_file_name = saved_file_name
        self.create_temp_folder = create_temp_folder
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
        selected_image = self.base_video[0,:,:,1] # selecting for the mask the first time point
        selected_masks, _,_ = Cellpose(selected_image,num_iterations=10, channels=[0],diameter=200,model_type='cyto',selection_method = 'max_area').calculate_masks() # options are 'max_area' or 'max_cells'
        selected_mask  = CellposeSelection(selected_masks,selected_image, slection_method = 'maximum_area').select_mask()
        # section that converts the mask in contours and resduces the size of the mask to ensure the spots are generated inside the cell.
        contours = np.array(find_contours(selected_mask, 0.5),dtype = int)
        polygon_array = contours[0] 
        self.polygon_array = mask_reduction(polygon_array, percentage_reduction= 0.25)
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
            Dataframe with fields: [cell_number,spot_number,frame,red_int_mean,green_int_mean,blue_int_mean,red_int_std,green_int_std,blue_int_std,x_position,y_position].
        '''
        def calculate_intensity_in_figure(tensor_image,time_vector,number_spots, spot_positions_movement, size_spot):
            # Function that allows the user to 
            if (size_spot % 2) ==0:
                size_spot = size_spot + 1
            disk_size =  int(size_spot/2)
            crop_size = disk_size+2
            tensor_mean_intensity_in_figure = np.zeros((len(time_vector),number_spots),dtype='float')
            tensor_std_intensity_in_figure = np.zeros((len(time_vector),number_spots),dtype='float')
            def gaussian_fit(test_im,mean_intensity_donut):
                # Function that calculates the intensity by using a gaussian fit.
                size_spot = test_im.shape[0]
                image_flat = test_im.ravel()
                def gaussian_function(size_spot, offset,sigma,mean_intensity_donut):
                    ax = np.linspace(-(size_spot - 1) / 2., (size_spot - 1) / 2., size_spot)        
                    xx, yy = np.meshgrid(ax, ax)
                    kernel =  mean_intensity_donut + (offset *(np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))))
                    return kernel.ravel()
                p0 = (np.mean(image_flat) ,2,mean_intensity_donut)
                popt, _ = curve_fit(gaussian_function,size_spot, image_flat, p0=p0)
                spot_intensity_gaussian = popt[0]#-mean_intensity_donut # Amplitude
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
            MIN_INTENSITY_ALL_SPOTS = 30000 # basal value for each spot
            MAX_INTENSITY_ALL_SPOTS = 35000 # maximum allowed intensity for a given spot
            MAX_INTENSITY_IN_UINT16 = 65535 # maximum value in a unint16 image
            # The following two constants are weights used to define a range of intensities for the simulated spots.
            MIN_INTENSITY_SPOT_WEIGHT = 1.05 # lower weight that multiplays the mean intensity value in the image to define the simulated spot intensity.
            MAX_INTENSITY_SPOT_WEIGHT = 1.5 # higher weight that multiplays the mean intensity value in the image to define the simulated spot intensity.
            for point_index in range(0,len(center_positions_vector)):
                # Section that creates the Gaussian Kernel Matrix
                ax = np.linspace(-(size_spot - 1) / 2., (size_spot - 1) / 2., size_spot)        
                xx, yy = np.meshgrid(ax, ax)
                kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(spot_sigma))
                #kernel = kernel / np.sum(kernel) # nrrmalized to one
                # creating a position for each spot
                center_position = center_positions_vector[point_index]
                pixel_size_arround_spot = size_spot + 1
                try:
                    # this section calculates the basal intensity arround an spots
                    temp_int_arround_spot = pixelated_image_no_spots[center_position[0]-int(pixel_size_arround_spot/2): center_position[0]+int(pixel_size_arround_spot/2)+1 ,center_position[1]-int(pixel_size_arround_spot/2): center_position[1]+int(pixel_size_arround_spot/2)+1 ]
                    temp_int_arround_spot.flatten()
                    temp_mean = np.mean(temp_int_arround_spot)
                    temp_std = np.std(temp_int_arround_spot)
                    dist_mean = abs(temp_int_arround_spot - temp_mean)
                    max_dev = 1
                    idx_inside_elements = dist_mean < max_dev * temp_std
                    inside_elements = temp_int_arround_spot[idx_inside_elements]
                    mean_int_arround_spot = np.mean(inside_elements)
                except:
                    mean_int_arround_spot = pixelated_image_no_spots[center_position[0],center_position[1]]
                if using_ssa == 1 :
                    min_int = np.amin((mean_int_arround_spot *MIN_INTENSITY_SPOT_WEIGHT,MIN_INTENSITY_ALL_SPOTS))
                    max_int = np.amin((mean_int_arround_spot *MAX_INTENSITY_SPOT_WEIGHT,MAX_INTENSITY_ALL_SPOTS))  # 65535 is the maximum value for np.uint16   # bast value 1.5
                    if max_int == MAX_INTENSITY_IN_UINT16:
                         print('Warning! Spot genrated in an area with high intensity, mean_int = ',mean_int_arround_spot)
                    # Scaling the spot kernel matrix with the intensity value form the SSA.                    
                    int_ssa = (simulated_trajectories_time_point[point_index] - min_SSA_value) / (max_SSA_value-min_SSA_value) # intensity normalized to min and max values in the SSA
                    int_tp = min_int + (int_ssa * (max_int-min_int)) # Scaling to the min and max values in the image
                    spot_intensity = int_tp 
                    if int_tp<0:
                        print(round(int_tp,1))
                    kernel_value_intensity = (kernel*spot_intensity) 
                else:
                    min_int = np.amin((mean_int_arround_spot *MIN_INTENSITY_SPOT_WEIGHT,MIN_INTENSITY_ALL_SPOTS))
                    max_int = np.amin((mean_int_arround_spot *MAX_INTENSITY_SPOT_WEIGHT,MAX_INTENSITY_ALL_SPOTS))
                    # Scaling the spot kernel matrix with a mean value obtained from uniformly random distribution.
                    spot_intensity = int(random.uniform(min_int,max_int))
                    kernel_value_intensity = (kernel*spot_intensity)
                # Setting thresholds
                kernel_value_intensity[kernel_value_intensity < min_int] = min_int # making spots with less intensity than the background equal to the background.
                kernel_value_intensity[kernel_value_intensity > MAX_INTENSITY_ALL_SPOTS] = MAX_INTENSITY_ALL_SPOTS # making spots with more intensity than maximum allowed intensity equal to max_int
                # Defining the current area in the matrix that will be replaced by the spot kernel.        
                pixelated_image[center_position[0]-int(size_spot/2): center_position[0]+int(size_spot/2)+1 ,center_position[1]-int(size_spot/2): center_position[1]+int(size_spot/2)+1 ] = kernel_value_intensity    
            return pixelated_image # final_image
        def make_spots_movement(polygon_array, number_spots, time_vector, step_size, image_size, diffusion_coefficient,base_video=None):
            # Function that creates the simulated spots inside a given polygon
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
                if not ( base_video is None):
                    selected_image= base_video
                    max_allowed_int_image = np.mean(selected_image) + 3* np.std(selected_image) 
                    size_spot =5
                    pixel_size_arround_spot = size_spot + 20
                    temp_crop_arround_spot = selected_image[test_points[0]-int(pixel_size_arround_spot/2): test_points[0]+int(pixel_size_arround_spot/2)+1 ,test_points[1]-int(pixel_size_arround_spot/2): test_points[1]+int(pixel_size_arround_spot/2)+1 ]
                    mean_int_tested_spot = np.mean(temp_crop_arround_spot)
                    if mean_int_tested_spot<max_allowed_int_image:
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
            return spot_positions_movement # vector with dimensions (time, spot, y, x )
        def make_simulation(base_video_slected_channel,masked_video_slected_channel, spot_positions_movement, time_vector, polygon_array, image_size,size_spot, spot_sigma, simulated_trajectories):
            # Main function that makes the simulated cell by calling multiple function.
            temp_image = masked_video_slected_channel[0,:,:]
            temp_image_nonzeros = temp_image.copy()
            temp_image_nonzeros.flatten
            #non_zeros_image = temp_image_nonzeros[temp_image_nonzeros!=0]
            # Calculate initial background intensity for each spot.
            tensor_image = np.zeros((len(time_vector),image_size[0],image_size[1]),dtype=np.uint16)
            counter =0
            for t_p,_ in enumerate(time_vector):
                if counter >= base_video_slected_channel.shape[0]:
                    counter = 0                 
                matrix_background = base_video_slected_channel[counter,:,:] 
                if not ( simulated_trajectories is None):
                    using_ssa = 1
                    simulated_trajectories_tp = simulated_trajectories[:,t_p]
                    max_SSA_value = simulated_trajectories.max()
                    min_SSA_value = simulated_trajectories.min()
                else:
                    using_ssa = 0
                    simulated_trajectories_tp = 0
                    max_SSA_value = 0
                    min_SSA_value =0
                # Making the pixelated spots    
                tensor_image[t_p,:,:] = make_replacement_pixelated_spots(matrix_background, spot_positions_movement[t_p,:,:], size_spot, spot_sigma, using_ssa, simulated_trajectories_tp,min_SSA_value,max_SSA_value)
                counter += 1 
            return tensor_image
        # Create the spots for all channels. Return array with 3-dimensions: T,Sop, XY-Coord
        spot_positions_movement = make_spots_movement(self.polygon_array, self.number_spots, self.time_vector, self.step_size, self.image_size, self.diffusion_coefficient,self.base_video[0,:,:,1])
        # This section of the code runs the for each channel    
        if self.ignore_ch0 == 0:
            tensor_image_ch0 = make_simulation(self.base_video[:,:,:,0], self.video_removed_mask[:,:,:,0], spot_positions_movement, self.time_vector, self.polygon_array, self.image_size, self.size_spot_ch0, self.spot_sigma_ch0, self.simulated_trajectories_ch0)
            tensor_mean_intensity_in_figure_ch0, tensor_std_intensity_in_figure_ch0 = calculate_intensity_in_figure(tensor_image_ch0,self.time_vector,self.number_spots, spot_positions_movement, self.size_spot_ch0)
        else:
            tensor_image_ch0 = np.zeros((self.number_frames,self.image_size[0],self.image_size[1]),dtype=np.uint16) 
            tensor_mean_intensity_in_figure_ch0 = np.zeros(( len(self.time_vector),self.number_spots),dtype=np.uint16)
            tensor_std_intensity_in_figure_ch0 = np.zeros((len(self.time_vector),self.number_spots),dtype=np.uint16)
        # Channel 1
        if self.ignore_ch1 == 0:
            tensor_image_ch1 = make_simulation(self.base_video[:,:,:,1],self.video_removed_mask[:,:,:,1],spot_positions_movement, self.time_vector, self.polygon_array, self.image_size, self.size_spot_ch1, self.spot_sigma_ch1, self.simulated_trajectories_ch1)
            tensor_mean_intensity_in_figure_ch1, tensor_std_intensity_in_figure_ch1 = calculate_intensity_in_figure(tensor_image_ch1,self.time_vector,self.number_spots, spot_positions_movement, self.size_spot_ch1)
        else:
            tensor_image_ch1 = np.zeros((self.number_frames,self.image_size[0],self.image_size[1]),dtype=np.uint16)    
            tensor_mean_intensity_in_figure_ch1 = np.zeros((len(self.time_vector),self.number_spots),dtype=np.uint16)
            tensor_std_intensity_in_figure_ch1 = np.zeros((len(self.time_vector),self.number_spots),dtype=np.uint16)
        # Channel 2
        if self.ignore_ch2 == 0:   
            tensor_image_ch2 = make_simulation(self.base_video[:,:,:,2], self.video_removed_mask[:,:,:,2], spot_positions_movement, self.time_vector, self.polygon_array, self.image_size, self.size_spot_ch2, self.spot_sigma_ch2, self.simulated_trajectories_ch2)
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
                save_to_path = 'temp/'
                if not os.path.exists(save_to_path):
                    os.makedirs(save_to_path)
                print ("The output is saved in the directory: ./" , save_to_path[0:-1])
            else:
                save_to_path=''
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
                tifffile.imwrite(save_to_path+self.saved_file_name+'_unit8_'+'.tif', normalized_tensor)
            if self.save_as_gif==1:
                # Saving the simulation as a gif. Complete image
                with imageio.get_writer(save_to_path+self.saved_file_name+'_unit8_'+'.gif', mode='I') as writer:
                    for i in range(0,num_images_for_gif):
                        image = normalized_tensor[i,:,:,:]
                        writer.append_data(image)
        if self.save_as_tif==1:
            if self.create_temp_folder == True:
                save_to_path = 'temp/'
                if not os.path.exists(save_to_path):
                    os.makedirs(save_to_path)
                print ("The output is saved in the directory: ./" , save_to_path[0:-1])
            else:
                save_to_path=''
            tifffile.imwrite(save_to_path+self.saved_file_name+'.tif', tensor_video)
        # Initialize the dataframe    
        init_dataFrame = {'spot_number': [],
            'frame': [],
            'red_int_mean': [],
            'green_int_mean': [],
            'blue_int_mean': [],
            'red_int_std': [],
            'green_int_std': [],
            'blue_int_std': [],
            'x_position': [],
            'y_position': []}
        dataframe_particles = pd.DataFrame(init_dataFrame)
        for n_spot in range (0,self.number_spots):
        # Section that append the information for each spots
            temp_data_frame = {'spot_number': n_spot,
                'frame': self.time_vector,
                'red_int_mean': tensor_mean_intensity_in_figure[:,n_spot,0],
                'green_int_mean': tensor_mean_intensity_in_figure[:,n_spot,1],
                'blue_int_mean': tensor_mean_intensity_in_figure[:,n_spot,2],
                'red_int_std': tensor_std_intensity_in_figure[:,n_spot,0],
                'green_int_std': tensor_std_intensity_in_figure[:,n_spot,1],
                'blue_int_std': tensor_std_intensity_in_figure[:,n_spot,2],
                'x_position': spot_positions_movement[:,n_spot,1],
                'y_position': spot_positions_movement[:,n_spot,0]}
            temp_dataframe = pd.DataFrame(temp_data_frame)
            dataframe_particles = dataframe_particles.append(temp_dataframe, ignore_index=True)
            dataframe_particles = dataframe_particles.astype({"spot_number": int, "frame": int}) # specify data type as integer for some columns    
        if self.save_dataframe==1:
            if self.create_temp_folder == True:
                save_to_path = 'temp/'
                if not os.path.exists(save_to_path):
                    os.makedirs(save_to_path)
                print ("The output is saved in the directory: ./" , save_to_path[0:-1])
            else:
                save_to_path=''
            dataframe_particles.to_csv(save_to_path+self.saved_file_name +'_df'+ '.csv', index = True)
        return tensor_video , tensor_for_image_j , spot_positions_movement, tensor_mean_intensity_in_figure, tensor_std_intensity_in_figure, dataframe_particles

