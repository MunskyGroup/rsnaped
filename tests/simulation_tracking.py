"""
Created on Tue Mar 30 08:42:24 2021

@author: luisub
"""
######################################
## Importing libraries
import os
import sys
parent_path = os.path.abspath('../rsnaped')
sys.path.append(parent_path)
import rsnaped as rsp

import argparse
import os; from os import listdir; from os.path import isfile, join
import re  
from skimage import io 
from skimage.io import imread
import pkg_resources
pkg_resources.require("numpy>=`1.20.1")  #  to use specific numpy version
import numpy as np
from tqdm.notebook import tqdm
import scipy
import pandas as pd
import shutil
import itertools


######################################
## User passed arguments
parser = argparse.ArgumentParser(description='Pass number of cells, as int.')
parser.add_argument('integers', metavar='N', type=int, nargs='+')
args = parser.parse_args().integers
print(args)
if len (args) >0:
    number_of_simulated_cells = args[0]
else:
    number_of_simulated_cells = 10
print ('The number of simulated cells is: ', number_of_simulated_cells, '\n')

######################################
## Deffining parameters
number_spots_per_cell = 41      
simulation_time_in_sec = 30     
diffusion_coefficient = 0.7 
sel_timepoint = 0#simulation_time_in_sec-1
mask_selection_method = 'max_area' # options are : 'max_spots' and 'max_area' 
particle_size = 5 # spot size for the simulation and tracking.
number_repetitions = 3
use_optimization_for_tracking =0

######################################
## Deffining main functions
def fun_simulated_cells(number_of_simulated_cells=3,number_spots_per_cell=80,simulation_time_in_sec =100,step_size_in_sec=1,particle_size=5, diffusion_coefficient =1,path_to_rSNAPsim= None, path_to_save_output='./temp',intensity_calculation_method='gaussian_fit'):
    spot_size = particle_size
    spot_sigma = 2
    # Code that creates the folder to store results.
    diffusion_coefficient_string = str(diffusion_coefficient).replace('.','_')
    directory_name = '/Simulation_V2__'+'ns_'+str(number_spots_per_cell) +'_diff_'+ diffusion_coefficient_string 
    path_to_save_output = './temp'
    save_to_path =  path_to_save_output + directory_name 
    if not os.path.exists(save_to_path):
        os.makedirs(save_to_path)
    else:
        shutil.rmtree(save_to_path)
        os.makedirs(save_to_path)
    # Loading trajectories from file
    ssa_trajectories = np.load('../DataBases/rsnapsim_simulations/bactin_ssa.npy')
    counter = 0
    ## Main loop that creates each cell and dataframe
    for cell_number in range (0, number_of_simulated_cells):
        ouput_directory_name = '../DataBases/videos_for_sim_cell'
        list_files_names = sorted([f for f in listdir(ouput_directory_name) if isfile(join(ouput_directory_name, f)) and ('.tif') in f], key=str.lower)  # reading all tif files in the folder
        list_files_names.sort(key=lambda f: int(re.sub('\D', '', f)))  # sorting the index in numerical order
        path_files = [ ouput_directory_name+'/'+f for f in list_files_names ] # creating the complete path for each file
        video_path = path_files[counter]        
        video = io.imread(video_path) 
        counter +=1
        if counter>=len(path_files):
            counter =0
        random_index_ch1 = np.random.randint(low=0, high=ssa_trajectories.shape[0]-1, size=(number_spots_per_cell,))
        random_index_ch2 = np.random.randint(low=0, high=ssa_trajectories.shape[0]-1, size=(number_spots_per_cell,))
        simulated_trajectories_ch1 = ssa_trajectories[random_index_ch1,0:simulation_time_in_sec:step_size_in_sec]
        simulated_trajectories_ch2 =  ssa_trajectories[random_index_ch2,0:simulation_time_in_sec:step_size_in_sec]
        saved_file_name = save_to_path+'/sim_cell_'+str(cell_number)
        tensor_video , tensor_for_image_j , spot_positions_movement, tensor_mean_intensity_in_figure, tensor_std_intensity_in_figure, DataFrame_particles_intensities = rsp.SimulatedCell( base_video=video, number_spots = number_spots_per_cell, number_frames=simulation_time_in_sec, step_size=step_size_in_sec, diffusion_coefficient =diffusion_coefficient, simulated_trajectories_ch0=None, size_spot_ch0=spot_size, spot_sigma_ch0=spot_sigma, simulated_trajectories_ch1=simulated_trajectories_ch1, size_spot_ch1=spot_size, spot_sigma_ch1=spot_sigma, simulated_trajectories_ch2=simulated_trajectories_ch2, size_spot_ch2=spot_size, spot_sigma_ch2=spot_sigma, ignore_ch0=0,ignore_ch1=0, ignore_ch2=1,save_as_tif_uint8=0,save_as_tif =1,save_as_gif=0, save_dataframe=1, saved_file_name=saved_file_name,create_temp_folder = False, intensity_calculation_method=intensity_calculation_method).make_simulation()      
        print ('The results are saved in folder: ', saved_file_name)
    return save_to_path

def remove_extrema(vector,min_percentile = 0 ,max_percentile = 100):
    '''This function is intended to remove extrema data given by the min and max percentiles specified by the user'''
    vector = vector [vector>0]
    max_val = np.percentile(vector, max_percentile)
    min_val =  np.percentile(vector, min_percentile)
    print(round(min_val,2),round(max_val,2))
    new_vector = vector [vector< max_val] # = np.percentile(vector,max_percentile)
    new_vector = new_vector [new_vector> min_val] # = np.percentile(vector, min_percentile)
    return new_vector

def running_tracking(number_of_simulated_cells,number_spots_per_cell,simulation_time_in_sec ,step_size_in_sec,particle_size, diffusion_coefficient,path_to_rSNAPsim,intensity_calculation_method,use_optimization_for_tracking):
    # running the simulation
    ouput_directory_name = fun_simulated_cells(number_of_simulated_cells=number_of_simulated_cells,number_spots_per_cell=number_spots_per_cell,simulation_time_in_sec =simulation_time_in_sec,step_size_in_sec=1,particle_size=particle_size, diffusion_coefficient=diffusion_coefficient,path_to_rSNAPsim= None,intensity_calculation_method=intensity_calculation_method)
    path = ouput_directory_name
    # Reads the folder with the results and import the simulations as lists
    list_files_names = sorted([f for f in listdir(ouput_directory_name) if isfile(join(ouput_directory_name, f)) and ('.tif') in f], key=str.lower)  # reading all tif files in the folder
    list_files_names.sort(key=lambda f: int(re.sub('\D', '', f)))  # sorting the index in numerical order
    path_files = [ ouput_directory_name+'/'+f for f in list_files_names ] # creating the complete path for each file
    # Reading the microscopy data
    list_videos = [imread(f)[:,:,:,:] for f in  path_files] # List with all the videos
    nimg = number_of_simulated_cells
    # Tracking
    list_DataFrame_particles_intensities= []
    list_array_intensities = []
    list_time_vector = []
    for i in tqdm(range(0,nimg)): 
        DataFrame_particles_intensities, array_intensities, time_vector, mean_intensities,std_intensities, mean_intensities_normalized, std_intensities_normalized = rsp.PipelineTracking(list_videos[i],particle_size=particle_size,file_name=list_files_names[i],selected_channel=0,intensity_calculation_method =intensity_calculation_method, mask_selection_method = mask_selection_method,show_plot=0, use_optimization_for_tracking =use_optimization_for_tracking).run()    
        list_DataFrame_particles_intensities.append(DataFrame_particles_intensities)
        list_array_intensities.append(array_intensities)
        list_time_vector.append(time_vector)
    # Intensity from trajectories
    ssa_trajectories = np.load('../DataBases/rsnapsim_simulations/bactin_ssa.npy')
    ssa_trajectories_timePoint = ssa_trajectories[:,sel_timepoint].flatten()
    ssa_trajectories_timePoint= remove_extrema(ssa_trajectories_timePoint)
    ssa_trajectories_timePoint_normalized = (ssa_trajectories_timePoint-np.amin(ssa_trajectories_timePoint))/ (np.amax(ssa_trajectories_timePoint)-np.amin(ssa_trajectories_timePoint))
    # Intensity from tracking
    intensity_values_tracking_flat =[]
    for i in range(0,nimg):
        if not ( list_DataFrame_particles_intensities[i] is None): 
            df_intensities_real = list_DataFrame_particles_intensities[i]  
            max_nspots = df_intensities_real['particle'].nunique()
            intensity_values_tracking = np.zeros((max_nspots)) # prealocating memory
            for j in range (0,max_nspots):
                intensity_values_tracking[j] = df_intensities_real[df_intensities_real['particle'] ==j].green_int_mean.values[sel_timepoint]         
            intensity_values_tracking_flat.append(intensity_values_tracking.tolist())
    merged = list(itertools.chain(*intensity_values_tracking_flat))
    merged = [num if num else 0 for num in merged] # removing zeros
    merged = np.asarray(merged)
    merged= remove_extrema(merged)
    intensity_values_tracking_normalized = (merged-np.amin(merged))/ (np.amax(merged)-np.amin(merged)).flatten()
    # Extracting the number of real simulations from folder name
    ind_str_start = path.find('_ns_') +4
    ind_str_end = path.find('_diff') 
    max_nspots = int(path[ind_str_start:ind_str_end])
    intensity_values_in_image = np.zeros((nimg,max_nspots)) # prealocating memory
    for i in range(0,nimg):
        for j in range (0,max_nspots):
            file_name = path+'/sim_cell_'+str(i)+'_df.csv'
            try:
                df_intensities_real = pd.read_csv(file_name)  
                intensity_values_in_image[i,j] = df_intensities_real[df_intensities_real['particle'] ==j].green_int_mean.values[sel_timepoint] 
            except:
                intensity_values_in_image[i,j] = 0 
    intensity_values_in_image_flat = intensity_values_in_image.flatten()
    intensity_values_in_image_flat =  intensity_values_in_image_flat[intensity_values_in_image_flat>0]
    intensity_values_in_image_flat= remove_extrema(intensity_values_in_image_flat)
    intensity_values_in_image_normalized = (intensity_values_in_image_flat-np.amin(intensity_values_in_image_flat))/ (np.amax(intensity_values_in_image_flat)-np.amin(intensity_values_in_image_flat)).flatten()
    # Data flatten
    data1 = ssa_trajectories_timePoint_normalized
    data_sorted_1 = np.sort(data1)
    data2 = intensity_values_tracking_normalized
    data_sorted_2 = np.sort(data2)
    data3 = intensity_values_in_image_normalized
    data_sorted_3 = np.sort(data3)
    # Calculating Kolmogorov distance
    ks_distance_tracking = scipy.stats.kstest(data_sorted_1,data_sorted_2).statistic
    ks_distance_image = scipy.stats.kstest(data_sorted_1,data_sorted_3).statistic
    return ks_distance_tracking, ks_distance_image

######################################
## Running the main function
intensity_calculation_method = 'disk_donut'  # options are : 'total_intensity' and 'disk_donut' 'gaussian_fit'
ks_distance_tracking = np.zeros((number_repetitions))
ks_distance_image = np.zeros((number_repetitions))
for i in range (0,number_repetitions):
    ks_distance_tracking[i], ks_distance_image[i] = running_tracking(number_of_simulated_cells=number_of_simulated_cells,number_spots_per_cell=number_spots_per_cell,simulation_time_in_sec =simulation_time_in_sec,step_size_in_sec=1,particle_size=particle_size, diffusion_coefficient=diffusion_coefficient,path_to_rSNAPsim= None,intensity_calculation_method=intensity_calculation_method,use_optimization_for_tracking=use_optimization_for_tracking)

## Printing results
print('The KS-distance between SSA and tracking is:' , ks_distance_tracking.round(2), '\n')
print('The KS-distance between SSA and image is:' , ks_distance_image.round(2), '\n', '\n')

