#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 17:23:24 2020

@author: luisaguilera
"""

# Importing libraries
from sys import platform
import os
from skimage import io ; from skimage.io import imread; from skimage.measure import find_contours
import numpy as np 
from random import randrange
import os; from os import listdir; from os.path import isfile, join
import pandas as pd

# Importing rSNAPsim_IP
cwd = os.getcwd(); os.chdir('..'); import rSNAPsim_IP as rss_IP ; os.chdir(cwd) # return to the working directory


def simulated_cells(number_of_simulated_cells,number_spots_per_cell=10,simulation_time_in_sec=50,step_size_in_sec=1, background_noise_percentage=0.01,diffusion_coeffient=None, path_to_rSNAPsim = None, path_to_save_output = os.getcwd() ):

    if diffusion_coeffient == None:
        average_diffusion_coeff_um_2_sec = np.mean([0.018,0.016,0.01,0.021,0.047]) # Calculating average diff coef in micrometers^2/s
        average_diffusion_coeff_px_2_sec = average_diffusion_coeff_um_2_sec * (1000/1) * (1/130) # conversion um^2/sec to px^2/sec = um^2/sec * (1000nm2/1um^2) * (1px^2/130nm^2)
        average_diffusion_coeff_px_2_sec = round(average_diffusion_coeff_px_2_sec, 4)
        diffusion_coeffient = average_diffusion_coeff_px_2_sec
        
    
    # Loading rSNAPsim-dev
    #rSNAPsim_directory = path_to_rSNAPsim
    #sequences_directory = path_to_rSNAPsim +'/gene_files/Bactin_withTags.txt'
    #importing rSNAPsim.
    #cwd = os.getcwd(); os.chdir(rSNAPsim_directory); import rSNAPsim; rss = rSNAPsim.rSNAPsim(); os.chdir(cwd); rss = rSNAPsim.rSNAPsim()


    ##################################
    ##################################
    ## Main simulation parameters
    number_of_cells =number_of_simulated_cells
    number_spots = number_spots_per_cell # number translation spots per cell
    n_frames = simulation_time_in_sec # in seconds
    step_size = step_size_in_sec # in seconds
    diffusion_coeffient = diffusion_coeffient # Diffusion coeffient for Brownian Motion
    bg_noise_percentage = background_noise_percentage # amount of noise (fraction from 0 to 1) where 0 is 0% and 1 is 100%.
    ##################################
    ##################################
    
    # Paramerters for channel 1 -> RED
    minimal_background_outside_cell_ch0 =30
    minimal_background_inside_cell_ch0 =100
    size_spot_ch0 = 5
    spot_sigma_ch0 = 1.5
    add_background_noise_ch0 = 1
    amount_background_noise_ch0 = bg_noise_percentage
    add_photobleaching_ch0 = 0
    photobleaching_exp_constant_ch0 = 0.0005
    simulated_trajectories_ch0 =0
    ignore_ch0 = 0
    use_triangulation_ch0 =0
    
    max_phoptobleaching = np.log(5)/ n_frames # this line defines the maximum value for the exponential constant that doesnt excedes 255(max uint8 value). 
    
    # Paramerters for channel 2 -> GREEN
    minimal_background_outside_cell_ch1 =30
    minimal_background_inside_cell_ch1 =100
    size_spot_ch1 = 5
    spot_sigma_ch1 = 1.5
    add_background_noise_ch1 = 1
    amount_background_noise_ch1 = bg_noise_percentage
    add_photobleaching_ch1 = 0
    photobleaching_exp_constant_ch1 = max_phoptobleaching*0.5
    ignore_ch1 = 0
    use_triangulation_ch1=1
    
    # Paramerters for channel 3 -> BLUE
    minimal_background_outside_cell_ch2 =30
    minimal_background_inside_cell_ch2 =100
    size_spot_ch2 = 5
    spot_sigma_ch2 = 1.5
    add_background_noise_ch2 = 1
    amount_background_noise_ch2 = bg_noise_percentage
    add_photobleaching_ch2 = 0
    photobleaching_exp_constant_ch2 = max_phoptobleaching*0.9
    ignore_ch2 = 0
    use_triangulation_ch2 = 0
    
    


    # Loading sequences for the rSNAPsim for intensity trajectories
    #rss.open_seq_file(sequences_directory)
    #rss.get_orfs(rss.sequence_str) # now to look for open reading frames
    #rss.orfs # reading the 3 ORF.
    #protein1 = rss.sequence_str  # Deffining a Protein passing the DNA sequence 
    #rss.nt2aa(protein1) # Converting nucleotides to Protein
    #rss.get_temporal_proteins()  #proteins sorted by
    #rss.analyze_poi(rss.pois[0],rss.pois_seq[0])  # Analyze the first protein, analyze function takes the aa_seq then nt_seq
    #rss.POI 
    #starting_time = 1000
    
    
    # Deffining the cell shape directory.
    cell_shapes_directory = '../DataBases/Cell_Shapes/'
    ## Section that determines the number of cell_shapes
    files = sorted([f for f in listdir(cell_shapes_directory) if isfile(join(cell_shapes_directory, f)) and ('.tif') in f], key=str.lower)  # reading all tif files in the folder
    if len(files)>0:
        counter =0
        for i in range(0,len(files)):
            val = files[i].find('.tif')
            if val >= 0:
                counter+=1
        number_cell_shapes = counter
    else:
        number_cell_shapes = 0
    
    # Code that creates the folder to store results.
    noise_string = str(bg_noise_percentage).replace('.','_')
    diffusion_coeffient_string = str(diffusion_coeffient).replace('.','_')
    save_to_path = path_to_save_output + '/Results_noise_'+noise_string+'_diff_coef_'+ diffusion_coeffient_string 
    if not os.path.exists(save_to_path):
        os.makedirs(save_to_path)
        
    
    # Loading trajectories from file
    ssa_trajectories = np.load('../DataBases/rsnapsim_simulations/bactin_ssa.npy')
    
    
    
    ## Main loop that creates each cell and dataframe
    for cell_number in range (0, number_of_cells):
        
        random_index_ch2 = np.random.randint(low=0, high=499, size=(number_spots_per_cell,))
        random_index_ch3 = np.random.randint(low=0, high=499, size=(number_spots_per_cell,))

        simulated_trajectories_ch2 = ssa_trajectories[random_index_ch2,0:simulation_time_in_sec:step_size_in_sec]
        simulated_trajectories_ch3 =  ssa_trajectories[random_index_ch3,0:simulation_time_in_sec:step_size_in_sec]
        
        # Simulations for intensity
#        ssa1 = rss.ssa_solver(n_traj = number_spots, start_time=starting_time,tf=starting_time+n_frames, tstep=starting_time+n_frames,k_elong_mean=3, k_initiation=.03)  # tstep = total number of steps including the burnin time 
#        simulated_trajectories = ssa1.intensity_vec
#        ssa2 = rss.ssa_solver(n_traj = number_spots, start_time=starting_time,tf=starting_time+n_frames, tstep=starting_time+n_frames,k_elong_mean=3, k_initiation=.03)  # tstep = total number of steps including the burnin time 
#        simulated_trajectories_blue = ssa2.intensity_vec
        # simulated trajectories for the green and blue channels
#        simulated_trajectories_ch2 = simulated_trajectories
#        simulated_trajectories_ch3 = simulated_trajectories_blue
        
        
        # Selecting the cell shape from the directly. If the number of cells is larger than the number of shapes, it will randomly select a shape.
        if cell_number < number_cell_shapes:
            cell_shape = io.imread(cell_shapes_directory+'cell_shape_'+str(cell_number)+'.tif') 
        else:
            cell_shape = io.imread(cell_shapes_directory+'cell_shape_'+str(randrange(number_cell_shapes))+'.tif') 
        
        # Parameter for the shape of the simulated cell
        contours = np.array(find_contours(cell_shape, 0.8),dtype = int)
        polygon_array = contours[0] 
        height = cell_shape.shape[0]
        width  = cell_shape.shape[1]
        image_size = [height,width,1]
            
        # Running the cell simulation
        saved_file_name = save_to_path+'/sim_cell_'+str(cell_number)
        obj_Simulated_Cell = rss_IP.Simulated_cell(number_spots, n_frames, step_size, diffusion_coeffient, polygon_array, image_size, simulated_trajectories_ch0, minimal_background_outside_cell_ch0, minimal_background_inside_cell_ch0, size_spot_ch0, spot_sigma_ch0, add_background_noise_ch0, amount_background_noise_ch0, add_photobleaching_ch0, photobleaching_exp_constant_ch0, simulated_trajectories_ch2, minimal_background_outside_cell_ch1, minimal_background_inside_cell_ch1, size_spot_ch1, spot_sigma_ch1, add_background_noise_ch1, amount_background_noise_ch1, add_photobleaching_ch1, photobleaching_exp_constant_ch1, simulated_trajectories_ch3, minimal_background_outside_cell_ch2, minimal_background_inside_cell_ch2, size_spot_ch2, spot_sigma_ch2, add_background_noise_ch2, amount_background_noise_ch2, add_photobleaching_ch2, photobleaching_exp_constant_ch2, ignore_ch0,ignore_ch1, ignore_ch2,use_triangulation_ch0,use_triangulation_ch1,use_triangulation_ch2,save_as_tif=1, save_dataframe=1, saved_file_name=saved_file_name)
        tensor_video, tensor_for_image_j, spot_positions_movement, tensor_mean_intensity_in_figure, tensor_std_intensity_in_figure = obj_Simulated_Cell.make_simulated_cell()
        
        print ('The results are saved in folder: ', saved_file_name)


