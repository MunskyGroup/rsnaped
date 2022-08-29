#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 04 03:01:24 2021

@author: luisub
"""
import os
import pkg_resources
pkg_resources.require("numpy>=`1.20.1")  #  to use specific numpy version
import numpy as np
import rsnapsim as rss
import sys
from sys import platform
from skimage import io ; from skimage.io import imread; from skimage.measure import find_contours
from random import randrange
import pandas as pd
import os; from os import listdir; from os.path import isfile, join
import re
import shutil
import pathlib
from pathlib import Path
from random import randrange

import argparse

# Deffining directories
current_dir = pathlib.Path().absolute()
sequences_dir = current_dir.parents[1].joinpath('DataBases','gene_files')
video_dir = current_dir.parents[1].joinpath('DataBases','videos_for_sim_cell')
rsnaped_dir = current_dir.parents[1].joinpath('rsnaped')
masks_dir = current_dir.parents[1].joinpath('DataBases','masks_for_sim_cell')


# Importing rSNAPsim_IP
sys.path.append(str(rsnaped_dir))
import rsnaped as rsp
import matplotlib
import matplotlib.pyplot as plt


######################################
## User passed arguments
parser = argparse.ArgumentParser(description='Pass parameters for simulation')
parser.add_argument('integers', metavar='N', type=float, nargs='+')
args = parser.parse_args().integers
# Parameters for simulation
total_number_of_spots = int(args[0])
simulation_time_in_sec = int(args[1])
ke_gene_0 =args[2]
ke_gene_1 = args[3]
ki_gene_0 = args[4]
ki_gene_1 =args[5]


print ('==== Running simulation ====', '\n',
    'total_number_of_spots = ', total_number_of_spots,  ' \n',
    'simulation time = ',simulation_time_in_sec, ' \n',
    'ke gene 0 = ',ke_gene_0, ' \n',
    'ke gene 1 = ',ke_gene_1, ' \n',
    'ki gene 0 = ',ki_gene_0, ' \n',
    'ki gene 1 = ',ki_gene_1, ' \n')
######################################

# Path to store data
current_folder = pathlib.Path().absolute()
multiplexing_path_folder = current_folder.joinpath('temp')
if not os.path.exists(str(multiplexing_path_folder)):
    os.makedirs(str(multiplexing_path_folder))

simulated_RNA_intensities_method = 'constant'

# Reading all empty cells in directory
list_files_names = sorted([f for f in listdir(video_dir) if isfile(join(video_dir, f)) and ('.tif') in f], key=str.lower)  # reading all tif files in the folder
list_files_names.sort(key=lambda f: int(re.sub('\D', '', f)))  # sorting the index in numerical order
path_files = [ str(video_dir.joinpath(f).resolve()) for f in list_files_names ] # creating the complete path for each file
num_cell_shapes = len(path_files)

# Coding sequence
gene_file_KDM5B_PP7 = str(sequences_dir.joinpath('pUB_SM_KDM5B_PP7_coding_sequence.txt')) # coding sequence for SM_KDM5B_PP7    ### 5685 nt   ### 1895 codons
gene_file_p300_MS2 = str(sequences_dir.joinpath('pUB_SM_p300_MS2_coding_sequence.txt'))  # coding sequence for SM_p300_MS2      ### 8268 nt   ### 2756 codons


# Constant Parameters
list_gene_sequences = [gene_file_KDM5B_PP7, gene_file_p300_MS2] # path to gene sequences
list_label_names = [0,1] # list of strings or int used to generate a classification field in the output data frame
step_size_in_sec = 1 # step size
save_as_tif = 0 # option to save the simulated video
save_dataframe = 0 # option to save the simulation output as a dataframe in format csv. 
create_temp_folder = 0 # saves the video and data frame in a temp folder
spot_size = 5 # size of spots in pixels
list_number_spots = [25, 25] # list of integers, where each element represents the number of spots
number_cells = int(total_number_of_spots // sum (list_number_spots))  # Number of simulated Cell

if number_cells <1:
    number_cells =1
    print('The minimal number of spots must be more than: ' ,sum (list_number_spots),' that is the default number of spots simulated in a cell.')

list_elongation_rates = [ke_gene_0, ke_gene_1] # elongation rates aa/sec
list_initiation_rates = [ki_gene_0, ki_gene_1] # initiation rates 1/sec
list_target_channels_proteins = [1, 1] # channel where the simulated protein spots will be located. Integer between 0 and 2. 
list_target_channels_mRNA = [0, 2] # channel where the simulated mRNA spots will be located. Integer between 0 and 2. 
list_diffusion_coefficients =[0.55, 0.55] # diffusion coefficients for each gene

# SSA to intensity conversion scales 
intensity_scale_ch0 = 100
intensity_scale_ch1 = 200
intensity_scale_ch2 = 200

def simulate_multiplexing( path_files,masks_dir,tested_list_elongation_rates,tested_list_initiation_rates,tested_list_diffusion_coefficients,multiplexing_path_folder,frame_selection_empty_video,simulated_RNA_intensities_method):
    # function  that simulates the multiplexing experiments    
    list_dataframe_simulated_cell =[]
    list_ssa_all_cells_and_genes =[]
    for i in range(0,number_cells): # for i in range (0,number_cells ):
        saved_file_name = 'cell_' + str(i)  # if the video or dataframe are save, this variable assigns the name to the files
        sel_shape = randrange(num_cell_shapes)
        video_path = path_files[sel_shape]
        initial_video = io.imread(video_path) # video with empty cell
        mask_image = imread(masks_dir.joinpath('mask_cell_shape_'+str(sel_shape)+'.tif'))
        # This step reduces the intensity of the empty video by a half. This is necessary to match the intensity in a video with spots. Check code "Analysis_simulated_cells.ipynb"
        _, single_dataframe_simulated_cell, list_ssa = rsp.SimulatedCellMultiplexing(initial_video,
                                                                                    list_gene_sequences,
                                                                                    list_number_spots,
                                                                                    list_target_channels_proteins,
                                                                                    list_target_channels_mRNA, 
                                                                                    tested_list_diffusion_coefficients,
                                                                                    list_label_names,
                                                                                    tested_list_elongation_rates,
                                                                                    tested_list_initiation_rates,
                                                                                    simulation_time_in_sec,
                                                                                    step_size_in_sec,
                                                                                    save_as_tif, 
                                                                                    save_dataframe, 
                                                                                    saved_file_name,
                                                                                    create_temp_folder,
                                                                                    mask_image=mask_image,
                                                                                    cell_number =i,
                                                                                    frame_selection_empty_video=frame_selection_empty_video,
                                                                                    spot_size =spot_size ,
                                                                                    intensity_scale_ch0 = intensity_scale_ch0,
                                                                                    intensity_scale_ch1 = intensity_scale_ch1,
                                                                                    intensity_scale_ch2 = intensity_scale_ch2,
                                                                                    dataframe_format='long',
                                                                                    simulated_RNA_intensities_method=simulated_RNA_intensities_method).make_simulation()
        # appending dataframes for each cell
        list_dataframe_simulated_cell.append(single_dataframe_simulated_cell)
        list_ssa_all_cells_and_genes.append(list_ssa)
    # Creating a folder
    folder_to_save_data = 'multiplexing_data__bg_' + frame_selection_empty_video + '__ke_' + str(tested_list_elongation_rates[0])+'_'+str(tested_list_elongation_rates[1])+'__ki_'+str(tested_list_initiation_rates[0])[2:]+'_'+str(tested_list_initiation_rates[1])[2:]+'__kdiff_'+str(tested_list_diffusion_coefficients[0])+'_'+str(tested_list_diffusion_coefficients[1])  + '__time_' + str(simulation_time_in_sec) + '__cells_' + str(number_cells) +'__spots_' +str(list_number_spots[0])+ '_' +str(list_number_spots[1]) +'__int_scale_ch0_' +str(intensity_scale_ch0)+'__int_scale_ch1_' +str(intensity_scale_ch1)+'__int_scale_ch2_' +str(intensity_scale_ch2)
    folder_to_save_data = folder_to_save_data.replace(".", "_")
    multiplexing_path = multiplexing_path_folder.joinpath(folder_to_save_data)
    dataframe_simulated_cell = pd.concat(list_dataframe_simulated_cell)
    ssas_multiplexing = np.array(list_ssa_all_cells_and_genes)
    return multiplexing_path,folder_to_save_data, dataframe_simulated_cell, ssas_multiplexing


def save_data (multiplexing_path,folder_to_save_data, dataframe_simulated_cell, ssas_multiplexing):
    # This function compresses and saves the data in the correct repository
    security_testing = multiplexing_path.parents[0].exists()
    # testing if the parent path exist
    if security_testing == True:
        if not multiplexing_path.exists():
            #multiplexing_path.mkdir()
            os.makedirs(multiplexing_path)
        else:
            shutil.rmtree(multiplexing_path)
            #multiplexing_path.mkdir()
            os.makedirs(multiplexing_path)
        # saving the dataframe
        dataframe_simulated_cell.to_csv( multiplexing_path.joinpath('multiplexing_csv.csv'), float_format="%.2f")
        # saving the ssa
        np.save(multiplexing_path.joinpath('ssas_multiplexing.npy') , ssas_multiplexing)
        # creating zip
        shutil.make_archive(multiplexing_path, 'zip', multiplexing_path.parents[0],folder_to_save_data)
        shutil.rmtree(multiplexing_path)
    else:
        print ('The folder does not exist')


frame_selection_empty_video = 'constant' # Options are: 'constant' , 'shuffle' and 'loop'
# Simulation_0
multiplexing_path,folder_to_save_data, dataframe_simulated_cell, ssas_multiplexing = simulate_multiplexing( path_files,
                                                                                                            masks_dir,
                                                                                                            list_elongation_rates,
                                                                                                            list_initiation_rates,
                                                                                                            list_diffusion_coefficients,
                                                                                                            multiplexing_path_folder,
                                                                                                            frame_selection_empty_video,
                                                                                                            simulated_RNA_intensities_method)


#dataframe_simulated_cell = reduce_dataframe(dataframe_simulated_cell_complete)
save_data (multiplexing_path,folder_to_save_data, dataframe_simulated_cell, ssas_multiplexing)
