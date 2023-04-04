# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
import scipy
import pandas as pd
import pathlib
import sys
import scipy.stats as stats
import matplotlib.pyplot as plt 
import seaborn as sns
from joblib import Parallel, delayed
import multiprocessing
import uuid
#NUMBER_OF_CORES = np.max((1,int(multiprocessing.cpu_count()/2)))
NUMBER_OF_CORES = int(multiprocessing.cpu_count()/2)
sns.set(font_scale = 1.5)
sns.set_style("white")

# Defining directories
current_dir = pathlib.Path().absolute()
sequences_dir = current_dir.parents[0].joinpath('DataBases','gene_files')
video_dir = current_dir.parents[0].joinpath('DataBases','videos_for_sim_cell')
rsnaped_dir = current_dir.parents[0].joinpath('rsnaped')
gene_file = current_dir.parents[0].joinpath('DataBases','gene_files','KDM5B_withTags.txt')
masks_dir = current_dir.parents[0].joinpath('DataBases','masks_for_sim_cell')

# Importing rSNAPed
sys.path.append(str(rsnaped_dir))
import rsnaped as rsp

########################################################
number_of_simulated_cells = 20
number_repetitions_for_statistics = 8
number_conditions = 6
number_ssa = 5000
variable_range_0 = np.linspace(start=30,stop=80,num=number_conditions).astype(int)  # number_spots
variable_range_1 = np.round(np.linspace(start=0.5,stop=4,num=number_conditions),2)  # SNR
variable_range_2 = np.round(np.logspace(np.log10(0.01), np.log10(0.5), num=number_conditions),3) # k_diff
variable_range_3 = np.linspace(start=20,stop=100,num=number_conditions).astype(int)  # simulation_time
################################################################

number_spots_per_cell = 30           # 
simulation_time_in_sec = 30          # 
min_percentage_time_tracking = 0.3   # (normalized) minimum time to consider a trajectory.
average_cell_diameter = 400
diffusion_coefficient = 0.1          # 
microns_per_pixel = 0.13
spot_size = 7 # spot size for the simulation and tracking.
spot_sigma = 1.5
elongation_rate = 10
initiation_rate = 0.01
intensity_scale_ch0 = 1
intensity_scale_ch1 = 1
intensity_scale_ch2 = None
simulated_RNA_intensities_method = 'random'
frame_selection_empty_video = 'gaussian' # Options are: 'constant' , 'shuffle' and 'loop' 'linear_interpolation', 'gaussian', 'poisson'
dataframe_format = 'short' # 'short'  'long'
store_videos_in_memory = False
select_background_cell_index = None # Integer in range 0 to 8, or use None to select a random value. 
perform_video_augmentation = True 
basal_intensity_in_background_video = 10000
scale_intensity_in_base_video=False
# Parameters for pipeline tracking
intensity_calculation_method = 'disk_donut'  # options are : 'total_intensity' and 'disk_donut' 'gaussian_fit'
mask_selection_method = 'max_area' # options are : 'max_spots' and 'max_area' 
use_optimization_for_tracking = 1 # 0 not using, 1 is using optimization
selected_channel_tracking = 0
selected_channel_segmentation = 1
particle_detection_size = spot_size


_,ssa_ump,_,_ = rsp.SSA_rsnapsim(gene_file = gene_file, ke = elongation_rate, ki = initiation_rate, frames = simulation_time_in_sec,frame_rate = 1,n_traj = number_ssa,).simulate() 
ssa_trajectories_timePoint = ssa_ump.flatten() #ssa_trajectories[:,:,:,:].flatten()
ssa_trajectories_timePoint_normalized = (ssa_trajectories_timePoint-np.amin(ssa_trajectories_timePoint))/ (np.amax(ssa_trajectories_timePoint)-np.amin(ssa_trajectories_timePoint))


# def running_conditions_simulated_cell(number_spots_per_cell,intensity_scale_ch1,diffusion_coefficient,simulation_time_in_sec):
#     _, list_masks, _, _, _, _, video_path, _ = rsp.simulate_cell( video_dir, 
#                                                                 list_gene_sequences = gene_file,
#                                                                 list_number_spots= number_spots_per_cell,
#                                                                 list_target_channels_proteins = 1,
#                                                                 list_target_channels_mRNA = 0, 
#                                                                 list_diffusion_coefficients=diffusion_coefficient,
#                                                                 list_elongation_rates=elongation_rate,
#                                                                 list_initiation_rates=initiation_rate,
#                                                                 masks_dir=masks_dir, 
#                                                                 list_label_names=1,
#                                                                 number_cells = number_of_simulated_cells,
#                                                                 simulation_time_in_sec = simulation_time_in_sec,
#                                                                 step_size_in_sec = 1,
#                                                                 save_as_gif = False,
#                                                                 frame_selection_empty_video=frame_selection_empty_video,
#                                                                 spot_size = spot_size,
#                                                                 spot_sigma = spot_sigma,
#                                                                 intensity_scale_ch0 = intensity_scale_ch0,
#                                                                 intensity_scale_ch1 = intensity_scale_ch1,
#                                                                 intensity_scale_ch2 = intensity_scale_ch2,
#                                                                 dataframe_format = dataframe_format,
#                                                                 simulated_RNA_intensities_method=simulated_RNA_intensities_method,
#                                                                 store_videos_in_memory= store_videos_in_memory,
#                                                                 scale_intensity_in_base_video=scale_intensity_in_base_video,
#                                                                 basal_intensity_in_background_video=basal_intensity_in_background_video,
#                                                                 microns_per_pixel=microns_per_pixel,
#                                                                 select_background_cell_index=select_background_cell_index,
#                                                                 perform_video_augmentation=perform_video_augmentation)
#     return list_masks,video_path

# def simulation_and_tracking(number_spots_per_cell=number_spots_per_cell,
#                             intensity_scale_ch1=intensity_scale_ch1,
#                             diffusion_coefficient=diffusion_coefficient,
#                             simulation_time_in_sec=simulation_time_in_sec,
#                             particle_detection_size=particle_detection_size,
#                             selected_channel_tracking = selected_channel_tracking,
#                             selected_channel_segmentation = selected_channel_segmentation,
#                             intensity_calculation_method =intensity_calculation_method, 
#                             mask_selection_method = mask_selection_method,
#                             use_optimization_for_tracking=use_optimization_for_tracking,
#                             average_cell_diameter=average_cell_diameter,
#                             min_percentage_time_tracking=min_percentage_time_tracking,
#                             dataframe_format=dataframe_format):
#     list_masks, video_path = running_conditions_simulated_cell(number_spots_per_cell=number_spots_per_cell,
#                                                                 intensity_scale_ch1=intensity_scale_ch1,
#                                                                 diffusion_coefficient=diffusion_coefficient,
#                                                                 simulation_time_in_sec=simulation_time_in_sec)
#     # processing simulated cell
#     list_DataFrame_tracking, _, _, _ = rsp.image_processing(files_dir_path_processing = video_path,
#                                                             list_masks = list_masks,
#                                                             particle_size=particle_detection_size,
#                                                             selected_channel_tracking = selected_channel_tracking,
#                                                             selected_channel_segmentation = selected_channel_segmentation,
#                                                             intensity_calculation_method =intensity_calculation_method, 
#                                                             mask_selection_method = mask_selection_method,
#                                                             show_plot=False,
#                                                             use_optimization_for_tracking=use_optimization_for_tracking,
#                                                             real_positions_dataframe = None,
#                                                             average_cell_diameter=average_cell_diameter,
#                                                             print_process_times=False,
#                                                             min_percentage_time_tracking=min_percentage_time_tracking,
#                                                             dataframe_format=dataframe_format)
#     DataFrame_particles_intensities_tracking_merged = pd.concat(list_DataFrame_tracking)
#     return DataFrame_particles_intensities_tracking_merged



# def image_processing_conditions (number_repetitions_for_statistics,number_spots_per_cell,intensity_scale_ch1,diffusion_coefficient,simulation_time_in_sec,ssa_trajectories_timePoint_normalized):
#     vector_KD = np.zeros((number_repetitions_for_statistics))
#     # running the cell simulation and particle tracking
#     DataFrame_particles_intensities_tracking_merged = Parallel(n_jobs = NUMBER_OF_CORES)(delayed(simulation_and_tracking)(number_spots_per_cell=number_spots_per_cell,
#                                                                             intensity_scale_ch1=intensity_scale_ch1,
#                                                                             diffusion_coefficient=diffusion_coefficient,
#                                                                             simulation_time_in_sec=simulation_time_in_sec,
#                                                                             particle_detection_size=particle_detection_size,
#                                                                             selected_channel_tracking = selected_channel_tracking,
#                                                                             selected_channel_segmentation = selected_channel_segmentation,
#                                                                             intensity_calculation_method =intensity_calculation_method, 
#                                                                             mask_selection_method = mask_selection_method,
#                                                                             use_optimization_for_tracking=use_optimization_for_tracking,
#                                                                             average_cell_diameter=average_cell_diameter,
#                                                                             min_percentage_time_tracking=min_percentage_time_tracking,
#                                                                             dataframe_format=dataframe_format) for i in range(0, number_repetitions_for_statistics))
#     for i in range (number_repetitions_for_statistics):
#         # Intensities tracking
#         intensities_tracking =  rsp.Utilities.extract_field_from_dataframe(dataframe=DataFrame_particles_intensities_tracking_merged[i],selected_time=simulation_time_in_sec-1,selected_field='ch1_int_mean')
#         intensities_tracking_normalized = (intensities_tracking-np.min(intensities_tracking))/ (np.max(intensities_tracking)-np.min(intensities_tracking))  
#         # Renaming vectors
#         data1 = ssa_trajectories_timePoint_normalized
#         data2 = intensities_tracking_normalized
#         # Calculating Kolmogorov distance
#         vector_KD[i] = scipy.stats.kstest(data1,data2).statistic
#     # Calculating statistics
#     ks_dist_mean = np.mean(vector_KD)
#     ks_dist_std = np.std(vector_KD)
#     return ks_dist_mean , ks_dist_std 


def running_conditions_simulated_cell(number_spots_per_cell,intensity_scale_ch1,diffusion_coefficient,simulation_time_in_sec):
    _, list_masks, _, _, _, _, video_path, _ = rsp.simulate_cell( video_dir, 
                                                                list_gene_sequences = gene_file,
                                                                list_number_spots= number_spots_per_cell,
                                                                list_target_channels_proteins = 1,
                                                                list_target_channels_mRNA = 0, 
                                                                list_diffusion_coefficients=diffusion_coefficient,
                                                                list_elongation_rates=elongation_rate,
                                                                list_initiation_rates=initiation_rate,
                                                                masks_dir=masks_dir, 
                                                                list_label_names=1,
                                                                number_cells = number_of_simulated_cells,
                                                                simulation_time_in_sec = simulation_time_in_sec,
                                                                step_size_in_sec = 1,
                                                                save_as_gif = False,
                                                                frame_selection_empty_video=frame_selection_empty_video,
                                                                spot_size = spot_size,
                                                                spot_sigma = spot_sigma,
                                                                intensity_scale_ch0 = intensity_scale_ch0,
                                                                intensity_scale_ch1 = intensity_scale_ch1,
                                                                intensity_scale_ch2 = intensity_scale_ch2,
                                                                dataframe_format = dataframe_format,
                                                                simulated_RNA_intensities_method=simulated_RNA_intensities_method,
                                                                store_videos_in_memory= store_videos_in_memory,
                                                                scale_intensity_in_base_video=scale_intensity_in_base_video,
                                                                basal_intensity_in_background_video=basal_intensity_in_background_video,
                                                                microns_per_pixel=microns_per_pixel,
                                                                select_background_cell_index=select_background_cell_index,
                                                                perform_video_augmentation=perform_video_augmentation,
                                                                name_folder= 'tem_folder_'+str(uuid.uuid4().hex))
    return list_masks, video_path


def tracking_from_simulation(video_path,
                            list_masks ,
                            particle_detection_size=particle_detection_size,
                            selected_channel_tracking = selected_channel_tracking,
                            selected_channel_segmentation = selected_channel_segmentation,
                            intensity_calculation_method =intensity_calculation_method, 
                            mask_selection_method = mask_selection_method,
                            use_optimization_for_tracking=use_optimization_for_tracking,
                            average_cell_diameter=average_cell_diameter,
                            min_percentage_time_tracking=min_percentage_time_tracking,
                            dataframe_format=dataframe_format):
    list_DataFrame_tracking, _, _, _ = rsp.image_processing(files_dir_path_processing = video_path,
                                                            list_masks = list_masks,
                                                            particle_size=particle_detection_size,
                                                            selected_channel_tracking = selected_channel_tracking,
                                                            selected_channel_segmentation = selected_channel_segmentation,
                                                            intensity_calculation_method =intensity_calculation_method, 
                                                            mask_selection_method = mask_selection_method,
                                                            use_optimization_for_tracking=use_optimization_for_tracking,
                                                            average_cell_diameter=average_cell_diameter,
                                                            min_percentage_time_tracking=min_percentage_time_tracking,
                                                            dataframe_format=dataframe_format)
    DataFrame_particles_intensities_tracking_merged = pd.concat(list_DataFrame_tracking)
    return DataFrame_particles_intensities_tracking_merged


def image_processing_conditions (number_repetitions_for_statistics,number_spots_per_cell,intensity_scale_ch1,diffusion_coefficient,simulation_time_in_sec,ssa_trajectories_timePoint_normalized):
    vector_KD = np.zeros((number_repetitions_for_statistics))
    
    simulation_outputs = Parallel(n_jobs = NUMBER_OF_CORES)(delayed(running_conditions_simulated_cell)
                                                            (number_spots_per_cell=number_spots_per_cell,
                                                            intensity_scale_ch1=intensity_scale_ch1,
                                                            diffusion_coefficient=diffusion_coefficient,
                                                            simulation_time_in_sec=simulation_time_in_sec) for i in range(0, number_repetitions_for_statistics))
    
    list_DataFrame_tracking = Parallel(n_jobs = NUMBER_OF_CORES)(delayed(tracking_from_simulation) 
                                                                        (video_path=simulation_outputs[i][1],
                                                                        list_masks=simulation_outputs[i][0] ,
                                                                        particle_detection_size=particle_detection_size,
                                                                        selected_channel_tracking = selected_channel_tracking,
                                                                        selected_channel_segmentation = selected_channel_segmentation,
                                                                        intensity_calculation_method =intensity_calculation_method, 
                                                                        mask_selection_method = mask_selection_method,
                                                                        use_optimization_for_tracking=use_optimization_for_tracking,
                                                                        average_cell_diameter=average_cell_diameter,
                                                                        min_percentage_time_tracking=min_percentage_time_tracking,
                                                                        dataframe_format=dataframe_format) for i in range(0, number_repetitions_for_statistics))
    
    for i in range (number_repetitions_for_statistics):
        intensities_tracking =  rsp.Utilities.extract_field_from_dataframe(dataframe=list_DataFrame_tracking[i],selected_time=simulation_time_in_sec-1,selected_field='ch1_int_mean')
        intensities_tracking_normalized = (intensities_tracking-np.min(intensities_tracking))/ (np.max(intensities_tracking)-np.min(intensities_tracking))  
        data1 = ssa_trajectories_timePoint_normalized
        data2 = intensities_tracking_normalized
        vector_KD[i] = scipy.stats.kstest(data1,data2).statistic
    ks_dist_mean = np.mean(vector_KD)
    ks_dist_std = np.std(vector_KD)
    return ks_dist_mean , ks_dist_std 



def plots_conditions(variable_range,ks_dist_mean_vector,ks_dist_std_vector,save_to_dir,plot_name='',label_x='' , extend_x_range= False):
    plt.figure(figsize=(5, 5))
    plt.errorbar(variable_range, ks_dist_mean_vector,  yerr=ks_dist_std_vector, ecolor='orangered',linestyle='')
    plt.plot(variable_range, ks_dist_mean_vector, marker='o', markersize=12, linestyle='none',color='orangered' )
    plt.title(plot_name+' ('+ str(number_of_simulated_cells) + ' Cells)')
    plt.ylabel('K-Dist (SSA-Tracking)')
    plt.xlabel(label_x)
    if extend_x_range == True:
        plt.xlim(variable_range[0]-2,variable_range[-1]+2)
    plt.savefig(save_to_dir.joinpath(plot_name+'_KD.pdf'), transparent=False,dpi=1200, bbox_inches = 'tight', format='pdf')
    plt.show()
    return

save_to_dir =  current_dir.joinpath('temp_images' )
rsp.Utilities.test_if_directory_exist_if_not_create(save_to_dir,remove_if_already_exist=False)

ks_dist_mean_vector_0 = np.zeros(number_conditions)
ks_dist_std_vector_0 = np.zeros(number_conditions)
for j,variable_0 in enumerate(variable_range_0):
    ks_dist_mean_vector_0[j] , ks_dist_std_vector_0[j]= image_processing_conditions (number_repetitions_for_statistics,
                                                                            number_spots_per_cell=variable_0,
                                                                            intensity_scale_ch1=intensity_scale_ch1,
                                                                            diffusion_coefficient=diffusion_coefficient,
                                                                            simulation_time_in_sec=simulation_time_in_sec,
                                                                            ssa_trajectories_timePoint_normalized=ssa_trajectories_timePoint_normalized)

plots_conditions(variable_range=variable_range_0,
                ks_dist_mean_vector=ks_dist_mean_vector_0,
                ks_dist_std_vector=ks_dist_std_vector_0,
                save_to_dir=save_to_dir,
                plot_name='spot_density',
                label_x='No Spots / Cell',
                extend_x_range=True)

ks_dist_mean_vector_1 = np.zeros(number_conditions)
ks_dist_std_vector_1 = np.zeros(number_conditions)

for j,variable_1 in enumerate(variable_range_1):
    ks_dist_mean_vector_1[j] , ks_dist_std_vector_1[j]= image_processing_conditions (number_repetitions_for_statistics,
                                                                            number_spots_per_cell=number_spots_per_cell,
                                                                            intensity_scale_ch1=variable_1,
                                                                            diffusion_coefficient=diffusion_coefficient,
                                                                            simulation_time_in_sec=simulation_time_in_sec,
                                                                            ssa_trajectories_timePoint_normalized=ssa_trajectories_timePoint_normalized)

plots_conditions(variable_range=variable_range_1,
                ks_dist_mean_vector=ks_dist_mean_vector_1,
                ks_dist_std_vector=ks_dist_std_vector_1,
                save_to_dir=save_to_dir,
                plot_name='SNR',
                label_x='SNR')

ks_dist_mean_vector_2 = np.zeros(number_conditions)
ks_dist_std_vector_2 = np.zeros(number_conditions)
LL_mean_vector_2 = np.zeros(number_conditions)
LL_std_vector_2 = np.zeros(number_conditions)

for j,variable_2 in enumerate(variable_range_2):
    ks_dist_mean_vector_2[j] , ks_dist_std_vector_2[j] = image_processing_conditions (number_repetitions_for_statistics,
                                                                            number_spots_per_cell=number_spots_per_cell,
                                                                            intensity_scale_ch1=intensity_scale_ch1,
                                                                            diffusion_coefficient=variable_2,
                                                                            simulation_time_in_sec=simulation_time_in_sec,
                                                                            ssa_trajectories_timePoint_normalized=ssa_trajectories_timePoint_normalized)
    
plots_conditions(variable_range=variable_range_2,
                ks_dist_mean_vector=ks_dist_mean_vector_2,
                ks_dist_std_vector=ks_dist_std_vector_2,
                save_to_dir=save_to_dir,
                plot_name='diff',
                label_x='k_D')

ks_dist_mean_vector_3 = np.zeros(number_conditions)
ks_dist_std_vector_3 = np.zeros(number_conditions)
for j,variable_3 in enumerate(variable_range_3):
    ks_dist_mean_vector_3[j] , ks_dist_std_vector_3[j] = image_processing_conditions (number_repetitions_for_statistics,
                                                                            number_spots_per_cell=number_spots_per_cell,
                                                                            intensity_scale_ch1=intensity_scale_ch1,
                                                                            diffusion_coefficient=diffusion_coefficient,
                                                                            simulation_time_in_sec=variable_3,
                                                                            ssa_trajectories_timePoint_normalized=ssa_trajectories_timePoint_normalized)

plots_conditions(variable_range=variable_range_3,
                ks_dist_mean_vector=ks_dist_mean_vector_3,
                ks_dist_std_vector=ks_dist_std_vector_3,
                save_to_dir=save_to_dir,
                plot_name='frames',
                label_x='Frames',
                extend_x_range=True)
