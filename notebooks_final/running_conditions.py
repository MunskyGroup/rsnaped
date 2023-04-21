#!/usr/bin/env python3
# -*- coding: utf-8 -*-
########################################################
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
NUMBER_OF_CORES = int(multiprocessing.cpu_count())
sns.set(font_scale = 1.5)
sns.set_style("white")
selected_color = '#1C00FE'
########################################################
# Defining directories
current_dir = pathlib.Path().absolute()
sequences_dir = current_dir.parents[0].joinpath('DataBases','gene_files')
video_dir = current_dir.parents[0].joinpath('DataBases','videos_for_sim_cell')
rsnaped_dir = current_dir.parents[0].joinpath('rsnaped')
gene_file = current_dir.parents[0].joinpath('DataBases','gene_files','KDM5B_withTags.txt')
masks_dir = current_dir.parents[0].joinpath('DataBases','masks_for_sim_cell')
########################################################
# Importing rSNAPed
sys.path.append(str(rsnaped_dir))
import rsnaped as rsp
########################################################
number_of_simulated_cells = 12
number_repetitions_for_statistics = 8
number_conditions = 8
number_ssa = 5000
variable_range_0 = np.linspace(start=30,stop=200,num=number_conditions).astype(int)  # number_spots
variable_range_1 = np.round(np.linspace(start=0.2,stop=2,num=number_conditions),2)  # SNR
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
initiation_rate = 0.03
intensity_scale = 1 # This variable defines an approximated SNR
intensity_scale_ch0 = intensity_scale
intensity_scale_ch1 = intensity_scale
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
########################################################

########################################################
_,ssa_ump,_,_ = rsp.SSA_rsnapsim(gene_file = gene_file, ke = elongation_rate, ki = initiation_rate, frames = simulation_time_in_sec,frame_rate = 1,n_traj = number_ssa,).simulate() 
ssa_trajectories_timePoint = ssa_ump.flatten()
ssa_trajectories_timePoint_normalized = (ssa_trajectories_timePoint-np.amin(ssa_trajectories_timePoint))/ (np.amax(ssa_trajectories_timePoint)-np.amin(ssa_trajectories_timePoint))
########################################################


########################################################
simulations_directory =  current_dir.joinpath('temp_simulation')
save_to_dir =  current_dir.joinpath('temp_images' )
rsp.Utilities.test_if_directory_exist_if_not_create(simulations_directory,remove_if_already_exist=True)
rsp.Utilities.test_if_directory_exist_if_not_create(current_dir.joinpath('temp_processing'),remove_if_already_exist=True)
rsp.Utilities.test_if_directory_exist_if_not_create(save_to_dir,remove_if_already_exist=True)
########################################################

########################################################
def running_conditions_simulated_cell(number_spots_per_cell,intensity_scale,diffusion_coefficient,simulation_time_in_sec):
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
                                                                save_dataframe = False, 
                                                                frame_selection_empty_video=frame_selection_empty_video,
                                                                spot_size = spot_size,
                                                                spot_sigma = spot_sigma,
                                                                intensity_scale_ch0 = intensity_scale,
                                                                intensity_scale_ch1 = intensity_scale,
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
########################################################

########################################################
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
########################################################

########################################################
def image_processing_conditions (number_repetitions_for_statistics,
                                number_spots_per_cell,
                                intensity_scale,
                                diffusion_coefficient,
                                simulation_time_in_sec,
                                ssa_trajectories_timePoint_normalized,
                                iterable_value,
                                case):
    if case == 0:
        number_spots_per_cell = iterable_value
    if case == 1:
        intensity_scale = iterable_value
    if case == 2:
        diffusion_coefficient = iterable_value
    if case == 3:
        simulation_time_in_sec = iterable_value
    vector_KD = np.zeros((number_repetitions_for_statistics))
    simulation_outputs = Parallel(n_jobs = NUMBER_OF_CORES)(delayed(running_conditions_simulated_cell)
                                                            (number_spots_per_cell=number_spots_per_cell,
                                                            intensity_scale=intensity_scale,
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
        intensities_tracking_complete =  rsp.Utilities.extract_field_from_dataframe(dataframe=list_DataFrame_tracking[i],selected_time=simulation_time_in_sec-1,selected_field='ch1_int_mean')
        intensities_tracking = rsp.Utilities.remove_extrema_in_vector(intensities_tracking_complete ,max_percentile = 98)
        intensities_tracking_normalized = (intensities_tracking-np.min(intensities_tracking))/ (np.max(intensities_tracking)-np.min(intensities_tracking))  
        data1 = ssa_trajectories_timePoint_normalized
        data2 = intensities_tracking_normalized
        vector_KD[i] = scipy.stats.kstest(data1,data2).statistic
    ks_dist_mean = np.round(np.mean(vector_KD),2)
    ks_dist_std = np.round(np.std(vector_KD),2)
    return ks_dist_mean , ks_dist_std 
########################################################

########################################################
def plots_conditions(variable_range,ks_dist_mean_vector,ks_dist_std_vector,save_to_dir,plot_name='',label_x='' , extend_x_range= False,use_log_scale=False,selected_color='#1C00FE'):
    plt.figure(figsize=(5, 4))
    plt.errorbar(variable_range, ks_dist_mean_vector,  yerr=ks_dist_std_vector, ecolor=selected_color,linestyle='')
    plt.plot(variable_range, ks_dist_mean_vector, marker='o', markersize=12, linestyle='none',color=selected_color )
    #plt.title(plot_name+' ('+ str(number_of_simulated_cells) + ' Cells)')
    plt.ylabel('KD (SSA-Image)')
    plt.xlabel(label_x)
    plt.xlabel(r'${}$'.format(label_x))
    if extend_x_range == True:
        plt.xlim(variable_range[0]-2,variable_range[-1]+2)
    if use_log_scale ==True:
        plt.xscale('log')
    plt.savefig(save_to_dir.joinpath(plot_name+'_KD.pdf'), transparent=False,dpi=1200, bbox_inches = 'tight', format='pdf')
    plt.show()
    return
########################################################




########################################################
def running_conditions (simulations_directory,
                        save_to_dir,
                        case,
                        variable_range,
                        plot_name,
                        label_x,
                        extend_x_range,
                        use_log_scale,
                        number_repetitions_for_statistics,
                        number_spots_per_cell,
                        intensity_scale,
                        diffusion_coefficient,
                        simulation_time_in_sec,
                        ssa_trajectories_timePoint_normalized):
    ks_dist_mean_vector = np.zeros(number_conditions)
    ks_dist_std_vector = np.zeros(number_conditions)
    # Thins loops replaces the condition to test. iterable_value replaces the value according to the selected case.
    # The cases are: 
                    # if case == 0:
                    #     number_spots_per_cell = iterable_value
                    # if case == 1:
                    #     intensity_scale = iterable_value
                    # if case == 2:
                    #     diffusion_coefficient = iterable_value
                    # if case == 3:
                    #     simulation_time_in_sec = iterable_value
    for j,iterable_value in enumerate(variable_range):
        ks_dist_mean_vector[j] , ks_dist_std_vector[j]= image_processing_conditions (number_repetitions_for_statistics,
                                                                                number_spots_per_cell=number_spots_per_cell,
                                                                                intensity_scale=intensity_scale,
                                                                                diffusion_coefficient=diffusion_coefficient,
                                                                                simulation_time_in_sec=simulation_time_in_sec,
                                                                                ssa_trajectories_timePoint_normalized=ssa_trajectories_timePoint_normalized,
                                                                                iterable_value=iterable_value,
                                                                                case=case)
    np.save(save_to_dir.joinpath('variable_range_'+str(case)+'.npy'),variable_range)
    np.save(save_to_dir.joinpath('ks_dist_mean_vector_'+str(case)+'.npy'),ks_dist_mean_vector)
    np.save(save_to_dir.joinpath('ks_dist_std_vector_'+str(case)+'.npy'),ks_dist_std_vector)
    plots_conditions(variable_range=variable_range,
                    ks_dist_mean_vector=ks_dist_mean_vector,
                    ks_dist_std_vector=ks_dist_std_vector,
                    save_to_dir=save_to_dir,
                    plot_name=plot_name,
                    label_x=label_x,
                    extend_x_range=extend_x_range,
                    use_log_scale=use_log_scale)
    rsp.Utilities.test_if_directory_exist_if_not_create(simulations_directory,remove_if_already_exist=True)
########################################################


########################################################
# condition 0 : number_spots_per_cell
running_conditions (simulations_directory,
                    save_to_dir,
                    case=0,
                    variable_range=variable_range_0,
                    plot_name='plot_spot_density',
                    label_x='No Spots / Cell',
                    extend_x_range=True,
                    use_log_scale=False,
                    number_repetitions_for_statistics=number_repetitions_for_statistics,
                    number_spots_per_cell=number_spots_per_cell,
                    intensity_scale=intensity_scale,
                    diffusion_coefficient=diffusion_coefficient,
                    simulation_time_in_sec=simulation_time_in_sec,
                    ssa_trajectories_timePoint_normalized=ssa_trajectories_timePoint_normalized)
########################################################
# condition 1 : intensity_scale
running_conditions (simulations_directory,
                    save_to_dir,
                    case=1,
                    variable_range=variable_range_1,
                    plot_name='plot_SNR',
                    label_x='SNR',
                    extend_x_range=False,
                    use_log_scale=False,
                    number_repetitions_for_statistics=number_repetitions_for_statistics,
                    number_spots_per_cell=number_spots_per_cell,
                    intensity_scale=intensity_scale,
                    diffusion_coefficient=diffusion_coefficient,
                    simulation_time_in_sec=simulation_time_in_sec,
                    ssa_trajectories_timePoint_normalized=ssa_trajectories_timePoint_normalized)
########################################################
# condition 2 : intensity_scale
running_conditions (simulations_directory,
                    save_to_dir,
                    case=2,
                    variable_range=variable_range_2,
                    plot_name='plot_diff',
                    label_x='D (\mu$m$^2/s) ',
                    extend_x_range=False,
                    use_log_scale=True,
                    number_repetitions_for_statistics=number_repetitions_for_statistics,
                    number_spots_per_cell=number_spots_per_cell,
                    intensity_scale=intensity_scale,
                    diffusion_coefficient=diffusion_coefficient,
                    simulation_time_in_sec=simulation_time_in_sec,
                    ssa_trajectories_timePoint_normalized=ssa_trajectories_timePoint_normalized)
########################################################
# condition 3 : intensity_scale
running_conditions (simulations_directory,
                    save_to_dir,
                    case=3,
                    variable_range=variable_range_3,
                    plot_name='plot_frames',
                    label_x='Frames (s)',
                    extend_x_range=True,
                    use_log_scale=False,
                    number_repetitions_for_statistics=number_repetitions_for_statistics,
                    number_spots_per_cell=number_spots_per_cell,
                    intensity_scale=intensity_scale,
                    diffusion_coefficient=diffusion_coefficient,
                    simulation_time_in_sec=simulation_time_in_sec,
                    ssa_trajectories_timePoint_normalized=ssa_trajectories_timePoint_normalized)
########################################################

