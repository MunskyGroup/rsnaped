#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 11:21:52 2021

@author: luisub
"""

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
        if not (type(list_selected_particles_dataframe) is list):
            list_selected_particles_dataframe = [list_selected_particles_dataframe] 
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
        if ( self.list_selected_particles_dataframe[0] is None):
            NUM_COLUMNS = 3
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
            NUM_COLUMNS = 3
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
                
                
                
                real_selected_particles_dataframe = self.list_real_particle_positions[counter]
                if not len (self.list_real_particle_positions[counter]) == 0:
                    number_particles = real_selected_particles_dataframe['particle'].nunique()
                    for k in range (0,number_particles):
                        frames_part =real_selected_particles_dataframe.loc[real_selected_particles_dataframe['particle']==real_selected_particles_dataframe['particle'].unique()[k]].frame.values  
                        index_time = self.selected_timepoint
                        if index_time in frames_part: # plotting the circles for each detected particle at a given time point
                            index_val=np.where(frames_part == index_time)
                            x_pos=int(real_selected_particles_dataframe.loc[real_selected_particles_dataframe['particle']==real_selected_particles_dataframe['particle'].unique()[k]].x.values[index_val])
                            y_pos=int(real_selected_particles_dataframe.loc[real_selected_particles_dataframe['particle']==real_selected_particles_dataframe['particle'].unique()[k]].y.values[index_val])
                        try:
                            circle = plt.Circle((x_pos, y_pos), self.particle_size/2, color='yellow', fill=False)
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
            NUM_COLUMNS = 3
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
                ax.set(title=title_str + ' Filtered' )
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