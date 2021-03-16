#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 10:52:17 2021

@author: luisub
"""

class Trackpy():
    '''
    This class is intended to detect spots in the video by using **Trackpy**. 

    Parameters
    ----------
    video : NumPy array 
        Array of images with dimensions [T,Y,X,C]. In case a FISH image is used, the format must be [Z,Y,X,C], and the user must specify the parameter FISH_image =1.
    mask : NumPy array, with Boolean values, where 1 represents the masked area, and 0 represents the area outside the mask. 
        An array of images with dimensions [Y,X].
    particle_size : int, optional
        Average particle size. The default is 5.
    selected_channel : int, optional
        Channel where the particles are detected and tracked. The default is 0.
    minimal_frames : int, optional
        This parameter defines the minimal number of frames that a particle should appear on the video to be considered on the final count. The default is 5.
    optimization_iterations : int, optional
        Number of iterations for the optimization process to select the best filter. The default is 30.
    use_defaul_filter : bool, optional
        This option allows the user to use a default filter if TRUE. Else, it uses an optimization process to select the best filter for the image. The default is= 1.
    show_plot : bool, optional
        Allows the user to show a plot for the optimization process. The default is 1.
    FISH_image : bool, optional.
        This parameter allows the user to use FISH images and connect spots detected along multiple z-slices. The default is 0.
    '''
    def __init__(self, video, mask, particle_size =5, selected_channel=0, minimal_frames=5,optimization_iterations =30,use_defaul_filter = 1, FISH_image = 0, show_plot =1):
        self.num_cores = multiprocessing.cpu_count()
        self.time_points = video.shape[0]
        self.selected_channel = selected_channel
        
        # function that remove outliers from the video
        video = RemoveExtrema(video, min_percentile=0, max_percentile=99,ignore_channel =None).remove_outliers()
        
        # Function to convert the video to uint
        def img_uint(image):
            temp_vid= img_as_uint(image)
            return temp_vid
        init_video = Parallel(n_jobs=self.num_cores)(delayed(img_uint)(video[i,:,:,self.selected_channel]) for i in range(0,self.time_points)) 
        self.video = np.asarray(init_video)
        
        self.video_complete = video.copy()
        self.mask = mask
        if (particle_size % 2) ==0:
            particle_size = particle_size + 1
            print('particle_size must be an odd number, this was automatically changed to: ', particle_size)
        self.particle_size = particle_size # according to the documentation must be an even number 3,5,7,9 etc.
        
        if self.time_points< 10: 
            self.min_time_particle_vanishes = 0
            self.max_distance_particle_moves = 5
        else:
            self.min_time_particle_vanishes = 1
            self.max_distance_particle_moves = 7
            
        if minimal_frames> self.time_points:     # this line is making sure that "minimal_frames" is always less or equal than the total number of frames
            minimal_frames = self.time_points
        self.minimal_frames = minimal_frames
        self.optimization_iterations = optimization_iterations 
        self.show_plot = show_plot
        self.use_defaul_filter =  use_defaul_filter
        # parameters for the filters
        self.low_pass_filter = 0.5
        self.default_highpass = 20
        self.gaussian_filter_sigma = 0.5
        self.median_filter_size = 5
        self.perecentile_intensity_slection = 95
        self.max_highpass = 10
        self.min_highpass = 0.1
        
        self.default_threshold_int_std =0.5
        
        # This section detects if a FISH image is passed and it adjust accordingly.
        self.FISH_image = FISH_image
        if self.FISH_image ==1:
            self.min_time_particle_vanishes = 0
            self.max_distance_particle_moves = 1
            self.minimal_frames = minimal_frames
    def perform_tracking(self):
        '''
        This method 

        Returns
        -------
        trackpy_dataframe : pandas data frame.
            Pandas data frame from trackpy with fields [x, y, mass, size, ecc, signal, raw_mass, ep, frame, particle].
        number_particles : int.
            The total number of detected particles in the data frame.
        video_filtered : np.uint16.
            Filtered video resulting from the bandpass process. Array with format [T,Y,X,C].
        '''
        #NUM_STD = 1
        num_detected_particles = np.zeros((self.optimization_iterations),dtype=np.float)
        #vector_highpass_filters = np.linspace(self.min_highpass,self.max_highpass,self.optimization_iterations).astype(np.uint16)
        min_int_vector = np.linspace(0,0.5,self.optimization_iterations) # range of std to test for optimization
        percentile = self.perecentile_intensity_slection
        # Funtions with the bandpass and gaussian filters
        def bandpass_filter (image, lowfilter, highpass):
            temp_vid= difference_of_gaussians(image, lowfilter, highpass,truncate=3.0)
            return img_as_uint(temp_vid)
        def gaussian_filter(image, sigma=0.1):
            temp_image= img_as_float64(image)
            filtered_image = gaussian(temp_image, sigma=sigma, output=None, mode='nearest', cval=0, multichannel=None, preserve_range=True, truncate=4.0)
            return img_as_uint(filtered_image)
        # non-linear filter
        def nl_filter(image):
            temp_image= img_as_float64(image)
            sigma_est = np.mean(estimate_sigma(temp_image, multichannel=True))
            denoise_img = denoise_nl_means(temp_image, h=sigma_est, fast_mode=True,patch_size=10, patch_distance=3, multichannel=False)
            return img_as_uint(denoise_img)
        
        def median_filter (image, size=1):
            temp_image= img_as_float64(image)
            filtered_image = nd.median_filter(temp_image,size=size)
            return img_as_uint(filtered_image)
        
        if self.use_defaul_filter ==1: # This section uses a default value for the filter size.
            temp_vid_dif_filter = Parallel(n_jobs=self.num_cores)(delayed(bandpass_filter)(self.video[i,:,:],self.low_pass_filter,self.default_highpass) for i in range(0,self.time_points)) 
            temp_video_bp_filtered = np.asarray(temp_vid_dif_filter)
            video_removed_mask = np.einsum('ijk,jk->ijk', temp_video_bp_filtered, self.mask)
            f_init = tp.locate(video_removed_mask[0,:,:], self.particle_size, minmass=0, max_iterations=100,preprocess=False, percentile=percentile) 
            #min_int_in_video = np.mean(f_init.mass.values) + self.default_threshold_int_std*np.std(f_init.mass.values)
            min_int_in_video = np.amax( (0, np.mean(f_init.mass.values) + self.default_threshold_int_std *np.std(f_init.mass.values)))
            f = tp.batch(video_removed_mask[:,:,:],self.particle_size, minmass=min_int_in_video, processes='auto',max_iterations=1000,preprocess=False, percentile=percentile)
            t = tp.link_df(f,(self.max_distance_particle_moves,self.max_distance_particle_moves), memory=self.min_time_particle_vanishes, adaptive_stop = 1,link_strategy='auto') # tp.link_df(data_frame, min_distance_particle_moves, min_time_particle_vanish). 
            t_sel = tp.filter_stubs(t, self.minimal_frames)  # selecting trajectories that appear in at least 10 frames.
            number_particles = t_sel['particle'].nunique()
            trackpy_dataframe = t_sel
        
        else: # This section uses optimization to select the optimal value for the filter size.
            #for index_p,highpass_filters in enumerate(vector_highpass_filters):
            for index_p,int_optimization_value in enumerate(min_int_vector):
                try:
                    #temp_vid_dif_filter = Parallel(n_jobs=self.num_cores)(delayed(bandpass_filter)(self.video[i,:,:],self.low_pass_filter,highpass_filters) for i in range(0,self.time_points)) 
                    temp_vid_dif_filter = Parallel(n_jobs=self.num_cores)(delayed(bandpass_filter)(self.video[i,:,:],self.low_pass_filter,self.default_highpass) for i in range(0,self.time_points))
                    temp_video_bp_filtered = np.asarray(temp_vid_dif_filter)
                    video_removed_mask = np.einsum('ijk,jk->ijk', temp_video_bp_filtered, self.mask)
                    f_init = tp.locate(video_removed_mask[0,:,:], self.particle_size, minmass=0, max_iterations=100,preprocess=False, percentile=percentile) 
                    #print(int_optimization_value)
                    #### OPTIMIZATION VARIABLE:  min_int_in_video
                    min_int_in_video = np.amax( (0,  np.mean(f_init.mass.values) + int_optimization_value *np.std(f_init.mass.values)))
                    f = tp.batch(video_removed_mask[:,:,:],self.particle_size, minmass=min_int_in_video, processes='auto',max_iterations=1000,preprocess=False, percentile=percentile)
                    t = tp.link_df(f,(self.max_distance_particle_moves,self.max_distance_particle_moves), memory=self.min_time_particle_vanishes, adaptive_stop = 1,link_strategy='auto') # tp.link_df(data_frame, min_distance_particle_moves, min_time_particle_vanish). 
                    t_sel = tp.filter_stubs(t, self.minimal_frames)  # selecting trajectories that appear in at least 10 frames.
                    num_detected_particles[index_p] = t_sel['particle'].nunique()
                    #print(num_detected_particles[index_p])
                except:
                     num_detected_particles[index_p]  = 0
            if self.optimization_iterations <=10:
                window_size = 2
            elif (self.optimization_iterations >10) and (self.optimization_iterations <= 20):
                window_size = 3
            else:
                window_size = 4
            conv_num_detected_particles = np.round( np.convolve(num_detected_particles, np.ones(window_size)/window_size, mode='same'))
            try:
                threshold = 3 # threshold to reject if 
                num_detected_particles_with_nans = conv_num_detected_particles.copy()
                num_detected_particles_with_nans[num_detected_particles_with_nans<=threshold]=np.nan
                mode_detected_particles = sps.mode(num_detected_particles_with_nans, nan_policy='omit')[0][0] # calculaitng the mode of nonzero num_detected_particles.
                center_modes = int (len(np.where(num_detected_particles_with_nans == mode_detected_particles)[0])/2)-1 # 
                if center_modes<0:
                    center_modes =0
                index_containin_central_mode = np.where(num_detected_particles_with_nans == mode_detected_particles)[0][center_modes]
                selected_int_optimized = min_int_vector[index_containin_central_mode] 
                
                #print(selected_int_optimized)
                
                #selected_filter = vector_highpass_filters[index_containin_central_mode] # selecting the intensity for the first instance where the mode appears in the vector_intensities
                temp_vid_dif_filter = Parallel(n_jobs=self.num_cores)(delayed(bandpass_filter)(self.video[i,:,:],self.low_pass_filter,self.default_highpass) for i in range(0,self.time_points))
                #temp_vid_dif_filter = Parallel(n_jobs=self.num_cores)(delayed(bandpass_filter)(self.video[i,:,:],self.low_pass_filter,selected_filter) for i in range(0,self.time_points)) 
                temp_video_bp_filtered = np.asarray(temp_vid_dif_filter)
                #selected_threshold_int = 
                video_removed_mask = np.einsum('ijk,jk->ijk', temp_video_bp_filtered, self.mask)
                f_init = tp.locate(video_removed_mask[0,:,:], self.particle_size, minmass=0, max_iterations=100,preprocess=False, percentile=percentile) 
                #### OPTIMIZATION VARIABLE:  min_int_in_video
                
                #min_int_in_video = np.mean(f_init.mass.values) + selected_int_optimized*np.std(f_init.mass.values)
                min_int_in_video = np.amax( (0,  np.mean(f_init.mass.values) + selected_int_optimized *np.std(f_init.mass.values)))
                f = tp.batch(video_removed_mask[:,:,:],self.particle_size, minmass=min_int_in_video, processes='auto',max_iterations=1000,preprocess=False, percentile=percentile)
                t = tp.link_df(f,(self.max_distance_particle_moves,self.max_distance_particle_moves), memory=self.min_time_particle_vanishes, adaptive_stop = 1,link_strategy='auto') # tp.link_df(data_frame, min_distance_particle_moves, min_time_particle_vanish). 
                t_sel = tp.filter_stubs(t, self.minimal_frames)  # selecting trajectories that appear in at least 10 frames.
                number_particles = t_sel['particle'].nunique()
                trackpy_dataframe = t_sel
            except:
                number_particles = 0
                trackpy_dataframe = []
                self.show_plot = 0
                selected_int_optimized =0
            
            if self.show_plot ==1:
                plt.figure(figsize=(7,7))
                plt.plot(min_int_vector, num_detected_particles, 'w-', linewidth=2, label = 'real')
                plt.plot(min_int_vector[1:-1], conv_num_detected_particles[1:-1], 'g-', linewidth = 1, label = 'smoothed')
                plt.plot([selected_int_optimized,selected_int_optimized],[np.amin(num_detected_particles),np.amax(num_detected_particles)],'r-',linewidth=2,label='Automatic selection')
                #plt.plot([vector_intensities[0],vector_intensities[-1]],[threshold,threshold],'k--',linewidth=1,label='min. selection threshold')
                plt.legend(loc='best')
                plt.xlabel('minimal value (int.)')
                plt.ylabel('Detected Spots')
                plt.title('Optimization: detecting the longest plateau')
                plt.show()
                print('')
                print('The number of detected trajectories is: ', number_particles)
                print('')
                print('')
        video_filtered =  self.video_complete.copy()
        video_filtered[:,:,:,self.selected_channel] = video_removed_mask
        return trackpy_dataframe, int(number_particles), video_filtered