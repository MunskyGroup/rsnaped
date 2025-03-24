# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 14:07:38 2025

@author: willi
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib

# cwd = os.getcwd()
# os.chdir('..')
# import rsnaped as rsp
# os.chdir(cwd)


import sys

current_dir = pathlib.Path().absolute()
sequences_dir = current_dir.parents[0].joinpath('DataBases','gene_files')
video_dir = current_dir.parents[0].joinpath('DataBases','videos_for_sim_cell')
rsnaped_dir = current_dir.parents[0].joinpath('rsnaped')
gene_file = current_dir.parents[0].joinpath('DataBases','gene_files','KDM5B_withTags.txt')
masks_dir = current_dir.parents[0].joinpath('DataBases','masks_for_sim_cell')

sys.path.append(str(rsnaped_dir))
import rsnaped as rsp

image = np.zeros([512,512],dtype=np.uint16);
spots_xy = np.random.randint(100,400,size=[100,2])
values_xy = np.random.randint(100,400,size=[100])
sigma = 5
spot_size = 11
intensity_scale = 1

frame = rsp.Frame2D().make(spots_xy, values_xy, spot_size, sigma,
         image, intensity_scale,)

plt.matshow(frame)




n_spots = 3
step_size = 1
t = np.linspace(0,99,100)
n_times = len(t)
initial_spots = np.random.randint(200,300,size=(2,1,n_spots))
diffusion_coefficient = np.random.pareto(1,size=(n_times,n_spots))
diffusion_coefficient = np.array([diffusion_coefficient, diffusion_coefficient])
isfloat_D = False
if isinstance(diffusion_coefficient, float):
    brownian_movement = np.sqrt(2*diffusion_coefficient*step_size)
    isfloat_D = True
else:
    brownian_movement = np.sqrt(2*diffusion_coefficient*step_size)
    
movement = np.random.randn(*brownian_movement.shape)*brownian_movement
trajs = initial_spots + np.cumsum(movement,axis=1)
plt.plot(*trajs.T)