# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 14:07:38 2025

@author: willi
"""

import numpy as np
import matplotlib.pyplot as plt
import os

cwd = os.getcwd()
os.chdir('../..')
from rsnaped import Frame2D
os.chdir(cwd)


image = np.zeros([512,512],dtype=np.uint16);
spots_xy = np.random.randint(100,400,size=[100,2])
values_xy = np.random.randint(100,400,size=[100])
sigma = 5
spot_size = 11
intensity_scale = 1

frame = Frame2D().make(spots_xy, values_xy, spot_size, sigma,
         image, intensity_scale,)

plt.matshow(frame)