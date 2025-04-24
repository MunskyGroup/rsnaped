# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 10:06:23 2025

@author: wsraymon
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

test_spot_one = np.random.randint(0,50,size=(50,50))
test_spot_two = np.random.randint(0,50,size=(50,50))

test_spots = []



def pdf_pixel_resolution(ax, spot_sigma=2):
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(spot_sigma))
    return kernel #kernel/np.max(kernel)

def gaussian_subpixel_erf(point, size_spot=5, spot_sigma=2):
    '''
    get a gaussian kernel from a point in a subpixel of the center

    point: iterable of x and y e.g. [x,y]
    size_spot: size of the kernel to generate **MUST BE ODD** to use the center pixel - consider adding check
    spot_sigma: std of the point spread function

    returns: size_spot x size_spot gaussian kernel with subpixel of frac(point) in the center
    '''
    x,y = point # get the point
    x_p = x - int(x) # get the fraction value, subpixel value of center pixel
    y_p = y - int(y) 
    # generate the N+1 x N+1 pixel grid with N/2 x N/2 as the center pixel
    pixelgrid = np.array([np.linspace( -(size_spot+1)/2+1, (size_spot+1)/2 , size_spot+1)])
    xbar = pixelgrid-x_p #subtract the subpixel value
    ybar = pixelgrid-y_p
    # get the gaussian cdf sum of this bin
    Fx = .5*(1+erf(xbar /(spot_sigma*np.sqrt(2))) ) 
    Fy = .5*(1+erf(ybar /(spot_sigma*np.sqrt(2))) )
    dFx = Fx[:,1:] - Fx[:,:-1] #subtract the differences of the cdfs in each direction
    dFy = Fy[:,1:] - Fy[:,:-1]
    K = dFx.T @ dFy # dot together to generate the NxN kernel
    return K*(1/np.sum(K))  #/np.max(K) # (K-np.min(K))/(np.max(K)-np.min(K))


kernel=gaussian_subpixel_erf(point=(25,25), size_spot=15, spot_sigma=3)

spot_intensity = 5000
kernel_value_intensity = (kernel*spot_intensity*3).astype(np.uint16)
    
photon_count = 8000
sampled_poisson_process_spot = np.array([np.random.poisson(x*photon_count) for x in kernel])
kernel_value_intensity2 = ((sampled_poisson_process_spot/np.max(sampled_poisson_process_spot))*spot_intensity/5).astype(np.uint16) 

center_position = (25,25)
size_spot = 15
spots_range_to_replace = np.linspace(-(size_spot - 1) / 2, (size_spot - 1) / 2, size_spot,dtype=int)

test_spot_one[center_position[0]+spots_range_to_replace[0]: center_position[0]+spots_range_to_replace[-1]+1 , center_position[1]+spots_range_to_replace[0]: center_position[1]+spots_range_to_replace[-1]+1 ] += kernel_value_intensity

test_spot_two[center_position[0]+spots_range_to_replace[0]: center_position[0]+spots_range_to_replace[-1]+1 , center_position[1]+spots_range_to_replace[0]: center_position[1]+spots_range_to_replace[-1]+1 ] += kernel_value_intensity2

plt.imshow(test_spot_one)
plt.figure()
plt.imshow(test_spot_two)


def return_donut(image, spot_size):
    tem_img = image.copy().astype('float')
    center_coordinates = int(tem_img.shape[0]/2)
    range_to_replace = np.linspace(-(spot_size - 1) / 2, (spot_size - 1) / 2, spot_size,dtype=int)
    min_index = center_coordinates+range_to_replace[0]
    max_index = (center_coordinates +range_to_replace[-1])+1
    tem_img[min_index: max_index , min_index: max_index] *= np.nan
    removed_center_flat = tem_img.copy().flatten()
    donut_values = removed_center_flat[~np.isnan(removed_center_flat)]
    return donut_values.astype('uint16')

def disk_donut(values_disk, values_donut):
    mean_intensity_disk = np.mean(values_disk.flatten().astype('float'))
    spot_intensity_disk_donut_std = np.std(values_disk.flatten().astype('float'))
    mean_intensity_donut = np.mean(values_donut.flatten().astype('float')) # mean calculation ignoring zeros
    spot_intensity_disk_donut = mean_intensity_disk - mean_intensity_donut
    #spot_intensity_disk_donut[np.isnan(spot_intensity_disk_donut)] = 0 # replacing nans with zero
    return spot_intensity_disk_donut, spot_intensity_disk_donut_std








