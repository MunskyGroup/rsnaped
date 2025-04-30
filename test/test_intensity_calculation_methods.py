# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 10:06:23 2025

@author: wsraymon
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import matplotlib.patches as patches
from scipy.optimize import curve_fit
from scipy.stats import norm

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


def return_disk_donut_square(image, spot_size, donut_r=5):
    tem_img = image.copy().astype('float')
    center_coordinates = int(tem_img.shape[0]/2)    
    range_to_replace = np.linspace(-(spot_size - 1) / 2, (spot_size - 1) / 2, spot_size,dtype=int)
    min_index = center_coordinates+range_to_replace[0]
    max_index = (center_coordinates +range_to_replace[-1])+1    
    disk = tem_img[min_index: max_index , min_index: max_index].copy()
    tem_img[min_index: max_index , min_index: max_index] *= np.nan
    donut = tem_img[min_index-donut_r:max_index+donut_r , min_index-donut_r:max_index+donut_r]
    
    disk_values = np.mean(disk.flatten())
    donut_values = donut[~np.isnan(donut)].flatten()
    return disk, donut, disk_values, donut_values, 

def return_donut(image, spot_size):
    tem_img = image.copy().astype('float')
    center_coordinates = int(tem_img.shape[0]/2)
    range_to_replace = np.linspace(-(spot_size - 1) / 2, (spot_size - 1) / 2, spot_size,dtype=int)
    min_index = center_coordinates+range_to_replace[0]
    max_index = (center_coordinates +range_to_replace[-1])+1
    tem_img[min_index: max_index , min_index: max_index] *= np.nan
    removed_center_flat = tem_img.copy().flatten()
    donut_values = removed_center_flat[~np.isnan(removed_center_flat)]
    return donut_values.astype('uint16'), tem_img

def disk_donut(values_disk, values_donut):
    mean_intensity_disk = np.mean(values_disk.flatten().astype('float'))
    spot_intensity_disk_donut_std = np.std(values_disk.flatten().astype('float'))
    mean_intensity_donut = np.mean(values_donut.flatten().astype('float')) # mean calculation ignoring zeros
    spot_intensity_disk_donut = mean_intensity_disk - mean_intensity_donut
    #spot_intensity_disk_donut[np.isnan(spot_intensity_disk_donut)] = 0 # replacing nans with zero
    return spot_intensity_disk_donut, spot_intensity_disk_donut_std


def get_crop(image, xy,  crop_size=25):
    rx = int(np.round(xy[0]))
    ry = int(np.round(xy[1])) #center pixel
    
    if crop_size%2 == 0:
        lc, rc = int(np.floor(crop_size/2)-1), int(np.floor(crop_size/2))
    else:
        lc, rc = int(np.floor(crop_size/2)), int(np.floor(crop_size/2))
    return image[rx-lc:rx+rc+1, ry-lc:ry+rc+1]
    

def gaussian_fit(test_im):
    size_spot = test_im.shape[0]
    image_flat = test_im.ravel()
    def gaussian_function(size_spot, offset, sigma):
        ax = np.linspace(-(size_spot - 1) / 2., (size_spot - 1) / 2., size_spot)
        xx, yy = np.meshgrid(ax, ax)
        kernel =  offset *(np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma)))
        return kernel.ravel()
    p0 = (np.min(image_flat) , np.std(image_flat) ) # int(size_spot/2))
    optimized_parameters, _ = curve_fit(gaussian_function, size_spot, image_flat, p0 = p0)
    spot_intensity_gaussian = optimized_parameters[0] # Amplitude
    spot_intensity_gaussian_std = optimized_parameters[1]
    return spot_intensity_gaussian, spot_intensity_gaussian_std

# method 1 - square disk donut

# method 1.5 - square disk donut fit

Is = []
bgs = []
for i in range(2,20):
    _,_, disk, donut = return_disk_donut_square(test_spot_one, i, donut_r = 10)
    I, bg = np.mean(disk), np.mean(donut)
    Is.append(I)
    bgs.append(bg)

plt.figure()
plt.plot(range(2,20), Is);plt.plot(range(2,20), bgs)
plt.figure()
plt.imshow(test_spot_one)
ax = plt.gca()
spot_r = 7
donut_r = 10
spot = patches.Rectangle((25-spot_r/2, 25-spot_r/2), spot_r, spot_r, linewidth=1, edgecolor='r', facecolor='none')
donut = patches.Rectangle((25-spot_r/2-donut_r/2, 25-spot_r/2-donut_r/2), spot_r+donut_r, spot_r+donut_r, linewidth=1, edgecolor='r', facecolor='none')
#plt.plot([24],[24],'rx')
ax.add_patch(spot)
ax.add_patch(donut)

plt.figure()
plt.plot(test_spot_one[25,:])
plt.plot(test_spot_one[24,:])
plt.plot(test_spot_one[23,:])
plt.plot(test_spot_one[22,:])
plt.plot(test_spot_one[21,:])

plt.figure()
center = int(np.round(test_spot_one.shape[0]/2))
horizontal_line = np.mean(test_spot_one[center-2:center+2,:],axis=0)
plt.plot(horizontal_line)
#plt.plot(norm.pdf(np.linspace(0,49,50),24.5, 3)*2000 + 25)

x = np.linspace(0,49,50)
center = int(np.round(test_spot_one.shape[0]/2))
y = np.mean(test_spot_one[center-2:center+2,:],axis=0)
f = lambda x, sigma, intensity, offset,xoffset: norm.pdf(x, 24.5-xoffset, sigma)*intensity + offset
pars,_ = curve_fit(f, x, y)
sigma, intensity, offset,xoffset  = pars

plt.plot(norm.pdf(x, center-xoffset, sigma)*intensity + offset)

def spot_size_guesser(centered_image):
    
    center = int(np.round(centered_image.shape[0]/2))
    horizontal_line = np.mean(centered_image[center-2:center+2,:],axis=0)
    vertical_line = np.mean(centered_image[:,center-2:center+2],axis=1)
    
    x = np.arange(len(centered_image))
    center = int(np.round(centered_image.shape[0]/2))
    
    y = horizontal_line
    f = lambda x, sigma, intensity, offset: norm.pdf(x, 24.1, sigma)*intensity + offset
    pars,_ = curve_fit(f, x, y)
    sigma_x, intensity_x, offset_x,  = pars
    
    y = vertical_line
    f = lambda x, sigma, intensity, offset: norm.pdf(x, 24.1, sigma)*intensity + offset
    pars,_ = curve_fit(f, x, y)
    sigma_y, intensity_y, offset_y,  = pars
    
    return np.mean([sigma_x,sigma_y]), np.mean([offset_y,offset_x])


def disk_donut_square(centered_image, disk_w, donut_w=10):
    '''
    Get a circular disk and donut mask for a given spot

    Parameters
    ----------
    centered_image : np.ndarray
        X by Y np array of image data with spot centered.
    disk_r : float
        radius of the disk to use.
    donut_r : float, optional
        radius of the donut AFTER the disk (ie disk_r = 5, and donut_r = 10 the true donut radius is 15). The default is 10.

    Returns
    -------
    disk_mask : np.ndarray
        boolean array of the disk mask.
    donut_mask : np.ndarray
        boolean array of the donut mask.

    '''
    # the short: convert every pixel to a distance from center array, then do a boolean comparison
    
    center_coordinates = int(centered_image.shape[0]/2)    
    range_to_replace = np.linspace(-(disk_w - 1) / 2, (disk_w - 1) / 2, disk_w, dtype=int)
    min_index = center_coordinates+range_to_replace[0]
    max_index = (center_coordinates +range_to_replace[-1])+1    
    disk_mask = np.ones_like(centered_image)
    disk_mask[min_index: max_index , min_index: max_index] +=1
    disk_mask -= 1
    donut_mask =  np.ones_like(centered_image)
    donut_mask[min_index-donut_w:max_index+donut_w , min_index-donut_w:max_index+donut_w] +=1
    donut_mask[min_index: max_index , min_index: max_index] -=1
    donut_mask -= 1
    
    return disk_mask, donut_mask

    
def disk_donut_circular(centered_image, disk_r, donut_r=10):
    h,w = centered_image.shape
    center = tuple([int(np.round(x)/2) for x in test_spot_one.shape])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    disk_mask = dist_from_center <= disk_r
    donut_mask = (dist_from_center > disk_r) *(dist_from_center <= disk_r+donut_r)
    
    return disk_mask, donut_mask


def get_intensity_disk_donut_circular(centered_spot_image, guessed_sigma=None, donut_r = 10):
    if guessed_sigma is None:
        guessed_sigma, guessed_bg = spot_size_guesser(centered_spot_image) #guess the sigma of gaussian fit of the spot
    
    # use this sigma to calculate the full-width-half-maximum of the gaussian + one pixel (this gives us a disk that covers the
    # entire spot if its circular gaussian)
    fwhm = 2*np.sqrt(2*np.log(2))*3 
    fwhm_plus_one = fwhm + 1
    
    # get the masks (circular inclusive of the guessed sized)
    disk_mask, donut_mask = disk_donut_circular(centered_spot_image, fwhm_plus_one, donut_r=donut_r)
    disk_av = np.mean(centered_spot_image[disk_mask])
    donut_av = np.mean(centered_spot_image[donut_mask])
    return disk_av, donut_av
    
    

# method 2 - round disk donut 

# method 3 - round disk donut fit

# method 4 - gaussian fit







