# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 22:49:50 2025

@author: willi
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib import patches

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

@staticmethod
def line_segment_intersect(pt1, pt2, vert1, vert2):
    # THIS DOES NOT WORK FOR COLINEAR SEGMENTS (in a line or touching)
    # but I dont expect colinearity to arise in this problem ever.
    # for more see: https://www.cs.cmu.edu/~quake/robust.html
    # and https://stackoverflow.com/questions/3838329/how-can-i-check-if-two-segments-intersect
    dx0 = pt2[0]-pt1[0]
    dx1 = vert2[0]-vert1[0]
    xdiff = (dx0, dx1)
    dy0 = pt2[1]-pt1[1]
    dy1 = vert2[1]-vert1[1]
    ydiff = (dy0, dy1)
    
    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]
    
    p0 = dy1*(vert2[0]-pt1[0]) - dx1*(vert2[1]-pt1[1])
    p1 = dy1*(vert2[0]-pt2[0]) - dx1*(vert2[1]-pt2[1])
    p2 = dy0*(pt2[0]-vert1[0]) - dx0*(pt2[1]-vert1[1])
    p3 = dy0*(pt2[0]-vert2[0]) - dx0*(pt2[1]-vert2[1])
    

    
    if (p0*p1<=0) & (p2*p3<=0):
        div = det(xdiff, ydiff)
        d = det(pt1,pt2), det(vert1,vert2)
        intersect = -det(d, xdiff)/ div, -det(d,ydiff)/div
        return (p0*p1<=0) & (p2*p3<=0), intersect
    else:
        return (p0*p1<=0) & (p2*p3<=0), None

@staticmethod
def get_slope(pt1,pt2):
    return (pt1[1] - pt2[1] ) / (pt1[0] - pt2[0])
@staticmethod
def distance(pt1,pt2):
    return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)


# negative slope left to right
vert1 = np.array([200., 100.])
vert2 = np.array([200., 200.])
pt1 = np.array([190.37076401, 143.27395815])
pt2 = np.array([206.26829747, 101.76580372])

s1, i1 = line_segment_intersect(pt1,pt2,vert1,vert2)

# reflect the part outside geometry back in (perfect elastic)
v1 = pt2 - i1 #vector that "left" the geometry

# normal of the geometry it left
normal = (vert2[0] - vert1[0], vert2[1] - vert1[1] )/distance(vert1,vert2)

#calculate new bounce vector and its endpoint
u = (v1@normal)*normal
r = v1-2*(u)
new_pt = i1-r 

plt.figure()
plt.plot([vert1[0], vert2[0]], [vert1[1], vert2[1]]);
plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]]);
plt.plot([i1[0], (i1 - r)[0]], [i1[1], (i1-r)[1]])


# positive slope right to left
vert1 = np.array([200., 100.])
vert2 = np.array([200., 200.])
pt1 = np.array([206.26829747, 101.76580372])
pt2 = np.array([190.37076401, 143.27395815])

s1, i1 = line_segment_intersect(pt1,pt2,vert1,vert2)

# reflect the part outside geometry back in (perfect elastic)
v1 = pt2 - i1 #vector that "left" the geometry

# normal of the geometry it left
normal = (vert2[0] - vert1[0], vert2[1] - vert1[1] )/distance(vert1,vert2)

#calculate new bounce vector and its endpoint
u = (v1@normal)*normal
r = v1-2*(u)
new_pt = i1-r 

plt.figure()
plt.plot([vert1[0], vert2[0]], [vert1[1], vert2[1]]);
plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]]);
plt.plot([i1[0], (i1 - r)[0]], [i1[1], (i1-r)[1]])



# negative slope slope right to left
vert1 = np.array([190., 130.])
vert2 = np.array([205., 130.])
pt1 = np.array([206.26829747, 101.76580372])
pt2 = np.array([190.37076401, 143.27395815])

s1, i1 = line_segment_intersect(pt1,pt2,vert1,vert2)

# reflect the part outside geometry back in (perfect elastic)
v1 = pt2 - i1 #vector that "left" the geometry

# normal of the geometry it left
normal = (vert2[0] - vert1[0], vert2[1] - vert1[1] )/distance(vert1,vert2)

#calculate new bounce vector and its endpoint
u = (v1@normal)*normal
r = v1-2*(u)
new_pt = i1-r 

plt.figure()
plt.plot([vert1[0], vert2[0]], [vert1[1], vert2[1]]);
plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]]);
plt.plot([i1[0], (i1 - r)[0]], [i1[1], (i1-r)[1]])


# positive slope slope left to right
vert1 = np.array([190., 130.])
vert2 = np.array([205., 130.])
pt1 = np.array([192, 101])
pt2 = np.array([206.26829747, 143]) 

s1, i1 = line_segment_intersect(pt1,pt2,vert1,vert2)

# reflect the part outside geometry back in (perfect elastic)
v1 = pt2 - i1 #vector that "left" the geometry

# normal of the geometry it left
normal = (vert2[0] - vert1[0], vert2[1] - vert1[1] )/distance(vert1,vert2)

#calculate new bounce vector and its endpoint
u = (v1@normal)*normal
r = (v1-2*(u))*.5
new_pt = i1-r 

plt.figure()
plt.plot([vert1[0], vert2[0]], [vert1[1], vert2[1]]);
plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]]);
plt.plot([i1[0], (i1 - r)[0]], [i1[1], (i1-r)[1]])




###############################################################################

vertices = np.array([[100,100],[150,50],[200,100],[200,200],[150,250],[100,200],[100,100 ]])

diff2D = rsp.Diffusion2D(vertices)

hexagon = Path(vertices)
fig, ax = plt.subplots()
patch = patches.PathPatch(hexagon, facecolor='orange', lw=2)
ax.add_patch(patch)

pts = diff2D.initialize_spots(1)
plt.scatter(*pts)

plt.xlim([0,300]); plt.ylim([0,300]);
plt.show()

@staticmethod
def get_slope(pt1,pt2):
    return (pt1[1] - pt2[1] ) / (pt1[0] - pt2[0])
@staticmethod
def distance(pt1,pt2):
    return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
@staticmethod
def line_segment_intersect(pt1, pt2, vert1, vert2):
    # THIS DOES NOT WORK FOR COLINEAR SEGMENTS (in a line or touching)
    # but I dont expect colinearity to arise in this problem ever.
    # for more see: https://www.cs.cmu.edu/~quake/robust.html
    # and https://stackoverflow.com/questions/3838329/how-can-i-check-if-two-segments-intersect
    dx0 = pt2[0]-pt1[0]
    dx1 = vert2[0]-vert1[0]
    xdiff = (dx0, dx1)
    dy0 = pt2[1]-pt1[1]
    dy1 = vert2[1]-vert1[1]
    ydiff = (dy0, dy1)
    
    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]
    
    p0 = dy1*(vert2[0]-pt1[0]) - dx1*(vert2[1]-pt1[1])
    p1 = dy1*(vert2[0]-pt2[0]) - dx1*(vert2[1]-pt2[1])
    p2 = dy0*(pt2[0]-vert1[0]) - dx0*(pt2[1]-vert1[1])
    p3 = dy0*(pt2[0]-vert2[0]) - dx0*(pt2[1]-vert2[1])
    

    
    if (p0*p1<=0) & (p2*p3<=0):
        div = det(xdiff, ydiff)
        d = det(pt1,pt2), det(vert1,vert2)
        intersect = -det(d, xdiff)/ div, -det(d,ydiff)/div
        return (p0*p1<=0) & (p2*p3<=0), intersect
    else:
        return (p0*p1<=0) & (p2*p3<=0), None
    
def check_if_segment_left_geometry(vertices, pt1, pt2, reflection = False, previous_vert = -1):
    vertices_left = []
    intersects = []
    vert_ids = []

    
    for i in range(len(vertices)-1):
        b, i1 = line_segment_intersect(pt1, pt2, vertices[i], vertices[i+1])

        if b:
            # IF WE ARE DOING AN ITERATIVE REFLECTION, ignore the vertex where the reflection
            # is originating from, so we dont "find" the closest intersection point to be what we 
            # are reflecting from
            if reflection:
                if i != previous_vert:
                    vertices_left.append(vertices[[i,i+1]])
                    intersects.append(i1)
                    vert_ids.append(i)
            else:
                vertices_left.append(vertices[[i,i+1]])
                intersects.append(i1)
                vert_ids.append(i)                
    
    if len(vertices_left) == 0: # segment did not leave geometry
        return False, None, None, None, None
    
    # segment did leave the geometry only once:
    if len(vertices_left) == 1: 
        return True, vertices_left[0][0], vertices_left[0][1], intersects[0], vert_ids[0]
    
    # segment left multiple times, find the closest segment intersection to first point
    if len(vertices_left) > 1:
        i = np.argmin([distance(pt1,i1) for i1 in intersects])
        print(intersects)
        return True, vertices_left[i][0], vertices_left[i][1], intersects[i], vert_ids[i]
        
    

initial_points = np.array([[150,150]]).T
t = np.array([0,100])
elasticity = 1
trajs = np.zeros([2,2,1])
trajs[:,0,0] = [150,150]
trajs[:,1,0] = [1500,250]

# reflect each trajectory when it crosses a boundary
for i in range(initial_points.shape[-1]):
    for j in range(1,len(t)):
        # FOR EACH LINE SEGMENT, check if it crosses the geometry
        # if it does, vert1, vert2, and intersection point
        pt1 = trajs[:,j-1,i] 
        pt2 = trajs[:,j,i] 
        left, vert1, vert2, i1, vert_id = check_if_segment_left_geometry(vertices,pt1,pt2)
        intersections = [i1]
        new_point = pt2
        iter_num = 0
        while left: #if the line segment left
            
            # reflect the part outside geometry back in (perfect elastic)
            v1 = new_point - i1 #vector that "left" the geometry
            
            #  FLIPPED normal of the geometry it left, this is not a normal normal.
            normal = (vert2[0] - vert1[0], vert2[1] - vert1[1] )/distance(vert1,vert2)
            
            #calculate new bounce vector and its endpoint
            u = (v1@normal)*normal
            r = (v1-2*(u))*elasticity
            new_point = (i1-r)
            print('***')
            print(new_point)
            
            #did the new line leave again??? reflect again from the first intersection point
            left, vert1, vert2, i1, vert_id = check_if_segment_left_geometry(vertices, i1, new_point, reflection=True, previous_vert=vert_id)
            if i1 != None:
                intersections.append(i1)
            
        trajs[:,j,i] =  new_point 
            

vertices = np.array([[100,100],[150,50],[200,100],[200,200],[150,250],[100,200],[100,100 ]])


hexagon = Path(vertices)
fig, ax = plt.subplots()
patch = patches.PathPatch(hexagon, facecolor='orange', lw=2)
ax.add_patch(patch)

pts = diff2D.initialize_spots(1)
plt.plot([pt1[0],*[i[0] for i in intersections], new_point[0]], [pt1[1],*[i[1] for i in intersections],new_point[1]],'o-')

plt.xlim([0,300]); plt.ylim([0,300]);
plt.show()