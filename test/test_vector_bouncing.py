# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 22:49:50 2025

@author: willi
"""

import numpy as np
import matplotlib.pyplot as plt


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
r = v1-2*(u)
new_pt = i1-r 

plt.figure()
plt.plot([vert1[0], vert2[0]], [vert1[1], vert2[1]]);
plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]]);
plt.plot([i1[0], (i1 - r)[0]], [i1[1], (i1-r)[1]])

