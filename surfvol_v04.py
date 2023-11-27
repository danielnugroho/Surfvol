# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 04:25:50 2023

@author: dnugr

v0.4-231127 - improvements
    features:
        - faster calculation using custom function
        - this custom function matches idealized math surface
        - removed -32767 values from arrays using numpy masked array
        
    todo:
        - test with various projections
        - prepare to integrate with QGIS as a plugin
        
v0.3-231126 - improvements
    features:
        - info for area, cut & fill.
        - use command line arguments instead hardcoded
        - exception handling
        
    to be improved:
        - faster calculation (esp xy transform part):
            problem: custom affine works fast, but the result is different
            from GLobal Mapper result.
            Will need to check with idealized mathematical surfaces.
            findings: Custom affine works more accurate and faster.

v0.2-231126 - improvements
    features:
        - can use two arbitrary size rasters and resolution
        
    to be improved:
        - improve speed using np.meshgrid
        - select the lowest res grid as the base for the calc

v0.1-231126 - first version 
    features:
        - good for two rasters of same size, same res, same projection
        - proved that the result is correct (compared to Global Mapper)
   
   to be improved:
        - use two arbitrary size arrays/rasters
        - improve speed using np.meshgrid

"""

import sys, os, argparse, time
import rasterio as rio
from rasterio.enums import Resampling
import numpy as np

from shapely.geometry import Point
from shapely.geometry import box
import geopandas as gpd

def xy_np(a, b, c, d, e, f, x, y):
    """
    Apply an affine transformation to (x, y) coordinates.

    Parameters:
    a, b, c, d, e, f (float): Transformation parameters.
    x (float or numpy array): Input x-coordinate(s).
    y (float or numpy array): Input y-coordinate(s).

    Returns:
    numpy array: Transformed (x, y) coordinates.
    
    """
    
    #if isinstance(x, (int, float)):
    #    # If single coordinates are provided, convert them to arrays for consistency.
    #    x = np.array([x])
    #    y = np.array([y])

    # Apply the affine transformation.
    transformed_x = a * x + b * y + c
    transformed_y = d * x + e * y + f

    return transformed_x, transformed_y

start = time.time()

# construct the argument parse and parse the arguments

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to model surface")
ap.add_argument("-b", "--base", required=True,
	help="path to extent boundary")
ap.add_argument("-e", "--extent", required=True,
	help="path to class raster")

args = vars(ap.parse_args())

modelpath = args["model"]
basepath = args["base"]
extentpath = args["extent"]

if not os.path.isfile(modelpath):
    print("Model file " + modelpath + " does not exist.")
    sys.exit(0)

if not os.path.isfile(basepath):
    print("Base file " + basepath + " does not exist.")
    sys.exit(0)

if not os.path.isfile(extentpath):
    print("Extent file " + extentpath + " does not exist.")
    sys.exit(0)
    

with rio.open(basepath) as ras1, rio.open(modelpath) as ras2:
    x_scale = ras2.transform.a / ras1.transform.a
    y_scale = ras2.transform.e / ras1.transform.e

    # scale image transform
    transform = ras2.transform * ras2.transform.scale(
        (ras2.width / ras2.shape[-1]),
        (ras2.height / ras2.shape[-2])
    )

    # find the intersection
    
    ext1 = box(*ras1.bounds)
    ext2 = box(*ras2.bounds)
    intersection = ext1.intersection(ext2)
    win1 = rio.windows.from_bounds(*intersection.bounds, ras1.transform)
    win2 = rio.windows.from_bounds(*intersection.bounds, transform)

    # resample the worse res to match higher res
    array1 = ras1.read(1, window=win1)
    array2 = ras2.read(1, 
        window=win2,
        out_shape=(
            ras2.count,
            round(win2.height * y_scale),
            round(win2.width * x_scale)
        ),
        resampling=Resampling.cubic
    )

# mask any NoData values (-32767)
array1 = np.ma.masked_where(array1 == -32767, array1)
array2 = np.ma.masked_where(array2 == -32767, array2)

xres, yres = ras1.res

xdim = array1.shape[0]
ydim = array1.shape[1]
arraysub = np.zeros((xdim, ydim))

intersection_transform = rio.transform.from_bounds(*intersection.bounds, 
                                                   width=xdim, height= ydim)

diffarray_all = np.subtract(array2, array1)

area_all = xdim*ydim*(xres*yres)
cut_all = np.sum(diffarray_all[diffarray_all<0])*(xres*yres)
fill_all = np.sum(diffarray_all[diffarray_all>=0])*(xres*yres)
volume_all = np.sum(diffarray_all)*(xres*yres)

print("Area of entire bound : " + str(area_all))
print("Net volume  : " + str(volume_all))
print("Cut volume  : " + str(cut_all))
print("Fill volume : " + str(fill_all))
print()

# Step 1: Define a point
# Replace with your point coordinates

# Step 2: Read a shapefile using Geopandas
# Replace with the path to your shapefile
shapefile_path = extentpath
gdf = gpd.read_file(shapefile_path)

# Step 3: Check if the point is within the polygon(s) in the shapefile
for i in range(xdim):
    for j in range(ydim):
        
        #xs, ys = rio.transform.xy(intersection_transform,i,j)
        xs, ys = xy_np(*intersection_transform[:6],i,j)
        point1 = Point(xs, ys)
        
        for polygon in gdf['geometry']:
            if point1.within(polygon):
                arraysub[i,j] = 1

diffarray_bnd = diffarray_all * arraysub

# find the total volume, cut & fill

area_bnd = np.count_nonzero(arraysub)*(xres*yres)
cut_bnd = np.sum(diffarray_bnd[diffarray_bnd<0])*(xres*yres)
fill_bnd = np.sum(diffarray_bnd[diffarray_bnd>=0])*(xres*yres)
volume_bnd = np.sum(diffarray_bnd)*(xres*yres)

end = time.time()

print("Area of extent bound : " + str(area_bnd))
print("Net volume  : " + str(volume_bnd))
print("Cut volume  : " + str(cut_bnd))
print("Fill volume : " + str(fill_bnd))
print()
print("Elapsed time : " + str(end - start) + " seconds")  # time in seconds