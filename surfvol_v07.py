# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 04:25:50 2023

@author: dnugr

Datasets:
    Datasets\SPHERE\
    Datasets\PYRAMID\
    Datasets\SPILES1\

-m Datasets\SPHERE\SPHERE.TIF -b Datasets\SPHERE\BASE.TIF -e Datasets\SPHERE\EXTENTS.SHP -n -32767
-m Datasets\PYRAMID\PYRAMID.TIF -b Datasets\PYRAMID\BASE.TIF -e Datasets\PYRAMID\EXTENTS.SHP -n -32767
-m Datasets\SPILES1\SURF2XR.TIF -b Datasets\SPILES1\SURF1X.TIF -e Datasets\SPILES1\TOES.SHP -n -32767
        
v0.7-231129 - improvements
    features:
        - much faster calcs using custom "within" algorithm, ditching Shapely
        - 2.8 secs execution vs 25 secs using v0.4.
        
    todo:
        - test with various projections
        - prepare to integrate with QGIS as a plugin


v0.6-231129 - improvements
    features:
        - even faster calcs using Numpy vectorization for xy transformation
        - 6.5 secs execution vs 25 secs using v0.4
    
    removed:
        - custom affine function removed. It is inaccurate.
        
    todo:
        - test with various projections
        - prepare to integrate with QGIS as a plugin
        
v0.4-231127 - improvements
    features:
        - faster calculation using custom affine function
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

from shapely.geometry import box
import geopandas as gpd

def is_point_in_polygon(point, polygon):
    """
    Check if a point is contained within a polygon using the even-odd (winding number) algorithm.

    Args:
    - point (tuple or list): Coordinates of the point as (x, y).
    - polygon (list of tuples or lists): Coordinates of the polygon vertices.

    Returns:
    - bool: True if the point is inside the polygon, False otherwise.
    """
    x, y = point
    n = len(polygon)
    odd_nodes = False
    j = n - 1

    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if yi < y and yj >= y or yj < y and yi >= y:
            if xi + (y - yi) / (yj - yi) * (xj - xi) < x:
                odd_nodes = not odd_nodes
        j = i

    return odd_nodes


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to model surface")
ap.add_argument("-b", "--base", required=True,
	help="path to extent boundary")
ap.add_argument("-e", "--extent", required=True,
	help="path to class raster")
ap.add_argument("-n", "--nodata", required=False,
	help="NoData value")

args = vars(ap.parse_args())

modelpath = args["model"]
basepath = args["base"]
extentpath = args["extent"]
nodatastr = args["nodata"]

# ensure NoData value is valid
if nodatastr == None:
    nodata = -32767
    print("NoData value was not specified, defaulting to -32767.")
else:
    try:
        nodata = int(nodatastr)
    except (TypeError, ValueError):
        # Handle the exception
        print("Please enter an integer for NoData value.")
        sys.exit(0)

#ensure the files exist
if not os.path.isfile(modelpath):
    print("Model file " + modelpath + " does not exist.")
    sys.exit(0)

if not os.path.isfile(basepath):
    print("Base file " + basepath + " does not exist.")
    sys.exit(0)

if not os.path.isfile(extentpath):
    print("Extent file " + extentpath + " does not exist.")
    sys.exit(0)
    
# read the raster files
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

# mask any NoData values (-32767 or any other values specified)
array1 = np.ma.masked_where(array1 == nodata, array1)
array2 = np.ma.masked_where(array2 == nodata, array2)

xres, yres = ras1.res

xdim = array1.shape[0]
ydim = array1.shape[1]
arraybnd = np.zeros((xdim, ydim))

# recreate the transformation that defines the intersection data
intersection_transform = rio.transform.from_bounds(*intersection.bounds, 
                                                   width=xdim, height= ydim)

start = time.time()

# calculate surface difference for each cell
diffarray_all = np.subtract(array2, array1)

# find the total volume, cut & fill
area_all = xdim*ydim*(xres*yres)
cut_all = np.sum(diffarray_all[diffarray_all<0])*(xres*yres)
fill_all = np.sum(diffarray_all[diffarray_all>=0])*(xres*yres)
volume_all = np.sum(diffarray_all)*(xres*yres)

end = time.time()

# print results for the entire raster intersection
print("Area of entire bound : " + str(area_all))
print("Net volume  : " + str(volume_all))
print("Cut volume  : " + str(cut_all))
print("Fill volume : " + str(fill_all))
print()
print("Elapsed time : " + str(end - start) + " seconds")  # time in seconds
print()


start = time.time()

# Create an array of x and y coordinates using NumPy
x_coords, y_coords = np.meshgrid(np.arange(xdim), np.arange(ydim))

# Apply the affine transformation to all coordinates at once using NumPy
transformed_coords = rio.transform.xy(intersection_transform, x_coords, y_coords)
flattened_coords = np.array(transformed_coords, dtype=np.float64).flatten(order='F')

# initialize list
pointlist = []
pointarr = []

# Make a list of points object
for i in range(0, len(flattened_coords), 2):
    pointlist.append((flattened_coords[i], flattened_coords[i+1]))

# read the actual shapefile
shapefile_path = extentpath
gdf = gpd.read_file(shapefile_path)

# Check if the point is within the polygon(s) in the shapefile
for polygon in gdf['geometry']:
    
    # get the list of the polygon vertices coordinates
    polygon_points = tuple(zip(*polygon.exterior.coords.xy))

    # check all the points and put it in point list
    for kk in range(xdim*ydim):
        pointarr.append(is_point_in_polygon(pointlist[kk], polygon_points))

# construct boundary array of 0 and 1 from the check result
# the 1s means it's within boundary and should be calculated
arraybnd = np.array(pointarr).reshape([xdim, ydim]).astype(int)
diffarray_bnd = diffarray_all * arraybnd

# find the total volume, cut & fill
area_bnd = np.count_nonzero(arraybnd)*(xres*yres)
cut_bnd = np.sum(diffarray_bnd[diffarray_bnd<0])*(xres*yres)
fill_bnd = np.sum(diffarray_bnd[diffarray_bnd>=0])*(xres*yres)
volume_bnd = np.sum(diffarray_bnd)*(xres*yres)

end = time.time()

# print results for the area within extent boundary
print("Area of extent bound : " + str(area_bnd))
print("Net volume  : " + str(volume_bnd))
print("Cut volume  : " + str(cut_bnd))
print("Fill volume : " + str(fill_bnd))
print()
print("Elapsed time : " + str(end - start) + " seconds")  # time in seconds