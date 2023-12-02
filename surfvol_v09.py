# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 04:25:50 2023

@author: dnugroho

USAGE: survfol -b BASEFILE -m MODELFILE -e EXTENTFILE -n NODATAVAL

BASEFILE   : Geotiff elevation raster file representing the base/original surface
MODELFILE  : Geotiff elevation raster file representing the expected/new surface
EXTENTFILE : Shapefile of the volume extent boundary
NODATAVAL  : NoData values, default -32767 if not specified.

TEST DATASETS:
    
Datasets:
    Datasets\SPHERE\
    Datasets\PYRAMID\
    Datasets\SPILES1\

-m Datasets\SPHERE\SPHERE.TIF -b Datasets\SPHERE\BASE.TIF -e Datasets\SPHERE\EXTENTS.SHP -n -32767
-m Datasets\PYRAMID\PYRAMID.TIF -b Datasets\PYRAMID\BASE.TIF -e Datasets\PYRAMID\EXTENTS.SHP -n -32767
-m Datasets\SPILES1\SURF2XR.TIF -b Datasets\SPILES1\SURF1X.TIF -e Datasets\SPILES1\TOES.SHP -n -32767
-m Datasets//LFILL//NOVEMBER.TIF -b Datasets//LFILL//OCTOBER.TIF -e Datasets//LFILL//EXTENTS.SHP -n -32767

VERSION HISTORY:

v0.9-231202 - pre-release
    features:
        - using higher resolution option for higher accuracy
        
v0.8-231202 - improvements
    features:
        - lightning fast execution by removing the need of checking "within" and coord
          transformation
        - Using rasterizer for shapefile instead.
        
    removed:
        - code to do within polygon checks & coord transformations
        - functions to do those 2 items
        
    todo:
        - automatically use raster with higher resolution as the primary raster
        - prepare to integrate with QGIS as a plugin

v0.7-231129 - improvements
    features:
        - much faster calcs using custom "within" algorithm, ditching Shapely
        - 2.8 secs execution vs 25 secs using v0.4.
        - use custom Affine transform instead of rasterio_transform for faster process
        
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
from rasterio import features
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LightSource
from shapely.geometry import box
import geopandas as gpd

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

print("Volume calculation between two surfaces")
print()
print("Base surface raster file   : " + os.path.basename(basepath))
print("Model surface raster file  : " + os.path.basename(modelpath))
print("Extent boundary shapefile  : " + os.path.basename(extentpath))
print("NoData value               : " + str(nodata))
print()

# read the raster files
with rio.open(modelpath) as ras1, rio.open(basepath) as ras2:
    
    xres1, yres1 = ras1.res
    xres2, yres2 = ras2.res
    
    # if ras1 lower res thatn ras2, then swap it
    # this is to ensure all calcs is based on the higher resolution
    
    if (xres1*yres1) > (xres2*yres2):
        xres, yres = xres2, yres2
        tempras = ras2
        ras2 = ras1
        ras1 = tempras
        swapped = True
    else:
        xres, yres = xres1, yres1
        swapped = False
    
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

xdim = array1.shape[1]
ydim = array1.shape[0]

# recreate the transformation that defines the intersection data
intersection_transform = rio.transform.from_bounds(*intersection.bounds, 
                                                   width=xdim, height=ydim)

start = time.time()

# calculate surface difference for each cell
# check if rasters were swapped for higher resolution to ensure correct layers
if swapped:
    diffarray_all = np.subtract(array2, array1)
else:
    diffarray_all = -np.subtract(array2, array1)

# find the total volume, cut & fill
area_all = xdim*ydim*(xres*yres)
cut_all = np.sum(diffarray_all[diffarray_all<0])*(xres*yres)
fill_all = np.sum(diffarray_all[diffarray_all>=0])*(xres*yres)
volume_all = np.sum(diffarray_all)*(xres*yres)

# avoid any masked values from causing errors in printout
if np.ma.is_masked(cut_all):
    cut_all = 0
if np.ma.is_masked(fill_all):
    fill_all = 0
if np.ma.is_masked(volume_all):
    volume_all = 0

# print results for the entire raster intersection
print("Area of entire raster intersection : " + str(round(area_all,3)))
print("Net volume  : " + str(round(volume_all,3)))
print("Cut volume  : " + str(round(cut_all,3)))
print("Fill volume : " + str(round(fill_all,3)))
print()


# read the actual shapefile
shapefile_path = extentpath
gdf = gpd.read_file(shapefile_path)

# construct boundary array of 0 and 1 from the check result
# the 1s means it's within boundary and should be calculated
arraybnd = features.rasterize(gdf['geometry'], out_shape=array1.shape,
            transform=intersection_transform)
    
# calculate surface difference for each cell within boundary
diffarray_bnd = diffarray_all * arraybnd

# find the total volume, cut & fill
area_bnd = np.count_nonzero(arraybnd)*(xres*yres)
cut_bnd = np.sum(diffarray_bnd[diffarray_bnd<0])*(xres*yres)
fill_bnd = np.sum(diffarray_bnd[diffarray_bnd>=0])*(xres*yres)
volume_bnd = np.sum(diffarray_bnd)*(xres*yres)

# avoid any masked values from causing errors in printout
if np.ma.is_masked(cut_bnd):
    cut_bnd = 0
if np.ma.is_masked(fill_bnd):
    fill_bnd = 0
if np.ma.is_masked(volume_bnd):
    volume_bnd = 0

end = time.time()
duration = end-start

# print results for the area within extent boundary
print("Area of extent boundary : " + str(round(area_bnd,3)))
print("Net volume   : " + str(round(volume_bnd,3)))
print("Cut volume   : " + str(round(cut_bnd,3)))
print("Fill volume  : " + str(round(fill_bnd,3)))
print()
print("X resolution : " + str(round(xres,3)))
print("Y resolution : " + str(round(yres,3)))
print("Elapsed time : " + str(round(duration,3)))  # time in seconds


# Illuminate the scene from the northwest
ls = LightSource(azdeg=315, altdeg=45)

#plt.imshow(ls.hillshade(diffarray_bnd, vert_exag=1), cmap=mpl.colormaps['grey'])
plt.imshow(ls.shade(diffarray_all, cmap=mpl.colormaps['jet'], vert_exag=1, blend_mode='hsv'))
plt.show() 
plt.imshow(ls.shade(diffarray_bnd, cmap=mpl.colormaps['jet'], vert_exag=1, blend_mode='hsv'))
plt.show() 
plt.imshow(arraybnd)
plt.show() 
