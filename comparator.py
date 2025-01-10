import os
import numpy as np
from osgeo import gdal, ogr, osr
try:
    from numba import njit, prange
except ModuleNotFoundError:
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def prange(n):
        return range(n)
    
gdal.UseExceptions()

########### INPUTS

floodmap_file = '/Users/ricky/Documents/arfs_sample/output/USGS_1_n40w111_20240130_buff__flood.tif'
boundary_file = '/Users/ricky/Documents/streamstats/boundary.shp'
# boundary_file = None
# reference_floodmap_file = '/Users/ricky/Downloads/10_4231_R71V5BZ5/UT/UT_SFM.shp'
reference_floodmap_file = '/Users/ricky/Downloads/10_4231_R71V5BZ5/true_raster.tif'
output_file = 'test.tif'
output_file = None

DEBUG = False
RASTERIZATION_SIZE = 10 # If using a vector file, this is the resolution of the intermediate rasters in meters

########### END INPUTS

def reproject_raster_gdal(src: gdal.Dataset, target_crs_wkt: str) -> gdal.Dataset:
    """Reproject a raster to the target CRS using GDAL."""
    target = gdal.Warp(
        "",
        src,
        dstSRS=target_crs_wkt,
        format="MEM",
        dstNodata=0
    )
    return target

def calculate_overlap_extent(raster1: gdal.Dataset, raster2: gdal.Dataset) -> tuple:
    """Calculate the overlapping extent of two rasters."""
    geo_transform1 = raster1.GetGeoTransform()
    geo_transform2 = raster2.GetGeoTransform()

    extent1 = (
        geo_transform1[0],  # minX
        geo_transform1[3] + geo_transform1[5] * raster1.RasterYSize,  # minY
        geo_transform1[0] + geo_transform1[1] * raster1.RasterXSize,  # maxX
        geo_transform1[3],  # maxY
    )
    extent2 = (
        geo_transform2[0],
        geo_transform2[3] + geo_transform2[5] * raster2.RasterYSize,
        geo_transform2[0] + geo_transform2[1] * raster2.RasterXSize,
        geo_transform2[3],
    )

    overlap_extent = (
        max(extent1[0], extent2[0]),  # minX
        max(extent1[1], extent2[1]),  # minY
        min(extent1[2], extent2[2]),  # maxX
        min(extent1[3], extent2[3]),  # maxY
    )

    return overlap_extent

def clip_raster_gdal(raster: gdal.Dataset, extent: tuple, xsize: int = None, ysize: int = None, xres = None, yres = None) -> gdal.Dataset:
    """Clip a raster to the given extent using GDAL."""
    minX, minY, maxX, maxY = extent
    if xsize is None or ysize is None:
        xsize = raster.RasterXSize
        ysize = raster.RasterYSize
        xres = raster.GetGeoTransform()[1]
        yres = raster.GetGeoTransform()[5]

    gdal.WarpOptions
    clipped = gdal.Warp(
        "",
        raster,
        outputBounds=[minX, maxY, maxX, minY],
        format="MEM",
        dstNodata=0,
        xRes=xres,
        yRes=yres,
    )
    return clipped

def process_rasters_gdal(raster1: gdal.Dataset, raster2: gdal.Dataset) -> tuple[gdal.Dataset, gdal.Dataset]:
    """Check and process rasters to have the same CRS and extent."""
    # Check CRS and reproject if needed
    crs1 = raster1.GetProjection()
    crs2 = raster2.GetProjection()
    if crs1 != crs2:
        raster2 = reproject_raster_gdal(raster2, crs1)

    # Calculate overlapping extent
    overlap_extent = calculate_overlap_extent(raster1, raster2)

    # Clip rasters to the overlapping extent
    xres = raster1.GetGeoTransform()[1]
    yres = raster1.GetGeoTransform()[5]
    raster1_clipped = clip_raster_gdal(raster1, overlap_extent)
    raster2_clipped = clip_raster_gdal(raster2, overlap_extent, raster1.RasterXSize, raster1.RasterYSize, xres, yres)

    return raster1_clipped, raster2_clipped

def reproject_and_align_to_reference(file_to_modify: str, ref_ds: gdal.Dataset) -> np.ndarray:
    """Reproject the current raster to match the projection and resolution of the reference raster."""
    if not file_to_modify.lower().endswith('.tif'):
        ds_to_modify = gdal.Rasterize("",
            file_to_modify,
            format="MEM",
            initValues=0,
            burnValues=[1],
            outputType=gdal.GDT_Byte,
            xRes=ref_ds.GetGeoTransform()[1],
            yRes=ref_ds.GetGeoTransform()[5],
        )
    else:
        ds_to_modify: gdal.Dataset = gdal.Open(file_to_modify)

    ref_projection = ref_ds.GetProjection()
    ref_geo_transform = ref_ds.GetGeoTransform()
    ref_cols = ref_ds.RasterXSize
    ref_rows = ref_ds.RasterYSize

    # Perform the reprojection and resampling
    reprojected = gdal.Warp(
        "",
        ds_to_modify,
        format="MEM",
        dstSRS=ref_projection,
        xRes=ref_geo_transform[1],
        yRes=-ref_geo_transform[5],  # Negative because GeoTransform uses negative Y res
        outputBounds=(ref_geo_transform[0],  # minX
                      ref_geo_transform[3] + ref_geo_transform[5] * ref_rows,  # minY
                      ref_geo_transform[0] + ref_geo_transform[1] * ref_cols,  # maxX
                      ref_geo_transform[3]),  # maxY
        dstNodata=0  # Set nodata value for out-of-bounds regions
    )

    # Read the data into a numpy array
    array = reprojected.GetRasterBand(1).ReadAsArray()

    return array

def rasterize(file: str) -> gdal.Dataset:
    # We assume this is some sort of geometry file; let's rasterize it
    ref_ogr_ds: ogr.DataSource = ogr.Open(file)
    lyr: ogr.Layer = ref_ogr_ds.GetLayer()
    ref: osr.SpatialReference = lyr.GetSpatialRef()
    if ref.IsProjected() == 1:
        units = RASTERIZATION_SIZE # Assume meters
    else:
        units = RASTERIZATION_SIZE * 0.00002777 # Assume degrees

    ref_ds: gdal.Dataset = gdal.Rasterize("",
        file,
        format="MEM",
        initValues=0,
        burnValues=[1],
        outputType=gdal.GDT_Byte,
        xRes=units,
        yRes=units,
    )
    ref_ds.FlushCache()
    if ref_ds is None:
        raise ValueError(f"Failed to rasterize {file}")

def format_number(n):
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"  # Format as millions
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"  # Format as thousands
    else:
        return str(n)  # Return as is for smaller numbers

@njit(cache=True, parallel=True)
def loop_assign(output_array: np.ndarray, flood_array: np.ndarray, ref_array: np.ndarray, mask: np.ndarray) -> tuple[int, int, int, int]:
    # Compare the datasets
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    nrows, ncols = output_array.shape
    for row in prange(nrows):
        for col in prange(ncols):
            if mask[row,col] != 1: # If not in boundry:
                output_array[row,col] = 0
                continue
            point = flood_array[row,col]
            if point == ref_array[row,col]: # If they match 
                if point == 0:
                    output_array[row,col] = 2
                    tn += 1
                else:
                    output_array[row,col] = 3
                    tp += 1
            elif point != 0 and ref_array[row,col]==0: # If predicted flood is over
                output_array[row,col] = 4
                fp += 1
            else:# point == 0 and True_Array[x,y]!=0: # If predicted flood is under
                output_array[row,col] = 1
                fn += 1
    """"
    0: No data
    1: Predicted no flood but there was
    2: Predicted correctly no flood here
    3: Predicted correctly flood here
    4: Predicted flood but there wasn't
    """
    return tp, tn, fp, fn
 
def main():
    assert floodmap_file and os.path.exists(floodmap_file), "Floodmap file not provided or does not exist."
    assert reference_floodmap_file and os.path.exists(reference_floodmap_file), "Reference floodmap file not provided or does not exist."

    # Read the dataset
    if floodmap_file.lower().endswith('.tif'):
        floodmap_ds: gdal.Dataset = gdal.Open(floodmap_file)
    else:
        floodmap_ds = rasterize(floodmap_file)

    geotransform = floodmap_ds.GetGeoTransform()
    flood_projection = floodmap_ds.GetProjection()

    # Get the reference floodmap
    if reference_floodmap_file.lower().endswith('.tif'):
        ref_ds: gdal.Dataset = gdal.Open(reference_floodmap_file)
    else:
        ref_ds = rasterize(reference_floodmap_file)
        
    # Process the rasters
    floodmap_ds, ref_ds = process_rasters_gdal(floodmap_ds, ref_ds)
    
    if boundary_file is not None:
        # Read in boundary mask
        mask: np.ndarray = reproject_and_align_to_reference(boundary_file, ref_ds)
    else:
        mask = np.ones((floodmap_ds.RasterYSize, floodmap_ds.RasterXSize))

    ncols = int(floodmap_ds.RasterXSize)
    nrows = int(floodmap_ds.RasterYSize)
   
    flood_array: np.ndarray = floodmap_ds.ReadAsArray()
    output_array = np.zeros([nrows,ncols], dtype=np.uint8)

    ref_array: np.ndarray = ref_ds.ReadAsArray()

    assert mask.shape == flood_array.shape == ref_array.shape, f"Array shapes do not match. Mask: {mask.shape}, Input Floodmap:{flood_array.shape}, Validation map: {ref_array.shape}"

    # Make sure arrays are same type and contain only 0 and 1
    unique_vals = np.unique(flood_array)
    if not (unique_vals == [0, 1]).all():
        if len(unique_vals) == 2:
            flood_array[flood_array == unique_vals[0]] = 0
            flood_array[flood_array == unique_vals[1]] = 1
        else:
            raise ValueError(f"Unexpected values in Flood Array: {unique_vals}")
    unique_vals = np.unique(ref_array)
    if not (unique_vals == [0, 1]).all():
        if len(unique_vals) == 2:
            ref_array[ref_array == unique_vals[0]] = 0
            ref_array[ref_array == unique_vals[1]] = 1
        else:
            raise ValueError(f"Unexpected values in True Array: {unique_vals}")

    # Now, if not uint8, convert to uint8
    if flood_array.dtype != np.uint8:
        flood_array = flood_array.astype(np.uint8)
    if ref_array.dtype != np.uint8:
        ref_array = ref_array.astype(np.uint8)

    if DEBUG:
        # Save the processed rasters for debugging
        hDriver: gdal.Driver = gdal.GetDriverByName("GTiff")
        output_ds: gdal.Dataset = hDriver.Create("Floodmap_clipped.tif", xsize=floodmap_ds.RasterXSize, ysize=floodmap_ds.RasterYSize, bands=1, eType=gdal.GDT_Byte)
        output_ds.SetGeoTransform(floodmap_ds.GetGeoTransform())
        output_ds.SetProjection(floodmap_ds.GetProjection())
        output_ds.GetRasterBand(1).WriteArray(floodmap_ds.ReadAsArray())
        output_ds = None

        ref_ds_out: gdal.Dataset = hDriver.Create("Reference_clipped.tif", xsize=ref_ds.RasterXSize, ysize=ref_ds.RasterYSize, bands=1, eType=gdal.GDT_Byte)
        ref_ds_out.SetGeoTransform(ref_ds.GetGeoTransform())
        ref_ds_out.SetProjection(ref_ds.GetProjection())
        ref_ds_out.GetRasterBand(1).WriteArray(ref_ds.ReadAsArray())
        ref_ds_out = None

    tp, tn, fp, fn = loop_assign(output_array, flood_array, ref_array, mask)

    pc = (tp+tn)/(tp+tn+fp+fn)
    b = (tp+fp)/(tp+fn)
    h = tp/(tp+fn)
    n = nrows*ncols
    k = (n*(tp+tn) - ((tp+fp)*(tp+fn) + (fp+tn)*(fn+tn)))/((n**2) - ((tp+fp)*(tp+fn) + (fp+tn)*(fn+tn)))
    f = tp / (tp + fp + fn)

    # Print error matrix
    print(f"\nTrue Positives: {format_number(tp)}")
    print(f"True Negatives: {format_number(tn)}")
    print(f"False Positives: {format_number(fp)}")
    print(f"False Negatives: {format_number(fn)}\n")

    print("Performance Indicators (1 = Perfect):")
    print("-" * 30)
    print(f"Proportion Correct:\t{pc:.3f}")
    print(f"Bias:\t\t\t{b:.3f}")
    print(f"Hit Rate:\t\t{h:.3f}")
    print(f"Kappa:\t\t\t{k:.3f}")
    print(f"Fitness-statistic:\t{f:.3f}\n")

    if not output_file:
        return

    hDriver: gdal.Driver = gdal.GetDriverByName("GTiff")

    out_ds: gdal.Dataset = hDriver.Create(output_file, xsize=ncols, ysize=nrows, bands=1, eType=gdal.GDT_Byte)
    out_ds.SetGeoTransform(geotransform)
    out_ds.SetProjection(flood_projection)
    out_ds.GetRasterBand(1).WriteArray(np.flipud(output_array))
    out_ds.GetRasterBand(1).SetNoDataValue(0)

    out_ds = None

if __name__ == "__main__":
    main()

