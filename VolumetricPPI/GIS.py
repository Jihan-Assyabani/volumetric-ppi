import rasterio
from rasterio.transform import from_origin
import numpy as np

def to_raster(X, Y, Z, CRS, save_path, print_info=True):
    """ 
    This function is used to export meshed lidar analysis result to raster.
    
    Input:
        X, Y: 2D arrays of coordinates (meshgrid)
        Z: 2D array of values (e.g., wind speed, turbulence intensity)
        CRS: Coordinate Reference System (e.g., 'EPSG:4326' for WGS 84)
        save_path: Path to save the raster file (e.g., 'output.tif')

    Output:
        Saves a GeoTIFF raster file to the destination save_path.
    """

    Z = np.flipud(Z)

    # Get pixel size from mesh
    pixel_size_x = X[0,1] - X[0,0]
    pixel_size_y = Y[1,0] - Y[0,0]

    # Create transform — note: raster origin is top-left
    transform = from_origin(
        west=X.min(),       # left boundary
        north=Y.max(),      # top boundary
        xsize=pixel_size_x, # pixel width
        ysize=pixel_size_y  # pixel height
    )

    # Save to GeoTIFF
    with rasterio.open(
        save_path,  # file name
        'w',
        driver='GTiff',
        height=Z.shape[0],
        width=Z.shape[1],
        count=1,
        dtype=Z.dtype,
        crs=CRS, #coordinate reference system
        transform=transform
    ) as dst:
        dst.write(Z, 1)

    if print_info:
        print(f"Raster saved to {save_path}")
