############ This code is developed by Nicolas Pucino and modified by Yakup Niyazi, for the artcile titled "Status of global seafloor mapping effort and priority areas for future mapping". ############

############################### Import Libraries ############################### 
import os
import glob
import rioxarray as rx
import xarray as xr
from shapely.geometry import mapping
import rasterio as ras
from rasterio.features import geometry_mask
from rasterio.io import MemoryFile
from rasterio.plot import show
from rasterio.mask import mask
import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from plottable import Table,ColDef
import matplotlib.pyplot as plt


########################################## Pixel-level information extraction ##########################################


############################### Paths to input files ############################### 
bathy_raster_path = r'Bathymetric_raster'
eez_poly_path = r"EEZ"
ocean_depth_poly_path = r"Ocean-depth"
high_seas_poly_path = r"High-seas"
grid_path = r"Grid"
years = ['2019', '2020', '2021', '2022', '2023', '2024']

############################### Output folders for saving results ############################### 
output_folder_dfs = r'output_dfs'
output_folder_images = r'output_images'

############################### Load input data for processing ############################### 
bathy_raster = ras.open(bathy_raster_path)

eez_poly = gpd.read_file(eez_poly_path)
ocean_depth_poly = gpd.read_file(ocean_depth_poly_path)
high_seas_poly = gpd.read_file(high_seas_poly_path)
grid = gpd.read_file(grid_path)

############################### Dictionary to map EEZ and ocean names to unique codes for raster band identification ############################### 
eez_dict = {name: code for code, name in enumerate(eez_poly.SOVEREIGN1.unique())}
ocean_dict = {name: code for code, name in enumerate(ocean_depth_poly.name.unique())}
high_seas_dict = {name: code for code, name in enumerate(high_seas_poly.name.unique())}
############################### Reproject shapefiles to match the CRS of the bathymetry raster ############################### 
eez_poly = eez_poly.to_crs(bathy_raster.crs)
ocean_depth_poly = ocean_depth_poly.to_crs(bathy_raster.crs)
high_seas_poly = high_seas_poly.to_crs(bathy_raster.crs)

############################### Process each tile in the grid iteratively ############################### 
for i_tile in tqdm(range(grid.shape[0]), desc="Processing Tiles", unit='tile'):
    tile_test = grid.iloc[[i_tile]]  # Select current tile for processing

    for year_in in years:
       ################ Load the method raster for the current year ################
        meth_raster_path = rf'GEBCO_{year_in}_TID.tif'
        meth_raster = ras.open(meth_raster_path)

       ################ Define output paths for the DataFrame and image ################
        out_folder_df = os.path.join(output_folder_dfs, f"out_dfs_{year_in}")
        os.makedirs(out_folder_df, exist_ok=True)
        out_folder_image = os.path.join(output_folder_images, f"out_tiles_{year_in}")
        os.makedirs(out_folder_image, exist_ok=True)

        ################ Read data from bathymetry and method rasters ################
        bathy = bathy_raster.read(1)
        meth = meth_raster.read(1)

        print(f"WORKING ON TILE: {i_tile}")

        ################ Step 1: Intersect the current tile with EEZ and ocean polygons and high sea for subsetting ################
        eez_inter = eez_poly[eez_poly.intersects(tile_test.geometry.iloc[0])]
        eez_inter_dissolved = eez_inter.dissolve(by="SOVEREIGN1").reset_index()
        ocean_inter = ocean_depth_poly[ocean_depth_poly.intersects(tile_test.geometry.iloc[0])]
        high_seas_inter= high_seas_poly[high_seas_poly.intersects(tile_test.geometry.iloc[0])]

        ################ Prepare list to hold clipped rasters ################
        images = []

        ################ Step 2: Clip both method and bathymetry rasters to tile extent ################ 
        rasters = [meth_raster, bathy_raster]
        for raster_in in rasters:
            geom = tile_test.to_crs(raster_in.crs).iloc[0].loc['geometry']
            out_image, out_transform = mask(raster_in, geom, crop=True)
            out_meta = raster_in.meta.copy()

            ################ Update metadata to match clipped raster dimensions and transformation ################ 
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })

            ################  Write the clipped raster data to an in-memory file ################ 
            single_memfile = MemoryFile()
            with single_memfile.open(**out_meta) as mem_writer:
                mem_writer.write(out_image)

            ################ Reopen in-memory file to append it to the images list ################ 
            mem_reader = single_memfile.open()
            images.append(mem_reader)

        ################ Step 3: Load clipped rasters as xarray DataArrays ################ 
        xds_meth = xr.open_dataarray(images[0].name, engine='rasterio')
        xds_depth = xr.open_dataarray(images[1].name, engine='rasterio')

        ################ Step 4: Reproject depth raster to match the method raster ################ 
        xds_depth_match = xds_depth.rio.reproject_match(xds_meth, resampling=ras.enums.Resampling.nearest)

        ################ Step 5: Process intersected EEZ polygons and assign each a unique code ################ 
        if eez_inter_dissolved.shape[0] > 0:
            clips_svrn = []
            for sovereign_state in eez_inter_dissolved.SOVEREIGN1.unique():
                ################ Clip method raster for each EEZ region ################ 
                eez_zone_sovr_geom = [mapping(eez_inter_dissolved.query(f"SOVEREIGN1 == '{sovereign_state}'").iloc[0]['geometry'])]
                clipped_tmp = xds_meth.rio.clip(geometries=eez_zone_sovr_geom, crs=xds_meth.rio.crs, drop=False)
                clipped_tmp = clipped_tmp.where(np.isnan(clipped_tmp), eez_dict[sovereign_state])
                clips_svrn.append(clipped_tmp)

            ################ Stack results into a DataArray for EEZ regions ################ 
            stacked_da_svrn = xr.concat(clips_svrn, dim="sovereign_state")
            sovereign_band = stacked_da_svrn.max(dim="sovereign_state", skipna=True)
        else:
            sovereign_band = xds_meth.copy()
            sovereign_band[:] = np.nan

        ################ Step 6: Process intersected ocean polygons and assign each a unique code ################ 
        if ocean_inter.shape[0] > 0:
            clips_ocean = []
            for i in range(ocean_inter.shape[0]):
                ocean_i = ocean_inter.iloc[[i]]
                ocean_name_i = ocean_i['name'].values[0]
                ocean_geom_i = [mapping(ocean_i.iloc[0]['geometry'])]

                clipped_tmp = xds_meth.rio.clip(geometries=ocean_geom_i, crs=xds_meth.rio.crs, drop=False)
                clipped_tmp = clipped_tmp.where(np.isnan(clipped_tmp), ocean_dict[ocean_name_i])
                clips_ocean.append(clipped_tmp)

            ################ Stack results into a DataArray for ocean regions ################ 
            stacked_da_ocean = xr.concat(clips_ocean, dim="ocean")
            ocean_band = stacked_da_ocean.max(dim="ocean", skipna=True)
        else:
            ocean_band = xds_meth.copy()
            ocean_band[:] = np.nan

        ################ Step 7: Process intersected high seas polygons and assign each a unique code ################ 
        if high_seas_inter.shape[0] > 0:
            clips_high_seas = []
            for i in range(high_seas_inter.shape[0]):
                high_seas_i = high_seas_inter.iloc[[i]]
                high_seas_name_i = high_seas_i['name'].values[0]
                high_seas_geom_i = [mapping(high_seas_i.iloc[0]['geometry'])]

                clipped_tmp = xds_meth.rio.clip(geometries=high_seas_geom_i, crs=xds_meth.rio.crs, drop=False)
                clipped_tmp = clipped_tmp.where(np.isnan(clipped_tmp), high_seas_dict[high_seas_name_i])
                clips_high_seas.append(clipped_tmp)

            ################ Stack results into a DataArray for high_seas ################ 
            stacked_da_high_seas = xr.concat(clips_high_seas, dim="high_seas")
            high_seas_band = stacked_da_high_seas.max(dim="high_seas", skipna=True)
        else:
            high_seas_band = xds_meth.copy()
            high_seas_band[:] = np.nan
        ################ Step 8: Combine all data into a multi-band xarray DataArray ################ 
        band_names = ['method', 'depth', 'eez', 'ocean', 'high_seas']
        stacked = xr.concat([xds_meth, xds_depth_match, sovereign_band, ocean_band, high_seas_band], dim='band')
        stacked = stacked.assign_coords(band=band_names)

        ################ Set description attributes for each band ################ 
        for i, band_name in enumerate(band_names):
            stacked[i].attrs['description'] = band_name

        ################ Step 9: Reproject to Mollweide projection ################ 
        stacked_mollweide = stacked.rio.reproject("ESRI:54009")

        ################ Step 10: Reshape data to find unique pixel combinations ################ 
        reshaped = stacked_mollweide.stack(pixel=("y", "x")).transpose("band", "pixel").values
        reshaped_no_nan = np.nan_to_num(reshaped, nan=-9999)
        unique_combinations, counts = np.unique(reshaped_no_nan.T, axis=0, return_counts=True)

        ################ Step 11: Create a DataFrame to store unique combinations and pixel area ################ 
        df = pd.DataFrame(unique_combinations, columns=['method', 'depth', 'eez', 'ocean', 'high_seas_band'])
        df['num_pixels'] = counts
        df.replace(-9999, np.nan, inplace=True)
        df = df[['num_pixels', 'ocean', 'depth', 'method', 'eez']]

        ################ Calculate pixel area and area in km² ################ 
        transform = stacked_mollweide.rio.transform()
        pixel_width, pixel_height = transform[0], abs(transform[4])
        pixel_area_m2 = pixel_width * pixel_height
        df['area_m2'] = df['num_pixels'] * pixel_area_m2
        df['area_km2'] = df['area_m2'] / 1e6

        ################ Step 12: Export stacked raster as GeoTIFF and save DataFrame ################ 
        out_image_path = os.path.join(out_folder_image, f"{i_tile}_tile_multistack.tif")
        with ras.open(out_image_path, 'w', driver='GTiff', crs=stacked.rio.crs,
                      transform=stacked.rio.transform(), width=stacked.rio.width,
                      height=stacked.rio.height, count=stacked.shape[0], dtype=stacked.dtype) as dst:
            for i in range(stacked.shape[0]):
                dst.write(stacked[i, :, :].values, i + 1)
                dst.set_band_description(i + 1, stacked[i].attrs['description'])


########################################## Assembling outputs from extraction ##########################################
eez_poly=gpd.read_file(r"ocean_data\World_EEZ_Depth_Class.shp")
eez_dict={name:code for code,name in enumerate(eez_poly.SOVEREIGN1.unique())} 

cean_dict={'Southern Ocean': 0,
 'South Atlantic Ocean': 1,
 'South Pacific Ocean': 2,
 'North Pacific Ocean': 3,
 'South China and Easter Archipelagic Seas': 4,
 'Indian Ocean': 5,
 'Mediterranean Region': 6,
 'Baltic Sea': 7,
 'North Atlantic Ocean': 8,
 'Arctic Ocean': 9}

high_seas_dict={'Southern Ocean': 0,
 'South Atlantic Ocean': 1,
 'South Pacific Ocean': 2,
 'North Pacific Ocean': 3,
 'South China and Easter Archipelagic Seas': 4,
 'Indian Ocean': 5,
 'Mediterranean Region': 6,
 'Baltic Sea': 7,
 'North Atlantic Ocean': 8,
 'Arctic Ocean': 9}

measurement_dict = {
    0: "Land",
    10: "Singlebeam echo-sounder",
    11: "Multibeam echo-sounder",
    12: "Seismic methods",
    13: "Isolated sounding",
    14: "ENC sounding",
    15: "Bathymetric lidar",
    16: "Optical light sensor",
    17: "Combination of direct methods",
    40: "Satellite gravity data",
    41: "Computer algorithm interpolation",
    42: "Bathymetric contours from charts",
    43: "Bathymetric contours from ENC",
    44: "Bathymetric sounding with interpolation",
    45: "Helicopter gravity data",
    46: "Iceberg freeboard calculation",
    70: "Pre-generated grid",
    71: "Unknown source",
    72: "Steering points"
}
df_dirs=[r"out_dfs_2019",r"out_dfs_2020",r"out_dfs_2021",
        r"out_dfs_2022",r"out_dfs_2023",r"out_dfs_2024"]
df_name_years=['2019','2020','2021','2022','2023','2024']

df_all=pd.DataFrame()

for dir_i, year_i in zip(df_dirs,df_name_years):
    print(f"working on {year_i}")
    for i in tqdm(glob.glob(f"{dir_i}\\*.csv")):
        df_tmp=pd.read_csv(i)
        df_tmp['year']=year_i
        df_all=pd.concat([df_all,df_tmp], ignore_index=True)

ocean_dict_r={c:n for n,c in ocean_dict.items()}
eez_dict_r={c:n for n,c in eez_dict.items()}
high_seas_r={c:n for n,c in high_seas_dict.items()}

depth_name_dict={1:'0-200', 2:'200-1000', 3:'1000-3000', 4:'3000-6000', 5:'>6000'}

land=0
direct=[10,11,12,13,14,15,16,17]
indirect=[40,41,42,43,44,45,46]
unknown=[70,71,72]

# Create the dictionary
meas_type_dict = {land: 'Land'}
meas_type_dict.update({key: 'Direct' for key in direct})
meas_type_dict.update({key: 'Indirect' for key in indirect})
meas_type_dict.update({key: 'Unknown' for key in unknown})

df_all['ocean_name']=df_all.ocean.map(ocean_dict_r)
df_all['eez_name']=df_all.eez.map(eez_dict_r)
df_all['depth_name']=df_all.depth.map(depth_name_dict)
df_all['meth_type']=df_all.method.map(meas_type_dict)
df_all['meth_desc']=df_all.method.map(measurement_dict)
df_all['high_seas_name']=df_all.high_seas.map(high_seas_dict_r)
df_all


