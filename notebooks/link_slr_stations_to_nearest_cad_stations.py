# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from des.join import join_nearest_points
# -

# # Get LA boundaries

# +
# --no-clobber skips downloading unless a new version exists
# !wget --no-clobber \
#     -O ../data/external/dublin_admin_county_boundaries.zip
    https://zenodo.org/record/4446778/files/dublin_admin_county_boundaries.zip

# -o forces overwriting
# !unzip -o \
#     -d ../data/external \
#     ../data/external/dublin_admin_county_boundaries.zip 
# -

dublin_admin_county_boundaries = (
    gpd.read_file("../data/external/dublin_admin_county_boundaries")
    .to_crs(epsg=2157) # read & convert to ITM or epsg=2157
    .loc[:, ["COUNTYNAME", "geometry"]]
)

# # Get Network data

# Must be downloaded from the Codema Google Shared Drive or <span style="color:red">**requested from the ESB**</span>

cad_data = "/home/wsl-rowanm/Data/dublin-electricity-network/"

# ## Get 38kV, 110kV & 220kV  stations
#
# ... there is no 400kV station in Dublin

cad_stations = (
    gpd.read_parquet(f"{cad_data}/dublin_hv_network.parquet")
    .query("`Level` == [20, 30, 40]")
    .explode() # un-dissolve station locations from multipoint to single points
    .reset_index()
    .drop(columns=["COUNTY", "index_right", "level_1"])
)

# # Get SLR stations

# +
# From https://www.esbnetworks.ie/demand-availability-capacity-map
capacity_keys = {
    "Blue": "mixed",
    "Green": "significant",
    "Orange": "limited",
    "Red": "none",
}

slr_stations_df = (
    pd.read_excel("../data/external/MapDetailsDemand.xlsx", engine="openpyxl")
    .assign(
        available_capacity=lambda df: df["Marker"].map(capacity_keys),
        station_name=lambda df: df["Title"].str.split(" - ").str.get(0).str.lower(),
        station_voltages=lambda df: df["Title"].str.split(" - ").str.get(1).str.lower(),
    )
    .dropna(how="all", axis="rows")
    .dropna(how="all", axis="columns")
)

slr_stations = (
    gpd.GeoDataFrame(
        slr_stations_df,
        geometry=gpd.points_from_xy(slr_stations_df.Longitude, slr_stations_df.Latitude),
        crs="epsg:4326",
    )
    .to_crs(epsg=2157)
)
# -

slr_stations_dublin = (
    gpd.sjoin(slr_stations, dublin_admin_county_boundaries)
    .drop(columns="index_right")
)

slr_station_voltages = (
    pd.read_excel("../data/external/slr-2019-20.xlsx", engine="openpyxl")
    .ffill()
    .drop_duplicates()
    .assign(station=lambda df: df.station.str.lower())
)

# # Plot CAD stations vs SLR stations

f, ax = plt.subplots(figsize=(25,25))
dublin_admin_county_boundaries.boundary.plot(ax=ax, edgecolor='red')
slr_stations_dublin.geometry.buffer(1000).plot(ax=ax, color=slr_stations_dublin["Marker"])
slr_stations_dublin.apply(
    lambda gdf: ax.annotate(
        text=gdf["station_name"],
        xy=gdf.geometry.centroid.coords[0],
        ha='center',
        fontsize=10,
    ),
    axis=1,
)
cad_stations.plot(ax=ax, markersize=10, color='blue');

# ## Link stations to nearest geocoded station

columns = [
    "Level",
    "Type",
    "GraphicGroup",
    "ColorIndex",
    "Weight",
    "Style",
    "EntityNum",
    "MSLink",
    "Text",
    "geometry",
    "Marker",
    "Title",
    "Description",
    "VoltageClass",
    "available_capacity",
    "station_name",
    "station_voltages",
    "COUNTYNAME",
]
cad_stations_linked = gpd.GeoDataFrame(
    join_nearest_points(cad_stations, slr_stations_dublin)
    .loc[:, columns]
)

dublin_admin_county_boundaries.to_file(
    "../data/outputs/dublin-admin-county-boundaries.geojson",
    driver="GeoJSON",
)

slr_stations_dublin.to_file(
    "../data/outputs/capacitymap-stations.geojson",
    driver="GeoJSON",
)

cad_stations_linked.to_file(
    "../data/outputs/cad-38-&-110-kv-stations-linked-to-nearest-capacitymap-station.geojson",
    driver="GeoJSON",
)
