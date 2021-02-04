# -*- coding: utf-8 -*-
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
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

from des.join import join_nearest_points
# -

# # Get Small Area boundaries

# +
# --no-clobber skips downloading unless a new version exists
# !wget --no-clobber \
#     -O ../data/external/Small_Areas_Ungeneralised_-_OSi_National_Statistical_Boundaries_-_2015-shp.zip \
#     https://opendata.arcgis.com/datasets/c85e610da1464178a2cd84a88020c8e2_3.zip

# -o forces overwriting
# !unzip -o \
#     -d ../data/external/Small_Areas_Ungeneralised_-_OSi_National_Statistical_Boundaries_-_2015-shp \
#     ../data/external/Small_Areas_Ungeneralised_-_OSi_National_Statistical_Boundaries_-_2015-shp.zip 
# -

dublin_small_area_boundaries = (
    gpd.read_file("../data/external/Small_Areas_Ungeneralised_-_OSi_National_Statistical_Boundaries_-_2015-shp")
    .query("`COUNTYNAME` == ['South Dublin', 'Dún Laoghaire-Rathdown', 'Fingal', 'Dublin City']")
    .loc[:, ["SMALL_AREA", "COUNTYNAME", "geometry"]]
    .to_crs(epsg=2157) # convert to ITM
    .reset_index()
)

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
    .reset_index()
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

capacitymap_stations_df = (
    pd.read_excel("../data/external/MapDetailsDemand.xlsx", engine="openpyxl")
    .assign(
        available_capacity=lambda df: df["Marker"].map(capacity_keys),
        station_name=lambda df: df["Title"].str.split(" - ").str.get(0).str.lower(),
        station_voltages=lambda df: df["Title"].str.split(" - ").str.get(1).str.lower(),
    )
    .dropna(how="all", axis="rows")
    .dropna(how="all", axis="columns")
)

capacitymap_stations = (
    gpd.GeoDataFrame(
        slr_stations_df,
        geometry=gpd.points_from_xy(slr_stations_df.Longitude, slr_stations_df.Latitude),
        crs="epsg:4326",
    )
    .to_crs(epsg=2157)
)
# -

capacitymap_stations_dublin = (
    gpd.sjoin(capacitymap_stations, dublin_admin_county_boundaries)
    .drop(columns="index_right")
)

# # Join Small Areas to Nearest SLR Station

dublin_small_area_stations = (
    join_nearest_points(dublin_small_area_boundaries.geometry.centroid, capacitymap_stations_dublin)
    .merge(dublin_small_area_boundaries, left_index=True, right_index=True)
    .loc[:, ["station_name", "Marker", "geometry"]]
 )

# # Plot 

# +
f, ax = plt.subplots(figsize=(100,100))

dublin_admin_county_boundaries.plot(ax=ax, facecolor="none", edgecolor='red')
dublin_small_area_stations.plot(ax=ax, column="station_name", cmap="binary")
slr_stations_dublin.plot(
    ax=ax,
    color=slr_stations_dublin["Marker"],
    markersize=5000,
)
slr_stations_dublin.apply(
    lambda gdf: ax.annotate(
        text=gdf["station_name"],
        xy=gdf.geometry.centroid.coords[0],
        ha='center',
        fontsize=12,
        color="black",
        path_effects=[pe.withStroke(linewidth=2, foreground="white")],
    ),
    axis=1,
);
# -

f.savefig("../data/outputs/small-areas-linked-to-nearest-capacitymap-station.png")

dublin_small_area_stations.to_file(
    "../data/outputs/small-areas-linked-to-nearest-capacitymap-station.geojson",
    driver="GeoJSON",
)
