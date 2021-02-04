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
import pickle

import geopandas as gpd
import matplotlib.pyplot as plt
import momepy
import networkx as nx
import pandas as pd
from tqdm import tqdm

from des.distance import get_nearest_node
from des.distance import get_nearest_nodes
from des.distance import get_network_paths_between_points
from des.distance import get_network_paths_between_points_lazy
from des.plot import plot_gdf_vs_nx
from des.plot import plot_path_n
from des.plot import plot_paths_to_files
# -

# # Get Dublin Small Area Boundaries

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

small_areas = (
    gpd.read_file("../data/external/Small_Areas_Ungeneralised_-_OSi_National_Statistical_Boundaries_-_2015-shp")
    .query("`COUNTYNAME` == ['South Dublin', 'Dún Laoghaire-Rathdown', 'Fingal', 'Dublin City']")
    .loc[:, ["SMALL_AREA", "COUNTYNAME", "geometry"]]
    .to_crs(epsg=2157) # convert to ITM
)

# # Get Dublin Local Authority admin boundaries

# +
# --no-clobber skips downloading unless a new version exists
# !wget --no-clobber \
#     -O ../data/external/dublin_admin_county_boundaries.zip \
#     https://zenodo.org/record/4446778/files/dublin_admin_county_boundaries.zip

# -o forces overwriting
# !unzip -o \
#     -d ../data/external \
#     ../data/external/dublin_admin_county_boundaries.zip 
# -

dublin_admin_county_boundaries = (
    gpd.read_file("../data/external/dublin_admin_county_boundaries")
    .to_crs(epsg=2157) # read & convert to ITM or epsg=2157
)

# # Get Dublin Station Totals

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
slr_stations = gpd.GeoDataFrame(
    slr_stations_df,
    geometry=gpd.points_from_xy(slr_stations_df.Longitude, slr_stations_df.Latitude),
    crs="epsg:4326",
).to_crs(epsg=2157)
# -

slr_stations_dublin = (
    gpd.sjoin(slr_stations, dublin_admin_county_boundaries)
    .drop(columns="index_right")
    [["station_name"]]
    .copy()
    .reset_index(drop=True)
)

slr_station_voltage_levels = (
    pd.read_excel("../data/external/slr-2019-20.xlsx", engine="openpyxl")
    .loc[:, ["station", "voltage_level"]]
    .assign(station=lambda df: df.station.str.lower())
    .ffill()
    .drop_duplicates()
)

slr_stations_dublin_grouped = slr_stations_dublin.merge(
    slr_station_voltage_levels,
    left_on="station_name",
    right_on="station",
    indicator=True,
    how="left",
)

dublin_slr_stations_38kv = slr_stations_dublin_grouped.query("`voltage_level` == '38/MV station'")
dublin_slr_stations_110kv = slr_stations_dublin_grouped.query("`voltage_level` == '110/MV station'")
dublin_slr_stations_38_and_110kv = slr_stations_dublin_grouped.query(
    "`voltage_level` == ['38/MV station', '110/MV station']"
)

# # Get MV network lines
#
# ... Must be downloaded from the Codema Google Shared Drive or <span style="color:red">**requested from the ESB**</span>

network_data = "/home/wsl-rowanm/Data/dublin-electricity-network/"

mv_network_lines = (
    gpd.read_parquet(f"{network_data}/dublin_lvmv_network.parquet")
    .query("`Level` == [10, 11, 14]")
    .explode()
    .reset_index(drop=True)
    .loc[:, ["geometry"]]
)

# # Link SLR stations to Small Areas via Network

# ## Convert to NetworkX
#
#

G = momepy.gdf_to_nx(mv_network_lines, approach="primal")

# Filter out all of the tiny sub-networks

largest_components = [
    component for component in sorted(nx.connected_components(G), key=len, reverse=True)
    if len(component) > 5
]

G_top = nx.compose_all(G.subgraph(component) for component in tqdm(largest_components))

# ## Link SA centroids to nearest station on MV network

orig_points = pd.DataFrame(
    {
        "x": small_areas.geometry.centroid.x,
        "y": small_areas.geometry.centroid.y,
    }
)

dest_points = pd.DataFrame(
    {
        "x": dublin_slr_stations_38_and_110kv.geometry.x,
        "y": dublin_slr_stations_38_and_110kv.geometry.y,
    }
)

paths = get_network_paths_between_points(
    G=G_top,
    orig_points=orig_points,
    dest_points=dest_points,
)

plot_path_n(
    G=G_top,
    paths=paths,
    orig_points=orig_points,
    dest_points=dest_points,
    boundaries=dublin_admin_county_boundaries.boundary,
    n=1,
)

plot_paths_to_files(
    G=G_top,
    paths=paths,
    orig_points=orig_points,
    dest_points=dest_points,
    boundaries=dublin_admin_county_boundaries.boundary,
    dirpath="../data/outputs/sa-centroids-to-slr-38kv-&-110kv-stations-via-mv-network",
)
