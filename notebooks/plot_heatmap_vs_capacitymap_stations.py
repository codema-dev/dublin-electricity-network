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
import geopandas as gpd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import pandas as pd

from des import io
from des import clean
from des import convert
from des import extract
from des import plot
# -

# # Get Dublin LA boundaries

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

dublin_admin_county_boundaries = io.read_dublin_admin_county_boundaries(
    "../data/external/dublin_admin_county_boundaries"
)

# # Get Dublin Capacitymap substations

capacitymap_raw = io.read_capacitymap("../data/external/MapDetailsDemand.xlsx")
capacitymap_clean = clean.clean_capacitymap(capacitymap_raw)
capacitymap_gdf = convert.convert_to_gdf_via_lat_long(capacitymap_clean)
capacitymap_dublin = extract.extract_polygon(
    capacitymap_gdf,
    dublin_admin_county_boundaries
)
capacitymap_dublin_station_names = capacitymap_dublin.loc[:, "station_name"]

# # Get Dublin Heatmap substations

heatmap_raw = io.read_heatmap("../data/external/heatmap-download-version-nov-2020.xlsx")
heatmap_clean = clean.clean_heatmap(heatmap_raw)
heatmap_gdf = convert.convert_to_gdf_via_lat_long(heatmap_clean)
heatmap_dublin = extract.extract_polygon(
    heatmap_gdf,
    dublin_admin_county_boundaries
)
heatmap_dublin_station_names = (
    heatmap_dublin.query("station_name != 'mv/lv'")
    .loc[:, "station_name"]
    .reset_index(drop=True)
)

# # Find exclusive stations 

merged_stations = pd.merge(
    capacitymap_dublin_station_names,
    heatmap_dublin_station_names,
    indicator=True,
    how="outer",
)

merged_stations.query("`_merge` == 'right_only'")

merged_stations.query("`_merge` == 'left_only'")

# # Plot

fig = plot.plot_heatmap_vs_capacitymap(
    heatmap=heatmap_dublin.query("station_name != 'mv/lv'"),
    capacitymap=capacitymap_dublin,
    boundaries=dublin_admin_county_boundaries,
)

fig.savefig("../data/outputs/capacitymap-vs-heatmap-stations.png")
