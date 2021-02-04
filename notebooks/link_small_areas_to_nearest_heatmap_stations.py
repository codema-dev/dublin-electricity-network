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

from des import clean
from des import convert
from des import extract
from des import io
from des import join
from des import plot

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

dublin_small_area_boundaries = io.read_dublin_small_areas(
    "../data/external/Small_Areas_Ungeneralised_-_OSi_National_Statistical_Boundaries_-_2015-shp"
)

# # Get Dublin LA boundaries

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

dublin_admin_county_boundaries = io.read_dublin_admin_county_boundaries(
    "../data/external/dublin_admin_county_boundaries"
)

# # Get Dublin Heatmap stations

heatmap_raw = io.read_heatmap("../data/external/heatmap-download-version-nov-2020.xlsx")
heatmap_clean = clean.clean_heatmap(heatmap_raw)
heatmap_gdf = convert.convert_to_gdf_via_lat_long(heatmap_clean)
heatmap_dublin = extract.extract_polygon(
    heatmap_gdf,
    dublin_admin_county_boundaries
)
heatmap_dublin_hv = heatmap_dublin.query("station_name != 'mv/lv'")

# # Join Small Areas to Nearest Heatmap Station

dublin_small_area_stations = (
    join.join_nearest_points(dublin_small_area_boundaries.geometry.centroid, heatmap_dublin_hv)
    .merge(dublin_small_area_boundaries, left_index=True, right_index=True)
    .loc[:, ["station_name", "geometry"]]
 )

# # Plot 

fig = plot.plot_small_areas_linked_to_stations(
    small_areas=dublin_small_area_stations,
    stations=heatmap_dublin_hv,
)

# # Save

fig.savefig("../data/outputs/small-areas-linked-to-nearest-heatmap-station.png")

dublin_small_area_stations.to_file(
    "../data/outputs/small-areas-linked-to-nearest-heatmap-station.geojson",
    driver="GeoJSON",
)
