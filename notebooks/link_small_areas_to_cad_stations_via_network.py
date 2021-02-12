# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.10.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Run `link_cad_stations_to_map_stations.ipynb` first so that Small Areas can be linked to Map stations via CAD stations

# # Setup

# +
from os import listdir

import pandas as pd
import geopandas as gpd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import momepy
from shapely.geometry import box

from des import distance
from des import download
from des import io
from des import join
from des import paths
from des import plot
from des import unzip

data_dir = "../data"
cad_data = "/home/wsl-rowanm/Data/ESBdata_20200124"
show_plots = True
# -

# # Get Small Area boundaries

download.download(
    url="https://opendata.arcgis.com/datasets/c85e610da1464178a2cd84a88020c8e2_3.zip",
    to_filepath=f"{data_dir}/Small_Areas_Ungeneralised_-_OSi_National_Statistical_Boundaries_-_2015-shp.zip",
)
unzip.unzip(
    filename=f"{data_dir}/Small_Areas_Ungeneralised_-_OSi_National_Statistical_Boundaries_-_2015-shp.zip",
    extract_dir=f"{data_dir}/Small_Areas_Ungeneralised_-_OSi_National_Statistical_Boundaries_-_2015-shp",
)

small_areas = io.read_dublin_small_areas(f"{data_dir}/Small_Areas_Ungeneralised_-_OSi_National_Statistical_Boundaries_-_2015-shp")

# # Get Local Authority boundaries

dublin_admin_county_boundaries = io.read_dublin_admin_county_boundaries(
    f"{data_dir}/dublin_admin_county_boundaries"
)

# # Get MV Network and Get 38kV, 110kV & 220kV  stations
#
# ... there is no 400kV station in Dublin

ireland_mv_index = io.read_mv_index(f"{cad_data}/Ancillary Data/mv_index.dgn")

dublin_boundary = gpd.GeoSeries(box(695000, 715000, 740000, 771000)).rename("geometry").to_frame()
dublin_mv_index = gpd.sjoin(ireland_mv_index, dublin_boundary, op="within")
dublin_mv_network_filepaths = [
    f"{cad_data}/Dig Request Style/MV-LV Data/{index}.dgn" for index in dublin_mv_index.Text
]

mv_network_lines = io.read_network(dublin_mv_network_filepaths, levels=[10, 11, 14]).reset_index(drop=True).explode()

hv_network_filepaths = [
    f"{cad_data}/Dig Request Style/HV Data/{filename}"
    for filename in listdir(f"{cad_data}/Dig Request Style/HV Data/")
]

hv_stations_ireland = io.read_network(hv_network_filepaths, levels=[20, 30, 40])
hv_stations_dublin = (
    gpd.sjoin(
        hv_stations_ireland,
        dublin_admin_county_boundaries,
        op="within",
    )
    .drop(columns=["index_right", "COUNTYNAME"])
    .reset_index(drop=True)
    .assign(station_id=lambda gdf: gdf.index)
)

if show_plots:
    
    ax = dublin_admin_county_boundaries.plot(figsize=(15,15), facecolor="orange", edgecolor="orange")
    mv_network_lines.plot(ax=ax)
    dublin_boundary.plot(ax=ax, facecolor="none", edgecolor="cyan")
    hv_stations_dublin.plot(ax=ax, color="black")

# # Link Each Small Area Centroid to a Station via Network
#
# Use `networkx` to find the station that is closest along the network to each small area centroid:
# - Convert `geopandas` `GeoDataFrame` to `networkx` `MultiGraph` via `momepy` for network analysis
# - Extract the largest unbroken network
# - Extract all stations and small areas near the network
# - Trace the path from each small area centroid to the nearest station along the network

G = momepy.gdf_to_nx(mv_network_lines, approach="primal")

G_largest = distance.get_largest_subgraph(G)

G_largest_nodes, G_largest_edges, G_largest_sw = momepy.nx_to_gdf(
    G_largest,
    points=True,
    lines=True,
    spatial_weights=True
)

G_largest_edges_buffered = (
    G_largest_edges[["geometry"]]
    .assign(geometry=lambda gdf: gdf.buffer(750))
    .dissolve()
)

small_areas_near_g_largest = join.centroids_within(small_areas, G_largest_edges_buffered)

hv_stations_near_g_largest = gpd.sjoin(
    hv_stations_dublin,
    G_largest_edges_buffered,
    op="within",
).drop(columns="index_right").reset_index(drop=True)

hv_stations_snapped_to_g_largest = join.snap_points_to_network(G_largest, hv_stations_near_g_largest)

shortest_paths = paths.get_network_paths_between_points(
    G=G_largest,
    orig_points=small_areas_near_g_largest,
    dest_points=hv_stations_snapped_to_g_largest,
)

if show_plots:
    n = 0 # change 'n' to change the path to be displayed
    f, ax = plt.subplots(figsize=(15, 15))

    dublin_admin_county_boundaries.plot(edgecolor="red", facecolor="orange", ax=ax)

    positions = {z: [z[0], z[1]] for z in list(G.nodes)}
    nx.draw(G_largest, positions, node_size=5, ax=ax, node_color="none", edge_color="teal")

    x, y = zip(*shortest_paths[n][1])
    ax.plot(x, y, c="k", lw=20, alpha=0.5)
    ax.scatter(
        small_areas_near_g_largest.iloc[n].geometry.centroid.x,
        small_areas_near_g_largest.iloc[n].geometry.centroid.y,
        color="green",
        s=50,
    )
    ax.scatter(x[0], y[0], color="red", s=50)
    ax.scatter(x[-1], y[-1], color="green", s=50)
    ax.scatter(
        hv_stations_snapped_to_g_largest.geometry.x,
        hv_stations_snapped_to_g_largest.geometry.y,
        color="black",
        s=10,
    )

# # Link Small Areas to stations

small_areas_linked_to_stations_via_network = paths.extract_nearest_dest(
    shortest_paths,
    small_areas_near_g_largest,
    hv_stations_snapped_to_g_largest
).drop_duplicates(subset="SMALL_AREA")


small_areas_remaining_centroids = gpd.GeoDataFrame(
    small_areas
    .merge(small_areas_near_g_largest, how="left", indicator=True)
    .query("`_merge` == 'left_only'")
    .drop(columns="_merge")
    .assign(geometry=lambda gdf: gdf.geometry.centroid)
)

small_areas_linked_to_stations_via_nearest = (
    join.join_nearest_points(small_areas_remaining_centroids, hv_stations_dublin)
    .drop(columns=["geometry", "COUNTYNAME"])
    .merge(small_areas)
    .drop_duplicates(subset="SMALL_AREA")
)

small_areas_linked_to_stations = pd.concat(
    [
        small_areas_linked_to_stations_via_network,
        small_areas_linked_to_stations_via_nearest,
    ]
).drop_duplicates(subset="SMALL_AREA")

if show_plots:
    
    f, ax = plt.subplots(figsize=(120,120))

    small_areas_linked_to_stations_via_network.plot(
        ax=ax,
        edgecolor="white",
        color="teal",
    )
    small_areas_linked_to_stations_via_network.apply(
        lambda x: ax.annotate(
            text=x["station_id"],
            xy=x.geometry.centroid.coords[0],
            va='bottom',
            color="white",
            path_effects=[pe.withStroke(linewidth=2, foreground="orange")],
        ),
        axis=1,
    )
    small_areas_linked_to_stations_via_nearest.plot(
        ax=ax,
        edgecolor="white",
        facecolor="cyan",
    )
    small_areas_linked_to_stations_via_nearest.apply(
        lambda x: ax.annotate(
            text=x["station_id"],
            xy=x.geometry.centroid.coords[0],
            va='bottom',
            color="white",
            path_effects=[pe.withStroke(linewidth=2, foreground="brown")],
        ),
        axis=1,
    )

    G_positions = {z: [z[0], z[1]] for z in list(G.nodes)}
    nx.draw(G, G_positions, node_size=5, ax=ax, node_color="none", edge_color="blue")
    G_largest_positions = {z: [z[0], z[1]] for z in list(G_largest.nodes)}
    nx.draw(
        G_largest,
        G_largest_positions,
        node_size=5,
        ax=ax,
        node_color="none",
        edge_color="red",
    )
    
    hv_stations_dublin.plot(ax=ax, color="black")
    hv_stations_dublin.apply(
        lambda x: ax.annotate(
            text=x["station_id"],
            xy=x.geometry.centroid.coords[0],
            va='top',
            color="white",
            path_effects=[pe.withStroke(linewidth=2, foreground="black")],
        ),
        axis=1,
    )
    hv_stations_dublin.apply(
        lambda x: ax.annotate(
            text=x["station_name"],
            xy=x.geometry.centroid.coords[0],
            va='bottom',
            color="white",
            path_effects=[pe.withStroke(linewidth=2, foreground="black")],
        ),
        axis=1,
    )

    # plt.legend(["Small Area Station IDs", "CAD Stations linked to Heat Map"], prop={'size': 25});
    plt.legend(list(range(5)), prop={'size': 25});

#
# # Save

small_areas_linked_to_stations.to_file(
    f"{data_dir}/small-areas-linked-to-map-stations.geojson",
    driver="GeoJSON",
)
