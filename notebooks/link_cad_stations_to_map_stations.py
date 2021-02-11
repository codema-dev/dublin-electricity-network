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

from des import io
from des import join
# -

data_dir = "../data"

# # Get LA boundaries

# +
# --no-clobber skips downloading unless a new version exists
# !wget --no-clobber \
#     -O {data_dir}/external/dublin_admin_county_boundaries.zip \
#     https://zenodo.org/record/4446778/files/dublin_admin_county_boundaries.zip

# -n skips overwriting
# !unzip -n \
#     -d {data_dir}/external \
#     {data_dir}/external/dublin_admin_county_boundaries.zip 
# -

dublin_admin_county_boundaries = io.read_dublin_admin_county_boundaries(
    f"{data_dir}/external/dublin_admin_county_boundaries"
)

# # Get 38kV, 110kV & 220kV stations from CAD data
#
# ... there is no 400kV station in Dublin

# Must be downloaded from the Codema Google Shared Drive or <span style="color:red">**requested from the ESB**</span>

cad_data = "/home/wsl-rowanm/Data/dublin-electricity-network/"

cad_stations = (
    gpd.read_parquet(f"{cad_data}/dublin_hv_network.parquet")
    .query("`Level` == [20, 30, 40]")
    .explode() # un-dissolve station locations from multipoint to single points
    .reset_index()
    .drop(columns=["COUNTY", "index_right", "level_1"])
)

# # Get Map stations

heatmap_ireland = io.read_heatmap(f"{data_dir}/external/heatmap-download-version-nov-2020.xlsx")
heatmap_dublin =  gpd.sjoin(
    heatmap_ireland,
    dublin_admin_county_boundaries,
    op="within",
).drop(columns="index_right")
heatmap_dublin_hv = heatmap_dublin.query("station_name != 'mv/lv'")

capacitymap_ireland = io.read_capacitymap(f"{data_dir}/external/MapDetailsDemand.xlsx")
capacitymap_dublin = gpd.sjoin(
    capacitymap_ireland,
    dublin_admin_county_boundaries,
    op="within",
).drop(columns="index_right")
capacitymap_dublin_hv = capacitymap_dublin.query("station_name != 'mv/lv'")

# ## Link stations to nearest geocoded station

cad_stations_linked_to_heatmap = join.join_nearest_points(cad_stations, heatmap_dublin_hv)

cad_stations_linked_to_capacitymap = join.join_nearest_points(cad_stations, capacitymap_dublin_hv)

# # Plot CAD stations vs Heatmap stations
#
# ... Open `png` version of below plot locally (see [save](#save)) to zoom in

# +
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize=(100, 100))

dublin_admin_county_boundaries.plot(ax=ax, facecolor="teal", edgecolor="white")

cad_stations_linked_to_heatmap.plot(ax=ax, color="black")
cad_stations_linked_to_heatmap.apply(
    lambda x: ax.annotate(
        text=x["station_name"],
        xy=x.geometry.centroid.coords[0],
        ha='center',
        color="white",
        path_effects=[pe.withStroke(linewidth=2, foreground="black")],
    ),
    axis=1,
);

heatmap_dublin_hv.plot(ax=ax,color="orange")
heatmap_dublin_hv.apply(
    lambda x: ax.annotate(
        text=x["station_name"],
        xy=x.geometry.centroid.coords[0],
        ha='center',
        color="white",
        path_effects=[pe.withStroke(linewidth=2, foreground="orange")],
    ),
    axis=1,
);

capacitymap_dublin_hv.plot(ax=ax,color="red")
capacitymap_dublin_hv.apply(
    lambda x: ax.annotate(
        text=x["station_name"],
        xy=x.geometry.centroid.coords[0],
        ha='center',
        color="white",
        path_effects=[pe.withStroke(linewidth=2, foreground="red")],
    ),
    axis=1,
);

plt.legend(["CAD", "Heat Map", "Capacity Map"], prop={'size': 50});
# -

# # Save

f.savefig(f"{data_dir}/outputs/cad-stations-linked-to-nearest-heatmap-station.png")

cad_stations_linked_to_heatmap.to_file(
    f"{data_dir}/outputs/cad-stations-linked-to-nearest-heatmap-station.geojson",
    driver="GeoJSON",
)
