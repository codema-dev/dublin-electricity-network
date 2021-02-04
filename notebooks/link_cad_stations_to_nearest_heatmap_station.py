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
from des import clean
from des import convert
from des import extract
from des import join
from des import plot
# -

# # Get LA boundaries

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

# # Get heatmap stations

heatmap_raw = io.read_heatmap("../data/external/heatmap-download-version-nov-2020.xlsx")
heatmap_clean = clean.clean_heatmap(heatmap_raw)
heatmap_gdf = convert.convert_to_gdf_via_lat_long(heatmap_clean)
heatmap_dublin = extract.extract_polygon(
    heatmap_gdf,
    dublin_admin_county_boundaries
)
heatmap_dublin_hv = heatmap_dublin.query("station_name != 'mv/lv'")

# # Plot CAD stations vs Heatmap stations
#
# ... Open `png` version of below plot locally (see [save](#save)) to zoom in

fig = plot.plot_cad_stations_vs_heatmap_stations(
    cad_stations=cad_stations,
    heatmap_stations=heatmap_dublin_hv,
    boundaries=dublin_admin_county_boundaries,
)

# ## Link stations to nearest geocoded station

cad_stations_linked = gpd.GeoDataFrame(
    join.join_nearest_points(cad_stations, heatmap_dublin_hv)
)

# # Save

fig.savefig("../data/outputs/cad-stations-linked-to-nearest-heatmap-station.png")

cad_stations_linked.to_file(
    "../data/outputs/cad-stations-linked-to-nearest-heatmap-station.geojson",
    driver="GeoJSON",
)
