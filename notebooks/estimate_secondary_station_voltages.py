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

import geopandas as gpd
import matplotlib.pyplot as plt

# # Get network stations & lines

network_data = "/home/wsl-rowanm/Data/dublin-electricity-network/"

mv_network = (
    gpd.read_parquet(f"{network_data}/dublin_lvmv_network.parquet")
    .query("`voltage_kv` == 'mv'")
    .explode() # un-dissolve station locations from multipoint to single points
    .reset_index()
    .loc[: , ["Level", "voltage_kv", "geometry"]]
)

mv_network_lines = mv_network.query("`Level` == [10, 11, 14]").dissolve(by="voltage_kv")

hv_network = (
    gpd.read_parquet(f"{network_data}/dublin_hv_network.parquet")
    .to_crs(epsg=2157)
    .explode() # un-dissolve station locations from multipoint to single points
    .reset_index()
    .loc[: , ["Level", "voltage_kv", "geometry"]]
)

hv_lines_38kv = hv_network.query("`Level` == [21, 24]").dissolve(by="voltage_kv")
hv_lines_110kv = hv_network.query("`Level` == [31, 34]").dissolve(by="voltage_kv")
hv_lines_220kv = hv_network.query("`Level` == [41, 44]").assign(voltage_kv="220kv")

hv_stations_38kv = hv_network.query("`Level` == 20")
hv_stations_110kv = hv_network.query("`Level` == 30")
hv_stations_220kv = hv_network.query("`Level` == 40")

# # Get HV stations that intersect ...

hv_stations_38kv_buffered = hv_stations_38kv.assign(geometry=lambda gdf: gdf.buffer(150))
hv_stations_110kv_buffered = hv_stations_110kv.assign(geometry=lambda gdf: gdf.buffer(150))

# ## 38kv/MV

hv_stations_38kv_mv = gpd.sjoin(hv_stations_38kv_buffered, mv_network_lines)[["geometry"]]

hv_stations_38kv_from_38kv_to_mv = gpd.sjoin(hv_stations_38kv_mv, hv_lines_38kv)[["geometry"]]

# +
f, ax = plt.subplots(figsize=(30, 30))

hv_stations_38kv_buffered.plot(ax=ax, color="white")
mv_network_lines.plot(ax=ax, color="blue")
hv_lines_38kv.plot(ax=ax, color="orange")
hv_stations_38kv_from_38kv_to_mv.plot(ax=ax, color="red")
ax.legend(["mv network", "38kv network"])
# -

f.savefig("../data/outputs/hv_stations_38kv.png")

# ## 110kv/MV

hv_stations_110kv_mv = gpd.sjoin(hv_stations_110kv_buffered, mv_network_lines)[["geometry"]]

hv_stations_110kv_from_110kv_to_mv = gpd.sjoin(hv_stations_110kv_mv, hv_lines_110kv)[["geometry"]]

# +
f, ax = plt.subplots(figsize=(30, 30))

hv_stations_110kv_buffered.plot(ax=ax, color="white")
mv_network_lines.plot(ax=ax, color="blue")
hv_lines_110kv.plot(ax=ax, color="orange")
hv_stations_110kv_from_110kv_to_mv.plot(ax=ax, color="red")
ax.legend(["mv network", "110kv network"])
# -

f.savefig("../data/outputs/hv_stations_110kv.png")
