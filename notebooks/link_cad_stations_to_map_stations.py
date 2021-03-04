# %%
# Uncomment if running on Google Colab
# Click RESTART RUNTIME if prompted
# !pip install git+https://github.com/codema-dev/esb-network-pylib

# %%
from os import listdir
from pathlib import Path
from shutil import unpack_archive

import geopandas as gpd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

import esb

data_dir = Path("../data")
cad_data = Path("/home/wsl-rowanm/Data/ESBdata_20200124")

# %% [markdown]
# # Get Dublin LA boundaries

# %%
esb.download(
    url="https://zenodo.org/record/4446778/files/dublin_admin_county_boundaries.zip",
    to_filepath=str(data_dir / "dublin_admin_county_boundaries.zip"),
)
unpack_archive(
    data_dir / "dublin_admin_county_boundaries.zip",
    data_dir,
)


# %%
dublin_admin_county_boundaries = esb.read_dublin_admin_county_boundaries(
    data_dir / "dublin_admin_county_boundaries"
)

# %% [markdown]
# # Get 38kV, 110kV & 220kV stations from CAD data
#
# ... there is no 400kV station in Dublin

# %% [markdown]
# Must be downloaded from the Codema Google Shared Drive or <span style="color:red">**requested from the ESB**</span>

# %%
hv_network_dirpath = cad_data / "Dig Request Style" / "HV Data"
hv_network_filepaths = [
    hv_network_dirpath / filename for filename in listdir(hv_network_dirpath)
]
cad_stations_ireland = esb.read_network(hv_network_filepaths, levels=[20, 30, 40])
cad_stations_dublin = gpd.sjoin(
    cad_stations_ireland,
    dublin_admin_county_boundaries,
    op="within",
).drop(columns=["index_right", "COUNTYNAME"])


# %% [markdown]
# # Get Map stations

# %%
esb.download(
    url="https://esbnetworks.ie/docs/default-source/document-download/heatmap-download-version-nov-2020.xlsx",
    to_filepath=str(data_dir / "heatmap-download-version-nov-2020.xlsx"),
)

# %%
heatmap_stations_ireland = esb.read_heatmap(
    data_dir / "heatmap-download-version-nov-2020.xlsx"
)
heatmap_stations_dublin = gpd.sjoin(
    heatmap_stations_ireland,
    dublin_admin_county_boundaries,
    op="within",
).drop(columns="index_right")
heatmap_stations_dublin_hv = heatmap_stations_dublin.query("station_name != 'mv/lv'")

# %% [markdown]
# ## Link stations to nearest geocoded station

# %%
cad_stations_linked_to_heatmap = esb.join_nearest_points(
    cad_stations_dublin, heatmap_stations_dublin_hv
)

# %% [markdown]
# # Plot CAD stations vs Heatmap stations
#
# ... Open `png` version of below plot locally (see [save](#save)) to zoom in

# %%
f, ax = plt.subplots(figsize=(100, 100))

dublin_admin_county_boundaries.plot(ax=ax, facecolor="teal", edgecolor="white")

cad_stations_linked_to_heatmap.plot(ax=ax, color="black")
cad_stations_linked_to_heatmap.apply(
    lambda x: ax.annotate(
        text=x["station_name"],
        xy=x.geometry.centroid.coords[0],
        ha="center",
        color="white",
        path_effects=[pe.withStroke(linewidth=2, foreground="black")],
    ),
    axis=1,
)

heatmap_stations_dublin_hv.plot(ax=ax, color="orange")
heatmap_stations_dublin_hv.apply(
    lambda x: ax.annotate(
        text=x["station_name"],
        xy=x.geometry.centroid.coords[0],
        ha="center",
        color="white",
        path_effects=[pe.withStroke(linewidth=2, foreground="orange")],
    ),
    axis=1,
)

plt.legend(["CAD", "Heat Map", "Capacity Map"], prop={"size": 50})

# %% [markdown]
# # Save

# %%
f.savefig(data_dir / "cad-stations-linked-to-nearest-heatmap-station.png")

# %%
cad_stations_linked_to_heatmap.to_file(
    data_dir / "cad-stations-linked-to-nearest-heatmap-station.geojson",
    driver="GeoJSON",
)

# %%
cad_stations_linked_to_heatmap_lat_long = (
    cad_stations_linked_to_heatmap.assign(
        Latitude=lambda gdf: gdf.to_crs(epsg=4326).geometry.y,
        Longitude=lambda gdf: gdf.to_crs(epsg=4326).geometry.x,
    )
    .loc[:, ["station_name", "Latitude", "Longitude"]]
    .sort_values(["Latitude", "Longitude"])
)

cad_stations_linked_to_heatmap_lat_long.to_csv(
    data_dir / "cad-stations-linked-to-nearest-heatmap-station.csv",
    index=False,
)
