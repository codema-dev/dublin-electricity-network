# %%
from pathlib import Path
from shutil import unpack_archive

import den

data_dir = Path("../data")

# %% [markdown]
# # Get Dublin Boundary
den.download(
    url="https://zenodo.org/record/4577018/files/dublin_boundary.geojson",
    to_filepath=str(data_dir / "dublin_boundary.geojson"),
)

# %% [markdown]
# # Get Dublin LA boundaries

# %%
den.download(
    url="https://zenodo.org/record/4446778/files/dublin_admin_county_boundaries.zip",
    to_filepath=str(data_dir / "dublin_admin_county_boundaries.zip"),
)

# %%
unpack_archive(
    data_dir / "dublin_admin_county_boundaries.zip",
    data_dir,
)

# %%
dublin_admin_county_boundaries = den.read_dublin_admin_county_boundaries(
    data_dir / "dublin_admin_county_boundaries"
)


# %% [markdown]
# # Get Dublin HV Heat Map stations

# %%
den.download(
    url="https://esbnetworks.ie/docs/default-source/document-download/heatmap-download-version-nov-2020.xlsx",
    to_filepath=str(data_dir / "heatmap-download-version-nov-2020.xlsx"),
)

# %%
heatmap_stations_ireland = den.read_heatmap(
    data_dir / "heatmap-download-version-nov-2020.xlsx"
)

# %%
heatmap_stations_dublin = gpd.sjoin(
    heatmap_stations_ireland,
    dublin_admin_county_boundaries,
    op="within",
).drop(columns="index_right")

# %%
heatmap_stations_dublin_hv = heatmap_stations_dublin.query("station_name != 'mv/lv'")

# %%
heatmap_stations_dublin_hv.to_file(
    data_dir / "heatmap_stations.geojson",
    driver="GeoJSON",
)

# %% [markdown]
# # Get 38kV, 110kV & 220kV Dublin stations from CAD data
# .. must be downloaded from the Codema Google Shared Drive or <span style="color:red">**requested from the ESB**</span>

# %%
hv_network_dirpath = cad_data / "Dig Request Style" / "HV Data"
hv_network_filepaths = [
    hv_network_dirpath / filename for filename in listdir(hv_network_dirpath)
]
# %%
cad_stations_ireland = den.read_network(hv_network_filepaths, levels=[20, 30, 40])
# %%
cad_stations_dublin = gpd.sjoin(
    cad_stations_ireland,
    dublin_admin_county_boundaries,
    op="within",
).drop(columns=["index_right", "COUNTYNAME"])
# %%
cad_stations_dublin.to_file(data_dir / "cad_stations_dublin.geojson", driver="GeoJSON")