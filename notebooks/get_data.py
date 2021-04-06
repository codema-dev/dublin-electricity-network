# %%
from os import listdir
from pathlib import Path
from shutil import unpack_archive

import geopandas as gpd
import pandas as pd

import dublin_electricity_network as den

data_dir = Path("../data")
cad_data = data_dir / "ESBdata_20210107"

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

# %% [markdown]
# # Get Census 2016 Dublin Small Area Boundaries & Statistics

# %%
den.download(
    url="https://zenodo.org/record/4562488/files/dublin_small_area_boundaries.geojson",
    to_filepath=str(data_dir / "dublin_small_area_boundaries.geojson"),
)

# %%
dublin_small_area_boundaries = gpd.read_file(
    data_dir / "dublin_small_area_boundaries.geojson", driver="GeoJSON"
)

# %%
filepath = data_dir / "SAPS2016_SA2017.csv"

# %%
den.download(
    url="https://www.cso.ie/en/media/csoie/census/census2016/census2016boundaryfiles/SAPS2016_SA2017.csv",
    filename=str(filepath),
)

# %%
unpack_archive(
    filepath.with_suffix(".zip"),
    filepath,
)

# %%
use_columns = ["SMALL_AREA", "T6_1_TH"]
dublin_small_area_hh = (
    pd.read_csv(filepath)
    .assign(SMALL_AREA=lambda df: df["GEOGID"].str[7:])
    .loc[:, use_columns]
    .rename(columns={"T6_1_TH": "total_hh"})
)

# %%
dublin_small_area_hh.to_csv(data_dir / "dublin_small_area_hh.csv", index=False)

# %% [markdown]
# # Get Dublin Boundary
filepath = data_dir / "dublin_boundary.geojson"
den.download(
    url="https://zenodo.org/record/4577018/files/dublin_boundary.geojson",
    filename=str(filepath),
)

dublin_boundary = gpd.read_file(filepath, driver="GeoJSON")

# %%
