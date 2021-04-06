# %%
from pathlib import Path

import pandas as pd
import geopandas as gpd
import mapclassify as mc
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import dublin_electricity_network as den
from dublin_electricity_network.cluster import cluster_itm_coords

sns.set()
data_dir = Path("../data")
power_factor = 0.95


def convert_to_gdf(df, x, y, crs, *args, **kwargs):
    return gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df[x], df[y], crs=crs)
    ).drop(columns=[x, y])


# %% [markdown]
# # Caveat: Only Data Centres with < 20MVA load link to the MV network
# ... so most don't effect the substation capacities

# %%
esbmap_stations = gpd.read_file(
    data_dir / "heatmap_stations.geojson",
    driver="GeoJSON",
)

# %%
esbmap_capacity_columns = [
    "slr_load_mva",
    "installed_capacity_mva",
    "planned_capacity_mva",
    "demand_available_mva",
    "gen_available_firm_mva",
]

# %%
esbmap_stations_clustered = gpd.read_file(
    data_dir / "esbmap_stations_clustered.geojson",
    driver="GeoJSON",
)

# %%
dublin_boundary = gpd.read_file(
    data_dir / "dublin_boundary.geojson",
    driver="GeoJSON",
)

# %%
dublin_small_area_boundaries = gpd.read_file(
    data_dir / "dublin_small_area_boundaries.geojson",
    driver="GeoJSON",
)

# %%
dublin_small_area_hh = pd.read_csv(data_dir / "dublin_small_area_hh.csv")

# %% [markdown]
# # Link Small Areas stations to Substation Cluster
dublin_small_area_boundaries["cluster_id"] = den.join_nearest_points(
    dublin_small_area_boundaries.assign(geometry=lambda gdf: gdf["geometry"].centroid),
    esbmap_stations_clustered[["cluster_id", "geometry"]],
).loc[:, "cluster_id"]

# %%
dublin_small_area_boundaries["total_hh"] = dublin_small_area_boundaries.merge(
    dublin_small_area_hh
).loc[:, "total_hh"]

# %%
esbmap_stations_clustered["residential_buildings"] = (
    dublin_small_area_boundaries.groupby("cluster_id")["total_hh"].sum().round()
)

# %%
peak_demand_mva_lower = 1.5 * (10 ** -3) * power_factor
esbmap_stations_clustered["resi_peak_mva_at_1_5kw"] = esbmap_stations_clustered.eval(
    "residential_buildings * @peak_demand_mva_lower"
).round()

peak_demand_mva_upper = 2 * (10 ** -3) * power_factor
esbmap_stations_clustered["resi_peak_mva_at_2kw"] = esbmap_stations_clustered.eval(
    "residential_buildings * @peak_demand_mva_upper"
).round()


# %% [markdown]
# # Get remaining Load at each cluster
esbmap_stations_clustered["remaining_load_mva_lower"] = esbmap_stations_clustered.eval(
    "slr_load_mva - resi_peak_mva_at_2kw"
)

esbmap_stations_clustered["remaining_load_mva_upper"] = esbmap_stations_clustered.eval(
    "slr_load_mva - resi_peak_mva_at_1_5kw"
)

# %% [markdown]
# # Link Small Areas to clustered stations

# %%
small_areas_clustered = dublin_small_area_boundaries[
    ["cluster_id", "geometry"]
].dissolve(by="cluster_id", as_index=True)

# %%
small_area_esbmap_stations = esbmap_stations_clustered.assign(
    geometry=small_areas_clustered["geometry"]
)

# %%
small_area_esbmap_stations.to_file(
    data_dir / "small_area_esbmap_stations.geojson", driver="GeoJSON"
)

# %%
