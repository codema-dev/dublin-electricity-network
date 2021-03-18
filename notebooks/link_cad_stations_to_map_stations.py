# %% [markdown]
# # Warning!
#  Must run `get_data.py` to generate the data for this Notebook

# %% [markdown]
# # Load External Dependencies
from pathlib import Path

import geopandas as gpd

import dublin_electricity_network as den

sns.set()
data_dir = Path("../data")
cad_data = Path("/home/wsl-rowanm/Data/ESBdata_20200124")

# %% [markdown]
# # Read Dublin Local Authority Boundaries
dublin_admin_county_boundaries = den.read_dublin_admin_county_boundaries(
    data_dir / "dublin_admin_county_boundaries"
)

# %% [markdown]
# # Read 38kV, 110kV & 220kV Dublin stations from CAD data
cad_stations_dublin = gpd.read_file(
    data_dir / "cad_stations_dublin.geojson",
    driver="GeoJSON",
)

# %% [markdown]
# # Read Dublin OSM | Heat Map stations
map_stations = gpd.read_file(
    data_dir / "osm_substations_linked_to_heatmap.geojson",
    driver="GeoJSON",
)

# %% [markdown]
# ## Link CAD stations to nearest Map station
cad_stations_linked_to_map = den.join_nearest_points(cad_stations_dublin, map_stations)


# %% [markdown]
# # Save

# %%
cad_stations_linked_to_map.to_file(
    data_dir / "cad-stations-linked-to-nearest-map-station.geojson",
    driver="GeoJSON",
)

# %%
cad_stations_linked_to_heatmap_lat_long = (
    cad_stations_linked_to_map.assign(
        Latitude=lambda gdf: gdf.to_crs(epsg=4326).geometry.y,
        Longitude=lambda gdf: gdf.to_crs(epsg=4326).geometry.x,
    )
    .loc[:, ["station_name", "Latitude", "Longitude"]]
    .sort_values(["Latitude", "Longitude"])
)

cad_stations_linked_to_heatmap_lat_long.to_csv(
    data_dir / "cad-stations-linked-to-nearest-map-station.csv",
    index=False,
)
