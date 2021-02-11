import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


def read_capacitymap(filepath):

    capacitymap_df = (
        pd.read_excel(filepath, engine="openpyxl")
        .dropna(how="all", axis="rows")
        .dropna(how="all", axis="columns")
    )

    capacitymap_df["available_capacity"] = capacitymap_df["Marker"].map(
        {
            "Blue": "mixed",
            "Green": "significant",
            "Orange": "limited",
            "Red": "none",
        }
    )  # From https://www.esbnetworks.ie/demand-availability-capacity-map

    capacitymap_df["station_name"] = (
        capacitymap_df["Title"].str.split(" - ").str.get(0).str.lower()
    )

    capacitymap_df["station_voltages"] = (
        capacitymap_df["Title"].str.split(" - ").str.get(1).str.lower()
    )

    capacitymap_gdf = gpd.GeoDataFrame(
        capacitymap_df,
        geometry=gpd.points_from_xy(
            capacitymap_df["Longitude"], capacitymap_df["Latitude"]
        ),
        crs="epsg:4326",
    ).to_crs(epsg=2157)

    return capacitymap_gdf


def read_dublin_admin_county_boundaries(filepath):

    return (
        gpd.read_file(filepath)
        .to_crs(epsg=2157)  # read & convert to ITM or epsg=2157
        .loc[:, ["COUNTYNAME", "geometry"]]
        .reset_index(drop=True)
    )


def read_heatmap(filepath):

    heatmap_df = (
        pd.read_excel(
            filepath,
            engine="openpyxl",
            header=1,
        )
        .drop(labels=[0, 1])  # 1st & 2nd rows are empty
        .reset_index(drop=True)
        .dropna(how="all")
    )

    hv_station_rows = ~heatmap_df["Station Name"].str.contains("MV/LV Substation")
    heatmap_df["station_name"] = (
        heatmap_df.loc[hv_station_rows]
        .loc[:, "Station Name"]
        .str.lower()
        .str.extract(r"(.*?) \d")
    )
    heatmap_df["station_name"] = heatmap_df["station_name"].fillna("mv/lv")

    heatmap_gdf = gpd.GeoDataFrame(
        heatmap_df,
        geometry=gpd.points_from_xy(heatmap_df["Longitude"], heatmap_df["Latitude"]),
        crs="epsg:4326",
    ).to_crs(epsg=2157)

    return heatmap_gdf


def read_dublin_small_areas(filepath):

    return (
        gpd.read_file(filepath)
        .loc[:, ["SMALL_AREA", "COUNTYNAME", "geometry"]]
        .to_crs(epsg=2157)  # convert to ITM
        .query(
            "`COUNTYNAME` == ['South Dublin', 'Dún Laoghaire-Rathdown', 'Fingal', 'Dublin City']"
        )
        .reset_index(drop=True)
    )


def read_mv_index(filepath):

    ireland_mv_index = gpd.read_file(filepath, driver="DGN")
    ireland_mv_index.crs = "EPSG:29903"
    ireland_mv_index = ireland_mv_index.to_crs(epsg=2157)
    point_rows = ireland_mv_index.geometry.apply(lambda x: isinstance(x, Point))
    return ireland_mv_index[point_rows].copy()


def read_network(filepaths, levels=None):

    network = []
    for filepath in filepaths:

        if levels:
            region = gpd.read_file(filepath, driver="DGN").query(
                f"`Level` == {str(levels)}"
            )
        else:
            region = gpd.read_file(filepath, driver="DGN")

        network.append(region)

    return gpd.GeoDataFrame(pd.concat(network))