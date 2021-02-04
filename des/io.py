import geopandas as gpd
import pandas as pd


def read_dublin_admin_county_boundaries(filepath):

    return (
        gpd.read_file(filepath)
        .to_crs(epsg=2157)  # read & convert to ITM or epsg=2157
        .loc[:, ["COUNTYNAME", "geometry"]]
        .reset_index(drop=True)
    )


def read_capacitymap(filepath):

    return (
        pd.read_excel(filepath, engine="openpyxl")
        .dropna(how="all", axis="rows")
        .dropna(how="all", axis="columns")
    )


def read_heatmap(filepath):

    return (
        pd.read_excel(
            filepath,
            engine="openpyxl",
            header=1,
        )
        .drop(labels=[0, 1])  # 1st & 2nd rows are empty
        .reset_index(drop=True)
        .dropna(how="all")
    )
