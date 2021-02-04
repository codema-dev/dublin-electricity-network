import geopandas as gpd


def convert_to_gdf_via_lat_long(df):

    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]),
        crs="epsg:4326",
    ).to_crs(epsg=2157)
