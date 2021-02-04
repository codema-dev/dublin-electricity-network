import geopandas as gpd


def extract_polygon(gdf, boundary):

    return gpd.sjoin(gdf, boundary).drop(columns="index_right").reset_index(drop=True)
