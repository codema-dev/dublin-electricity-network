import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def join_nearest_points(gdA, gdB):
    nA = np.array(list(gdA.geometry.apply(lambda x: (x.x, x.y))))
    nB = np.array(list(gdB.geometry.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1)
    gdB_nearest = gdB.iloc[idx].drop(columns="geometry").reset_index(drop=True)
    gdf = pd.concat(
        [
            gdA.reset_index(drop=True),
            gdB_nearest,
        ],
        axis=1,
    )
    return gdf


def centroids_within(left, right):

    left_centroids = left.geometry.centroid.rename("geometry").to_frame()
    return (
        gpd.sjoin(left_centroids, right, op="within")
        .drop(columns=["geometry", "index_right"])
        .merge(left, left_index=True, right_index=True)
        .reset_index(drop=True)
    )
