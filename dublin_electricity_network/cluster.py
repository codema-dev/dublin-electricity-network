import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from shapely.geometry import MultiPoint
from scipy.spatial import cKDTree


def _join_nearest_points(gdA, gdB):
    nA = np.array(list(gdA.geometry.apply(lambda x: (x.x, x.y))))
    nB = np.array(list(gdB.geometry.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1)
    gdB_nearest = gdB.iloc[idx].reset_index(drop=True)
    gdf = pd.concat(
        [
            gdA.reset_index(drop=True).drop(columns="geometry"),
            gdB_nearest,
        ],
        axis=1,
    )
    return gdf


def cluster_itm_coords(
    gdf,
    coords,
    keep_columns,
    how="knearest",
    n_clusters=8,
    max_km_distance_between_points=2500,
):
    if how == "knearest":
        model = KMeans(n_clusters)
    elif how == "dbscan":
        model = DBSCAN(
            eps=max_km_distance_between_points,
            min_samples=1,
            algorithm="ball_tree",
        )
    else:
        raise NotImplementedError("Only 'knearest' or 'dbscan' implemented...")

    model.fit(coords)
    cluster_labels = model.labels_
    num_clusters = len(set(cluster_labels))

    clusters = gpd.GeoDataFrame(
        geometry=[MultiPoint(coords[cluster_labels == n]) for n in range(num_clusters)]
    )

    centermost_points = (
        clusters.assign(geometry=lambda gdf: gdf.geometry.centroid)
        .reset_index()
        .rename(columns={"index": "cluster_id"})
    )

    gdf_linked_to_clusters = _join_nearest_points(
        gdf,
        centermost_points,
    )

    return gdf_linked_to_clusters[keep_columns].dissolve(
        by="cluster_id", aggfunc="sum", as_index=False
    )
