from dask import compute
from dask import delayed
from dask.diagnostics import ProgressBar
import networkx as nx
import pandas as pd


def get_network_paths_between_points(G, orig_points, dest_points):
    """
    Find the nearest dest to each orig.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    orig_points : geopandas.GeoDataFrame
        The points for which we will find the nearest dest_point
    dest_points : geopandas.GeoDataFrame
        The points to be compared to orig_points

    Returns
    -------
    list of (tuple of (int, list of (coords))
        Shortest paths from orig to nearest dest

    Adapted from
    ------------
    https://stackoverflow.com/questions/63690631/osmnx-shortest-path-how-to-skip-node-if-not-reachable-and-take-the-next-neares/63713539#63713539
    """
    paths = []
    target_nodes = delayed(get_nearest_nodes)(G, dest_points)
    for orig_point in orig_points.itertuples():
        orig_node = delayed(get_nearest_node)(
            G, (orig_point.geometry.centroid.x, orig_point.geometry.centroid.y)
        )
        path = delayed(nx.multi_source_dijkstra)(
            G,
            sources=target_nodes,
            target=(orig_node),
        )
        paths.append(path)

    with ProgressBar():
        shortest_paths = compute(*paths)

    return shortest_paths


def extract_nearest_dest(paths, orig, dest):

    nearest_dest_coords = [path[1][0] for path in paths]
    nearest_dest_coords_df = pd.DataFrame(nearest_dest_coords, columns=["x", "y"])
    orig_with_dest_coords = nearest_dest_coords_df.merge(
        orig,
        left_index=True,
        right_index=True,
    )

    dest["x"] = dest.geometry.centroid.x
    dest["y"] = dest.geometry.centroid.y

    return gpd.GeoDataFrame(
        orig_with_dest_coords.merge(
            dest.drop(columns="geometry"),
            on=["x", "y"],
        ).drop(columns=["x", "y"])
    )
