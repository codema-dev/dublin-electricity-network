from dask import compute
from dask import delayed
from dask.diagnostics import ProgressBar
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from tqdm import tqdm


def euclidean_dist_vec(y1, x1, y2, x2):
    """
    Calculate Euclidean distances between points.

    Vectorized function to calculate the Euclidean distance between two
    points' coordinates or between arrays of points' coordinates. For most
    accurate results, use projected coordinates rather than decimal degrees.

    Parameters
    ----------
    y1 : float or np.array of float
        first point's y coordinate
    x1 : float or np.array of float
        first point's x coordinate
    y2 : float or np.array of float
        second point's y coordinate
    x2 : float or np.array of float
        second point's x coordinate

    Returns
    -------
    dist : float or np.array of float
        distance or array of distances from (x1, y1) to (x2, y2) in
        coordinates' units

    Copied from
    -----------
    https://github.com/gboeing/osmnx/blob/master/osmnx/distance.py
    """
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5  # Pythagorean theorem


def get_nearest_node(G, point):
    """
    Find the nearest node to a point.

    Return the graph node nearest to some (lng, lat) or (x, y) point

    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    point : tuple
        The (lng, lat) or (x, y) point for which we will find the nearest node
        in the graph

    Returns
    -------
    int or tuple of (int, float)
        Nearest node ID
    """
    # dump graph node coordinates into a pandas dataframe with x and y columns
    coords = (d[0] for d in G.nodes(data=True))
    target = pd.DataFrame(coords, columns=["x", "y"])
    tree = BallTree(target[["x", "y"]].values, leaf_size=2)

    distance_nearest, id_nearest = tree.query(
        np.array(point).reshape(1, -1),
        k=1,  # The number of nearest neighbors
    )

    return tuple(target.iloc[id_nearest.item()])


def get_nearest_nodes(G, points):
    """
    Find the nearest node to each point.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    points : pandas.DataFrame
        The points for which we will find the nearest node in the graph

    Returns
    -------
    int or tuple of (int, float)
        Nearest node ID

    Adapted from
    ------------
    https://stackoverflow.com/questions/58893719/find-nearest-point-in-other-dataframe-with-a-lot-of-data
    """
    # dump graph node coordinates into a pandas dataframe with x and y columns
    coords = (d[0] for d in G.nodes(data=True))
    target = pd.DataFrame(coords, columns=["x", "y"])
    tree = BallTree(target[["x", "y"]].values, leaf_size=2)

    points["distance_nearest"], points["id_nearest"] = tree.query(
        points[["x", "y"]].values,
        k=1,  # The number of nearest neighbors
    )
    target_ids = points["id_nearest"]
    target_coords = target.iloc[target_ids]
    return [tuple(x) for x in target_coords.to_numpy()]


def get_network_paths_between_points_recursively(G, orig_points, dest_points):
    """
    Find the nearest dest_point to each orig_point.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    orig_points : pandas.DataFrame
        The points for which we will find the nearest dest_point
    dest_points : pandas.DataFrame
        The points to be compared to orig_points

    Returns
    -------
    list of (tuple of (int, list of (coords))
        Shortest paths from orig_point to nearest dest_point

    Adapted from
    ------------
    https://stackoverflow.com/questions/63690631/osmnx-shortest-path-how-to-skip-node-if-not-reachable-and-take-the-next-neares/63713539#63713539
    """
    S = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    paths = []
    target_nodes = get_nearest_nodes(G, dest_points)
    for orig_point in tqdm(orig_points.itertuples(), total=len(orig_points)):
        G_copy = G.copy()
        solved = False
        while not solved:
            orig_node = get_nearest_node(G_copy, (orig_point.x, orig_point.y))
            try:
                path = nx.multi_source_dijkstra(
                    G_copy,
                    sources=target_nodes,
                    target=(orig_node),
                )
            except nx.exception.NetworkXNoPath:
                remove_subgraph(G, S, orig_node)
            else:
                paths.append(path)
                solved = True
    return paths


def remove_subgraph(G, S, orig_node):
    # if no path to any target_nodes remove entire subgraph
    Gs = [subgraph for subgraph in S if orig_node in subgraph][0]
    return G.remove_nodes_from(list(Gs.nodes()))


def get_network_paths_between_points(G, orig, dest):
    """
    Find the nearest dest to each orig.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    orig : pandas.DataFrame
        The points for which we will find the nearest dest_point
    dest : pandas.DataFrame
        The points to be compared to orig_points

    Returns
    -------
    list of (tuple of (int, list of (coords))
        Shortest paths from orig to nearest dest

    Adapted from
    ------------
    https://stackoverflow.com/questions/63690631/osmnx-shortest-path-how-to-skip-node-if-not-reachable-and-take-the-next-neares/63713539#63713539
    """
    orig_points = pd.DataFrame(
        {
            "x": orig.geometry.centroid.x,
            "y": orig.geometry.centroid.y,
        }
    )
    dest_points = pd.DataFrame(
        {
            "x": dest.geometry.x,
            "y": dest.geometry.y,
        }
    )
    paths = []
    target_nodes = delayed(get_nearest_nodes)(G, dest_points)
    for orig_point in orig_points.itertuples():
        orig_node = delayed(get_nearest_node)(G, (orig_point.x, orig_point.y))
        path = delayed(nx.multi_source_dijkstra)(
            G,
            sources=target_nodes,
            target=(orig_node),
        )
        paths.append(path)

    with ProgressBar():
        shortest_paths = compute(*paths)

    return shortest_paths


def get_largest_subgraph(G):

    nodes = max(nx.connected_components(G), key=len)
    return G.subgraph(nodes)


def get_large_subgraphs(G, size=5):

    largest_components = [
        component
        for component in sorted(nx.connected_components(G), key=len, reverse=True)
        if len(component) > 5
    ]
    return nx.compose_all(
        G.subgraph(component) for component in tqdm(largest_components)
    )