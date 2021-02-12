from dask import compute
from dask import delayed
from dask.diagnostics import ProgressBar
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from tqdm import tqdm


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

    points_x = points.geometry.x
    points_y = points.geometry.y
    points_xy = pd.concat([points_x, points_y], axis="columns").to_numpy()
    points["distance_nearest"], points["id_nearest"] = tree.query(
        points_xy,
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