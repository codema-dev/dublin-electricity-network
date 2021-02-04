import os

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm


def plot_gdf_vs_nx(G, gdf, boundaries):
    positions = {n: [n[0], n[1]] for n in list(G.nodes)}

    f, ax = plt.subplots(1, 2, figsize=(30, 30), sharex=True, sharey=True)

    for i, facet in enumerate(ax):
        facet.set_title(("gdf", "nx")[i])
        facet.axis("off")

    boundaries.plot(edgecolor="red", ax=ax[0])
    gdf.plot(color="k", ax=ax[0])

    boundaries.plot(edgecolor="red", ax=ax[1])
    nx.draw(G, positions, ax=ax[1], node_size=5)


def plot_path_n(G, paths, orig_points, dest_points, boundaries, n):
    positions = {z: [z[0], z[1]] for z in list(G.nodes)}
    x, y = zip(*paths[n][1])

    f, ax = plt.subplots(figsize=(30, 30))
    boundaries.plot(edgecolor="red", ax=ax)
    nx.draw(G, positions, node_size=5, ax=ax)
    ax.plot(x, y, c="k", lw=20, alpha=0.5)
    ax.scatter(orig_points.iloc[n].x, orig_points.iloc[n].y, color="green", s=500)
    ax.scatter(x[0], y[0], color="red", s=500)
    ax.scatter(x[-1], y[-1], color="green", s=500)
    ax.scatter(dest_points.x, dest_points.y, color="k", s=250)


def plot_paths_to_files(G, paths, orig_points, dest_points, boundaries, dirpath):
    os.mkdir(dirpath)
    for n in tqdm(range(len(paths))):
        positions = {z: [z[0], z[1]] for z in list(G.nodes)}
        x, y = zip(*paths[n][1])

        f, ax = plt.subplots(figsize=(30, 30))
        boundaries.plot(edgecolor="red", ax=ax)
        nx.draw(G, positions, node_size=5, ax=ax)
        ax.plot(x, y, c="k", lw=20, alpha=0.5)

        ax.scatter(orig_points.iloc[n].x, orig_points.iloc[n].y, color="green", s=500)
        ax.scatter(x[0], y[0], color="red", s=500)
        ax.scatter(x[-1], y[-1], color="green", s=500)
        ax.scatter(dest_points.x, dest_points.y, color="k", s=250)
        f.savefig(f"{dirpath}/{n}.png")
        plt.close()


def plot_heatmap_vs_capacitymap(heatmap, capacitymap, boundaries):

    f, ax = plt.subplots(figsize=(100, 100))

    boundaries.plot(ax=ax, facecolor="orange", edgecolor="red")
    heatmap.query("station_name != 'mv/lv'").plot(ax=ax, color="blue", markersize=3000)
    capacitymap.plot(ax=ax, color="cyan", markersize=3000)

    heatmap.query("station_name != 'mv/lv'").apply(
        lambda gdf: ax.annotate(
            text=gdf["station_name"],
            xy=gdf.geometry.centroid.coords[0],
            ha="center",
            fontsize=14,
            color="white",
            path_effects=[pe.withStroke(linewidth=2, foreground="black")],
        ),
        axis=1,
    )
    capacitymap.apply(
        lambda gdf: ax.annotate(
            text=gdf["station_name"],
            xy=gdf.geometry.centroid.coords[0],
            ha="center",
            fontsize=14,
            color="white",
            path_effects=[pe.withStroke(linewidth=2, foreground="black")],
        ),
        axis=1,
    )

    plt.legend(["heatmap", "capacitymap"], prop={"size": 100})

    return f


def plot_small_areas_linked_to_stations(small_areas, stations):

    f, ax = plt.subplots(figsize=(100, 100))

    small_areas.plot(ax=ax, column="station_name", cmap="binary")
    stations.plot(
        ax=ax,
        color="teal",
        markersize=5000,
    )
    stations.apply(
        lambda gdf: ax.annotate(
            text=gdf["station_name"],
            xy=gdf.geometry.centroid.coords[0],
            ha="center",
            fontsize=12,
            color="black",
            path_effects=[pe.withStroke(linewidth=2, foreground="white")],
        ),
        axis=1,
    )

    return f


def plot_cad_stations_vs_heatmap_stations(cad_stations, heatmap_stations, boundaries):

    f, ax = plt.subplots(figsize=(100, 100))
    boundaries.plot(ax=ax, facecolor="orange", edgecolor="teal")

    heatmap_stations.plot(
        ax=ax,
        color="teal",
        markersize=2000,
    )
    heatmap_stations.apply(
        lambda gdf: ax.annotate(
            text=gdf["station_name"],
            xy=gdf.geometry.centroid.coords[0],
            ha="center",
            fontsize=12,
            color="black",
            path_effects=[pe.withStroke(linewidth=2, foreground="white")],
        ),
        axis=1,
    )

    cad_stations.plot(ax=ax, markersize=100, color="black")

    return f