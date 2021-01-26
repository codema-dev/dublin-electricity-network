import os

import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm


def plot_gdf_vs_nx(G, gdf, boundaries):
    positions = {n: [n[0], n[1]] for n in list(G.nodes)}

    f, ax = plt.subplots(1, 2, figsize=(30, 30), sharex=True, sharey=True)

    for i, facet in enumerate(ax):
        facet.set_title(("gdf", "nx")[i])
        facet.axis("off")

    boundaries.plot(edgecolor='red', ax=ax[0])
    gdf.plot(color="k", ax=ax[0])

    boundaries.plot(edgecolor='red', ax=ax[1])
    nx.draw(G, positions, ax=ax[1], node_size=5)


def plot_path_n(G, paths, orig_points, dest_points, boundaries, n):
    positions = {z: [z[0], z[1]] for z in list(G.nodes)}
    x, y = zip(*paths[n][1])

    f, ax = plt.subplots(figsize=(30, 30))
    boundaries.plot(edgecolor='red', ax=ax)
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
        boundaries.plot(edgecolor='red', ax=ax)
        nx.draw(G, positions, node_size=5, ax=ax)
        ax.plot(x, y, c="k", lw=20, alpha=0.5)

        ax.scatter(orig_points.iloc[n].x, orig_points.iloc[n].y, color="green", s=500) 
        ax.scatter(x[0], y[0], color="red", s=500)
        ax.scatter(x[-1], y[-1], color="green", s=500)
        ax.scatter(dest_points.x, dest_points.y, color="k", s=250)
        f.savefig(f"{dirpath}/{n}.png")
        plt.close()
