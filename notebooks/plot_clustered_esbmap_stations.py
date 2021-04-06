# %%
from pathlib import Path

from bokeh.palettes import diverging_palette
import geopandas as gpd
import numpy as np
import pandas as pd

import pandas_bokeh

data_dir = Path("../data")
html_dir = Path("../html")
pandas_bokeh.output_notebook()

# %%
esbmap_stations = (
    gpd.read_file(
        data_dir / "heatmap_stations.geojson",
        driver="GeoJSON",
    )
    .to_crs(epsg=2157)
    .assign(
        name=lambda df: df["Station Name"],
        voltages=lambda df: df["Secondary Voltage(s)"],
        slr_load_mva=lambda gdf: gdf["SLR Load MVA"].round(),
        installed_capacity_mva=lambda gdf: gdf["Installed Capacity MVA"].round(),
        planned_capacity_mva=lambda gdf: gdf["Demand Planning Capacity"].round(),
        demand_available_mva=lambda gdf: gdf["Demand Available MVA"].round(),
        gen_available_firm_mva=lambda gdf: gdf["Gen Available Firm"].round(),
        scaled_installed_capacity_mva=lambda gdf: gdf["installed_capacity_mva"] / 4,
    )
)

# %%
esbmap_stations_clustered = gpd.read_file(
    data_dir / "esbmap_stations_clustered.geojson", driver="GeoJSON"
)

# %%
small_area_esbmap_stations = gpd.read_file(
    data_dir / "small_area_esbmap_stations.geojson", driver="GeoJSON"
)

# %%[markdown]
# # Plot Station clusters via Bokeh

# %%
# pandas_bokeh.output_file(html_dir / "substations_clustered_to_10_points.html")

# %%
hovertool_string = """  
    <table style="background-color:#084594;color:#ffffff">
        <tr>
            <td>Demand Available [MVA]</th>
            <td>@demand_available_mva</td>
        </tr>
        <tr>
            <td>Installed Capacity [MVA]</th>
            <td>@installed_capacity_mva</td>
        </tr>
        <tr>
            <td>SLR Load [MVA]</th>
            <td>@slr_load_mva</td>
        </tr>
    </table>
"""
figure = small_area_esbmap_stations.plot_bokeh(
    figsize=(700, 900),
    dropdown=["demand_available_mva", "installed_capacity_mva", "slr_load_mva"],
    # colormap=(
    #     "#f7fbff",
    #     "#3182bd",
    # ),
    colormap_range=(0, 50),
    hovertool_string=hovertool_string,
    fill_alpha=0.5,
)

hovertool_string = """
    <table style="background-color:#084594;color:#ffffff">
        <tr>
            <td>Name</th>
            <td>@station_name</td>
        </tr>
        <tr>
            <td>Secondary Voltage(s)</th>
            <td>@voltages</td>
        </tr>
        <tr>
            <td>Demand Available [MVA]</th>
            <td>@demand_available_mva</td>
        </tr>
        <tr>
            <td>Installed Capacity [MVA]</th>
            <td>@installed_capacity_mva</td>
        </tr>
        <tr>
            <td>SLR Load [MVA]</th>
            <td>@slr_load_mva</td>
        </tr>
    </table>
"""
figure = esbmap_stations.plot_bokeh(
    figure=figure,
    marker="inverted_triangle",
    hovertool_string=hovertool_string,
    legend="Substations",
    size="scaled_installed_capacity_mva",
    fill_alpha=0.2,
)

# %% [markdown]
# # Plot Station Clusters via Seaborn

# %%
def plot_clusters(boundary, unclustered, clustered, column_name):
    f, ax = plt.subplots(figsize=(20, 20))
    boundary.plot(ax=ax, alpha=0.5)
    clustered.plot(ax=ax, c="#99cc99", edgecolor="None", alpha=0.7, markersize=120)
    clustered.apply(
        lambda gdf: ax.annotate(
            "ID = " + str(gdf["cluster_id"]),
            xy=gdf.geometry.centroid.coords[0],
            va="top",
            path_effects=[pe.withStroke(linewidth=4, foreground="white")],
        ),
        axis="columns",
    )
    clustered.apply(
        lambda gdf: ax.annotate(
            gdf[column_name],
            xy=gdf.geometry.centroid.coords[0],
            va="bottom",
            path_effects=[pe.withStroke(linewidth=4, foreground="white")],
        ),
        axis="columns",
    )
    unclustered.plot(ax=ax, c="k", alpha=0.9, markersize=3)
    return f, ax


# %% [markdown]
# ## Plot Remaining SLR load after bottom-up cluster load

# %%
f, ax = plot_clusters(
    dublin_boundary,
    esbmap_stations,
    esbmap_stations_clustered,
    "remaining_load_mva_upper",
)
plt.title(
    "Remaining SLR Substation capacity - Low Demand Scenario",
    fontsize=20,
)
props = dict(boxstyle="round", facecolor="yellow", alpha=0.5)
textstr = "Assuming:\n" "Mean residential peak load of 1.5kW\n"
# place a text box in upper left in axes coords
ax.text(
    0.67,
    0.99,
    textstr,
    transform=ax.transAxes,
    fontsize=12,
    verticalalignment="top",
    bbox=props,
)
f.savefig(data_dir / "Remaining Capacity 2kW Residential Load & Eirgrid DC Load.png")

# %%
f, ax = plot_clusters(
    dublin_boundary,
    esbmap_stations,
    esbmap_stations_clustered,
    "remaining_load_mva_lower",
)
plt.title(
    "Remaining SLR Substation capacity - High Demand Scenario",
    fontsize=20,
)
props = dict(boxstyle="round", facecolor="yellow", alpha=0.5)
textstr = "Assuming:\n" "Mean residential peak load of 2kW\n"
# place a text box in upper left in axes coords
ax.text(
    0.67,
    0.99,
    textstr,
    transform=ax.transAxes,
    fontsize=12,
    verticalalignment="top",
    bbox=props,
)

# %% [markdown]
# ## Plot Small Area Remaining Capacity

# %%
def replace_legend_items(legend, mapping):
    for txt in legend.texts:
        for k, v in mapping.items():
            if txt.get_text() == str(k):
                txt.set_text(v)


f, ax = plt.subplots(figsize=(20, 20))
bins = mc.UserDefined(
    esbmap_stations_clustered["demand_available_mva"], [-np.inf, 10, 60, np.inf]
)
mapping = dict([(i, s) for i, s in enumerate(bins.get_legend_classes())])
esbmap_stations_clustered.assign(hdd=bins.yb).plot(
    column="hdd",
    categorical=True,
    cmap="OrRd",
    legend=True,
    legend_kwds={"loc": "lower right"},
    ax=ax,
)
ax.set_axis_off()
replace_legend_items(ax.get_legend(), mapping)
plt.title("Demand Availability [MVA]", fontsize=20)
plt.show()
