---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.9.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from sklearn.neighbors import BallTree
```

# Get LA boundaries

```python
dublin_admin_county_boundaries = (
    gpd.read_file("/home/wsl-rowanm/Data/dublin_admin_county_boundaries")
    .to_crs(epsg=2157) # read & convert to ITM or epsg=2157
)
```

# Get MV Network


- 10
 - Three Phase Overhead Electricity Lines
- 11
 - Single Phase Overhead Electricity Lines
- 12
 - Pole [p]
 - Pole Mounted Capacitor [0]
 - Pole Mounted Regulator [S]
 - Pole Mounted Single Phase Substation [t]
 - Pole Mounted Three Phase Substation [T]
 - Ground Mounted HV Station Location [T]
- 13
 - Ground Mounted MV Substation
- 14
 - Underground Electricity Cables
- 15
 - Ground Mounted MV URD Substation [U]
 - Ground Mounted MV Kiosk [K]
 - Underground Cable Joint [H]

... where **[ ] = Text Character**

... from ESB Networks Data Style Mappings v1.1

```python
lvmv_network = gpd.read_parquet(
    "/home/wsl-rowanm/Data/dublin-electricity-network/dublin_lvmv_network.parquet",
)
```

```python
lv_network = lvmv_network.query("`voltage_kv` == 'lv'").copy()
```

```python
mv_network = lvmv_network.query("`voltage_kv` == 'mv'").copy()
```

```python
mv_network_lines = lvmv_network.query("`Level` == [10, 11, 14]").copy()
```

```python
base = dublin_admin_county_boundaries.boundary.plot(edgecolor='red', figsize=(25,25))
mv_network_lines.plot(ax=base, markersize=0.1)
```

# Get 38kV, 110kV & 220kV  stations

... there is no 400kV station in Dublin

```python
hv_network = (
    gpd.read_parquet(
        "/home/wsl-rowanm/Data/dublin-electricity-network/dublin_hv_network.parquet",
    )
    .to_crs(epsg=2157)
)
```

```python
hv_stations = (
    hv_network.query("`Level` == [20, 30, 40]")
    .copy()
    .explode() # un-dissolve station locations from multipoint to single points
    .reset_index()
    .drop(columns="level_1")
)
```

# Get Geocoded stations

```python
slr_stations = (
    gpd.read_file(
        "/home/wsl-rowanm/Data/dublin-stations-slr-totals.geojson",
        driver="GeoJSON",
    )
    .to_crs(epsg=2157)
)
```

# Extract stations by voltage level

```python
network_stations_38kv = hv_stations.query("`Level` == 20")
```

```python
network_stations_110kv = hv_stations.query("`Level` == 30")
```

```python
network_stations_220kv = hv_stations.query("`Level` == 40")
```

# Link stations via MV network

```python
slr_stations_38kv = slr_stations.query("`voltage_level` == '38/MV station'")
```

```python
slr_stations_110kv = slr_stations.query("`voltage_level` == '110/MV station'")
```

```python
slr_stations_220kv = slr_stations.query("`voltage_level` == '220kV station'")
```

# Plot network stations vs slr_stations

```python
base = dublin_admin_county_boundaries.boundary.plot(edgecolor='red', figsize=(25,25))
slr_stations_38kv.geometry.buffer(1000).plot(ax=base, color='green')
network_stations_38kv.plot(ax=base, markersize=10, color='blue')
```

```python
base = dublin_admin_county_boundaries.boundary.plot(edgecolor='red', figsize=(25,25))
slr_stations_110kv.geometry.buffer(1000).plot(ax=base, color='green')
network_stations_110kv.plot(ax=base, markersize=10, color='blue')
```

```python
base = dublin_admin_county_boundaries.boundary.plot(edgecolor='red', figsize=(25,25))
slr_stations_220kv.geometry.buffer(1000).plot(ax=base, color='green')
network_stations_220kv.plot(ax=base, markersize=10, color='blue')
```

## Link stations to nearest geocoded station


From https://stackoverflow.com/questions/58893719/find-nearest-point-in-other-dataframe-with-a-lot-of-data

Also see:
- https://stackoverflow.com/questions/56520780/how-to-use-geopanda-or-shapely-to-find-nearest-point-in-same-geodataframe
- https://gis.stackexchange.com/questions/222315/geopandas-find-nearest-point-in-other-dataframe

```python
def link_nearest_points(left, right):

    left = left.copy()
    right = right.copy()
    
    left["x"] = left.geometry.x
    left["y"] = left.geometry.y
    
    right["x"] = right.geometry.centroid.x
    right["y"] = right.geometry.centroid.y

    tree = BallTree(right[['x', 'y']].values, leaf_size=2)

    left['distance_nearest'], left['id_nearest'] = tree.query(
        left[['x', 'y']].values,
        k=1, # The number of nearest neighbors
    )

    return pd.merge(
        left,
        right,
        left_on="id_nearest",
        right_index=True,
    )
```

```python
network_stations_38kv_linked = link_nearest_points(slr_stations_38kv, network_stations_38kv)
```

```python
network_stations_110kv_linked = link_nearest_points(slr_stations_110kv, network_stations_110kv)
```

```python
network_stations_220kv_linked = link_nearest_points(slr_stations_220kv, network_stations_220kv)
```
