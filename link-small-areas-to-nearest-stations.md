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
from shapely.geometry import Point
from sklearn.neighbors import BallTree
```

# Get Dublin Station Totals

```python
station_totals = (
    gpd.read_file("data/outputs/dublin-station-slr-totals.geojson", driver="GeoJSON")
    .to_crs(epsg=2157) # convert to ITM
)
```

```python
station_totals
```

```python
station_totals.plot()
```

# Get Dublin Small Area Boundaries

```python
!wget -O data/external/Small_Areas_Ungeneralised_-_OSi_National_Statistical_Boundaries_-_2015-shp.zip https://opendata.arcgis.com/datasets/c85e610da1464178a2cd84a88020c8e2_3.zip
```

```python
!echo "y" unzip data/external/Small_Areas_Ungeneralised_-_OSi_National_Statistical_Boundaries_-_2015-shp.zip -d data/external/Small_Areas_Ungeneralised_-_OSi_National_Statistical_Boundaries_-_2015-shp
```

```python
small_areas = (
    gpd.read_file("data/external/Small_Areas_Ungeneralised_-_OSi_National_Statistical_Boundaries_-_2015-shp")
    .query("`COUNTYNAME` == ['South Dublin', 'Dún Laoghaire-Rathdown', 'Fingal', 'Dublin City']")
    .loc[:, ["SMALL_AREA", "COUNTYNAME", "geometry"]]
    .to_crs(epsg=2157) # convert to ITM
)
```

```python
small_areas
```

# Get Small Area demands

... <span style="color:red">**Not yet publicly released**</span>.

```python
small_area_demand = (
    gpd.read_file("data/external/small_area_residential_demand.geojson", driver="GeoJSON")
    .to_crs(epsg=2157) # convert to ITM
    .loc[:, ["GEOGID", "sa_demand"]]
    .rename(columns={"GEOGID": "SMALL_AREA"})
)
```

```python
small_area_demand
```

# Link Small Areas to nearest station 


From https://stackoverflow.com/questions/58893719/find-nearest-point-in-other-dataframe-with-a-lot-of-data

Also see:
- https://stackoverflow.com/questions/56520780/how-to-use-geopanda-or-shapely-to-find-nearest-point-in-same-geodataframe
- https://gis.stackexchange.com/questions/222315/geopandas-find-nearest-point-in-other-dataframe

```python
station_totals["x"] = station_totals.geometry.x
station_totals["y"] = station_totals.geometry.y
```

```python
small_areas["x"] = small_areas.geometry.centroid.x
small_areas["y"] = small_areas.geometry.centroid.y
```

```python
tree = BallTree(station_totals[['x', 'y']].values, leaf_size=2)

small_areas['distance_nearest'], small_areas['id_nearest'] = tree.query(
    small_areas[['x', 'y']].values,
    k=1, # The number of nearest neighbors
)
```

```python
small_area_stations = pd.merge(
    station_totals.station,
    small_areas,
    left_on=station_totals.index,
    right_on="id_nearest"
)
```

```python
small_areas_with_stations = small_areas.merge(small_area_stations).loc[:, ["SMALL_AREA", "station", "geometry"]]
```

```python
base = small_areas_with_stations.plot(
    column="station", 
    legend=True,
    figsize=(20,20),
    cmap='tab20c',
)

station_totals.plot(
    ax=base,
    color='k',
)
```

# Link small area demands to stations

```python
small_area_demands_with_stations = small_areas_with_stations.merge(small_area_demand)
```

```python
demand_at_stations = small_area_demands_with_stations.groupby("station").sum() / 10**6
```

```python
demand_at_stations
```

```python
stations_with_demands = pd.merge(
    demand_at_stations,
    station_totals,
    left_index=True,
    right_on="station"
)
```

```python
stations_with_demands.to_csv("data/outputs/station_demands.csv", index=False)
```
