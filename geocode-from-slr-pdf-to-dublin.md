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
import geopandas as gpd
import pandas as pd
```

```python
def get_percentage(left, right):
    
    return str(round(len(left) / len(right) * 100, 2))
```

# Get Dublin stations

... guessed from a mixture of geocoding and local knowledge

```python
station_names = pd.read_csv(
    "data/raw/dublin-station-names.txt",
    header=None,
    names=["station"],
    squeeze=True
)
```

```python
station_addresses = station_names + ", Dublin, Ireland"
```

# Get Dublin Boundary

... so can filter out non-Dublin results

```python
!wget -q -O data/external/dublin_boundary.geojson https://zenodo.org/record/4432494/files/dublin_boundary.geojson 
```

```python
dublin_boundary = gpd.read_file("data/dublin_boundary.geojson", driver="GeoJSON").geometry
```

# Geocode stations

```python
station_locations = gpd.tools.geocode(
    station_addresses,
    provider="Nominatim",
    user_agent="codema",
    timeout=20,
)
```

```python
raw_station_locations = station_locations.join(station_names)
```

```python
display("Percentage located within Dublin: " + get_percentage(station_locations, raw_station_locations) + "%") 
```

```python
stations = gpd.sjoin(raw_station_locations, dublin_boundary).drop(columns=["index_right"])
```

```python
stations.to_file(
    "data/outputs/dublin-stations-nominatim.geojson",
    driver="GeoJSON",
)
```

# Get missing substations

```python
raw_station_locations.merge(station_locations, how="left", indicator=True).query("`_merge` == 'left_only'")
```
