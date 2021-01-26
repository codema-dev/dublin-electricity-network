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

```python
!wget -O data/external/dublin-substation-names.xlsx https://zenodo.org/record/4446622/files/dublin-substation-names.xlsx
```

```python
dublin_station_names = pd.read_excel(
    "data/external/dublin-substation-names.xlsx",
    engine="openpyxl",
)
```

```python
dublin_station_addresses = dublin_station_names["town"] + ", Dublin, Ireland"
```

# Get Dublin Boundary

... so can filter out failed non-Dublin results

```python
!wget -O data/external/dublin_boundary.geojson https://zenodo.org/record/4432494/files/dublin_boundary.geojson 
```

```python
dublin_boundary = gpd.read_file("data/external/dublin_boundary.geojson", driver="GeoJSON")[["geometry"]]
```

# Geocode stations

```python
raw_locations = gpd.tools.geocode(
    new_addresses,
    provider="Google",
    api_key="<YOUR API KEY>", # https://developers.google.com/maps/documentation/geocoding/start
)
```

```python
dublin_station_locations = raw_locations.join(dublin_station_names)
```

# Check if results in Dublin boundary

```python
stations_within_dublin_boundary = gpd.sjoin(dublin_station_locations, dublin_boundary)
```

```python
display("Percentage located within Dublin: " + get_percentage(stations_within_dublin_boundary, dublin_station_locations) + "%") 
```

```python
incorrectly_geocoded_results = (
    pd.merge(dublin_station_locations, stations_within_dublin_boundary, indicator=True, how="left")
    .query("`_merge` == 'left_only'")
)
```

```python
incorrectly_geocoded_results
```

# Manually verify validity of geocode results

... mark uncertain & incorrect results via [`pd.DataFrame.iloc`](https://pandas.pydata.org/docs/user_guide/10min.html#selection)

```python
# dublin_station_locations["incorrect"] = False
```

```python
# dublin_station_locations.iloc[109, -2] = True
```

# Save result

```python
dublin_station_locations.to_file(
    "data/outputs/dublin-stations-google.geojson",
    driver="GeoJSON",
)
```
