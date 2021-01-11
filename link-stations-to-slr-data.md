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

# Load data

```python
station_locations = gpd.read_file(
    "data/outputs/dublin-stations-nominatim.geojson",
    driver="GeoJSON",
)
```

```python
station_slr = (
    pd.read_excel("data/raw/dublin-station-slr.xlsx", engine="openpyxl")
)
```

# Link station totals to station coordinates

```python
station_totals = station_slr.query("station.notnull()")
# station_totals.to_excel("data/dublin-station-slr-totals.xlsx", engine="openpyxl")
```

```python
station_totals_by_location =  station_totals.groupby("station", as_index=False).sum()
```

```python
stations = station_locations.merge(station_totals_by_location)
```

```python
stations.plot()
```

```python
stations.to_file("data/dublin-station-slr-totals.geojson", driver="GeoJSON")
```
