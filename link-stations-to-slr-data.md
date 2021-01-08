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
station_locations = gpd.read_file(
    "data/dublin-stations-nominatim.geojson",
    driver="GeoJSON",
)
```

```python
station_slr = (
    pd.read_excel("data/dublin-station-slr.xlsx", engine="openpyxl")
    .assign(station_name=lambda x: x.station_name.ffill())
    .rename(columns={"station_name": "station"})
)
```

```python
station_totals = station_slr.groupby("station", as_index=False).sum()
```

```python
stations = station_locations.merge(station_totals)
```

```python
stations.to_file("data/dublin-station-slr-totals.geojson", driver="GeoJSON")
```

```python
stations.drop(columns="geometry").to_excel("data/dublin-station-slr-totals.xlsx", engine="openpyxl")
```

```python
stations
```

```python
stations.plot()
```

# Customer Stations

```python
mask = station_slr["capacity_feeder_customer"].fillna("").astype(str).str.contains("Customer")
customer_stations = station_slr[mask]
```

```python
customer_stations
```
