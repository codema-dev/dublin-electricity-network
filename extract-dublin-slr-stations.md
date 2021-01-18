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
import re

import geopandas as gpd
import pandas as pd
import numpy as np
```

# Get Dublin station names

```python
dublin_station_names = (
    pd.read_csv("~/Data/dublin-substation-names.csv")
    .assign(slr_station=lambda df: df.slr_station.fillna(df.station)) # replace finglas/mcdermott etc with mesh: orange etc
    .slr_station
    .rename("station")
    .tolist()
)
```

# Get Ireland Station Special Load Readings (SLR)

```python
ireland_slr = (
    pd.read_csv("~/Data/slr-2019-20.csv")
    .assign(total=lambda df: df.station.notnull())
    .assign(station=lambda df: df.station.str.lower().ffill())
)
```

# Get Dublin station locations

```python
dublin_station_locations = gpd.read_file("/home/wsl-rowanm/Data/dublin-stations-google.geojson", driver="GeoJSON")
```

# Extract SLR Dublin stations


see https://stackoverflow.com/questions/11350770/select-by-partial-string-from-a-pandas-dataframe/55335207#55335207

```python
# escape characters that can be interpreted as regex metacharacters like ^ $ * + ? { } [ ] \ | ( )
# ... so can handle 'grange (dr)'
dublin_station_names_escaped = map(re.escape, dublin_station_names)
dublin_station_names_regex = '|'.join(dublin_station_names_escaped)

mask = (
    (ireland_slr.station.str.contains(dublin_station_names_regex))
    & (ireland_slr.station != 'ennistymon') # false positive result
)
dublin_slr = ireland_slr.loc[mask].copy()
```

# Check if all stations have been matched

```python
unique_dublin_slr_stations = set(dublin_slr.station.unique())
```

```python
unique_dublin_station_names = set(dublin_station_names)
```

```python
unique_dublin_station_names - unique_dublin_slr_stations
```

```python
unique_dublin_slr_stations - unique_dublin_station_names 
```

# Get station totals

```python
dublin_slr_totals = dublin_slr.query("`total` == True")
```

# Link station totals to station coordinates

```python
dublin_slr_totals_geolocated = dublin_station_locations.merge(dublin_slr_totals)
```

# Save

```python
dublin_slr_totals_geolocated.to_file("/home/wsl-rowanm/Data/dublin-stations-slr-totals.geojson", driver="GeoJSON")
```

```python
dublin_slr_totals_geolocated.to_csv("/home/wsl-rowanm/Data/dublin-stations-slr-totals.csv")
```
