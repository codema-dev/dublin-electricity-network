import pandas as pd


def clean_capacitymap(capacitymap_df):

    capacitymap_df["available_capacity"] = capacitymap_df["Marker"].map(
        {
            "Blue": "mixed",
            "Green": "significant",
            "Orange": "limited",
            "Red": "none",
        }
    )  # From https://www.esbnetworks.ie/demand-availability-capacity-map

    capacitymap_df["station_name"] = (
        capacitymap_df["Title"].str.split(" - ").str.get(0).str.lower()
    )

    capacitymap_df["station_voltages"] = (
        capacitymap_df["Title"].str.split(" - ").str.get(1).str.lower()
    )

    return capacitymap_df


def clean_heatmap(heatmap_df):

    hv_station_rows = ~heatmap_df["Station Name"].str.contains("MV/LV Substation")
    heatmap_df["station_name"] = (
        heatmap_df.loc[hv_station_rows]
        .loc[:, "Station Name"]
        .str.lower()
        .str.extract(r"(.*?) \d")
    )
    heatmap_df["station_name"] = heatmap_df["station_name"].fillna("mv/lv")

    return heatmap_df