import os
import sys
from multiprocessing import Process
import folium
from threading import Thread
import polars as pl
from folium.plugins import HeatMap, MarkerCluster, AntPath
from constants import YEAR, MONTH, DAY, EVENT_TYPE, COUNTRY, LATITUDE, LONGITUDE, REGION, NATURAL_EVENT_ID, \
    LATITUDE_END, LONGITUDE_END, EVENTS_MAPS, INTENSITY_MAPS, INTENSITY, MAPS_PATH, \
    LEGEND_HTML_OUTLIERS_HEATMAPS
from sql_engine import SqlEngine

def print_progress_bar(percentuale, lunghezza_barra=20):
    blocchi_compilati = int(lunghezza_barra * percentuale)
    barra = "[" + "=" * (blocchi_compilati - 1) + ">" + " " * (lunghezza_barra - blocchi_compilati) + "]"
    sys.stdout.write(f"\r{barra} {percentuale * 100:.2f}% completo")
    sys.stdout.flush()
    if percentuale == 1:
        print("")

def generate_maps(df_events, df_tornadoes, maps_to_generate):
    print("🗺️ Starting generation maps with Folium...")

    # Filtro dati
    df_locations = df_events.filter(pl.col(LATITUDE).is_not_null() & pl.col(LONGITUDE).is_not_null()) \
        .join(df_tornadoes, how="left", left_on="id", right_on=NATURAL_EVENT_ID)

    # Esempio: se vuoi un'unica mappa “all”


    for event_type in maps_to_generate:
        mappa = folium.Map(location=[20, 0], zoom_start=2)
        marker_cluster = MarkerCluster().add_to(mappa)
        subset = df_locations if event_type == "all" else df_locations.filter(pl.col(EVENT_TYPE) == event_type)

        for i, row in enumerate(subset.iter_rows(named=True)):
            print_progress_bar(i / len(subset), lunghezza_barra=40)
            event_color = (
                'orange' if row[EVENT_TYPE] == 'earthquake' else
                'blue' if row[EVENT_TYPE] == 'tsunami' else
                'red' if row[EVENT_TYPE] == 'eruption' else
                'purple' if row[EVENT_TYPE] == 'tornado' else
                'green'
            )

            folium.CircleMarker(
                location=[row[LATITUDE], row[LONGITUDE]],
                radius=3,
                color=event_color,
                fill=True,
                popup="event location"
            ).add_to(marker_cluster)

            if row.get(LATITUDE_END) and row.get(LONGITUDE_END):
                AntPath(
                    locations=[
                        [row[LATITUDE], row[LONGITUDE]],
                        [row[LATITUDE_END], row[LONGITUDE_END]]
                    ],
                    color=event_color,
                    weight=3,
                    opacity=0.8,
                    popup=f"{row[EVENT_TYPE]} trajectory"
                ).add_to(marker_cluster)

                folium.CircleMarker(
                    location=[row[LATITUDE_END], row[LONGITUDE_END]],
                    radius=3,
                    color=event_color,
                    fill=True,
                    popup="End location"
                ).add_to(marker_cluster)

        filename = f'{MAPS_PATH}events_maps/events_map_{event_type}.html'
        print(f"💾 Saving {filename} ...")
        mappa.save(filename)
        print(f"✅ {filename} saved!")

    # Heatmap
    heat_data = df_events.filter(pl.col(LATITUDE).is_not_null() & pl.col(LONGITUDE).is_not_null()) \
        .select([LATITUDE, LONGITUDE]).to_numpy().tolist()

    heat_map = folium.Map(location=[20, 0], zoom_start=2)
    HeatMap(heat_data, radius=15).add_to(heat_map)
    heat_map.save(f'{MAPS_PATH}events_maps/events_heatmap.html')
    print("✅ Heatmap saved!")

    print("🎉 Finished generating maps!")





def generate_intensity_maps(engine: SqlEngine, maps_to_generate):
    print(f"🗺️ Starting generation intensity heat maps: {maps_to_generate}...")
    subset_default = engine.get_intensity_df(True, maps=True)
    for event_type in maps_to_generate:
        subset = subset_default if event_type == "all" else subset_default.filter(pl.col(EVENT_TYPE) == event_type)
        m = folium.Map(location=[20, 0], zoom_start=2)
        heat_data = []
        for i, row in enumerate(subset.iter_rows(named=True)):
            print_progress_bar(i / len(subset))
            heat_data.append([row[LATITUDE], row[LONGITUDE], row[INTENSITY]])
        HeatMap(heat_data,
                radius=15,
                blur=20,
                max_zoom=13,
                gradient={0.0: 'blue', 0.5: 'yellow', 1.0: 'red'}).add_to(m)
        filename = f"{MAPS_PATH}intensity_maps/intensity_map_{event_type}.html"
        print(f"💾 Saving {filename} ...")
        m.save(filename)
        print(f"✅ {filename} saved!")


def create_outliers_heatmap(df_outliers, map_name):
    """
    Creates a geographic heatmap of outliers with different colors for upper and lower outliers using Folium.
    Saves the map as an HTML file in the templates folder.

    Parameters:
    -----------
    df_outliers : polars.DataFrame
        DataFrame containing outliers with the following columns:
        - latitude: float, latitude coordinate
        - longitude: float, longitude coordinate
        - outlier_type: str, 'upper' or 'lower' to indicate if outlier is above or below bounds
    output_filename : str
        Name of the output HTML file (default: 'outliers_heatmap.html')

    Returns:
    --------
    str
        Path to the saved HTML file
    """

    # Check if dataframe is empty
    if df_outliers.height == 0:
        # Return empty map centered on world
        m = folium.Map(location=[20, 0], zoom_start=2)
        folium.Marker(
            location=[20, 0],
            popup="No outliers to display",
            icon=folium.Icon(color='gray', icon='info-sign')
        ).add_to(m)
    else:
        # Calculate center coordinates
        center_lat = df_outliers['latitude'].mean()
        center_lon = df_outliers['longitude'].mean()

        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=4,
            tiles='OpenStreetMap'
        )

        # Separate upper and lower outliers
        upper_outliers = df_outliers.filter(pl.col('outlier_type') == 'high')
        lower_outliers = df_outliers.filter(pl.col('outlier_type') == 'low')

        # Create feature groups for layer control
        upper_group = folium.FeatureGroup(name='Upper Outliers (Above Bound)', show=True)
        lower_group = folium.FeatureGroup(name='Lower Outliers (Below Bound)', show=True)

        # Add upper outliers heatmap (red)
        if upper_outliers.height > 0:
            upper_data = [
                [row['latitude'], row['longitude']]
                for row in upper_outliers.iter_rows(named=True)
            ]

            HeatMap(
                upper_data,
                radius=15,
                blur=20,
                gradient={
                    0.0: 'rgba(255, 255, 255, 0)',
                    0.2: 'rgba(255, 200, 200, 0.3)',
                    0.5: 'rgba(255, 100, 100, 0.6)',
                    0.8: 'rgba(255, 50, 50, 0.8)',
                    1.0: 'rgba(255, 0, 0, 1)'
                },
                min_opacity=0.3,
                max_val=1.0
            ).add_to(upper_group)

            # Add markers for upper outliers
            for row in upper_outliers.iter_rows(named=True):
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=5,
                    popup=f"<b>Upper Outlier</b><br>Lat: {row['latitude']:.4f}<br>Lon: {row['longitude']:.4f}",
                    color='darkred',
                    fill=True,
                    fillColor='red',
                    fillOpacity=0.7,
                    weight=2
                ).add_to(upper_group)

        # Add lower outliers heatmap (blue)
        if lower_outliers.height > 0:
            lower_data = [
                [row['latitude'], row['longitude']]
                for row in lower_outliers.iter_rows(named=True)
            ]

            HeatMap(
                lower_data,
                radius=15,
                blur=20,
                gradient={
                    0.0: 'rgba(255, 255, 255, 0)',
                    0.2: 'rgba(200, 200, 255, 0.3)',
                    0.5: 'rgba(100, 100, 255, 0.6)',
                    0.8: 'rgba(50, 50, 255, 0.8)',
                    1.0: 'rgba(0, 0, 255, 1)'
                },
                min_opacity=0.3,
                max_val=1.0
            ).add_to(lower_group)

            # Add markers for lower outliers
            for row in lower_outliers.iter_rows(named=True):
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=5,
                    popup=f"<b>Lower Outlier</b><br>Lat: {row['latitude']:.4f}<br>Lon: {row['longitude']:.4f}",
                    color='darkblue',
                    fill=True,
                    fillColor='blue',
                    fillOpacity=0.7,
                    weight=2
                ).add_to(lower_group)

        # Add feature groups to map
        upper_group.add_to(m)
        lower_group.add_to(m)

        # Add layer control
        folium.LayerControl().add_to(m)

        m.get_root().html.add_child(folium.Element(LEGEND_HTML_OUTLIERS_HEATMAPS))

    # Save the map to the templates folder
    output_path = f'{MAPS_PATH}outliers_maps/{map_name}'
    print(f"saving {output_path} ...")
    m.save(output_path)
    print(f"✅ Saved {output_path}")
    return

def start_map_generation(engine: SqlEngine):
    """Avvia un processo separato per generare le mappe, se non esistono già."""

    missing = [f for f in EVENTS_MAPS if not os.path.exists(f"{MAPS_PATH}events_maps/events_map_{f}.html")]
    missing_heatmaps = [f for f in INTENSITY_MAPS if not os.path.exists(f"{MAPS_PATH}intensity_maps/intensity_map_{f}.html")]

    if missing:
        print(f"🧩 Missing some maps: {missing}")
        p = Process(target=generate_maps, args=(engine.get_all_events(), engine.get_all_tornado_traces(), missing))
        p.start()
        print("🚀 Map generations for events starting in background.")
    if missing_heatmaps:
        print(f"🧩 Missing some intensity maps: {missing_heatmaps}")
        p = Process(target=generate_intensity_maps, args=(engine, missing_heatmaps))
        p.start()
        print("🚀 Map generations for intensities starting in background.")
    elif not missing:
        print("✅ All maps already generated.")

def generate_map_outliers(df_outliers, event_type, column):
    map_name = f"outliers_heatmap_{event_type}_{column}.html"
    if os.path.exists(os.path.join(MAPS_PATH, f"outliers_maps/{map_name}")):
        print(f"✅ Outliers heatmap for {event_type} and column '{column}' already exists.")
        return
    thread = Thread(target=create_outliers_heatmap, args=(df_outliers.clone(), map_name))
    thread.daemon = True
    thread.start()
    print(f"🚀 Outliers heatmap for {event_type} and column '{column}' started in background.")