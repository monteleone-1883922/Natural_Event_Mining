import os
import sys
from multiprocessing import Process
import folium
from threading import Thread
import polars as pl
from folium.plugins import HeatMap, MarkerCluster, AntPath, TimestampedGeoJson
from constants import YEAR, MONTH, DAY, EVENT_TYPE, COUNTRY, LATITUDE, LONGITUDE, REGION, NATURAL_EVENT_ID, \
    LATITUDE_END, LONGITUDE_END, EVENTS_MAPS, INTENSITY_MAPS, INTENSITY, MAPS_PATH, \
    LEGEND_HTML_OUTLIERS_HEATMAPS, TEMPLATES_PATH
from sql_engine import SqlEngine
import matplotlib.cm as cm
import matplotlib.colors as mcolors



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

        filename = f'{TEMPLATES_PATH}{MAPS_PATH}events_maps/events_map_{event_type}.html'
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


def generate_cluster_map(df_clustered: pl.DataFrame, n_clusters: int, cell_size: int, id: int):
    """
    Genera una mappa Folium con punti colorati per cluster e dimensione proporzionale all'intensità.
    """

    # Colori per i cluster
    cmap = cm.get_cmap('tab10', n_clusters)
    cluster_colors = [mcolors.to_hex(cmap(i)) for i in range(n_clusters)]

    # Crea mappa centrata sul centro medio
    center_lat = df_clustered["lat_grid"].mean()
    center_lon = df_clustered["lon_grid"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=2, tiles='cartodb positron')

    # Usa MarkerCluster per performance migliori
    marker_cluster = MarkerCluster().add_to(m)

    # Aggiungi un punto per ciascun evento
    for row in df_clustered.iter_rows(named=True):
        cluster_id = int(row.get("cluster", 0))
        color = cluster_colors[cluster_id % len(cluster_colors)]

        intensity = float(row.get("intensity", 0) or 0)
        radius = max(3.0, min(15.0, intensity * 2.0))  # scala proporzionale all’intensità

        event_type = row.get(EVENT_TYPE, "unknown")
        deaths = row.get("deaths", 0)
        damage = row.get("damagemillionsdollars", 0)
        houses = row.get("housesdestroyed", 0)

        popup_html = f"""
        <b>Cluster:</b> {cluster_id}<br>
        <b>Event Type:</b> {event_type}<br>
        <b>Intensity:</b> {intensity:.2f}<br>
        <b>Deaths:</b> {deaths}<br>
        <b>Damage (M$):</b> {damage}<br>
        <b>Houses Destroyed:</b> {houses}
        """

        folium.CircleMarker(
            location=[row["lat_grid"], row["lon_grid"]],
            radius=radius,
            color=color,
            fill=True,
            fill_opacity=0.6,
            popup=folium.Popup(popup_html, max_width=250)
        ).add_to(marker_cluster)

    # Aggiungi una legenda
    legend_html = """
    <div style="
        position: fixed;
        bottom: 20px; left: 20px; width: 220px;
        background-color: white; border:2px solid grey; z-index:9999; font-size:14px;
        padding: 10px; border-radius: 8px;">
        <b>Cluster Colors</b><br>
    """
    for i, color in enumerate(cluster_colors):
        legend_html += f'<i style="background:{color};width:15px;height:15px;float:left;margin-right:5px;opacity:0.8;"></i> Cluster {i}<br>'
    legend_html += "</div>"

    m.get_root().html.add_child(folium.Element(legend_html))

    # Salva la mappa in HTML
    filename = f'{TEMPLATES_PATH}{MAPS_PATH}cluster_maps/geographic_map_{n_clusters}_clusters_{cell_size}_cell_size_{id}.html'
    print(f"💾 Saving {filename} ...")
    m.save(filename)
    print(f"✅ {filename} saved!")

    return



def generate_intensity_maps(engine: SqlEngine, maps_to_generate):
    print(f"🗺️ Starting generation intensity heat maps: {maps_to_generate}...")
    subset_default = engine.get_joined_events_df(True, "intensity_map")
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
        filename = f"{TEMPLATES_PATH}{MAPS_PATH}intensity_maps/intensity_map_{event_type}.html"
        print(f"💾 Saving {filename} ...")
        m.save(filename)
        print(f"✅ {filename} saved!")
        return


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
    output_path = f'{TEMPLATES_PATH}{MAPS_PATH}outliers_maps/{map_name}'
    print(f"saving {output_path} ...")
    m.save(output_path)
    print(f"✅ Saved {output_path}")
    return


def generate_temporal_cluster_map(df_clustered: pl.DataFrame, n_clusters, eps_spatial, eps_temporal, min_samples, id):
    """
    Genera una mappa Folium con time slider per visualizzare l'evoluzione temporale dei cluster.

    Args:
        df_clustered: DataFrame con colonne cluster_st, lat, lon, days_epoch, intensity, event_type
        n_clusters: numero di cluster trovati
        id: identificatore per il filename
    """

    # Colori per i cluster (-1 = grigio per noise)
    cmap = cm.get_cmap('tab10', n_clusters + 1)
    cluster_colors = {-1: '#808080'}  # grigio per noise
    for i in range(n_clusters):
        cluster_colors[i] = mcolors.to_hex(cmap(i))

    # Crea mappa centrata sul centro medio
    center_lat = df_clustered[LATITUDE].mean()
    center_lon = df_clustered[LONGITUDE].mean()
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=3,
        tiles='cartodb positron'
    )

    # Prepara features per TimestampedGeoJson
    features = []

    # AGGIUNGI QUESTO: Contatore per debug
    total_rows = df_clustered.height
    processed_rows = 0

    for row in df_clustered.iter_rows(named=True):
        cluster_id = int(row.get("cluster_st", -1))
        color = cluster_colors.get(cluster_id, '#808080')

        # Estrai dati
        year = int(row.get(YEAR, 0))
        month = int(row.get(MONTH, 0))
        day = int(row.get(DAY, 0))

        # VERIFICA: Skippa righe con date invalide
        if year == 0 or month == 0 or day == 0:
            print(f"⚠️ Skipping row with invalid date: {year}-{month}-{day}")
            continue

        event_date = f"{year}-{month:02d}-{day:02d}"
        lat = float(row.get(LATITUDE, 0))
        lon = float(row.get(LONGITUDE, 0))
        intensity = float(row.get("intensity", 0) or 0)
        event_type = row.get(EVENT_TYPE, "unknown")
        deaths = row.get("deaths", 0) or 0
        damage = row.get("damagemillionsdollars", 0) or 0
        houses = row.get("housesdestroyed", 0) or 0
        days_epoch = row.get("days_epoch", 0)

        # Raggio proporzionale all'intensità
        radius = max(5, min(20, intensity * 2.5))

        # Popup HTML
        popup_text = f"""
        <b>Cluster:</b> {cluster_id if cluster_id >= 0 else 'Noise'}<br>
        <b>Date:</b> {event_date}<br>
        <b>Event Type:</b> {event_type}<br>
        <b>Intensity:</b> {intensity:.2f}<br>
        <b>Deaths:</b> {deaths}<br>
        <b>Damage (M$):</b> {damage}<br>
        <b>Houses Destroyed:</b> {houses}
        """

        # Feature GeoJSON con timestamp COMPLETO (non solo anno!)
        feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [lon, lat]
            },
            'properties': {
                'time': event_date,  # ✅ USA DATA COMPLETA invece di solo year
                'popup': popup_text,
                'icon': 'circle',
                'iconstyle': {
                    'fillColor': color,
                    'fillOpacity': 0.7,
                    'stroke': 'true',
                    'radius': radius,
                    'color': color,
                    'weight': 2
                }
            }
        }
        features.append(feature)
        processed_rows += 1

    # AGGIUNGI QUESTO: Verifica che tutti gli eventi siano stati processati
    print(f"📊 Processed {processed_rows}/{total_rows} events for map")
    if processed_rows < total_rows:
        print(f"⚠️ WARNING: {total_rows - processed_rows} events were skipped!")

    # Aggiungi TimestampedGeoJson con slider annuale
    TimestampedGeoJson(
        {
            'type': 'FeatureCollection',
            'features': features
        },
        period='P1Y',  # Intervallo ANNUALE
        add_last_point=True,
        auto_play=False,
        loop=False,
        max_speed=2,
        loop_button=True,
        date_options='YYYY',  # Mostra solo l'anno
        time_slider_drag_update=True,
        duration='P1Y'  # Durata di 1 anno per step
    ).add_to(m)

    # Aggiungi legenda cluster
    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 20px; left: 20px; width: 240px;
        background-color: white; border:2px solid grey; z-index:9999; font-size:13px;
        padding: 12px; border-radius: 8px; box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
        <b>🗺️ Spatio-Temporal Clusters</b><br>
        <small style="color: #666;">Use time slider to explore evolution</small><br><br>
    """

    # Aggiungi solo i primi 10 cluster per non sovraffollare
    for i in range(min(10, n_clusters)):
        color = cluster_colors[i]
        legend_html += f'<i style="background:{color};width:15px;height:15px;float:left;margin-right:8px;opacity:0.8;border-radius:50%;"></i> Cluster {i}<br>'

    if n_clusters > 10:
        legend_html += f'<small style="color: #999;">... and {n_clusters - 10} more clusters</small><br>'

    legend_html += f'<i style="background:#808080;width:15px;height:15px;float:left;margin-right:8px;opacity:0.5;border-radius:50%;"></i> Noise<br>'
    legend_html += "</div>"

    m.get_root().html.add_child(folium.Element(legend_html))

    # Aggiungi info box in alto a destra
    info_html = f"""
    <div style="
        position: fixed;
        top: 20px; right: 20px; width: 200px;
        background-color: rgba(255,255,255,0.95); border:2px solid #2c5aa0; z-index:9999; font-size:12px;
        padding: 10px; border-radius: 8px; box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
        <b>📊 Analysis Summary</b><br>
        <b>Total Clusters:</b> {n_clusters}<br>
        <b>Total Events:</b> {processed_rows}<br>
        <b>Clustered:</b> {df_clustered.filter(pl.col('cluster_st') >= 0).height}<br>
        <small style="color: #666;">Circle size = intensity</small>
    </div>
    """

    m.get_root().html.add_child(folium.Element(info_html))

    # Salva la mappa
    filename = f'{TEMPLATES_PATH}{MAPS_PATH}cluster_maps/spatial_temporal_map_{eps_spatial}_spatial_{eps_temporal}_temporal_{min_samples}_samples_{id}.html'
    print(f"💾 Saving temporal map {filename} ...")
    m.save(filename)
    print(f"✅ Temporal map saved with {processed_rows} events!")

    return filename


def start_map_generation(engine: SqlEngine):
    """Avvia un processo separato per generare le mappe, se non esistono già."""

    missing = [f for f in EVENTS_MAPS if not os.path.exists(f"{TEMPLATES_PATH}{MAPS_PATH}events_maps/events_map_{f}.html")]
    missing_heatmaps = [f for f in INTENSITY_MAPS if not os.path.exists(f"{TEMPLATES_PATH}{MAPS_PATH}intensity_maps/intensity_map_{f}.html")]

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
    if os.path.exists(f"{TEMPLATES_PATH}{MAPS_PATH}outliers_maps/{map_name}"):
        print(f"✅ Outliers heatmap for {event_type} and column '{column}' already exists.")
        return
    thread = Thread(target=create_outliers_heatmap, args=(df_outliers.clone(), map_name))
    thread.daemon = True
    thread.start()
    print(f"🚀 Outliers heatmap for {event_type} and column '{column}' started in background.")

def generate_map_geographic_cluster(df_clustered, n_clusters, cell_size, id):
    thread = Thread(target=generate_cluster_map, args=(df_clustered, n_clusters, cell_size, id))
    thread.daemon = True
    thread.start()
    print(f"🚀 Cluster map for {n_clusters} clusters and cell size '{cell_size}' started in background.")

def generate_map_spatiotemporal_cluster(df_clustered, n_clusters, eps_spatial, eps_temporal, min_samples, id):
    thread = Thread(target=generate_temporal_cluster_map, args=(df_clustered, n_clusters, eps_spatial, eps_temporal, min_samples, id))
    thread.daemon = True
    thread.start()
    print(f"🚀 Spatio-temporal cluster map for eps spatial '{eps_spatial}', eps temporal '{eps_temporal}' and min samples '{min_samples}' started in background.")