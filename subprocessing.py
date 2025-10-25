import os
from multiprocessing import Process
import folium
import polars as pl
from folium.plugins import HeatMap, MarkerCluster, AntPath
from constants import YEAR, MONTH, DAY, EVENT_TYPE, COUNTRY, LATITUDE, LONGITUDE, REGION, NATURAL_EVENT_ID, \
    LATITUDE_END, LONGITUDE_END, HTMLS_PATH

def generate_maps(df_events, df_tornadoes):
    print("🗺️ Avvio generazione mappe Folium...")

    # Filtro dati
    df_locations = df_events.filter(pl.col(LATITUDE).is_not_null() & pl.col(LONGITUDE).is_not_null()) \
        .join(df_tornadoes, how="left", left_on="id", right_on=NATURAL_EVENT_ID)

    # Esempio: se vuoi un'unica mappa “all”
    types = df_events[EVENT_TYPE].unique().to_list() + ["all"]

    for event_type in types:
        mappa = folium.Map(location=[20, 0], zoom_start=2)
        marker_cluster = MarkerCluster().add_to(mappa)
        subset = df_locations if event_type == "all" else df_locations.filter(pl.col(EVENT_TYPE) == event_type)

        for i, row in enumerate(subset.iter_rows(named=True)):
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

        filename = f'{HTMLS_PATH}events_map_{event_type}.html'
        print(f"💾 Salvataggio {filename} ...")
        mappa.save(filename)
        print(f"✅ {filename} salvata!")

    # Heatmap
    heat_data = df_events.filter(pl.col(LATITUDE).is_not_null() & pl.col(LONGITUDE).is_not_null()) \
        .select([LATITUDE, LONGITUDE]).to_numpy().tolist()

    heat_map = folium.Map(location=[20, 0], zoom_start=2)
    HeatMap(heat_data, radius=15).add_to(heat_map)
    heat_map.save(HTMLS_PATH + 'events_heatmap.html')
    print("✅ Heatmap salvata!")

    print("🎉 Generazione mappe completata!")

def start_map_generation():
    """Avvia un processo separato per generare le mappe, se non esistono già."""
    required_maps = ["events_map_all.html", "events_heatmap.html"]
    missing = [f for f in required_maps if not os.path.exists(HTMLS_PATH + f)]

    if missing:
        print(f"🧩 Mancano alcune mappe: {missing}")
        p = Process(target=generate_maps)
        p.start()
        print("🚀 Processo di generazione mappe avviato in background.")
    else:
        print("✅ Tutte le mappe già presenti, nessuna generazione necessaria.")