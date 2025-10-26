import os
import sys
from multiprocessing import Process
import folium
import polars as pl
from folium.plugins import HeatMap, MarkerCluster, AntPath
from constants import YEAR, MONTH, DAY, EVENT_TYPE, COUNTRY, LATITUDE, LONGITUDE, REGION, NATURAL_EVENT_ID, \
    LATITUDE_END, LONGITUDE_END, HTMLS_PATH, EVENTS_MAPS, INTENSITY_MAPS, INTENSITY
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

        filename = f'{HTMLS_PATH}events_map_{event_type}.html'
        print(f"💾 Saving {filename} ...")
        mappa.save(filename)
        print(f"✅ {filename} saved!")

    # Heatmap
    heat_data = df_events.filter(pl.col(LATITUDE).is_not_null() & pl.col(LONGITUDE).is_not_null()) \
        .select([LATITUDE, LONGITUDE]).to_numpy().tolist()

    heat_map = folium.Map(location=[20, 0], zoom_start=2)
    HeatMap(heat_data, radius=15).add_to(heat_map)
    heat_map.save(HTMLS_PATH + 'events_heatmap.html')
    print("✅ Heatmap saved!")

    print("🎉 Finished generating maps!")


def generate_intensity_maps(engine: SqlEngine, maps_to_generate):
    print(f"🗺️ Starting generation intensity heat maps: {maps_to_generate}...")
    subset_default = engine.get_intensity_df_maps(True)
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
        filename = f"{HTMLS_PATH}intensity_map_{event_type}.html"
        print(f"💾 Saving {filename} ...")
        m.save(filename)
        print(f"✅ {filename} saved!")



def start_map_generation(engine: SqlEngine):
    """Avvia un processo separato per generare le mappe, se non esistono già."""

    missing = [f for f in EVENTS_MAPS if not os.path.exists(f"{HTMLS_PATH}events_map_{f}.html")]
    missing_heatmaps = [f for f in INTENSITY_MAPS if not os.path.exists(f"{HTMLS_PATH}intensity_map_{f}.html")]

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