from folium import CircleMarker

from sql_engine import engine
import polars as pl
import seaborn as sns
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import sys
import folium
from folium.plugins import HeatMap, MarkerCluster, AntPath
from constants import YEAR, MONTH, DAY, EVENT_TYPE, COUNTRY, LATITUDE, LONGITUDE, REGION, NATURAL_EVENT_ID, \
    LATITUDE_END, LONGITUDE_END, HTMLS_PATH


def print_progress_bar(percentuale, lunghezza_barra=20):
    blocchi_compilati = int(lunghezza_barra * percentuale)
    barra = "[" + "=" * (blocchi_compilati - 1) + ">" + " " * (lunghezza_barra - blocchi_compilati) + "]"
    sys.stdout.write(f"\r{barra} {percentuale * 100:.2f}% completo")
    sys.stdout.flush()
    if percentuale == 1:
        print("")


def base_analysis(df_events):
    print(df_events.glimpse())
    print(df_events.head(20))
    print(df_events.describe())

    # verify dimensions
    print(f"Total number of events: {len(df_events)}")


    # Event type distribution
    print("\nDistribution by event type:")
    print(df_events[EVENT_TYPE].value_counts())


    # Temporal range
    print(f"\nCovered period: {df_events[YEAR].cast(int).min()} - {df_events[YEAR].cast(int).max()}")


    # Geographic coverage
    print(f"\nUnique countries: {df_events[COUNTRY].n_unique()}")
    print(f"\nCoordinate ranges:")
    print(f"  Latitude: {df_events[LATITUDE].min():.2f} to {df_events[LATITUDE].max():.2f}")
    print(f"  Longitude: {df_events[LONGITUDE].min():.2f} to {df_events[LONGITUDE].max():.2f}")


def missing_values_analysis(df_events):
    # count missing
    missing = df_events.null_count()
    print("num missing values:\n", missing)

    # missing percentage per column
    missing_pct = df_events.select([
        ((pl.col(col).is_null().sum() / df_events.height) * 100).alias(col)
        for col in df_events.columns
    ])
    print("missing percentages:\n", missing_pct)

    df_missing_viz = df_events.to_pandas()

    plt.figure(figsize=(12, 6))
    sns.heatmap(df_missing_viz.isnull(), cbar=True, yticklabels=False, cmap='viridis')
    plt.title('Pattern dei Missing Values')
    plt.tight_layout()
    plt.show()

def temporal_distribution(df_events):
    # Convert date to datetime if necessary
    df_events = df_events.with_columns([
        pl.col(YEAR),
        pl.col(MONTH),
        pl.col(DAY)
    ])

    # Events per year
    events_by_year = df_events.group_by([YEAR, EVENT_TYPE]).agg([
        pl.len().alias('count')
    ])

    fig = px.line(events_by_year.to_pandas(), x=YEAR, y='count', color=EVENT_TYPE,
                  title='Event Frequency per Year',
                  labels={'count': 'Number of Events', YEAR: 'Year'})
    fig.show()

    # Monthly heatmap
    # df_events_w_month = df_events.filter(pl.col(MONTH).is_not_null())
    # df_events_w_month = df_events_w_month.with_columns([
    #     pl.col(YEAR).cast(pl.Int32),
    #     pl.col(MONTH).cast(pl.Int32)
    # ])
    #
    # month_events = df_events_w_month.group_by([YEAR, MONTH, EVENT_TYPE]).agg([
    #     pl.len().alias('count')
    # ])
    #
    # for event_type in df_events[EVENT_TYPE].unique().to_list():
    #     subset = month_events.filter(pl.col(EVENT_TYPE) == event_type)
    #
    #     pivot = subset.pivot(
    #         values='count',
    #         index=MONTH,
    #         on=YEAR,
    #         aggregate_function='first'
    #     ).sort(MONTH)
    #
    #
    #     pivot_pandas = pivot.to_pandas()
    #     pivot_pandas = pivot_pandas.set_index(MONTH)
    #     pivot_pandas = pivot_pandas.reindex(
    #         sorted(pivot_pandas.columns),
    #         axis=1
    #     )
    #     pivot_pandas.columns = pivot_pandas.columns.astype(str)  # forza colonne = anni stringa
    #
    #     plt.figure(figsize=(15, 6))
    #     sns.heatmap(pivot_pandas, cmap='YlOrRd', annot=False, fmt='g')
    #     plt.title(f'Monthly Heatmap - {event_type}')
    #     plt.xlabel('Year')
    #     plt.ylabel('Month')
    #     plt.xticks(rotation=90)
    #     plt.yticks(np.arange(1, 13),  # mesi 1–12
    #                ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
    #                rotation=0)
    #     plt.show()
    #
    # # Seasonality (aggregated by month)
    # seasonality = df_events.filter(pl.col(MONTH).is_not_null()).group_by([MONTH, EVENT_TYPE]) \
    #     .agg([
    #         pl.len().alias('count')
    #     ])
    #
    # fig = px.bar(seasonality.to_pandas(), x=MONTH, y='count', color=EVENT_TYPE,
    #              title='Seasonal Distribution of Events',
    #              labels={'count': 'Number of Events', MONTH: 'Month'},
    #              barmode='group')
    # fig.show()

def geographic_distribution(df_events, df_tornadoes):
    # Events by country (top 20)
    top_countries = (
        df_events
        .group_by(COUNTRY)
        .agg(pl.len().alias('count'))
        .sort('count', descending=True)
        .head(20)
        .sort('count')
    )

    plt.figure(figsize=(12, 6))
    plt.barh(top_countries[COUNTRY].to_list(), top_countries['count'].to_list())
    plt.title('Top 20 Countries by Number of Events')
    plt.xlabel('Number of Events')
    plt.ylabel('Country')
    plt.tight_layout()
    plt.show()

    # By continent (if you have this information)

    events_by_region = df_events.filter(pl.col(REGION).is_not_null()) \
        .group_by([REGION, EVENT_TYPE]) \
        .agg(pl.len().alias('count'))


    fig = px.sunburst(events_by_region, path=[REGION, EVENT_TYPE], values='count',
                      title='Event Distribution by Continent and Type')
    fig.show()

    events_by_region = df_events.filter(pl.col(EVENT_TYPE) != 'tornado') \
        .filter(pl.col(REGION).is_not_null()) \
        .group_by([REGION, EVENT_TYPE]) \
        .agg(pl.len().alias('count'))

    fig = px.sunburst(events_by_region, path=[REGION, EVENT_TYPE], values='count',
                      title='Event Distribution by Continent and Type')
    fig.show()


    df_locations = df_events.filter(pl.col(LATITUDE).is_not_null() & pl.col(LONGITUDE).is_not_null()) \
        .join(df_tornadoes, how="left", left_on="id", right_on=NATURAL_EVENT_ID)
    types = [] #df_events[EVENT_TYPE].unique().to_list() + ["all"]
    for event_type in types:
        mappa = folium.Map(location=[20, 0], zoom_start=2)
        marker_cluster = MarkerCluster().add_to(mappa)
        subset = df_locations if event_type == "all" else df_locations.filter(pl.col(EVENT_TYPE) == event_type)
        for i, row in enumerate(subset.iter_rows(named=True)):
            print_progress_bar(i/len(subset))
            event_color = 'orange' if row[EVENT_TYPE] == 'earthquake' else \
                'blue' if row[EVENT_TYPE] == 'tsunami' else \
                    'red' if row[EVENT_TYPE] == 'eruption' else \
                        'purple' if row[EVENT_TYPE] == 'tornado' else 'green'
            CircleMarker(
                location=[row[LATITUDE], row[LONGITUDE]],
                radius=3,
                color=event_color,
                fill=True,
                popup="event location"
            ).add_to(marker_cluster)

            if row.get(LATITUDE) and row.get(LONGITUDE) and \
                    row.get(LATITUDE_END) and row.get(LONGITUDE_END):
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

                CircleMarker(
                    location=[row[LATITUDE_END], row[LONGITUDE_END]],
                    radius=3,
                    color=event_color,
                    fill=True,
                    popup="End location"
                ).add_to(marker_cluster)

        filename = f'{HTMLS_PATH}events_map_{event_type}.html'
        print(f"\nsaving {filename}...")
        mappa.save(filename)
        print(f"✓ {filename} saved!")


    # Geographic heatmap
    heat_data = df_events.filter(pl.col(LATITUDE).is_not_null() & pl.col(LONGITUDE).is_not_null())\
        .select([LATITUDE, LONGITUDE]).to_numpy().tolist()

    heat_map = folium.Map(location=[20, 0], zoom_start=2)
    HeatMap(heat_data, radius=15).add_to(heat_map)
    heat_map.save(HTMLS_PATH + 'events_heatmap.html')

if __name__ == "__main__":
    df_events = engine.get_all_events()
    # base_analysis(df_events)
    # missing_values_analysis(df_events)
    temporal_distribution(df_events)
    # df_tornadoes = engine.get_all_tornado_traces()
    # geographic_distribution(df_events, df_tornadoes)
