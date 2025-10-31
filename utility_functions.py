import datetime

import polars as pl
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go

from flask import jsonify
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.cm as cm

from constants import MULTIPLICATOR_OUTLIERS_ANALYSIS, EVENT_SPECIFIC_COLUMNS, LATITUDE, LONGITUDE, EVENT_TYPE, YEAR, \
    MONTH, DAY
from sql_engine import SqlEngine


def spearman_corr(df: pl.DataFrame) -> pd.DataFrame:
    """Calcola la correlazione di Spearman tra le colonne specificate."""
    ranked_df = df.select([df[col].rank().alias(col) for col in df.columns if df[col].null_count() < df.height and not (df[col] == 0).all()])
    corr_df = ranked_df.to_pandas().corr().fillna(-2)

    return corr_df

def fig_to_json_response(fig, jsonfy = True):
    """
    Converte un oggetto Plotly Figure in JSON compatibile con Plotly.js,
    risolvendo il problema dei dati binari (bdata) per qualsiasi tipo di grafico.

    Args:
        fig: oggetto Plotly Figure (es. px.line, go.Figure, ecc.)
        df: DataFrame Pandas/Polars usato per generare il grafico (opzionale)
        x_col, y_col, color_col, z_col: nomi delle colonne (opzionali, per figure non standard)

    Returns:
        Flask `jsonify` con i dati compatibili per Plotly.js
    """


    fig_json = json.loads(fig.to_json())

    for i, trace in enumerate(fig_json.get("data", [])):
        x =  getattr(fig.data[i], 'x', None)
        y =  getattr(fig.data[i], 'y', None)
        values = getattr(fig.data[i], 'values', None)
        z = getattr(fig.data[i], 'z', None)
        # --- Fix x ---
        if isinstance(trace.get("x", []), dict) and "bdata" in trace["x"]:
            if x is not None:
                trace["x"] = x.tolist()

        # --- Fix y ---
        if isinstance(trace.get("y", []), dict) and "bdata" in trace["y"]:
            if y is not None:
                trace["y"] = y.tolist()

        if isinstance(trace.get("values", []), dict) and "bdata" in trace["values"]:
            if values is not None:
                trace["values"] = values.tolist()

        # --- Fix z (es. heatmap) ---
        if isinstance(trace.get("z", []), dict) and "bdata" in trace["z"]:
            if z is not None:
                trace["z"] = z.tolist()
            elif "zsrc" not in trace:  # fallback: svuota z se non specificato
                trace["z"] = []

    return jsonify(fig_json) if jsonfy else fig_json


def get_outliers_analysis(engine: SqlEngine, column, event_type, return_table=False):
    dataframe = engine.get_from_full_event(event_type, (['ne.*'] + [f"e.{column}"]
                                                        if column in EVENT_SPECIFIC_COLUMNS else []) \
        if return_table else [column], not_null_columns=[column])

    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - MULTIPLICATOR_OUTLIERS_ANALYSIS * IQR
    upper_bound = Q3 + MULTIPLICATOR_OUTLIERS_ANALYSIS * IQR

    df_outliers = dataframe.filter((pl.col(column) < lower_bound) | (pl.col(column) > upper_bound)).with_columns(
        pl.when(pl.col(column) < lower_bound)
        .then(pl.lit('low'))
        .when(pl.col(column) > upper_bound)
        .then(pl.lit('high'))
        .alias('outlier_type')
    )
    stats = {
        'event_type': event_type,
        'column': column,
        'total_elements': dataframe.height,
        'num_outliers': df_outliers.height,
        'perc_outliers': (df_outliers.height / dataframe.height * 100),
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'val_min': dataframe[column].min(),
        'val_max': dataframe[column].max(),
        'mean': dataframe[column].mean(),
        'median': dataframe[column].median()
    }
    return df_outliers if return_table else stats

def prepare_geo_clustering_data(df: pl.DataFrame, n_clusters: int = 6, cell_size: int = 2):
    """
    Esegue il clustering geografico e restituisce i dati pronti per il frontend JS.
    """

    stats = geographical_clustering(df, n_clusters=n_clusters, cell_size=cell_size)
    cluster_profiles = stats['cluster_profiles']
    df_clustered = stats['df_clustered']

    # Palette di colori per i cluster
    colors = [cm.tab10(i / n_clusters) for i in range(n_clusters)]
    hex_colors = [f"#{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}" for r, g, b, _ in colors]

    clusters_data = []
    for i, row in enumerate(cluster_profiles.iter_rows(named=True)):
        dominant_type = max(
            ["earthquake", "tsunami", "eruption", "tornado"],
            key=lambda t: row[f"pct_{t}s"] if f"pct_{t}s" in row else row.get(f"pct_{t}", 0)
        )

        clusters_data.append({
            "name": f"Cluster {int(row['cluster'])}" if "cluster" in row else f"Cluster {i}",
            "color": hex_colors[i % len(hex_colors)],
            "num_events": int(row["total_events"]),
            "num_cells": int(row["n_cells"]),
            "mean_intensity": float(row["mean_intensity"]),
            "std_intensity": float(row["std_intensity"]),
            "total_deaths": int(row["total_deaths"]),
            "mean_deaths": float(row["mean_deaths"]),
            "std_deaths": float(row["std_deaths"]),
            "mean_damage": float(row["mean_damage"]),
            "std_damage": float(row["std_damage"]),
            "total_damage": float(row["total_damage"]),
            "mean_houses_destroyed": float(row["mean_houses_destroyed"]),
            "std_houses_destroyed": float(row["std_houses_destroyed"]),
            "total_houses_destroyed": int(row["total_houses_destroyed"]),
            "pct_earthquakes": float(row["pct_earthquakes"]),
            "pct_tsunami": float(row["pct_tsunami"]),
            "pct_eruptions": float(row["pct_eruptions"]),
            "pct_tornadoes": float(row["pct_tornadoes"]),
            "dominant_type": dominant_type,
            "centroid": {"lat": float(row["lat_centroid"]), "lon": float(row["lon_centroid"])}
        })


    # === 2. Heatmap ===
    feature_cols = [
        "mean_intensity", "mean_deaths", "mean_damage", "mean_houses_destroyed",
        "pct_earthquakes", "pct_tsunami", "pct_eruptions", "pct_tornadoes"
    ]
    z = np.array([
        [row[f] for f in feature_cols]
        for row in cluster_profiles.iter_rows(named=True)
    ])

    # Normalizza tra 0 e 1
    scaler = MinMaxScaler()
    z = scaler.fit_transform(z)
    heatmap_fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=feature_cols,
            y=[f"Cluster {i}" for i in range(n_clusters)],
            colorscale="YlOrRd"
        )
    )
    heatmap_fig.update_layout(
        title="Feature Intensity per Cluster",
        xaxis_title="Feature",
        yaxis_title="Cluster"
    )

    # === 3. Istogramma eventi per cluster ===
    events_fig = px.bar(
        cluster_profiles.to_pandas(),
        x=[f"Cluster {i}" for i in range(n_clusters)],
        y="total_events",
        color_discrete_sequence=colors,
        labels={"x": "Cluster", "y": "Number of Events"},
        title="Number of Events per Cluster"
    )

    # === 4. Istogramma distribuzione tipi di evento ===
    df_types = pl.DataFrame({
        "cluster": [f"Cluster {i}" for i in range(n_clusters)],
        "Earthquakes": [row["pct_earthquakes"] * row["total_events"] for row in
                        cluster_profiles.iter_rows(named=True)],
        "Tsunami": [row["pct_tsunami"] * row["total_events"] for row in cluster_profiles.iter_rows(named=True)],
        "Eruptions": [row["pct_eruptions"] * row["total_events"] for row in cluster_profiles.iter_rows(named=True)],
        "Tornado": [row["pct_tornadoes"] * row["total_events"] for row in cluster_profiles.iter_rows(named=True)],
    })
    df_types_pd = df_types.to_pandas().melt(id_vars="cluster", var_name="type", value_name="count")
    type_fig = px.bar(
        df_types_pd,
        x="cluster",
        y="count",
        color="type",
        barmode="group",
        title="Event Types Distribution per Cluster"
    )

    # Dati riassuntivi
    summary = {
        "total_events": int(df.height),
        "num_clusters": n_clusters,
        "silhouette_score": float(stats["silhouette_score"])
    }


    plots = {
        "heatmap": fig_to_json_response(heatmap_fig, False),
        "events_per_cluster": fig_to_json_response(events_fig, False),
        "events_per_type": fig_to_json_response(type_fig, False)
    }

    return {
        "summary": summary,
        "clusters": clusters_data,
        "plots": plots,
        "df_clustered": df_clustered
    }


def days_since_year0_expr(year_col: str, month_col: str, day_col: str) -> pl.Expr:
    """
    Restituisce un'espressione Polars che calcola il numero di giorni
    dall'anno 0, anche per anni negativi (prima di Cristo).
    Usa un calendario gregoriano semplificato, ignorando le correzioni storiche.
    """
    # giorni cumulativi fino all'inizio di ogni mese (anno non bisestile)
    cum_days = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]

    # Numero giorni fino all’anno precedente
    days_year = (
        pl.col(year_col) * 365
        + pl.floor(pl.col(year_col) / 4)        # aggiungi 1 giorno ogni 4 anni (bisestili)
        - pl.floor(pl.col(year_col) / 100)      # togli 1 ogni 100
        + pl.floor(pl.col(year_col) / 400)      # aggiungi 1 ogni 400
    )

    # Aggiungi giorni dell’anno corrente
    days_month = pl.when(pl.col(month_col) > 0).then(
        pl.lit(cum_days)[pl.col(month_col) - 1]
    ).otherwise(0)

    # Correzione per febbraio bisestile
    is_leap = (
        ((pl.col(year_col) % 4 == 0) & (pl.col(year_col) % 100 != 0))
        | (pl.col(year_col) % 400 == 0)
    )
    leap_correction = pl.when((is_leap) & (pl.col(month_col) > 2)).then(1).otherwise(0)

    return days_year + days_month + pl.col(day_col) - 1 + leap_correction


def geographical_clustering(df: pl.DataFrame,n_clusters: int = 6, cell_size: int = 2):
    """
    Clustering che considera:
    - Posizione geografica
    - Frequenza eventi nella zona
    - Intensità media
    - Distribuzione tipi di eventi
    """

    # Feature engineering: crea griglia geografica
    # Dividi il mondo in celle 2°x2° (circa 220km x 220km)
    df_grid = df.with_columns([
        (pl.col(LATITUDE) // cell_size * cell_size).alias('lat_grid'),
        (pl.col(LONGITUDE) // cell_size * cell_size).alias('lon_grid')
    ])

    # Aggrega per cella
    grid_features = df_grid.group_by(['lat_grid', 'lon_grid']).agg([
        pl.count().alias('frequency'),
        pl.col('intensity').mean().alias('mean_intensity'),
        pl.col('intensity').std().alias('std_intensity'),
        pl.col('deaths').mean().alias('mean_deaths'),
        pl.col('deaths').std().alias('std_deaths'),
        pl.col('deaths').sum().alias('total_deaths'),
        pl.col('damagemillionsdollars').mean().alias('mean_damage'),
        pl.col('damagemillionsdollars').std().alias('std_damage'),
        pl.col('damagemillionsdollars').sum().alias('total_damage'),
        pl.col('housesdestroyed').mean().alias('mean_houses_destroyed'),
        pl.col('housesdestroyed').std().alias('std_houses_destroyed'),
        pl.col('housesdestroyed').sum().alias('total_houses_destroyed'),

        # Percentuale di ogni tipo di evento
        (pl.col(EVENT_TYPE).filter(pl.col(EVENT_TYPE) == 'earthquake').count() / pl.count()).alias(
            'pct_earthquakes'),
        (pl.col(EVENT_TYPE).filter(pl.col(EVENT_TYPE) == 'tsunami').count() / pl.count()).alias('pct_tsunami'),
        (pl.col(EVENT_TYPE).filter(pl.col(EVENT_TYPE) == 'eruption').count() / pl.count()).alias('pct_eruptions'),
        (pl.col(EVENT_TYPE).filter(pl.col(EVENT_TYPE) == 'tornado').count() / pl.count()).alias('pct_tornadoes'),
    ])

    # Prepara features per clustering
    feature_cols = ['lat_grid', 'lon_grid', 'frequency', 'mean_intensity', 'std_intensity',
                    'mean_deaths', 'std_deaths', 'total_deaths',
                    'mean_damage', 'std_damage', 'total_damage', 'mean_houses_destroyed', 'std_houses_destroyed',
                    'total_houses_destroyed', 'pct_earthquakes', 'pct_tsunami', 'pct_eruptions', 'pct_tornadoes']

    X = grid_features.select(feature_cols).fill_null(0).to_numpy()

    # Normalizza features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # Aggiungi cluster al grid
    grid_features = grid_features.with_columns([
        pl.Series('cluster', labels)
    ])

    # Associa cluster agli eventi originali
    df_clustered = df_grid.join(
        grid_features.select(['lat_grid', 'lon_grid', 'cluster']),
        on=['lat_grid', 'lon_grid'],
        how='left'
    )

    # Statistiche per cluster
    cluster_profiles = grid_features.group_by('cluster').agg([
        pl.count().alias('n_cells'),
        pl.col('frequency').sum().alias('total_events'),
        pl.col('frequency').mean().alias('mean_frequency'),
        pl.col('mean_intensity').mean().alias('mean_intensity'),
        pl.col('std_intensity').mean().alias('std_intensity'),
        pl.col('mean_deaths').mean().alias('mean_deaths'),
        pl.col('std_deaths').mean().alias('std_deaths'),
        pl.col('total_deaths').sum().alias('total_deaths'),
        pl.col('mean_damage').mean().alias('mean_damage'),
        pl.col('std_damage').mean().alias('std_damage'),
        pl.col('total_damage').sum().alias('total_damage'),
        pl.col('mean_houses_destroyed').mean().alias('mean_houses_destroyed'),
        pl.col('std_houses_destroyed').mean().alias('std_houses_destroyed'),
        pl.col('total_houses_destroyed').sum().alias('total_houses_destroyed'),
        pl.col('pct_earthquakes').mean().alias('pct_earthquakes'),
        pl.col('pct_tsunami').mean().alias('pct_tsunami'),
        pl.col('pct_eruptions').mean().alias('pct_eruptions'),
        pl.col('pct_tornadoes').mean().alias('pct_tornadoes'),
        pl.col('lat_grid').mean().alias('lat_centroid'),
        pl.col('lon_grid').mean().alias('lon_centroid')
    ]).sort('total_events', descending=True)

    # Identifica tipo dominante per cluster
    cluster_profiles = cluster_profiles.with_columns([
        pl.max_horizontal(['pct_earthquakes', 'pct_tsunami', 'pct_eruptions', 'pct_tornadoes']).alias('pct_max')
    ])

    # Metriche
    silhouette = silhouette_score(X_scaled, labels)
    davies_bouldin = davies_bouldin_score(X_scaled, labels)

    stats = {
        'features_used': feature_cols,
        'silhouette_score': silhouette,
        'davies_bouldin_score': davies_bouldin,
        'cluster_profiles': cluster_profiles,
        'df_clustered': df_clustered
    }

    return stats


# ============================================
# 3. CLUSTERING SPAZIO-TEMPORALE (ST-DBSCAN)
# ============================================

def geospatial_temporal_clustering(df: pl.DataFrame,
                               eps_spatial: float = 2.0,  # gradi (circa 220 km)
                               eps_temporal: int = 30,  # giorni
                               min_samples: int = 5):
    """
    Trova cluster di eventi vicini sia nello spazio che nel tempo
    Utile per identificare:
    - Sequenze sismiche (terremoti + aftershocks)
    - Zone di attività vulcanica temporanea
    - Pattern spazio-temporali

    Args:
        eps_spatial: distanza massima in gradi
        eps_temporal: distanza massima in giorni
        min_samples: minimo eventi per formare un cluster
    """

    # Converti date in numerico (giorni da epoch)
    df_prep = df.with_columns([
        days_since_year0_expr(YEAR, MONTH, DAY).alias('days_epoch')
    ])

    # Prepara features: [lat, lon, tempo_normalizzato]
    # Normalizziamo il tempo per pesarlo correttamente rispetto allo spazio
    time_scale = eps_spatial / eps_temporal

    coords_time = np.column_stack([
        df_prep[LATITUDE].to_numpy(),
        df_prep[LONGITUDE].to_numpy(),
        df_prep['days_epoch'].to_numpy() * time_scale
    ])

    # ST-DBSCAN (usiamo DBSCAN standard su features spazio-temporali)
    dbscan = DBSCAN(eps=eps_spatial, min_samples=min_samples, metric='euclidean')
    labels = dbscan.fit_predict(coords_time)

    # Aggiungi cluster
    df_clustered = df_prep.with_columns([
        pl.Series('cluster_st', labels)
    ])

    # Statistiche per cluster (escludi noise = -1)
    valid_clusters = df_clustered.filter(pl.col('cluster_st') >= 0)

    if valid_clusters.height > 0:
        cluster_stats = valid_clusters.group_by('cluster_st').agg([
            pl.count().alias('n_events'),
            pl.col('days_epoch').min().alias('starting_days_from_year_0'),
            pl.col('days_epoch').max().alias('ending_days_from_year_0'),
            pl.col(LATITUDE).mean().alias('lat_centroid'),
            pl.col(LONGITUDE).mean().alias('lon_centroid'),
            pl.col(EVENT_TYPE).value_counts().alias('different_types_events'),
            pl.col('intensity').mean().alias('mean_intensity'),
            pl.col('intensity').max().alias('max_intensity')
        ]).sort('n_events', descending=True)

        # Calcola durata cluster
        cluster_stats = cluster_stats.with_columns([
            (pl.col('ending_days_from_year_0') - pl.col('starting_days_from_year_0')).alias(
                'days_period_duration')
        ])
    else:
        cluster_stats = None

    n_clusters = len(np.unique(labels[labels >= 0]))
    n_noise = (labels == -1).sum()


    stats = {
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'pct_clustered': ((df.height - n_noise) / df.height * 100) if df.height > 0 else 0,
        'cluster_stats': cluster_stats,
        'df_clustered': df_clustered
    }

    return stats


# # ============================================
# # 4. CLUSTERING DI PAESI (ANALISI RISCHIO)
# # ============================================
#
# def clustering_countries(df: pl.DataFrame,
#                      n_clusters: int = 5):
#     """
#     Raggruppa paesi con profili di rischio simili
#
#     Features:
#     - Frequenza totale eventi
#     - Distribuzione per tipo
#     - Intensità media
#     - Variabilità intensità
#     """
#
#     # Aggrega per paese
#     country_features = df.group_by('country').agg([
#         pl.count().alias('n_events'),
#         pl.col('intensity').mean().alias('mean_intensity'),
#         pl.col('intensity').std().alias('std_intensity'),
#         pl.col('intensity').max().alias('max_intensity'),
#         pl.col('deaths').mean().alias('mean_deaths'),
#         pl.col('deaths').std().alias('std_deaths'),
#         pl.col('deaths').max().alias('max_deaths'),
#         pl.col('damagemillionsdollars').mean().alias('mean_damage'),
#         pl.col('damagemillionsdollars').std().alias('std_damage'),
#         pl.col('damagemillionsdollars').max().alias('max_damage'),
#         pl.col('housesdestroyed').mean().alias('mean_houses_destroyed'),
#         pl.col('housesdestroyed').std().alias('std_houses_destroyed'),
#         pl.col('housesdestroyed').max().alias('max_houses_destroyed'),
#
#         # Conteggi per tipo
#         (pl.col(EVENT_TYPE).filter(pl.col(EVENT_TYPE) == 'earthquake').count()).alias('n_earthquakes'),
#         (pl.col(EVENT_TYPE).filter(pl.col(EVENT_TYPE) == 'tsunami').count()).alias('n_tsunami'),
#         (pl.col(EVENT_TYPE).filter(pl.col(EVENT_TYPE) == 'eruption').count()).alias('n_eruptions'),
#         (pl.col(EVENT_TYPE).filter(pl.col(EVENT_TYPE) == 'tornado').count()).alias('n_tornado'),
#
#         # Percentuali
#         (pl.col(EVENT_TYPE).filter(pl.col(EVENT_TYPE) == 'earthquake').count() / pl.count()).alias(
#             'pct_earthquakes'),
#         (pl.col(EVENT_TYPE).filter(pl.col(EVENT_TYPE) == 'tsunami').count() / pl.count()).alias('pct_tsunami'),
#         (pl.col(EVENT_TYPE).filter(pl.col(EVENT_TYPE) == 'eruption').count() / pl.count()).alias('pct_eruptions'),
#         (pl.col(EVENT_TYPE).filter(pl.col(EVENT_TYPE) == 'tornado').count() / pl.count()).alias('pct_tornadoes'),
#
#         # Diversità eventi (quanti tipi diversi?)
#         pl.col(EVENT_TYPE).n_unique().alias('n_different_event_types')
#     ])
#
#     # Features per clustering
#     feature_cols = ['n_events', 'mean_intensity', 'std_intensity', 'max_intensity',
#                     'mean_deaths', 'std_deaths', 'max_deaths',
#                     'mean_damage', 'std_damage', 'max_damage',
#                     'mean_houses_destroyed', 'std_houses_destroyed', 'max_houses_destroyed',
#                     'n_earthquakes', 'n_tsunami', 'n_eruptions', 'n_tornado',
#                     'pct_earthquakes', 'pct_tsunami', 'pct_eruptions', 'pct_tornadoes',
#                     'n_different_event_types']
#
#     X = country_features.select(feature_cols).fill_null(0).to_numpy()
#
#     # Normalizza
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#
#     # K-means
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
#     labels = kmeans.fit_predict(X_scaled)
#
#     # Aggiungi cluster
#     country_features = country_features.with_columns([
#         pl.Series('cluster', labels)
#     ])
#
#     # Profilo cluster
#     cluster_profiles = country_features.group_by('cluster').agg([
#         pl.count().alias('n_countries'),
#         pl.col('n_events').sum().alias('total_events_cluster'),
#         pl.col('n_events').mean().alias('mean_n_events_cluster'),
#         pl.col('mean_intensity').mean().alias('mean_intensity_cluster'),
#         pl.col('std_intensity').mean().alias('std_intensity_cluster'),
#         pl.col('max_intensity').mean().alias('max_intensity_cluster'),
#         pl.col('mean_deaths').mean().alias('mean_deaths_cluster'),
#         pl.col('std_deaths').mean().alias('std_deaths_cluster'),
#         pl.col('max_deaths').mean().alias('max_deaths_cluster'),
#         pl.col('mean_damage').mean().alias('mean_damage_cluster'),
#         pl.col('std_damage').mean().alias('std_damage_cluster'),
#         pl.col('max_damage').mean().alias('max_damage_cluster'),
#         pl.col('mean_houses_destroyed').mean().alias('mean_houses_destroyed_cluster'),
#         pl.col('std_houses_destroyed').mean().alias('std_houses_destroyed_cluster'),
#         pl.col('max_houses_destroyed').mean().alias('max_houses_destroyed_cluster'),
#         pl.col('pct_earthquakes').mean().alias('mean_pct_earthquakes'),
#         pl.col('pct_tsunami').mean().alias('mean_pct_tsunami'),
#         pl.col('pct_eruptions').mean().alias('mean_pct_eruptions'),
#         pl.col('pct_tornadoes').mean().alias('mean_pct_tornado'),
#         pl.col('n_different_event_types').mean().alias('mean_n_different_event_types')
#     ]).sort('total_events_cluster', descending=True)
#
#     # Identifica paesi esempio per ogni cluster (top 3 più rappresentativi)
#     cluster_examples = []
#     for cluster_id in range(n_clusters):
#         top_countries = (country_features
#                      .filter(pl.col('cluster') == cluster_id)
#                      .sort('n_events', descending=True)
#                      .head(3)
#                      .select(['country', 'n_events']))
#         cluster_examples.append({
#             'cluster': cluster_id,
#             'example_countries': top_countries['country'].to_list()
#         })
#
#     # Metriche
#     silhouette = silhouette_score(X_scaled, labels)
#     davies_bouldin = davies_bouldin_score(X_scaled, labels)
#
#     stats = {
#         'features_used': feature_cols,
#         'silhouette_score': silhouette,
#         'davies_bouldin_score': davies_bouldin,
#         'cluster_profiles': cluster_profiles,
#         'examples_per_cluster': cluster_examples,
#         'country_features': country_features
#     }
#
#     return stats
