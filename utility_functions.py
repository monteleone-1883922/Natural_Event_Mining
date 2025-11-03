import datetime

import polars as pl
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial.distance import cdist

from flask import jsonify
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
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


def create_time_index(df_polars):
    """
    Crea un indice temporale numerico progressivo dai campi anno, mese, giorno.
    Gestisce anche anni negativi (a.C.)
    Input: Polars DataFrame
    Output: Polars DataFrame con time_index e date_label
    """
    # Assicurati che le colonne siano numeriche e ordina
    df = df_polars.filter(
        (pl.col(YEAR).is_not_null()) &
        (pl.col(MONTH).is_not_null()) &
        (pl.col(DAY).is_not_null())
    ).with_columns([
        pl.col(YEAR).cast(pl.Int32),
        pl.col(MONTH).cast(pl.Int32),
        pl.col(DAY).cast(pl.Int32)
    ]).sort([YEAR, MONTH, DAY])

    # Crea indice progressivo
    df = df.with_row_count('time_index')

    # Crea etichetta leggibile per i grafici
    df = df.with_columns([
        pl.when(pl.col(YEAR) < 0)
        .then(
            pl.format(
                "{} a.C.-{}-{}",
                pl.col(YEAR).abs(),
                pl.col(MONTH).cast(pl.Utf8).str.zfill(2),
                pl.col(DAY).cast(pl.Utf8).str.zfill(2)
            )
        )
        .otherwise(
            pl.format(
                "{} d.C.-{}-{}",
                pl.col(YEAR),
                pl.col(MONTH).cast(pl.Utf8).str.zfill(2),
                pl.col(DAY).cast(pl.Utf8).str.zfill(2)
            )
        )
        .alias('date_label')
    ])

    return df

def clean_nan_for_json(value):
    """Converte NaN in None (null in JSON)"""
    if isinstance(value, (list, np.ndarray, pd.Series)):
        return [None if pd.isna(v) else v for v in value]
    return None if pd.isna(value) else value

def get_outliers_analysis(engine: SqlEngine, column, event_type, return_table=False):
    not_null_columns = [column, LATITUDE, LONGITUDE] if return_table else [column]
    dataframe = engine.get_from_full_event(event_type, (['ne.*'] + [f"e.{column}"]
                                                        if column in EVENT_SPECIFIC_COLUMNS else []) \
        if return_table else [column], not_null_columns=not_null_columns)
    if event_type == 'eruption':
        dataframe = dataframe.with_columns(
            pl.col("vei").cast(pl.Float64)
        )
    elif event_type == 'tornado':
        dataframe = dataframe.with_columns(
            pl.col("f_scale").cast(pl.Float64)
        )

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
        + (pl.col(year_col) / 4).floor()        # aggiungi 1 giorno ogni 4 anni (bisestili)
        - (pl.col(year_col) / 100).floor()     # togli 1 ogni 100
        + (pl.col(year_col) / 400).floor()    # aggiungi 1 ogni 400
    )

    # Aggiungi giorni dell’anno corrente
    days_month = (
        pl.when(pl.col(month_col) == 1).then(cum_days[0])
        .when(pl.col(month_col) == 2).then(cum_days[1])
        .when(pl.col(month_col) == 3).then(cum_days[2])
        .when(pl.col(month_col) == 4).then(cum_days[3])
        .when(pl.col(month_col) == 5).then(cum_days[4])
        .when(pl.col(month_col) == 6).then(cum_days[5])
        .when(pl.col(month_col) == 7).then(cum_days[6])
        .when(pl.col(month_col) == 8).then(cum_days[7])
        .when(pl.col(month_col) == 9).then(cum_days[8])
        .when(pl.col(month_col) == 10).then(cum_days[9])
        .when(pl.col(month_col) == 11).then(cum_days[10])
        .when(pl.col(month_col) == 12).then(cum_days[11])
        .otherwise(0)
    )

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
def calculate_dunn_index(coords, labels):
    """
    Calcola il Dunn Index per valutare la qualità del clustering.

    Dunn Index = min_inter_cluster_distance / max_intra_cluster_distance

    Valori più alti = clustering migliore
    - Massimizza la distanza tra cluster (separazione)
    - Minimizza la distanza dentro i cluster (coesione)

    Args:
        coords: array di coordinate (n_samples, n_features)
        labels: array di label dei cluster (n_samples,)

    Returns:
        dunn_index: float, più alto è meglio
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters < 2:
        return None

    # 1. Calcola distanza MINIMA tra cluster diversi (inter-cluster)
    min_inter_cluster_dist = np.inf

    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            # Punti del cluster i e j
            cluster_i_points = coords[labels == unique_labels[i]]
            cluster_j_points = coords[labels == unique_labels[j]]

            # Distanza minima tra tutti i punti dei due cluster
            distances = cdist(cluster_i_points, cluster_j_points, metric='euclidean')
            min_dist = np.min(distances)

            if min_dist < min_inter_cluster_dist:
                min_inter_cluster_dist = min_dist

    # 2. Calcola distanza MASSIMA dentro ogni cluster (intra-cluster)
    max_intra_cluster_dist = 0

    for label in unique_labels:
        cluster_points = coords[labels == label]

        if len(cluster_points) < 2:
            continue

        # Distanza massima tra punti dello stesso cluster (diametro)
        distances = cdist(cluster_points, cluster_points, metric='euclidean')
        max_dist = np.max(distances)

        if max_dist > max_intra_cluster_dist:
            max_intra_cluster_dist = max_dist

    # 3. Dunn Index
    if max_intra_cluster_dist == 0:
        return None  # Evita divisione per zero

    dunn_index = min_inter_cluster_dist / max_intra_cluster_dist

    return dunn_index

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
    mask_no_noise = labels >= 0
    coords_no_noise = coords_time[mask_no_noise]
    labels_no_noise = labels[mask_no_noise]
    silhouette_value = silhouette_score(
        coords_no_noise,
        labels_no_noise,
        metric='euclidean'
    )
    # dunn_index = calculate_dunn_index(coords_no_noise, labels_no_noise)

    stats = {
        'n_clusters': n_clusters,
        'n_noise': int(n_noise),
        'silhouette_value': silhouette_value,
        # 'dunn_index': dunn_index,
        'pct_clustered': float(((df.height - n_noise) / df.height * 100) if df.height > 0 else 0),
        'cluster_stats': cluster_stats,
        'df_clustered': df_clustered
    }

    return stats


def prepare_spatiotemporal_payload(df ,
                                   eps_spatial: float,
                                   eps_temporal: int,
                                   min_samples: int):
    """
    Prepara il payload JSON per la visualizzazione del clustering spazio-temporale.

    Args:
        stats: dizionario ritornato da geospatial_temporal_clustering
        eps_spatial: parametro epsilon spaziale usato
        eps_temporal: parametro epsilon temporale usato
        min_samples: parametro min_samples usato

    Returns:
        Dictionary con summary, top_clusters e plots
    """
    stats = geospatial_temporal_clustering(df, eps_spatial, eps_temporal, min_samples)
    df_clustered = stats['df_clustered']
    cluster_stats = stats['cluster_stats']

    # ==========================================
    # 1. SUMMARY STATISTICS
    # ==========================================
    summary = {
        'n_clusters': stats['n_clusters'],
        'n_noise': stats['n_noise'],
        'silhouette_value': round(stats['silhouette_value'], 2),
        # 'dunn_index': round(stats['dunn_index'], 2),
        'pct_clustered': round(stats['pct_clustered'], 1),
        'total_events': df_clustered.height,
        'eps_spatial': eps_spatial,
        'eps_temporal': eps_temporal,
        'min_samples': min_samples
    }

    # ==========================================
    # 2. TOP CLUSTERS (primi 5 per numero eventi)
    # ==========================================
    top_clusters = []

    if cluster_stats is not None and cluster_stats.height > 0:
        # Prendi i top 5 cluster
        top_5 = cluster_stats.head(5)

        for row in top_5.iter_rows(named=True):
            # Estrai tipo dominante da different_types_events
            types_list = row.get('different_types_events', [])
            dominant_type = None
            if types_list and len(types_list) > 0:
                # types_list è una lista di struct con campi 'event_type' e 'count'
                # Ordina per count e prendi il primo
                sorted_types = sorted(types_list, key=lambda x: x.get('count', 0), reverse=True)
                if sorted_types:
                    dominant_type = sorted_types[0].get(EVENT_TYPE, 'Unknown')

            cluster_info = {
                'cluster_id': int(row['cluster_st']),
                'n_events': int(row['n_events']),
                'duration_days': int(row['days_period_duration']),
                'starting_days': int(row['starting_days_from_year_0']),
                'ending_days': int(row['ending_days_from_year_0']),
                'lat_centroid': float(row['lat_centroid']),
                'lon_centroid': float(row['lon_centroid']),
                'mean_intensity': float(row['mean_intensity']),
                'max_intensity': float(row['max_intensity']),
                'dominant_type': dominant_type
            }
            top_clusters.append(cluster_info)

    # ==========================================
    # 3. PLOTS
    # ==========================================
    plots = {}


    # --- 3.2 DURATION DISTRIBUTION (Istogramma durate) ---
    durations = cluster_stats['days_period_duration'].to_list()
    if cluster_stats is not None and cluster_stats.height > 0:
        cluster_stats_sorted_duration = cluster_stats.sort('days_period_duration', descending=True)

        cluster_ids = [f"Cluster {x}" for x in cluster_stats_sorted_duration['cluster_st'].to_list()]


        fig_duration = go.Figure(data=[
            go.Bar(
                x=cluster_ids,
                y=durations,
                marker_color='coral',
                textposition='outside'
            )
        ])


        fig_duration.update_layout(
            title='Cluster Duration Distribution',
            xaxis_title='Duration (days)',
            yaxis_title='Cluster ID',
            showlegend=False,
            height=400,
            xaxis={'categoryorder': 'total descending'}
        )

        plots['duration'] = fig_to_json_response(fig_duration, False)
    else:
        plots['duration'] = {
            'data': [],
            'layout': {'title': 'No clusters to display'}
        }

    # --- 3.3 EVENTS PER CLUSTER (Bar chart) ---
    if cluster_stats is not None and cluster_stats.height > 0:
        # Ordina per numero eventi
        cluster_stats_sorted = cluster_stats.sort('n_events', descending=True)

        cluster_ids = [f"Cluster {x}" for x in cluster_stats_sorted['cluster_st'].to_list()]
        n_events_list = cluster_stats_sorted['n_events'].to_list()

        fig_events = go.Figure(data=[
            go.Bar(
                x=cluster_ids,
                y=n_events_list,
                marker_color='coral',
                text=n_events_list,
                textposition='outside'
            )
        ])

        fig_events.update_layout(
            title='Number of Events per Cluster',
            xaxis_title='Cluster ID',
            yaxis_title='Number of Events',
            showlegend=False,
            height=400,
            xaxis={'categoryorder': 'total descending'}
        )

        plots['events_per_cluster'] = fig_to_json_response(fig_events, False)
    else:
        plots['events_per_cluster'] = {
            'data': [],
            'layout': {'title': 'No clusters to display'}
        }

    # ==========================================
    # 4. ASSEMBLA PAYLOAD FINALE
    # ==========================================
    payload = {
        'summary': summary,
        'top_clusters': top_clusters,
        'plots': plots,
        'status': 'success',
        'df_clustered': df_clustered
    }

    return payload


def calculate_country_statistics(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calcola statistiche aggregate per paese.
    """

    country_stats = df.group_by("country").agg([
        pl.count().alias("total_events"),
        pl.col("deaths").sum().alias("total_deaths"),
        pl.col("deaths").mean().alias("avg_deaths"),
        pl.col("damagemillionsdollars").sum().alias("total_damage"),
        pl.col("damagemillionsdollars").mean().alias("avg_damage"),
        pl.col("housesdestroyed").sum().alias("total_houses"),
        pl.col("housesdestroyed").mean().alias("avg_houses"),
        pl.col("intensity").mean().alias("avg_intensity"),
        pl.col("intensity").max().alias("max_intensity"),
        (
            pl.when((pl.col(YEAR).max() - pl.col(YEAR).min()) > 0)
            .then(pl.count() / (pl.col(YEAR).max() - pl.col(YEAR).min() + 1))
            .otherwise(pl.count()).cast(pl.Float64)
            .alias("frequency")
        ),

        pl.col(YEAR).min().alias("first_year").cast(pl.Int64),
        pl.col(YEAR).max().alias("last_year").cast(pl.Int64),
        (pl.col(EVENT_TYPE).filter(pl.col(EVENT_TYPE) == 'earthquake').count() / pl.count()).alias(
                        'pct_earthquakes'),
                    (pl.col(EVENT_TYPE).filter(pl.col(EVENT_TYPE) == 'tsunami').count() / pl.count()).alias('pct_tsunami'),
                    (pl.col(EVENT_TYPE).filter(pl.col(EVENT_TYPE) == 'eruption').count() / pl.count()).alias('pct_eruptions'),
                    (pl.col(EVENT_TYPE).filter(pl.col(EVENT_TYPE) == 'tornado').count() / pl.count()).alias('pct_tornadoes'),
        pl.col(EVENT_TYPE).n_unique().alias("event_types_count"),
    ]).sort("total_events", descending=True)

    country_stats = country_stats.with_columns([
        (pl.col("last_year") - pl.col("first_year") + 1).alias("years_covered"),
    ]).with_columns([
        (pl.col("total_events") / pl.col("years_covered")).cast(pl.Int64).alias("events_per_year")
    ])

    country_stats = country_stats.fill_null(0)

    return country_stats


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
