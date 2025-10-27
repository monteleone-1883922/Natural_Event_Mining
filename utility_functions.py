
import polars as pl
import pandas as pd
import json

from flask import jsonify

from constants import MULTIPLICATOR_OUTLIERS_ANALYSIS, EVENT_SPECIFIC_COLUMNS
from sql_engine import SqlEngine


def spearman_corr(df: pl.DataFrame) -> pd.DataFrame:
    """Calcola la correlazione di Spearman tra le colonne specificate."""
    ranked_df = df.select([df[col].rank().alias(col) for col in df.columns if df[col].null_count() < df.height and not (df[col] == 0).all()])
    corr_df = ranked_df.to_pandas().corr().fillna(-2)

    return corr_df

def fig_to_json_response(fig):
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

    return jsonify(fig_json)


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
