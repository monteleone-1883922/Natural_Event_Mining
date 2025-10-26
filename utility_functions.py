
import polars as pl
import pandas as pd
import json

from flask import jsonify


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