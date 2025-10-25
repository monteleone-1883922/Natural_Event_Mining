import json

from flask import Flask, render_template, jsonify, send_file
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots
from sql_engine import engine
from constants import YEAR, MONTH, DAY, EVENT_TYPE, COUNTRY, LATITUDE, LONGITUDE, REGION, NATURAL_EVENT_ID, \
    LATITUDE_END, LONGITUDE_END, HTMLS_PATH, TEMPLATES_PATH
app = Flask(__name__)

df_events = engine.get_all_events()

@app.route('/')
def index():
    """Homepage con overview generale"""

    stats = {
        'total_events': len(df_events),
        'years_covered': f"{df_events[YEAR].cast(int).min()} - {df_events[YEAR].cast(int).max()}",
        'countries': df_events[COUNTRY].n_unique(),
        'event_types': df_events[EVENT_TYPE].value_counts().to_dicts()
    }

    return render_template('index.html', stats=stats)


@app.route('/missing_values')
def api_missing_values():
    """Statistiche riepilogative sui missing values"""


    # Calcola missing per ogni colonna
    total_rows = len(df_events)
    column_stats = []

    for col in df_events.columns:
        missing_count = df_events[col].null_count()
        missing_pct = (missing_count / total_rows) * 100

        column_stats.append({
            'column_name': col,
            'missing_count': missing_count,
            'total_rows': total_rows,
            'missing_percentage': missing_pct,
            'completeness': 100 - missing_pct
        })

    # Sort by missing percentage (descending)
    column_stats.sort(key=lambda x: x['missing_percentage'], reverse=True)


    complete_columns = sum(1 for c in column_stats if c['missing_percentage'] == 0)
    partial_columns = sum(1 for c in column_stats if 0 < c['missing_percentage'] < 50)
    critical_columns = sum(1 for c in column_stats if c['missing_percentage'] >= 50)

    # Quality score: media della completeness
    quality_score = sum(c['completeness'] for c in column_stats) / len(column_stats)
    stats = {
        'total_columns': len(df_events.columns),
        'complete_columns': complete_columns,
        'partial_columns': partial_columns,
        'critical_columns': critical_columns,
        'quality_score': quality_score
    }

    return render_template('missing_values.html', stats=stats)

@app.route('/temporal_analysis')
def temporal_analysis():
    return render_template('temporal_analysis.html')


@app.route('/api/missing-values/percentage-chart')
def api_missing_percentage_chart():
    """Grafico percentuali missing per colonna"""

    # Calcola percentuali missing
    total_rows = len(df_events)
    missing_pct = []
    columns = []

    for col in df_events.columns:
        missing_count = df_events[col].null_count()
        pct = (missing_count / total_rows) * 100
        missing_pct.append(pct)
        columns.append(col)

    # Crea DataFrame per sorting
    data = list(zip(columns, missing_pct))
    data.sort(key=lambda x: x[1], reverse=True)
    columns_sorted = [x[0] for x in data]
    missing_sorted = [x[1] for x in data]

    # Colora le barre in base alla severità
    colors = []
    for pct in missing_sorted:
        if pct == 0:
            colors.append('#27ae60')  # Verde
        elif pct < 30:
            colors.append('#3498db')  # Blu
        elif pct < 60:
            colors.append('#f39c12')  # Arancione
        else:
            colors.append('#e74c3c')  # Rosso

    fig = go.Figure(data=[
        go.Bar(
            x=missing_sorted,
            y=columns_sorted,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='white', width=1)
            ),
            text=[f'{v:.1f}%' for v in missing_sorted],
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>' +
                          'Missing: %{x:.2f}%<br>' +
                          '<extra></extra>'
        )
    ])

    fig.update_layout(
        title='Missing Values Percentage by Column',
        xaxis_title='Missing Percentage (%)',
        yaxis_title='Column',
        template='plotly_white',
        height=max(400, len(columns) * 25),
        showlegend=False,
        margin=dict(l=150, r=50, t=80, b=50)
    )

    # Aggiungi linee di riferimento
    fig.add_vline(x=25, line_dash="dash", line_color="orange", opacity=0.5)
    fig.add_vline(x=50, line_dash="dash", line_color="red", opacity=0.5)

    return jsonify(json.loads(fig.to_json()))


@app.route('/api/missing-values/count-chart')
def api_missing_count_chart():
    """Grafico conteggio assoluto missing per colonna"""

    # Calcola conteggi missing
    missing_counts = []
    columns = []

    for col in df_events.columns:
        missing_count = df_events[col].null_count()
        if missing_count > 0:  # Mostra solo colonne con missing
            missing_counts.append(missing_count)
            columns.append(col)

    # Sort
    data = list(zip(columns, missing_counts))
    data.sort(key=lambda x: x[1], reverse=True)
    columns_sorted = [x[0] for x in data]
    counts_sorted = [x[1] for x in data]

    fig = go.Figure(data=[
        go.Bar(
            x=counts_sorted,
            y=columns_sorted,
            orientation='h',
            marker=dict(
                color='#e74c3c',
                line=dict(color='white', width=1)
            ),
            text=[f'{v:,}' for v in counts_sorted],
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>' +
                          'Missing Count: %{x:,}<br>' +
                          '<extra></extra>'
        )
    ])

    fig.update_layout(
        title='Absolute Missing Count by Column',
        xaxis_title='Number of Missing Values',
        yaxis_title='Column',
        template='plotly_white',
        height=max(400, len(columns_sorted) * 25),
        showlegend=False,
        margin=dict(l=150, r=50, t=80, b=50)
    )

    return jsonify(json.loads(fig.to_json()))


@app.route('/api/missing-values/heatmap')
def api_missing_heatmap():
    """Heatmap pattern missing values (sample di righe)"""

    # Prendi un campione di righe (max 100) per visualizzare il pattern
    sample_size = min(500, len(df_events))
    df_sample = df_events.sample(n=sample_size)

    # Crea matrice di booleani (True = missing, False = presente)
    missing_matrix = []
    for col in df_events.columns:
        missing_matrix.append(df_sample[col].is_null().to_list())

    # Converti in numeri (1 = missing, 0 = presente)
    missing_matrix = [[1 if x else 0 for x in row] for row in missing_matrix]

    fig = go.Figure(data=go.Heatmap(
        z=missing_matrix,
        x=list(range(sample_size)),
        y=df_events.columns,
        colorscale=[[0, '#27ae60'], [1, '#e74c3c']],
        showscale=True,
        colorbar=dict(
            title='Status',
            tickvals=[0, 1],
            ticktext=['Present', 'Missing']
        ),
        hovertemplate='Column: %{y}<br>' +
                      'Row: %{x}<br>' +
                      'Status: %{z}<br>' +
                      '<extra></extra>'
    ))

    fig.update_layout(
        title=f'Missing Values Pattern (Sample of {sample_size} rows)',
        xaxis_title='Row Sample',
        yaxis_title='Column',
        template='plotly_white',
        height=max(400, len(df_events.columns) * 20),
        margin=dict(l=150, r=50, t=80, b=50)
    )

    return jsonify(json.loads(fig.to_json()))

@app.route('/api/temporal_analysis/temporal_distribution')
def api_event_temporal_distribution():
    temporal_df = df_events.with_columns([
        pl.col(YEAR),
        pl.col(MONTH),
        pl.col(DAY)
    ])

    # Events per year
    events_by_year = temporal_df.group_by([YEAR, EVENT_TYPE]).agg([
        pl.len().alias('count')
    ]).sort([YEAR, EVENT_TYPE])
    events_by_year = events_by_year.with_columns(pl.col(YEAR).cast(pl.Int64))

    fig_px = px.line(events_by_year.to_pandas(), x=YEAR, y='count', color=EVENT_TYPE,
                  title='Event Frequency per Year',
                  labels={'count': 'Number of Events', YEAR: 'Year'})
    fig_px.update_layout(
        template='plotly_white',
        title_x=0.5,
        xaxis=dict(title='Year'),
        yaxis=dict(title='Number of Events'),
        legend=dict(title='Event Type'),
        autosize=True
    )
    fig_px.show()
    fi = fig_px.to_json()
    with open("trial.json", "w") as f:
        json.dump(fi, f)


    return jsonify(fig_px.to_json())


# @app.route('/api/my-endpoint')
# def my_chart():
#
#     # Elabora dati
#     data = df_events.group_by('column').agg([...])
#
#     # Crea grafico Plotly (GIÀ INTERATTIVO!)
#     fig = px.bar(data.to_pandas(), x='x', y='y')
#
#     return jsonify(json.loads(fig.to_json()))


if __name__ == '__main__':
    app.run(debug=True)