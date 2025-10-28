import json
import os

import plotly
from flask import Flask, render_template, jsonify, send_file
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from utility_functions import *
from markupsafe import Markup

from plotly.subplots import make_subplots
from sql_engine import engine
from constants import YEAR, MONTH, DAY, EVENT_TYPE, COUNTRY, GEOGRAPHIC_SECTIONS, LONGITUDE, REGION, NATURAL_EVENT_ID, \
    LATITUDE_END, LONGITUDE_END,  TEMPLATES_PATH, EVENT_INTENSITY, INTENSITY_LABELS, INTENSITY_SECTIONS, \
    MAPS_PATH
from subprocessing import start_map_generation, generate_map_outliers

app = Flask(__name__)

df_events = None

outliers_cache = {
    "df" : pl.DataFrame({}),
    "event_type": "",
    "column": ""
}

def get_df_event(event_type):
    if event_type == 'earthquake':
        return engine.get_all_earthquakes()
    elif event_type == 'tornado':
        return engine.get_all_tornado_traces()
    elif event_type == 'eruption':
        return engine.get_all_eruptions()
    elif event_type == 'tsunami':
        return engine.get_all_tsunamis()



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
    event_types = df_events.select(EVENT_TYPE).unique().to_series().to_list()
    return render_template('temporal_analysis.html', event_types=event_types)

@app.route('/geographic_analysis')
def geographic_analysis():

    return render_template('geographic_analysis_home.html', geographic_sections=GEOGRAPHIC_SECTIONS)

@app.route('/intensity_analysis')
def intensity_analysis():
    return render_template('intensity_analysis.html', intensity_sections=INTENSITY_SECTIONS)

@app.route('/correlation_analysis')
def correlation_analysis():
    return render_template('correlation_analysis.html')

@app.route('/outlier_analysis')
def outliers_analysis():
    return render_template('outlier_analysis.html')


@app.route("/geographic_analysis/maps/<string:map>")
def maps_page(map):
    return render_template(f'{MAPS_PATH}events_maps/events_map_{map}.html')

@app.route("/intensity_analysis/heatmaps/<string:map>")
def heatmaps_page(map):
    return render_template(f'{MAPS_PATH}intensiity_maps/intensity_map_{map}.html')

@app.route('/outlier_analysis/heatmap/<string:event_type>/<string:column>')
def outlier_heatmap(event_type, column):
    return render_template(f'{MAPS_PATH}outliers_maps/outliers_heatmap_{event_type}_{column}.html')




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
    df_pandas = events_by_year.to_pandas()
    fig_px = px.line(df_pandas, x=YEAR, y='count', color=EVENT_TYPE,
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

    return fig_to_json_response(fig_px)

@app.route('/api/temporal_analysis/monthly_distribution/<string:event_type>')
def api_monthly_distribution(event_type):
    df_events_w_month = df_events.filter((pl.col(MONTH).is_not_null()) & (pl.col(EVENT_TYPE) == event_type))
    df_events_w_month = df_events_w_month.with_columns([
        pl.col(YEAR).cast(pl.Int32),
        pl.col(MONTH).cast(pl.Int32)
    ])

    month_events = df_events_w_month.group_by([YEAR, MONTH]).agg([
        pl.len().alias('count')
    ])
    pivot = month_events.pivot(
        values='count',
        index=MONTH,
        on=YEAR,
        aggregate_function='first'
    ).sort(MONTH)

    pivot_pandas = pivot.to_pandas()
    pivot_pandas = pivot_pandas.set_index(MONTH).reindex(
        sorted(pivot_pandas.columns, key=lambda x: int(x) if x.isdigit() else -1000000),
        axis=1
    ).fillna(0)
    pivot_pandas.columns = pivot_pandas.columns.astype(str)  # forza colonne = anni stringa
    pivot_pandas = pivot_pandas.iloc[:, 1:]

    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    z_values = pivot_pandas.values.astype(float)
    z_max = z_values.max()
    if z_max > 0:
        z_values /= z_max
    z_values = z_values.tolist()

    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=pivot_pandas.columns.tolist(),
        y=month_labels,
        colorscale='YlOrRd',
        hovertemplate='Year: %{x}<br>Month: %{y}<br>Count: %{z}<extra></extra>',
        colorbar=dict(title='Count')
    ))

    fig.update_layout(
        title=f'Monthly Heatmap - {event_type}',
        xaxis_title='Year',
        yaxis_title='Month',
        height=500,
        xaxis=dict(tickangle=90),
        yaxis=dict(autorange='reversed')  # Gennaio in alto
    )

    # Restituisci il JSON per Plotly.js
    return fig_to_json_response(fig)


@app.route('/api/temporal_analysis/seasonality')
def api_seasonality():
    ordered_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_map = {
        1.0: "Jan", 2.0: "Feb", 3.0: "Mar", 4.0: "Apr",
        5.0: "May", 6.0: "Jun", 7.0: "Jul", 8.0: "Aug",
        9.0: "Sep", 10.0: "Oct", 11.0: "Nov", 12.0: "Dec"
    }
    seasonality = df_events.filter(pl.col(MONTH).is_not_null()).group_by([MONTH, EVENT_TYPE]) \
        .agg([
        pl.len().alias('count')
    ]).with_columns(
        pl.col(MONTH).replace_strict(month_map, return_dtype=pl.Utf8)
    )


    fig = px.bar(seasonality.to_pandas(), x=MONTH, y='count', color=EVENT_TYPE,
                 title='Seasonal Distribution of Events',
                 labels={'count': 'Number of Events', MONTH: 'Month'},
                 barmode='group', category_orders={MONTH: ordered_months})
    return fig_to_json_response(fig)


@app.route('/api/geographic_analysis/top_countries_by_events/<string:event_type>/<int:limit>/<int:offset>')
def api_top_countries_by_events(event_type, limit, offset):
    if event_type != 'all':
        df_events_filtered = df_events.filter(pl.col(EVENT_TYPE) == event_type)
    else:
        df_events_filtered = df_events
    top_countries = (
        df_events_filtered
        .group_by(COUNTRY)
        .agg(pl.len().alias('count'))
        .sort('count', descending=True)
        .slice(offset, limit)
        .sort('count')
    )

    top_countries_pandas = top_countries.to_pandas()

    # Crea il grafico interattivo con Plotly
    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=top_countries_pandas[COUNTRY],
        x=top_countries_pandas['count'],
        orientation='h',
        marker=dict(
            color='lightblue',
            line=dict(color='darkblue', width=1)
        ),
        hovertemplate='<b>%{y}</b><br>Events: %{x}<extra></extra>'
    ))

    # Layout del grafico
    fig.update_layout(
        title=dict(
            text=f'Top {limit} Countries by Number of Events with offset {offset}',
            x=0.5,
            xanchor='center',
            font=dict(size=20)
        ),
        xaxis=dict(
            title='Number of Events',
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title='Country',
            categoryorder='total ascending'
        ),
        plot_bgcolor='white',
        height=500 + (limit * 20),  # Altezza dinamica in base al numero di paesi
        margin=dict(l=80, r=30, t=80, b=60)
    )

    # Restituisci il grafico come JSON
    return fig_to_json_response(fig)

@app.route('/api/geographic_analysis/events_by_region/<int:region_first>')
def api_events_by_region(region_first):
    group = [REGION, EVENT_TYPE] if region_first > 0 else [EVENT_TYPE, REGION]
    events_by_region = df_events.filter(pl.col(REGION).is_not_null()) \
        .group_by(group) \
        .agg(pl.len().alias('count'))

    fig = px.sunburst(events_by_region, path=group, values='count',
                      title='Event Distribution by Region and Type')

    return fig_to_json_response(fig)

@app.route('/api/intensity_analysis/count_by_intensity/<string:event_type>')
def api_count_by_intensity(event_type):
    column = EVENT_INTENSITY[event_type]
    intensity_df = (get_df_event(event_type).filter(
        pl.col(column).is_not_null()).with_columns(
        pl.col(column).cast(pl.Float64).round()
    ).filter(pl.col(column) > 0).group_by(column).agg(pl.len().alias("count")).sort(column))
    df_plot = intensity_df.to_pandas()
    fig = px.bar(
        df_plot,
        x=column,
        y="count",
        orientation="v",
        title="Frequency of intensities",
        labels={column: INTENSITY_LABELS[event_type], 'count': 'Number of events'}
    )
    return fig_to_json_response(fig)

    # Crea il grafico a barre

@app.route('/api/intensity_analysis/temporal_distribution/<int:normalize>')
def api_intensity_temporal_distribution(normalize):


    mean_intensity_by_year = engine.get_intensity_df(normalize > 0).sort([YEAR, EVENT_TYPE])
    # del df_intensity
    fig_px = px.line(mean_intensity_by_year.to_pandas(), x=YEAR, y='mean_intensity', color=EVENT_TYPE,
                     title='Event mean intensity per Year',
                     labels={'mean_intensity': 'Mean intensity', YEAR: 'Year'})
    fig_px.update_layout(
        template='plotly_white',
        title_x=0.5,
        xaxis=dict(title='Year'),
        yaxis=dict(title='Mean intensity'),
        legend=dict(title='Event Type - intensity type'),
        autosize=True
    )

    return fig_to_json_response(fig_px)


@app.route('/api/correlation_analysis/correlation_matrix/<string:event_type>')
def api_correlation_matrix(event_type):
    corr_df = spearman_corr(engine.get_correlation_df(event_type))

    fig = px.imshow(
        corr_df,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title=f"Spearman Correlation Matrix for {event_type} events",
        aspect="auto",
    )
    text_matrix = [
        ["NaN" if val == -2 else val for val in row]
        for row in fig.data[0].z
    ]

    # Aggiorna il testo nel trace principale
    fig.data[0].text = text_matrix
    fig.data[0].texttemplate = "%{text}"
    fig.data[0].textfont = dict(color="black")
    fig.data[0].hovertemplate = "%{x} vs %{y}<br>r = %{z:.2f}<extra></extra>"

    # Personalizza il layout
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=60, r=60, t=80, b=60),
        xaxis=dict(title=""),
        yaxis=dict(title=""),
        coloraxis_colorbar=dict(
            title="Correlation",
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticks="outside"
        ),
    )
    return fig_to_json_response(fig)


@app.route('/api/correlation_analysis/related_events')
def api_related_events():
    df = engine.get_related_events_counts().with_columns(
        pl.concat_str([pl.col("event_1"), pl.col("event_2")], separator=" - ").alias("relation")
    ).select(["relation", "num_events"])
    fig = px.bar(
        df.to_pandas(),
        x="relation",
        y="num_events",
        orientation="v",
        title="Relations between different events",
        labels={"relation": "Events Relation", 'num_events': 'Number of events'}
    )
    return fig_to_json_response(fig)

@app.route('/api/correlation_analysis/stats/<int:radius>')
def api_stats(radius):
    df = engine.get_concatenated_events(radius)
    df_by_event = df.group_by(EVENT_TYPE).agg(
        pl.col("distance").mean().alias("avg_distance"),
        pl.col("intensity").mean().alias("avg_intensity")
    )
    df_max_in_day = (
        df.group_by([YEAR, MONTH, DAY, EVENT_TYPE])
        .agg(
            pl.concat_list(["id1", "id2"]).alias("all_ids")
        )
        .with_columns(
            pl.col("all_ids").list.n_unique().alias("count")
        )
        .select([YEAR, MONTH, DAY, EVENT_TYPE, "count"])
    )
    eruptions_avg = df_by_event.filter(pl.col(EVENT_TYPE) == "eruption")
    eruptions_day = df_max_in_day.filter(pl.col(EVENT_TYPE) == "eruption")
    earthquakes_avg = df_by_event.filter(pl.col(EVENT_TYPE) == "earthquake")
    earthquakes_day = df_max_in_day.filter(pl.col(EVENT_TYPE) == "earthquake")
    tsunamis_avg = df_by_event.filter(pl.col(EVENT_TYPE) == "tsunami")
    tsunamis_day = df_max_in_day.filter(pl.col(EVENT_TYPE) == "tsunami")
    tornadoes_avg = df_by_event.filter(pl.col(EVENT_TYPE) == "tornado")
    tornadoes_day = df_max_in_day.filter(pl.col(EVENT_TYPE) == "tornado")
    stats = {
        "eruption": {
            "avg_distance": eruptions_avg.get_column("avg_distance").item() if eruptions_avg.height > 0 else None,
            "avg_intensity": eruptions_avg.get_column("avg_intensity").item() if eruptions_avg.height > 0 else None,
            "max_events": eruptions_day.get_column("count").max() if eruptions_day.height > 0 else None,
            "avg_multiple": eruptions_day.get_column("count").mean() if eruptions_day.height > 0 else None
        },
        "tsunami": {
            "avg_distance": tsunamis_avg.get_column("avg_distance").item() if tsunamis_avg.height > 0 else None,
            "avg_intensity": tsunamis_avg.get_column("avg_intensity").item() if tsunamis_avg.height > 0 else None,
            "max_events": tsunamis_day.get_column("count").max() if tsunamis_day.height > 0 else None,
            "avg_multiple": tsunamis_day.get_column("count").mean() if tsunamis_day.height > 0 else None
        },
        "tornado": {
            "avg_distance": tornadoes_avg.get_column("avg_distance").item() if tornadoes_avg.height > 0 else None,
            "avg_intensity": tornadoes_avg.get_column("avg_intensity").item() if tornadoes_avg.height > 0 else None,
            "max_events": tornadoes_day.get_column("count").max() if tornadoes_day.height > 0 else None,
            "avg_multiple": tornadoes_day.get_column("count").mean() if tornadoes_day.height > 0 else None
        },
        "earthquake": {
            "avg_distance": earthquakes_avg.get_column("avg_distance").item() if earthquakes_avg.height > 0 else None,
            "avg_intensity": earthquakes_avg.get_column("avg_intensity").item() if earthquakes_avg.height > 0 else None,
            "max_events": earthquakes_day.get_column("count").max() if earthquakes_day.height > 0 else None,
            "avg_multiple": earthquakes_day.get_column("count").mean() if earthquakes_day.height > 0 else None
        }
    }
    return jsonify(stats)

@app.route('/api/correlation_analysis/missed_relations/<int:radius>')
def api_missed_relations(radius):
    df = engine.get_missed_correlations(radius).group_by(["type1", "type2"]).agg(
        pl.count().alias("count")
    ).with_columns(
        pl.concat_str([pl.col("type1"), pl.col("type2")], separator=" - ").alias("relation")
    ).select(["relation", "count"])
    fig = px.bar(
        df.to_pandas(),
        x="relation",
        y="count",
        orientation="v",
        title="Missed Relations between different events",
        labels={"relation": "Events Relation", 'count': 'Number of events'}
    )
    return fig_to_json_response(fig)


@app.route('/api/outlier_analysis/analyze_column/<string:event_type>/<string:column>')
def api_analyze_column(event_type, column):
    return jsonify(get_outliers_analysis(engine, column, event_type))

@app.route('/api/outlier_analysis/analyze_column/<string:event_type>/<string:column>/table/<int:page>/<int:page_size>')
def api_analyze_column_table(event_type, column, page, page_size):
    df = outliers_cache["df"] if outliers_cache["event_type"] == event_type and outliers_cache["column"] == column \
        else get_outliers_analysis(engine, column, event_type, return_table=True)
    generate_map_outliers(df, event_type, column)
    total_outliers = df.height
    if event_type != outliers_cache["event_type"] or column != outliers_cache["column"]:
        outliers_cache["df"] = df
        outliers_cache["event_type"] = event_type
        outliers_cache["column"] = column
    df = df.slice((page-1)*page_size, page_size)
    result = {
        "outliers" : df.to_dicts(),
        "total_outliers": total_outliers
    }
    return jsonify(result)

@app.route('/api/outlier_analysis/heatmap/<string:event_type>/<string:column>/status')
def api_heatmap_status(event_type, column):
    file_path = f'{MAPS_PATH}outliers_maps/outliers_heatmap_{event_type}_{column}.html'
    return jsonify({
        'ready': os.path.exists(file_path)
    })





if __name__ == '__main__':
    if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        # Solo nel processo principale, non nel reloader
        start_map_generation(engine)
    df_events = engine.get_all_events()
    app.run(debug=True)
