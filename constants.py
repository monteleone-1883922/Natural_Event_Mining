HOST = "localhost"
PORT = 5433
USER = "user"
PASSWORD = "password1234"
DB_NAME = "natural_event_db"
DEFAULT_SCHEMA =  "public"
DEFAULT_CONTAINER = "database_service"
COMPOSE_FILE_PATH = "DockerCompose.yml"
TEMPLATES_PATH = "templates/"
MAPS_PATH = "maps/"
EVENT_SPECIFIC_COLUMNS = {
    "intensity",
    "eqdepth",
    "eqmagnitude",
    "eqmagms",
    "eqmagml",
    "eqmagmw",
    "eqmagmb",
    "eqmagmfa",
    "eqmagunk",
    "eruption",
    "eruption_location",
    "significant",
    "vei",
    "agent",
    "eruption_status",
    "volcano_id",
    "f_scale",
    "latitudeend",
    "longitudeend",
    "trace_length",
    "width",
    "alteredmagnitude",
    "order_idx",
    "tornado_id",
    "eventvalidity",
    "causecode",
    "numdeposits",
    "numrunups",
    "tsintensity",
    "oceanictsunami",
    "maxwaterheight",
    "tsmtii",
    "tsmtabe",
    "warningstatusid",
    "cause",
    "validity",
    "warningstatus"
}
EVENTS_MAPS = ["all", "earthquake", "eruption", "tornado", "tsunami", "heatmap"]
INTENSITY_MAPS = ["earthquake", "eruption", "tsunami", "tornado", "all"]
TABLE_BY_EVENT = {"earthquake": "earthquake", "eruption": "eruption", "tornado": "tornado_trace", "tsunami": "tsunami"}
MULTIPLICATOR_OUTLIERS_ANALYSIS = 1.5
# SQL Queries
ALL_EVENTS = "SELECT * FROM natural_event"
ALL_TORNADOES = "SELECT * FROM tornado_trace"
ALL_ERUPTIONS = "SELECT * FROM eruption"
ALL_TSUNAMIS = "SELECT * FROM tsunami"
ALL_EARTHQUAKES = "SELECT * FROM earthquake"
FULL_EVENT_QUERY = "select {columns} from {event} as e join natural_event as ne on e.natural_event_id=ne.id {where_clause}"
MAX_INTENSITY_EARTHQUAKE = "select max(cast(e.eqmagnitude as float)) from earthquake as e"
MAX_INTENSITY_ERUPTION = "select max(cast(er.vei as float)) from eruption as er"
MAX_INTENSITY_TSUNAMI = "select max(cast(ts.maxwaterheight as float)) from tsunami as ts"
MAX_INTENSITY_TORNADO = "select max(cast(t.f_scale as float)) from tornado_trace as t"
INTENSITY_QUERY = """select p.event_year, p.event_type, avg(p.event_intensity) as mean_intensity
from (select concat(ne.event_type,'-magnitude') as event_type, ne.event_year, cast(e.eqmagnitude as float) / {earthquake_intensity} as event_intensity
from earthquake as e left join natural_event as ne on e.natural_event_id=ne.id
where e.eqmagnitude is not null union select concat(ne.event_type,'-f scale') as event_type, ne.event_year, cast(t.f_scale as float) / {tornado_intensity} as event_intensity
from tornado_trace as t left join natural_event as ne on t.natural_event_id=ne.id
where t.f_scale is not null and cast(t.f_scale as float) <> -9.0 union select concat(ne.event_type,'-max water height') as event_type, ne.event_year, cast(ts.maxwaterheight as float) / {tsunami_intensity} as event_intensity
from tsunami as ts left join natural_event as ne on ts.natural_event_id=ne.id
where ts.maxwaterheight is not null union select concat(ne.event_type,'-vei') as event_type, ne.event_year, cast(er.vei as float) / {eruption_intensity} as event_intensity
from eruption as er left join natural_event as ne on er.natural_event_id=ne.id
where er.vei is not null) as p group by p.event_type, p.event_year;"""
INTENSITY_MAP_QUERY = """select ne.event_type, ne.longitude, ne.latitude, CASE
    WHEN ne.event_type='earthquake'
        THEN cast(e.eqmagnitude as float) / {earthquake_intensity}
    WHEN ne.event_type='tornado'
        THEN cast(t.f_scale as float) / {tornado_intensity}
    WHEN ne.event_type='tsunami'
        THEN cast(ts.maxwaterheight as float) / {tsunami_intensity}
    WHEN ne.event_type='eruption'
        THEN cast(er.vei as float) / {eruption_intensity}
END AS event_intensity
FROM natural_event AS ne LEFT JOIN  earthquake AS e ON e.natural_event_id = ne.id
LEFT JOIN tornado_trace AS t ON t.natural_event_id = ne.id
LEFT JOIN tsunami AS ts ON ts.natural_event_id = ne.id
LEFT JOIN eruption AS er ON er.natural_event_id = ne.id
WHERE ne.latitude IS NOT NULL AND ne.longitude IS NOT NULL AND
      (e.eqmagnitude IS NOT NULL OR ne.event_type <> 'earthquake') AND
      ((t.f_scale IS NOT NULL AND CAST(t.f_scale AS FLOAT) <> -9.0) OR ne.event_type <> 'tornado') AND
      (ts.maxwaterheight IS NOT NULL OR ne.event_type <> 'tsunami') AND
      (er.vei IS NOT NULL OR ne.event_type <> 'eruption');"""
CORRELATION_NATURAL_EVENT_QUERY="""select ne.latitude, ne.longitude, ne.damagemillionsdollars, ne.deaths,
       ne.missing, ne.housesdamaged, ne.housesdestroyed, ne.injuries, ne.event_year, ne.milliondollarscropsdamage {additional_selection}
from natural_event as ne {join_condition}"""
EVENTS_RELATED_QUERY="""select ne1.event_type as event_1, ne2.event_type as event_2, count(1) as num_events
from related_event as re join natural_event as ne1 on re.event1_id = ne1.id
    join natural_event as ne2 on re.event2_id = ne2.id
where ne1.event_type <> 'tornado'
group by ne1.event_type, ne2.event_type;"""

# ne1.longitude as event1_, ne1.latitude, ne2.longitude, ne2.latitude,
#        ne1.id, ne2.id, ne1.event_timestamp, ne2.event_timestamp,
#        tr1.tornado_id as tornado1_id, tr2.tornado_id as tornado2_id,
CONCATENATED_EVENTS_QUERY="""select ne1.event_type, ne1.event_year, ne1.event_month, ne1.event_day,ne1.id as id1, ne2.id as id2,
                            6371 * 2 * ASIN(SQRT(
                                    POWER(SIN((RADIANS(ne1.latitude) - RADIANS(ne2.latitude)) / 2), 2) +
                                    COS(RADIANS(ne1.latitude)) * COS(RADIANS(ne2.latitude)) *
                                    POWER(SIN((RADIANS(ne1.longitude) - RADIANS(ne2.longitude)) / 2), 2)
                                )) as distance, CASE
    WHEN ne1.event_type='earthquake'
        THEN cast(ea.eqmagnitude as float)
    WHEN ne1.event_type='tornado'
        THEN cast(tr1.f_scale as float)
    WHEN ne1.event_type='tsunami'
        THEN cast(ts.maxwaterheight as float)
    WHEN ne1.event_type='eruption'
        THEN cast(er.vei as float) END as intensity
from natural_event as ne1 left join tornado_trace as tr1 on ne1.id = tr1.natural_event_id
    left join earthquake as ea on ne1.id = ea.natural_event_id
    left join eruption as er on ne1.id = er.natural_event_id
    left join tsunami as ts on ne1.id = ts.natural_event_id
    join natural_event as ne2 on ne1.id < ne2.id and
                                 ne1.event_year = ne2.event_year and
                                 ne1.event_type = ne2.event_type and
                                 ne1.event_month = ne2.event_month and
                                 ne1.event_day = ne2.event_day and 6371 * 2 * ASIN(SQRT(
                                    POWER(SIN((RADIANS(ne1.latitude) - RADIANS(ne2.latitude)) / 2), 2) +
                                    COS(RADIANS(ne1.latitude)) * COS(RADIANS(ne2.latitude)) *
                                    POWER(SIN((RADIANS(ne1.longitude) - RADIANS(ne2.longitude)) / 2), 2)
                                )) < {radius}
left join tornado_trace as tr2 on ne2.id = tr2.natural_event_id
where ne1.event_month is not null and ne1.event_day is not null and
      ne2.event_month is not null and ne2.event_day is not null and
      ne1.longitude is not null and ne1.latitude is not null and
      ne2.longitude is not null and ne2.latitude is not null and
      (ne1.event_type <> 'tornado' or ne2.event_type <> 'tornado' or
       (tr1.tornado_id <> tr2.tornado_id and tr1.tornado_id is not null and tr2.tornado_id is not null));"""
MISSED_CORRELATIONS_QUERY="""select distinct ne1.event_type as type1, ne2.event_type as type2, ne1.event_year, ne1.event_month, ne1.event_day,
        ne1.id as id1, ne2.id as id2,
                            6371 * 2 * ASIN(SQRT(
                                    POWER(SIN((RADIANS(ne1.latitude) - RADIANS(ne2.latitude)) / 2), 2) +
                                    COS(RADIANS(ne1.latitude)) * COS(RADIANS(ne2.latitude)) *
                                    POWER(SIN((RADIANS(ne1.longitude) - RADIANS(ne2.longitude)) / 2), 2)
                                )) as distance, CASE
    WHEN ne1.event_type='earthquake'
        THEN cast(ea.eqmagnitude as float)
    WHEN ne1.event_type='tornado'
        THEN cast(tr1.f_scale as float)
    WHEN ne1.event_type='tsunami'
        THEN cast(ts.maxwaterheight as float)
    WHEN ne1.event_type='eruption'
        THEN cast(er.vei as float) END as intensity
from natural_event as ne1 left join tornado_trace as tr1 on ne1.id = tr1.natural_event_id
    left join earthquake as ea on ne1.id = ea.natural_event_id
    left join eruption as er on ne1.id = er.natural_event_id
    left join tsunami as ts on ne1.id = ts.natural_event_id
    join natural_event as ne2 on ne1.id < ne2.id and
                                 ne1.event_year = ne2.event_year and
                                 ne1.event_type <> ne2.event_type and
                                 ne1.event_month = ne2.event_month and
                                 ne1.event_day = ne2.event_day and 6371 * 2 * ASIN(SQRT(
                                    POWER(SIN((RADIANS(ne1.latitude) - RADIANS(ne2.latitude)) / 2), 2) +
                                    COS(RADIANS(ne1.latitude)) * COS(RADIANS(ne2.latitude)) *
                                    POWER(SIN((RADIANS(ne1.longitude) - RADIANS(ne2.longitude)) / 2), 2)
                                )) < {radius}
left join related_event as re1 on ne1.id = re1.event1_id and ne2.id = re1.event2_id
left join related_event as re2 on ne1.id = re2.event2_id and ne2.id = re2.event1_id
where ne1.event_month is not null and ne1.event_day is not null and
      ne2.event_month is not null and ne2.event_day is not null and
      ne1.longitude is not null and ne1.latitude is not null and
      ne2.longitude is not null and ne2.latitude is not null and
      re1.event1_id is null and re2.event1_id is null;"""
COMPLETE_EVENTS_CLUSTERING_QUERY="""select distinct ne.event_type, ne.latitude, ne.longitude, ne.deaths, ne.damagemillionsdollars, ne.housesdestroyed,
                ne.event_year, ne.event_month, ne.event_day,
    CASE WHEN ne.event_type='earthquake'
        THEN cast(ea.eqmagnitude as float) / {earthquake_intensity}
    WHEN ne.event_type='tornado'
        THEN CASE WHEN cast(tr.f_scale as float) = -9.0 THEN null ELSE cast(tr.f_scale as float) / {tornado_intensity} END
    WHEN ne.event_type='tsunami'
        THEN cast(ts.maxwaterheight as float) / {tsunami_intensity}
    WHEN ne.event_type='eruption'
        THEN cast(er.vei as float) / {eruption_intensity} END as intensity 
from natural_event as ne left join tornado_trace as tr on ne.id = tr.natural_event_id
    left join earthquake as ea on ne.id = ea.natural_event_id
    left join eruption as er on ne.id = er.natural_event_id
    left join tsunami as ts on ne.id = ts.natural_event_id
where ne.event_month is not null and ne.event_day is not null and
      ne.longitude is not null and ne.latitude is not null;"""
JOIN_WITH_NATURAL_EVENT="join {event} as e on e.natural_event_id = ne.id "

CORRELATION_FIELDS_EARTHQUAKE=",e.eqmagnitude, e.intensity, e.eqdepth "
CORRELATION_FIELDS_ERUPTION=",cast(e.vei as float) "
CORRELATION_FIELDS_TSUNAMI=",e.maxwaterheight, e.numdeposits, e.numrunups, e.tsintensity, e.tsmtii"
CORRELATION_FIELDS_TORNADO=",CASE WHEN e.f_scale='-9' THEN null ELSE cast(e.f_scale as float) END, e.width, e.latitudeend, e.longitudeend, e.trace_length "
JOINED_EVENT_QUERIES={
    "intensity": INTENSITY_QUERY,
    "intensity_map": INTENSITY_MAP_QUERY,
    "clustering": COMPLETE_EVENTS_CLUSTERING_QUERY
}
CORRELATION_FIELDS_BY_EVENT={
    "earthquake": CORRELATION_FIELDS_EARTHQUAKE,
    "eruption": CORRELATION_FIELDS_ERUPTION,
    "tsunami": CORRELATION_FIELDS_TSUNAMI,
    "tornado": CORRELATION_FIELDS_TORNADO,
    "all": ""
}
#columns
YEAR = "event_year"
MONTH = "event_month"
DAY = "event_day"
EVENT_TYPE = "event_type"
INTENSITY = "event_intensity"
COUNTRY = "country"
LONGITUDE = "longitude"
LATITUDE = "latitude"
REGION = "region"
NATURAL_EVENT_ID = "natural_event_id"
LONGITUDE_END = "longitudeend"
LATITUDE_END = "latitudeend"
# other configs

EVENT_INTENSITY = {
    "earthquake": "eqmagnitude",
    "eruption": "vei",
    "tsunami": "maxwaterheight",
    "tornado": "f_scale"

}

INTENSITY_LABELS = {
    "earthquake": "Magnitude",
    "eruption": "VEI",
    "tsunami": "Max Water Height (m)",
    "tornado": "F Scale"
}
GEOGRAPHIC_SECTIONS = [
            {
                'title': 'Map of all events',
                'description': 'Analyze events distribution across the world',
                'url': '/geographic_analysis/maps/all',
                'icon': 'globe',
                'color': 'primary'  # Blu - neutro per "tutti"
            },
            {
                'title': 'Map of earthquakes',
                'description': 'Analyze earthquakes distribution across the world',
                'url': '/geographic_analysis/maps/earthquake',
                'icon': 'mountain',  # Icona più appropriata
                'color': 'danger'  # Rosso - pericolo/impatto forte
            },
            {
                'title': 'Map of eruptions',
                'description': 'Analyze eruptions distribution across the world',
                'url': '/geographic_analysis/maps/eruption',
                'icon': 'fire',  # Icona vulcano
                'color': 'warning'  # Arancione - fuoco/lava
            },
            {
                'title': 'Map of tornadoes',
                'description': 'Analyze tornadoes distribution across the world',
                'url': '/geographic_analysis/maps/tornado',
                'icon': 'wind',  # Icona tornado
                'color': 'info'  # Azzurro - cielo/tempesta
            },
            {
                'title': 'Map of tsunamis',
                'description': 'Analyze tsunamis distribution across the world',
                'url': '/geographic_analysis/maps/tsunami',
                'icon': 'water',  # Icona onda
                'color': 'primary'  # Blu - acqua/mare
            },
            {
                'title': 'Heatmap of all events',
                'description': 'Heatmap of all events across the world',
                'url': '/geographic_analysis/maps/heatmap',
                'icon': 'layer-group',  # Icona più appropriata per heatmap
                'color': 'success'  # Verde - densità/aggregazione
            }
        ]
INTENSITY_SECTIONS=[
            {
                'title': 'HeatMap of all event intensities',
                'description': 'Analyze event intensities distribution across the world',
                'url': '/intensity_analysis/heatmaps/all',
                'icon': 'globe',
                'color': 'primary'  # Blu - neutro per "tutti"
            },
            {
                'title': 'Heatmap of earthquakes magnitude',
                'description': 'Analyze earthquakes magnitude across the world',
                'url': '/intensity_analysis/heatmaps/earthquake',
                'icon': 'mountain',  # Icona più appropriata
                'color': 'danger'  # Rosso - pericolo/impatto forte
            },
            {
                'title': 'Heatmap of eruptions VEI',
                'description': 'Analyze eruptions VEI across the world',
                'url': '/intensity_analysis/heatmaps/eruption',
                'icon': 'fire',  # Icona vulcano
                'color': 'warning'  # Arancione - fuoco/lava
            },
            {
                'title': 'Heatmap of tornadoes F scale',
                'description': 'Analyze tornadoes F scale across the world',
                'url': '/intensity_analysis/heatmaps/tornado',
                'icon': 'wind',  # Icona tornado
                'color': 'info'  # Azzurro - cielo/tempesta
            },
            {
                'title': 'Heatmap of tsunamis max water height',
                'description': 'Analyze tsunamis max water height across the world',
                'url': '/intensity_analysis/heatmaps/tsunami',
                'icon': 'water',  # Icona onda
                'color': 'primary'  # Blu - acqua/mare
            }
        ]
LEGEND_HTML_OUTLIERS_HEATMAPS = '''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 200px; height: 120px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px; border-radius: 5px;
                    box-shadow: 0 0 15px rgba(0,0,0,0.2);">
            <h4 style="margin-top:0;">Outliers Legend</h4>
            <p style="margin: 5px 0;">
                <span style="background-color: red; width: 20px; height: 20px; 
                             display: inline-block; margin-right: 5px; border-radius: 50%;"></span>
                Upper Outliers
            </p>
            <p style="margin: 5px 0;">
                <span style="background-color: blue; width: 20px; height: 20px; 
                             display: inline-block; margin-right: 5px; border-radius: 50%;"></span>
                Lower Outliers
            </p>
        </div>
        '''