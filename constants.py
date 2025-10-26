HOST = "localhost"
PORT = 5433
USER = "user"
PASSWORD = "password1234"
DB_NAME = "natural_event_db"
DEFAULT_SCHEMA =  "public"
DEFAULT_CONTAINER = "database_service"
COMPOSE_FILE_PATH = "DockerCompose.yml"
TEMPLATES_PATH = "templates/"
HTMLS_PATH = "templates/"
EVENTS_MAPS = ["all", "earthquake", "eruption", "tornado", "tsunami", "heatmap"]
INTENSITY_MAPS = ["earthquake", "eruption", "tsunami", "tornado", "all"]
# SQL Queries
ALL_EVENTS = "SELECT * FROM natural_event"
ALL_TORNADOES = "SELECT * FROM tornado_trace"
ALL_ERUPTIONS = "SELECT * FROM eruption"
ALL_TSUNAMIS = "SELECT * FROM tsunami"
ALL_EARTHQUAKES = "SELECT * FROM earthquake"
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