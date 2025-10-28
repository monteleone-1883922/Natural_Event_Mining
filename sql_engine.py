
import connectorx as cx
import polars as pl
import subprocess

from constants import *






class SqlEngine:

    def __init__(self, host: str = HOST, port: int = PORT, user: str = USER, password: str = PASSWORD, db: str = DB_NAME, container: str  = ""):
            self.connection = f"postgresql://{user}:{password}@{host}:{port}/{db}"
            self.host = host
            self.port = port
            self.user = user
            self.db = db
            self.container = container
            if not self.check_container_status():
                if not self.start_specific_container_with_compose():
                    print("Could not start database container!")
            else:
                print(f"{self.container} already running!")



    def check_container_status(self):
        """Check if a Docker container is running"""
        if self.container == "":
            return True
        try:
            result = subprocess.run(
                ['docker', 'ps', '--filter', f'name={self.container}', '--format', '{{.Names}}'],
                capture_output=True, text=True, check=True
            )
            return self.container in result.stdout
        except subprocess.CalledProcessError:
            return False

    def start_specific_container_with_compose(self, compose_file=COMPOSE_FILE_PATH):
        """Start a specific container using docker-compose"""
        try:
            cmd = ['docker', 'compose', '-f', compose_file, 'up', '-d', self.container]


            subprocess.run(cmd, check=True)
            print(f"Container '{self.container}' started successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error starting container '{self.container}': {e}")
            return False


    def query(self, sql_query: str) -> pl.DataFrame:
        return pl.read_database_uri(sql_query, self.connection)

    def get_all_events(self) -> pl.DataFrame:
        return self.query(ALL_EVENTS)

    def get_all_tornado_traces(self):
        return self.query(ALL_TORNADOES)

    def get_intensity_df(self, normalized, maps=False):
        earthquake_intensity = self.query(MAX_INTENSITY_EARTHQUAKE)["max"].item() if normalized else 1
        eruption_intensity = self.query(MAX_INTENSITY_ERUPTION)["max"].item() if normalized else 1
        tsunami_intensity = self.query(MAX_INTENSITY_TSUNAMI)["max"].item() if normalized else 1
        tornado_intensity = self.query(MAX_INTENSITY_TORNADO)["max"].item() if normalized else 1
        return self.query((INTENSITY_MAP_QUERY if maps else INTENSITY_QUERY).format(earthquake_intensity=earthquake_intensity,
                                                 eruption_intensity=eruption_intensity,
                                                 tsunami_intensity=tsunami_intensity,
                                                 tornado_intensity=tornado_intensity))

    def get_missed_correlations(self, radius):
        return self.query(MISSED_CORRELATIONS_QUERY.format(radius=radius))


    def get_correlation_df(self, event_type='all'):
        additional_selection = CORRELATION_FIELDS_BY_EVENT[event_type]
        join_condition = "" if event_type == "all" else JOIN_WITH_NATURAL_EVENT.format(event=TABLE_BY_EVENT[event_type])
        return self.query(CORRELATION_NATURAL_EVENT_QUERY
                          .format(additional_selection=additional_selection, join_condition=join_condition))

    def get_from_full_event(self, event_type, columns, not_null_columns=[]):
        where_clause = "" if not_null_columns == [] else f"WHERE {not_null_columns[0]} IS NOT NULL " + \
                                                          "AND ".join([f"{column_not_null} IS NOT NULL" for
                                                                       column_not_null in not_null_columns[1:]])
        return self.query(FULL_EVENT_QUERY.format(columns=", ".join(columns),
                                                  event=TABLE_BY_EVENT[event_type], where_clause=where_clause))

    def get_related_events_counts(self):
        return self.query(EVENTS_RELATED_QUERY)

    def get_concatenated_events(self, radius):
        return self.query(CONCATENATED_EVENTS_QUERY.format(radius=radius))


    def get_all_earthquakes(self):
        return self.query(ALL_EARTHQUAKES)

    def get_all_eruptions(self):
        return self.query(ALL_ERUPTIONS)

    def get_all_tsunamis(self):
        return self.query(ALL_TSUNAMIS)

engine = SqlEngine(container=DEFAULT_CONTAINER)