import configparser
from datetime import datetime
import pandas as pd
import os

class LTMConfig:
    def __init__(self, file_path="namelist.txt"):
        self.config = configparser.ConfigParser()
        self.config.read(file_path)
        # Store the directory of the namelist to help find the CSV file
        self.base_dir = os.path.dirname(os.path.abspath(file_path))

    def get_simulation_times(self):
        start = datetime.strptime(self.config.get('Simulation', 'start_simulation'), '%Y-%m-%d %H:%M:%S')
        end = datetime.strptime(self.config.get('Simulation', 'end_simulation'), '%Y-%m-%d %H:%M:%S')
        return start, end

    def get_sources(self):
        """
        Modified to read sources from a CSV file and apply global release parameters.
        """
        # 1. Load global release parameters from the [Release] section
        rel_cfg = self.config['Release']
        start_rel = datetime.strptime(rel_cfg['start_date_release'], '%Y-%m-%d %H:%M:%S')
        end_rel = datetime.strptime(rel_cfg['end_date_release'], '%Y-%m-%d %H:%M:%S')
        rate_day = int(rel_cfg['release_rate_per_day'])
        diameter = float(rel_cfg['diameter'])
        
        # 2. Locate and read the CSV file defined in the [Files] section
        csv_name = self.config.get('Files', 'source_csv')
        csv_path = os.path.join(self.base_dir, csv_name)
        
        # The CSV is expected to have: station_name, longitude, latitude, depth
        df = pd.read_csv(csv_path, names=['station_name', 'lon', 'lat', 'depth'])
        
        sources = []
        for _, row in df.iterrows():
            sources.append({
                "name": str(row['station_name']),
                "lon": float(row['lon']),
                "lat": float(row['lat']),
                "depth": float(row['depth']),
                "start_rel": start_rel,
                "end_rel": end_rel,
                "rate_day": rate_day,
                "diameter": diameter
            })
        return sources
