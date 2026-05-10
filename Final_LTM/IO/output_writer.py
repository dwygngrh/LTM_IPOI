import os
import netCDF4 as nc

class TrajectoryWriter:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        if not os.path.exists(output_dir): 
            os.makedirs(output_dir)

    def create_file(self, name, cfg_ref):
        """Initializes the NetCDF file with dimensions, variables, and global metadata."""
        file_path = os.path.join(self.output_dir, f"{name}.nc")
        mode = cfg_ref.config['Simulation']['mode']
        
        # PROJECT NAME METADATA
        proj_name = cfg_ref.config['Simulation'].get('project_name', 'LTM_Project')
        
        dt = int(cfg_ref.config['Simulation']['dt'])
        running_type = "forward" if dt > 0 else "backward"

        with nc.Dataset(file_path, 'w', format='NETCDF4') as root:
            root.createDimension('obs', None) 

            time_var = root.createVariable('time', 'f8', ('obs',))
            time_var.units = "seconds since 1970-01-01 00:00:00"
            time_var.calendar = "gregorian"
            
            root.createVariable('lon', 'f4', ('obs',))
            root.createVariable('lat', 'f4', ('obs',))
            root.createVariable('z', 'f4', ('obs',))
            root.createVariable('status', 'i1', ('obs',))
            root.createVariable('particle_id', 'i4', ('obs',))

            # GLOBAL METADATA ATTRIBUTES
            root.Project = proj_name
            root.Author = "Dr Dwiyoga Nugroho, RCO-BRIn 2026"
            root.mode = mode
            root.running_type = running_type 

    def write_step(self, file_name, time, lons, lats, depths, status, p_ids):
        """Appends a new time step of data to the NetCDF file."""
        file_path = os.path.join(self.output_dir, f"{file_name}.nc")
        if not lons: return
        
        with nc.Dataset(file_path, 'a') as root:
            start_idx = len(root.dimensions['obs'])
            count = len(lons)
            end_idx = start_idx + count
            
            t_val = nc.date2num(time, units=root.variables['time'].units, 
                                calendar=root.variables['time'].calendar)
            
            root.variables['time'][start_idx:end_idx] = [t_val] * count
            root.variables['lon'][start_idx:end_idx] = lons
            root.variables['lat'][start_idx:end_idx] = lats
            root.variables['z'][start_idx:end_idx] = depths
            root.variables['status'][start_idx:end_idx] = status
            root.variables['particle_id'][start_idx:end_idx] = p_ids
