import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import netCDF4 as nc
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation, PillowWriter

# --- Configuration ---
NC_FILE = 'test_run_microplastic_backward_20240220_20240101.nc'
PRODUCE_ANIMATION = True 
FRAME_SKIP = 2  # Plot every Nth frame to speed up generation
TRAIL_LIMIT = 20 # Limit trail length to the last N positions

def plot_trajectories():
    if not os.path.exists(NC_FILE):
        print(f'Error: File {NC_FILE} not found.')
        return

    # 1. Load Data and Metadata
    ds = nc.Dataset(NC_FILE)
    project_name = getattr(ds, 'project_name', 'LTM').replace(" ", "_")
    mode = getattr(ds, 'mode', 'microplastic')
    dimension = getattr(ds, 'dimension', '3D')
    direction = getattr(ds, 'direction', 'forward')
    
    obs_time = ds.variables['time'][:]
    time_units = ds.variables['time'].units
    calendar = getattr(ds.variables['time'], 'calendar', 'standard')
    times_dt = nc.num2date(obs_time, units=time_units, calendar=calendar)
    
    # Stability fix for cftime objects
    times_raw = times_dt.compressed() if hasattr(times_dt, 'compressed') else times_dt
    t_min, t_max = np.min(times_raw), np.max(times_raw)
    start_str, end_str = t_min.strftime('%Y%m%d'), t_max.strftime('%Y%m%d')
    
    obs_lon, obs_lat = ds.variables['lon'][:], ds.variables['lat'][:]
    obs_z, obs_pid = ds.variables['z'][:], ds.variables['particle_id'][:]
    unique_times = np.unique(obs_time)
    
    # 2. Setup Map
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([obs_lon.min()-1.5, obs_lon.max()+1.5, 
                   obs_lat.min()-1.5, obs_lat.max()+1.5])
    
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=3)
    ax.add_feature(cfeature.OCEAN, facecolor='aliceblue')
    
    gl = ax.gridlines(draw_labels=True, dms=True, alpha=0.2, zorder=4)
    gl.top_labels = gl.right_labels = False

    cmap = plt.get_cmap('jet_r')
    z_max = np.max(obs_z)
    norm = plt.Normalize(vmin=0, vmax=z_max if z_max > 0 else 100)
    
    title_base = f"{project_name} {mode} {dimension} {direction} {start_str}, {end_str}"
    
    # 3. Animation Components
    scat = ax.scatter([], [], c=[], cmap=cmap, norm=norm, s=15, edgecolors='black', linewidth=0.5, zorder=6)
    trail_collection = LineCollection([], cmap=cmap, norm=norm, alpha=0.4, zorder=5)
    ax.add_collection(trail_collection)
    anim_title = ax.set_title('', fontsize=12)

    # Dictionary to store trail history per particle
    particle_paths = {}

    def update(frame_time):
        idx = np.where(obs_time == frame_time)[0]
        if len(idx) == 0: return scat, trail_collection, anim_title
        
        current_pids = obs_pid[idx]
        current_lons = obs_lon[idx]
        current_lats = obs_lat[idx]
        current_zs = obs_z[idx]

        all_segments = []
        all_depths = []

        for i, pid in enumerate(current_pids):
            if pid not in particle_paths:
                particle_paths[pid] = {'lon': [], 'lat': [], 'z': []}
            
            # Append current position
            particle_paths[pid]['lon'].append(current_lons[i])
            particle_paths[pid]['lat'].append(current_lats[i])
            particle_paths[pid]['z'].append(current_zs[i])

            # SPEED OPTIMIZATION: Limit the trail length to the last N steps
            if len(particle_paths[pid]['lon']) > TRAIL_LIMIT:
                particle_paths[pid]['lon'].pop(0)
                particle_paths[pid]['lat'].pop(0)
                particle_paths[pid]['z'].pop(0)

            lons = np.array(particle_paths[pid]['lon'])
            lats = np.array(particle_paths[pid]['lat'])
            zs = np.array(particle_paths[pid]['z'])
            
            if len(lons) > 1:
                pts = np.array([lons, lats]).T.reshape(-1, 1, 2)
                segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                all_segments.extend(segs)
                all_depths.extend(zs[1:])

        # Update visual elements
        trail_collection.set_segments(all_segments)
        trail_collection.set_array(np.array(all_depths))
        scat.set_offsets(np.c_[current_lons, current_lats])
        scat.set_array(current_zs)
        
        current_dt = nc.num2date(frame_time, units=time_units, calendar=calendar)
        anim_title.set_text(f"{title_base}\nTime: {current_dt.strftime('%Y-%m-%d %H:%M')}")
        
        return scat, trail_collection, anim_title

    # SPEED OPTIMIZATION: Downsample frames using FRAME_SKIP
    frames = unique_times[::FRAME_SKIP] if direction == 'forward' else unique_times[::-FRAME_SKIP]
    
    ani = FuncAnimation(fig, update, frames=frames, blit=True)
    output_gif = f"{project_name}_{mode}_{dimension}_{direction}_{start_str},{end_str}.gif"
    
    print(f"Generating fast GIF with {TRAIL_LIMIT}-step trails: {output_gif}...")
    # Increase FPS slightly if you skip frames to keep it looking fluid
    ani.save(output_gif, writer=PillowWriter(fps=12))
    print("Animation complete.")
    ds.close()

if __name__ == '__main__':
    plot_trajectories()
