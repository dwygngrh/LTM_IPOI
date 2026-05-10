import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.collections import LineCollection
import netCDF4 as nc
from matplotlib.animation import FuncAnimation, PillowWriter
import pandas as pd

# --- Configuration ---
NC_FILE = 'test_run_microplastic_forward_20240101_20240210.nc'

# NEW OPTIONS
PRODUCE_ANIMATION = True  # Set to True to generate GIF, False for static image only
USE_DEPTH_COLOR = True    
SHOW_INACTIVE = True    

def plot_mixed_trajectories():
    if not os.path.exists(NC_FILE):
        print(f"Error: File {NC_FILE} not found.")
        return

    # 1. Load Data
    ds = nc.Dataset(NC_FILE)
    mode = getattr(ds, 'mode', 'microplastic')
    output_base = NC_FILE.replace('.nc', '')
    
    obs_lon = ds.variables['lon'][:]
    obs_lat = ds.variables['lat'][:]
    obs_z = ds.variables['z'][:]
    obs_status = ds.variables['status'][:]
    obs_pid = ds.variables['particle_id'][:]
    obs_time = ds.variables['time'][:] # Hours since reference
    
    # Handle Time Formatting
    # Assuming CMEMS standard time units (check your nc_handler.py for units)
    time_units = ds.variables['time'].units
    times_dt = nc.num2date(obs_time, units=time_units)
    
    unique_pids = np.unique(obs_pid)
    unique_times = np.sort(np.unique(obs_time))
    
    # 2. Setup Map
    fig = plt.figure(figsize=(15, 10))
    proj = ccrs.PlateCarree()
    ax = plt.axes(projection=proj)
    ax.set_extent([obs_lon.min()-2, obs_lon.max()+2, obs_lat.min()-2, obs_lat.max()+2], crs=proj)
    ax.add_feature(cfeature.LAND, facecolor='#e0e0e0', zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=3)
    
    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.6, zorder=1)
    gl.top_labels = gl.right_labels = False

    # Color Setup
    cmap = plt.get_cmap('viridis_r')
    norm = plt.Normalize(vmin=0, vmax=max(obs_z.max(), 1.0))

    # Static plot elements
    title = ax.set_title("", fontsize=14, pad=20)
    if USE_DEPTH_COLOR and mode == 'microplastic':
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02, aspect=40, label='Depth (m)').ax.invert_yaxis()

    # --- ANIMATION LOGIC ---
    if PRODUCE_ANIMATION:
        print(f"--- [PROCESS] Generating Animation for {len(unique_times)} steps ---")
        
        # Lists to hold plot objects for updating
        paths = {pid: ax.plot([], [], color='grey', linewidth=0.8, alpha=0.3, zorder=4)[0] for pid in unique_pids}
        dots = ax.plot([], [], 'o', color='red', markersize=3, zorder=6)[0]
        beached = ax.plot([], [], 'x', color='black', markersize=4, zorder=5)[0]

        def update(frame_time):
            # Convert time to YYYY-DD-MM
            dt_obj = nc.num2date(frame_time, units=time_units)
            timestamp_str = dt_obj.strftime('%Y-%d-%m') # Format requested: YYYY-DD-MM
            title.set_text(f"LTM 2026 - {mode.upper()} | {timestamp_str}")

            curr_lons, curr_lats = [], []
            beach_lons, beach_lats = [], []

            for pid in unique_pids:
                # Get data up to current frame time
                p_idx = np.where((obs_pid == pid) & (obs_time <= frame_time))[0]
                if len(p_idx) == 0: continue
                
                sort_p = np.argsort(obs_time[p_idx])
                lon = obs_lon[p_idx][sort_p]
                lat = obs_lat[p_idx][sort_p]
                status = obs_status[p_idx][sort_p][-1]

                # Update trajectory trail
                paths[pid].set_data(lon, lat)
                
                if status == 1:
                    curr_lons.append(lon[-1])
                    curr_lats.append(lat[-1])
                elif SHOW_INACTIVE:
                    beach_lons.append(lon[-1])
                    beach_lats.append(lat[-1])

            dots.set_data(curr_lons, curr_lats)
            beached.set_data(beach_lons, beach_lats)
            return list(paths.values()) + [dots, beached, title]

        # Reduce frame count for GIF performance if needed (e.g., [::2])
        anim = FuncAnimation(fig, update, frames=unique_times[::2], interval=100, blit=True)
        
        gif_name = f"{output_base}.gif"
        print(f"Saving animation to: {gif_name}")
        anim.save(gif_name, writer=PillowWriter(fps=10))
    
    else:
        # Static plotting (original logic)
        print("--- [PROCESS] Generating Static Plot ---")
        # ... (Your existing loop for static plotting) ...
        plt.savefig(f"{output_base}.jpg", dpi=300, bbox_inches='tight')

    plt.close()

if __name__ == "__main__":
    plot_mixed_trajectories()
