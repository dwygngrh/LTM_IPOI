import numpy as np

def spawn(src, now, dt_seconds):
    """
    Spawns particles using a probabilistic threshold to handle low release rates.
    """
    # Only release if current time is within the release window
    if not (src['start_rel'] <= now <= src['end_rel']):
        return [], [], []
    
    # Calculate the exact floating-point number of particles for this step
    # Example: (2 per day / 86400 seconds) * 3600 seconds = 0.0833
    n_float = (src['rate_day'] / 86400.0) * abs(dt_seconds)
    
    # 1. Get the base integer (number of guaranteed particles)
    n = int(n_float)
    
    # 2. Use the fractional part as a probability for one extra particle
    # If np.random.rand() is less than 0.0833, spawn a particle.
    if np.random.rand() < (n_float - n):
        n += 1
    
    if n <= 0: 
        return [], [], []
    
    # Random spatial distribution within the specified diameter
    r_deg = (src['diameter'] / 2.0) / 111000.0
    r = r_deg * np.sqrt(np.random.rand(n))
    theta = np.random.rand(n) * 2 * np.pi
    
    lons = (src['lon'] + r * np.cos(theta)).tolist()
    lats = (src['lat'] + r * np.sin(theta)).tolist()
    depths = [src['depth']] * n
    
    return lons, lats, depths
