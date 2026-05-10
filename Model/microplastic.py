import numpy as np
from numba import njit

@njit
def get_polymer_density(polymer_type):
    """Specific density mapping[cite: 2, 5]."""
    if polymer_type == "PE": return 910.0 
    if polymer_type == "HDPE": return 950.0
    return 1025.0

@njit
def get_fouled_density(rho_p, r0_mm, rho_bf, bt_mm):
    """Calculates fouled density based on core-shell volume ratios[cite: 4]."""
    r0 = r0_mm / 1000.0
    bt = bt_mm / 1000.0
    vol_ratio = (r0**3) / ((r0 + bt)**3)
    return (rho_p * vol_ratio) + (rho_bf * (1.0 - vol_ratio))

@njit
def calculate_3d_ws(rho_p, d_mm, rho_bf, bt_mm, shape_idx):
    """Calculates dynamic ws based on Jalón-Rojas et al. (2019)[cite: 4]."""
    g, rho_w, nu = 9.81, 1025.0, 1e-6
    radius_tot_m = (d_mm / 2.0 + bt_mm) / 1000.0
    rho_fouled = get_fouled_density(rho_p, d_mm/2.0, rho_bf, bt_mm)
    
    # Buoyancy: positive for sinking, negative for rising[cite: 4, 5]
    d_star = 2 * radius_tot_m * (g * np.abs(rho_fouled - rho_w) / (rho_w * nu**2))**(1/3)
    
    if shape_idx == 0: # sphere
        w_val = (nu / (2 * radius_tot_m)) * d_star**3 * (38.1 + 0.93 * d_star**(12/7))**-0.875
    else: # cylinder/other
        w_val = (np.pi / 2) * (1/nu) * g * (np.abs(rho_fouled - rho_w)/rho_w) * (2*(radius_tot_m**2) / 55.0)
    
    return w_val if rho_fouled > rho_w else -w_val
