import georinex as gr
import numpy as np
import pandas as pd
import xarray as xr
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="georinex")

# Load navigation data
nav_data = gr.load('ephimeredes.nav')

# Constants
GM = 3.986005e14  # m³/s²
omega_e = 7.2921151467e-5  # rad/s
c = 299792458  # m/s (speed of light)

# Observation data from the project document
observation_data = {
    'G08': {'P_L1': 22550792.660, 'dt_j': 0.000133456e-3},
    'G10': {'P_L1': 22612136.900, 'dt_j': 0.000046155711e-3},
    'G21': {'P_L1': 20754631.240, 'dt_j': -0.000151820e-3},
    'G24': {'P_L1': 23974471.500, 'dt_j': 0.000265875e-3},
    'G17': {'P_L1': 24380357.760, 'dt_j': -0.000721440e-3},
    'G03': {'P_L1': 24444143.500, 'dt_j': 0.000221870e-3},
    'G14': {'P_L1': 22891323.280, 'dt_j': -0.000130207e-3}
}

# Observation epoch
T = 558000  # seconds

def compute_satellite_coordinates(nav_data, sv_id, transmission_time, include_corrections=True):
    """
    Compute satellite coordinates at transmission time using broadcast ephemeris
    
    Parameters:
    nav_data: xarray dataset with navigation data
    sv_id: satellite ID (e.g., 'G08')
    transmission_time: time at which satellite transmitted signal
    include_corrections: whether to include the 9 correction terms
    
    Returns:
    tuple: (X, Y, Z) coordinates in meters
    """
    
    # Extract ephemeris parameters for the satellite
    # Find the appropriate time index (closest to transmission time)
    time_idx = 0  # Using first time index as it matches our data
    
    # Extract orbital parameters
    sqrt_a = float(nav_data['sqrtA'].sel(sv=sv_id, time=nav_data.time[time_idx]))
    e = float(nav_data['Eccentricity'].sel(sv=sv_id, time=nav_data.time[time_idx]))
    M0 = float(nav_data['M0'].sel(sv=sv_id, time=nav_data.time[time_idx]))
    omega = float(nav_data['omega'].sel(sv=sv_id, time=nav_data.time[time_idx]))
    i0 = float(nav_data['Io'].sel(sv=sv_id, time=nav_data.time[time_idx]))
    Omega0 = float(nav_data['Omega0'].sel(sv=sv_id, time=nav_data.time[time_idx]))
    Delta_n = float(nav_data['DeltaN'].sel(sv=sv_id, time=nav_data.time[time_idx]))
    i_dot = float(nav_data['IDOT'].sel(sv=sv_id, time=nav_data.time[time_idx]))
    Omega_dot = float(nav_data['OmegaDot'].sel(sv=sv_id, time=nav_data.time[time_idx]))
    toe = float(nav_data['Toe'].sel(sv=sv_id, time=nav_data.time[time_idx]))
    
    # Correction terms (set to 0 if not including corrections)
    if include_corrections:
        Cuc = float(nav_data['Cuc'].sel(sv=sv_id, time=nav_data.time[time_idx]))
        Cus = float(nav_data['Cus'].sel(sv=sv_id, time=nav_data.time[time_idx]))
        Crc = float(nav_data['Crc'].sel(sv=sv_id, time=nav_data.time[time_idx]))
        Crs = float(nav_data['Crs'].sel(sv=sv_id, time=nav_data.time[time_idx]))
        Cic = float(nav_data['Cic'].sel(sv=sv_id, time=nav_data.time[time_idx]))
        Cis = float(nav_data['Cis'].sel(sv=sv_id, time=nav_data.time[time_idx]))
    else:
        Cuc = Cus = Crc = Crs = Cic = Cis = 0.0
    
    # Semi-major axis
    a = sqrt_a**2
    
    # Time from ephemeris reference epoch
    tk = transmission_time - toe
    
    # Account for beginning or end of GPS week crossovers
    if tk > 302400:
        tk -= 604800
    elif tk < -302400:
        tk += 604800
    
    # Corrected mean motion
    n0 = np.sqrt(GM / a**3)
    n = n0 + Delta_n
    
    # Mean anomaly
    Mk = M0 + n * tk
    
    # Eccentric anomaly (solve by iteration)
    E0 = Mk  # Initial value
    for j in range(3):  # Three iterations as specified
        E_prev = E0 if j == 0 else E_new
        E_new = E_prev + (Mk - E_prev + e * np.sin(E_prev)) / (1 - e * np.cos(E_prev))
    Ek = E_new
    
    # True anomaly
    fk = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(Ek / 2))
    
    # Argument of latitude
    uk = omega + fk + Cuc * np.cos(2 * (omega + fk)) + Cus * np.sin(2 * (omega + fk))
    
    # Radius
    rk = a * (1 - e * np.cos(Ek)) + Crc * np.cos(2 * (omega + fk)) + Crs * np.sin(2 * (omega + fk))
    
    # Inclination
    ik = i0 + i_dot * tk + Cic * np.cos(2 * (omega + fk)) + Cis * np.sin(2 * (omega + fk))
    
    # Longitude of ascending node
    Lambda_k = Omega0 + (Omega_dot - omega_e) * tk - omega_e * toe
    
    # Satellite position in orbital plane
    x_orbital = rk * np.cos(uk)
    y_orbital = rk * np.sin(uk)
    
    # Rotation matrices
    def R3(angle):
        """Rotation matrix about z-axis"""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    
    def R1(angle):
        """Rotation matrix about x-axis"""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])
    
    # Transform to Earth-fixed coordinates
    R = R3(-Lambda_k) @ R1(-ik) @ R3(-uk)
    orbital_pos = np.array([rk, 0, 0])
    earth_fixed_pos = R @ orbital_pos
    
    return earth_fixed_pos

def compute_transmission_time(T, pseudorange, dt_j):
    """
    Compute transmission time accounting for signal travel time and satellite clock error
    
    Parameters:
    T: observation epoch (reception time)
    pseudorange: measured pseudorange in meters
    dt_j: satellite clock error in seconds
    
    Returns:
    transmission time in seconds
    """
    return T - pseudorange / c + dt_j

# Task 1: Compute satellite coordinates with all correction terms
print("=" * 60)
print("TASK 1: GPS SATELLITE COORDINATES WITH CORRECTION TERMS")
print("=" * 60)
print(f"Observation epoch T = {T} seconds")
print()

satellite_coords_corrected = {}

for sv in observation_data.keys():
    # Compute transmission time
    P_L1 = observation_data[sv]['P_L1']
    dt_j = observation_data[sv]['dt_j']
    
    ts = compute_transmission_time(T, P_L1, dt_j)
    
    print(f"Satellite {sv}:")
    print(f"  Pseudorange P(L1) = {P_L1:.3f} m")
    print(f"  Satellite clock error dt_j = {dt_j:.9f} s")
    print(f"  Transmission time ts = {ts:.6f} s")
    
    # Compute coordinates with corrections
    coords = compute_satellite_coordinates(nav_data, sv, ts, include_corrections=True)
    satellite_coords_corrected[sv] = coords
    
    print(f"  Coordinates [X, Y, Z] = [{coords[0]:.3f}, {coords[1]:.3f}, {coords[2]:.3f}] m")
    print()

# Verify with given hint for SV03
print("Verification with given hint:")
print("Given coordinates for SV03:")
print("  [23098433.065, -12669412.772, 2685881.089] m")
print(f"Computed coordinates for G03:")
print(f"  [{satellite_coords_corrected['G03'][0]:.3f}, {satellite_coords_corrected['G03'][1]:.3f}, {satellite_coords_corrected['G03'][2]:.3f}] m")

# Calculate differences
diff_x = satellite_coords_corrected['G03'][0] - 23098433.065
diff_y = satellite_coords_corrected['G03'][1] - (-12669412.772)
diff_z = satellite_coords_corrected['G03'][2] - 2685881.089
print(f"Differences: [{diff_x:.3f}, {diff_y:.3f}, {diff_z:.3f}] m")

print("\n" + "=" * 60)
print("Summary of all satellite coordinates (with corrections):")
print("=" * 60)
for sv, coords in satellite_coords_corrected.items():
    print(f"{sv}: [{coords[0]:12.3f}, {coords[1]:12.3f}, {coords[2]:12.3f}] m")