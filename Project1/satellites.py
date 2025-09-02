import georinex as gr
import pandas as pd
import warnings
import numpy as np
from prettytable import PrettyTable
# Ignore FutureWarnings from georinex
warnings.filterwarnings("ignore", category=FutureWarning, module="georinex")

# Load the navigation data
nav_data = gr.load('Project1/ephemerides.nav')
# Convert to DataFrame
df = nav_data.to_dataframe().reset_index()

# keep only the 7 satellites
wanted = ["G08","G10","G21","G24","G17","G03","G14"]
df = df[df["sv"].isin(wanted)]

# drop empty rows
df = df.dropna(subset=["sqrtA"])

# rename columns
rename_map = {
    "Eccentricity": "e",
    "Io": "i0",         
    "IDOT": "iDot",
}
df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns})

# Read observations
obs_data = pd.read_csv('Project1/observations.csv')

# Merge with navigation data
df = df.merge(obs_data, on="sv", how="inner")

print(df.columns)

# Transmission time
def ts(T, P, c, dt):
    return T - (P/c) + dt

# Constants 
T = 558000 # s 
GM = 3.986005e14 # m^3/s^2
omega_e = 7.2921151467e-5 # rad/s
pi = np.pi
c = 299792458 # m/s

# Time from ephemerides to reference epoch
def tk(t, toe):
    tk = t - toe 
    if tk > 302400:
        tk -= 604800
    elif tk < -302400:
        tk += 604800
    return tk

def Mk(M0, GM, a, DeltaN, tk):
    return M0 + (np.sqrt(GM/(a**3)) + DeltaN) * tk

def Ek(Mk, e):
    Ek = Mk
    for j in range(3):
        Ek = Ek + (Mk-Ek + e*np.sin(Ek))/(1-e*np.cos(Ek))
    return Ek

def fk(e, E):
    return 2 * np.arctan(np.sqrt((1+e)/(1-e)) * np.tan(E/2))

def uk(omega, fk, Cuc, Cus):
    return omega + fk + Cuc*np.cos(2*(omega + fk)) + Cus*np.sin(2*(omega + fk))

def rk(a, e, Ek, Crc, Crs, omega, fk):
    return a*(1-e*np.cos(Ek)) + Crc*np.cos(2*(omega + fk)) + Crs*np.sin(2*(omega + fk))

def ik(i0, iDot, tk, Cic, Cis, omega, fk):
    return i0 + iDot*tk + Cic*np.cos(2*(omega + fk)) + Cis*np.sin(2*(omega + fk))

def lambda_k(lambda0, OmegaDot, tk, omega_e, toe):
    return lambda0 + (OmegaDot - omega_e)*tk - omega_e*toe

def R1(i):
    return np.array([[1, 0, 0],
                    [0, np.cos(i), np.sin(i)],
                    [0, -np.sin(i), np.cos(i)]])

def R3(lam):
    return np.array([[np.cos(lam), np.sin(lam), 0],
                    [-np.sin(lam), np.cos(lam), 0],
                    [0, 0, 1]])

def satellite_position(sv, data, corr=True):
    # Extract satellite data
    sat_data = data[data["sv"] == sv]
    if sat_data.empty:
        return None
    
    P = sat_data["P"].values[0]
    dt = sat_data["dt"].values[0]
    toe = sat_data["Toe"].values[0]
    sqrtA = sat_data["sqrtA"].values[0]
    M0 = sat_data["M0"].values[0]
    e = sat_data["e"].values[0]
    omega = sat_data["omega"].values[0]
    i0 = sat_data["i0"].values[0]
    Omega0 = sat_data["Omega0"].values[0]

    # secular correction terms
    DeltaN = sat_data["DeltaN"].values[0] if corr else 0
    iDot = sat_data["iDot"].values[0] if corr else 0
    OmegaDot = sat_data["OmegaDot"].values[0] if corr else 0

    # periodic correction terms
    Cuc = sat_data["Cuc"].values[0] if corr else 0
    Cus = sat_data["Cus"].values[0] if corr else 0
    Crc = sat_data["Crc"].values[0] if corr else 0
    Crs = sat_data["Crs"].values[0] if corr else 0
    Cic = sat_data["Cic"].values[0] if corr else 0
    Cis = sat_data["Cis"].values[0] if corr else 0



    ts_ = ts(T, P, c, dt)
    tk_ = tk(ts_, toe)
    Mk_ = Mk(M0, GM, sqrtA**2, DeltaN, tk_)
    Ek_ = Ek(Mk_, e)
    fk_ = fk(e, Ek_)
    uk_ = uk(omega, fk_, Cuc, Cus)
    rk_ = rk(sqrtA**2, e, Ek_, Crc, Crs, omega, fk_)
    ik_ = ik(i0, iDot, tk_, Cic, Cis, omega, fk_)
    lambda_k_ = lambda_k(Omega0, OmegaDot, tk_, omega_e, toe)

    X, Y, Z = R3(-lambda_k_) @ R1(-ik_) @ R3(-uk_) @ np.array([rk_, 0, 0])
    print(Y)
    return np.array([X, Y, Z])

satellite_positions = {}
satellite_positions_no_corr = {}
for sv in wanted:
    pos = satellite_position(sv, df)
    satellite_positions[sv] = pos
    pos_no_corr = satellite_position(sv, df, corr=False)
    satellite_positions_no_corr[sv] = pos_no_corr

print("Satellite positions with corrections:")
t = PrettyTable()
t.field_names = ["Satellite", "X (m)", "Y (m)", "Z (m)"]
for sv, pos in satellite_positions.items():
    t.add_row([sv, pos[0], pos[1], pos[2]])
print(t)

print("Satellite positions without corrections:")
t_no_corr = PrettyTable()
t_no_corr.field_names = ["Satellite", "X (m)", "Y (m)", "Z (m)"]
for sv, pos in satellite_positions_no_corr.items():
    t_no_corr.add_row([sv, pos[0], pos[1], pos[2]])
print(t_no_corr)

print("Difference between corrected and uncorrected positions:")
t_diff = PrettyTable()
t_diff.field_names = ["Satellite", "dX (m)", "dY (m)", "dZ (m)"]
for sv in satellite_positions.keys():
    if sv in satellite_positions_no_corr:
        dX = satellite_positions[sv][0] - satellite_positions_no_corr[sv][0]
        dY = satellite_positions[sv][1] - satellite_positions_no_corr[sv][1]
        dZ = satellite_positions[sv][2] - satellite_positions_no_corr[sv][2]
        t_diff.add_row([sv, dX, dY, dZ])
print(t_diff)

